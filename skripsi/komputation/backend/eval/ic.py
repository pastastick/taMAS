"""Evaluasi ekspresi DSL + RankIC di CPU, tanpa GPU dan tanpa Qlib.

Replika jalur evaluasi produksi lama (`coder/template.jinjia2` →
parse_symbol + parse_expression + eval(function_lib); `factors/runner.py` →
_factor_label / _oos_window / _compute_factor_ic). Kedua berkas sumber itu ikut
terhapus saat perombakan branch `exp/empat-metode-v1`; modul ini adalah
penerusnya dan kini satu-satunya jalur penilaian ekspresi.

Divalidasi: IC yang dihasilkan modul ini identik (7+ desimal) dengan
`extra_info.factor_ic` batch 2026-07-05 untuk faktor berbasis harga/volume.

Dua mode data:
  full  — 2015-01-01.. (setara runtime runpod), ~11 jt baris, lambat
  fast  — hanya jendela OOS + warmup, ~1 jt baris, IC identik untuk window <= 60

Pemakaian:
    from eval.ic import Lab
    lab = Lab(mode="fast")
    print(lab.ic("RANK($volume)"))
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent      # .../quantalatent/backend
QL_ROOT = BACKEND.parent                               # .../quantalatent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Batasi worker joblib SEBELUM `function_lib` (dan karenanya joblib) di-import.
# REGBETA/REGRESI/BB_* dipanggil dengan `n_jobs=-1`, jadi satu ekspresi yang
# memakai salah satunya men-spawn satu worker per core; tiap worker mem-fork
# induknya bersama data pasar (~320 MB terukur di mesin 16-core → +5 GB sekali
# jalan). Di mesin GPU 46 GB itu tak terasa; di laptop/CPU-box ia membunuh
# proses skoring lewat OOM di tengah korpus. Batasnya bisa dinaikkan lewat env.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", os.environ.get("LAB_MAX_WORKERS", "3"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# jendela test dari conf_combined_factors.yaml (dibaca, bukan di-hardcode)
_CONF = BACKEND / "eval" / "qlib_conf" / "conf_combined_factors.yaml"
CACHE_DIR = QL_ROOT / "results" / ".cache"

# ── PASAR: dari mana panel harga-volume dibaca ────────────────────────────────
# Default = panel A-share (`daily_pv.h5`, ~5.982 instrumen, ~4.240 saham/hari).
# Diganti lewat env `LAB_PV_FILE`, mis. panel IDX dari `scripts/ambil_data_idx.py`:
#
#     LAB_PV_FILE=backend/hf_data_id/daily_pv_idx_top100.h5
#
# Jalur relatif dihitung dari AKAR REPO, bukan CWD. Nama berkas ikut masuk ke
# nama berkas cache — tanpa itu cache panel A-share dipakai ulang diam-diam
# untuk panel IDX pada jendela yang sama, dan seluruh angka IC-nya palsu.
_PV_DEFAULT = BACKEND / "hf_data" / "daily_pv.h5"


def pv_source() -> Path:
    v = os.environ.get("LAB_PV_FILE", "").strip()
    if not v:
        return _PV_DEFAULT
    p = Path(v)
    return p if p.is_absolute() else (QL_ROOT / p)


def pasar_tag() -> str:
    """Penanda pasar, aman dipakai di nama berkas cache."""
    return pv_source().stem  # mis. "daily_pv" / "daily_pv_idx_top100"


def oos_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    import yaml

    seg = yaml.safe_load(_CONF.read_text())["task"]["dataset"]["kwargs"]["segments"]["test"]
    return pd.Timestamp(seg[0]), pd.Timestamp(seg[1])


@dataclass
class ICResult:
    ic: float | None          # mean_t RankIC harian
    icir: float | None        # mean/std deret IC harian
    n_days: int               # jumlah hari dengan IC terdefinisi
    tstat: float | None       # icir * sqrt(n_days) — uji H0: IC = 0
    coverage: float           # rata-rata jumlah saham per hari / universe
    n_unique: float           # rata-rata nilai unik per hari (deteksi konstan)
    error: str | None = None

    @property
    def significant(self) -> bool:
        return self.tstat is not None and abs(self.tstat) >= 1.96

    def __repr__(self) -> str:
        if self.error:
            return f"<ICResult ERROR {self.error[:60]}>"
        ic = "None" if self.ic is None else f"{self.ic:+.5f}"
        t = "None" if self.tstat is None else f"{self.tstat:+.2f}"
        return (f"<IC={ic} t={t} n={self.n_days} cov={self.coverage:.0f} "
                f"uniq={self.n_unique:.0f}>")


class Lab:
    def __init__(self, mode: str = "fast", warmup_days: int = 400,
                 window: tuple[str, str] | None = None):
        """window=None → segmen `test` config (2021, = jendela yang dipakai
        evolution untuk SELEKSI). Beri window eksplisit untuk holdout sejati,
        mis. ("2022-01-01", "2025-12-26") = split test QuantaAlpha."""
        self.mode = mode
        if window is None:
            self.oos_start, self.oos_end = oos_window()
        else:
            self.oos_start, self.oos_end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        self._df = None
        self._label = None
        self._warmup = warmup_days
        self._universe = None

    # ── data ──────────────────────────────────────────────────────────────
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tag = (self.mode if self.mode == "full"
               else f"fast_{self.oos_start.date()}_{self.oos_end.date()}")
        pasar = pasar_tag()
        # panel bawaan tetap memakai nama cache lama supaya cache A-share yang
        # sudah ada di mesin tidak jadi mubazir; panel lain dapat namanya sendiri.
        cache = CACHE_DIR / (f"pv_{tag}.parquet" if pasar == "daily_pv"
                             else f"pv_{pasar}_{tag}.parquet")
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            src = pv_source()
            df = pd.read_hdf(src, key="data")
            start = ("2015-01-01" if self.mode == "full"
                     else str((self.oos_start - pd.Timedelta(days=self._warmup)).date()))
            # butuh 2 hari sesudah OOS untuk label Ref(-2)
            end = str((self.oos_end + pd.Timedelta(days=10)).date())
            df = df.loc[start:end]
            df = df.drop(columns=["$factor"], errors="ignore").sort_index()
            # $return dibentuk seperti factors/data_template/generate.py
            df["$return"] = (df.groupby(level="instrument")["$close"]
                             .pct_change(fill_method=None).fillna(0))
            df.to_parquet(cache)
        return df

    @property
    def label(self) -> pd.Series:
        """Ref($close,-2)/Ref($close,-1)-1 — identik runner._factor_label."""
        if self._label is None:
            close = self.df["$close"].sort_index()
            g = close.groupby(level="instrument")
            self._label = (g.shift(-2) / g.shift(-1) - 1.0).rename("label")
        return self._label

    # ── evaluasi ekspresi ─────────────────────────────────────────────────
    def values(self, expr: str) -> pd.Series:
        """Jalur identik coder/template.jinjia2."""
        from dsl.expr_parser import parse_expression, parse_symbol
        import dsl.function_lib as FL

        df = self.df  # noqa: F841 — dipakai di eval()
        code = parse_symbol(expr, df.columns)
        # parse_expression punya print debug bawaan produksi → dibisukan di lab
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            code = parse_expression(code)
        for col in df.columns:
            code = code.replace(col[1:], f"df['{col}']")
        env = {k: getattr(FL, k) for k in dir(FL) if not k.startswith("_")}
        env.update({"df": df, "np": np, "pd": pd})
        out = eval(code, env)  # noqa: S307 — sama dengan produksi
        if isinstance(out, pd.DataFrame):
            out = out.iloc[:, 0]
        # Ekspresi LLM yang cacat bisa mengevaluasi ke OBJEK, bukan angka —
        # mis. `MAX` telanjang (tanpa argumen) menghasilkan objek fungsi, yang
        # kalau di-broadcast ke pd.Series akan meledak jauh di hilir
        # (`float() argument must be ... not 'function'`) dan menjatuhkan
        # seluruh loop skoring. Tolak di sini dengan pesan yang jelas; `ic()`
        # menangkapnya jadi ICResult(error=...) seperti kegagalan lainnya.
        if not isinstance(out, pd.Series):
            if callable(out) or isinstance(out, (str, bytes, type)):
                raise TypeError(
                    f"ekspresi menghasilkan {type(out).__name__}, bukan deret angka"
                )
            out = pd.Series(out, index=df.index)
        if not np.issubdtype(out.dtype, np.number):
            out = pd.to_numeric(out, errors="coerce")
        return out

    # ── metrik ────────────────────────────────────────────────────────────
    def _ic_core(self, vals: pd.Series) -> tuple[ICResult, "pd.Series | None"]:
        """Hitung ICResult DAN deret IC harian dalam satu kali groupby.

        Dipisah karena `ic_of_values` dan `ic_full` dulu menghitung `per_day`
        yang persis sama dua kali — dan groupby-spearman atas ±1 jt baris itulah
        biaya dominan skoring korpus (bukan evaluasi ekspresinya). Angkanya
        tidak berubah sedikit pun; hanya dihitung sekali.
        """
        d = pd.DataFrame({"f": vals, "y": self.label})
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        if not d.empty:
            dts = d.index.get_level_values("datetime")
            d = d[(dts >= self.oos_start) & (dts <= self.oos_end)]
        if d.empty:
            return (ICResult(None, None, 0, None, 0.0, 0.0,
                             error="empty after dropna/OOS"), None)

        grp = d.groupby(level="datetime")
        per_day = grp.apply(
            lambda x: x["f"].corr(x["y"], method="spearman") if len(x) > 2 else np.nan
        )
        cov = grp.size().mean()
        uniq = grp["f"].nunique().mean()
        ic, sd = per_day.mean(), per_day.std()
        n = int(per_day.notna().sum())
        icir = float(ic / sd) if pd.notna(ic) and pd.notna(sd) and sd > 0 else None
        t = float(icir * np.sqrt(n)) if icir is not None and n > 1 else None
        return (ICResult(float(ic) if pd.notna(ic) else None, icir, n, t,
                         float(cov), float(uniq)), per_day)

    def ic_of_values(self, vals: pd.Series) -> ICResult:
        return self._ic_core(vals)[0]

    def ic(self, expr: str) -> ICResult:
        try:
            return self.ic_of_values(self.values(expr))
        except Exception as e:  # noqa: BLE001
            return ICResult(None, None, 0, None, 0.0, 0.0,
                            error=f"{type(e).__name__}: {e}")

    def ic_full(self, expr: str) -> tuple[ICResult, "pd.Series | None"]:
        """IC + deret IC harian dari SATU kali evaluasi ekspresi DAN satu kali
        groupby (dua-duanya dulu dihitung dobel; itu yang membuat skoring korpus
        sangat lambat)."""
        try:
            vals = self.values(expr)
        except Exception as e:  # noqa: BLE001
            return ICResult(None, None, 0, None, 0.0, 0.0,
                            error=f"{type(e).__name__}: {e}"), None
        res, series = self._ic_core(vals)
        if res.error is not None or res.ic is None:
            return res, None
        return res, series

    def ic_series(self, expr: str) -> pd.Series | None:
        """Deret IC harian (untuk uji beda antar-faktor / Newey-West)."""
        try:
            vals = self.values(expr)
        except Exception:  # noqa: BLE001
            return None
        d = pd.DataFrame({"f": vals, "y": self.label})
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        dts = d.index.get_level_values("datetime")
        d = d[(dts >= self.oos_start) & (dts <= self.oos_end)]
        if d.empty:
            return None
        return d.groupby(level="datetime").apply(
            lambda x: x["f"].corr(x["y"], method="spearman") if len(x) > 2 else np.nan
        )
