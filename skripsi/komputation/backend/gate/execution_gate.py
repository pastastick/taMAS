"""Execution gate (B12) — jalankan ekspresi pada sampel kecil sebelum backtest.

Kenapa ada. Sembilan gate yang sudah terpasang semuanya STRUKTURAL: parsable,
arity, variabel dikenal, degenerate-args, semantik-numerik, kompleksitas,
redundansi. Semuanya bekerja pada TEKS ekspresi. Akibatnya sebuah ekspresi bisa
sah di semua sumbu itu dan tetap mati saat dijalankan:

    ZSCORE((TS_ZSCORE($volume,5) > 2) ? TS_PCTCHANGE($close,1) : 0)

sah secara sintaksis dan semantis, tetapi kolomnya kosong setelah dropna (11 dari
198 ekspresi di G5). Satu-satunya cara menangkapnya adalah MENJALANKANNYA.

Biaya. Sampel disimpan sebagai parquet (beberapa ratus instrumen × ~1 tahun);
evaluasi satu ekspresi ≈ 0,05–1 detik CPU — dua sampai tiga kali lipat lebih
murah daripada satu putaran repair LLM, apalagi daripada satu backtest.

Yang ditolak (dengan pesan ramah-LLM untuk agen repair):
  - ekspresi yang MELEDAK saat dijalankan (kelas galat yang selama ini baru
    ketahuan di factor.py, setelah gate dinyatakan lolos);
  - kolom yang seluruhnya NaN / kosong setelah dropna;
  - kolom KONSTAN atau nyaris-konstan (≤ `min_unique` nilai unik per hari) —
    faktor semacam ini tak bisa me-ranking saham berapa pun ICnya.

Sengaja TIDAK menolak berdasarkan IC: gate ini soal apakah faktornya HIDUP,
bukan apakah faktornya BAGUS. Menyeleksi berdasarkan IC di dalam gate akan
mencemari perbandingan antar-lengan (dan memilih berdasarkan data uji).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from qlog import logger

# Rombakan 9d4e0bf memindahkan berkas ini dari `backend/factors/regulator/`
# (3 tingkat di bawah root) ke `backend/gate/` (2 tingkat), tapi rantai
# `.parent.parent.parent`-nya ikut terbawa — sehingga `_BACKEND` menunjuk ROOT
# PROYEK, bukan `backend/`, dan `daily_pv.h5` tak pernah ketemu. Gate ini
# fail-open saat datanya hilang, jadi kegagalannya SENYAP: ekspresi yang
# kolomnya kosong/konstan tetap lolos tanpa satu pun pesan error. Dipakai
# `paths.BACKEND` (jalur kanonik repo) supaya tidak bisa melenceng lagi kalau
# berkas ini dipindah sekali lagi.
from paths import BACKEND as _BACKEND

# Panel sumber mengikuti env `LAB_PV_FILE` yang sama dengan `eval/ic.py`, supaya
# gate dan penilai IC tidak pernah melihat pasar yang berbeda dalam satu proses.
# Nama berkas cache-nya ikut menyertakan penanda pasar; tanpa itu sampel A-share
# akan dipakai ulang diam-diam untuk panel IDX.
def _sumber() -> Path:
    v = os.environ.get("LAB_PV_FILE", "").strip()
    if not v:
        return _BACKEND / "hf_data" / "daily_pv.h5"
    p = Path(v)
    return p if p.is_absolute() else (_BACKEND.parent / p)


_SOURCE = _sumber()
_CACHE = (_BACKEND / "hf_data" / ".cache" /
          ("exec_gate_sample.parquet" if _SOURCE.stem == "daily_pv"
           else f"exec_gate_sample_{_SOURCE.stem}.parquet"))

# Jendela sampel. Sengaja BUKAN jendela OOS yang dipakai untuk mengukur IC:
# gate hanya perlu tahu apakah ekspresi menghasilkan kolom hidup, dan memakai
# jendela penilaian di sini akan membuat gate ikut menyeleksi di atas data uji.
# Panel IDX mulai 2019 → jendelanya bisa digeser lewat env tanpa mengubah kode.
_SAMPLE_START = os.environ.get("LAB_GATE_SAMPLE_START", "2018-01-01")
_SAMPLE_END = os.environ.get("LAB_GATE_SAMPLE_END", "2018-12-31")
_MAX_INSTRUMENTS = 300


class ExecutionGate:
    """Evaluator ekspresi di sampel kecil. Data dimuat sekali, lalu dipakai ulang."""

    def __init__(
        self,
        min_unique: float = 2.0,
        min_coverage_days: int = 20,
        max_seconds: float = 20.0,
        enabled: bool = True,
    ) -> None:
        self.min_unique = min_unique
        self.min_coverage_days = min_coverage_days
        self.max_seconds = max_seconds
        self.enabled = enabled
        self._df = None
        self._unavailable = False

    # ── data ────────────────────────────────────────────────────────────────
    def _load(self):
        import pandas as pd

        if _CACHE.exists():
            return pd.read_parquet(_CACHE)
        if not _SOURCE.exists():
            raise FileNotFoundError(f"sumber data sampel tak ada: {_SOURCE}")
        df = pd.read_hdf(_SOURCE, key="data")
        df = df.loc[_SAMPLE_START:_SAMPLE_END]
        df = df.drop(columns=["$factor"], errors="ignore").sort_index()
        insts = df.index.get_level_values("instrument").unique()
        if len(insts) > _MAX_INSTRUMENTS:
            keep = set(sorted(insts)[:_MAX_INSTRUMENTS])
            df = df[df.index.get_level_values("instrument").isin(keep)]
        # $return dibentuk seperti factors/data_template/generate.py
        df["$return"] = (df.groupby(level="instrument")["$close"]
                         .pct_change(fill_method=None).fillna(0))
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE)
        return df

    @property
    def df(self):
        if self._df is None and not self._unavailable:
            try:
                self._df = self._load()
                logger.info(f"[ExecutionGate] sampel dimuat: {self._df.shape[0]} baris, "
                            f"{self._df.index.get_level_values('instrument').nunique()} instrumen")
            except Exception as e:  # noqa: BLE001
                # FAIL-OPEN: tanpa data, gate ini tak boleh menjatuhkan pipeline.
                self._unavailable = True
                logger.warning(f"[ExecutionGate] dinonaktifkan (data tak tersedia): {e!r}")
        return self._df

    # ── evaluasi ────────────────────────────────────────────────────────────
    def values(self, expression: str):
        """Jalur evaluasi IDENTIK produksi (coder/template.jinjia2)."""
        import contextlib
        import io

        import numpy as np
        import pandas as pd

        from dsl.expr_parser import parse_expression, parse_symbol
        import dsl.function_lib as FL

        df = self.df  # noqa: F841 — dipakai di eval()
        code = parse_symbol(expression, df.columns)
        with contextlib.redirect_stdout(io.StringIO()):
            code = parse_expression(code)
        for col in df.columns:
            code = code.replace(col[1:], f"df['{col}']")
        env = {k: getattr(FL, k) for k in dir(FL) if not k.startswith("_")}
        env.update({"df": df, "np": np, "pd": pd})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = eval(code, env)  # noqa: S307 — sama dengan produksi
        if isinstance(out, pd.DataFrame):
            out = out.iloc[:, 0]
        if not isinstance(out, pd.Series):
            if callable(out) or isinstance(out, (str, bytes, type)):
                raise TypeError(
                    f"the expression evaluates to a {type(out).__name__}, not a series "
                    f"of numbers — check for a function name used without arguments"
                )
            out = pd.Series(out, index=df.index)
        if not np.issubdtype(out.dtype, np.number):
            out = pd.to_numeric(out, errors="coerce")
        return out

    def check(self, expression: str) -> Tuple[bool, str]:
        """(ok, pesan). Fail-open bila data/waktu tak tersedia."""
        if not self.enabled or self.df is None:
            return True, ""
        import numpy as np

        try:
            with _time_budget(self.max_seconds):
                vals = self.values(expression)
        except TimeoutError:
            # Ekspresi yang terlalu lambat (mis. REGBETA per-instrumen) tidak
            # dinyatakan salah — hanya tak bisa diperiksa di sini.
            logger.debug(f"[ExecutionGate] lewat batas waktu, dilewati: {expression[:60]}")
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, (f"the expression fails when it is actually evaluated: "
                           f"{type(e).__name__}: {e}")

        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            return False, ("the expression evaluates to an empty column (every value is "
                           "NaN or infinite). A conditional gate that is almost never "
                           "true, a window longer than the data, or a division by zero "
                           "does this. Widen the condition or drop it.")
        by_day = vals.groupby(level="datetime")
        n_days = by_day.ngroups
        if n_days < self.min_coverage_days:
            return False, (f"the expression produces values on only {n_days} days in the "
                           f"probe window; it needs at least {self.min_coverage_days}. "
                           f"Shorten the lookback window or loosen the condition.")
        uniq = float(by_day.nunique().mean())
        if uniq <= self.min_unique:
            return False, (f"the expression gives only {uniq:.1f} distinct values per day, "
                           f"so it cannot rank stocks against each other. Let the data "
                           f"reach the VALUE (gate a magnitude, e.g. "
                           f"`(C) ? (TS_ZSCORE($return,5)) : (0)`), not just the condition.")
        return True, ""


class _time_budget:
    """Batas waktu per-ekspresi agar satu operator lambat tak menyandera gate."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        import signal

        def _raise(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"melebihi {self.seconds}s")

        try:
            self._old = signal.signal(signal.SIGALRM, _raise)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        except ValueError:
            # signal hanya bisa dipasang di main thread; di worker → tanpa batas.
            self._old = None
        return self

    def __exit__(self, *exc):
        import signal

        if self._old is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
        return False


_SHARED: Optional[ExecutionGate] = None


def get_execution_gate() -> Optional[ExecutionGate]:
    """Instance bersama (data dimuat sekali per proses).

    Dimatikan lewat env `LATENTMAS_EXEC_GATE=0` — dipakai saat ingin mengukur
    laju kegagalan eksekusi apa adanya, tanpa gate ini menyaringnya lebih dulu.
    """
    global _SHARED
    if os.environ.get("LATENTMAS_EXEC_GATE", "1") == "0":
        return None
    if _SHARED is None:
        _SHARED = ExecutionGate()
    return _SHARED


__all__ = ["ExecutionGate", "get_execution_gate"]
