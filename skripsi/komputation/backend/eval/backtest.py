"""Metrik backtest portofolio long–short di atas ekspresi DSL, di CPU.

Kenapa modul ini ada terpisah dari `eval/ic.py`. `ic.py` menjawab "apakah
ekspresi ini punya daya prediksi lintas-saham" (RankIC harian, ICIR, t-stat) —
itu metrik yang sudah divalidasi identik dengan produksi dan yang dipakai
seluruh dokumen di `docs/`. Tapi RankIC tidak menjawab "kalau sinyal ini
benar-benar diperdagangkan, apa hasilnya" — return, drawdown, dan biaya
perputaran posisi tidak muncul di korelasi peringkat sama sekali. Skripsi
menjanjikan "metrik hasil backtest", jadi lapisan itu ditambahkan di sini.

Desain sengaja SEDERHANA dan tanpa Qlib: portofolio kuantil ekual-bobot,
rebalance harian, tanpa biaya transaksi kecuali diminta eksplisit. Alasannya
ada di `docs/AUDIT_KRITIS.md` §S3/B8 — backtest LightGBM gabungan milik
QuantaAlpha lama TIDAK sah dipakai membandingkan mode, karena skornya
mencampur banyak faktor sekaligus sehingga kontribusi satu ekspresi tak bisa
diisolasi. Portofolio satu-faktor di bawah ini bisa.

PERINGATAN INTERPRETASI. Angka di sini adalah backtest kasar, bukan simulasi
perdagangan: tanpa slippage, tanpa batas likuiditas, tanpa aturan suspensi
bursa, dan rebalance harian penuh (turnover tinggi). `turnover` dan
`cost_bps` disediakan supaya besarnya biaya yang diabaikan bisa dilaporkan
alih-alih disembunyikan. Perbandingan ANTAR-METODE tetap adil karena semua
metode dinilai dengan pipeline yang sama persis.

Pemakaian:
    from eval.ic import Lab
    from eval.backtest import backtest_expression
    lab = Lab(mode="fast")
    print(backtest_expression(lab, "RANK($volume)"))
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Hari perdagangan per tahun. Dipakai HANYA untuk anualisasi (ann_return,
# ann_vol, dan karenanya Sharpe); nilainya HARUS sama di semua lengan yang
# dibandingkan agar angkanya sebanding — tapi HARUS ikut pasar, karena kalender
# bursanya berbeda:
#   A-share (qlib cn_data)          : ≈243 hari/tahun
#   IDX     (kalender 2021–2025)    : ≈241 hari/tahun (247/246/239/237/236)
# Ganti lewat env `LAB_TRADING_DAYS` bersamaan dengan `LAB_PV_FILE`.
# Perhatikan: RankIC/ICIR/t-stat TIDAK tersentuh konstanta ini sama sekali.
import os as _os

TRADING_DAYS = int(_os.environ.get("LAB_TRADING_DAYS", "243"))


@dataclass
class BacktestResult:
    """Ringkasan satu ekspresi yang diperdagangkan sebagai portofolio long–short."""
    ann_return: Optional[float]     # return tahunan rata-rata (geometric-free, sum×TRADING_DAYS)
    ann_vol: Optional[float]        # volatilitas tahunan
    sharpe: Optional[float]         # ann_return / ann_vol (risk-free = 0)
    max_drawdown: Optional[float]   # drawdown terdalam kurva ekuitas kumulatif (negatif)
    turnover: Optional[float]       # rata-rata harian |Δbobot| / 2 ∈ [0, 1]
    hit_rate: Optional[float]       # fraksi hari dengan return long–short > 0
    n_days: int
    n_long: float                   # rata-rata jumlah saham sisi long per hari
    quantile: float                 # fraksi universe per sisi (mis. 0.1 = desil)
    cost_bps: float                 # biaya satu-arah yang dibebankan (basis poin)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        if self.error:
            return f"<Backtest ERROR {self.error[:60]}>"
        def _f(x, p=4):
            return "None" if x is None else f"{x:+.{p}f}"
        return (f"<ann={_f(self.ann_return)} sharpe={_f(self.sharpe, 2)} "
                f"mdd={_f(self.max_drawdown)} turn={_f(self.turnover, 3)} "
                f"n={self.n_days}>")


def _weights_from_scores(scores: pd.Series, quantile: float) -> pd.Series:
    """Bobot long–short ekual per hari dari skor faktor.

    Peringkat dilakukan PER HARI (cross-sectional) sesuai definisi faktor alpha:
    `quantile` teratas diberi bobot +1/n_long, `quantile` terbawah −1/n_short,
    sisanya nol. Jumlah bobot = 0 (dollar-neutral) dan jumlah |bobot| = 2,
    sehingga return harian langsung terbaca sebagai spread long−short.
    """
    def _one_day(s: pd.Series) -> pd.Series:
        s = s.dropna()
        n = len(s)
        k = int(n * quantile)
        if k < 1 or n < 3:
            return pd.Series(0.0, index=s.index)
        order = s.rank(method="first")
        w = pd.Series(0.0, index=s.index)
        w[order > n - k] = 1.0 / k       # k teratas
        w[order <= k] = -1.0 / k         # k terbawah
        return w

    return scores.groupby(level="datetime", group_keys=False).apply(_one_day)


def _max_drawdown(daily_ret: pd.Series) -> Optional[float]:
    """Drawdown terdalam kurva ekuitas kumulatif aritmetik (nilai ≤ 0)."""
    if daily_ret.empty:
        return None
    equity = daily_ret.cumsum()
    peak = equity.cummax()
    return float((equity - peak).min())


def backtest_values(
    lab: Any,
    values: pd.Series,
    *,
    quantile: float = 0.1,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Backtest deret nilai faktor yang SUDAH dihitung (hemat: tanpa eval ulang).

    `lab` adalah `eval.ic.Lab` — dipakai untuk label return dan jendela OOS-nya,
    supaya jendela di sini identik dengan jendela RankIC yang dilaporkan
    berdampingan. Tidak ada jendela kedua yang diam-diam berbeda.
    """
    empty = BacktestResult(None, None, None, None, None, None, 0, 0.0,
                           quantile, cost_bps)
    try:
        d = pd.DataFrame({"f": values, "y": lab.label})
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        if not d.empty:
            dts = d.index.get_level_values("datetime")
            d = d[(dts >= lab.oos_start) & (dts <= lab.oos_end)]
        if d.empty:
            empty.error = "empty after dropna/OOS"
            return empty

        w = _weights_from_scores(d["f"], quantile)
        # Bobot hari-t diterapkan pada label hari-t, yang di `Lab.label` sudah
        # berupa return Ref(-2)/Ref(-1)−1 (yaitu forward return, bukan return
        # hari yang sama) — jadi tidak ada look-ahead yang diperkenalkan di sini.
        gross = (w * d["y"]).groupby(level="datetime").sum()

        wide = w.unstack(level="instrument").fillna(0.0).sort_index()
        # Turnover = setengah dari total perubahan bobot absolut, jadi 1.0
        # berarti seluruh portofolio berganti isi dalam sehari.
        turnover = (wide.diff().abs().sum(axis=1) / 2.0).iloc[1:]
        net = gross.copy()
        if cost_bps:
            net = net.sub(turnover.reindex(gross.index).fillna(0.0)
                          * 2.0 * cost_bps * 1e-4, fill_value=0.0)

        n = int(net.notna().sum())
        if n < 2:
            empty.error = f"hanya {n} hari dengan return terdefinisi"
            return empty
        mu, sd = float(net.mean()), float(net.std())
        ann_ret = mu * TRADING_DAYS
        ann_vol = sd * np.sqrt(TRADING_DAYS) if sd > 0 else None
        return BacktestResult(
            ann_return=ann_ret,
            ann_vol=float(ann_vol) if ann_vol else None,
            sharpe=float(ann_ret / ann_vol) if ann_vol else None,
            max_drawdown=_max_drawdown(net.dropna()),
            turnover=float(turnover.mean()) if len(turnover) else None,
            hit_rate=float((net > 0).mean()),
            n_days=n,
            n_long=float((wide > 0).sum(axis=1).mean()),
            quantile=quantile,
            cost_bps=cost_bps,
        )
    except Exception as e:  # noqa: BLE001 — ekspresi LLM bisa gagal ribuan cara
        empty.error = f"{type(e).__name__}: {e}"
        return empty


def backtest_expression(
    lab: Any,
    expr: str,
    *,
    quantile: float = 0.1,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Evaluasi ekspresi DSL lalu backtest hasilnya."""
    try:
        values = lab.values(expr)
    except Exception as e:  # noqa: BLE001
        return BacktestResult(None, None, None, None, None, None, 0, 0.0,
                              quantile, cost_bps, error=f"{type(e).__name__}: {e}")
    return backtest_values(lab, values, quantile=quantile, cost_bps=cost_bps)


def score_expression(
    lab: Any,
    expr: str,
    *,
    quantile: float = 0.1,
    cost_bps: float = 0.0,
) -> Dict[str, Any]:
    """IC + backtest dari SATU kali evaluasi ekspresi.

    Ini jalur yang dipakai lengan faktor: mengevaluasi ekspresi dua kali (sekali
    untuk `Lab.ic`, sekali untuk `backtest_expression`) memboroskan bagian
    termahal dari skoring korpus, dan pada ekspresi rolling bersarang itu bisa
    berarti puluhan detik per ekspresi.
    """
    try:
        values = lab.values(expr)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        return {"expression": expr, "error": err, "ic": None, "backtest": None}
    ic = lab.ic_of_values(values)
    bt = backtest_values(lab, values, quantile=quantile, cost_bps=cost_bps)
    return {
        "expression": expr,
        "error": ic.error or bt.error,
        "ic": {"ic": ic.ic, "icir": ic.icir, "tstat": ic.tstat,
               "n_days": ic.n_days, "coverage": ic.coverage,
               "n_unique": ic.n_unique},
        "backtest": bt.as_dict(),
    }


if __name__ == "__main__":  # smoke manual: python backend/eval/backtest.py
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import bootstrap
    bootstrap()

    from eval.ic import Lab

    _lab = Lab(mode="fast")
    for _e in ("RANK($volume)", "-1 * TS_CORR(RANK($close), RANK($volume), 10)"):
        print(f"{_e:<50} {backtest_expression(_lab, _e)}")
