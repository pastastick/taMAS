"""
mas/operator_families.py
====================
Klasifikasi DETERMINISTIK operator DSL → family, plus penyusun "diversity hint"
untuk melawan monokultur operator (model 4B default ke RANK/TS_ZSCORE/TS_PCTCHANGE).

Dua peran:
  1. `families_of(expr)` — family apa saja yang dipakai sebuah ekspresi (dari
     nama fungsi via regex; deterministik, tanpa AST/LLM).
  2. `diversity_hint(history)` — teks prompt yang (a) menyebut family yang BARU
     SAJA dipakai, (b) mengarahkan ke family RICH yang jarang dipakai, dan (c)
     menyertakan AFORDANS tiap family (mis. "SMA/WMA = ekstraksi tren/momentum")
     sehingga model TAHU operator langka itu relevan untuk mekanisme tertentu —
     menambal celah pengetahuan tanpa mengorbankan kesetiaan ke hipotesis.

Family "RICH" = yang kronis absen (smoothing/regression/technical/math/
conditional/quantile/ts_pair). Family "BASELINE" (cross_sectional/time_series)
sengaja TIDAK dipush karena memang sudah over-used.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

# ── operator → family (sumber: operator reference prompts.yaml / function_lib) ──
FAMILIES: Dict[str, List[str]] = {
    "cross_sectional": ["RANK", "ZSCORE", "MEAN", "STD", "SKEW", "KURT",
                        "MAX", "MIN", "MEDIAN", "SCALE"],
    "time_series": ["DELTA", "DELAY", "TS_MEAN", "TS_SUM", "TS_RANK", "TS_ZSCORE",
                    "TS_MEDIAN", "TS_STD", "TS_VAR", "TS_MIN", "TS_MAX", "TS_ARGMAX",
                    "TS_ARGMIN", "TS_MAD", "SUMAC", "HIGHDAY", "LOWDAY",
                    "TS_PCTCHANGE", "TS_KURT", "TS_SKEW"],
    "ts_pair": ["TS_CORR", "TS_COVARIANCE"],
    "smoothing": ["SMA", "WMA", "EMA", "DECAYLINEAR"],
    "math": ["LOG", "SQRT", "SIGN", "EXP", "ABS", "INV", "FLOOR", "POW", "PROD"],
    "conditional": ["COUNT", "SUMIF", "FILTER"],
    "regression": ["REGBETA", "REGRESI", "SEQUENCE"],
    "technical": ["RSI", "MACD", "BB_UPPER", "BB_MIDDLE", "BB_LOWER"],
    "quantile": ["TS_QUANTILE", "PERCENTILE"],
}

OP_TO_FAMILY: Dict[str, str] = {
    op: fam for fam, ops in FAMILIES.items() for op in ops
}

# Afordans 1-baris: untuk apa family ini secara mekanisme (knowledge injection).
AFFORDANCES: Dict[str, str] = {
    "smoothing":   "trend/momentum extraction & its decay (SMA/WMA/EMA/DECAYLINEAR)",
    "regression":  "beta / lead-lag / residual vs market or another series (REGBETA/REGRESI)",
    "technical":   "momentum oscillators & volatility bands (RSI/MACD/BB_UPPER/MIDDLE/LOWER)",
    "math":        "nonlinear shaping — emphasize tails or compress scale (LOG/SQRT/POW/SIGN/ABS/INV)",
    "conditional": "regime gating & event counting (COUNT/SUMIF/FILTER, and (C)?(A):(B))",
    "ts_pair":     "co-movement of two series over a window (TS_CORR/TS_COVARIANCE)",
    "quantile":    "distributional position within a window (TS_QUANTILE/PERCENTILE)",
    "cross_sectional": "rank/normalize a signal ACROSS stocks each day (RANK/ZSCORE/…)",
    "time_series": "rolling stat of ONE series over time (TS_MEAN/TS_ZSCORE/DELTA/…)",
}

# Family yang layak DIPUSH (rich, kronis absen). Baseline tak dipush.
RICH_FAMILIES: List[str] = ["smoothing", "regression", "technical",
                            "conditional", "math", "ts_pair", "quantile"]

_FUNC_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s*\(")


def families_of(expr: str) -> Set[str]:
    """Set family operator yang dipakai `expr` (deterministik, by function name).
    Ternary (C)?(A):(B) dihitung sebagai conditional walau tanpa nama fungsi."""
    fams: Set[str] = set()
    if not expr:
        return fams
    for name in _FUNC_RE.findall(expr):
        fam = OP_TO_FAMILY.get(name)
        if fam:
            fams.add(fam)
    if "?" in expr and ":" in expr:
        fams.add("conditional")
    return fams


def rich_families_used(exprs: Sequence[str]) -> Set[str]:
    """Family RICH yang muncul di kumpulan ekspresi (untuk metrik diversitas)."""
    used: Set[str] = set()
    for e in exprs:
        used |= (families_of(e) & set(RICH_FAMILIES))
    return used


def diversity_hint(history: Sequence[Set[str]], *, k_target: int = 2) -> str:
    """Susun teks hint untuk prompt judger_reason/construct.

    history : daftar set-family per faktor yang BARU diterima (terbaru di akhir).
              Kosong → mode round-1 (push semua family rich generik).
    k_target: berapa family under-used yang ditonjolkan.

    Hint mengarahkan ke family rich yang jarang/belum dipakai DAN memberi afordans-
    nya, lalu menutup dengan klausa kesetiaan ("hanya bila benar melayani mekanisme")
    supaya diversitas tak mengorbankan konsistensi-hipotesis.
    """
    # hitung pemakaian family rich di history
    counts: Dict[str, int] = {fam: 0 for fam in RICH_FAMILIES}
    for fams in history:
        for fam in fams:
            if fam in counts:
                counts[fam] += 1

    recent_used = sorted({fam for fams in history for fam in fams})
    # under-used = family rich dengan count terkecil (0 dulu), stabil by urutan RICH
    under = sorted(RICH_FAMILIES, key=lambda f: (counts[f], RICH_FAMILIES.index(f)))
    target = under[:max(1, k_target)]

    lines: List[str] = []
    if recent_used:
        lines.append(f"Operator families used in recent factors: {', '.join(recent_used)}.")
    else:
        lines.append("Prior factor-mining over-used RANK / TS_ZSCORE / TS_PCTCHANGE "
                     "(cross-sectional & basic time-series).")
    lines.append("To avoid operator monoculture, this round PRIORITIZE at least one "
                 "operator from an under-used family below — use it ONLY where it "
                 "genuinely serves the hypothesis mechanism (do not bolt it on):")
    for fam in target:
        lines.append(f"  - {fam}: {AFFORDANCES.get(fam, '')}")
    return "\n".join(lines)


# ── HYBRID part-2: penalti diversitas pada SKOR SELEKSI (evolution) ───────────
# effective_score = primary_metric − λ · family_penalty. Diterapkan saat memilih
# parent / best, sehingga faktor yang family operatornya REDUNDAN terhadap populasi
# kalah dari yang memperkenalkan family baru — tanpa hard-reject (tetap di pool).

def trajectory_families(factors) -> Set[str]:
    """Family operator GABUNGAN semua ekspresi faktor sebuah trajectory.
    `factors`: list[dict {expression}] | list[str] | str tunggal."""
    if isinstance(factors, str):
        return families_of(factors)
    fams: Set[str] = set()
    for f in factors or []:
        expr = f.get("expression", "") if isinstance(f, dict) else str(f)
        fams |= families_of(expr)
    return fams


def population_family_counts(family_sets: Iterable[Set[str]]) -> Counter:
    """Berapa trajectory yang memuat tiap family (penilai redundansi populasi)."""
    c: Counter = Counter()
    for fams in family_sets:
        for fam in fams:
            c[fam] += 1
    return c


def family_penalty(my_families: Set[str], pop_counts: Counter, n_pop: int) -> float:
    """Penalti redundansi family ∈ [0,1]. Dinilai dari family TERLANGKA yang dipakai
    faktor ini (min rarity = count/n_pop): memperkenalkan SATU family langka sudah
    menyelamatkan dari penalti (reward NOVELTY, bukan menghukum yang juga memakai
    family umum). Faktor yang SEMUA family-nya ubiquit → penalti ~1."""
    if not my_families or n_pop <= 0:
        return 0.0
    return min(pop_counts.get(f, 0) / n_pop for f in my_families)


def diversity_penalized(
    items: list,
    get_families: "Callable[[object], Set[str]]",
    get_metric: "Callable[[object], float]",
    lam: float,
) -> "List[Tuple[object, float]]":
    """[(item, effective_score)] dengan effective = metric − λ·family_penalty,
    penalty relatif ke distribusi family SELURUH `items`. λ≤0 → no-op (metric apa
    adanya). Pure → tiap titik seleksi tinggal sort by effective score, descending."""
    if lam <= 0:
        return [(it, (get_metric(it) or 0.0)) for it in items]
    fam_list = [get_families(it) for it in items]
    counts = population_family_counts(fam_list)
    n = len(items)
    out: List[Tuple[object, float]] = []
    for it, fams in zip(items, fam_list):
        m = get_metric(it) or 0.0
        out.append((it, m - lam * family_penalty(fams, counts, n)))
    return out
