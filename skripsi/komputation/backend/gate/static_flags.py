"""Cacat ekspresi yang bisa dideteksi TANPA mengeksekusi ekspresi.

Asalnya `lab/audit_batch.py` (audit batch 2026-07-05). Berkas itu ikut terhapus
di rombakan `9d4e0bf` bersama seluruh `lab/`, TAPI `factor/run_factor.py`
masih mengimpor `static_flags` darinya — sehingga seluruh tahap skoring lengan
faktor mati dengan `ModuleNotFoundError: No module named 'lab'`. Fungsi ini
dipulihkan di sini (verbatim dari commit `06d6b1c`), bukan di `lab/`, karena
ia dipakai berdampingan dengan `validate_semantics` dari `gate.factor_regulator`
dan memang berjenis sama: penilaian mutu ekspresi.

Sisa `lab/audit_batch.py` (fungsi `collect()`/`main()`) SENGAJA tidak dibawa —
keduanya membaca `backend/runs/prod_*/trajectory_pool*.json`, artefak run lama
yang juga sudah dihapus di rombakan yang sama.

Bedanya dengan `validate_semantics`: yang itu memutuskan LOLOS/TOLAK gate,
yang ini hanya MENANDAI pola mencurigakan untuk analisis — sebuah flag tidak
menggugurkan faktor, ia menjelaskan kenapa IC-nya mungkin jelek.
"""
from __future__ import annotations

import re

# Operator time-series yang butuh window > 1 agar tidak degenerate.
# Nilainya = posisi argumen window (0-indexed) di dalam pemanggilan.
TS_OPS = {
    "TS_ZSCORE": 1, "TS_RANK": 1, "TS_MEAN": 1, "TS_MEDIAN": 1, "TS_STD": 1,
    "TS_VAR": 1, "TS_MAX": 1, "TS_MIN": 1, "TS_SUM": 1, "TS_ARGMAX": 1,
    "TS_ARGMIN": 1, "TS_SKEW": 1, "TS_KURT": 1, "TS_MAD": 1, "TS_CORR": 2,
    "TS_COVARIANCE": 2, "TS_QUANTILE": 1,
}
# Window minimal agar statistiknya terdefinisi (std butuh >= 2, skew >= 3,
# kurtosis >= 4).
MIN_WIN = {"TS_ZSCORE": 2, "TS_STD": 2, "TS_VAR": 2, "TS_CORR": 2,
           "TS_COVARIANCE": 2, "TS_SKEW": 3, "TS_KURT": 4, "TS_MAD": 2}


def static_flags(expr: str) -> list[str]:
    """Cacat yang bisa dideteksi TANPA menjalankan ekspresi."""
    flags = []
    # 1. window degenerate: TS_OP(..., 1) atau window < minimum statistik
    for op, wpos in TS_OPS.items():
        for m in re.finditer(rf"\b{op}\s*\(", expr):
            args, depth, i = [], 0, m.end()
            cur = ""
            while i < len(expr):
                c = expr[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth == 0:
                        args.append(cur)
                        break
                    depth -= 1
                elif c == "," and depth == 0:
                    args.append(cur)
                    cur = ""
                    i += 1
                    continue
                cur += c
                i += 1
            if len(args) > wpos:
                w = args[wpos].strip()
                if re.fullmatch(r"\d+", w):
                    wi = int(w)
                    need = MIN_WIN.get(op, 2)
                    if wi < need:
                        flags.append(f"degenerate-window:{op}(w={wi}<{need})")
    # 2. ternary/kondisi dengan ekspresi kontinu (bukan perbandingan) sebagai syarat
    for m in re.finditer(r"([^?]*)\?", expr):
        cond = m.group(1)
        cond = cond[max(cond.rfind("("), cond.rfind(":")) + 1:].strip()
        if cond and not re.search(r"[<>=!]", cond):
            flags.append("nonboolean-condition")
            break
    # 3. threshold absolut pada besaran yang tidak sebanding lintas saham
    if re.search(r"\$volume[^)]*\)?\s*[<>]\s*\d{4,}", expr) or re.search(
            r"TS_(MEAN|MIN|MAX|SUM)\(\s*\$volume[^)]*\)\s*[<>]\s*\d{4,}", expr):
        flags.append("absolute-volume-threshold")
    # 4. perbandingan pada rank persentil dengan ambang > 1 (TS_RANK pct=True ∈ [0,1])
    for m in re.finditer(r"TS_RANK\([^)]*\)\s*[<>]=?\s*(\d+(?:\.\d+)?)", expr):
        if float(m.group(1)) > 1.0:
            flags.append("pct-rank-vs-threshold>1")
    return sorted(set(flags))
