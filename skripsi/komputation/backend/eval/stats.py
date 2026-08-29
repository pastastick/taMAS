"""Uji berpasangan yang dipakai SEMUA lengan skripsi.

Ketiga fungsi ini lahir di `compare_channel_modes.py` (Tahap 0) dan angka yang
mereka hasilkan sudah dikutip di `docs/HASIL_TAHAP0.md`. Ia dipindah ke sini,
apa adanya, supaya lengan benchmark dan lengan faktor memakai uji yang SAMA —
bukan reimplementasi kedua yang bisa diam-diam berbeda konvensinya (dua sisi vs
satu sisi, penanganan seri, jumlah iterasi bootstrap) dan membuat dua bab
skripsi tak bisa dibandingkan.

Kenapa uji BERPASANGAN, bukan uji dua-sampel: setiap sel eksperimen dijalankan
atas himpunan item yang identik (muatan yang sama di Tahap 0, soal yang sama di
lengan benchmark lewat `--sample-seed`, arah+seed yang sama di lengan faktor).
Uji berpasangan memanfaatkan itu dan membuang varians antar-item, yang pada
n≈20–200 adalah selisih antara "terdeteksi" dan "tenggelam di derau".

  wilcoxon  → skor kontinu/ordinal (recall, IC)
  mcnemar   → hasil biner (exact-match, benar/salah per soal)
  boot_ci   → CI 95% selisih rata-rata, dilaporkan BERSAMA p karena pada n kecil
              nilai p sendirian menyesatkan ke dua arah
"""
from __future__ import annotations

import random
import statistics as st
from typing import List, Sequence, Tuple


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, int]:
    """Wilcoxon signed-rank dua sisi; return (p, n_nonzero). scipy bila ada."""
    d = [x - y for x, y in zip(a, b) if x != y]
    if not d:
        return 1.0, 0
    try:
        from scipy.stats import wilcoxon as _w
        return float(_w(a, b, zero_method="wilcox").pvalue), len(d)
    except Exception:  # noqa: BLE001
        # Fallback permutasi tanda (eksak untuk n kecil, deterministik).
        obs = abs(sum(d))
        rng = random.Random(0)
        hits = sum(1 for _ in range(20000)
                   if abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12)
        return hits / 20000, len(d)


def mcnemar(a: Sequence[float], b: Sequence[float]) -> Tuple[float, int, int]:
    """McNemar eksak (binomial dua sisi) atas pasangan yang berbeda.

    Return (p, b01, b10) dengan b01 = jumlah item yang BENAR di `a` tapi salah
    di `b`, dan b10 kebalikannya. Pasangan yang sama-sama benar atau sama-sama
    salah tidak membawa informasi dan memang tidak dihitung.
    """
    b01 = sum(1 for x, y in zip(a, b) if x > y)
    b10 = sum(1 for x, y in zip(a, b) if x < y)
    n = b01 + b10
    if n == 0:
        return 1.0, b01, b10
    from math import comb
    lo = min(b01, b10)
    p = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    return min(1.0, p), b01, b10


def boot_ci(a: Sequence[float], b: Sequence[float],
            iters: int = 20000) -> Tuple[float, float]:
    """CI 95% bootstrap berpasangan untuk mean(a) − mean(b)."""
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return 0.0, 0.0
    rng = random.Random(0)
    n = len(d)
    means = sorted(st.mean(rng.choice(d) for _ in range(n)) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def paired_report(a: Sequence[float], b: Sequence[float],
                  *, binary: bool = False) -> dict:
    """Ringkasan lengkap satu perbandingan berpasangan (delta + CI + p)."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError(f"panjang tak sama: {len(a)} vs {len(b)}")
    delta = st.mean(a) - st.mean(b) if a else 0.0
    lo, hi = boot_ci(a, b)
    out = {"n": len(a), "mean_a": st.mean(a) if a else None,
           "mean_b": st.mean(b) if b else None,
           "delta": delta, "ci95": [lo, hi]}
    if binary:
        p, b01, b10 = mcnemar(a, b)
        out.update({"test": "mcnemar_exact", "p": p, "b01": b01, "b10": b10})
    else:
        p, n_nz = wilcoxon(a, b)
        out.update({"test": "wilcoxon", "p": p, "n_nonzero": n_nz})
    return out


__all__ = ["wilcoxon", "mcnemar", "boot_ci", "paired_report"]
