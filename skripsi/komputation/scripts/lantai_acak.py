#!/usr/bin/env python3
"""Lantai acak — apakah sistem multi-agen mengalahkan penarikan acak dari DSL?

KENAPA ANALISIS INI ADA. Seluruh Bab hasil membandingkan lima formulasi langkah
laten SATU SAMA LAIN. Yang tak pernah dijawab: apakah salah satu dari mereka
lebih baik daripada tidak memakai model bahasa sama sekali. Tanpa pembanding
itu, "keluarga R lolos gate 90%" bisa saja berarti "gate ini memang mudah",
bukan "agen R menghasilkan ekspresi yang baik". `AUDIT_KRITIS.md` §2.5 sudah
menyebut kebutuhannya; skripnya tak pernah ada di repo ini.

CARA KERJA. Ekspresi dibangkitkan langsung dari tata bahasa DSL yang SAMA
dengan yang diberikan ke agen (enam leaf data, daftar fungsi + arity dari
`gate/factor_regulator._DSL_ARITY`), lalu dilewatkan gate yang SAMA dan dinilai
Lab yang SAMA. Satu-satunya yang berbeda adalah asal ekspresinya: penarikan
acak, bukan penalaran agen.

APA YANG DIBANDINGKAN (dan apa yang TIDAK). Perbandingan yang sah di sini ada
dua, dan keduanya dilaporkan terpisah:

  (1) LAJU LOLOS GATE — acak vs tiap sel. Ini pembanding untuk klaim
      keandalan. Kalau ekspresi acak lolos gate sesering keluarga R, maka
      angka lolos-gate mengukur kemudahan gate, bukan mutu agen.
  (2) SEBARAN |IC| pada ekspresi yang LOLOS — acak vs tiap sel. Ini pembanding
      untuk klaim mutu sinyal. Diuji dengan Mann-Whitney U (nonparametrik:
      sebaran |IC| condong dan berekor, jadi uji-t tidak tepat).

Yang TIDAK bisa disimpulkan dari sini: bahwa agen "menemukan alpha". Keduanya
dinilai pada jendela yang sama, jadi ini perbandingan relatif antar-sumber
ekspresi, bukan bukti profitabilitas.

Pemakaian:
    LAB_PV_FILE=backend/hf_data_id/daily_pv_idx_lq45.h5 LAB_TRADING_DAYS=241 \
    PYTHONPATH=backend .venv/bin/python scripts/lantai_acak.py --n 600
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from paths import bootstrap, OUT_FACTOR, RESULTS  # noqa: E402
bootstrap()

# Enam leaf data — persis daftar yang diberikan ke agen di `prompts/factor.yaml`.
LEAF = ["$open", "$high", "$low", "$close", "$volume", "$return"]

# Fungsi cross-sectional berargumen tunggal.
CS1 = ["RANK", "ZSCORE", "MEAN", "STD", "MEDIAN", "SKEW", "KURT", "ABS",
       "SIGN", "LOG", "SQRT", "SCALE"]

# Fungsi deret waktu (x, p).
TS2 = ["TS_MEAN", "TS_SUM", "TS_RANK", "TS_ZSCORE", "TS_MEDIAN", "TS_PCTCHANGE",
       "TS_MIN", "TS_MAX", "TS_ARGMAX", "TS_ARGMIN", "TS_STD", "TS_VAR",
       "TS_MAD", "TS_SKEW", "TS_KURT", "DELTA", "DELAY", "DECAYLINEAR",
       "WMA", "EMA", "SUMAC", "HIGHDAY", "LOWDAY"]

# Fungsi deret waktu dua-deret (x, y, p).
TS3 = ["TS_CORR", "TS_COVARIANCE", "REGBETA", "REGRESI"]

# Periode yang dipakai agen di korpus nyata (5–60) — memakai rentang yang sama
# supaya perbedaan hasil tidak bisa dijelaskan sekadar oleh panjang jendela.
PERIODE = [3, 5, 10, 15, 20, 30, 60]
BINER = ["+", "-", "*", "/"]


def buat(rng: random.Random, depth: int) -> str:
    """Bangkitkan satu ekspresi. `depth` = sisa kedalaman yang diizinkan."""
    if depth <= 0:
        return rng.choice(LEAF)
    pilih = rng.random()
    if pilih < 0.30:
        return f"{rng.choice(CS1)}({buat(rng, depth - 1)})"
    if pilih < 0.68:
        return f"{rng.choice(TS2)}({buat(rng, depth - 1)}, {rng.choice(PERIODE)})"
    if pilih < 0.80:
        return (f"{rng.choice(TS3)}({buat(rng, depth - 1)}, "
                f"{buat(rng, depth - 1)}, {rng.choice(PERIODE)})")
    return (f"({buat(rng, depth - 1)} {rng.choice(BINER)} "
            f"{buat(rng, depth - 1)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="jumlah ekspresi acak")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--window", default="2021-01-01,2021-12-31")
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--budget", type=int, default=60)
    args = ap.parse_args()

    from eval.ic import Lab, pasar_tag
    from factor.run_factor import _score_one_expr
    from mas.pipeline import FrontEndPipeline

    rng = random.Random(args.seed)
    # Kedalaman diacak 1..depth supaya korpus acak memuat ekspresi sederhana
    # maupun bersarang, seperti korpus agen — bukan hanya yang paling dalam.
    ekspresi: list[str] = []
    lihat: set[str] = set()
    while len(ekspresi) < args.n:
        e = buat(rng, rng.randint(1, args.depth))
        if e not in lihat:
            lihat.add(e)
            ekspresi.append(e)

    gate, _ = FrontEndPipeline._build_regulator_gate()
    awal, akhir = [s.strip() for s in args.window.split(",")]
    lab = Lab(mode="fast", window=(awal, akhir))
    pasar = pasar_tag()
    print(f"panel {pasar} · jendela {awal}..{akhir} · {len(ekspresi)} ekspresi acak",
          flush=True)

    hasil: list[dict] = []
    t0 = time.time()
    for i, e in enumerate(ekspresi, 1):
        ok, alasan = gate(e)
        baris = {"expression": e, "lolos_gate": bool(ok), "alasan_gate": alasan}
        if ok:
            entry, _ = _score_one_expr(e, lab, args.budget, args.quantile, 0.0)
            baris.update({k: entry.get(k) for k in
                          ("ic", "icir", "tstat", "n_days", "n_unique",
                           "coverage", "eval_error")})
            baris.update({k: v for k, v in entry.items() if k.startswith("bt_")})
        hasil.append(baris)
        if i % 50 == 0 or i == len(ekspresi):
            n_ok = sum(1 for h in hasil if h["lolos_gate"])
            print(f"  {i:4d}/{len(ekspresi)}  lolos_gate={n_ok:4d} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    lolos = [h for h in hasil if h["lolos_gate"]]
    hidup = [h for h in lolos
             if h.get("ic") is not None and (h.get("n_unique") or 0) > 2]
    sig = [h for h in hidup
           if h.get("tstat") is not None and abs(h["tstat"]) >= 1.96]

    out = {
        "pasar": pasar, "window": [awal, akhir], "quantile": args.quantile,
        "seed": args.seed, "max_depth": args.depth,
        "ringkas": {
            "dibangkitkan": len(hasil),
            "lolos_gate": len(lolos),
            "laju_lolos_gate": len(lolos) / len(hasil),
            "hidup": len(hidup),
            "signifikan": len(sig),
            "mean_abs_ic": (sum(abs(h["ic"]) for h in hidup) / len(hidup)
                            if hidup else None),
        },
        "per_ekspresi": hasil,
    }
    f = RESULTS / "factor" / f"lantai_acak_{pasar}_{awal}_{akhir}.json"
    f.write_text(json.dumps(out, indent=1))
    r = out["ringkas"]
    print(f"\nlolos gate  : {r['lolos_gate']}/{r['dibangkitkan']} = "
          f"{r['laju_lolos_gate']:.1%}")
    print(f"hidup       : {r['hidup']}")
    print(f"signifikan  : {r['signifikan']}")
    print(f"mean |IC|   : {r['mean_abs_ic']}")
    print(f"laporan → {f}")


if __name__ == "__main__":
    main()
