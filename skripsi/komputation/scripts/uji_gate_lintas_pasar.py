#!/usr/bin/env python3
"""Uji apakah keputusan EXECUTION GATE berpindah pasar.

KENAPA UJI INI ADA. Klaim utama skripsi (laju lolos gate keluarga R vs `raw`)
diukur saat generasi, dan satu-satunya komponen generasi yang menyentuh data
pasar adalah `gate/execution_gate.py`: ia mengevaluasi tiap ekspresi pada
sampel kecil (A-share 2018) dan menolak yang menghasilkan kolom mati/konstan.
Karena data pasar diganti ke IDX, seorang penguji berhak bertanya: apakah
keputusan gate itu masih berlaku, atau laju lolosnya hanya berlaku untuk pasar
Tiongkok?

Skrip ini menjawabnya secara langsung: jalankan gate yang SAMA pada panel IDX,
lalu bandingkan keputusannya dengan yang tercatat di `frontend_*.json`.

Yang dilaporkan:
  sama       - keputusan identik di kedua pasar
  beda       - keputusan berbeda (inilah angka yang dilaporkan sebagai batas)
  per_sel    - rincian per sel supaya bisa dilihat apakah ketidaksesuaian
               terkonsentrasi di satu metode (yang akan mengancam klaim) atau
               tersebar merata (yang tidak).

Pemakaian:
    LAB_PV_FILE=backend/hf_data_id/daily_pv_idx_lq45.h5 \
    LAB_GATE_SAMPLE_START=2019-01-01 LAB_GATE_SAMPLE_END=2019-12-31 \
    PYTHONPATH=backend .venv/bin/python scripts/uji_gate_lintas_pasar.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from paths import bootstrap, OUT_FACTOR, RESULTS  # noqa: E402
bootstrap()


def main() -> None:
    from gate.execution_gate import get_execution_gate, _SOURCE, _SAMPLE_START, _SAMPLE_END

    gate = get_execution_gate()
    if gate is None:
        raise SystemExit("gate dinonaktifkan (LATENTMAS_EXEC_GATE=0)")
    print(f"panel  : {_SOURCE}")
    print(f"sampel : {_SAMPLE_START} .. {_SAMPLE_END}")
    if gate.df is None:
        raise SystemExit("gate tak bisa memuat data — periksa LAB_PV_FILE / jendela sampel")
    print(f"sampel dimuat: {len(gate.df):,} baris, "
          f"{gate.df.index.get_level_values('instrument').nunique()} emiten\n")

    per_sel: list[dict] = []
    total_sama = total_beda = 0
    ringkas_beda: list[dict] = []
    cache: dict[str, bool] = {}

    for path in sorted(OUT_FACTOR.glob("frontend_*.json")):
        tag = path.stem[len("frontend_"):]
        doc = json.loads(path.read_text())
        sama = beda = 0
        t0 = time.time()
        for r in doc["runs"]:
            lolos_tercatat = set(r.get("passing") or [])
            for f in (r.get("factors") or []):
                e = f.get("expression", "")
                if not e:
                    continue
                lama = e in lolos_tercatat
                if e not in cache:
                    try:
                        ok, _ = gate.check(e)
                    except Exception:  # noqa: BLE001
                        ok = False
                    cache[e] = bool(ok)
                baru = cache[e]
                if lama == baru:
                    sama += 1
                else:
                    beda += 1
                    if len(ringkas_beda) < 40:
                        ringkas_beda.append({"tag": tag, "expression": e[:110],
                                             "lolos_a_share": lama, "lolos_idx": baru})
        n = sama + beda
        per_sel.append({"tag": tag, "ekspresi": n, "sama": sama, "beda": beda,
                        "kesesuaian": (sama / n if n else None)})
        total_sama += sama
        total_beda += beda
        print(f"{tag:28s} n={n:4d}  sama={sama:4d}  beda={beda:3d}  "
              f"kesesuaian={sama / n if n else 0:.3f}  ({time.time() - t0:.0f}s)")

    n = total_sama + total_beda
    out = {"panel": str(_SOURCE), "sampel": [_SAMPLE_START, _SAMPLE_END],
           "total": {"ekspresi": n, "sama": total_sama, "beda": total_beda,
                     "kesesuaian": (total_sama / n if n else None)},
           "per_sel": per_sel, "contoh_beda": ringkas_beda}
    f = RESULTS / "factor" / "gate_lintas_pasar.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"\nTOTAL n={n}  sama={total_sama}  beda={total_beda}  "
          f"kesesuaian={total_sama / n if n else 0:.4f}")
    print(f"laporan → {f}")


if __name__ == "__main__":
    main()
