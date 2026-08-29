"""Analisis berpasangan antar-sel lengan benchmark.

Membaca semua `results/bench/bench_*.json`, mengelompokkannya per (benchmark ×
jumlah soal × sample_seed), MEMVERIFIKASI bahwa sel-sel dalam satu kelompok
benar-benar melihat soal yang sama (bukan mengasumsikannya), lalu menguji tiap
pasangan sel dengan McNemar eksak + CI bootstrap 95% — metodologi yang sama
dengan `eval/compare_modes.py` untuk Tahap 0, lewat `eval/stats.py` yang sama.

Verifikasi kesamaan soal itu bukan formalitas. `--limit`, `--sample-seed`, dan
versi dataset di Hub semuanya bisa menggeser himpunan soal tanpa bersuara, dan
uji berpasangan atas dua himpunan yang berbeda menghasilkan angka yang terlihat
sah tapi tidak berarti apa-apa. Sel yang tak cocok DIKELUARKAN dari kelompok
dan dilaporkan, bukan dipaksa masuk.

Dua metrik diuji terpisah, sesuai disosiasi yang diusung skripsi
(`docs/HASIL_TAHAP4.md` §B2):
  correct    — benar/salah jawaban  → mutu penalaran
  format_ok  — jawaban keluar dalam format yang diminta → keandalan format

    python backend/bench/compare.py
    python backend/bench/compare.py --task gsm8k --out results/bench/analisis.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import bootstrap, ensure_out, OUT_BENCH
bootstrap()

from eval.stats import boot_ci, mcnemar  # noqa: E402

OUT = ensure_out(OUT_BENCH)


def cell_label(meta: dict) -> str:
    comm = "baseline" if meta.get("baseline") else meta.get("comm_mode")
    lab = f"{meta.get('latent_mode')}/{comm}"
    if meta.get("latent_steps") is not None:
        lab += f"/m{meta['latent_steps']}"
    return lab


def group_key(meta: dict) -> Tuple:
    """Sel hanya sebanding bila benchmark, jumlah soal, dan seed sampel sama."""
    return (meta.get("task"), meta.get("limit"), meta.get("sample_seed"),
            meta.get("model"))


def item_fingerprint(results: List[dict]) -> str:
    """Sidik jari himpunan soal — mendeteksi sel yang diam-diam melihat soal lain."""
    h = hashlib.sha256()
    for r in results:
        h.update(str(r.get("index")).encode())
        h.update(b"\x00")
        h.update((r.get("question") or "").encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


def load_cells(paths: List[Path]) -> List[dict]:
    cells = []
    for p in sorted(paths):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! lewati {p.name}: {e}")
            continue
        if "_meta" not in d or "results" not in d:
            print(f"  ! lewati {p.name}: bukan keluaran run_bench")
            continue
        d["_path"] = p
        d["_label"] = cell_label(d["_meta"])
        d["_fp"] = item_fingerprint(d["results"])
        cells.append(d)
    return cells


def _vec(cell: dict, field: str) -> List[float]:
    return [1.0 if r.get(field) else 0.0 for r in cell["results"]]


def analyse(cells: List[dict]) -> dict:
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for c in cells:
        groups[group_key(c["_meta"])].append(c)

    report = {"groups": [], "excluded": []}
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        task, limit, sseed, model = key
        # Sidik jari mayoritas = acuan; sel lain yang beda dikeluarkan.
        counts: Dict[str, int] = defaultdict(int)
        for m in members:
            counts[m["_fp"]] += 1
        ref_fp = max(counts, key=lambda k: counts[k])
        keep = [m for m in members if m["_fp"] == ref_fp]
        for m in members:
            if m["_fp"] != ref_fp:
                report["excluded"].append({
                    "file": m["_path"].name, "cell": m["_label"],
                    "reason": f"himpunan soal berbeda (fp={m['_fp']} vs {ref_fp})",
                })

        g = {
            "task": task, "limit": limit, "sample_seed": sseed, "model": model,
            "n_items": len(keep[0]["results"]) if keep else 0,
            "fingerprint": ref_fp,
            "cells": [{"cell": m["_label"], "file": m["_path"].name,
                       **m["summary"],
                       "time_s": m["_meta"].get("total_time_s")}
                      for m in sorted(keep, key=lambda x: x["_label"])],
            "pairs": [],
        }
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                a, b = keep[i], keep[j]
                row = {"a": a["_label"], "b": b["_label"]}
                for field in ("correct", "format_ok"):
                    va, vb = _vec(a, field), _vec(b, field)
                    p, b01, b10 = mcnemar(va, vb)
                    lo, hi = boot_ci(va, vb)
                    row[field] = {
                        "delta": sum(va) / len(va) - sum(vb) / len(vb),
                        "ci95": [lo, hi], "p_mcnemar": p,
                        "a_only": b01, "b_only": b10,
                    }
                g["pairs"].append(row)
        g["pairs"].sort(key=lambda r: r["correct"]["p_mcnemar"])
        report["groups"].append(g)
    return report


def _fmt(x, p=4):
    return "  None" if x is None else f"{x:+.{p}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(OUT), help="direktori hasil run_bench")
    ap.add_argument("--task", default="", help="saring satu benchmark saja")
    ap.add_argument("--out", default="", help="tulis laporan JSON ke path ini")
    args = ap.parse_args()

    paths = sorted(Path(args.dir).glob("bench_*.json"))
    if args.task:
        paths = [p for p in paths if p.name.startswith(f"bench_{args.task}_")]
    if not paths:
        print(f"tak ada bench_*.json di {args.dir}")
        return

    cells = load_cells(paths)
    report = analyse(cells)

    for g in report["groups"]:
        print(f"\n══ {g['task']}  n={g['n_items']}  seed_sampel={g['sample_seed']}"
              f"  ({g['fingerprint']}) ══")
        print(f"{'sel':<28}{'akurasi':>9}{'format':>9}{'benar':>7}{'detik':>9}")
        for c in g["cells"]:
            print(f"{c['cell']:<28}{c['accuracy']:>9.4f}{c['format_rate']:>9.4f}"
                  f"{c['n_correct']:>7}{(c['time_s'] or 0):>9.0f}")
        if g["pairs"]:
            print(f"\n{'perbandingan (akurasi)':<44}{'Δ':>9}{'CI95':>22}{'p':>9}")
            for r in g["pairs"]:
                d = r["correct"]
                ci = f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
                star = " *" if d["p_mcnemar"] < 0.05 else ""
                print(f"{r['a'] + ' − ' + r['b']:<44}{_fmt(d['delta'])}"
                      f"{ci:>22}{d['p_mcnemar']:>9.4f}{star}")

    if report["excluded"]:
        print("\n! sel DIKELUARKAN (himpunan soal tak cocok):")
        for e in report["excluded"]:
            print(f"  {e['file']}: {e['reason']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nlaporan → {args.out}")


if __name__ == "__main__":
    main()
