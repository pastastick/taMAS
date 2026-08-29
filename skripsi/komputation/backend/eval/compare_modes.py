"""Tahap 0 — bandingkan kapasitas kanal laten antar persamaan langkah laten.

Pertanyaan yang dijawab (dan HANYA itu): apakah mengganti persamaan langkah
laten LatentMAS dari ridge `W_a` resmi paper (`raw`) ke proyeksi convex-hull
(`soft`) / proyeksi + noise Gumbel (`gumbel`) menaikkan berapa banyak muatan
SIMBOLIK yang benar-benar lewat kanal laten murni (`kv_latent_only`)?

Perbandingannya BERPASANGAN. `channel_capacity.py` membangkitkan muatan dengan
`random.Random(seed)` sebelum lengan mana pun dijalankan, jadi dua run dengan
`--seed`/`--k`/`--trials`/`--payload` yang sama melihat muatan yang identik,
trial-per-trial. Skrip ini MEMVERIFIKASI kesamaan itu dulu (bukan
mengasumsikannya) lalu menguji selisihnya sebagai data berpasangan:

  recall  → Wilcoxon signed-rank (kontinu, k+1 nilai diskret, tak normal)
  exact   → McNemar exact (biner; hanya pasangan yang berbeda yang informatif)

Keduanya dilaporkan bersama efek mentahnya + CI bootstrap, karena pada n=20 nilai
p sendirian menyesatkan ke dua arah.

    python backend/eval/compare_modes.py                     # semua yang ada di results/probe
    python backend/eval/compare_modes.py --arm kv_full       # lengan lain
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from itertools import combinations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ensure_out, OUT_PROBE as _OUT
from eval.stats import boot_ci, mcnemar, wilcoxon

OUT = ensure_out(_OUT)


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for p in sorted(paths):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! lewati {p.name}: {e}")
            continue
        if "_meta" not in d or "records" not in d:
            continue
        d["_path"] = p
        runs.append(d)
    return runs


def key_of(meta: dict) -> tuple:
    """Sel eksperimen: dua run sebanding HANYA jika seluruh kunci ini sama."""
    return (meta.get("model"), meta.get("k"), meta.get("trials"),
            meta.get("latent_steps"), meta.get("seed"))


def series(run: dict, arm: str, kind: str) -> tuple[list, list, list]:
    """(truth, recall, exact) terurut menurut nomor trial."""
    rec = [r for r in run["records"]
           if r.get("arm") == arm and r.get("payload_kind") == kind
           and "error" not in r]
    rec.sort(key=lambda r: r["trial"])
    return ([tuple(r["truth"]) for r in rec],
            [r["recall"] for r in rec],
            [r["exact"] for r in rec])


def by_position(run: dict, arm: str, kind: str, k: int) -> list[float]:
    rec = [r for r in run["records"]
           if r.get("arm") == arm and r.get("payload_kind") == kind
           and "error" not in r]
    if not rec:
        return []
    return [round(sum(1 for r in rec if r["truth"][i] in r["pred"]) / len(rec), 2)
            for i in range(k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="channel_capacity_*.json")
    ap.add_argument("--arm", default="kv_latent_only")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    runs = load_runs(list(OUT.glob(a.glob)))
    if not runs:
        raise SystemExit(f"tak ada berkas cocok di {OUT}")

    # Kelompokkan per sel; di dalam satu sel, mode adalah satu-satunya yang beda.
    cells: dict[tuple, dict[str, dict]] = {}
    for r in runs:
        mode = r["_meta"].get("latent_mode", "?")
        # `use_realign` hanya membedakan sesuatu di mode raw. Berkas lama tak
        # memuat kuncinya; semuanya dijalankan dengan realign aktif → default True.
        if mode == "raw" and not r["_meta"].get("use_realign", True):
            mode = "raw(M=I)"
        # beta adalah sumbu bebas untuk mode "moi" (bukan biner) — setiap nilai
        # HARUS jadi label mode terpisah, atau sweep beta akan saling menimpa
        # satu sama lain di dalam satu sel dan sweep-nya tak pernah terlihat.
        if mode == "moi":
            mode = f"moi(β={r['_meta'].get('latent_beta', 1.0):g})"
        cell = cells.setdefault(key_of(r["_meta"]), {})
        if mode in cell:
            print(f"  ! dua run mode={mode} untuk sel yang sama; "
                  f"pakai {r['_path'].name}, abaikan {cell[mode]['_path'].name}")
        cell[mode] = r

    print(f"\nberkas terbaca: {len(runs)} | lengan diuji: {a.arm}\n")
    report = []

    for cell_key, by_mode in sorted(cells.items(), key=lambda kv: str(kv[0])):
        model, k, trials, m, seed = cell_key
        print(f"── {model}  k={k} trials={trials} m={m} seed={seed}")
        print(f"   mode tersedia: {', '.join(sorted(by_mode))}")

        kinds = sorted({r["payload_kind"] for run in by_mode.values()
                        for r in run["records"] if "payload_kind" in r})

        for kind in kinds:
            print(f"\n   muatan `{kind}`")
            print(f"   {'mode':10s} {'n':>3s} {'recall':>7s} {'exact':>7s} "
                  f"{'halus':>7s}  posisi")
            for mode in sorted(by_mode):
                truth, rec, exa = series(by_mode[mode], a.arm, kind)
                if not rec:
                    print(f"   {mode:10s}   – (lengan/muatan tak ada)")
                    continue
                hal = [r["hallucinate"] for r in by_mode[mode]["records"]
                       if r.get("arm") == a.arm and r.get("payload_kind") == kind
                       and "error" not in r]
                pos = by_position(by_mode[mode], a.arm, kind, k)
                print(f"   {mode:10s} {len(rec):3d} {st.mean(rec):7.3f} "
                      f"{st.mean(exa):7.3f} {st.mean(hal):7.3f}  "
                      f"[{' '.join(f'{p:.2f}' for p in pos)}]")

            for m1, m2 in combinations(sorted(by_mode), 2):
                t1, r1, e1 = series(by_mode[m1], a.arm, kind)
                t2, r2, e2 = series(by_mode[m2], a.arm, kind)
                if not r1 or not r2:
                    continue
                if len(r1) != len(r2):
                    print(f"   ! {m1} vs {m2}: jumlah trial beda "
                          f"({len(r1)} vs {len(r2)}) — dilewati")
                    continue
                if t1 != t2:
                    n_beda = sum(1 for x, y in zip(t1, t2) if x != y)
                    print(f"   ! {m1} vs {m2}: muatan TIDAK identik "
                          f"({n_beda}/{len(t1)} trial beda) — bukan berpasangan, "
                          f"uji dilewati")
                    continue
                pw, nnz = wilcoxon(r1, r2)
                pm, b01, b10 = mcnemar(e1, e2)
                lo, hi = boot_ci(r1, r2)
                d = st.mean(r1) - st.mean(r2)
                print(f"   {m1} − {m2}: Δrecall={d:+.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}] Wilcoxon p={pw:.3f} (n≠={nnz}) | "
                      f"Δexact={st.mean(e1) - st.mean(e2):+.3f} "
                      f"McNemar p={pm:.3f} ({b01}/{b10})")
                report.append({
                    "model": model, "k": k, "trials": trials, "m": m, "seed": seed,
                    "arm": a.arm, "payload": kind, "mode_a": m1, "mode_b": m2,
                    "recall_a": round(st.mean(r1), 3), "recall_b": round(st.mean(r2), 3),
                    "d_recall": round(d, 3), "ci95": [round(lo, 3), round(hi, 3)],
                    "p_wilcoxon": round(pw, 4), "n_nonzero": nnz,
                    "exact_a": round(st.mean(e1), 3), "exact_b": round(st.mean(e2), 3),
                    "p_mcnemar": round(pm, 4), "discordant": [b01, b10],
                })
        print()

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"tersimpan → {a.out}")


if __name__ == "__main__":
    main()
