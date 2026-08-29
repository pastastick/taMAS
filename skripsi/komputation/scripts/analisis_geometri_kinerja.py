#!/usr/bin/env python3
"""Korelasi geometri (b7_probe) vs kinerja hilir — kuantifikasi klaim README #3.

KENAPA ADA. README §"Key results" #3 menulis "the mechanism is geometric, and
it is visible" lalu menyandingkan dua tabel angka (cosinus embedding vs recall
kanal, cosinus embedding vs akurasi) tanpa satu koefisien korelasi pun — klaim
dibaca lewat mata, bukan diukur. Tiga sumber data yang dibutuhkan SEMUANYA
sudah lengkap tanpa GPU tambahan:
  - geometri per mode      : results/probe/b7_probe_Qwen_Qwen3-8B.json (5 mode + moi, 2026-08-27)
  - recall kanal per mode  : results/probe/channel_capacity_Qwen_Qwen3-8B_<mode>_m10.json
  - akurasi bench per mode : results/bench/analisis.json (groups[*].cells[*], medium kv)

Ini BUKAN pengganti sumbu interpolasi (`mix`) — mix menguji BENTUK hubungan di
sepanjang lintasan kontinu raw->soft; skrip ini menguji ARAH hubungan di lima
titik diskret yang sudah ada. Keduanya saling melengkapi, bukan salah satu
menggantikan yang lain. n=5 (lima mode) berarti p-value di sini deskriptif,
bukan inferensial — dilaporkan sebagai ukuran asosiasi, bukan uji hipotesis.

    python scripts/analisis_geometri_kinerja.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import RESULTS  # noqa: E402

MODES = ["raw", "soft", "gumbel", "sample", "moi"]
PROBE = RESULTS / "probe"
OUT = RESULTS / "pendukung"


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho tanpa scipy — cukup untuk n=5, tak menambah dependency."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan")


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    vx = sum((x[i] - mx) ** 2 for i in range(n))
    vy = sum((y[i] - my) ** 2 for i in range(n))
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan")


def geometry() -> dict[str, float]:
    d = json.loads((PROBE / "b7_probe_Qwen_Qwen3-8B.json").read_text())
    return {m: d["geometry"][m]["max_cos_embed_mean"] for m in MODES}


def channel_recall() -> dict[str, dict[str, float]]:
    """recall per mode per payload pada arm `kv_latent_only`, m=10 (satu-satunya
    lengan yang menguji ekspresivitas vektor laten murni — matriks.yaml `probe.arm`).
    `_summary` memuat SEMUA payload x arm dalam satu berkas per mode."""
    out: dict[str, dict[str, float]] = {m: {} for m in MODES}
    for m in MODES:
        # `gumbel` adalah mode default Tahap 0 asli — berkasnya TANPA sufiks
        # mode (`..._m10.json`), bukan `..._gumbel_m10.json` seperti mode lain
        # (lihat docs/HASIL_TAHAP0.md §1: "dibandingkan terhadap data gumbel
        # yang sudah ada dari sesi sebelumnya").
        name = "m10" if m == "gumbel" else f"{m}_m10"
        fp = PROBE / f"channel_capacity_Qwen_Qwen3-8B_{name}.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        for row in d.get("_summary", []):
            if row.get("arm") == "kv_latent_only" and row.get("payload"):
                out[m][row["payload"]] = row["recall"]
    return out


def bench_accuracy() -> dict[str, dict[str, float]]:
    """akurasi per (task, mode) pada medium kv, limit=100 (sel uji, bukan lampiran)."""
    d = json.loads((RESULTS / "bench" / "analisis.json").read_text())
    out: dict[str, dict[str, float]] = {}
    for g in d["groups"]:
        if g["limit"] != 100:
            continue
        task = g["task"]
        out.setdefault(task, {})
        for c in g["cells"]:
            cell = c["cell"]  # "<mode>/<comm|baseline>/m10"
            mode, comm, _ = cell.split("/")
            if comm not in ("kv", "baseline"):
                continue
            # raw/baseline dan raw/kv keduanya ada; kv adalah medium uji utama —
            # utamakan itu, baseline hanya dipakai kalau kv tak ada (tak terjadi di data ini).
            if mode not in out[task] or comm == "kv":
                out[task][mode] = c["accuracy"]
    return out


def main() -> None:
    geo = geometry()
    recall = channel_recall()
    acc = bench_accuracy()

    x = [geo[m] for m in MODES]
    result: dict = {"_meta": {
        "catatan": "n=5 (lima mode); korelasi DESKRIPTIF, bukan uji inferensial. "
                   "Melengkapi, bukan menggantikan, sumbu interpolasi (mix).",
        "geometry_source": "results/probe/b7_probe_Qwen_Qwen3-8B.json",
    }, "geometry": geo, "korelasi": {}}

    lines = ["# Geometri vs kinerja hilir (deskriptif, n=5 mode)",
             "", "Regenerasi: `python scripts/analisis_geometri_kinerja.py`", "",
             "| mode | cos ke embedding terdekat |", "|---|---:|"]
    for m in MODES:
        lines.append(f"| {m} | {geo[m]:.4f} |")

    lines += ["", "## Korelasi dengan recall kanal laten murni (m=10, k=5, trials=20)", ""]
    for payload in ("dsl", "token"):
        y = [recall[m].get(payload) for m in MODES]
        if any(v is None for v in y):
            continue
        y = [float(v) for v in y]
        rho, r = _spearman(x, y), _pearson(x, y)
        result["korelasi"][f"recall_{payload}"] = {
            "spearman": rho, "pearson": r,
            "per_mode": {m: {"geometry": geo[m], "recall": y[i]} for i, m in enumerate(MODES)},
        }
        lines.append(f"- **{payload}**: Spearman ρ={rho:.3f}, Pearson r={r:.3f} "
                     f"(n=5 titik: {', '.join(MODES)})")

    lines += ["", "## Korelasi dengan akurasi bench (medium kv, limit=100)", ""]
    for task, per_mode in sorted(acc.items()):
        y = [per_mode.get(m) for m in MODES]
        if any(v is None for v in y):
            lines.append(f"- **{task}**: data tak lengkap, dilewati "
                         f"(punya: {sorted(k for k,v in per_mode.items() if v is not None)})")
            continue
        y = [float(v) for v in y]
        rho, r = _spearman(x, y), _pearson(x, y)
        result["korelasi"][f"acc_{task}"] = {
            "spearman": rho, "pearson": r,
            "per_mode": {m: {"geometry": geo[m], "accuracy": y[i]} for i, m in enumerate(MODES)},
        }
        lines.append(f"- **{task}**: Spearman ρ={rho:.3f}, Pearson r={r:.3f}")

    lines += ["", "## Pembacaan", "",
              "Korelasi tinggi pada `token`/`humanevalplus` (payload/tugas yang paling "
              "menuntut presisi simbolik) dan lemah/tak bermakna pada tugas penalaran umum "
              "akan mengkonfirmasi disosiasi §1 README secara numerik, bukan hanya lewat "
              "rentang p-value McNemar per tugas. n=5 tetap kecil — baca sebagai ARAH "
              "hubungan, bukan bukti bentuknya (itu tugas sumbu `mix`)."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "geometri_vs_kinerja.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    (OUT / "geometri_vs_kinerja.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[tersimpan] {OUT / 'geometri_vs_kinerja.json'}")


if __name__ == "__main__":
    main()
