#!/usr/bin/env python3
"""Figur Bab IV — satu skrip, banyak figur, PNG ke results/visual/.

KENAPA SATU BERKAS. Tiap figur adalah satu fungsi `vNN_...()`, independen dan
aman dipanggil sendirian atau lewat `--all`. Yang belum punya data melapor
"[lewati]" dan keluar bersih — TIDAK membuat PNG kosong atau menebak angka.
Ini supaya skrip yang sama dipanggil ulang otomatis oleh watcher CPU begitu
sel faktor baru mendarat (lihat scripts/kekuatan_uji_faktor.py dan
scripts/agregasi_agent_trace.py yang jadi sumber sebagian figur di sini),
tanpa perlu tahu figur mana yang sudah siap.

Palet & rcParams DIWARISI dari `scripts/plot_readme_figures.py` — satu warna
per formulasi, TAK PERNAH dirotasi lintas figur (README dan Bab IV memakai
kanal warna yang sama, supaya pembaca yang melihat keduanya tak perlu belajar
ulang legenda).

    python scripts/visual_bab4.py --all
    python scripts/visual_bab4.py v03_heatmap_akurasi v06_geometri_vs_kinerja
    python scripts/visual_bab4.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "visual"
OUT.mkdir(parents=True, exist_ok=True)

# ── palet — identik dengan scripts/plot_readme_figures.py, jangan bercabang ──
COLOR = {
    "raw": "#e34948", "soft": "#2a78d6", "gumbel": "#1baf7a",
    "sample": "#eda100", "moi": "#4a3aa7",
}
GREY, DARK_GREY, INK = "#9a9a95", "#52514e", "#0b0b0b"
GOOD, WARN, BAD = "#0ca30c", "#eda100", "#e34948"   # status — jangan dipakai utk seri
ORDER = ["raw", "soft", "gumbel", "sample", "moi"]
R_FAMILY = ["soft", "gumbel", "sample", "moi"]
TASKS = ["gsm8k", "arc_challenge", "humanevalplus"]
TASK_LABEL = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}
AGENTS_BENCH = ["planner", "critic", "refiner", "judger"]
AGENTS_FACTOR = ["proposal", "innovate", "construct"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c9c4", "axes.labelcolor": INK,
    "xtick.color": DARK_GREY, "ytick.color": DARK_GREY,
    "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 170, "savefig.dpi": 170,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})


def _save(fig, name: str) -> None:
    fp = OUT / f"{name}.png"
    fig.savefig(fp)
    plt.close(fig)
    print(f"  [OK] {fp.relative_to(ROOT)}")


def _lewati(name: str, sebab: str) -> None:
    print(f"  [lewati] {name}: {sebab}")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ══════════════════════════════════════════════════════════════════════════
# LAPISAN 0 — peta & arsitektur (tanpa data, murni diagram)
# ══════════════════════════════════════════════════════════════════════════

def v01_peta_eksperimen() -> None:
    """Peta penelitian: 5 formulasi x 2 lengan x tugas — bukan hasil, konteks."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f2", ec="#c9c9c4", fs=8.2, weight="normal", tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
               fontsize=fs, color=tc, weight=weight, linespacing=1.35)

    def arrow(x0, y0, x1, y1, color=DARK_GREY):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=11, color=color, lw=1.2))

    # baris 1 — lima formulasi
    box(1.0, 8.7, 8.0, 1.0, "5 FORMULASI LANGKAH LATEN  ℳ = {raw, soft, sample, gumbel, moi}",
       fc="#efeeea", fs=9.2, weight="bold")
    xs = np.linspace(1.35, 8.15, 5)
    for x, m in zip(xs, ORDER):
        box(x - 0.55, 7.35, 1.1, 0.75, m, fc=COLOR[m], tc="white", fs=8.6, weight="bold")
        arrow(x, 8.7, x, 8.1)

    arrow(5.0, 7.35, 5.0, 6.85)
    box(1.0, 6.05, 8.0, 0.8, "ℛ = ℳ ∖ {raw}  —  keluarga relaksasi diskret (convex hull embedding)",
       fc="#eef4fc", fs=8.4)

    # baris 2 — dua lengan
    arrow(2.6, 6.05, 2.6, 5.55); arrow(7.4, 6.05, 7.4, 5.55)
    box(0.7, 4.55, 3.8, 1.0, "LENGAN BENCH\n\"apakah agen masih bisa BERNALAR?\"\nplanner→critic→refiner→judger",
       fc="#fdf3ea", fs=8.0)
    box(5.5, 4.55, 3.8, 1.0, "LENGAN FAKTOR\n\"apakah agen masih bisa membawa STRUKTUR?\"\nproposal→innovate→construct",
       fc="#fdf3ea", fs=8.0)

    arrow(2.6, 4.55, 1.4, 3.7); arrow(2.6, 4.55, 2.6, 3.7); arrow(2.6, 4.55, 3.8, 3.7)
    for x, t in zip([1.4, 2.6, 3.8], ["GSM8K\n(math)", "ARC-C\n(commonsense)", "HumanEval+\n(code)"]):
        box(x - 0.68, 3.0, 1.36, 0.7, t, fc="#fbeee8", fs=7.3)

    arrow(7.4, 4.55, 7.4, 3.7)
    box(5.6, 3.0, 3.6, 0.7,
       "parse → evaluable → fidelitas →\nkeberagaman → RankIC → holdout  (6 level)",
       fc="#fbeee8", fs=7.3)

    arrow(2.6, 3.0, 2.6, 2.5); arrow(7.4, 3.0, 7.4, 2.5)
    box(0.7, 1.5, 3.8, 1.0,
       "medium: text · kv · kv_and_text\nn=100 soal/sel, sample-seed sama",
       fc="#eef4fc", fs=7.6)
    box(5.5, 1.5, 3.8, 1.0,
       "4 arah × 5 seed = 20 jalan/sel\n(d0, d1, opp_mom, opp_rev)",
       fc="#eef4fc", fs=7.6)

    ax.text(5.0, 0.35, "Sumbu C (mix, α∈[0,1]) menginterpolasi raw↔soft secara kontinu — mengisi jarak antar 5 titik di atas",
           ha="center", fontsize=7.4, color=DARK_GREY, style="italic")
    ax.set_title("Peta eksperimen — bukan hasil, kerangka yang menghasilkannya",
                fontsize=10.5, weight="bold", pad=10)
    _save(fig, "v01_peta_eksperimen")


def v02_arsitektur_sistem() -> None:
    """Diagram alir: rantai agen sekuensial + titik cabang 5 formulasi."""
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f2", fs=8.4, weight="normal", tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                                    fc=fc, ec="#c9c9c4", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
               fontsize=fs, color=tc, weight=weight, linespacing=1.3)

    def arrow(x0, y0, x1, y1, color=DARK_GREY, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=12, color=color, lw=1.3))

    box(0.1, 4.6, 1.7, 0.8, "PERTANYAAN\n(prompt awal)", fc="#eef4fc", fs=7.6)
    agents_x = [2.6, 5.0, 7.4]
    labels = ["Agen A₁\n(mis. proposal)", "Agen A₂\n(mis. innovate)", "Agen A₃\n(mis. construct)"]
    arrow(1.8, 5.0, 2.5, 5.0)
    for i, (x, lab) in enumerate(zip(agents_x, labels)):
        box(x - 0.75, 4.6, 1.5, 0.8, lab, fc="#fdf3ea", fs=7.4)
        if i < 2:
            nx = agents_x[i + 1]
            arrow(x + 0.75, 5.0, nx - 0.75, 5.0, color="#4a3aa7")
            ax.text((x + 0.75 + nx - 0.75) / 2, 5.28, "KV-cache\n(medium kv)",
                   ha="center", fontsize=6.4, color="#4a3aa7", style="italic")
    arrow(8.15, 5.0, 9.3, 5.0)
    box(9.0, 4.6, 0.9, 0.8, "OUTPUT", fc="#eef4fc", fs=7.2)

    ax.text(5.0, 3.85, "di setiap hop laten, hidden state h harus dipetakan balik ke ruang embedding masukan:",
           ha="center", fontsize=7.6, color=DARK_GREY)
    box(3.9, 3.05, 2.2, 0.6, "z = Φ(h)  — SUMBU A", fc="#efeeea", fs=8.6, weight="bold")
    arrow(5.0, 3.05, 5.0, 2.55)

    xs = np.linspace(1.2, 8.8, 5)
    for x, m in zip(xs, ORDER):
        box(x - 0.68, 1.75, 1.36, 0.7, m, fc=COLOR[m], tc="white", fs=8.2, weight="bold")
        arrow(5.0, 2.55, x, 2.45)
    ax.text(5.0, 1.35,
           "raw = ridge $W_a$ (LatentMAS, Teorema A.1)  |  soft/gumbel/sample/moi = proyeksi convex-hull $W_\\mathrm{in}$",
           ha="center", fontsize=7.2, color=DARK_GREY, style="italic")
    ax.set_title("Arsitektur: rantai agen sekuensial dengan satu titik cabang formulasi",
                fontsize=10.5, weight="bold", pad=8)
    _save(fig, "v02_arsitektur_sistem")


# ══════════════════════════════════════════════════════════════════════════
# LAPISAN 1 — apakah formulasi berbeda? (lengan bench, data lengkap)
# ══════════════════════════════════════════════════════════════════════════

def _bench_cells() -> dict[tuple[str, str], dict]:
    d = _load(RESULTS / "bench" / "analisis.json")
    if not d:
        return {}
    out = {}
    for g in d["groups"]:
        if g["limit"] != 100:
            continue
        for c in g["cells"]:
            mode, comm, _ = c["cell"].split("/")
            out[(g["task"], mode, comm)] = c
    return out


def v03_heatmap_akurasi() -> None:
    cells = _bench_cells()
    if not cells:
        return _lewati("v03", "results/bench/analisis.json tak ada")
    cols = ORDER + ["raw(text)", "raw(single)"]
    data = np.full((len(TASKS), len(cols)), np.nan)
    for i, t in enumerate(TASKS):
        for j, m in enumerate(ORDER):
            c = cells.get((t, m, "kv"))
            if c:
                data[i, j] = c["accuracy"]
        if cells.get((t, "raw", "text")):
            data[i, 5] = cells[(t, "raw", "text")]["accuracy"]
        if cells.get((t, "raw", "baseline")):
            data[i, 6] = cells[(t, "raw", "baseline")]["accuracy"]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(TASKS))); ax.set_yticklabels([TASK_LABEL[t] for t in TASKS], fontsize=8.5)
    ax.grid(False)
    for i in range(len(TASKS)):
        for j in range(len(cols)):
            if not np.isnan(data[i, j]):
                tc = "white" if data[i, j] < 0.6 or data[i, j] > 0.93 else INK
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8, color=tc)
    ax.axvline(4.5, color="white", lw=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("akurasi", fontsize=7.5)
    ax.set_title("Akurasi per formulasi × tugas (medium kv, n=100/sel; dua kolom kanan = referensi text/single)",
                fontsize=9.3, weight="bold")
    _save(fig, "v03_heatmap_akurasi")


def v04_pareto_token_waktu() -> None:
    cells = _bench_cells()
    tok = _load(RESULTS / "pendukung" / "token_bench.json")
    if not cells or not tok:
        return _lewati("v04", "analisis.json atau token_bench.json tak ada")
    tok_by_sel = {r["sel"]: r for r in tok}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        for m in ORDER:
            c = cells.get((task, m, "kv"))
            if not c:
                continue
            name = f"{task}_{m}_kv_s0"
            tinfo = tok_by_sel.get(name)
            if not tinfo:
                continue
            x = tinfo.get("token_per_soal")
            y = c["accuracy"] * 100
            size = max(30, c["time_s"] / c["n"] * 6)
            ax.scatter(x, y, s=size, color=COLOR[m], alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(m, (x, y), fontsize=7, color=COLOR[m], weight="bold",
                       xytext=(4, 3), textcoords="offset points")
        ax.set_title(TASK_LABEL[task], fontsize=9)
        ax.set_xlabel("token keluaran / soal", fontsize=8)
    axes[0].set_ylabel("akurasi (%)", fontsize=8.5)
    fig.suptitle("Akurasi vs biaya token (ukuran bubble ∝ detik/soal) — trade-off per formulasi",
               fontsize=10, weight="bold", y=1.03)
    fig.text(0.5, -0.02,
            "Sudut kiri-atas = murah & akurat. `raw` konsisten paling kanan-bawah pada HumanEval+ — mahal DAN kurang akurat sekaligus.",
            ha="center", fontsize=7.6, color=DARK_GREY, style="italic")
    _save(fig, "v04_pareto_token_waktu")


# ══════════════════════════════════════════════════════════════════════════
# LAPISAN 2 — mengapa berbeda? (geometri)
# ══════════════════════════════════════════════════════════════════════════

def v05_geometri_rentang() -> None:
    d = _load(RESULTS / "probe" / "b7_probe_Qwen_Qwen3-8B.json")
    if not d:
        return _lewati("v05", "b7_probe belum ada")
    geo = d["geometry"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, m in enumerate(ORDER):
        g = geo[m]
        lo, mean, hi = g["max_cos_embed_min"], g["max_cos_embed_mean"], g["max_cos_embed_max"]
        ax.plot([i, i], [lo, hi], color=COLOR[m], lw=3, solid_capstyle="round", alpha=0.55)
        ax.scatter([i], [mean], color=COLOR[m], s=90, zorder=3, edgecolor="white", linewidth=1)
        ax.text(i, hi + 0.03, f"{mean:.3f}", ha="center", fontsize=7.6, color=COLOR[m], weight="bold")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"$\max_i \cos(z, W_{in}[i])$", fontsize=9)
    ax.set_title("Kedekatan vektor laten ke embedding token terdekat\n"
               "(titik = rerata 10 langkah, 1 prompt; batang = rentang min–maks langkah)",
               fontsize=9, weight="bold")
    _save(fig, "v05_geometri_rentang")


def v06_geometri_vs_kinerja() -> None:
    d = _load(RESULTS / "pendukung" / "geometri_vs_kinerja.json")
    if not d:
        return _lewati("v06", "results/pendukung/geometri_vs_kinerja.json tak ada")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    for key, marker in [("acc_gsm8k", "o"), ("acc_arc_challenge", "s"), ("acc_humanevalplus", "^")]:
        entry = d["korelasi"].get(key)
        if not entry:
            continue
        label = key.replace("acc_", "")
        for m, pt in entry["per_mode"].items():
            ax.scatter(pt["geometry"], pt["accuracy"] * 100, color=COLOR[m], marker=marker, s=55, zorder=3)
        rho = entry["spearman"]
        ax.scatter([], [], color=DARK_GREY, marker=marker, label=f"{TASK_LABEL.get(label,label)} (ρ={rho:.2f})")
    ax.set_xlabel(r"geometri: $\max\cos(z,e)$", fontsize=8.5)
    ax.set_ylabel("akurasi bench (%)", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="lower right", frameon=False)
    ax.set_title("Geometri vs akurasi", fontsize=9.3, weight="bold")

    ax = axes[1]
    for key, marker in [("recall_dsl", "o"), ("recall_token", "D")]:
        entry = d["korelasi"].get(key)
        if not entry:
            continue
        for m, pt in entry["per_mode"].items():
            ax.scatter(pt["geometry"], pt["recall"] * 100, color=COLOR[m], marker=marker, s=55, zorder=3)
        rho = entry["spearman"]
        ax.scatter([], [], color=DARK_GREY, marker=marker,
                 label=f"payload {key.replace('recall_','')} (ρ={rho:.2f})")
    ax.set_xlabel(r"geometri: $\max\cos(z,e)$", fontsize=8.5)
    ax.set_ylabel("recall kanal laten murni (%)", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="lower right", frameon=False)
    ax.set_title("Geometri vs recall simbolik (Tahap 0)", fontsize=9.3, weight="bold")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[m], markersize=7, label=m)
              for m in ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Geometri → kinerja hilir (deskriptif, n=5 titik/panel; ASOSIASI, bukan kausalitas)",
               fontsize=9.5, weight="bold", y=1.15)
    _save(fig, "v06_geometri_vs_kinerja")


def v07_pencarian_beta_moi() -> None:
    """Sweep beta MoI — TAMBAHAN, bukan dari saran awal: datanya sudah ada
    (results/probe/channel_capacity_..._moi_b*_m10.json, Tahap 0B) tapi tak
    pernah divisualkan. beta mengontrol pseudo-count observasi one-hot vs
    prior entropi (§Rumus Sumbu A) — sweep ini menjawab APAKAH default β=1
    (setelan produksi) memang di titik yang masuk akal, atau kebetulan."""
    betas_files = {
        "0.25": "channel_capacity_Qwen_Qwen3-8B_moi_b0.25_m10.json",
        "0.5": "channel_capacity_Qwen_Qwen3-8B_moi_b0.5_m10.json",
        "1 (default)": "channel_capacity_Qwen_Qwen3-8B_moi_m10.json",
        "2": "channel_capacity_Qwen_Qwen3-8B_moi_b2_m10.json",
        "4": "channel_capacity_Qwen_Qwen3-8B_moi_b4_m10.json",
        "8": "channel_capacity_Qwen_Qwen3-8B_moi_b8_m10.json",
    }
    rows = []
    for label, fname in betas_files.items():
        d = _load(RESULTS / "probe" / fname)
        if not d:
            continue
        beta_num = float(label.split()[0])
        rec = {row["payload"]: row["recall"] for row in d.get("_summary", [])
              if row.get("arm") == "kv_latent_only"}
        rows.append((beta_num, label, rec.get("dsl"), rec.get("token")))
    if len(rows) < 3:
        return _lewati("v07", f"cuma {len(rows)} titik beta ditemukan (butuh >=3)")
    rows.sort()

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = [r[0] for r in rows]
    ax.plot(xs, [r[2] for r in rows], "o-", color=COLOR["moi"], label="payload dsl", lw=1.8)
    ax.plot(xs, [r[3] for r in rows], "s--", color=COLOR["moi"], alpha=0.55, label="payload token", lw=1.8)
    ax.axvline(1.0, color=GREY, lw=1, ls=":")
    ax.text(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] else 0.4, " β=1 (produksi)",
           fontsize=7, color=DARK_GREY, va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ (pseudo-count MoI)", fontsize=8.8)
    ax.set_ylabel("recall kanal laten murni", fontsize=8.8)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Sweep β pada MoI: apakah default produksi masuk akal?\n"
               "(Tahap 0B, m=10, k=5, trials=20 per titik)", fontsize=9, weight="bold")
    _save(fig, "v07_pencarian_beta_moi")


def v08_interpolasi_geometri() -> None:
    d = _load(RESULTS / "probe" / "b7_probe_Qwen_Qwen3-8B.json")
    if not d or "geometry_mix" not in d:
        return _lewati("v08", "geometry_mix tak ada di b7_probe")
    kurva = d["geometry_mix"]
    xs = sorted(float(k) for k in kurva)
    ys = [kurva[f"{x}" if x not in (0.0, 1.0) else ("0.0" if x == 0.0 else "1.0")]["max_cos_embed_mean"]
         for x in xs]
    # kunci float bisa "0.0"/"0"/"1.0" tergantung sumber; cocokkan lebih toleran
    ys = []
    for x in xs:
        key = next((k for k in kurva if abs(float(k) - x) < 1e-9), None)
        ys.append(kurva[key]["max_cos_embed_mean"] if key else np.nan)

    mix = _load(RESULTS / "pendukung" / "sumbu_mix.json")
    if not mix:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(xs, ys, "o-", color="#4a3aa7", lw=2, ms=8, zorder=3)
        ax.set_xlabel(r"$\alpha$", fontsize=9)
        ax.set_ylabel(r"$\max\cos(z_\alpha, e)$", fontsize=9)
        ax.set_title("Sumbu C — separuh GEOMETRI\n(kinerja belum dianalisis: "
                     "jalankan scripts/analisis_mix.py)", fontsize=8.8, weight="bold")
        return _save(fig, "v08_interpolasi_geometri")

    fig, (axg, axp) = plt.subplots(1, 2, figsize=(10.4, 3.9))

    # ── kiri: GEOMETRI. Poinnya sumbu-x TIDAK linier — ia jenuh. ───────────
    axg.plot(xs, ys, "o-", color="#4a3aa7", lw=2, ms=8, zorder=3)
    axg.scatter([0.0], [ys[0]], color=COLOR["raw"], s=110, zorder=4, edgecolor="white", linewidth=1.2)
    axg.scatter([1.0], [ys[-1]], color=COLOR["soft"], s=110, zorder=4, edgecolor="white", linewidth=1.2)
    axg.annotate("= raw", (0.0, ys[0]), xytext=(8, -12), textcoords="offset points",
                 fontsize=7.5, color=COLOR["raw"])
    axg.annotate("= soft", (1.0, ys[-1]), xytext=(-40, -12), textcoords="offset points",
                 fontsize=7.5, color=COLOR["soft"])
    axg.annotate("jenuh:\n0,75→1 hanya +0,0025", (0.86, 0.93), xytext=(-6, -46),
                 textcoords="offset points", fontsize=7, color=DARK_GREY,
                 ha="center", arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    axg.set_xlabel(r"$\alpha$  (interpolasi $z_{raw}\to z_{soft}$)", fontsize=9)
    axg.set_ylabel(r"$\max\cos(z_\alpha, e)$  — TERUKUR", fontsize=9)
    axg.set_title("(a) GEOMETRI — sumbu dosis tidak linier", fontsize=9, weight="bold")

    # ── kanan: KINERJA terhadap cos TERUKUR, bukan terhadap alpha ─────────
    titik = mix["titik"]
    cx = [t["cos_embed_mean"] for t in titik]
    gate = [t["faktor"]["laju_lolos_gate"] for t in titik]
    he = [t["bench_humanevalplus"].get("akurasi") for t in titik]
    cjk = [t["bench_humanevalplus"].get("n_jawaban_ber_cjk") or 0 for t in titik]

    axp.axvspan(cx[1], cx[2], color="#f2c9c9", alpha=0.45, zorder=0)
    axp.text((cx[1] + cx[2]) / 2, 1.15, "zona ambang", ha="center", fontsize=7.8,
             color="#8f2f2f", weight="bold", zorder=5)
    axp.plot(cx, gate, "o-", color=COLOR["gumbel"], lw=2, ms=7.5, zorder=3,
             label="lolos gate, lengan faktor (n=20 jalan)")
    axp.plot(cx, he, "s-", color=COLOR["soft"], lw=2, ms=7, zorder=3,
             label="akurasi HumanEval+ (n=100 soal)")
    # dua titik kanan berimpit di sumbu cos (0,9244 vs 0,9269) — labelnya
    # digeser vertikal supaya tak saling menimpa
    geser = {0.0: (0, -15), 0.25: (0, -15), 0.5: (0, -15), 0.75: (-16, 10), 1.0: (16, -16)}
    for x, y, a in zip(cx, he, [t["alpha"] for t in titik]):
        dx, dy = geser.get(a, (0, -15))
        axp.annotate(f"α={a:g}", (x, y), xytext=(dx, dy), textcoords="offset points",
                     fontsize=6.8, color=DARK_GREY, ha="center")
    axp.set_ylim(0, 1.22)
    axp.set_xlabel(r"$\max\cos(z_\alpha, e)$ terukur  (BUKAN $\alpha$)", fontsize=9)
    axp.set_ylabel("proporsi", fontsize=9)
    axp.legend(fontsize=7.2, loc="lower right", framealpha=0.92)

    axc = axp.twinx()
    axc.bar(cx, cjk, width=0.030, color=BAD, alpha=0.42, zorder=2)
    axc.set_ylim(0, 11)
    axc.set_ylabel("jawaban ber-aksara CJK", fontsize=8, color="#8f2f2f")
    axc.tick_params(axis="y", labelsize=7.5, colors="#8f2f2f")
    axc.annotate("korupsi simbolik MEMUNCAK di sini\n(4 jawaban ber-CJK; raw 1; nol setelah ambang)",
                 (cx[1], cjk[1]), xytext=(-58, 88), textcoords="offset points", fontsize=6.9,
                 color="#8f2f2f", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#8f2f2f", lw=0.9,
                                 connectionstyle="arc3,rad=-0.25"))

    axp.set_title("(b) KINERJA — fungsi TANGGA, bukan tanjakan", fontsize=9, weight="bold")
    fig.suptitle("Sumbu C (`mix`) — formulasi dipegang tetap, hanya posisi di ruang laten yang digeser",
                 fontsize=9.6, weight="bold", y=1.005)
    _save(fig, "v08_interpolasi_geometri")


# ══════════════════════════════════════════════════════════════════════════
# LAPISAN 3 — apakah laten mempertahankan simbol? (lengan faktor)
# ══════════════════════════════════════════════════════════════════════════

def _muat_faktor(comm_mode: str, mode: str, allow_archive: bool = True) -> dict | None:
    tag = "text" if (mode == "raw" and comm_mode == "text") else f"{comm_mode}_{mode}"
    fp = RESULTS / "factor" / f"frontend_{tag}.json"
    if fp.exists():
        return json.loads(fp.read_text()), "matriks(20 jalan)"
    if allow_archive:
        fp2 = RESULTS / "arsip_faktor_6jalan_2026-08-10" / f"frontend_{tag}.json"
        if fp2.exists():
            return json.loads(fp2.read_text()), "arsip(6 jalan)"
    return None, None


def v09_funnel_fidelitas(comm_mode: str = "kv") -> None:
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.4), sharey=True)
    any_data = False
    for ax, m in zip(axes, ORDER):
        d, sumber = _muat_faktor(comm_mode, m)
        if not d:
            ax.axis("off")
            ax.set_title(f"{m}\n(belum ada)", fontsize=8, color=GREY)
            continue
        any_data = True
        runs = d["runs"]
        n = len(runs)
        n_expr = sum(1 for r in runs for f in (r.get("factors") or []))
        n_pass = sum(1 for r in runs for f in (r.get("factors") or []) if f.get("passed_gate") or
                    (r.get("passing") and f.get("expression") in r["passing"]))
        n_ic = sum(1 for r in runs for f in (r.get("factors") or []) if f.get("ic") is not None)
        n_pos = sum(1 for r in runs for f in (r.get("factors") or [])
                   if (f.get("ic") or 0) != 0 and abs(f.get("ic") or 0) > 0.02)
        stages = [("ekspresi\ndihasilkan", n_expr), ("lolos\ngate", n_pass),
                 ("evaluable\n(ic≠None)", n_ic), ("|IC|>0.02", n_pos)]
        base = max(1, stages[0][1])
        widths = [s[1] / base for s in stages]
        ys = range(len(stages))
        ax.barh(list(ys), widths, color=COLOR[m], height=0.6, alpha=0.9)
        for y, (label, val) in zip(ys, stages):
            ax.text(-0.05, y, label, ha="right", va="center", fontsize=6.6, color=INK)
            ax.text(min(widths[y] + 0.03, 1.02), y, f"{val}", ha="left", va="center", fontsize=6.8, color=DARK_GREY)
        ax.set_xlim(0, 1.35)
        ax.invert_yaxis()
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{m}\n({sumber}, n={n})", fontsize=8.4, color=COLOR[m], weight="bold")
    if not any_data:
        plt.close(fig)
        return _lewati("v09", f"tak ada frontend_{comm_mode}_*.json sama sekali")
    fig.suptitle(f"Corong fidelitas simbolik per formulasi — comm_mode={comm_mode}\n"
               "(lebar batang relatif thd jumlah ekspresi dihasilkan mode itu sendiri)",
               fontsize=9.6, weight="bold", y=1.05)
    _save(fig, f"v09_funnel_fidelitas_{comm_mode}")


def v10_akumulasi_kv_hop(comm_mode: str = "kv") -> None:
    """Pertumbuhan KV-cache sepanjang rantai — BUKAN 'parse-rate per-hop'.

    Kenapa diganti dari rencana awal (parse rate menurun A1->A2->A3). Dicoba,
    dan hasilnya menyesatkan: pada arsitektur ini HANYA `construct` menulis
    JSON — `proposal`/`innovate` berkomunikasi murni lewat KV-cache dan
    `parsed_ok` mereka SELALU False by design (bukan kegagalan bertahap, lihat
    `agent_trace` mentah). Memplotnya sebagai 'degradasi' akan menyiratkan pola
    yang tak pernah ada di data. `kv_len` adalah metrik yang BENAR-BENAR terukur
    di ketiga hop dan menjawab pertanyaan yang sejenis: berapa banyak konteks
    yang diwariskan tiap agen berikutnya.
    """
    d = _load(RESULTS / "pendukung" / "agent_trace_perhop.json")
    if not d:
        return _lewati("v10", "agent_trace_perhop.json tak ada — jalankan scripts/agregasi_agent_trace.py")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    any_line = False
    for m in ORDER:
        tag = f"{comm_mode}_{m}" if not (m == "raw" and comm_mode == "text") else "text"
        rows = None
        for sumber in ("matriks", "arsip_faktor_6jalan_2026-08-10"):
            cand = d.get(sumber, {}).get(tag)
            if cand:
                rows = cand
                break
        if not rows:
            continue
        by_agent = {r["agent"]: r for r in rows}
        xs, ys = [], []
        for i, ag in enumerate(AGENTS_FACTOR):
            r = by_agent.get(ag)
            kv = r["metrik"].get("kv_len", {}).get("median") if r else None
            if kv is not None:
                xs.append(i); ys.append(kv)
        if len(xs) >= 2:
            ax.plot(xs, ys, "o-", color=COLOR[m], label=m, lw=2, ms=7)
            any_line = True
    if not any_line:
        plt.close(fig)
        return _lewati("v10", "tak ada agen dgn kv_len lengkap")
    ax.set_xticks(range(len(AGENTS_FACTOR))); ax.set_xticklabels(AGENTS_FACTOR, fontsize=9)
    ax.set_ylabel("kv_len median (token)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Akumulasi KV-cache sepanjang rantai — comm_mode={comm_mode}\n"
               "(bukan parse-rate: hanya `construct` menulis JSON, lihat docstring)",
               fontsize=9, weight="bold")
    _save(fig, f"v10_akumulasi_kv_hop_{comm_mode}")


def v11_upaya_perbaikan() -> None:
    """PENGGANTI 'lineage/evolution tree'. Desain ini SINGLE-PASS, tanpa loop
    evolusi (DESAIN_EKSPERIMEN §6) — pohon silsilah hipotesis->faktor->umpan
    balik ala QuantaAlpha TIDAK ADA padanannya di sini, dan memaksakannya akan
    menggambarkan struktur yang tak pernah dijalankan. Yang benar-benar ada:
    `construct` boleh mencoba ulang hingga `max_repair` kali dalam SATU
    trajectory. Ini figur yang jujur terhadap desainnya."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    any_data = False
    width = 0.15
    xs = np.arange(4)  # 0,1,2,3 percobaan repair
    for i, m in enumerate(ORDER):
        d, _ = _muat_faktor("kv", m)
        if not d:
            continue
        any_data = True
        counts = [0, 0, 0, 0]
        for r in d["runs"]:
            n = min(int(r.get("repair_attempts") or 0), 3)
            counts[n] += 1
        total = max(1, sum(counts))
        ax.bar(xs + (i - 2) * width, [c / total * 100 for c in counts], width,
              color=COLOR[m], label=m)
    if not any_data:
        plt.close(fig)
        return _lewati("v11", "tak ada data repair_attempts")
    ax.set_xticks(xs); ax.set_xticklabels(["0\n(langsung jadi)", "1", "2", "3\n(maks)"], fontsize=8)
    ax.set_xlabel("jumlah percobaan `repair` per trajectory", fontsize=8.5)
    ax.set_ylabel("% trajectory", fontsize=8.5)
    # x=1,2 kosong pada data ini (repair mayoritas 0 atau maks) — tempat aman
    # utk legenda TANPA menimpa judul atau batang.
    ax.legend(fontsize=7.5, frameon=False, ncol=1, loc="upper center")
    ax.set_title("Upaya perbaikan per trajectory (comm_mode=kv)\n"
               "pengganti jujur 'lineage': desain ini single-pass, tak ada loop evolusi",
               fontsize=8.8, weight="bold", pad=10)
    _save(fig, "v11_upaya_perbaikan")


def v12_matriks_fidelitas() -> None:
    cols = ["parse_rate", "lolos_gate", "evaluable_rate", "positive_ic_rate"]
    data = np.full((len(ORDER), len(cols)), np.nan)
    sumber_label = {}
    for i, m in enumerate(ORDER):
        d, sumber = _muat_faktor("kv", m)
        if not d:
            continue
        sumber_label[m] = sumber
        runs = d["runs"]
        n = len(runs)
        n_head_ok = sum(1 for r in runs if (r.get("construct_text_head") or "").lstrip().startswith("{"))
        n_gate = sum(1 for r in runs if r.get("passing"))
        n_expr = sum(1 for r in runs for f in (r.get("factors") or []))
        n_ic = sum(1 for r in runs for f in (r.get("factors") or []) if f.get("ic") is not None)
        n_pos = sum(1 for r in runs for f in (r.get("factors") or [])
                   if (f.get("ic") or 0) != 0 and abs(f.get("ic") or 0) > 0.02)
        data[i, 0] = n_head_ok / n if n else np.nan
        data[i, 1] = n_gate / n if n else np.nan
        data[i, 2] = (n_ic / n_expr) if n_expr else np.nan
        data[i, 3] = (n_pos / n_expr) if n_expr else np.nan
    if np.all(np.isnan(data)):
        return _lewati("v12", "tak ada data faktor sama sekali")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["parse\n(construct)", "≥1 lolos\ngate/traj", "evaluable\n/ekspresi", "|IC|>0.02\n/ekspresi"], fontsize=7.6)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([f"{m}  ({sumber_label.get(m,'-')})" for m in ORDER], fontsize=7.8)
    ax.grid(False)
    for i in range(len(ORDER)):
        for j in range(len(cols)):
            if not np.isnan(data[i, j]):
                tc = "white" if data[i, j] < 0.35 or data[i, j] > 0.8 else INK
                ax.text(j, i, f"{data[i,j]*100:.0f}%", ha="center", va="center", fontsize=8, color=tc)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("proporsi", fontsize=7.5)
    ax.set_title("Matriks fidelitas simbolik — comm_mode=kv\n(kolom kanan butuh skoring CPU; kosong = belum diskor)",
               fontsize=9, weight="bold")
    _save(fig, "v12_matriks_fidelitas")


# ══════════════════════════════════════════════════════════════════════════
# LAPISAN 4 — apakah faktor yang dihasilkan berkualitas? (butuh skoring CPU)
# ══════════════════════════════════════════════════════════════════════════

def v13_distribusi_ic() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    series, labels, colors = [], [], []
    for m in ORDER:
        d, _ = _muat_faktor("kv", m)
        if not d:
            continue
        vals = [f["ic"] for r in d["runs"] for f in (r.get("factors") or [])
               if f.get("ic") is not None]
        if len(vals) >= 3:
            series.append(vals); labels.append(m); colors.append(COLOR[m])
    if not series:
        return _lewati("v13", "belum ada expr ber-IC (menunggu rescore_all/skoring)")
    parts = ax.violinplot(series, showmeans=True, showextrema=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.65); pc.set_edgecolor(c)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(DARK_GREY)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.set_ylabel("IC (RankIC harian, jendela seleksi 2021)", fontsize=8.8)
    ax.set_title("Distribusi IC per formulasi — comm_mode=kv (ekspresi ber-IC saja)",
               fontsize=9.3, weight="bold")
    _save(fig, "v13_distribusi_ic")


def v14_holdout_vs_seleksi() -> None:
    # berkas holdout bisa berada di results/factor/ (run aktif) atau sudah
    # dipindah ke results/arsip_*/ (mis. arsip 6-jalan 2026-08-10) — cari
    # keduanya, pakai yang termuda.
    kandidat = sorted(RESULTS.glob("factor/holdout_*.json")) \
        + sorted(RESULTS.glob("arsip_*/holdout_*.json"))
    kandidat = sorted(kandidat, key=lambda p: p.stat().st_mtime, reverse=True)
    hold = _load(kandidat[0]) if kandidat else None
    if not hold:
        return _lewati("v14", "results/{factor,arsip_*}/holdout_*.json belum ada — jalankan skor_holdout.py")
    per_ex = hold.get("per_ekspresi", [])
    if not per_ex:
        return _lewati("v14", "'per_ekspresi' kosong di berkas holdout")
    # skor_holdout.py menulis IC jendela holdout sebagai "ic" (bukan
    # "ic_holdout"); "ic_seleksi" barulah IC jendela seleksi.
    pasangan = [(e["ic_seleksi"], e["ic"]) for e in per_ex
                if e.get("ic_seleksi") is not None and e.get("ic") is not None]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    xs = [p[0] for p in pasangan]
    ys = [p[1] for p in pasangan]
    if not xs:
        plt.close(fig)
        return _lewati("v14", "tak ada pasangan ic_seleksi/ic")
    ax.scatter(xs, ys, s=22, color="#4a3aa7", alpha=0.6, edgecolor="white", linewidth=0.4)
    lim = max(abs(min(xs + ys, default=0)), abs(max(xs + ys, default=0.1))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1, ls="--", label="y = x (stabil sempurna)")
    ax.axhline(0, color="#e8e8e4", lw=0.8); ax.axvline(0, color="#e8e8e4", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("IC — jendela seleksi (2021)", fontsize=9)
    win = hold.get("window") or []
    lab_win = f"{win[0][:4]}–{win[1][:4]}" if len(win) == 2 else "holdout"
    ax.set_ylabel(f"IC — jendela holdout ({lab_win})", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Stabilitas IC: seleksi vs holdout (n={len(xs)} ekspresi)\n"
               "titik di kuadran II/IV = berbalik tanda", fontsize=9.2, weight="bold")
    _save(fig, "v14_holdout_vs_seleksi")


# ══════════════════════════════════════════════════════════════════════════
# LAPISAN 5 — putaran kedua (saran lanjutan 2026-08-27): cakupan DSL,
# Pareto lengan faktor, korupsi CJK, panjang/repetisi bench, kanal per-arm
# (A9/HASIL_TAHAP4 §3), kedalaman ekspresi, statistik early-stop, keragaman
# SEMANTIK (embedding Qwen, bukan sintaksis) — semua CPU-only.
# ══════════════════════════════════════════════════════════════════════════
import re  # noqa: E402

_FUNC_RE = re.compile(r"\b([A-Z_][A-Z0-9_]*)\s*\(")
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-퟿]")


def _semua_ekspresi_per_mode(comm_mode: str = "kv") -> dict[str, list[str]]:
    out = {}
    for m in ORDER:
        d, _ = _muat_faktor(comm_mode, m)
        if not d:
            continue
        out[m] = [f["expression"] for r in d["runs"] for f in (r.get("factors") or [])
                 if f.get("expression")]
    return out


def v15_cakupan_fungsi_dsl(comm_mode: str = "kv") -> None:
    """Heatmap mode x fungsi DSL — apakah `raw` kolaps ke kosakata sempit?
    (saran #3 — sintaksis: fungsi APA yang dipakai, pelengkap v22 yang semantik)"""
    per_mode = _semua_ekspresi_per_mode(comm_mode)
    if not per_mode:
        return _lewati("v15", "tak ada ekspresi faktor sama sekali")
    counts = {m: {} for m in per_mode}
    all_funcs: set[str] = set()
    for m, exprs in per_mode.items():
        for e in exprs:
            for fn in set(_FUNC_RE.findall(e)):
                counts[m][fn] = counts[m].get(fn, 0) + 1
                all_funcs.add(fn)
    if not all_funcs:
        return _lewati("v15", "regex fungsi tak menangkap apa pun")
    # urutkan fungsi menurun berdasar total pemakaian lintas mode
    funcs = sorted(all_funcs, key=lambda fn: -sum(counts[m].get(fn, 0) for m in per_mode))[:20]
    modes_ada = [m for m in ORDER if m in per_mode]
    data = np.array([[counts[m].get(fn, 0) / max(1, len(per_mode[m])) for fn in funcs]
                     for m in modes_ada])

    fig, ax = plt.subplots(figsize=(max(6.5, 0.42 * len(funcs)), 0.55 * len(modes_ada) + 1.6))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_xticks(range(len(funcs))); ax.set_xticklabels(funcs, rotation=60, ha="right", fontsize=6.8)
    ax.set_yticks(range(len(modes_ada)))
    ax.set_yticklabels([f"{m} (n={len(per_mode[m])})" for m in modes_ada], fontsize=8)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("pemakaian / ekspresi", fontsize=7)
    ax.set_title(f"Cakupan fungsi DSL per formulasi — comm_mode={comm_mode} (top {len(funcs)} fungsi)",
               fontsize=9.3, weight="bold")
    _save(fig, f"v15_cakupan_fungsi_dsl_{comm_mode}")


def v16_pareto_faktor(comm_mode: str = "kv") -> None:
    """Pareto lengan faktor: biaya (n_out_tok construct) vs mutu (gate-pass-rate
    atau |IC| rerata) per mode — pendamping v04 (saran #7)."""
    trace = _load(RESULTS / "pendukung" / "agent_trace_perhop.json")
    if not trace:
        return _lewati("v16", "agent_trace_perhop.json tak ada")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    any_pt = False
    for m in ORDER:
        d, sumber_key = _muat_faktor(comm_mode, m)
        if not d:
            continue
        runs = d["runs"]
        n = len(runs)
        gate_rate = sum(1 for r in runs if r.get("passing")) / n if n else None
        ics = [abs(f["ic"]) for r in runs for f in (r.get("factors") or []) if f.get("ic") is not None]
        mean_abs_ic = (sum(ics) / len(ics)) if ics else None
        tag = f"{comm_mode}_{m}" if not (m == "raw" and comm_mode == "text") else "text"
        rows = trace.get("matriks", {}).get(tag) or trace.get("arsip_faktor_6jalan_2026-08-10", {}).get(tag)
        if not rows or gate_rate is None:
            continue
        construct = next((r for r in rows if r["agent"] == "construct"), None)
        if not construct:
            continue
        n_out = construct["metrik"].get("n_out_tok", {}).get("median")
        if n_out is None:
            continue
        y = mean_abs_ic if mean_abs_ic is not None else gate_rate
        size = 60 + gate_rate * 200
        ax.scatter(n_out, y, s=size, color=COLOR[m], alpha=0.85, edgecolor="white", linewidth=0.9, zorder=3)
        ax.annotate(f"{m}\n(gate {gate_rate*100:.0f}%)", (n_out, y), fontsize=6.8, color=COLOR[m],
                   weight="bold", xytext=(6, 4), textcoords="offset points")
        any_pt = True
    if not any_pt:
        plt.close(fig)
        return _lewati("v16", "tak ada mode dengan trace+gate lengkap")
    ax.set_xlabel("n_out_tok median (construct)", fontsize=9)
    ax.set_ylabel("|IC| rerata ekspresi ber-IC (fallback: gate-pass-rate)", fontsize=8.5)
    ax.set_title(f"Pareto lengan faktor: biaya token vs mutu — comm_mode={comm_mode}\n"
               "(ukuran bubble ∝ gate-pass-rate; label = gate-pass-rate)",
               fontsize=9, weight="bold")
    _save(fig, f"v16_pareto_faktor_{comm_mode}")


def v17_korupsi_cjk_posisi() -> None:
    """Histogram posisi relatif kemunculan PERTAMA aksara CJK dalam jawaban
    bench, per mode (medium kv) — saran #6. Regex sama dgn hitung_token.py."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    any_pt = False
    for m in ORDER:
        positions = []
        for task in TASKS:
            fp = RESULTS / "bench" / f"bench_{task}_{m}_kv_s0.json"
            d = _load(fp)
            if not d:
                continue
            for r in d.get("results", []):
                txt = r.get("answer_text") or ""
                mobj = _CJK_RE.search(txt)
                if mobj and len(txt) > 0:
                    positions.append(mobj.start() / len(txt))
        if positions:
            any_pt = True
            jitter = np.random.RandomState(0).uniform(-0.15, 0.15, len(positions))
            y0 = ORDER.index(m)
            ax.scatter(positions, y0 + jitter, color=COLOR[m], s=26, alpha=0.7, zorder=3)
            ax.text(1.03, y0, f"n={len(positions)}", fontsize=7, color=COLOR[m], va="center")
    if not any_pt:
        plt.close(fig)
        return _lewati("v17", "tak ada jawaban bench dengan aksara CJK ditemukan")
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=9)
    ax.set_xlim(-0.05, 1.15)
    ax.set_xlabel("posisi relatif kemunculan PERTAMA aksara CJK dalam teks jawaban (0=awal, 1=akhir)",
                fontsize=8)
    ax.set_title("Di mana korupsi lintas-bahasa muncul dalam jawaban? (lengan bench, medium kv)\n"
               "kecil tapi hidup — tiap titik satu jawaban yang terjangkit",
               fontsize=9, weight="bold")
    _save(fig, "v17_korupsi_cjk_posisi")


def v18_panjang_repetisi_jawaban() -> None:
    """Violin panjang jawaban (karakter) per mode, lengan bench, medium kv —
    saran #4. Rasio-repetisi bench diaproksimasi word-level (1 - unik/total)
    karena bench TAK menyimpan agent_trace/rep_ratio (hanya lengan faktor)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    len_series, rep_series, labels, colors = [], [], [], []
    for m in ORDER:
        lens, reps = [], []
        for task in TASKS:
            d = _load(RESULTS / "bench" / f"bench_{task}_{m}_kv_s0.json")
            if not d:
                continue
            for r in d.get("results", []):
                txt = r.get("answer_text") or ""
                if not txt:
                    continue
                lens.append(len(txt))
                words = txt.split()
                if len(words) >= 5:
                    reps.append(1 - len(set(words)) / len(words))
        if lens:
            len_series.append(lens); rep_series.append(reps or [0]); labels.append(m); colors.append(COLOR[m])
    if not len_series:
        plt.close(fig)
        return _lewati("v18", "tak ada answer_text bench ditemukan")

    for ax, series, title, ylab in [
        (axes[0], len_series, "Panjang jawaban (karakter)", "panjang (karakter)"),
        (axes[1], rep_series, "Rasio repetisi kata (1 − unik/total)", "rasio repetisi"),
    ]:
        parts = ax.violinplot(series, showmeans=True, showextrema=True)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_alpha(0.65); pc.set_edgecolor(c)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color(DARK_GREY)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Panjang & repetisi jawaban per formulasi — lengan bench, medium kv, 3 tugas digabung",
               fontsize=9.6, weight="bold", y=1.04)
    _save(fig, "v18_panjang_repetisi_jawaban")


def v19_kanal_per_arm() -> None:
    """Recall per ARM (text/kv_full/kv_prompt_only/kv_latent_only/none) x mode
    — reproduksi visual dari temuan A9 (HASIL_TAHAP4 §3): mode `kv` produksi
    lossless BUKAN karena vektor latennya ekspresif, tapi karena token prompt
    hulu ikut diwariskan verbatim (`kv_prompt_only` ≈ `kv_full`, jauh di atas
    `kv_latent_only`)."""
    arms = ["text", "kv_full", "kv_prompt_only", "kv_latent_only", "none"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(arms))
    width = 0.15
    any_bar = False
    for i, m in enumerate(ORDER):
        name = "m10" if m == "gumbel" else f"{m}_m10"
        d = _load(RESULTS / "probe" / f"channel_capacity_Qwen_Qwen3-8B_{name}.json")
        if not d:
            continue
        by_arm = {}
        for row in d.get("_summary", []):
            by_arm.setdefault(row["arm"], []).append(row["recall"])
        vals = [np.mean(by_arm[a]) if a in by_arm else np.nan for a in arms]
        if any(not np.isnan(v) for v in vals):
            ax.bar(x + (i - 2) * width, vals, width, color=COLOR[m], label=m)
            any_bar = True
    if not any_bar:
        plt.close(fig)
        return _lewati("v19", "tak ada channel_capacity json ditemukan")
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.2)
    ax.set_ylabel("recall (rerata payload dsl+token)", fontsize=8.8)
    # x="none" selalu 0 di semua mode (lantai tebakan) — tempat aman utk legenda.
    ax.legend(fontsize=7.5, frameon=False, ncol=1, loc="upper right")
    ax.set_title("Recall per ARM kanal — replikasi temuan A9 (HASIL_TAHAP4 §3)\n"
               "kv_full ≈ kv_prompt_only ≫ kv_latent_only: prompt yg diwariskan, bukan vektor laten",
               fontsize=8.6, weight="bold", pad=10)
    _save(fig, "v19_kanal_per_arm")


def v20_kedalaman_ekspresi(comm_mode: str = "kv") -> None:
    """Kedalaman nesting tanda kurung per ekspresi, proksi kompleksitas
    struktural — bonus non-plot #1 dari saran, dijadikan plot."""
    def depth(expr: str) -> int:
        d = cur = 0
        for ch in expr:
            if ch == "(":
                cur += 1; d = max(d, cur)
            elif ch == ")":
                cur = max(0, cur - 1)
        return d

    per_mode = _semua_ekspresi_per_mode(comm_mode)
    if not per_mode:
        return _lewati("v20", "tak ada ekspresi faktor")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    series, labels, colors = [], [], []
    for m in ORDER:
        if m not in per_mode or not per_mode[m]:
            continue
        series.append([depth(e) for e in per_mode[m]]); labels.append(m); colors.append(COLOR[m])
    if not series:
        plt.close(fig)
        return _lewati("v20", "korpus ekspresi kosong")
    parts = ax.violinplot(series, showmeans=True, showextrema=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.65); pc.set_edgecolor(c)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(DARK_GREY)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("kedalaman nesting tanda kurung", fontsize=8.8)
    ax.set_title(f"Kompleksitas struktural ekspresi per formulasi — comm_mode={comm_mode}",
               fontsize=9.3, weight="bold")
    _save(fig, f"v20_kedalaman_ekspresi_{comm_mode}")


def v21_statistik_latent_stop() -> None:
    """Alasan berhenti rollout laten (budget vs fixed_point) per mode — bonus
    non-plot #2. Uji-ulang temuan B6 (HASIL_TAHAP4 §1): early-stop nyaris tak
    pernah menyala utk gumbel/soft/sample/moi (stokastik, tak pernah membeku)."""
    d = _load(RESULTS / "pendukung" / "agent_trace_perhop.json")
    if not d:
        return _lewati("v21", "agent_trace_perhop.json tak ada")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    labels, budget_pct, fixed_pct = [], [], []
    for m in ORDER:
        tag = f"kv_{m}" if m != "raw" else "kv_raw"
        rows = d.get("matriks", {}).get(tag) or d.get("arsip_faktor_6jalan_2026-08-10", {}).get(tag)
        if not rows:
            continue
        tot_budget = tot_fixed = tot_lain = 0
        for r in rows:
            sr = r.get("latent_stop_reasons") or {}
            # Label sebenarnya di engine.py: "budget" | "early_stop" | "off"
            # (backend/llm/engine.py:590-634) — BUKAN "fixed_point"/"cos_threshold".
            tot_budget += sr.get("budget", 0)
            tot_fixed += sr.get("early_stop", 0)
            tot_lain += sum(v for k, v in sr.items() if k not in ("budget", "early_stop"))
        tot = tot_budget + tot_fixed + tot_lain
        if tot == 0:
            continue
        labels.append(m); budget_pct.append(100 * tot_budget / tot); fixed_pct.append(100 * tot_fixed / tot)
    if not labels:
        plt.close(fig)
        return _lewati("v21", "tak ada latent_stop_reasons terekam")
    x = np.arange(len(labels))
    ax.bar(x, fixed_pct, 0.55, label="berhenti dini (fixed-point)", color="#0ca30c")
    ax.bar(x, budget_pct, 0.55, bottom=fixed_pct, label="habis anggaran (budget)", color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% hop", fontsize=8.8)
    # Batang menutupi 0-100% penuh — tak ada ruang kosong; legenda diberi kotak
    # putih semi-transparan alih-alih didorong ke atas judul (itu yang tabrakan
    # di v11/v19 sebelum diperbaiki).
    ax.legend(fontsize=7.5, frameon=True, facecolor="white", framealpha=0.92,
             edgecolor="none", loc="center", ncol=1)
    ax.set_title("Sebab berhenti rollout laten per formulasi (B6, HASIL_TAHAP4 §1)\n"
               "prediksi: raw bisa membeku dini; gumbel/soft/sample/moi stokastik, jarang membeku",
               fontsize=8.6, weight="bold", pad=10)
    _save(fig, "v21_statistik_latent_stop")


_EMB_CACHE: dict = {}


def _muat_embedding_qwen(model_id: str = "Qwen/Qwen3-8B"):
    """W_in (CPU, hanya safetensors, TANPA memuat model penuh) + tokenizer.
    Dipakai HANYA untuk mean-pooling token->vektor sebagai representasi
    semantik ekspresi — bukan forward pass, jadi tak menyentuh GPU sama
    sekali dan aman dijalankan sementara lengan faktor masih memakai kartu."""
    if model_id in _EMB_CACHE:
        return _EMB_CACHE[model_id]
    from eval.realign_probe import load_matrices
    from transformers import AutoTokenizer
    print(f"  [v22] memuat embedding {model_id} dari safetensors (CPU, sekali saja) ...")
    W_in, _, _ = load_matrices(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    _EMB_CACHE[model_id] = (W_in.float(), tok)
    return _EMB_CACHE[model_id]


def v22_kemiripan_semantik(comm_mode: str = "kv") -> None:
    """Keragaman ekspresi lewat EMBEDDING SEMANTIK (bukan sintaksis/AST) —
    mean-pool token embedding Qwen3-8B (W_in, sama dgn ruang yg dipakai
    seluruh Sumbu A) per ekspresi, lalu kemiripan kosinus berpasangan dalam
    korpus tiap mode. Rerata TINGGI = ekspresi mode itu berkerumun di satu
    ide (mode collapse); rerata RENDAH = keragaman ide lebih luas. Permintaan
    eksplisit: kemiripan dari embedding semantik saja, bukan AST/himpunan fungsi
    (v15/v20 sudah menutupi sisi sintaksisnya)."""
    per_mode = _semua_ekspresi_per_mode(comm_mode)
    if not per_mode:
        return _lewati("v22", "tak ada ekspresi faktor")
    try:
        W_in, tok = _muat_embedding_qwen()
    except Exception as e:  # noqa: BLE001
        return _lewati("v22", f"gagal memuat embedding: {type(e).__name__}: {e}")

    import torch

    def embed(expr: str) -> "torch.Tensor":
        ids = tok(expr, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        vecs = W_in[ids]
        return torch.nn.functional.normalize(vecs.mean(dim=0), dim=0)

    labels, means, colors, ns = [], [], [], []
    for m in ORDER:
        exprs = list(dict.fromkeys(per_mode.get(m, [])))  # unik, urutan stabil
        if len(exprs) < 3:
            continue
        V = torch.stack([embed(e) for e in exprs])          # [n, d]
        sim = (V @ V.T).clamp(-1, 1)
        n = sim.shape[0]
        off_diag = sim[~torch.eye(n, dtype=torch.bool)]
        labels.append(m); means.append(float(off_diag.mean())); colors.append(COLOR[m]); ns.append(n)
    if not labels:
        return _lewati("v22", "korpus per-mode terlalu kecil (<3 ekspresi unik)")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, means, 0.55, color=colors)
    for xi, v, n in zip(x, means, ns):
        ax.text(xi, v + 0.01, f"{v:.3f}\n(n={n})", ha="center", fontsize=7.4, color=DARK_GREY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("kemiripan kosinus berpasangan rerata\n(embedding Qwen3-8B, mean-pool token)", fontsize=7.8)
    ax.set_ylim(0, max(means) * 1.35 if means else 1)
    ax.set_title(f"Keragaman SEMANTIK korpus ekspresi per formulasi — comm_mode={comm_mode}\n"
               "lebih tinggi = ekspresi berkerumun di satu ide (mode collapse)",
               fontsize=9, weight="bold")
    _save(fig, f"v22_kemiripan_semantik_{comm_mode}")


# ══════════════════════════════════════════════════════════════════════════
# LAPISAN 6 — dari scripts/faktor_perhop.py (diadaptasi dari `09_faktor_perhop.py`,
# 2026-08-27): fidelitas terhadap ARAH yang ditugaskan (muatan DIKETAHUI pada
# tugas nyata, bukan probe sintetis) + biaya percobaan-ulang per jalan.
# ══════════════════════════════════════════════════════════════════════════

def _muat_perhop() -> dict | None:
    return _load(RESULTS / "pendukung" / "faktor_perhop.json")


def v23_fidelitas_arah() -> None:
    """Dua panel: (a) peluruhan fidelitas-arah sepanjang rantai pada medium
    yang teksnya lengkap (text, kv_and_text) — inilah versi SAH dari 'degradasi
    per-hop' yang v10 tolak untuk parsed_ok; di sini metriknya (Jaccard kata
    thd kalimat arah yang diketahui) memang terukur di SEMUA hop yang berteks.
    (b) fidelitas construct SAJA lintas kelima formulasi kv — perbandingan yang
    adil karena kv-only memang cuma punya satu titik teks (construct)."""
    d = _muat_perhop()
    if not d:
        return _lewati("v23", "results/pendukung/faktor_perhop.json tak ada — jalankan scripts/faktor_perhop.py")
    by_tag = {s["tag"]: s for s in d["per_sel"]}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # panel A — rantai penuh (medium ber-teks di setiap hop)
    ax = axes[0]
    any_a = False
    multi_hop_tags = [("text", "raw", "text"), ("kv_and_text_soft", "soft", "kv_and_text"),
                      ("kv_and_text_raw", "raw", "kv_and_text"),
                      ("kv_and_text_gumbel", "gumbel", "kv_and_text"),
                      ("kv_and_text_sample", "sample", "kv_and_text"),
                      ("kv_and_text_moi", "moi", "kv_and_text")]
    for tag, mode, medium in multi_hop_tags:
        s = by_tag.get(tag)
        if not s:
            continue
        pts = [(i, h["fidelitas_arah"]) for i, h in enumerate(s["per_hop"])
              if h["hop"] in AGENTS_FACTOR and h["fidelitas_arah"] is not None]
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ls = "--" if medium == "text" else "-"
            ax.plot(xs, ys, marker="o", ls=ls, color=COLOR[mode],
                   label=f"{mode} ({medium})", lw=1.8, ms=6)
            any_a = True
    if any_a:
        ax.set_xticks(range(len(AGENTS_FACTOR))); ax.set_xticklabels(AGENTS_FACTOR, fontsize=8.5)
        ax.set_ylabel("fidelitas-arah (Jaccard kata thd arah yang ditugaskan)", fontsize=7.6)
        ax.legend(fontsize=6.6, frameon=False, loc="upper right")
        ax.set_title("(a) Peluruhan sepanjang rantai\n(medium berteks di tiap hop)", fontsize=8.6)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "[belum ada medium ber-teks-penuh]",
                                ha="center", transform=ax.transAxes, fontsize=8, color=GREY)

    # panel B — construct saja, lintas 5 formulasi kv
    ax = axes[1]
    labels, fid, mech, colors = [], [], [], []
    for m in ORDER:
        s = by_tag.get(f"kv_{m}" if m != "raw" else "kv_raw")
        if not s:
            continue
        h = next((x for x in s["per_hop"] if x["hop"] == "construct"), None)
        if h and h["fidelitas_arah"] is not None:
            labels.append(m); fid.append(h["fidelitas_arah"]); mech.append(h["mech_arah"] or 0)
            colors.append(COLOR[m])
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.18, fid, 0.34, color=colors, label="fidelitas-arah (kata)")
        ax.bar(x + 0.18, mech, 0.34, color=colors, alpha=0.45, label="mech-arah (mekanisme)")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("kesetiaan thd arah (construct)", fontsize=8)
        ax.legend(fontsize=6.8, frameon=False, loc="upper right")
        ax.set_title("(b) construct saja — lintas formulasi\n(medium kv, satu2nya titik teks)", fontsize=8.6)
    else:
        plt.close(fig)
        return _lewati("v23", "tak ada hop construct dengan fidelitas_arah")

    fig.suptitle("Kesetiaan terhadap arah eksplorasi yang DIKETAHUI — muatan pada tugas nyata\n"
               "(bukan probe sintetis; Jaccard kata thd kalimat arah, backend/factor/run_factor.py::DIRECTIONS)",
               fontsize=9, weight="bold", y=1.08)
    _save(fig, "v23_fidelitas_arah")


def v24_biaya_percobaan() -> None:
    """percobaan_construct_per_jalan (retry gate+repair) per formulasi —
    kuantifikasi bersih dari klaim 'raw butuh lebih banyak percobaan', dgn
    denominator SERAGAM per jalan (bukan per panggilan LLM, lihat docstring
    faktor_perhop.py butir 1)."""
    d = _muat_perhop()
    if not d:
        return _lewati("v24", "results/pendukung/faktor_perhop.json tak ada")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels, vals, gate, colors = [], [], [], []
    for m in ORDER:
        s = next((x for x in d["per_sel"] if x["tag"] == (f"kv_{m}" if m != "raw" else "kv_raw")), None)
        if not s:
            continue
        labels.append(m); vals.append(s["percobaan_construct_per_jalan"])
        gate.append(s["laju_jalan_lolos_gate"]); colors.append(COLOR[m])
    if not labels:
        plt.close(fig)
        return _lewati("v24", "tak ada sel kv_* di faktor_perhop.json")
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.55, color=colors)
    ax.axhline(1.0, color=GREY, lw=1, ls=":")
    for xi, v, g in zip(x, vals, gate):
        ax.text(xi, v + 0.03, f"{v:.2f}×\n(gate {g*100:.0f}%)", ha="center", fontsize=7.6, color=DARK_GREY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("percobaan construct+repair / jalan", fontsize=8.8)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("Biaya percobaan-ulang per jalan — comm_mode=kv\n"
               "denominator seragam (per arah×seed, bukan per panggilan LLM)",
               fontsize=9, weight="bold")
    _save(fig, "v24_biaya_percobaan")


FIGURES = {
    "v01_peta_eksperimen": v01_peta_eksperimen,
    "v02_arsitektur_sistem": v02_arsitektur_sistem,
    "v03_heatmap_akurasi": v03_heatmap_akurasi,
    "v04_pareto_token_waktu": v04_pareto_token_waktu,
    "v05_geometri_rentang": v05_geometri_rentang,
    "v06_geometri_vs_kinerja": v06_geometri_vs_kinerja,
    "v07_pencarian_beta_moi": v07_pencarian_beta_moi,
    "v08_interpolasi_geometri": v08_interpolasi_geometri,
    "v09_funnel_fidelitas": v09_funnel_fidelitas,
    "v10_akumulasi_kv_hop": v10_akumulasi_kv_hop,
    "v11_upaya_perbaikan": v11_upaya_perbaikan,
    "v12_matriks_fidelitas": v12_matriks_fidelitas,
    "v13_distribusi_ic": v13_distribusi_ic,
    "v14_holdout_vs_seleksi": v14_holdout_vs_seleksi,
    "v15_cakupan_fungsi_dsl": v15_cakupan_fungsi_dsl,
    "v16_pareto_faktor": v16_pareto_faktor,
    "v17_korupsi_cjk_posisi": v17_korupsi_cjk_posisi,
    "v18_panjang_repetisi_jawaban": v18_panjang_repetisi_jawaban,
    "v19_kanal_per_arm": v19_kanal_per_arm,
    "v20_kedalaman_ekspresi": v20_kedalaman_ekspresi,
    "v21_statistik_latent_stop": v21_statistik_latent_stop,
    "v22_kemiripan_semantik": v22_kemiripan_semantik,
    "v23_fidelitas_arah": v23_fidelitas_arah,
    "v24_biaya_percobaan": v24_biaya_percobaan,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n in FIGURES:
            print(n)
        return

    todo = list(FIGURES) if (args.all or not args.names) else args.names
    print(f"=== membangkitkan {len(todo)} figur -> {OUT} ===")
    for n in todo:
        fn = FIGURES.get(n)
        if not fn:
            print(f"  [?] '{n}' bukan nama figur dikenal — lihat --list")
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — satu figur gagal tak boleh menghentikan sisanya
            print(f"  [GAGAL] {n}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
