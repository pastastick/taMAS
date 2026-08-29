#!/usr/bin/env python3
"""Gambar Bab 4 — bentuk visual mengikuti Figure 1/6/7 LatentMAS (arXiv:2511.20639).

Keluaran PNG 300 dpi ke skripsi/assets/images/ — direktori itu memang untuk
berkas gambar, jadi keluarannya raster, bukan PDF. 300 dpi aman untuk cetak
skripsi; sumber kebenarannya tetap skrip ini, jadi gambar bisa dibangkitkan
ulang kapan saja. Palet kategorial diambil dari
palet rujukan tervalidasi (lolos enam pemeriksaan; dua slot di bawah kontras
3:1 sehingga SEMUA batang diberi label nilai langsung — aturan relief).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analisis"
IMG = ROOT / "skripsi" / "assets" / "images"
IMG.mkdir(parents=True, exist_ok=True)

# warna per ENTITAS (tetap, tak pernah dirotasi)
WARNA = {
    "raw": "#e34948", "soft": "#2a78d6", "gumbel": "#1baf7a",
    "sample": "#eda100", "moi": "#4a3aa7",
}
ABU, ABU_TUA = "#9a9a95", "#52514e"
URUT = ["raw", "soft", "gumbel", "sample", "moi"]
TUGAS = ["gsm8k", "arc_challenge", "humanevalplus"]
LABEL = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c9c4", "axes.labelcolor": "#0b0b0b",
    "xtick.color": ABU_TUA, "ytick.color": ABU_TUA,
    "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150,
})


def muat():
    df = pd.read_csv(AN / "bench_tabel.csv")
    ring = json.loads((AN / "bench_ringkas.json").read_text())
    return df, ring


# ── Gambar 1: akurasi · token · waktu (meniru Figure 1 LatentMAS) ──────────
def gambar_utama(df):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.55))
    x = np.arange(len(TUGAS))
    lebar = 0.16

    # (a) akurasi — label langsung di tiap batang (aturan relief: dua slot
    #     warna di bawah kontras 3:1, jadi identitas tak boleh warna saja)
    ax = axes[0]
    for i, m in enumerate(URUT):
        v = [df[(df.tugas == t) & (df.sel == f"{m}/kv")].akurasi.iloc[0] for t in TUGAS]
        b = ax.bar(x + (i - 2) * lebar, v, lebar * 0.86, color=WARNA[m], label=m,
                   zorder=3)
        ax.bar_label(b, fmt="%.2f", fontsize=4.9, padding=1.2, color=ABU_TUA,
                     rotation=90)
    for t_i, t in enumerate(TUGAS):
        for sel, gaya in (("raw/text", "--"), ("raw/baseline", ":")):
            v = df[(df.tugas == t) & (df.sel == sel)].akurasi.iloc[0]
            ax.plot([t_i - 2.7 * lebar, t_i + 2.7 * lebar], [v, v], gaya,
                    color=ABU_TUA, lw=1.0, zorder=2)
    ax.plot([], [], "--", color=ABU_TUA, lw=1.0, label="teks")
    ax.plot([], [], ":", color=ABU_TUA, lw=1.0, label="agen tunggal")
    ax.set_xticks(x, [LABEL[t] for t in TUGAS], fontsize=7)
    ax.set_xlim(-0.58, 2.58)
    ax.set_ylim(0.3, 1.12)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Akurasi")
    ax.set_title("(a) Akurasi — medium KV", fontsize=8.5, loc="left")
    ax.legend(fontsize=6.0, ncol=4, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.40), columnspacing=0.9, handlelength=1.2)

    # (b)/(c) angka lengkapnya ada di tabel utama Bab 4; di sini hanya satu
    # sorotan per tugas supaya bentuknya terbaca (mengikuti Figure 1 LatentMAS)
    for ax, kolom, judul, ylab in (
            (axes[1], "token_per_soal", "(b) Biaya token", "Token keluaran per soal"),
            (axes[2], "detik_per_soal", "(c) Waktu", "Detik per soal")):
        for i, m in enumerate(URUT):
            v = [df[(df.tugas == t) & (df.sel == f"{m}/kv")][kolom].iloc[0]
                 for t in TUGAS]
            ax.bar(x + (i - 2) * lebar, v, lebar * 0.86, color=WARNA[m], zorder=3)
        for t_i, t in enumerate(TUGAS):
            d = df[df.tugas == t]
            ref = d[d.sel == "raw/text"][kolom].iloc[0]
            kv_min = min(d[d.sel == f"{m}/kv"][kolom].iloc[0] for m in URUT)
            ax.plot([t_i - 2.7 * lebar, t_i + 2.7 * lebar], [ref, ref], "--",
                    color=ABU_TUA, lw=1.0, zorder=4)
            ax.annotate(f"teks {ref:.0f}", (t_i, ref), fontsize=5.8, color=ABU_TUA,
                        ha="center", va="bottom", xytext=(0, 2),
                        textcoords="offset points")
            ax.annotate(f"↓{(1 - kv_min / ref) * 100:.0f}%", (t_i, kv_min),
                        fontsize=6.2, color="#b03a39", ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points", weight="bold")
        ax.set_xticks(x, [LABEL[t] for t in TUGAS], fontsize=7)
        ax.set_xlim(-0.58, 2.58)
        ax.set_ylabel(ylab)
        ax.set_title(judul, fontsize=8.5, loc="left")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)

    fig.tight_layout()
    fig.savefig(IMG / "hasil_utama.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Gambar 2: disosiasi (gambar kepala Bab 4) ──────────────────────────────
def gambar_disosiasi(ring):
    k = {c["tugas"]: c for c in ring["kontras_keluarga"]}
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    y = np.arange(len(TUGAS))[::-1]
    for i, t in enumerate(TUGAS):
        c = k[t]
        d, lo, hi = c["delta"], c["ci_lo"], c["ci_hi"]
        warna = "#e34948" if lo > 0 else ABU
        ax.errorbar(d, y[i], xerr=[[d - lo], [hi - d]], fmt="o", ms=7,
                    color=warna, ecolor=warna, elinewidth=1.8, capsize=4, zorder=3)
        ax.annotate(f"{d:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]", (hi, y[i]),
                    xytext=(8, 0), textcoords="offset points", fontsize=7.5,
                    va="center", color=ABU_TUA)
    ax.axvline(0, color="#52514e", lw=1.0, zorder=2)
    ax.set_yticks(y, [LABEL[t] for t in TUGAS])
    ax.set_xlim(-0.06, 0.62)
    ax.set_xlabel("Selisih akurasi: rerata keluarga relaksasi − ridge $W_a$ (`raw`)")
    ax.set_title("Kerusakan langkah laten resmi menurut tuntutan presisi simbolik",
                 fontsize=9, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "disosiasi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Gambar 3: geometri langkah laten (tandingan Figure 6 LatentMAS) ────────
def gambar_geometri():
    g = json.loads((ROOT / "quantalatent" / "results" / "probe" /
                    "b7_probe_Qwen_Qwen3-8B.json").read_text())["geometry"]
    mode = [m for m in URUT if m in g]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for i, m in enumerate(mode):
        v = g[m]
        ax.bar(i, v["max_cos_embed_mean"], 0.6, color=WARNA[m], zorder=3)
        ax.plot([i, i], [v["max_cos_embed_min"], v["max_cos_embed_max"]],
                color="#0b0b0b", lw=1.2, zorder=4)
        ax.annotate(f"{v['max_cos_embed_mean']:.2f}",
                    (i, v["max_cos_embed_max"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=ABU_TUA)
    ax.set_xticks(range(len(mode)), mode)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"$\max_i \cos(\Phi(h),\, W_{\mathrm{in}}[i])$")
    ax.set_title("Kedekatan vektor laten ke embedding token terdekat (Qwen3-8B)",
                 fontsize=9, loc="left")
    ax.axhline(1.0, color=ABU, lw=0.9, ls=":", zorder=2)
    ax.annotate("tepat di embedding token", (len(mode) - 0.5, 1.0),
                xytext=(0, 3), textcoords="offset points", ha="right",
                fontsize=6.5, color=ABU_TUA)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "geometri_langkah_laten.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Gambar 4: akurasi vs biaya token ───────────────────────────────────────
def gambar_efisiensi(df):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=False)
    for ax, t in zip(axes, TUGAS):
        d = df[df.tugas == t]
        for m in URUT:
            r = d[d.sel == f"{m}/kv"].iloc[0]
            ax.scatter(r.token_per_soal, r.akurasi, s=64, color=WARNA[m],
                       zorder=3, edgecolor="white", linewidth=1.2)
            ax.annotate(m, (r.token_per_soal, r.akurasi), xytext=(5, 3),
                        textcoords="offset points", fontsize=6.8, color=ABU_TUA)
        for sel, mark, lab in (("raw/text", "s", "teks"),
                               ("raw/baseline", "^", "agen tunggal")):
            r = d[d.sel == sel].iloc[0]
            ax.scatter(r.token_per_soal, r.akurasi, s=64, color=ABU_TUA,
                       marker=mark, zorder=3, edgecolor="white", linewidth=1.2)
            ax.annotate(lab, (r.token_per_soal, r.akurasi), xytext=(5, 3),
                        textcoords="offset points", fontsize=6.8, color=ABU_TUA)
        ax.set_xscale("log")
        ax.set_xlabel("Token keluaran per soal (skala log)")
        ax.set_title(LABEL[t], fontsize=9, loc="left")
    axes[0].set_ylabel("Akurasi")
    fig.tight_layout()
    fig.savefig(IMG / "akurasi_vs_token.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df, ring = muat()
    gambar_utama(df)
    gambar_disosiasi(ring)
    gambar_geometri()
    gambar_efisiensi(df)
    print("[tulis]", *(p.name for p in sorted(IMG.glob("*.png"))))


if __name__ == "__main__":
    main()
