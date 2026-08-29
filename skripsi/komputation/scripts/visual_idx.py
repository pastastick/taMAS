#!/usr/bin/env python3
"""Figur bab hasil untuk pasar Indonesia (LQ45).

Figur v01--v24 yang sudah ada dibangun dari `frontend_*.json`, yang medan
`ic`-nya berisi angka pasar A-share. Karena penelitian ini berpindah ke LQ45,
figur yang bergantung IC TIDAK boleh dipakai ulang apa adanya. Berkas ini
membangun ulang figur-figur itu dari laporan IDX, dan menambah beberapa yang
hanya masuk akal setelah dua jendela penilaian tersedia.

Palet warna diwarisi apa adanya dari `scripts/plot_readme_figures.py` supaya
pembaca tidak perlu mempelajari dua legenda untuk lima formulasi yang sama.

Keluaran: results/visual_idx/*.png (300 dpi)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from paths import bootstrap, RESULTS, OUT_FACTOR  # noqa: E402
bootstrap()

KELUAR = RESULTS / "visual_idx"
IDX = RESULTS / "idx"
SELEKSI = "holdout_daily_pv_idx_lq45_2021-01-01_2021-12-31_q0.2.json"
HOLDOUT = "holdout_daily_pv_idx_lq45_2022-01-01_2025-12-26_q0.2.json"

WARNA = {"raw": "#e34948", "soft": "#2a78d6", "gumbel": "#1baf7a",
         "sample": "#eda100", "moi": "#4a3aa7",
         "mix_a025": "#b06fa0", "mix_a05": "#8f7fb8", "mix_a075": "#6a7fc0",
         "-": "#777777"}
URUT = ["raw", "soft", "gumbel", "sample", "moi"]
LBL = {"text": "teks", "kv": "kv", "kv_and_text": "kv+teks"}
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def belah(tag):
    for m in ("kv_and_text", "kv", "text"):
        if tag == m:
            return m, "-"
        if tag.startswith(m + "_"):
            return m, tag[len(m) + 1:]
    return tag, "-"


def simpan(fig, nama):
    KELUAR.mkdir(parents=True, exist_ok=True)
    f = KELUAR / f"{nama}.png"
    fig.tight_layout()
    fig.savefig(f, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {f.name}")


def muat(nama):
    p = OUT_FACTOR / nama
    return json.loads(p.read_text()) if p.exists() else None


# ── i01: disosiasi — validitas vs mutu sinyal, berdampingan ─────────────────
def i01_disosiasi(d):
    """Figur utama skripsi: dua panel, satu klaim per panel."""
    u = d["uji_formal"]
    ms = d["uji_mutu_sinyal_seleksi"]
    media = [m for m in ("kv", "kv_and_text") if m in u]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))

    x = np.arange(len(media))
    lebar = 0.36
    pR = [u[m]["lolos_gate_per_jalan"]["p_R"] * 100 for m in media]
    pr = [u[m]["lolos_gate_per_jalan"]["p_raw"] * 100 for m in media]
    ax[0].bar(x - lebar/2, pR, lebar, label=r"keluarga $\mathcal{R}$", color="#2a78d6")
    ax[0].bar(x + lebar/2, pr, lebar, label="raw", color=WARNA["raw"])
    for i, m in enumerate(media):
        p = u[m]["lolos_gate_per_jalan"]["fisher_p"]
        ax[0].text(i, max(pR[i], pr[i]) + 4, f"$p$={p:.1e}", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([LBL[m] for m in media])
    ax[0].set_ylabel("lolos gate per jalan (%)"); ax[0].set_ylim(0, 118)
    ax[0].set_title("(a) VALIDITAS keluaran — berbeda tajam", fontsize=9.5)
    ax[0].legend(fontsize=8, frameon=False)

    mR = [ms[m]["median_abs_ic_R"] for m in media if m in ms]
    mr = [ms[m]["median_abs_ic_raw"] for m in media if m in ms]
    xm = np.arange(len(mR))
    ax[1].bar(xm - lebar/2, mR, lebar, label=r"keluarga $\mathcal{R}$", color="#2a78d6")
    ax[1].bar(xm + lebar/2, mr, lebar, label="raw", color=WARNA["raw"])
    atas = max(mR + mr) * 1.28
    ax[1].set_ylim(0, atas)
    for i, m in enumerate([m for m in media if m in ms]):
        p = ms[m]["mannwhitney_p"]
        ax[1].text(i, max(mR[i], mr[i]) + atas * 0.04, f"$p$={p:.2f}",
                   ha="center", fontsize=8)
    ax[1].set_xticks(xm)
    ax[1].set_xticklabels([LBL[m] for m in media if m in ms])
    ax[1].set_ylabel(r"median $|IC|$ (seleksi 2021)")
    ax[1].set_title("(b) MUTU sinyal — tidak berbeda", fontsize=9.5)
    ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Disosiasi: relaksasi diskret memperbaiki validitas, bukan mutu sinyal",
                 fontsize=10.5, y=1.04)
    simpan(fig, "i01_disosiasi")


# ── i02: sebaran |IC| per formulasi, dua jendela ────────────────────────────
def i02_sebaran_ic():
    s, h = muat(SELEKSI), muat(HOLDOUT)
    if not s:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for k, (doc, judul, T) in enumerate([(s, "Seleksi 2021", 247),
                                         (h, "Holdout 2022--2025", 958)]):
        if doc is None:
            continue
        data, label, warna = [], [], []
        for m in URUT:
            tag = f"kv_{m}"
            v = [abs(r["ic"]) for r in doc["per_ekspresi"]
                 if r["tag"] == tag and r.get("ic") is not None
                 and (r.get("n_unique") or 0) > 2]
            if v:
                data.append(v); label.append(m); warna.append(WARNA[m])
        bp = ax[k].boxplot(data, labels=label, patch_artist=True, showfliers=False,
                           widths=0.6, medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], warna):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        amb = 1.96 / np.sqrt(36 * T)
        ax[k].axhline(amb, ls="--", lw=1, color="#333")
        ax[k].text(0.55, amb * 1.05, f"ambang signifikansi = {amb:.4f}",
                   fontsize=7.5, color="#333")
        ax[k].set_title(f"{judul} ($T$={T})", fontsize=9.5)
        ax[k].set_xlabel("formulasi langkah laten (medium kv)")
    ax[0].set_ylabel(r"$|IC|$ per ekspresi")
    fig.suptitle("Sebaran mutu sinyal dan ambang deteksi yang menyertainya",
                 fontsize=10.5, y=1.03)
    simpan(fig, "i02_sebaran_ic")


# ── i03: stabilitas seleksi → holdout ──────────────────────────────────────
def i03_stabilitas():
    h = muat(HOLDOUT)
    if not h:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))
    xs, ys, cs = [], [], []
    for r in h["per_ekspresi"]:
        a, b = r.get("ic_seleksi"), r.get("ic")
        if a is None or b is None:
            continue
        _, mt = belah(r["tag"])
        xs.append(a); ys.append(b); cs.append(WARNA.get(mt, "#777"))
    ax[0].axhline(0, lw=0.8, color="#999"); ax[0].axvline(0, lw=0.8, color="#999")
    ax[0].scatter(xs, ys, s=7, c=cs, alpha=0.55, linewidths=0)
    lim = max(max(map(abs, xs)), max(map(abs, ys))) * 1.05
    ax[0].plot([-lim, lim], [-lim, lim], ls=":", lw=1, color="#333")
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel("IC jendela seleksi (2021)")
    ax[0].set_ylabel("IC jendela holdout (2022--2025)")
    n_sama = sum(1 for a, b in zip(xs, ys) if a * b > 0)
    ax[0].set_title(f"(a) {n_sama}/{len(xs)} ekspresi mempertahankan tanda "
                    f"({100*n_sama/len(xs):.1f}\\%)", fontsize=9.5)

    tag_urut = [f"kv_{m}" for m in URUT]
    per = {t["tag"]: t for t in h["per_tag"]}
    lab, val, col = [], [], []
    for t in tag_urut:
        if t in per and per[t]["berpasangan"]:
            lab.append(belah(t)[1])
            val.append(100 * per[t]["berbalik_tanda"] / per[t]["berpasangan"])
            col.append(WARNA[belah(t)[1]])
    ax[1].bar(lab, val, color=col, alpha=0.85)
    ax[1].axhline(50, ls="--", lw=1, color="#333")
    ax[1].text(len(lab) - 0.5, 51, "acak = 50\\%", ha="right", fontsize=8)
    ax[1].set_ylabel("ekspresi berbalik tanda (\\%)"); ax[1].set_ylim(0, 60)
    ax[1].set_title("(b) Laju berbalik tanda per formulasi", fontsize=9.5)
    simpan(fig, "i03_stabilitas")


# ── i04: efisiensi — biaya vs keandalan ────────────────────────────────────
def i04_efisiensi(d):
    g = d["generasi"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    penanda = {"text": "s", "kv": "o", "kv_and_text": "^"}
    for tag, v in g.items():
        m, mt = belah(tag)
        if mt not in WARNA:
            continue
        ax.scatter(v["token_keluar_per_jalan"], 100 * v["laju_lolos_gate"],
                   s=90, marker=penanda.get(m, "o"), color=WARNA[mt],
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(f"{mt}\n{m}", (v["token_keluar_per_jalan"],
                                   100 * v["laju_lolos_gate"]),
                    textcoords="offset points", xytext=(7, -3), fontsize=6.5)
    ax.set_xlabel("token keluaran per jalan (lebih kecil lebih baik)")
    ax.set_ylabel("lolos gate per jalan (\\%)")
    ax.set_title("Biaya token vs keandalan keluaran", fontsize=10)
    ax.set_ylim(20, 108)
    simpan(fig, "i04_efisiensi")


# ── i05: peluruhan alpha per tahun ─────────────────────────────────────────
def i05_peluruhan(d):
    pl = d.get("peluruhan_alpha", {}).get("per_tahun")
    if not pl:
        return
    th = sorted(pl)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(th, [pl[t]["mean_abs_ic"] for t in th], "o-", color="#2a78d6",
            label=r"rerata $|IC|$")
    ax.plot(th, [pl[t]["median_abs_ic"] for t in th], "s--", color="#1baf7a",
            label=r"median $|IC|$")
    ax.axvspan(-0.5, 0.5, color="#eda100", alpha=0.15)
    ax.text(0, ax.get_ylim()[1] * 0.95, "jendela seleksi", ha="center", fontsize=7.5)
    ax.set_xlabel("tahun"); ax.set_ylabel(r"$|IC|$ rerata lintas-ekspresi")
    ax.set_title("Peluruhan sinyal setelah jendela seleksi", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    simpan(fig, "i05_peluruhan")


def main():
    d = json.loads((IDX / "analisis_idx.json").read_text())
    print("membangun figur IDX:")
    i01_disosiasi(d)
    i02_sebaran_ic()
    i03_stabilitas()
    i04_efisiensi(d)
    i05_peluruhan(d)
    print(f"selesai → {KELUAR}")


if __name__ == "__main__":
    main()
