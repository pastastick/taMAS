#!/usr/bin/env python3
"""Figures for README.md — English labels, regenerated straight from results/.

The thesis figures live in `analisis/03_gambar.py` (Indonesian labels, vector
PDF). This script is deliberately separate and self-contained so the README
stays reproducible from this repository alone:

    python scripts/plot_readme_figures.py

Inputs (all tracked in git):
  results/bench/bench_*_s0.json               accuracy and wall-clock per cell
  results/pendukung/token_bench.json          output tokens per cell
  results/probe/b7_probe_Qwen_Qwen3-8B.json   latent-step geometry

Outputs: assets/main_results.png, assets/geometry.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# One colour per latent-step formulation, never rotated across figures.
COLOR = {
    "raw": "#e34948", "soft": "#2a78d6", "gumbel": "#1baf7a",
    "sample": "#eda100", "moi": "#4a3aa7",
}
GREY, DARK_GREY = "#9a9a95", "#52514e"
ORDER = ["raw", "soft", "gumbel", "sample", "moi"]
TASKS = ["gsm8k", "arc_challenge", "humanevalplus"]
LABEL = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c9c4", "axes.labelcolor": "#0b0b0b",
    "xtick.color": DARK_GREY, "ytick.color": DARK_GREY,
    "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150,
})


def load_cells() -> dict[tuple[str, str], dict]:
    """Main cells only: 100 questions, seed 0. The 5-question `kv_and_text`
    cells were transcript suppliers for the appendix, not experiment cells.

    Accuracy and wall-clock come from the run files themselves; output tokens
    are joined in from token_bench.json. The GSM8K single-agent cell has no
    token entry — it ran before the transcript collector existed — so `tokens`
    stays None there. It is only used as an accuracy reference line, which is
    why panels (b) and (c) never look it up.
    """
    tokens = {
        r["sel"]: r["token_per_soal"]
        for r in json.loads(
            (ROOT / "results" / "pendukung" / "token_bench.json").read_text())
    }
    cells: dict[tuple[str, str], dict] = {}
    for path in sorted((ROOT / "results" / "bench").glob("bench_*_s0.json")):
        run = json.loads(path.read_text())
        meta, summary = run["_meta"], run["summary"]
        if summary["n"] != 100:
            continue
        medium = "baseline" if meta.get("baseline") else meta["comm_mode"]
        name = path.stem.removeprefix("bench_")
        cells[(meta["task"], f"{meta['latent_mode']}/{medium}")] = {
            "accuracy": summary["accuracy"],
            "tokens": tokens.get(name),
            "seconds": meta["total_time_s"] / summary["n"],
        }
    return cells


def figure_main(cells: dict[tuple[str, str], dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.55))
    x = np.arange(len(TASKS))
    width = 0.16

    ax = axes[0]
    for i, m in enumerate(ORDER):
        vals = [cells[(t, f"{m}/kv")]["accuracy"] for t in TASKS]
        bars = ax.bar(x + (i - 2) * width, vals, width * 0.86, color=COLOR[m],
                      label=m, zorder=3)
        ax.bar_label(bars, fmt="%.2f", fontsize=4.9, padding=1.2,
                     color=DARK_GREY, rotation=90)
    for ti, t in enumerate(TASKS):
        for cell, style in (("raw/text", "--"), ("raw/baseline", ":")):
            v = cells[(t, cell)]["accuracy"]
            ax.plot([ti - 2.7 * width, ti + 2.7 * width], [v, v], style,
                    color=DARK_GREY, lw=1.0, zorder=2)
    ax.plot([], [], "--", color=DARK_GREY, lw=1.0, label="text")
    ax.plot([], [], ":", color=DARK_GREY, lw=1.0, label="single agent")
    ax.set_xticks(x, [LABEL[t] for t in TASKS], fontsize=7)
    ax.set_xlim(-0.58, 2.58)
    ax.set_ylim(0.3, 1.12)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Accuracy — KV medium", fontsize=8.5, loc="left")
    ax.legend(fontsize=6.0, ncol=4, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.40), columnspacing=0.9, handlelength=1.2)

    for ax, col, title, ylab in (
            (axes[1], "tokens", "(b) Output tokens", "Output tokens / question"),
            (axes[2], "seconds", "(c) Wall-clock", "Seconds / question")):
        for i, m in enumerate(ORDER):
            vals = [cells[(t, f"{m}/kv")][col] for t in TASKS]
            ax.bar(x + (i - 2) * width, vals, width * 0.86, color=COLOR[m], zorder=3)
        for ti, t in enumerate(TASKS):
            ref = cells[(t, "raw/text")][col]
            kv_min = min(cells[(t, f"{m}/kv")][col] for m in ORDER)
            ax.plot([ti - 2.7 * width, ti + 2.7 * width], [ref, ref], "--",
                    color=DARK_GREY, lw=1.0, zorder=4)
            ax.annotate(f"text {ref:.0f}", (ti, ref), fontsize=5.8, color=DARK_GREY,
                        ha="center", va="bottom", xytext=(0, 2),
                        textcoords="offset points")
            ax.annotate(f"↓{(1 - kv_min / ref) * 100:.0f}%", (ti, kv_min),
                        fontsize=6.2, color="#b03a39", ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points", weight="bold")
        ax.set_xticks(x, [LABEL[t] for t in TASKS], fontsize=7)
        ax.set_xlim(-0.58, 2.58)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)

    fig.tight_layout()
    fig.savefig(ASSETS / "main_results.png", bbox_inches="tight")
    plt.close(fig)


def figure_geometry() -> None:
    probe = json.loads(
        (ROOT / "results" / "probe" / "b7_probe_Qwen_Qwen3-8B.json").read_text())
    geom = probe["geometry"]
    modes = [m for m in ORDER if m in geom]

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for i, m in enumerate(modes):
        v = geom[m]
        ax.bar(i, v["max_cos_embed_mean"], 0.6, color=COLOR[m], zorder=3)
        ax.plot([i, i], [v["max_cos_embed_min"], v["max_cos_embed_max"]],
                color="#0b0b0b", lw=1.2, zorder=4)
        ax.annotate(f"{v['max_cos_embed_mean']:.2f}", (i, v["max_cos_embed_max"]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=7.5, color=DARK_GREY)
    ax.set_xticks(range(len(modes)), modes)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"$\max_i \cos(\Phi(h),\, W_{\mathrm{in}}[i])$")
    ax.set_title("Latent step vs. nearest token embedding (Qwen3-8B)",
                 fontsize=9, loc="left")
    ax.axhline(1.0, color=GREY, lw=0.9, ls=":", zorder=2)
    ax.annotate("exactly on a token embedding", (len(modes) - 0.5, 1.0),
                xytext=(0, 3), textcoords="offset points", ha="right",
                fontsize=6.5, color=DARK_GREY)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "geometry.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    figure_main(load_cells())
    figure_geometry()
    print("[written]", *(p.name for p in sorted(ASSETS.glob("*.png"))))


if __name__ == "__main__":
    main()
