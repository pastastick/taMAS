#!/usr/bin/env python3
"""Mutu ekspresi (Level 5) per sel — baru mungkin setelah 14/14 sel diskor.

KENAPA BARU SEKARANG. Sampai 2026-08-27 hanya tiga sel (`kv_soft`, `kv_sample`,
`kv_raw`) punya `icseries_*.parquet`; sisanya berhenti sebelum selesai karena
scorer lama menulis atomik di akhir. Perbandingan mutu lintas formulasi karena
itu tak bisa dibuat sama sekali — yang bisa dilaporkan cuma Level 4 (lolos gate).
Skoring 2026-08-28 melengkapi keempatbelasnya, jadi pertanyaan "apakah keluarga R
menghasilkan ekspresi yang LEBIH BAIK, bukan sekadar LEBIH SERING VALID" akhirnya
punya data.

STATUSNYA DESKRIPTIF, BUKAN UJI FORMAL. Kebijakan TEORI.md §4.6 mematok SATU
kontras formal di lengan faktor — keluarga R vs `raw` pada laju lolos gate, unit
analisis satu jalan. Angka di sini unitnya satu EKSPRESI, dan ekspresi dari jalan
yang sama tidak independen (satu panggilan `construct` melahirkan beberapa
sekaligus, berbagi hipotesis dan arah). Statistik uji tetap dicetak sebagai
ukuran besaran, tapi *p*-nya TIDAK boleh dikutip sebagai bukti konfirmatori;
menambah uji formal kedua juga akan merusak kebijakan satu-kontras itu sendiri.

Yang paling berarti dibaca di sini justru DISOSIASINYA: bandingkan kolom
`laju_lolos_gate` (Level 4) dengan `mean_abs_ic` (Level 5). Kalau geometri
menaikkan yang pertama tanpa menyentuh yang kedua, maka yang diperbaiki
formulasi R adalah KEMAMPUAN MENGHASILKAN SIMBOL YANG SAH, bukan mutu sinyalnya.

    python scripts/analisis_mutu_ic.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "pendukung"
R_FAMILY = ["soft", "sample", "gumbel", "moi"]


def _metode(tag: str) -> str:
    for m in R_FAMILY + ["raw"]:
        if tag.endswith("_" + m) or tag == m:
            return m
    return "mix" if "mix" in tag else "raw"


def _medium(tag: str) -> str:
    if tag == "text":
        return "text"
    return "kv_and_text" if tag.startswith("kv_and_text") else "kv"


def per_sel() -> list[dict]:
    baris = []
    for p in sorted((RESULTS / "factor").glob("frontend_*.json")):
        tag = p.stem[len("frontend_"):]
        runs = json.loads(p.read_text())["runs"]
        fs = [f for r in runs for f in (r.get("factors") or []) if f.get("expression")]
        ics = [float(f["ic"]) for f in fs if f.get("ic") is not None]
        aic = [abs(v) for v in ics]
        ts = [abs(float(f["tstat"])) for f in fs if f.get("tstat") is not None]
        sh = [float(f["bt_sharpe"]) for f in fs if f.get("bt_sharpe") is not None]
        baris.append({
            "tag": tag,
            "medium": _medium(tag),
            "metode": _metode(tag),
            "n_jalan": len(runs),
            "n_ekspresi": len(fs),
            "n_ber_ic": len(ics),
            "laju_ber_ic": round(len(ics) / len(fs), 4) if fs else None,
            "mean_abs_ic": round(statistics.fmean(aic), 5) if aic else None,
            "median_abs_ic": round(statistics.median(aic), 5) if aic else None,
            "p90_abs_ic": round(sorted(aic)[int(0.9 * (len(aic) - 1))], 5) if aic else None,
            "maks_abs_ic": round(max(aic), 5) if aic else None,
            "n_tstat_ge2": sum(1 for t in ts if t >= 2.0),
            "laju_tstat_ge2": round(sum(1 for t in ts if t >= 2.0) / len(ts), 4) if ts else None,
            "mean_bt_sharpe": round(statistics.fmean(sh), 4) if sh else None,
            "mean_abs_bt_sharpe": round(statistics.fmean(abs(v) for v in sh), 4) if sh else None,
            "_aic": aic,
        })
    return baris


def banding(baris: list[dict]) -> dict:
    """Keluarga R vs raw pada |IC|, per medium. Mann-Whitney U: tak berpasangan
    (ekspresi di kedua kelompok berbeda) dan tak mengandaikan normalitas, yang
    tepat karena sebaran |IC| menumpuk di dekat nol dan berekor panjang."""
    from scipy.stats import mannwhitneyu
    hasil = {}
    for med in ("kv", "kv_and_text"):
        sel = {b["metode"]: b for b in baris if b["medium"] == med and b["metode"] != "mix"}
        if "raw" not in sel:
            continue
        r_aic = [v for m in R_FAMILY if m in sel for v in sel[m]["_aic"]]
        raw_aic = sel["raw"]["_aic"]
        if not r_aic or not raw_aic:
            continue
        u = mannwhitneyu(r_aic, raw_aic, alternative="two-sided")
        n1, n2 = len(r_aic), len(raw_aic)
        hasil[med] = {
            "n_ekspresi_R": n1, "n_ekspresi_raw": n2,
            "mean_abs_ic_R": round(statistics.fmean(r_aic), 5),
            "mean_abs_ic_raw": round(statistics.fmean(raw_aic), 5),
            "median_abs_ic_R": round(statistics.median(r_aic), 5),
            "median_abs_ic_raw": round(statistics.median(raw_aic), 5),
            "rasio_mean": round(statistics.fmean(r_aic) / statistics.fmean(raw_aic), 3),
            "mannwhitney_u": float(u.statistic),
            "p_deskriptif": float(u.pvalue),
            "rank_biserial": round(2 * float(u.statistic) / (n1 * n2) - 1, 4),
            "PERINGATAN": "unit = ekspresi (TIDAK independen dalam satu jalan); "
                          "deskriptif — bukan uji konfirmatori, lihat TEORI.md §4.6",
        }
    return hasil


def main() -> None:
    baris = per_sel()
    bandingan = banding(baris)
    for b in baris:
        b.pop("_aic", None)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mutu_ic.json").write_text(json.dumps(
        {"_meta": {"catatan": "Level 5 (mutu ekspresi). DESKRIPTIF — kebijakan "
                              "satu-kontras-formal TEORI.md §4.6 tetap berlaku; "
                              "unit di sini ekspresi, bukan jalan."},
         "per_sel": baris, "keluarga_R_vs_raw": bandingan},
        indent=2, ensure_ascii=False))

    L = ["# Mutu ekspresi per sel (Level 5) — 14/14 sel sudah diskor", "",
         "Regenerasi: `python scripts/analisis_mutu_ic.py`", "",
         "DESKRIPTIF. Uji formal lengan faktor tetap satu: keluarga R vs `raw` pada",
         "laju lolos gate (unit = jalan). Di sini unitnya ekspresi, dan ekspresi dari",
         "jalan yang sama tidak independen.", "",
         "| sel | medium | metode | ekspresi | ber-IC | mean abs IC | median | p90 | maks | t>=2 | mean abs Sharpe |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in baris:
        L.append(f"| {b['tag']} | {b['medium']} | {b['metode']} | {b['n_ekspresi']} | "
                 f"{b['laju_ber_ic']:.0%} | {b['mean_abs_ic']} | {b['median_abs_ic']} | "
                 f"{b['p90_abs_ic']} | {b['maks_abs_ic']} | "
                 f"{b['laju_tstat_ge2']:.0%} | {b['mean_abs_bt_sharpe']} |")

    L += ["", "## Keluarga R vs `raw` pada |IC| (deskriptif)", ""]
    for med, d in bandingan.items():
        L += [f"### medium `{med}`",
              f"- R: {d['n_ekspresi_R']} ekspresi, mean |IC| {d['mean_abs_ic_R']}, median {d['median_abs_ic_R']}",
              f"- raw: {d['n_ekspresi_raw']} ekspresi, mean |IC| {d['mean_abs_ic_raw']}, median {d['median_abs_ic_raw']}",
              f"- rasio mean R/raw: **{d['rasio_mean']}×**",
              f"- Mann-Whitney U={d['mannwhitney_u']:.0f}, p={d['p_deskriptif']:.4g} "
              f"(DESKRIPTIF), rank-biserial {d['rank_biserial']}", ""]

    (OUT / "mutu_ic.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nditulis → {OUT}/mutu_ic.json + mutu_ic.md")


if __name__ == "__main__":
    main()
