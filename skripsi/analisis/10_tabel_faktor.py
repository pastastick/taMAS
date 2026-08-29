#!/usr/bin/env python3
"""Tabel LaTeX lengan faktor — enam level bukti, denominator seragam.

Menggantikan `faktor_parsing.tex`, yang menghitung persentase kegagalan atas
denominator yang berbeda-beda antar sel (7 sampai 18 "panggilan") karena agen
`construct` dipanggil ulang setiap kali gate memicu repair. Persentase atas
basis yang tak sama tidak bisa dibandingkan; di sini semua laju memakai
denominator yang sama menurut rancangan, yaitu jumlah JALAN (arah x seed), dan
cacah percobaan dilaporkan sebagai kolom biaya tersendiri.

Masukan  : analisis/faktor_perhop.json          (skrip 09)
           quantalatent/results/factor/holdout_*.json  (eval/skor_holdout.py)
Keluaran : skripsi/assets/tables/faktor_keandalan.tex
           skripsi/assets/tables/faktor_efisiensi.tex
           skripsi/assets/tables/faktor_holdout.tex
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analisis"
FACTOR = ROOT / "quantalatent" / "results" / "factor"
TAB = ROOT / "skripsi" / "assets" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

URUT_METODE = ["raw", "soft", "gumbel", "sample", "moi"]
LABEL_MEDIUM = {"text": "teks", "kv": "kv", "kv_and_text": "kv+teks"}


def koma(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def urut(sel: list[dict]) -> list[dict]:
    """teks dulu, lalu kv, lalu kv+teks; di dalamnya urut sesuai M."""
    prio_medium = {"text": 0, "kv": 1, "kv_and_text": 2}
    return sorted(sel, key=lambda s: (prio_medium.get(s["medium"], 9),
                                      URUT_METODE.index(s["metode"])
                                      if s["metode"] in URUT_METODE else 9))


def tabel_keandalan(per_sel: list[dict]) -> str:
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Keandalan lengan faktor per JALAN (satu jalan = satu "
        r"pasangan arah $\times$ \textit{seed}). Denominatornya sama di semua "
        r"sel menurut rancangan, sehingga laju antar-sel sebanding. Kolom "
        r"terakhir mencatat berapa kali agen \textit{construct} harus "
        r"dipanggil per jalan: nilai di atas 1 berarti \textit{gate} memicu "
        r"perbaikan, sehingga ia sekaligus ukuran biaya dan ukuran keandalan.}",
        r"\label{tab:faktor-keandalan}", r"\small",
        r"\begin{tabular}{llrrrr}", r"\hline",
        r"Medium & Langkah laten & Jalan & Menghasilkan ekspresi & "
        r"Lolos \textit{gate} & Panggilan/jalan \\", r"\hline",
    ]
    medium_terakhir = None
    for s in urut(per_sel):
        if medium_terakhir is not None and s["medium"] != medium_terakhir:
            b.append(r"\hline")
        medium_terakhir = s["medium"]
        met = "---" if s["metode"] == "-" else f"\\texttt{{{s['metode']}}}"
        b.append(
            f"{LABEL_MEDIUM.get(s['medium'], s['medium'])} & {met} & "
            f"{s['n_jalan']} & "
            f"{s['jalan_berekspresi']} ({koma(s['laju_jalan_berekspresi'] * 100, 0)}\\%) & "
            f"{s['jalan_lolos_gate']} ({koma(s['laju_jalan_lolos_gate'] * 100, 0)}\\%) & "
            f"{koma(s['percobaan_construct_per_jalan'])}" + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def tabel_efisiensi(per_sel: list[dict], efis: list[dict]) -> str:
    by_tag = {e["tag"]: e for e in efis}
    teks = next((s for s in per_sel if s["medium"] == "text"), None)
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Biaya lengan faktor per jalan, dan penghematannya terhadap "
        r"medium teks. Rentangnya sejajar dengan yang terukur pada lengan "
        r"\textit{benchmark}, sehingga klaim efisiensi kolaborasi laten "
        r"terreplikasi pada lengan kedua dengan rantai agen, prompt, dan tugas "
        r"yang seluruhnya berbeda.}",
        r"\label{tab:faktor-efisiensi}", r"\small",
        r"\begin{tabular}{llrrrr}", r"\hline",
        r"Medium & Langkah laten & Token keluaran & Penghematan & "
        r"Detik & Percepatan \\", r"\hline",
    ]
    if teks:
        b.append(f"teks & --- & {teks['token_keluar_per_jalan']:.0f} & --- & "
                 f"{teks['detik_per_jalan']:.0f} & ---" + r" \\")
        b.append(r"\hline")
    medium_terakhir = None
    for s in urut(per_sel):
        if s["medium"] == "text":
            continue
        if medium_terakhir is not None and s["medium"] != medium_terakhir:
            b.append(r"\hline")
        medium_terakhir = s["medium"]
        e = by_tag.get(s["tag"], {})
        hemat = e.get("penghematan_token_vs_text")
        cepat = e.get("percepatan_vs_text")
        b.append(
            f"{LABEL_MEDIUM.get(s['medium'], s['medium'])} & "
            f"\\texttt{{{s['metode']}}} & "
            f"{s['token_keluar_per_jalan']:.0f} & "
            + (koma(hemat * 100, 1) + r"\%" if hemat is not None else "---")
            + f" & {s['detik_per_jalan']:.0f} & "
            + (f"$\\times{koma(cepat, 1)}$" if cepat is not None else "---")
            + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def tabel_holdout(hold: dict) -> str:
    awal, akhir = hold["window"]
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Ketahanan ekspresi di luar jendela pembentukannya. Angka "
        r"seleksi dihitung pada 2021, yaitu jendela yang juga dipakai sistem "
        r"untuk menyaring ekspresi; angka \textit{holdout} dihitung pada "
        + f"{awal}--{akhir}, "
        + r"periode yang tak pernah dipakai menyaring apa pun. "
        r"\textsc{Berbalik} adalah cacah ekspresi yang tanda RankIC-nya "
        r"berlawanan antara kedua jendela.}",
        r"\label{tab:faktor-holdout}", r"\small",
        r"\begin{tabular}{llrrrr}", r"\hline",
        r"Medium & Langkah laten & Hidup & $\overline{|\mathrm{RankIC}|}$ & "
        r"Signifikan & Berbalik \\", r"\hline",
    ]
    per_tag = {t["tag"]: t for t in hold["per_tag"]}
    tags = urut([{"tag": t, "medium": _medium(t), "metode": _metode(t)}
                 for t in per_tag])
    medium_terakhir = None
    for s in tags:
        t = per_tag[s["tag"]]
        if t["ekspresi"] == 0:
            continue
        if medium_terakhir is not None and s["medium"] != medium_terakhir:
            b.append(r"\hline")
        medium_terakhir = s["medium"]
        met = "---" if s["metode"] == "-" else f"\\texttt{{{s['metode']}}}"
        mai = t["mean_abs_ic"]
        b.append(
            f"{LABEL_MEDIUM.get(s['medium'], s['medium'])} & {met} & "
            f"{t['hidup']} & "
            + (koma(mai, 4) if mai is not None else "---")
            + f" & {t['signifikan']} & {t['berbalik_tanda']}" + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def _medium(tag: str) -> str:
    for m in ("kv_and_text", "kv"):
        if tag.startswith(m + "_"):
            return m
    return tag


def _metode(tag: str) -> str:
    m = _medium(tag)
    return "-" if m == tag else tag[len(m) + 1:]


def main() -> None:
    d = json.loads((AN / "faktor_perhop.json").read_text())
    (TAB / "faktor_keandalan.tex").write_text(tabel_keandalan(d["per_sel"]) + "\n")
    (TAB / "faktor_efisiensi.tex").write_text(
        tabel_efisiensi(d["per_sel"], d["efisiensi_vs_text"]) + "\n")
    ditulis = ["faktor_keandalan.tex", "faktor_efisiensi.tex"]

    # PEMILIHAN BERKAS HOLDOUT — EKSPLISIT, BUKAN `sorted(...)[-1]`.
    # Sejak panel IDX ditambahkan, direktori ini berisi LEBIH DARI SATU berkas
    # holdout: `holdout_<awal>_<akhir>.json` (A-share) dan
    # `holdout_daily_pv_idx_<indeks>_<awal>_<akhir>_q<q>.json` (IDX). Keduanya
    # cocok dengan pola `holdout_*.json`, dan secara alfabet berkas IDX ada di
    # BELAKANG — sehingga `[-1]` akan diam-diam mengganti isi tabel A-share
    # dengan angka IDX tanpa mengubah satu kata pun di keterangan tabelnya.
    # Itu persis jenis kesalahan yang tak terlihat sampai sidang. Karena itu
    # berkasnya dipilih lewat argumen, dan pasarnya dicetak saat dijalankan.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="",
                    help="nama berkas di results/factor (kosong = pilih otomatis: "
                         "A-share bila ada, jika tidak berkas satu-satunya)")
    ap.add_argument("--keluaran", default="faktor_holdout.tex",
                    help="nama berkas .tex keluaran (ganti untuk pasar kedua)")
    a = ap.parse_args()

    semua = sorted(FACTOR.glob("holdout_*.json"))
    if a.holdout:
        pilih = FACTOR / a.holdout
        if not pilih.exists():
            raise SystemExit(f"tidak ada: {pilih}")
    else:
        ashare = [p for p in semua if "idx" not in p.stem]
        pilih = ashare[-1] if ashare else (semua[-1] if semua else None)

    if pilih is not None:
        h = json.loads(pilih.read_text())
        pasar = h.get("pasar", "daily_pv")
        print(f"[holdout] berkas : {pilih.name}")
        print(f"[holdout] pasar  : {pasar}  kuantil={h.get('quantile', 0.1)}  "
              f"jendela={h.get('window')}")
        if len(semua) > 1:
            print("[holdout] ADA " + str(len(semua)) + " berkas holdout; "
                  "pakai --holdout <nama> untuk memilih yang lain: "
                  + ", ".join(p.name for p in semua))
        (TAB / a.keluaran).write_text(tabel_holdout(h) + "\n")
        ditulis.append(a.keluaran)
    else:
        print("[lewati] faktor_holdout.tex — jalankan "
              "`PYTHONPATH=backend python backend/eval/skor_holdout.py` dulu")

    print("[tulis]", *ditulis)


if __name__ == "__main__":
    main()
