#!/usr/bin/env python3
"""Bangkitkan tabel LaTeX Bab 4 dari berkas analisis (bukan salin tangan).

Bentuk Tabel utama meniru Tabel 1 LatentMAS (arXiv:2511.20639): baris
dikelompokkan per tugas dengan tiga baris metrik (Akurasi/Token/Waktu), kolom
dikelompokkan per perlakuan, ditutup satu kolom selisih.

Keluaran → skripsi/assets/tables/
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analisis"
TAB = ROOT / "skripsi" / "assets" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

TUGAS = ["gsm8k", "arc_challenge", "humanevalplus"]
LABEL = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}
KOLOM = ["raw/baseline", "raw/text", "raw/kv", "soft/kv", "gumbel/kv",
         "sample/kv", "moi/kv"]
JUDUL_KOLOM = ["Tunggal", "Teks", r"\texttt{raw}", r"\texttt{soft}",
               r"\texttt{gumbel}", r"\texttt{sample}", r"\texttt{moi}"]
RELAKSASI = ["soft/kv", "gumbel/kv", "sample/kv", "moi/kv"]


def f(x, p=2, kosong="---"):
    return kosong if x is None or pd.isna(x) else f"{x:.{p}f}".replace(".", ",")


def tabel_utama(df: pd.DataFrame) -> str:
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Hasil lengan \textit{benchmark}: akurasi, biaya token, dan "
        r"waktu untuk tujuh perlakuan pada tiga tugas ($n=100$ soal identik per "
        r"tugas, Qwen3-8B, $m=10$ langkah laten). Kolom \textsc{Selisih} "
        r"membandingkan rerata empat anggota keluarga relaksasi terhadap "
        r"\texttt{raw} pada medium KV.}",
        r"\label{tab:hasil-bench}",
        r"\small",
        r"\begin{tabular}{llrrrrrrrr}", r"\hline",
        r" & & \multicolumn{2}{c}{Acuan} & \multicolumn{5}{c}{Medium KV} & \\",
        r"\cline{3-4}\cline{5-9}",
        "Tugas & Metrik & " + " & ".join(JUDUL_KOLOM) + r" & Selisih \\", r"\hline",
    ]
    for t in TUGAS:
        d = df[df.tugas == t].set_index("sel")
        akur = {k: d.loc[k, "akurasi"] if k in d.index else None for k in KOLOM}
        tok = {k: d.loc[k, "token_per_soal"] if k in d.index else None for k in KOLOM}
        wkt = {k: d.loc[k, "detik_per_soal"] if k in d.index else None for k in KOLOM}
        rel = sum(akur[k] for k in RELAKSASI) / len(RELAKSASI)
        selisih = rel - akur["raw/kv"]

        def baris(nama, nilai, p, sel=None):
            isi = []
            for k in KOLOM:
                v = nilai[k]
                tebal = (k in RELAKSASI + ["raw/kv"] and v is not None
                         and v == max(x for x in
                                      (nilai[j] for j in RELAKSASI + ["raw/kv"])
                                      if x is not None) and nama == "Akurasi")
                s = f(v, p)
                isi.append(r"\textbf{" + s + "}" if tebal else s)
            return f"& {nama} & " + " & ".join(isi) + " & " + (sel or "") + r" \\"

        b.append(rf"\multirow{{3}}{{*}}{{{LABEL[t]}}} " +
                 baris("Akurasi", akur, 2, f"{selisih:+.3f}".replace(".", ",")))
        b.append(baris("Token", tok, 0))
        b.append(baris("Waktu (dtk)", wkt, 1))
        b.append(r"\hline")
    b += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def tabel_uji(du: pd.DataFrame) -> str:
    d = du[du.p_holm < 0.05].sort_values("p_mcnemar")
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Seluruh pasangan yang tetap signifikan setelah koreksi Holm "
        r"atas 63 uji McNemar eksak. Tak satu pun pasangan pada GSM8K dan ARC-C "
        r"bertahan; seluruh pasangan yang bertahan berasal dari HumanEval+ dan "
        r"melibatkan \texttt{raw} pada medium KV.}",
        r"\label{tab:uji-signifikan}", r"\small",
        r"\begin{tabular}{llrrrr}", r"\hline",
        r"Tugas & Pasangan & $\Delta$ & SK 95\% & $p$ & $p_{\text{Holm}}$ \\",
        r"\hline",
    ]
    def notasi_p(x: float) -> str:
        """p dalam notasi ilmiah LaTeX dengan koma desimal."""
        m, e = f"{x:.1e}".split("e")
        return (rf"${m.replace('.', ',')}\times 10^{{{int(e)}}}$")

    for _, r in d.iterrows():
        a = r.a.replace("_", r"\_")
        bb = r.b.replace("_", r"\_")
        b.append(
            f"{r.tugas_label} & \\texttt{{{a}}} vs \\texttt{{{bb}}} & "
            + f"{r.delta:+.3f}".replace(".", ",")
            + f" & [{r.ci_lo:+.2f}; {r.ci_hi:+.2f}]".replace(".", ",")
            + " & " + notasi_p(r.p_mcnemar)
            + " & " + notasi_p(r.p_holm) + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def tabel_efisiensi(ring: dict, df: pd.DataFrame) -> str:
    """Efisiensi per FORMULASI, berdampingan dengan akurasinya.

    Kolom akurasi ditambahkan 2026-08-27. Tanpa kolom itu tabel ini hanya
    menjawab "apakah laten lebih murah dari teks"; RM3 juga menanyakan pengaruh
    formulasi terhadap efisiensi, dan pengaruh itu baru terbaca kalau biaya dan
    akurasi berdiri di satu baris — pembaca harus bisa melihat sendiri bahwa
    formulasi yang paling merusak akurasi juga yang paling boros token.
    """
    akurasi = {(r.tugas, r.metode): r.akurasi
               for r in df[df.medium == "kv"].itertuples()}
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Efisiensi medium KV relatif terhadap medium teks, per "
        r"persamaan langkah laten. Penghematan token dihitung atas token "
        r"keluaran per soal; percepatan atas waktu dinding per soal. Kolom "
        r"akurasi diulang dari Tabel~\ref{tab:hasil-bench} supaya biaya dan "
        r"mutu terbaca pada satu baris.}",
        r"\label{tab:efisiensi}", r"\small",
        r"\begin{tabular}{llrrr}", r"\hline",
        r"Tugas & Langkah laten & Akurasi & Penghematan token & Percepatan \\",
        r"\hline",
    ]
    tugas_terakhir = None
    for e in ring["efisiensi"]:
        m = e["sel"].split("/")[0]
        if tugas_terakhir is not None and e["tugas"] != tugas_terakhir:
            b.append(r"\hline")
        tugas_terakhir = e["tugas"]
        akur = akurasi.get((e["tugas"], m))
        b.append(f"{LABEL[e['tugas']]} & \\texttt{{{m}}} & "
                 + (f"{akur:.2f}".replace(".", ",") if akur is not None else "---")
                 + " & "
                 + f"{e['penghematan_token_vs_text'] * 100:.1f}\\%".replace(".", ",")
                 + f" & $\\times{e['percepatan_vs_text']:.1f}$".replace(".", ",")
                 + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── Tabel yang DULU diketik tangan di Bab4_TA2.tex ──────────────────────────
# Cochran Q, kontras keluarga, disosiasi, dan korupsi token semuanya ditulis
# manual di berkas .tex, sementara angkanya hidup di bench_ringkas.json dan
# bench_tabel.csv. Setiap kali analisis dijalankan ulang, angka di .tex tidak
# ikut berubah — dan tak ada yang memberi tahu. Dipindah ke sini supaya satu
# sumber angka, bukan dua.

def _p_teks(p: float, ambang: float = 1e-4) -> str:
    """p sebagai teks tabel: kecil sekali jadi '<0,0001', sisanya 4 desimal."""
    if p < ambang:
        return r"$<0{,}0001$"
    s = f"{p:.4f}".rstrip("0")
    return s.replace(".", ",")


def tabel_cochran(ring: dict) -> str:
    baris = sorted(ring["cochran_q"], key=lambda r: TUGAS.index(r["tugas"]))
    k = len(baris[0]["sel"]) if baris else 5
    b = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{Uji Cochran $Q$ atas {k} persamaan langkah laten pada "
        r"medium KV ($n=100$ soal berpasangan).}",
        r"\label{tab:cochran}",
        r"\begin{tabular}{lrrr}", r"\hline",
        r"Tugas & $Q$ & db & $p$ \\", r"\hline",
    ]
    for r in baris:
        b.append(f"{LABEL[r['tugas']]} & "
                 + f"{r['Q']:.2f}".replace(".", ",")
                 + f" & {r['df']} & " + _p_teks(r["p"]) + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def tabel_kontras(ring: dict) -> str:
    baris = sorted(ring["kontras_keluarga"], key=lambda r: TUGAS.index(r["tugas"]))
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Kontras akurasi antara rerata keluarga relaksasi diskret "
        r"dan ridge $W_a$ (\texttt{raw}) pada medium KV, dengan selang "
        r"kepercayaan \textit{bootstrap} 95\%.}",
        r"\label{tab:kontras}",
        r"\begin{tabular}{lrrl}", r"\hline",
        r"Tugas & \texttt{raw} & Keluarga & $\Delta$ dan SK 95\% \\", r"\hline",
    ]
    for r in baris:
        b.append(f"{LABEL[r['tugas']]} & "
                 + f"{r['akurasi_raw']:.2f}".replace(".", ",") + " & "
                 + f"{r['akurasi_keluarga_rerata']:.3f}".replace(".", ",") + " & "
                 + f"${r['delta']:+.3f}$".replace(".", "{,}")
                 + r" \quad [" + f"${r['ci_lo']:+.3f}$".replace(".", "{,}")
                 + "; " + f"${r['ci_hi']:+.3f}$".replace(".", "{,}") + "]"
                 + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def tabel_disosiasi(ring: dict) -> str:
    """Perbandingan besar kerusakan antar-tugas.

    Orientasi baris dibalik supaya tugas dengan kerusakan LEBIH BESAR selalu
    disebut lebih dulu, dan $\\Delta\\Delta$ karena itu selalu positif. Data
    mentahnya menyimpan pasangan menurut urutan alfabet tugas, yang membuat
    tanda selisihnya berpindah-pindah — persis jenis ketidakkonsistenan yang
    membuat tabel ketik-tangan sebelumnya sulit dicocokkan dengan sumbernya.
    """
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Perbandingan besar kerusakan $\Delta$ antar-tugas. Tugas "
        r"dengan kerusakan lebih besar disebut lebih dulu pada tiap baris.}",
        r"\label{tab:disosiasi}",
        r"\begin{tabular}{lrll}", r"\hline",
        r"Pasangan tugas & $\Delta\Delta$ & SK 95\% & $p$ \\", r"\hline",
    ]
    for r in ring["disosiasi"]:
        a, bb = r["tugas_a"], r["tugas_b"]
        d, lo, hi = r["selisih_delta"], r["ci_lo"], r["ci_hi"]
        if d < 0:                       # balik supaya yang lebih rusak di depan
            a, bb = bb, a
            d, lo, hi = -d, -hi, -lo
        b.append(f"{LABEL[a]} vs {LABEL[bb]} & "
                 + f"${d:+.3f}$".replace(".", "{,}") + " & ["
                 + f"${lo:+.3f}$".replace(".", "{,}") + "; "
                 + f"${hi:+.3f}$".replace(".", "{,}") + "] & "
                 + _p_teks(r["p_dua_sisi"]) + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def tabel_korupsi(df: pd.DataFrame) -> str:
    """Cacah jawaban ber-aksara Tionghoa pada tugas berbahasa Inggris."""
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Jumlah jawaban yang memuat aksara Tionghoa pada tugas "
        r"berbahasa Inggris, dari 100 jawaban per sel.}",
        r"\label{tab:korupsi}", r"\small",
        r"\begin{tabular}{l" + "r" * len(KOLOM) + "}", r"\hline",
        "Tugas & " + " & ".join(JUDUL_KOLOM) + r" \\", r"\hline",
    ]
    cacah = {(r.tugas, r.sel): r.n_jawaban_ber_cjk for r in df.itertuples()}
    jumlah = {k: 0 for k in KOLOM}
    for t in TUGAS:
        sel = []
        for k in KOLOM:
            v = cacah.get((t, k))
            # Sel tanpa agregat token (sel agen tunggal GSM8K, lihat Bab IV
            # §Cakupan Data) tak punya cacah korupsi — dibedakan dari nol.
            sel.append("---" if v is None or v != v else f"{int(v)}")
            if v is not None and v == v:
                jumlah[k] += int(v)
        b.append(f"{LABEL[t]} & " + " & ".join(sel) + r" \\")
    b += [r"\hline",
          "Jumlah & " + " & ".join(str(jumlah[k]) for k in KOLOM) + r" \\",
          r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def tabel_geometri() -> str:
    g = json.loads((ROOT / "quantalatent" / "results" / "probe" /
                    "b7_probe_Qwen_Qwen3-8B.json").read_text())["geometry"]
    p = json.loads((ROOT / "quantalatent" / "results" / "probe" /
                    "realign_probe_Qwen_Qwen3-8B.json").read_text())
    b = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Geometri vektor langkah laten pada Qwen3-8B. Kolom terakhir "
        r"adalah kosinus terhadap baris matriks embedding masukan yang paling "
        r"dekat; nilai $1$ berarti vektor tepat berimpit dengan embedding sebuah "
        r"token.}",
        r"\label{tab:geometri}", r"\small",
        r"\begin{tabular}{lccc}", r"\hline",
        r"Langkah laten & Bentuk & Anggota ($\star$) & "
        r"$\max_i \cos(\Phi(h), W_{\text{in}}[i])$ \\", r"\hline",
    ]
    bentuk = {
        "raw": r"$\rho\,hM/\lVert hM\rVert$",
        "soft": r"$w = p$",
        "gumbel": r"$w = \mathrm{softmax}((\ell+g)/T)$",
        "sample": r"$w = e_y$",
        "moi": r"$w = \lambda p + (1-\lambda)e_y$",
    }
    for m in ["raw", "soft", "gumbel", "sample", "moi"]:
        if m not in g:
            continue
        v = g[m]
        b.append(f"\\texttt{{{m}}} & {bentuk[m]} & "
                 + ("tidak" if m == "raw" else "ya") + " & "
                 + f"{v['max_cos_embed_mean']:.3f}".replace(".", ",")
                 + f" [{v['max_cos_embed_min']:.2f}; {v['max_cos_embed_max']:.2f}]"
                 .replace(".", ",") + r" \\")
    b += [r"\hline", r"\multicolumn{4}{l}{\footnotesize "
          r"$\lVert M-I\rVert_F/\lVert I\rVert_F = "
          + f"{p['relative_deviation']:.2f}".replace(".", ",")
          + r"$;\quad $\cos(h, hM)$ rerata $= "
          + f"{p['cos(h, hM)_mean']:.4f}".replace(".", ",")
          + r"$;\quad $\cos(W_{\text{in}}[i], W_{\text{out}}[i])$ rerata $= "
          + f"{p['cos(W_in_row, W_out_row)_mean']:.4f}".replace(".", ",")
          + r"$.} \\",
          r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def main() -> None:
    df = pd.read_csv(AN / "bench_tabel.csv")
    du = pd.read_csv(AN / "bench_uji.csv")
    ring = json.loads((AN / "bench_ringkas.json").read_text())
    (TAB / "hasil_bench.tex").write_text(tabel_utama(df) + "\n")
    (TAB / "uji_signifikan.tex").write_text(tabel_uji(du) + "\n")
    (TAB / "efisiensi.tex").write_text(tabel_efisiensi(ring, df) + "\n")
    (TAB / "geometri.tex").write_text(tabel_geometri() + "\n")
    (TAB / "cochran.tex").write_text(tabel_cochran(ring) + "\n")
    (TAB / "kontras.tex").write_text(tabel_kontras(ring) + "\n")
    (TAB / "disosiasi.tex").write_text(tabel_disosiasi(ring) + "\n")
    (TAB / "korupsi.tex").write_text(tabel_korupsi(df) + "\n")
    print("[tulis]", *(p.name for p in sorted(TAB.glob("*.tex"))))


if __name__ == "__main__":
    main()
