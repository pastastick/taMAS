#!/usr/bin/env python3
"""Bangkitkan berkas lampiran LaTeX dari data hasil.

  Lampiran A  tabel penuh 21 sel lengan benchmark
  Lampiran B  tabel penuh 63 uji berpasangan + p terkoreksi
  Lampiran C  transkrip rantai empat agen (satu soal, dua persamaan laten)
  Lampiran F  hiperparameter dan lingkungan eksekusi

Keluaran → skripsi/bab/lampiran_*.tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analisis"
BAB = ROOT / "skripsi" / "bab"
SNAP = Path("/tmp/results/bench/llm_outputs")
BENCH = Path("/tmp/results/bench")

LABEL = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}


def esc(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def koma(x, p=2, kosong="---"):
    return kosong if x is None or pd.isna(x) else f"{x:.{p}f}".replace(".", ",")


# ── Lampiran A ──────────────────────────────────────────────────────────────
def lampiran_a(df: pd.DataFrame) -> str:
    b = [r"\chapter{Tabel Lengkap Sel Lengan \textit{Benchmark}}",
         r"\label{lamp:sel-bench}", "",
         r"Seluruh 21 sel utama lengan \textit{benchmark}. Kolom "
         r"\textit{format} adalah proporsi keluaran berbentuk sah; kolom CJK "
         r"adalah cacah jawaban yang memuat aksara Tionghoa pada tugas "
         r"berbahasa Inggris.", "",
         r"\begin{longtable}{llrrrrrr}", r"\hline",
         r"Tugas & Sel & $n$ & Akurasi & SK 95\% & Format & Token/soal & CJK \\",
         r"\hline\endhead"]
    for t in ["gsm8k", "arc_challenge", "humanevalplus"]:
        for _, r in df[df.tugas == t].sort_values("sel").iterrows():
            sel = r.sel.replace("_", r"\_")
            b.append(
                f"{LABEL[t]} & \\texttt{{{sel}}} & {int(r.n)} & {koma(r.akurasi)}"
                f" & [{koma(r.akurasi_ci_lo)}; {koma(r.akurasi_ci_hi)}]"
                f" & {koma(r.format_rate)} & {koma(r.token_per_soal, 0)}"
                f" & {koma(r.n_jawaban_ber_cjk, 0)} \\\\")
        b.append(r"\hline")
    b += [r"\end{longtable}"]
    return "\n".join(b)


# ── Lampiran B ──────────────────────────────────────────────────────────────
def lampiran_b(du: pd.DataFrame) -> str:
    b = [r"\chapter{Hasil Lengkap Uji Berpasangan}", r"\label{lamp:uji}", "",
         r"Seluruh 63 uji McNemar eksak beserta nilai $p$ terkoreksi Holm dan "
         r"Benjamini--Hochberg. Baris diurutkan menaik menurut $p$ dalam tiap "
         r"tugas. Kolom $n_a$ dan $n_b$ adalah cacah soal diskordan, yaitu soal "
         r"yang hanya dijawab benar oleh salah satu perlakuan.", "",
         r"\begin{longtable}{llrlrrrr}", r"\hline",
         r"Tugas & Pasangan & $\Delta$ & SK 95\% & $n_a$ & $n_b$ & $p$ & "
         r"$p_{\text{Holm}}$ \\", r"\hline\endhead"]
    for t in ["gsm8k", "arc_challenge", "humanevalplus"]:
        d = du[du.tugas == t].sort_values("p_mcnemar")
        for _, r in d.iterrows():
            a = r.a.replace("_", r"\_")
            bb = r.b.replace("_", r"\_")
            b.append(
                f"{LABEL[t]} & \\texttt{{{a}}}--\\texttt{{{bb}}}"
                f" & {r.delta:+.3f}".replace(".", ",")
                + f" & [{r.ci_lo:+.2f}; {r.ci_hi:+.2f}]".replace(".", ",")
                + f" & {int(r.n_a_saja)} & {int(r.n_b_saja)}"
                + f" & {koma(r.p_mcnemar, 4)} & {koma(r.p_holm, 4)} \\\\")
        b.append(r"\hline")
    b += [r"\end{longtable}"]
    return "\n".join(b)


# ── Lampiran C ──────────────────────────────────────────────────────────────
def _aman(s: str) -> str:
    """Ganti aksara non-ASCII dengan penanda titik kode.

    Mesin pdfLaTeX tak dapat menyusun aksara di luar ASCII di dalam
    `lstlisting`. Penggantian ini bukan sekadar penyelamat kompilasi: aksara
    non-ASCII pada transkrip berbahasa Inggris justru merupakan korupsi token
    yang dibahas pada Bab~IV, sehingga menampilkannya sebagai titik kode
    membuatnya terbaca dan dapat dirujuk.
    """
    return "".join(c if ord(c) < 128 else f"[U+{ord(c):04X}]" for c in s)


def _potong(p: Path) -> tuple[str, str]:
    t = p.read_text(encoding="utf-8", errors="replace")
    peran = re.search(r"^# Call \d+ . `(\w+)`", t, re.M)
    i = t.find("## Response")
    m = re.search(r"```text\n(.*?)\n```", t[i:], re.S) if i >= 0 else None
    return (peran.group(1) if peran else "?"), _aman(m.group(1) if m else "")


def lampiran_c() -> str:
    b = [r"\chapter{Transkrip Rantai Agen}", r"\label{lamp:transkrip}", "",
         r"Keluaran verbatim keempat agen untuk satu soal GSM8K yang sama, "
         r"dibandingkan antara persamaan langkah laten resmi (\texttt{raw}) dan "
         r"salah satu anggota keluarga relaksasi diskret (\texttt{gumbel}), "
         r"pada medium gabungan KV dan teks. Teks tidak disunting; pemenggalan "
         r"baris ditambahkan agar muat di halaman.", ""]
    for mode in ("raw", "gumbel"):
        sesi = sorted(SNAP.glob(f"gsm8k_{mode}_kv_and_text_lampiran/*/"))
        if not sesi:
            continue
        berkas = sorted(sesi[-1].glob("000[1-4]_*.md"))
        b.append(rf"\section*{{Persamaan langkah laten \texttt{{{mode}}}}}")
        for p in berkas:
            peran, teks = _potong(p)
            teks = teks.strip()
            if len(teks) > 1100:
                teks = teks[:1100] + "\n[...dipotong...]"
            b += [rf"\subsection*{{Agen \texttt{{{peran}}}}}",
                  r"\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,"
                  r"breaklines=true,frame=single]",
                  teks, r"\end{lstlisting}", ""]

    # Contoh korupsi token yang terekam otomatis di seluruh sel.
    tok = json.loads((Path("/tmp/results/pendukung") / "token_bench.json").read_text())
    rusak = [r for r in tok if r.get("contoh_korupsi")
             and not r["sel"].endswith("_lampiran")]
    if rusak:
        b += [r"\section*{Contoh Korupsi Token yang Terekam}", "",
              r"Cuplikan jawaban yang memuat aksara di luar ASCII pada tugas "
              r"berbahasa Inggris. Aksara tersebut ditampilkan sebagai titik "
              r"kode Unicode.", "",
              r"\begin{longtable}{lp{8cm}}", r"\hline",
              r"Sel & Cuplikan \\", r"\hline\endhead"]
        for r in sorted(rusak, key=lambda r: r["sel"]):
            sel = r["sel"].replace("_", r"\_")
            b.append(f"\\texttt{{{sel}}} & "
                     f"\\texttt{{{esc(_aman(r['contoh_korupsi'].strip()))}}} \\\\")
        b += [r"\hline", r"\end{longtable}"]
    return "\n".join(b)


# ── Lampiran F ──────────────────────────────────────────────────────────────
def lampiran_f() -> str:
    meta = json.loads((BENCH / "bench_gsm8k_gumbel_kv_s0.json").read_text())["_meta"]
    probe = json.loads((ROOT / "quantalatent" / "results" / "probe" /
                        "realign_probe_Qwen_Qwen3-8B.json").read_text())
    baris = [
        ("Model", r"\texttt{Qwen/Qwen3-8B}"),
        ("Ukuran kosakata $V$", f"{probe['vocab']:,}".replace(",", ".")),
        ("Dimensi \\textit{hidden} $d$", str(probe["d_h"])),
        ("Embedding terikat", "tidak"),
        ("Langkah laten $m$", str(meta["latent_steps"])),
        ("Suhu langkah laten $T$", koma(meta["latent_temp"])),
        ("Regularisasi ridge $\\lambda$", "$10^{-5}$"),
        ("Parameter MoI $\\beta$", "1"),
        ("Suhu pembangkitan", koma(meta["temperature"])),
        ("\\textit{top-p}", koma(meta["top_p"])),
        ("Token baru maksimum", str(meta["max_new_tokens"])),
        ("Rantai agen", ", ".join(rf"\texttt{{{a}}}" for a in meta["chain"])),
        ("Subsampel per sel", str(meta["limit"])),
        ("\\textit{Seed} pengambilan sampel", str(meta["sample_seed"])),
        ("\\textit{Seed} pembangkitan", str(meta["seed"])),
        ("GPU", "NVIDIA A40 46 GB, CUDA 12.8"),
        ("Kerangka kerja", "PyTorch 2.6.0+cu124, HuggingFace Transformers"),
    ]
    b = [r"\chapter{Hiperparameter dan Lingkungan Eksekusi}",
         r"\label{lamp:hparam}", "",
         r"\begin{longtable}{ll}", r"\hline",
         r"Parameter & Nilai \\", r"\hline\endhead"]
    b += [f"{k} & {v} \\\\" for k, v in baris]
    b += [r"\hline", r"\end{longtable}"]
    return "\n".join(b)


def main() -> None:
    df = pd.read_csv(AN / "bench_tabel.csv")
    du = pd.read_csv(AN / "bench_uji.csv")
    (BAB / "lampiran_a_sel.tex").write_text(lampiran_a(df) + "\n")
    (BAB / "lampiran_b_uji.tex").write_text(lampiran_b(du) + "\n")
    (BAB / "lampiran_c_transkrip.tex").write_text(lampiran_c() + "\n")
    (BAB / "lampiran_f_hparam.tex").write_text(lampiran_f() + "\n")
    print("[tulis]", *(p.name for p in sorted(BAB.glob("lampiran_*.tex"))))


if __name__ == "__main__":
    main()
