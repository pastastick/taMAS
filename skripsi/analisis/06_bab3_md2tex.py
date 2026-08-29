#!/usr/bin/env python3
"""Konversi draf Bab III (markdown) menjadi berkas LaTeX skripsi.

Tulang punggung: `assets/briefing/bab 3/BAB III.md` — draf paling lengkap
sebagai bab metodologi (desain, objek, variabel, protokol, hipotesis, metrik,
statistik, disiplin klaim, batasan). Dua draf lain dipakai sebagai pelengkap:
tabel parameter dari `bab_3_metode_penelitian.md`, sedangkan penurunan panjang
di `BAB III (1).md` sengaja TIDAK diikutkan karena materinya sudah menjadi
bagian Bab II menurut kerangka landasan teori.

Aturan naskah yang ditegakkan di sini:
  - tanpa \\textbf di badan teks; penekanan istilah memakai \\textit
  - istilah asing dimiringkan
  - matematika bertanda `=` diberi nomor sebagai `equation`
"""
from __future__ import annotations

import re
from pathlib import Path

import argparse

ROOT = Path(__file__).resolve().parents[1]


def inline(s: str) -> str:
    """Konversi penanda markdown sebaris.

    Rentang matematika `\\( ... \\)` DILINDUNGI lebih dulu: di dalamnya `%`
    dan `&` sudah bermakna LaTeX yang benar, dan penanda `*` adalah perkalian
    atau superskrip, bukan penekanan. Tanpa perlindungan ini `\\%` di dalam
    math berubah menjadi `\\\\%` (ganti baris) dan merusak kompilasi.
    """
    simpan: list[str] = []

    def titip(m):
        simpan.append(m.group(0))
        return f"\x00{len(simpan) - 1}\x00"

    s = re.sub(r"\\\(.+?\\\)", titip, s, flags=re.S)
    s = re.sub(r"`([^`]+)`", lambda m: r"\code{" + m.group(1).replace("_", r"\_") + "}", s)
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"\\textit{\1}", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textit{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\textit{\1}", s)
    s = re.sub(r"(?<!\\)&", r"\\&", s)
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    s = s.replace("∎", r"$\blacksquare$").replace("□", r"$\square$")
    s = s.replace("→", r"$\rightarrow$").replace("×", r"$\times$")
    s = s.replace("≈", r"$\approx$").replace("≤", r"$\le$").replace("≥", r"$\ge$")
    s = s.replace("’", "'").replace("“", "``").replace("”", "''")
    for i, asli in enumerate(simpan):
        s = s.replace(f"\x00{i}\x00", asli)
    return s


def konversi(teks: str) -> str:
    baris = teks.splitlines()
    out: list[str] = []
    i, n = 0, len(baris)
    n_eq = 0
    while i < n:
        l = baris[i]
        s = l.strip()

        if s.startswith("# "):                      # judul bab
            i += 1
            continue
        if s == "---" or s == "":
            out.append("")
            i += 1
            continue

        m = re.match(r"^(#{2,4})\s+[\d.]*\s*(.+)$", s)
        if m:
            level = len(m.group(1))
            perintah = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
            out.append(rf"\{perintah}{{{inline(m.group(2).strip())}}}")
            i += 1
            continue

        if s.startswith(r"\["):                     # matematika tampilan
            blok = []
            while i < n and r"\]" not in baris[i]:
                blok.append(baris[i])
                i += 1
            if i < n:
                blok.append(baris[i])
                i += 1
            isi = "\n".join(blok).replace(r"\[", "").replace(r"\]", "").strip()
            if "=" in isi or r"\sim" in isi or r"\arg\min" in isi:
                n_eq += 1
                out += [r"\begin{equation}", isi, r"\end{equation}"]
            else:
                out += [r"\[", isi, r"\]"]
            continue

        if s.startswith("|"):                       # tabel
            tab = []
            while i < n and baris[i].strip().startswith("|"):
                tab.append(baris[i].strip())
                i += 1
            out += tabel(tab)
            continue

        if re.match(r"^\d+\.\s", s):                # daftar bernomor
            item = []
            while i < n and (re.match(r"^\d+\.\s", baris[i].strip())
                             or (baris[i].startswith("   ") and baris[i].strip())):
                item.append(re.sub(r"^\d+\.\s*", "", baris[i].strip()))
                i += 1
            out.append(r"\begin{enumerate}")
            out += [rf"    \item {inline(x.rstrip(';').rstrip('.'))}" for x in item]
            out.append(r"\end{enumerate}")
            continue

        if s.startswith("- "):                      # daftar butir
            item = []
            while i < n and baris[i].strip().startswith("- "):
                item.append(baris[i].strip()[2:])
                i += 1
            out.append(r"\begin{itemize}")
            out += [rf"    \item {inline(x.rstrip(';').rstrip('.'))}" for x in item]
            out.append(r"\end{itemize}")
            continue

        if s.startswith("> "):                      # kutipan hipotesis
            kut = []
            while i < n and baris[i].strip().startswith("> "):
                kut.append(baris[i].strip()[2:])
                i += 1
            out += [r"\begin{quote}", inline(" ".join(kut)), r"\end{quote}"]
            continue

        out.append(inline(s))
        i += 1

    hasil = "\n".join(out)
    hasil = re.sub(r"\n{3,}", "\n\n", hasil)
    return hasil, n_eq


def tabel(tab: list[str]) -> list[str]:
    baris = [[c.strip() for c in r.strip("|").split("|")] for r in tab]
    kepala, isi = baris[0], baris[2:]
    ncol = len(kepala)
    align = "l" * ncol
    out = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{\textit{[lengkapi keterangan tabel]}}",
           rf"\begin{{tabular}}{{{align}}}", r"\hline",
           " & ".join(inline(c) for c in kepala) + r" \\", r"\hline"]
    for r in isi:
        r = (r + [""] * ncol)[:ncol]
        out.append(" & ".join(inline(c) for c in r) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--judul", required=True)
    a = ap.parse_args()
    isi, n_eq = konversi(Path(a.src).read_text())
    Path(a.dst).write_text(rf"\chapter{{{a.judul}}}" + "\n\n" + isi + "\n")
    print(f"[tulis] {a.dst}  ({len(isi.splitlines())} baris, "
          f"{n_eq} persamaan bernomor)")


if __name__ == "__main__":
    main()
