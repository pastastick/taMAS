# Naskah Tugas Akhir — Statistika UGM

Dua dokumen dibangun dari satu himpunan bab, memakai template resmi
Statistika UGM (`laporanta1statugm.cls`).

| Driver | Isi | Halaman |
|---|---|---|
| `skripsi_ta1.tex` | TA 1 — Bab 1–3 + Bab 4 metode & hasil sementara | 88 |
| `skripsi_ta2.tex` | TA 2 — Bab 1–5 lengkap + Lampiran A–F | 114 |

Perbedaannya hanya pada bab hasil: TA 1 memakai `bab/Bab4_TA1.tex`,
TA 2 memakai `bab/Bab4_TA2.tex` plus `bab/Bab5.tex` dan enam lampiran.

## Kompilasi

```bash
cd skripsi/
latexmk -pdf skripsi_ta2.tex        # keluaran: skripsi_ta2.pdf
latexmk -pdf skripsi_ta1.tex        # keluaran: skripsi_ta1.pdf
latexmk -C                          # bersihkan artefak build
```

Butuh TeX Live: `texlive-latex-extra`, `texlive-fonts-extra`,
`texlive-lang-other`, `latexmk`. `logougm.png` dan `laporanta1statugm.cls`
harus berada satu folder dengan driver.

## Struktur

```
skripsi/
├── skripsi_ta1.tex        driver TA 1 — metadata (judul, nama, NIM) di sini
├── skripsi_ta2.tex        driver TA 2 — idem
├── laporanta1statugm.cls  class resmi UGM (jangan diubah)
├── logougm.png            logo cover
├── bab/
│   ├── Bab1.tex           PENDAHULUAN
│   ├── Bab2.tex           LANDASAN TEORI
│   ├── Bab3.tex           METODOLOGI PENELITIAN
│   ├── Bab4_TA1.tex       hasil sementara (TA 1)
│   ├── Bab4_TA2.tex       HASIL DAN PEMBAHASAN (TA 2)
│   ├── Bab5.tex           KESIMPULAN DAN SARAN (TA 2)
│   ├── lampiran_a..f.tex  sel eksperimen, uji, transkrip, kartu faktor,
│   │                      kode, hiperparameter
│   └── DaftarPustaka.tex  manual, natbib + \href
├── assets/
│   ├── tables/            8 tabel LaTeX — DIBANGKITKAN SKRIP, jangan disunting
│   ├── images/            gambar hasil (PNG 300 dpi) — juga dibangkitkan skrip
│   ├── briefing/          draf awal Bab 2 & Bab 3 + paper arXiv rujukan
│   └── another_skripsi/   rujukan format (gitignored, 32 MB)
├── pemikiran/             catatan & perencanaan (gitignored)
└── template senior/       template asli kakak tingkat (rujukan)
```

## Angka dan gambar dibangkitkan, bukan diketik

Seluruh isi `assets/tables/` dan `assets/images/` dihasilkan skrip di
`../analisis/`. Menyuntingnya langsung akan hilang saat regenerasi — ubah
skripnya.

| Skrip | Keluaran |
|---|---|
| `analisis/03_gambar.py` | `assets/images/*.png` (300 dpi) |
| `analisis/04_tabel_tex.py` | `hasil_bench` · `uji_signifikan` · `efisiensi` · `geometri` |
| `analisis/05_lampiran.py` | `bab/lampiran_a,b,c,f.tex` |
| `analisis/07_faktor_pelengkap.py` | `faktor_parsing` · `faktor_korpus` · `faktor_kartu` |
| `analisis/08_lampiran_kartu.py` | `bab/lampiran_d_kartu.tex` |

Empat tabel di Bab 4 (Cochran $Q$, kontras, disosiasi, korupsi token) masih
**diketik tangan** di dalam `bab/Bab4_TA2.tex`, padahal angkanya tersedia di
`analisis/bench_ringkas.json`. Kalau angka diperbarui, keempatnya harus
disesuaikan manual.

## Konvensi penulisan (dari template UGM)

- Sitasi `natbib`: `\citep{key}`, `\citeauthor{key}`, `\citeyear{key}`.
- Daftar pustaka manual di `bab/DaftarPustaka.tex` dengan `\bibitem`, urut
  abjad, wajib menyertakan `\href` ke sumber unduhan.
- Jangan menyitir skripsi S1.
- Pemanggilan objek pakai huruf kapital: "Persamaan 2.1", "Tabel 3.1",
  "Gambar 4.1".
- Tiap akhir persamaan diberi tanda titik.
- Penekanan istilah memakai `\textit`, bukan `\textbf`, di badan teks.

## Perintah bantu (didefinisikan di kedua driver)

| Perintah | Fungsi |
|---|---|
| `\code{...}` | kode inline |
| `\important{...}` | penekanan |
| `\todo{...}` | penanda merah bagian yang belum ditulis |
| `\draft` | penanda [DRAFT] |
| `\needref{...}` | formula hasil rekonstruksi yang butuh rujukan |

Sebelum final, pastikan tak ada penanda tersisa:

```bash
grep -rn '\\todo{\|\\draft\|\\needref{' bab/ skripsi_ta1.tex skripsi_ta2.tex
```

## Yang masih terbuka

- Judul kedua driver masih memuat literal `JUDUL SEMENTARA:`.
- TA 2 masih memakai `laporanta1statugm.cls`, yaitu class TA 1; class TA 2
  yang sesuai belum tersedia.
