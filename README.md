# Skripsi — Evaluasi Formulasi Langkah Laten pada Komunikasi Sistem Multi-Agen

Harun Yahya · 22/505239/PA/21749 · Program Studi Statistika, Universitas Gadjah Mada · 2026

Repositori ini memuat **naskah, kode, dan angka** dalam satu riwayat versi.
Satu `git clone` memberi seluruhnya.

---

## Pertanyaan penelitian, dalam satu paragraf

LatentMAS membuat agen-agen model bahasa berkomunikasi lewat ruang laten alih-alih
teks, dan menjembatani ruang keluaran ke ruang masukan dengan satu matriks hasil
regresi ridge. Penelitian ini menguji empat formulasi alternatif yang
memproyeksikan langkah laten kembali ke dalam *convex hull* embedding token nyata,
pada dua kategori tugas sekaligus: penalaran umum (GSM8K, ARC-Challenge,
HumanEval+) dan pembangkitan faktor simbolik atas data Bursa Efek Indonesia
(LQ45, 2021–2025).

**Temuan utamanya berupa disosiasi.** Proyeksi ke *convex hull* memperbaiki
**validitas** keluaran secara sangat besar — 97,5% berbanding 50,0% jalan
menghasilkan ekspresi yang sah, $p = 5{,}7\times10^{-7}$ — tetapi **tidak
memperbaiki mutu** sinyal keluaran yang berhasil dihasilkan ($p = 0{,}21$).
Yang diperbaiki adalah kemampuan kanal laten membawa *simbol*, bukan
kemampuannya membawa *gagasan*.

---

## Isi repositori

```
skripsi/
  skripsi_v_claude.tex     naskah lengkap, satu berkas mandiri  ← BACA INI
  skripsi_ta1.tex          naskah TA1 (versi sebelumnya)
  skripsi_ta2.tex          naskah TA2 (versi sebelumnya, memanggil bab/*.tex)
  bab/                     bab terpisah untuk naskah TA1/TA2
  assets/
    tables/                tabel LaTeX (idx_*.tex dibangkitkan skrip)
    images/                figur (i0*.png dari data IDX)
  pemikiran/
    fundamental_matematika.md   turunan matematis lengkap + padanan istilah
    draf/                  draf subbab yang belum masuk naskah
  komputation/             SALINAN kode & hasil eksperimen  ← lihat README-nya
arsip/                     bahan tahap awal, tidak dipakai naskah
```

`quantalatent/` (bila ada di mesin ini) adalah repositori pengembangan yang
terpisah dan **tidak** di-track di sini; salinan yang angkanya dikutip naskah
ada di `skripsi/komputation/`.

---

## Membangun naskah

```bash
cd skripsi
latexmk -pdf skripsi_v_claude.tex
```

## Membangun ulang angkanya

Tanpa GPU. Lihat `skripsi/komputation/README_KOMPUTASI.md`.

---

## Catatan versi

Riwayat git sebelum 2026-08-29 dihapus sepenuhnya, tanpa cadangan, atas
permintaan penulis. Repositori ini dimulai dari commit pertama pada tanggal
tersebut. Data pasar juga berpindah pada tanggal yang sama: seluruh penilaian
faktor kini memakai Bursa Efek Indonesia (LQ45), menggantikan panel A-share
yang dipakai versi sebelumnya.
