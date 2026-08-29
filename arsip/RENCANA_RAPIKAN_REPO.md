# Rencana Merapikan Repositori — usulan, menunggu keputusan

Ditulis 2026-08-28. **Belum dieksekusi.** Setiap langkah destruktif diberi tanda
⚠️ dan tidak akan dijalankan tanpa persetujuan eksplisit.

---

## 1. Keadaan sekarang — enam masalah nyata

### M1. `quantalatent/` adalah *submodule rusak* ⚠️ paling parah
Repo luar melacak `quantalatent` sebagai **gitlink** (mode `160000`, commit
`287a24c`) tetapi **tidak ada berkas `.gitmodules`**. Akibatnya:

- `git clone` repo luar menghasilkan direktori `quantalatent/` **kosong**;
- commit yang dilacak (`287a24c`) tertinggal jauh di belakang isi sebenarnya
  (`71c35af`) — itulah sebab `git status` selalu menunjukkan `M quantalatent`;
- tidak ada satu commit pun yang merekam "skripsi versi X dibangun dari kode
  versi Y". Untuk skripsi, itu masalah reprodusibilitas, bukan sekadar kerapian.

### M2. `analisis/` tidak dilacak git sama sekali
Sepuluh skrip yang membangkitkan **setiap tabel dan gambar Bab IV** hanya ada di
disk. Kalau mesin ini hilang, tabel-tabel itu tidak bisa dibangun ulang. Ini
juga penjelasan langsung kenapa tabel hasil sulit ditemukan saat ditanya
pembimbing: berkasnya memang tidak ada di riwayat mana pun.

### M3. Riwayat repo dalam 575 MB, isinya sampah lama
`quantalatent/.git` = 602 MB. Dua belas objek terbesar semuanya
`backend/runs/*/factor_logs/*factor_values.csv(.gz)` berukuran 33–53 MB —
buangan antara dari run Juni–Agustus 2026 yang **sudah tidak ada di working
tree** dan tidak dipakai satu angka pun di skripsi.

### M4. Lima belas branch, sebagian besar mati
`exp/*`, `prod/*`, `experiment/*`, `newEvol`, `backup/*`, `feat/*` — tidak ada
penanda mana yang masih berlaku selain ingatan.

### M5. Nama branch tak konsisten antar repo
Repo luar: `master` + `feat/prompt-redesign-v2`. Repo dalam: `main` + 14 lainnya.

### M6. Berkas liar di akar
`penjelasan proposal.py` (99 KB, bukan kode yang dijalankan), `original-prompt/`
(21 MB), `transfer-memory/` (catatan sesi lama).

---

## 2. Usulan struktur akhir

```
skripsi-quantalatent/                 ← SATU repo, satu riwayat
├── README.md                         ← peta: apa di mana, cara reproduksi
├── skripsi/                          ← naskah LaTeX + aset
│   ├── bab/ assets/ pemikiran/
│   └── skripsi_ta2.tex
├── analisis/                         ← 10 skrip pembangkit tabel & gambar  ← MASUK GIT
├── kode/                             ← isi `quantalatent/` sekarang
│   ├── backend/ scripts/ configs/ docs/
│   └── data/                         ← panel pasar (IDX kecil; A-share via unduhan)
├── hasil/                            ← results/ yang DIPAKAI skripsi saja
│   ├── factor/ bench/ probe/ visual/
└── arsip/                            ← tarball + catatan, tidak dipakai langsung
```

Satu `git clone` memberi **semuanya**: naskah, kode, angka, dan skrip yang
menghubungkan ketiganya.

---

## 3. Langkah, berurutan

| # | langkah | risiko |
|---|---|---|
| L1 | **Cadangkan dulu**: `git bundle create arsip/quantalatent-full-20260828.bundle --all` di repo dalam. Satu berkas berisi 15 branch + seluruh riwayat, bisa di-`clone` kembali kapan saja. | nol |
| L2 | Buat branch arsip di repo dalam: `arsip/agustus-2026` dari `main` sekarang, push ke origin. | nol |
| L3 | Kemas `results/` yang belum ter-tarball → `arsip/` | nol |
| L4 | Buat repo baru `skripsi-quantalatent` dengan struktur §2, **impor working tree** (bukan riwayat 575 MB) sebagai satu commit awal | nol (repo lama utuh) |
| L5 | Salin riwayat naskah skripsi dari repo luar (5 commit) ke repo baru — riwayatnya kecil dan berguna | nol |
| L6 | Tulis `README.md` + `REPRODUKSI.md`: satu perintah per angka di Bab IV | nol |
| L7 | ⚠️ Ganti `main` repo lama dengan pointer ke repo baru (atau arsipkan repo lama sepenuhnya) | **butuh keputusan** |
| L8 | ⚠️ Hapus branch mati di repo lama sesudah bundle terverifikasi | **butuh keputusan** |

**Aturan yang kupakai:** tidak ada `git push --force`, tidak ada penghapusan
branch, dan tidak ada `rm -rf` sebelum L1 selesai dan bundle-nya diuji dengan
`git clone` percobaan.

---

## 4. Keputusan yang sudah diambil (2026-08-28)

| # | keputusan | catatan |
|---|---|---|
| K1 | **SATU repo, dan itu berarti repositori BARU** | bukan `pastastick/multi-agent-system`. Repo lama dibiarkan utuh sebagai arsip; tidak ada `push --force` ke sana, tidak ada branch dihapus di sana. |
| K2 | **Riwayat 575 MB dibuang dari repo aktif** | disimpan utuh sebagai `git bundle` di `arsip/`; bisa di-`clone` kembali kapan saja |
| K3 | **Dieksekusi SETELAH angka IDX masuk naskah** | supaya tidak ada perpindahan direktori di tengah pekerjaan analisis |

Masih terbuka:

- **Nama repo baru** — usul: `skripsi-quantalatent`
- **Bahasa nama direktori** — Indonesia (`kode/`, `hasil/`, `arsip/`) atau
  Inggris (`src/`, `results/`, `archive/`)
- **Repo lama di GitHub** — di-*archive* lewat setelan GitHub (read-only,
  tetap bisa dibuka) atau dibiarkan aktif

---

## 5. ⚠️ TEMUAN MENDESAK — naskah skripsi tidak punya cadangan mana pun

Repo LUAR (`/root/projects/first-experiment`, yang memuat `skripsi/`)
**tidak punya remote sama sekali**:

```
$ git remote -v
(kosong)
```

Artinya seluruh naskah — `skripsi/bab/*.tex`, aset, tabel, gambar — hanya ada di
disk mesin ini. Tidak ada di GitHub, tidak ada di mana pun. Ditambah
`analisis/` yang bahkan tidak dilacak git, satu kegagalan disk menghapus
naskah **dan** alat pembangkit tabelnya sekaligus.

Ini tidak perlu menunggu K3. Yang paling cepat dan tidak mengganggu apa pun:

1. `git add analisis/` lalu commit di repo luar (menit, nol risiko);
2. buat repo privat baru di GitHub, `git remote add origin`, `git push`.

Sesudah itu barulah perapian besar bisa ditunggu dengan tenang.
