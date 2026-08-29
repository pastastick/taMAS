# Panduan sesi cloud — apa yang harus dikerjakan dan bagaimana

Ditulis 2026-08-28 untuk sesi yang **tidak punya riwayat percakapan sebelumnya**.
Baca §1–§2 dulu; §3 adalah daftar kerja; §4 berisi jebakan yang sudah pernah
memakan waktu berjam-jam.

Sumber angka terkini: [`docs/HASIL_TAHAP5.md`](../docs/HASIL_TAHAP5.md)
(sudah diverifikasi ulang terhadap arsip run — angkanya akurat).

---

## 0. Status per 2026-08-28 (sesi pod A40)

Matriks faktor **lengkap: 14 sel × 20 jalan, semuanya sudah diskor IC.**
Yang tersisa dari §3 hanya **§3.3 (holdout)** — sengaja tidak dijalankan di pod
karena murni CPU dan diperkirakan 8–16 jam (ia menskor seluruh korpus di jendela
2022–2025, 4× data jendela seleksi; jendela seleksi sendiri makan 4,2 jam pada
16 pekerja). Menyewa GPU untuk itu membayar tarif kartu demi kerja tanpa kartu.

| tugas | status |
|---|---|
| §3.1 `kv_and_text` ×3 → 20 jalan | **selesai** — digabung, 20/20 lolos gate, 0 error |
| §3.2 skoring IC 8 sel | **selesai** — 3 batch, verifikasi replika CPU nol selisih |
| §3.3 holdout 2022–2025 | **BELUM** — jalankan di mesin CPU, lihat §3.3 |
| §3.4 sumbu `mix` | **selesai** — 6 sel, 105 mnt |
| §3.5 finalisasi | **selesai sebagian** — analisis + 24/24 figur regenerasi dengan `LEWATI_HOLDOUT=1`; **ulangi penuh setelah §3.3** supaya v14 & Level 6 ikut baru |

Hasil sumbu `mix` (§3.4), kedua lengan sepakat dan bentuknya **ber-ambang**:

| α | 0 (`raw`) | 0,25 | 0,5 | 0,75 | 1 (`soft`) |
|---|---:|---:|---:|---:|---:|
| cos ke embedding | 0,3120 | 0,4519 | 0,7197 | 0,9244 | 0,9269 |
| lolos gate | 10/20 | 11/20 | 20/20 | 20/20 | 20/20 |
| HumanEval+ | — | 0,28 | 0,68 | 0,69 | — |

Loncatannya di ruas 0,25→0,5 — ruas tercuram kurva geometri. α=0,75, titik
falsifikasi terkuat yang disebut §3.4, berkinerja seperti `soft`.

⚠️ **Volume `/workspace` ini FUSE dan tidak tepercaya.** Dua kali menggigit
dalam satu sesi: (a) `import torch` gagal dengan berkas `.so` "hilang" yang
muncul lagi setelah beberapa kali coba — retry dulu sebelum memasang ulang apa
pun; (b) `backend/eval/rescore_all.py`, `backend/factor/run_factor.py`, dan
`scripts/kemas_hasil.sh` **mundur ke isi pra-commit di working tree** padahal
HEAD benar — ketahuan karena `--checkpoint-detik` tiba-tiba "unrecognized".
Jalankan `git status` sebelum percaya bahwa skrip yang dipanggil adalah skrip
yang di-commit.

---

## 1. Konteks 60 detik

Skripsi ini membandingkan **lima formulasi langkah laten** dalam sistem
multi-agen LLM (Qwen3-8B):

| | |
|---|---|
| `raw` | jalur produksi LatentMAS: hidden state → realigner ridge. **Kontrol.** |
| `soft`, `sample`, `gumbel`, `moi` | keluarga **R** — semuanya memproyeksikan lewat distribusi token dulu, jadi hasilnya selalu di dalam lambung konveks embedding nyata |

Dijalankan di **dua lengan setara** (bukan satu lengan utama + satu tambahan):

- **lengan bench** — GSM8K / ARC-C / HumanEval+
- **lengan faktor** — agen menulis ekspresi faktor alpha dalam DSL, lalu dinilai
  RankIC terhadap data pasar. Rantai `proposal → innovate → construct`.

Dan **tiga medium komunikasi**: `kv` (KV-cache saja), `text`, `kv_and_text`.

**Temuan inti yang sudah berdiri:** keluarga R jauh mengungguli `raw` dalam laju
lolos gate ekspresi — di medium `kv` 97,5% (78/80) vs 50% (10/20), Fisher eksak
*p* = 5,7 × 10⁻⁷, Cohen's *h* = 1,25; keempat anggota signifikan sendiri-sendiri.
Penghematan token 78–80% datang dari **medium**, bukan dari formulasi.

---

## 2. Setup

```bash
cd <akar repo>
bash scripts/setup_pod.sh
```

Ia memeriksa venv + dependensi, VRAM, `backend/hf_data/daily_pv.h5`, cache data
pasar, dan mencetak **status tiap sel** (berapa jalan, sudah diskor atau belum).
Baca keluarannya sebelum menjalankan apa pun — ia sekaligus memberitahu apa yang
tersisa.

Semua skrip menurunkan akar repo dari lokasinya sendiri, jadi tak ada jalur yang
dipatok. Semua memakai `.venv/bin/python` kalau ada.

**`results/` TIDAK di-track git** (181 MB, artefak run). Kalau pod baru,
pastikan `results/` sudah di-scp masuk — tanpa itu tak ada yang bisa dianalisis.

---

## 3. Daftar kerja, urut prioritas

### 3.1 Naikkan 3 sel `kv_and_text` dari 8 → 20 jalan · ~45 mnt GPU

**Kenapa ini nomor satu.** `kv_and_text_sample`, `kv_and_text_gumbel`, dan
`kv_and_text_moi` berhenti di 8 jalan (seed 0–1 saja) karena orkestrator malam
itu di-hardcode `target 8`. Pada n=8 dengan efek sempurna (b01=5, b10=0), *p*
terkecil yang **mungkin** dari McNemar eksak adalah 2 × 0,5⁵ = **0,0625** > 0,05.
Ketiga sel itu **tidak bisa signifikan berapa pun bagusnya hasilnya** — itu batas
resolusi rancangan, bukan sifat data. Selain itu *p* famili medium `kv_and_text`
kini berdiri di atas n timpang (20/8/8/8), sehingga bobot `soft` 2,5× tiga
lainnya.

```bash
bash scripts/lanjutkan_kv_and_text.sh --dry-run   # lihat perintahnya
bash scripts/lanjutkan_kv_and_text.sh             # jalankan
bash scripts/lanjutkan_kv_and_text.sh --gabung    # setelah diperiksa
```

⚠️ **Jangan** menjalankan ulang dengan `--seeds 0,1,2,3,4` ke tag asli.
`run_factor.py` tidak punya resume — tag yang sama **ditimpa**. Itu membuang 24
jalan GPU yang sudah jadi dan membangkitkan ulang seed 0–1 dengan
`--temperature 0.8` + batching vLLM yang tak menjamin reproduksi bit-per-bit,
sehingga angka yang sudah terdokumentasi bisa bergeser tanpa alasan ilmiah.
Skrip di atas menjalankan **hanya seed 2,3,4** ke tag sementara, lalu
[`scripts/gabung_jalan.py`](gabung_jalan.py) menggabungkannya.

`gabung_jalan.py` menolak menggabung kalau 15 argumen penentu perilaku berbeda,
menolak duplikat `(arah, seed)`, mengurutkan ulang seperti `run_factor.py`,
mencadangkan berkas lama, dan menanam sha256 tiap pecahan sebagai jejak asal.

### 3.2 Skoring IC 8 sel yang belum diskor · CPU, berjam-jam

Jalankan ini **bersamaan** dengan pekerjaan GPU — skoring tak butuh kartu.

```bash
bash scripts/skor_cpu.sh              # latar belakang
bash scripts/skor_cpu.sh --status     # progres
bash scripts/skor_cpu.sh --stop       # berhenti RAPI, progres tersimpan
```

Default-nya **hanya menyentuh sel yang belum punya `icseries_<tag>.parquet`**.
Itu disengaja: jalur ini menimpa field `ic`, dan menimpa sel yang angkanya sudah
dikutip dokumen (`kv_soft`, `kv_sample`, `kv_raw`) berarti mempertaruhkan dasar
Bab IV tanpa alasan.

Sifat yang membuatnya aman ditumpangkan pada umur pod:

- **paralel** — fork copy-on-write, N pekerja berbagi SATU salinan data pasar
  (terukur 2,4× di 3 pekerja)
- **ber-checkpoint** — cache disimpan tiap 180 detik; proses yang dibunuh
  kehilangan paling banyak 3 menit kerja, bukan berjam-jam
- **berhenti anggun** — `--stop` (SIGTERM) menyimpan progres lalu menulis tag
  yang sudah lengkap; tag yang belum lengkap dilewati, bukan diskor serial
- **bisa dilanjutkan** — jalankan perintah yang sama, ia membaca checkpoint

⚠️ `--stop`, **bukan** `kill -9`. SIGKILL membuang semua yang belum ter-checkpoint.

Sebelum 2026-08-28 jalur ini serial dan menulis atomik di akhir; itulah sebabnya
`text` (3 j 16 m) dan `kv_gumbel` (2 j 27 m) hilang total saat pod dimatikan.

### 3.3 Ulang `skor_holdout.py` di atas matriks 20-jalan

Angka Level 6 / figur v14 yang ada sekarang berasal dari **arsip run 6-jalan**
(`results/arsip_faktor_6jalan_2026-08-10/`), belum sebanding dengan §2. Ini ikut
dijalankan oleh `finalisasi.sh`, atau sendiri:

```bash
PYTHONPATH=backend .venv/bin/python backend/eval/skor_holdout.py \
    --budget 900 --workers 16
```

⚠️ **Jalankan di mesin CPU, jangan di pod GPU.** Ia menskor seluruh korpus
(~1000 ekspresi unik) di jendela 2022–2025 — 4× data jendela seleksi, yang
sendirian sudah makan 4,2 jam pada 16 pekerja. Perkiraan 8–16 jam tanpa
menyentuh kartu sama sekali. Default `--workers` adalah `min(4, ncpu)`; naikkan
kalau mesinnya besar. Cache-nya `results/.cache/holdout_cache_<jendela>.json`
dan ditulis tiap ekspresi, jadi proses yang mati bisa dilanjutkan.

Setelah ini selesai, ulang `finalisasi.sh` **penuh** (tanpa `LEWATI_HOLDOUT=1`)
supaya v14 dan Level 6 ikut terbarui.

### 3.4 Sumbu interpolasi `mix` · ~1,5–2 jam GPU · 6 sel

```bash
bash scripts/jalankan_interpolasi.sh --dry-run
bash scripts/jalankan_interpolasi.sh
```

`z(α) = normalisasi((1−α)·z_raw + α·z_soft)`. Ini **alat ukur, bukan usulan
metode**: lima formulasi memberi lima titik terpisah sehingga hubungan
geometri↔kinerja hanya bisa dibaca "searah"; `mix` mengisi jaraknya supaya
**bentuk**-nya yang diuji. Titik ujung tak dijalankan — α=0 identik bit-per-bit
dengan `raw`, α=1 dengan `soft`.

⚠️ **Hipotesisnya sengaja tak berarah.** Monoton / ber-ambang / tak berpola
ketiganya temuan; yang ketiga berarti klaim mekanistik Bab IV harus dilemahkan.
Jangan menulis "makin dekat embedding makin baik" sebelum datanya ada.

⚠️ **Sumbu-x sudah diukur dan tidak linier** — ia **jenuh**:

| α | 0 | 0,25 | 0,5 | 0,75 | 1 |
|---|---:|---:|---:|---:|---:|
| cos ke embedding terdekat | 0,3120 | 0,4519 | 0,7197 | 0,9244 | 0,9269 |

Paling curam di 0,25→0,5 (+0,268), praktis **datar** di 0,75→1 (+0,0025).
Plot kinerja terhadap **cos terukur**, bukan terhadap α. Dan α=0,75 yang secara
geometri sudah ≈`soft` justru titik falsifikasi terkuat: kalau geometri memang
penjelasnya, ia harus berkinerja seperti `soft`.

### 3.5 Regenerasi analisis + 24 figur

```bash
bash scripts/finalisasi.sh            # analisis + figur
bash scripts/finalisasi.sh --kemas    # + arsip tar.gz
```

**Jangan dilewati.** Berkas di `results/pendukung/` adalah turunan, dan turunan
basi tidak memberi tanda apa pun: 2026-08-27 delapan berkas di sana masih
memerikan korpus **174 ekspresi** padahal matriksnya sudah **910** — angka salah
tanpa peringatan, siap masuk naskah.

---

## 4. Jebakan yang sudah pernah memakan waktu

| jebakan | akibat | cara menghindar |
|---|---|---|
| `--budget 90` saat skoring | ekspresi `TS_SKEW`/`TS_KURT`/`TS_MAD`/`REGRESI` ada di bibir 90 dtk → ekspresi yang **sama** bisa ber-IC atau `ic=None` tergantung beban mesin | selalu `--budget 900` |
| 3 sel faktor serentak | OOM — sel faktor ~21 GB (`max_new_tokens 4096` × 3 agen), A40 46 GB | `--slots 2`; skrip sudah memasang gerbang VRAM 24 GB |
| `kill -9` pada skoring | seluruh progres yang belum ter-checkpoint hilang | `bash scripts/skor_cpu.sh --stop` |
| menyatukan run 6-jalan dengan 20-jalan | tidak sah — sejak 2026-08-27 `construct` memakai `prefill:`, prosedurnya beda | arsip 6-jalan sudah dipisah; `gabung_jalan.py` menolak pecahan yang argumennya beda |
| `"lengkap": true` di `uji_formal_*.json` | **menyesatkan** — ia hanya mengecek keempat anggota hadir, bukan n-nya seimbang | periksa `r_n` per anggota sendiri |
| menjalankan `visual_bab4.py` lebih dulu | figur terbangkit dari angka lama tanpa keluhan | pakai `finalisasi.sh` yang urutannya benar |
| menghapus pod sebelum scp | `results/` tak di-track git — arsipnya **satu-satunya salinan** | `bash scripts/kemas_hasil.sh semua` lalu scp keluar |
| kunci `--score-only` tak berlaku | dua proses bisa menyekor tag yang sama bersamaan (terjadi 2026-08-27 pada `kv_moi`) | jalankan skoring hanya lewat `skor_cpu.sh` |

Peringatan `UserWarning: Loky-backed parallel loops cannot be called in a
multiprocessing` **normal** — itu justru tanda pengaman `LOKY_MAX_CPU_COUNT=1`
bekerja. Peringatan LAPACK `DLASCL parameter had an illegal value` di log
`kv_gumbel` **belum diperiksa**; catat kalau muncul lagi.

---

## 5. Cara memeriksa hasil sehat

```bash
bash scripts/setup_pod.sh          # status tiap sel: jalan & sudah diskor
bash scripts/skor_cpu.sh --status  # progres skoring
tail -f results/logs/<sel>.log     # satu sel GPU
```

Sel faktor yang sehat: 20 jalan, `error` kosong, `passing` tak kosong pada
sebagian besar jalan, `construct_text_head` mulai dengan `{\n  "hypothesis": `
(itu prefill — kalau mulai di tengah JSON, prefill tidak aktif).

---

## 6. Sebelum pod dihapus

```bash
bash scripts/finalisasi.sh --kemas
# lalu dari mesin lokal:
scp -P <PORT> root@<HOST>:/workspace/hasil_*.tar.gz .
```

Repo boleh di-push, tapi `results/` tidak ikut. Push **harus lewat URL SSH** —
`origin` memakai HTTPS dan pod tak punya kredensial HTTPS:

```bash
git push git@github.com:pastastick/multi-agent-system.git main
```

---

## 7. Berkas rujukan

| berkas | isi |
|---|---|
| [`docs/HASIL_TAHAP5.md`](../docs/HASIL_TAHAP5.md) | angka terkini: matriks 20 jalan, uji formal, efisiensi, holdout |
| [`docs/TEORI.md`](../docs/TEORI.md) §4.6 | kebijakan uji + tabel daya (n=6 daya nol; n=20 hanya kontras R-vs-`raw` yang berdaya) |
| [`docs/PANDUAN.md`](../docs/PANDUAN.md) | panduan repo umum |
| [`configs/matriks.yaml`](../configs/matriks.yaml) | sumber kebenaran tunggal matriks eksperimen |
| `results/arsip_faktor_6jalan_2026-08-10/README.md` | kenapa run lama diarsipkan, bukan ditimpa |
| `results/arsip_pecahan_gabung_2026-08-28/README.md` | pecahan `_s234` + cadangan penggabungan §3.1; kenapa ia HARUS di luar `results/factor/` |
