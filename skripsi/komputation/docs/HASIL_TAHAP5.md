# Hasil Tahap 5 — matriks faktor 20-jalan + pipeline visual Bab IV

> Status per **2026-08-27 15:30 UTC**, saat seluruh proses GPU/CPU dihentikan
> dan `results/` dikeluarkan dari git (lihat §6). Dokumen ini merekam apa yang
> **sudah jadi angka**, apa yang **belum**, dan apa yang harus dijalankan
> berikutnya. Lanjutan dari [`HASIL_TAHAP4.md`](HASIL_TAHAP4.md).

---

## Ringkasan satu paragraf

Lengan faktor dinaikkan dari 6 jalan (arsip 2026-08-10) ke **20 jalan per sel**
(5 seed × 4 arah) untuk medium `kv`, dan hasilnya menutup pertanyaan utama
lengan ini: keluarga formulasi terelaksasi **R = {soft, sample, gumbel, moi}**
lolos gate ekspresi **97,5% (78/80)** melawan **50% (10/20)** untuk `raw` —
Fisher eksak *p* = 5,7 × 10⁻⁷, Cohen's *h* = 1,25. Efeknya konsisten di kedua
medium dan bertambah besar di `kv_and_text` (100% vs 30%, *p* = 8,1 × 10⁻¹⁰).
Jalur KV murni sekaligus **menghemat ~79% token keluar dan ~3,2× lebih cepat**
dibanding baseline teks penuh, tanpa kehilangan laju lolos gate. Yang **belum**
selesai: tiga sel `kv_and_text` (`sample`, `gumbel`, `moi`) berhenti di 8 jalan,
bukan 20 — pada n = 8 uji McNemar eksak **tidak bisa** mencapai *p* < 0,05
sekalipun efeknya sempurna, jadi klaim per-anggota di medium itu belum berdiri
sendiri. Skoring IC juga baru tuntas di 3 dari 11 sel. Sumbu interpolasi
(`mix`, 6 sel) belum disentuh sama sekali.

---

## 1. Status matriks lengan faktor

Target per sel: 20 jalan = 5 seed (0–4) × 4 arah (`d0`, `d1`, `opp_mom`,
`opp_rev`), rantai `proposal → innovate → construct`, `--max-repair 3`.

| sel | jalan | lolos gate | tok keluar/jalan | detik/jalan | coba construct/jalan | skoring IC |
|---|---:|---:|---:|---:|---:|:--:|
| `kv_soft` | 20 | **100%** | 481 | 52 | 1,00 | ✅ |
| `kv_gumbel` | 20 | **100%** | 502 | 58 | 1,05 | ❌ |
| `kv_sample` | 20 | 95% | 517 | 49 | 1,05 | ✅ |
| `kv_moi` | 20 | 95% | 524 | 56 | 1,10 | ❌ |
| `kv_raw` | 20 | 50% | 803 | 77 | 1,90 | ✅ |
| `text` | 20 | 100% | 2 436 | 181 | 1,15 | ❌ |
| `kv_and_text_soft` | 20 | **100%** | 1 920 | 186 | 1,00 | ❌ |
| `kv_and_text_raw` | 20 | 30% | 3 690 | 296 | 1,70 | ❌ |
| `kv_and_text_sample` | **8** ⚠️ | 100% | 1 968 | 139 | 1,00 | ❌ |
| `kv_and_text_gumbel` | **8** ⚠️ | 100% | 1 956 | 154 | 1,00 | ❌ |
| `kv_and_text_moi` | **8** ⚠️ | 100% | 2 003 | 102 | 1,00 | ❌ |

Tiga sel bertanda ⚠️ hanya menjalankan seed 0–1. Penyebabnya bukan kegagalan
run: orkestrator konsolidasi yang dipakai malam itu memang di-hardcode
`target 8` dan punya penjaga *"kalau berkasnya sudah ada, lewati"*, sehingga
ketiganya tidak akan pernah naik ke 20 tanpa intervensi manual.
`run_factor.py` **tidak punya flag resume/append**, jadi menaikkannya berarti
menjalankan ulang sel penuh dengan `--seeds 0,1,2,3,4` (menimpa), atau
menjalankan `--seeds 2,3,4` ke tag sementara lalu menggabungkan JSON-nya.

### 1.1. Kenapa n = 8 tidak cukup, secara aritmetika

Uji per-anggota memakai **McNemar eksak berpasangan** (pasangan = arah × seed).
Pada n = 8 dengan efek sempurna (b01 = 5, b10 = 0), *p* terkecil yang mungkin
adalah 2 × 0,5⁵ = **0,0625** — di atas α = 0,05. Artinya tiga sel itu
**tidak bisa signifikan sendiri berapa pun bagusnya hasilnya**; itu batas
resolusi desain, bukan sifat datanya. Itulah alasan target 20 dipilih sejak
awal (lihat tabel daya [`TEORI.md`](TEORI.md) §4.6, yang diverifikasi ulang
cocok ±0,005 di setiap sel pada putaran ini).

---

## 2. Uji formal — keluarga R vs `raw`

Kebijakan uji mengikuti [`TEORI.md`](TEORI.md) §4.6: klaim utama diuji di
tingkat **keluarga** R = {soft, sample, gumbel, moi} lawan `raw`, dengan uji
per-anggota dilaporkan sebagai diagnostik, bukan sebagai klaim terpisah.

### 2.1. `comm_mode = kv` — LENGKAP, n = 80 vs 20

| | lolos gate | n |
|---|---:|---:|
| keluarga R | **97,5%** | 80 |
| `raw` | 50,0% | 20 |

- selisih **+47,5 pp**, CI95 [0,25 – 0,70]
- **Fisher eksak dua-sisi *p* = 5,69 × 10⁻⁷**
- **Cohen's *h* = 1,25** (efek besar)

Diagnostik per-anggota (McNemar eksak, semua n = 20):

| anggota | lolos | vs raw | *p* | delta | CI95 |
|---|---:|---:|---:|---:|---|
| soft | 100% | 50% | **0,0020** | +0,500 | [0,3 – 0,7] |
| gumbel | 100% | 50% | **0,0020** | +0,500 | [0,3 – 0,7] |
| sample | 95% | 50% | **0,0117** | +0,450 | [0,2 – 0,7] |
| moi | 95% | 50% | **0,0117** | +0,450 | [0,2 – 0,7] |

Keempat anggota signifikan sendiri-sendiri. Ini hasil paling kuat di repo.

### 2.2. `comm_mode = kv_and_text` — TIDAK SEIMBANG, n = 44 vs 20

| | lolos gate | n |
|---|---:|---:|
| keluarga R | **100%** | 44 (20 + 8 + 8 + 8) |
| `raw` | 30,0% | 20 |

- selisih **+70,0 pp**, CI95 [0,50 – 0,90]
- **Fisher eksak dua-sisi *p* = 8,10 × 10⁻¹⁰**, Cohen's *h* = 1,98

| anggota | n | lolos | vs raw | *p* | catatan |
|---|---:|---:|---:|---:|---|
| soft | 20 | 100% | 30% | **0,00012** | signifikan |
| sample | 8 | 100% | 38% | 0,0625 | ⚠️ batas resolusi n=8 |
| gumbel | 8 | 100% | 38% | 0,0625 | ⚠️ batas resolusi n=8 |
| moi | 8 | 100% | 38% | 0,0625 | ⚠️ batas resolusi n=8 |

**Peringatan pelaporan.** Berkas `uji_formal_faktor_kv_and_text.json` menandai
keluarga ini `"lengkap": true`. Tanda itu **menyesatkan** — ia hanya mengecek
keempat anggota hadir, bukan bahwa n-nya seimbang. Angka *p* famili di medium
ini sah secara aritmetika tetapi berdiri di atas 44 jalan yang timpang
(20/8/8/8), sehingga bobot `soft` jauh lebih besar dari tiga lainnya. Sebelum
dipakai di naskah, sel-selnya harus disamakan ke 20.

---

## 3. Efisiensi — KV murni vs baseline teks

Diukur terhadap sel `text` (2 436 token keluar/jalan, 181 detik/jalan):

| sel | hemat token | percepatan |
|---|---:|---:|
| `kv_soft` | **80,3%** | 3,52× |
| `kv_sample` | 78,8% | **3,72×** |
| `kv_moi` | 78,5% | 3,26× |
| `kv_gumbel` | 79,4% | 3,13× |
| `kv_raw` | 67,0% | 2,35× |
| `kv_and_text_soft` | 21,2% | 0,98× |
| `kv_and_text_raw` | **−51,5%** | 0,61× |

Dua bacaan. Pertama, **penghematan datang dari medium, bukan dari formulasi**:
seluruh sel `kv_*` berkumpul di 78–80%, sedangkan `kv_and_text_*` di 18–21%.
Kedua, `raw` adalah satu-satunya formulasi yang **merugi** — di `kv_and_text`
ia justru 51,5% lebih boros dari teks penuh dan 0,61× lebih lambat, karena
laju lolos gate-nya rendah memaksa siklus repair (1,70–1,90 percobaan construct
per jalan lawan 1,00 untuk keluarga R). Jadi biaya `raw` bukan cuma akurasi;
kegagalannya berbunga jadi token dan waktu.

---

## 4. Geometri vs kinerja hilir (deskriptif, n = 5 mode)

| mode | cos ke embedding terdekat |
|---|---:|
| raw | 0,3120 |
| soft | 0,9269 |
| gumbel | 0,9848 |
| sample | 1,0000 |
| moi | 1,0000 |

Korelasi dengan recall kanal laten murni (m=10, k=5, 20 trial):
**dsl** ρ = 0,975 (r = 0,998) · **token** ρ = 0,684 (r = 0,820).

Korelasi dengan akurasi bench (medium kv, limit 100):
**humanevalplus** ρ = 0,975 (r = 0,998) · **gsm8k** ρ = 0,564 · **arc_challenge** ρ = 0,433.

Polanya searah dengan disosiasi §1 README: hubungan geometri→kinerja kuat pada
tugas yang menuntut presisi simbolik dan lemah pada penalaran umum. **n = 5
titik** — ini arah hubungan, bukan bentuknya. Membuktikan bentuknya adalah
tugas sumbu interpolasi (§7).

---

## 5. Stabilitas IC seleksi → holdout (figur v14)

Figur v14 sebelumnya **selalu dilewati** karena dua bug sekaligus di
[`scripts/visual_bab4.py`](../scripts/visual_bab4.py); keduanya diperbaiki di
putaran ini (§6.1). Sekarang terbaca:

- **84 pasangan ekspresi** dengan IC di kedua jendela (seleksi 2021 → holdout
  2022-01-01…2025-12-26)
- **Pearson r = 0,955**
- **berbalik tanda: 2 dari 84 (2,4%)**
- mean |IC| seleksi 0,0167 → holdout 0,0216

Per tag, jumlah ekspresi yang **hidup** di jendela holdout memperkuat temuan
§2 dari sisi lain:

| tag | ekspresi | hidup | signifikan | balik tanda |
|---|---:|---:|---:|---:|
| `kv_soft` | 33 | 17 | 17 | 0 |
| `kv_gumbel` | 30 | 20 | 16 | 0 |
| `text` | 36 | 16 | 16 | 0 |
| `kv_sample` | 22 | 15 | 11 | 0 |
| `kv_moi` | 28 | 14 | 11 | 1 |
| `kv_raw` | 11 | **1** | 1 | 0 |
| `kv_and_text_raw` | 14 | 6 | 5 | 1 |

`kv_raw` hanya menyisakan **1 ekspresi hidup dari 11**, sementara keluarga R
menyisakan 14–20. Ekspresi yang lolos gate ternyata juga yang bertahan
out-of-sample — kegagalan `raw` bukan sekadar gagal parse di depan.

> **Batas berlaku — penting.** Berkas holdout yang dipakai v14 berasal dari
> **arsip run 6-jalan 2026-08-10**
> (`results/arsip_faktor_6jalan_2026-08-10/holdout_2022-01-01_2025-12-26.json`),
> bukan dari matriks 20-jalan. Angka §5 belum sebanding langsung dengan §2.
> `skor_holdout.py` harus dijalankan ulang di atas matriks sekarang sebelum
> v14 dipakai sebagai bukti di naskah.

---

## 6. Perbaikan perkakas di putaran ini

### 6.1. v14 — dua bug, bukan satu

1. **Path hardcoded.** Fungsi mencari `results/factor/holdout_*.json`, padahal
   berkasnya sudah dipindah ke `results/arsip_faktor_6jalan_2026-08-10/`.
   Sekarang mencari di `results/factor/` **dan** `results/arsip_*/`, memakai
   yang termuda.
2. **Nama field salah.** Fungsi membaca `ic_holdout`, sedangkan
   [`backend/eval/skor_holdout.py`](../backend/eval/skor_holdout.py) menulis IC
   jendela holdout sebagai **`ic`** (dan IC seleksi sebagai `ic_seleksi`).
   Field `ic_holdout` tidak pernah ada, sehingga filter selalu menghasilkan
   daftar kosong. Ini yang membuat v14 gagal **diam-diam** — pesan lewatnya
   menuduh berkasnya belum ada, padahal berkasnya ada dan lengkap.

Label sumbu Y kini juga diambil dari `window` di berkas, bukan dikunci di
string, supaya tidak salah label kalau jendela holdout diganti.

Hasil: **24 dari 24 figur Bab IV** terbangkit.

### 6.2. `results/` dikeluarkan dari git

`results/` (181 MB, 3 245 berkas) tidak lagi di-track. Isinya artefak run yang
dibaca lewat ringkasan di `docs/` dan figur Bab IV — bukan sumber kode — dan
terus tumbuh. Salinan lengkapnya dikemas ke
**`/workspace/hasil_results_lengkap_20260827_1530.tar.gz`** (9,8 MB, 3 295
entri, integritas gzip terverifikasi, `.cache/` dikecualikan karena
regenerable dari `backend/hf_data/daily_pv.h5`).

> ⚠️ Arsip itu berada **di luar repo**. Begitu pod RunPod dihapus, ia hilang
> permanen. `scp` keluar sebelum menghapus pod — lihat
> [`scripts/kemas_hasil.sh`](../scripts/kemas_hasil.sh).

### 6.3. Bug yang ditemukan tapi BELUM diperbaiki

- **Skoring `--score-only` tidak mengunci per sel.** Kunci per-sel yang dibuat
  untuk `jalankan_matriks.py` tidak berlaku di jalur `--score-only`. Akibatnya
  dua proses sempat menyekor `kv_moi` bersamaan (dijadwalkan 14:12 dan 14:49)
  dan menulis ke `frontend_kv_moi.json` + `icseries_kv_moi.parquet` yang sama.
  Kali ini tidak ada yang rusak — keduanya dihentikan sebelum menulis, dan
  ke-11 JSON diverifikasi tetap parse bersih — tapi race-nya nyata.
  Penyebabnya: penjadwal mengecek *keberadaan berkas parquet*, bukan
  *ada tidaknya proses yang sedang mengerjakan tag itu*.
- **Penanda `"lengkap": true`** di keluaran uji formal tidak mengecek
  keseimbangan n (lihat §2.2).

---

## 7. Yang tertunda — urut prioritas

1. **Naikkan 3 sel `kv_and_text` dari 8 → 20 jalan** (36 jalan GPU, ± 1,5–2 jam
   di A40 dengan 2 slot). Tanpa ini, klaim per-anggota di medium `kv_and_text`
   tidak bisa signifikan (§1.1), dan *p* famili berdiri di atas n timpang.
   Ingat `run_factor.py` tidak punya resume — pakai `--seeds 0,1,2,3,4`
   (menimpa) atau tag sementara lalu gabung.
2. **Selesaikan skoring IC untuk 8 sel sisa** (`text`, `kv_gumbel`, `kv_moi`,
   dan 5 sel `kv_and_text`). CPU-only, tapi mahal: `text` sudah berjalan 3 jam
   16 menit dan `kv_gumbel` 2 jam 27 menit tanpa selesai saat dihentikan.
   Scorer menulis **atomik di akhir**, jadi progres yang belum tuntas hilang
   total — jalankan saat tidak akan diinterupsi. Log `kv_gumbel` penuh
   peringatan LAPACK `DLASCL parameter had an illegal value`; perlu diperiksa
   apakah itu sekadar berisik atau menandakan matriks singular.
3. **Jalankan ulang `skor_holdout.py` di atas matriks 20-jalan**, supaya v14
   (§5) sebanding dengan §2 dan bukan lagi angka dari arsip 6-jalan.
4. **Sumbu interpolasi (`mix`) — 6 sel, belum disentuh.** Dispatch-nya sudah
   siap sejak `--arm interpolasi` diterima (commit `dfc9561`):
   3 sel faktor (`kv_mix_a025/a05/a075`) + 3 sel bench
   (`humanevalplus_mix_kv_s0_a025/a05/a075`). Inilah yang mengubah §4 dari
   "arah hubungan" jadi bukti bentuknya.
5. **Perbaiki kunci `--score-only`** dan penanda `"lengkap"` (§6.3).
6. Masih terbuka dari Tahap 4: **A10** (sensitivitas arah), **A11** (stabilitas
   jangka panjang), Tahap 5 (`mutation`/`crossover`/`feedback`), dan pencarian
   `latent_steps` optimal.

---

## 8. Cara mereproduksi

```bash
cd /workspace/project/multi-agent-system
source .venv/bin/activate

# (1) sel faktor 20 jalan — contoh menaikkan kv_and_text_gumbel ke 20
PYTHONPATH=backend python backend/factor/run_factor.py \
    --model Qwen/Qwen3-8B --comm-mode kv_and_text --latent-mode gumbel \
    --latent-steps 10 --latent-temp 0.7 \
    --seeds 0,1,2,3,4 --directions d0,d1,opp_mom,opp_rev \
    --chain proposal,innovate,construct --max-repair 3 \
    --tag kv_and_text_gumbel --skip-score

# (2) skoring IC satu sel (CPU, bisa berjam-jam, menulis atomik di akhir)
PYTHONPATH=backend python backend/factor/run_factor.py --score-only --tag kv_gumbel

# (3) holdout di atas matriks sekarang
PYTHONPATH=backend python backend/eval/skor_holdout.py --budget 90

# (4) sumbu interpolasi (6 sel: 3 faktor + 3 bench)
python scripts/jalankan_matriks.py --arm interpolasi --slots 2 --dry-run
python scripts/jalankan_matriks.py --arm interpolasi --slots 2

# (5) refresh seluruh analisis + 24 figur Bab IV
python scripts/agregasi_agent_trace.py
python scripts/faktor_perhop.py
python scripts/kekuatan_uji_faktor.py --comm-mode kv
python scripts/kekuatan_uji_faktor.py --comm-mode kv_and_text
python scripts/analisis_geometri_kinerja.py
python scripts/visual_bab4.py --all         # satu figur: visual_bab4.py v14_holdout_vs_seleksi

# (6) kemas results/ sebelum pod dihapus — WAJIB, results/ tidak di-track git
tar czf /workspace/hasil_results_lengkap_$(date +%Y%m%d_%H%M).tar.gz \
    --exclude=.cache results/
```
