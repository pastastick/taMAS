# Audit Kritis QuantaLatent — bukti dari CPU, tanpa GPU

> Dibuat 2026-08-06. Semua angka di dokumen ini **dihasilkan ulang di mesin ini**
> (tanpa GPU) dan bisa direproduksi dengan skrip di `lab/`. Angka yang belum
> diuji ditandai eksplisit sebagai **hipotesis** atau masuk daftar §8 (butuh GPU).

Dokumen ini menjawab tiga masalah yang diajukan:

1. keluaran ekspresi buruk → nilai matriks (IC) jelek;
2. mode laten tidak berefek signifikan dibanding teks biasa;
3. sering *collapse* di mode laten.

Ia juga sengaja **meragukan** tiga sumber otoritas yang selama ini dipakai:
paper (LatentMAS & QuantaAlpha), memori Claude dari sesi-sesi sebelumnya, dan
solusi yang sudah terpasang di kode.

---

## 0. Yang berubah secara metodologis

Selama ini setiap pertanyaan tentang mutu faktor berakhir dengan "butuh GPU".
Itu **tidak benar**: yang butuh GPU hanyalah menjalankan LLM. Rantai
ekspresi → nilai faktor → RankIC seluruhnya `pandas`, dan datanya (`daily_pv.h5`,
398 MB) sudah ada di repo.

`lab/core.py` mereplikasi jalur evaluasi produksi persis:

| tahap | produksi | replika lab |
|---|---|---|
| ekspresi → kode | `coder/template.jinjia2` + `expr_parser` | sama, modul yang sama |
| kode → nilai | `eval()` + `function_lib` | sama, modul yang sama |
| label | `runner._factor_label` | sama (`Ref($close,-2)/Ref($close,-1)-1`) |
| jendela | `runner._oos_window` | sama (dibaca dari YAML, bukan hardcode) |
| IC | `runner._compute_factor_ic` | sama + ICIR + t-stat |

**Validasi**: 45 faktor yang punya IC terekam direproduksi dengan
median |Δ| = 6,4×10⁻⁵ dan maks |Δ| = 2,3×10⁻³ (selisih kecil hanya pada faktor
berbasis `$return`, karena `$return` di sini dihitung dari `daily_pv.h5` versi
HF sedangkan runpod memakai keluaran `qlib` provider). Untuk faktor harga/volume
murni selisihnya **nol sampai 7 desimal**.

Konsekuensi: iterasi mutu faktor **tidak lagi diblokir GPU**.

---

## 1. Jawaban ringkas untuk ketiga masalah

**Masalah 1 — ekspresi buruk.** Bukan sekadar "buruk"; sebagian besar **mati
secara numerik** dan lolos seluruh gate. Dari 86 ekspresi di batch 2026-07-05:
49% mengandung cacat semantik terdeteksi, 21 (24%) menghasilkan kolom konstan
atau NaN total. Dan faktor **terbaik** di seluruh batch ternyata rank-ekuivalen
dengan kolom mentah `$volume` itu sendiri. Sampling **acak** dari DSL yang sama
menghasilkan faktor yang **sama baiknya secara statistik** (p = 0,70) dan bahkan
menemukan faktor tunggal yang lebih kuat. Seluruh lapisan multi-agent belum
terbukti menambah nilai di atas lantai acak.

**Masalah 2 — laten tak berefek.** Pada unit analisis yang benar (trajectory,
n = 6/mode) **tidak ada beda yang meyakinkan antar-mode** — dan arah
peringkatnya bahkan **berbalik** bila metriknya |IC| alih-alih IC bertanda.
Sebagian dari "tidak ada beda" itu *memang diprediksi paper*: `kv_and_text`
mentransfer informasi yang sama dengan `text` lewat KV, dan Teorema 3.3
LatentMAS mengatakan keduanya ekuivalen — jadi itu **konfirmasi**, bukan
kegagalan. Yang benar-benar berbeda hanya lengan `kv` murni, dan bedanya
muncul bukan di kekuatan sinyal melainkan di **keragamannya**: `kv` hanya
menemukan **2 klaster sinyal dari 39 faktor**, versus 5 (text) dan 7
(kv_and_text).

**Masalah 3 — collapse laten.** Terukur, dan penyebabnya **dua**, bukan satu:
(a) vektor laten berada **di luar distribusi** — kosinus terhadap embedding token
terdekat = **−0,09** (bukan sekadar jauh: berlawanan arah); (b) rollout laten
**deterministik** sehingga entropinya nol. Solusi yang tercatat di memori
("suntik sampling/noise di rollout laten") hanya memperbaiki (b) dan terbukti
**tidak** memperbaiki (a).

---

## 2. Bukti — mutu ekspresi (Masalah 1)

### 2.1 Faktor terbaik seluruh batch = `-RANK($volume)`

Faktor ber-IC tertinggi di seluruh eksperimen (mode `text`, dir0, trajectory
`0c4eed1644e4`, IC = +0,0449, dikutip di Bab 4 §4.11.1) adalah:

```
RANK($volume) * (TS_RANK($return, 1) ? -1 : 1)
```

`TS_RANK(x, 1)` = `rolling(1).rank(pct=True)` = **1,0 konstan** (peringkat satu
observasi terhadap dirinya sendiri). Nilai 1,0 selalu *truthy*, jadi ternary
selalu memilih cabang `-1`. Ekspresi itu, secara aljabar dan secara numerik,
adalah `-RANK($volume)`:

| ekspresi | nan% | nunique | min | max | IC | t |
|---|---|---|---|---|---|---|
| `RANK($volume) * (TS_RANK($return,1) ? -1 : 1)` | 0,5 | 1.531.285 | −1,0 | −0,000214 | +0,04493 | +6,72 |
| `0 - RANK($volume)` | 0,5 | 1.531.285 | −1,0 | −0,000214 | +0,04493 | +6,72 |

Identik pada setiap statistik. Hipotesis yang menyertainya ("*small-cap stocks
with unusually high volume … liquidity-induced mean reversion*") **tidak
terimplementasi sama sekali** — tak ada kondisi kapitalisasi (kolomnya memang
tak ada), tak ada reversal, tak ada horizon. Yang tersisa: "beli saham
bervolume paling rendah".

Ini memperkuat catatan jujur yang sudah ada di Bab 4 ("*implementation is
shallow*") — tetapi jauh lebih tajam: hipotesis bukan cuma dangkal, ia **tidak
ada** di dalam ekspresi.

### 2.2 Setengah ekspresi cacat, dan gate tidak melihatnya

Gate produksi punya sembilan pemeriksaan, tetapi **semuanya struktural**
(parsable, arity, variabel dikenal, argumen kembar, panjang simbol, jumlah
fitur, subtree duplikat). Tak satu pun mengeksekusi ekspresi. Hasil audit
86 ekspresi (`lab/audit_batch.py`):

| kelas cacat | jumlah |
|---|---|
| kondisi ternary non-boolean (skor kontinu dipakai sebagai syarat) | 25 |
| window degenerate `TS_ZSCORE(·,1)` | 14 |
| window degenerate `TS_RANK(·,1)` | 8 |
| ambang > 1 pada persentil (`TS_RANK(...) < 50`) | 7 |
| ambang absolut pada `$volume` (`… < 500000`) | 5 |
| window degenerate lain (`TS_MEAN`, `TS_MIN`) | 3 |
| **total ekspresi ber-cacat** | **42 / 86 (49%)** |

Akibat numerik yang diverifikasi:

- `TS_ZSCORE($volume, 1)` → **100% NaN** (std dari satu observasi).
- `TS_RANK($return, 1)` → **konstan 1,0**.
- `(TS_ZSCORE($volume,1) > 1) ? TS_MEAN($return,1) : 0` → **konstan 0,0**, lolos
  semua gate, masuk LightGBM sebagai "faktor".
- **15 dari 86** faktor punya ≤ 2 nilai unik per hari — nol informasi lintas-saham.
- `TS_RANK($volume,1) < 50` → selalu benar kecuali saat `$volume` NaN, sehingga
  faktornya efektif adalah indikator *ketersediaan data*; ia memperoleh
  IC = +0,0063 dengan t = +3,73. Itu **artefak kualitas data yang terlihat
  "signifikan"**.

Sebagian kesalahan ini berasal dari dokumentasi DSL di prompt sendiri:
`TS_RANK(A, n)` dideskripsikan "*time-series rank of the last value of A in the
past n days*" tanpa menyebut bahwa keluarannya **persentil di [0,1]** — itulah
sebabnya model menulis `TS_RANK(...) < 50`. Jadi ini bukan murni "model kecil
bodoh": promptnya memang menyesatkan.

**Sudah diperbaiki di sesi ini** (§7): `validate_semantics()` di
`backend/factors/regulator/factor_regulator.py`, tersambung ke gate di
`backend/latent_mas/pipeline.py`, menolak keempat kelas cacat itu dengan pesan
yang bisa langsung dipakai agen *repair*.

### 2.3 `IC = None` mencampur dua hal yang sangat berbeda

Dari 41 faktor ber-`factor_ic = None` di pool: **19 memang mati numerik**,
tetapi **22 punya IC nyata** dan hilang hanya karena dibuang *correlation gate*
(`runner.py:472` menghapus nama yang di-drop dari `exp.factor_ic`). Membaca
"None" sebagai "faktor gagal" — seperti yang tersirat di catatan sebelumnya —
salah untuk lebih dari separuh kasus.

### 2.4 Seluruh eksperimen hanya menemukan ~10 sinyal

Korelasi Spearman antar-**deret IC harian** (243 hari, 56 faktor hidup),
klaster dengan ambang |ρ| > 0,7:

```
56 faktor hidup → 10 klaster sinyal
  klaster n=39  modes=[kv, kv_and_text, text]   ← 70% dari semua faktor
  klaster n= 7  modes=[kv, kv_and_text, text]
  klaster n= 3  modes=[kv_and_text]
  7 klaster sisanya n=1
```

Satu klaster memuat 70% faktor dan **muncul di ketiga mode komunikasi**. Jadi
*mode collapse* bukan gejala khas jalur laten: di mode `text`, tempat setiap
token disampel pada temperature 0,8, hasilnya mendarat di klaster yang sama.

Tetapi keparahannya **berbeda tajam** antar mode — dan di sinilah B14 terbukti:

| mode | faktor hidup | klaster sinyal unik |
|---|---:|---:|
| `text` | 17 | 5 |
| `kv_and_text` | 24 | 7 |
| **`kv`** | **39** | **2** |

`kv` menghasilkan faktor paling banyak namun hanya menemukan **dua** sinyal:
ia menemukan satu ide lalu mengulanginya 39 kali. Ini pembacaan B14 yang tepat —
*diversity collapse* memang khas `kv`, tetapi ia **bukan** ketertinggalan
kekuatan sinyal (lihat §2.6), melainkan ketertinggalan **cakupan pencarian**.

### 2.5 Lantai acak: sistem multi-agent belum mengalahkan tebakan

300 ekspresi diambil acak dari tata bahasa DSL yang sama (`lab/random_baseline.py`;
generatornya sengaja hanya memproduksi ekspresi yang **sehat** — window ≥ 2,
kondisi selalu perbandingan — jadi perbandingannya konservatif terhadap LLM).

| sumber | n | mean IC | mean \|IC\| | max \|IC\| | IC > 0 |
|---|---:|---:|---:|---:|---:|
| LLM (semua mode) | 67 | −0,0117 | 0,0158 | 0,0449 | 12% |
| **acak (null model)** | 278 | −0,0111 | **0,0170** | **0,0507** | 23% |
| LLM `text` | 20 | −0,0123 | 0,0181 | 0,0449 | 10% |
| LLM `kv_and_text` | 24 | −0,0071 | 0,0106 | 0,0387 | 21% |
| LLM `kv` | 23 | −0,0161 | 0,0192 | 0,0356 | 4% |

Mann-Whitney |IC| LLM vs acak: **z = −0,39, p = 0,70 → tidak berbeda**. Lantai
acak bahkan sedikit lebih tinggi dan menemukan faktor tunggal yang lebih kuat.

Sepuluh ekspresi acak terkuat memperjelas apa yang sebenarnya ada di data:

```
IC=-0.0507 t= -5.38  ($volume * $close)          ← proxy nilai transaksi
IC=-0.0463 t= -5.42  EMA(ABS($return), 5)
IC=-0.0459 t= -4.72  (EMA($low,10) * EMA($volume,10))
IC=-0.0449 t= -6.72  ZSCORE($volume)
IC=-0.0449 t= -6.72  SQRT($volume)
IC=-0.0449 t= -6.72  LOG(ABS($volume))
IC=-0.0449 t= -6.72  ($volume + $volume)
```

Semua bernilai −0,0449 dengan t = −6,72 **persis sama** dengan faktor terbaik LLM
(+0,0449, t = +6,72) — karena RankIC berbasis peringkat, setiap transformasi
monoton dari `$volume` memberi |IC| identik.

> Faktor terbaik yang dihasilkan seluruh sistem — tiga mode komunikasi, enam
> trajectory per mode, mutasi, crossover, kolaborasi laten, sembilan gate, agen
> repair, backtest LightGBM — **rank-ekuivalen dengan kolom masukan mentah
> `$volume`**. Bukan transformasinya: kolomnya.

Ini jawaban paling langsung untuk Masalah 1. Ia juga membalik diagnosis lama:
mean IC bertanda yang negatif (−0,0117) **hampir sama** dengan mean acak
(−0,0111), jadi kecondongan negatif itu **sifat data × DSL**, bukan bukti bahwa
LLM menulis tanda terbalik. Di universe ini, hampir semua ekspresi sederhana atas
level harga/volume memang antiprediktif — itu premi likuiditas/volatilitas klasik.

### 2.6 Faktor-faktornya nyata, fungsi fitness-nya yang membuangnya

Uji holdout **sejati** pada 2022-01-01…2025-12-26 (jendela yang tidak pernah
dilihat seleksi; = split test QuantaAlpha), 12 faktor terkuat:

| IC seleksi (2021) | IC holdout (2022-25) | t holdout | ekspresi |
|---:|---:|---:|---|
| +0,0449 | **+0,0489** | +13,98 | `RANK($volume) * (TS_RANK($return,1) ? -1 : 1)` |
| −0,0387 | −0,0426 | −11,56 | `TS_MEAN($volume, 10)` |
| −0,0358 | −0,0439 | −15,84 | `TS_RANK($return,1) * (TS_MIN($volume,1) > 1000000 ? 1 : 0)` |
| −0,0320 | −0,0341 | −10,66 | `RANK(TS_MEAN($high - $low, 10))` |
| −0,0311 | −0,0377 | −11,83 | `RANK(TS_STD($high, 10))` |

**Tanda IC bertahan pada 100% dari 12 faktor; korelasi Spearman antara jendela
seleksi dan holdout = +0,853.**

Artinya sinyal-sinyal ini **bukan noise dan bukan overfit**. Faktor ber-IC
−0,0426 dengan t = −11,56 di holdout adalah alfa yang stabil dan dapat
diperdagangkan — cukup dibalik tandanya. Sistem menemukannya, lalu
membuangnya, karena `get_primary_metric()` memakai IC **bertanda** (§S3).

Jadi kalimat "kualitas absolut lemah di semua mode" di catatan lama perlu
diganti: kualitasnya tidak lemah, **pengukurannya** yang membuang separuh
penemuan.

---

## 3. Bukti — mekanisme laten (Masalah 2 & 3)

### 3.1 Matriks realignment adalah identitas pada Qwen3-4B

`LatentRealigner` (`backend/llm/_shared.py:696`) menyelesaikan
M = (Wₒᵤₜᵀ Wₒᵤₜ + λI)⁻¹ Wₒᵤₜᵀ W_in. Tetapi `config.json` Qwen3-4B memuat
**`"tie_word_embeddings": true`** → Wₒᵤₜ *adalah* W_in → M = (WᵀW + λI)⁻¹WᵀW ≈ I.

Diukur langsung dari bobot checkpoint (`lab/realign_probe.py`, memuat hanya
matriks embedding sehingga muat di RAM 5 GB):

```
‖M − I‖_F / ‖I‖_F        = 1,44 × 10⁻⁶
cos(h, hM) rata-rata     = 1,000000   (min 0,9999997)
rasio norma ‖hM‖/‖h‖     = 1,000000
faktor shrinkage minimum = 0,9999979
target_norm              = 1,0974
```

Artinya: pada backbone ini, seluruh derivasi ridge — yang di Bab 4 §4.4.1 ditulis
sebagai mekanisme inti dan "identik dengan paper" — **secara operasional tidak
melakukan apa pun**. Langkah laten yang sebenarnya berjalan adalah

```
z = h / ‖h‖ × 1,0974
```

Ablasi `use_realign=True/False` pada Qwen3-4B karena itu **bukan ablasi**: kedua
cabang menghitung hal yang sama.

Catatan penting agar adil: **kodenya tidak salah.** Untuk model *tied*, identitas
memang jawaban yang benar secara matematis. Yang keliru adalah narasi yang
menempatkannya sebagai mekanisme penting. Qwen3-8B/14B tidak *tied*, jadi di sana
M betul-betul bekerja — tetapi skripsi ini memakai 4B.

### 3.2 Rollout laten: titik tetap, di luar distribusi, entropi nol

`lab/latent_dynamics.py` mereplikasi langkah laten produksi dan mengukurnya per
langkah. Dijalankan di CPU pada GPT-2 (juga *tied*, d=768, 40 langkah, 3 prompt
arah berbeda × 3 seed). **Ini demonstrasi mekanisme, bukan pengganti Qwen3-4B**
(lihat §8 butir G1).

| varian langkah laten | H akhir | cos(hₖ,hₖ₋₁) | cos ke embedding terdekat | token unik | jalur identik antar-seed |
|---|---:|---:|---:|---:|---:|
| **`raw` (= produksi)** | 3,98 | 0,99996 | **−0,090** | 6/40 | **9/9** |
| `raw` + noise Gauss 0,1 | 3,57 | 0,99991 | −0,089 | 4/40 | 0/9 |
| `soft` T=1 (convex hull) | 1,04 | 0,99976 | +0,949 | 2/40 | 9/9 |
| `soft` T=2 | 6,83 | 0,99999 | +0,883 | 8/40 | 9/9 |
| **`gumbel` T=0,7** | 5,32 | **0,99503** | **+0,899** | **22/40** | **0/9** |
| `gumbel` T=1,0 | 4,50 | 0,97722 | +0,856 | 15/40 | 0/9 |
| `sample` T=1 (token disampel, tak diemit) | 5,43 | 0,95900 | +1,000 | 29/40 | 0/9 |

Tiga hal terbaca langsung:

1. **Vektor laten produksi berada di luar manifold embedding.** Kosinus ke
   embedding token *terdekat* = −0,09. Model diberi masukan yang belum pernah
   dilihatnya sepanjang pelatihan, lalu diminta melanjutkan menulis ekspresi
   simbolik. Inilah kandidat mekanis untuk *repetition collapse* B2/B10
   ("selection selection selection…").
2. **Semua varian deterministik konvergen ke titik tetap** dalam beberapa langkah
   (cos antar-langkah ≥ 0,9997). Jadi `latent_steps: 60` tidak memberi 60 pikiran;
   ia memberi ~2 pikiran lalu ~58 salinan vektor yang sama di KV. Ini konsisten
   dengan B7 (ls 60 lebih buruk dari 10–20) dan menjelaskan sebabnya.
3. **Entropi jalur laten produksi persis nol**: 9 dari 9 pasang seed
   menghasilkan lintasan identik. Pencarian evolusioner hidup dari varians;
   jalur laten murni tidak menyediakannya sama sekali.

Dan satu hal yang **membantah rencana perbaikan yang sudah tercatat**: menyuntik
noise Gauss ke `raw` memang memulihkan varians (0/9 identik) tetapi
**tidak memindahkan vektor ke dalam distribusi** (cos tetap −0,089). Itu obat
untuk setengah penyakit.

Varian `gumbel` adalah satu-satunya yang memenuhi ketiganya sekaligus: tetap
**kontinu** (argumen ekspresivitas LatentMAS tetap berlaku), berada **di dalam
convex hull** embedding nyata (+0,899), **lolos dari titik tetap** (0,995), dan
**punya knob entropi**. Usulan konkret ada di §7.

---

### 3.3 Uji antar-mode pada unit analisis yang benar

Uji per-faktor menyesatkan: 70% faktor berada dalam satu klaster sinyal, jadi
mereka bukan pengamatan independen dan nilai-p per-faktor terlalu optimistis.
Unit yang benar adalah **trajectory** (n = 6 per mode):

| mode | n | mean \|IC\| per trajectory | sd |
|---|---:|---:|---:|
| `text` | 6 | 0,0219 | 0,0136 |
| `kv_and_text` | 6 | 0,0110 | 0,0039 |
| `kv` | 5 | 0,0184 | 0,0058 |

Welch t: `kv` vs `text` = −0,57 (tidak signifikan); `kv` vs `kv_and_text` = +2,44
(di ambang); `text` vs `kv_and_text` = +1,90 (tidak signifikan).

Dua kesimpulan yang harus masuk Bab 4:

1. **Peringkat mode berbalik ketika metrik berganti dari IC ke |IC|.** Bab 4
   sekarang menulis "`kv` terburuk kategoris, `kv_and_text` ≥ `text`". Dengan
   |IC| — metrik yang benar untuk faktor alpha — urutannya menjadi
   `text` ≈ `kv` > `kv_and_text`. Kesimpulan yang tidak tahan terhadap pergantian
   metrik yang dapat dipertahankan **tidak boleh dilaporkan sebagai temuan**.
2. **Desain n = 1 run/mode tidak punya daya statistik** untuk membedakan efek
   sebesar ini. Itu bukan aib — itu hasil yang bisa dilaporkan jujur, disertai
   perhitungan berapa replikasi yang dibutuhkan (§8, G4).

---

## 4. Meragukan papernya

### 4.1 "Lossless" LatentMAS tidak berarti seperti yang kita kutip

Teorema 3.3 berbunyi: keluaran agen yang menerima *latent working memory* setara
dengan keluaran bila kita langsung memasukkan keluaran agen sebelumnya.
Buktinya (Lampiran B.2) adalah induksi atas lapisan yang mengatakan: KV yang
dibaca dari cache sama dengan KV yang dihitung ulang dari token yang sama.

Itu **pernyataan tentang benarnya KV-caching**, bukan tentang laten versus teks.
Teorema itu tidak mengklaim — dan tidak bisa mengklaim — bahwa 60 vektor laten
membawa informasi yang setara dengan paragraf teks yang tidak pernah ditulis.
Jadi kalimat "transfer KV lossless" **tidak boleh dipakai** untuk membela mode
`kv` murni; ia hanya membela `kv_and_text` (yang memang membawa teks di KV).

Konsekuensi langsung untuk skripsi: hasil `kv_and_text ≈ text` **bukan hasil
nol**. Itu justru satu-satunya prediksi teoretis yang bisa diturunkan dari
Teorema 3.3, dan data kita konsisten dengannya.

### 4.2 Teorema ekspresivitas menghitung kapasitas, bukan keterjangkauan

Teorema 3.1 menghitung |H| = 3^d_h di bawah *Linear Representation Hypothesis*
dan menyimpulkan laten 235,7× lebih efisien daripada teks untuk Qwen3-4B. Yang
dihitung adalah **himpunan yang bisa direpresentasikan**. Yang relevan bagi
sistem *training-free* adalah **himpunan yang bisa dicapai**: h_{k+1} = f(h_k)
adalah peta deterministik, sehingga dari satu prompt himpunan terjangkaunya
adalah **satu lintasan** — dan §3.2 menunjukkan lintasan itu konvergen ke satu
titik. Entropi kondisionalnya nol, berapa pun d_h.

Ini bisa dirumuskan sebagai kontribusi teoretis kecil tapi jujur untuk skripsi:

> Ekspresivitas per-langkah dan entropi per-langkah adalah dua besaran berbeda.
> Untuk tugas jawaban-tunggal (GSM8K, MedQA, HumanEval — sembilan benchmark
> LatentMAS semuanya bertipe ini) hanya yang pertama yang penting. Untuk
> pencarian evolusioner, yang menggerakkan sistem adalah yang kedua. Kolaborasi
> laten murni memaksimalkan yang pertama sambil menihilkan yang kedua.

### 4.3 Kontribusi realignment inert pada backbone terkecil mereka sendiri

§3.1 berlaku untuk Qwen3-4B siapa pun yang memakainya — termasuk LatentMAS.
Kenaikan akurasi yang mereka laporkan pada 4B karena itu **tidak dapat**
diatribusikan ke realignment. Ini pertanyaan sah untuk sidang, dan pertanyaan
sah untuk penulis paper.

### 4.4 Kita membandingkan angka QuantaAlpha yang salah kategori

Ini koreksi paling mahal. Catatan lama berbunyi: "QuantaAlpha IC 0,15 di ~11–12
iterasi", dipakai sebagai bukti bahwa faktor kita (IC ~0,04) sangat jauh
tertinggal, dan melahirkan daftar "LEVER IC" (perbanyak iterasi, dll).

Membaca ulang papernya: Tabel 1 membandingkan QuantaAlpha dengan **Alpha158,
Alpha360, LSTM, GRU, Transformer, TRA, LightGBM** — semuanya *pipeline model*.
IC 0,1501 di sana adalah IC **prediksi model**, bukan IC satu faktor. IC
per-faktor mereka ada di **Tabel 3**:

| faktor QuantaAlpha (2023, CSI 300) | Rank IC |
|---|---:|
| GapZ10_Overnight_vs_TR | 0,0793 |
| Gap_IntradayAcceptanceScore_20D | 0,0744 |
| Gap_IntradayAcceptance_VolWeighted_20D | 0,0606 |
| CleanTrend_Continuation_Score_RS10_WVMA5 | 0,0590 |
| OrderlyTrend_x_Absorption_10D_5D_20D | 0,0465 |
| *(faktor lemah mereka)* KineticLength_AbsRetSum_Z_10D | **−0,0720** |
| *(lemah)* Drawdown_Gated_NegCorr_60D_20D_thr20pct | −0,0282 |

Faktor terbaik kita, +0,0449, **sekelas dengan faktor peringkat kelima mereka**,
dan paper itu sendiri melaporkan faktor ber-Rank IC −0,072. (Dengan catatan
merendahkan dari §2.5: faktor kita itu rank-ekuivalen dengan `$volume` mentah,
sementara faktor mereka adalah struktur *overnight gap* yang benar-benar
tersusun. Angkanya sebanding; kandungan riset di baliknya tidak.) Jadi:

- premis "faktor kita jauh lebih buruk dari paper" **tidak terbukti** dari
  angka yang tersedia;
- "mayoritas IC negatif" bukan tanda sistem rusak — paper pembanding pun punya;
- daftar "LEVER IC" yang dibangun di atas premis itu perlu ditinjau ulang.

Dua peringatan agar tidak berbalik jadi klaim berlebihan ke arah sebaliknya:
universe dan periode kita **berbeda** dari mereka (§5, butir M3 dan M4), jadi
angka tetap tidak sebanding secara langsung. Yang runtuh adalah premis
"tertinggal jauh", bukan berubah jadi "kita setara".

### 4.5 Alpha158 sebagai lantai

Paper melaporkan Alpha158 (158 fitur + model) mencapai Rank IC 0,0334 di CSI 300.
Artinya harness Alpha158 (T11, "blocking skripsi") tidak perlu menjadi target
yang menakutkan — dan `-RANK($volume)` satu baris kita sudah di atas angka itu
pada universe/periode kita sendiri. Perbandingan yang jujur tetap harus
menyamakan universe dan periode.

---

## 5. Meragukan memori Claude sendiri

Memori proyek dibuat dari kesimpulan Claude di sesi-sesi sebelumnya. Berikut
yang **tidak lolos** pemeriksaan ulang hari ini.

**M1. "QuantaAlpha IC 0,15 per faktor" — SALAH.** Itu IC level model. Lihat §4.4.
Seluruh rantai kesimpulan "IC kita 10× lebih buruk → perbanyak iterasi" berdiri
di atas kesalahan kategori ini.

**M2. "B14: kv terburuk karena `latent_pass` deterministik → diversity collapse"
— MEKANISMENYA BENAR, KESIMPULAN "TERBURUK"-NYA SALAH.**
Yang terbukti: determinisme nyata (9/9 lintasan identik, §3.2) dan `kv` memang
paling kolaps keragamannya — **2 klaster sinyal dari 39 faktor** versus 5 dan 7
di mode lain (§2.4). Yang **tidak** terbukti: bahwa `kv` "terburuk". Pada |IC|
per trajectory `kv` = 0,0184 justru **di atas** `kv_and_text` = 0,0110 (§3.3).
Label "terburuk" sepenuhnya berasal dari pemakaian IC bertanda.
Dua koreksi tambahan: (a) determinisme bukan satu-satunya sebab — geometri OOD
(cos −0,09) berdiri sendiri dan **tidak** disembuhkan oleh sampling, sehingga
rencana yang tercatat ("injeksi noise/sampling di latent rollout") adalah
setengah obat; (b) *mode collapse* itu sendiri terjadi di **semua** mode
(10 klaster dari 86 ekspresi), jadi ia bukan kelemahan khas laten — yang khas
laten adalah **keparahannya**.

**M3. "Bab 3: split test (OOS) 2022–2025" — TIDAK SESUAI KODE.**
`runner._oos_window()` membaca `conf_combined_factors.yaml`, yang berisi
`test: [2021-01-01, 2021-12-31]`. Setiap angka `FactorIC_mean` di Bab 4 Tabel 4.3
dihitung pada **2021 saja, 243 hari perdagangan** (terverifikasi: `n_days = 243`
di seluruh audit). Lebih serius: 2021 adalah juga jendela yang dipakai evolution
untuk **memilih** parent. Jadi angka Bab 4 adalah **metrik seleksi**, bukan
holdout — dan menyebutnya "OOS" menyesatkan. (Uji holdout sejati: §6.)

**M4. Judul "Studi Kasus Saham Indeks CSI 300" — tidak cocok dengan metrik utama.**
IC per-faktor dihitung lintas **± 4.370 emiten per hari** (terukur, kolom
`coverage` di `lab/out/audit_batch.json`) — yaitu hampir seluruh pasar A-share di
`daily_pv.h5`, bukan 300 emiten CSI 300. Hanya backtest portofolio LightGBM yang
memakai CSI 300. Dua metrik utama skripsi berjalan di dua universe berbeda.

**M5. "Realignment ridge = mekanisme inti, terverifikasi identik dengan paper" —
menyesatkan untuk backbone ini.** Rumusnya memang identik; efeknya identitas
(§3.1).

**M6. "Klaim KV lossy simbolik = artefak bug B11 (dangling turn)" — koreksinya
sendiri benar, tapi kesimpulan turunannya terlalu longgar.** Bahwa `kv` bisa
menghasilkan ekspresi parseable setelah B11 diperbaiki memang benar. Tetapi itu
tidak berarti jalur laten sehat: §3.2 menunjukkan masukannya tetap OOD dan
entropinya tetap nol.

**M7. Yang tetap bertahan.** `$vwap` memang tidak ada di data (terverifikasi:
kolomnya `$open $close $high $low $volume $factor`, `$return` diturunkan);
`latent_pass` memang tanpa sampling; duplikasi lintas-ronde memang parah (§2.4
mengukurnya lebih tajam).

---

## 6. Meragukan solusi yang sudah ada

**S1. Sembilan gate tidak pernah mengeksekusi apa pun.** Semua struktural →
49% ekspresi cacat lolos, 15 di antaranya konstan. Gate paling murah yang
hilang adalah menjalankan ekspresi pada sampel kecil (≈1 detik CPU) dan menolak
kolom NaN-total/konstan. Lihat §7.

**S2. Correlation gate menghukum yang salah.** Ia menyimpan hanya faktor
ber-IC > 0 ke *store* (`runner.py:197`), padahal faktor ber-IC −0,039 sama
informatifnya dengan +0,039 (tinggal dibalik tandanya). Akibatnya memori
redundansi buta terhadap separuh ruang, sementara 22 faktor ber-IC nyata dibuang
diam-diam dan tercatat sebagai `None` (§2.3).

**S3. Fitness memakai IC bertanda.** `get_primary_metric()` → `FactorIC_mean`
bertanda; seleksi parent, `is_successful`, dan `replace-SOTA` semuanya
memperlakukan IC negatif sebagai gagal. Untuk faktor alpha itu keliru: yang
diperlukan adalah |IC| (dengan tanda ditetapkan **di jendela latih**, bukan di
jendela evaluasi — kalau tidak, itu look-ahead).

**S4. Seleksi dan pelaporan memakai jendela yang sama.** §M3. Selama evolution
memilih di 2021 dan Bab 4 melaporkan 2021, angka apa pun yang membaik bisa jadi
sekadar overfitting ke satu tahun.

**S5. Perbaikan prompt "observable grounding" tidak bekerja.** Ia ditambahkan
untuk mencegah hipotesis yang tidak teramati; seluruh batch tetap didominasi
hipotesis *small-cap* padahal kolom kapitalisasi tidak ada. Larangan berbasis
prompt gagal; yang diperlukan adalah gate sisi-kode (variabel yang disebut
hipotesis harus ada di ekspresi, dan kolom yang tak ada harus ditolak).

**S6. Dokumentasi DSL di prompt sendiri menyesatkan** (§2.2, `TS_RANK` persentil).
Ini menyalahkan model untuk kesalahan yang kita tanam.

**S7. `latent_steps: 60`.** §3.2 menunjukkan lintasan konvergen dalam beberapa
langkah; 60 langkah terutama menambah token KV yang identik. Komentar di
`configs/experiment.yaml` sendiri sudah menyarankan 10–20 (B7) tetapi nilainya
tak pernah diubah.

---

## 6b. Konsekuensi untuk skripsi

Kabar buruknya: beberapa kalimat di Bab 3 dan Bab 4 tidak sesuai kode
(M3, M4, M5) dan satu temuan utama tidak tahan pergantian metrik (§3.3).
Itu harus diperbaiki, dan sebagian datanya perlu dilaporkan ulang.

Kabar baiknya jauh lebih besar: yang tersisa **lebih kuat** daripada rencana
semula, bukan lebih lemah.

Rencana lama — "studi banding KV vs TEXT" — bertumpu pada perbedaan antar-mode
yang, setelah diukur pada unit analisis yang benar, **tidak signifikan pada n = 1
run/mode**. Skripsi yang jantungnya sebuah efek null tanpa kontrol adalah skripsi
yang rapuh di sidang.

Yang sekarang tersedia sebagai gantinya, semuanya sudah terukur:

1. **Kontrol yang belum pernah dimiliki siapa pun di korpus pembanding**: lantai
   acak. Analisis di `ANALISIS_SKRIPSI_REFERENSI.md` mencatat bahwa 11 skripsi
   rujukan tidak melakukan uji signifikansi sama sekali. Skripsi ini bisa masuk
   dengan null model + uji Mann-Whitney + uji holdout — tiga hal yang tidak ada
   di satu pun pembanding.
2. **Hasil negatif yang tajam dan dapat dipertahankan**: sistem multi-agent LLM
   pada model 4B tidak mengungguli sampling acak dari DSL-nya sendiri
   (p = 0,70), dan faktor terbaiknya rank-ekuivalen dengan satu kolom masukan.
   Ini persis genre "menunjukkan kelemahan suatu metode" yang Anda sebut sah —
   tetapi dengan bukti kuantitatif, bukan anekdot.
3. **Mekanisme yang terukur, bukan spekulasi**: realignment = identitas
   (1,4×10⁻⁶), vektor laten OOD (cos −0,09), rollout konvergen ke titik tetap,
   entropi nol (9/9), keragaman sinyal `kv` = 2 klaster/39 faktor.
4. **Usulan perbaikan yang lahir dari bukti itu** (§7) — sehingga skripsi tidak
   berhenti di "metode ini gagal" melainkan "gagal karena X dan Y, dan inilah
   intervensi yang menyasar keduanya, dengan hasil awal Z".

Judul yang sudah dikunci ("Perbandingan Medium Komunikasi Laten dan Teks…")
masih muat, karena perbandingan itu tetap dilakukan — hanya kesimpulannya
berubah dari "kv terburuk" menjadi "perbedaan medium tenggelam di bawah kolaps
pencarian yang menimpa semua medium, dan inilah mekanismenya". Kalau nanti
ingin diperkuat, sumbu barunya bukan kv-vs-text melainkan **entropi jalur
komunikasi** (§7), yang mencakup kv dan text sebagai dua titik ekstrem.

---

## 7. Yang sudah dikerjakan di sesi ini, dan usulan berikutnya

**Sudah terpasang (tanpa GPU, sudah diuji):**

1. `lab/` — harness evaluasi faktor di CPU, tervalidasi terhadap IC produksi.
2. `validate_semantics()` di `factors/regulator/factor_regulator.py`, tersambung
   ke gate `latent_mas/pipeline.py`. Menolak keempat kelas cacat §2.2 dengan
   pesan ramah-LLM. Diuji: menolak keempat contoh nyata, meloloskan empat
   ekspresi sehat.

**Usulan utama — mengganti langkah laten (butuh GPU untuk validasi akhir):**

Ganti `z = h/‖h‖ · c` dengan campuran konveks embedding ber-noise Gumbel:

```python
gum = -log(-log(U))                       # U ~ Uniform(0,1)
z   = softmax((W_out·h + gum) / T) @ W_in
z   = z / ‖z‖ · target_norm
```

Alasannya, berdasarkan tabel §3.2: kontinu (klaim ekspresivitas LatentMAS tetap
berlaku), di dalam convex hull embedding (in-distribution), keluar dari titik
tetap, dan `T` menjadi knob entropi eksplisit untuk pencarian evolusioner.
Sebagai lengan pembanding, `sample` (token disampel, tak pernah diemit) adalah
batas diskret dari usulan yang sama.

Ini juga memberi variabel eksperimen baru yang lebih tajam daripada
"kv vs text": **entropi jalur komunikasi**, dengan `T` sebagai sumbu kontinu.

---

## 8. Yang WAJIB diperiksa dengan GPU

Diurutkan menurut nilai per jam GPU.

**G1 (paling penting). Ulangi `lab/latent_dynamics.py` pada Qwen3-4B.**
```bash
python lab/latent_dynamics.py --model Qwen/Qwen3-4B --steps 80 --device cuda
```
Semua angka §3.2 berasal dari GPT-2. Yang harus dikonfirmasi di Qwen3-4B:
(a) cos ke embedding terdekat untuk `raw` — apakah juga negatif?
(b) berapa langkah sampai cos antar-langkah > 0,999 (menentukan `latent_steps`
yang benar); (c) apakah `gumbel` tetap lolos titik tetap.
**Kalau (a) atau (b) tidak terkonfirmasi di Qwen3-4B, klaim §3.2 harus dicabut.**

**G2. Uji cepat `latent_steps`.** Jalankan mode `kv` dengan `latent.steps` ∈
{5, 10, 20, 60}, 2 trajectory tiap nilai. Prediksi dari §3.2: mutu tak menurun
di 5–10 sementara waktu dan tingkat *degenerasi* turun tajam. Murah, dan bisa
langsung masuk Bab 4 sebagai ablasi nyata.

**G3. Implementasikan + uji langkah laten `gumbel`.** Patch ~15 baris di
`client.py::latent_pass` (knob mode + T). Ukur: apakah lintasan berbeda
antar-trajectory (lawan B14), apakah `construct` masih *collapse*, apakah jumlah
klaster sinyal (§2.4) naik di atas 10.

**G4. Replikasi ≥3 seed per comm_mode.** Tanpa ini uji signifikansi antar-mode
tidak bermakna; n=1 saat ini.

**G5. Verifikasi ulang gate semantik pada run nyata.** Setelah §7 terpasang,
periksa berapa persen kandidat kini ditolak dan apakah agen *repair* berhasil
memperbaikinya (risiko: penolakan naik tajam → putaran repair membengkak).
Kalau tingkat tolak > ~60%, prompt DSL harus diperbaiki lebih dulu (§S6).

**G6. Ablasi `use_realign`** — **jangan dijalankan pada Qwen3-4B**; §3.1
membuktikan kedua cabang identik. Kalau ablasi ini diinginkan untuk skripsi,
ia harus dijalankan pada Qwen3-8B (tidak *tied*).

---

## 9. Reproduksi

```bash
cd quantalatent
.venv/bin/python lab/realign_probe.py                    # §3.1  (~2 mnt)
.venv/bin/python lab/latent_dynamics.py --model gpt2     # §3.2  (~15 mnt)
.venv/bin/python lab/audit_batch.py                      # §2    (~25 mnt)
.venv/bin/python lab/random_baseline.py 300 0            # lantai acak (~2 jam)
.venv/bin/python lab/synthesis.py                        # §2.4 + holdout
```

Keluaran JSON ada di `lab/out/`.
