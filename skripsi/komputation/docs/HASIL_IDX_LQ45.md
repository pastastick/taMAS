# Hasil penilaian korpus di pasar Indonesia (LQ45)

> Dibuat 2026-08-28. Jendela seleksi **selesai**; jendela holdout 2022–2025
> masih berjalan (`bash` → `results/logs/skor_idx.log`).
> Berkas hasil: `results/factor/holdout_daily_pv_idx_lq45_<awal>_<akhir>_q0.2.json`.

## 1. Apa yang diganti dan apa yang tidak

| tahap | pasar A-share | pasar IDX | dijalankan ulang? |
|---|---|---|---|
| pembangkitan ekspresi oleh agen LLM (GPU) | — | — | **TIDAK** |
| gate ekspresi saat generasi | sampel 2018 | — | tidak (lihat §4) |
| penilaian RankIC + backtest (CPU) | `daily_pv.h5` | `daily_pv_idx_lq45.h5` | **ya** |

Korpus yang dinilai **identik**: 1.200 ekspresi unik dari 14 sel × 20 jalan.
Ini sah karena prompt agen tidak pernah menyebut bursa mana pun
(`market_context` tak pernah diisi pemanggil mana pun) dan DSL-nya hanya
mengenal enam kolom data — `$open $high $low $close $volume $return` — yang ada
di kedua panel.

## 2. Deskripsi data yang benar

| | A-share (data lama) | IDX LQ45 (data baru) |
|---|---:|---:|
| instrumen di panel | 5.982 | 37 |
| rerata saham/hari (2021) | **4.343** | **37** |
| hari bursa 2021 | 243 | 247 |
| hari bursa 2022–2025 | — | 958 |
| kuantil portofolio | 0,1 (desil, ~434/sisi) | 0,2 (kuintil, 7/sisi) |
| hari/tahun anualisasi | 243 | 241 |

**Data lama BUKAN CSI 300.** `eval/ic.py` memuat seluruh panel tanpa filter
universe; medan `coverage` pada 1.129 ekspresi berskor menunjukkan rerata 4.343
saham/hari. Nama "CSI300" hanya berasal dari nama dataset HuggingFace dan dari
`conf_combined_factors.yaml` (`market: csi300`) yang hanya dibaca jalur
LightGBM Qlib — jalur yang sudah dibuang dari skripsi.

## 3. Hasil jendela seleksi (2021) — temuan utama BERTAHAN

Perbandingan keluarga relaksasi diskret **R = {soft, sample, gumbel, moi}**
melawan **`raw`** pada medium `kv`, memakai laju "ekspresi menghasilkan kolom
hidup" (IC terdefinisi dan lebih dari dua nilai unik):

| pasar | R | `raw` | selisih | Fisher eksak | odds ratio | Cohen's *h* |
|---|---|---|---:|---|---:|---:|
| A-share (4.343 saham/hari) | 373/438 = **85,2 %** | 26/59 = **44,1 %** | +41,1 pp | *p* = 2,84 × 10⁻¹¹ | 7,28 | 0,899 |
| **IDX LQ45 (37 saham/hari)** | 394/438 = **90,0 %** | 28/59 = **47,5 %** | **+42,5 pp** | ***p* = 2,19 × 10⁻¹³** | 9,91 | 0,977 |

**Selisihnya praktis identik (+41,1 vs +42,5 poin persen)** meskipun lebar
lintas-saham turun 117×. Temuan utama skripsi tidak bergantung pasar.

### Rincian per sel

| sel | ekspresi | hidup (A-share) | hidup (IDX) | sig (A-share) | sig (IDX) | mean\|IC\| (A-share) | mean\|IC\| (IDX) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `text` | 120 | 82 | 98 | 48 | 11 | 0,0146 | 0,0114 |
| `kv_raw` | 59 | 26 | 28 | 17 | 5 | 0,0125 | 0,0116 |
| `kv_soft` | 111 | 96 | 100 | 62 | 14 | 0,0162 | 0,0121 |
| `kv_sample` | 109 | 95 | 99 | 57 | 28 | 0,0162 | 0,0185 |
| `kv_gumbel` | 112 | 95 | 103 | 64 | 22 | 0,0157 | 0,0157 |
| `kv_moi` | 106 | 87 | 92 | 47 | 21 | 0,0163 | 0,0154 |
| `kv_mix_a025` | 68 | 52 | 54 | 39 | 11 | 0,0186 | 0,0153 |
| `kv_mix_a05` | 116 | 96 | 102 | 52 | 17 | 0,0145 | 0,0129 |
| `kv_mix_a075` | 119 | 100 | 105 | 55 | 20 | 0,0148 | 0,0152 |
| `kv_and_text_raw` | 29 | 13 | 17 | 8 | 3 | 0,0137 | 0,0136 |
| `kv_and_text_soft` | 120 | 94 | 103 | 58 | 23 | 0,0147 | 0,0156 |
| `kv_and_text_sample` | 119 | 88 | 99 | 13 | 13 | 0,0134 | 0,0124 |
| `kv_and_text_gumbel` | 120 | 93 | 109 | 57 | 19 | 0,0140 | 0,0153 |
| `kv_and_text_moi` | 120 | 94 | 104 | 59 | 26 | 0,0143 | 0,0162 |

## 4. Cara membaca turunnya jumlah ekspresi signifikan

Jumlah ekspresi signifikan turun tajam (mis. `kv_soft` 62 → 14) **sementara
rerata |IC| hampir tidak berubah** (0,0162 → 0,0121). Itu bukan penurunan mutu
sinyal, melainkan kenaikan ambang deteksi.

Di bawah hipotesis nol, simpangan baku korelasi Spearman lintas-saham adalah
$1/\sqrt{N-1}$, sehingga

$$\operatorname{se}(\overline{\mathrm{IC}})=\frac{1}{\sqrt{(N-1)T}},\qquad
|\overline{\mathrm{IC}}|_{\min}=\frac{1{,}96}{\sqrt{(N-1)T}}$$

| pasar | $N$ | $T$ | se($\overline{\mathrm{IC}}$) | $\|\overline{\mathrm{IC}}\|_{\min}$ |
|---|---:|---:|---:|---:|
| A-share | 4 343 | 243 | 0,00097 | **0,0019** |
| IDX LQ45 | 37 | 247 | 0,01060 | **0,0208** |

Ambang naik **11×**. Ekspresi ber-|IC| 0,015 sangat signifikan di A-share
($t\approx15$) dan tidak signifikan di LQ45 ($t\approx1{,}4$) — faktor yang
sama, pengukuran yang berbeda. Kalimat ini harus masuk Bab 4.

## 5. Yang belum selesai

- [ ] jendela holdout 2022-01-01..2025-12-26 (sedang berjalan)
- [ ] uji gate lintas-pasar (`scripts/uji_gate_lintas_pasar.py`) — menjawab
      "apakah keputusan gate saat generasi masih berlaku di pasar lain"
- [ ] tabel LaTeX: `python analisis/10_tabel_faktor.py --holdout
      holdout_daily_pv_idx_lq45_2022-01-01_2025-12-26_q0.2.json
      --keluaran faktor_holdout_idx.tex`
- [ ] subbab Bab 3 "Data Pasar dan Universe Penilaian"
      (draf: `skripsi/pemikiran/draf/bab3_data_pasar.tex`)
