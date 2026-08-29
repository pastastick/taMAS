# Indeks figur & data pendukung bab hasil

> **Status per 2026-08-29.** Penelitian berpindah pasar: penilaian faktor kini
> dilakukan pada **Bursa Efek Indonesia, indeks LQ45 (37 emiten, 2021--2025)**,
> menggantikan panel A-share. Indeks ini sudah disesuaikan dengan keputusan itu.
>
> Regenerasi figur A-share (yang masih berlaku): `python scripts/visual_bab4.py --all`
> Regenerasi figur IDX: `python scripts/visual_idx.py`
> Regenerasi tabel: `python scripts/analisis_idx.py && python scripts/tabel_idx.py`

---

## 0. Aturan pakai — baca ini dulu

Perpindahan pasar membelah figur yang ada menjadi tiga kelompok. Membaca
indeks ini tanpa memperhatikan pembagiannya akan menghasilkan bab yang
mencampur dua bursa dalam satu tabel.

| kelompok | arti | boleh dipakai? |
|---|---|---|
| 🟢 **netral-pasar** | tidak menyentuh data pasar sama sekali (geometri, token, waktu, korupsi teks, kapasitas kanal) | **ya, apa adanya** |
| 🔵 **IDX** | dibangun dari laporan penilaian LQ45 | **ya** |
| 🔴 **kedaluwarsa** | membaca medan `ic` dari `frontend_*.json`, yang berisi angka **A-share** | **tidak** — sudah digantikan |

Alasan kelompok 🔴 ada: `skor_holdout.py` sengaja bekerja pada SALINAN dan
tidak pernah menimpa `frontend_*.json`, supaya angka A-share tetap dapat
diverifikasi. Konsekuensinya, medan `ic` di berkas itu selamanya A-share, dan
figur mana pun yang membacanya juga A-share.

---

## 1. 🔵 Figur IDX — dipakai bab hasil (results/visual_idx/*.png)

| # | berkas | pertanyaan yang dijawab | sumber data |
|---|---|---|---|
| i01 | `i01_disosiasi.png` | **FIGUR UTAMA.** Apakah validitas dan mutu sinyal berubah ke arah yang sama? | `results/idx/analisis_idx.json` |
| i02 | `i02_sebaran_ic.png` | Bagaimana sebaran \|IC\| tiap formulasi relatif terhadap ambang deteksi $1{,}96/\sqrt{(N-1)T}$? | laporan IDX dua jendela |
| i03 | `i03_stabilitas.png` | Apakah tanda IC bertahan dari seleksi 2021 ke holdout 2022--2025? | laporan IDX holdout |
| i04 | `i04_efisiensi.png` | Bagaimana posisi tiap sel pada bidang biaya token vs keandalan? | `analisis_idx.json` |
| i05 | `i05_peluruhan.png` | Bagaimana \|IC\| meluruh per tahun setelah jendela seleksi? | `icseries_daily_pv_idx_lq45_*.parquet` |

**Temuan yang dibawa i01** — inilah kontribusi utama penelitian: pada korpus
ekspresi yang sama persis, kontras keluarga $\mathcal{R}$ vs `raw` memberi
$p = 5{,}7\times10^{-7}$ (kv) dan $p = 8{,}8\times10^{-13}$ (kv+teks) untuk
**validitas** keluaran, tetapi $p = 0{,}21$ dan $p = 0{,}64$ untuk **mutu
sinyal**. Selisih lima sampai dua belas orde besaran pada data yang sama.

---

## 2. 🟢 Figur netral-pasar — tetap dipakai (results/visual/*.png)

| # | berkas | pertanyaan yang dijawab | sumber data |
|---|---|---|---|
| 01 | `v01_peta_eksperimen.png` | Apa struktur eksperimen ini? | `configs/matriks.yaml` |
| 02 | `v02_arsitektur_sistem.png` | Di mana Sumbu A bercabang dalam rantai agen? | diagram |
| 03 | `v03_heatmap_akurasi.png` | Formulasi mana konsisten, tugas mana sensitif? | `bench/analisis.json` |
| 04 | `v04_pareto_token_waktu.png` | Trade-off akurasi vs biaya token | `analisis.json` + `pendukung/token_bench.json` |
| 05 | `v05_geometri_rentang.png` | Seberapa dekat tiap formulasi ke embedding token nyata? | `probe/b7_probe_*.json` |
| 06 | `v06_geometri_vs_kinerja.png` | Apakah kedekatan geometris berasosiasi dgn kinerja? (n=5, deskriptif) | `pendukung/geometri_vs_kinerja.json` |
| 07 | `v07_pencarian_beta_moi.png` | Apakah β=1 masuk akal, atau kebetulan? | `probe/channel_capacity_*_moi_b*` |
| 08 | `v08_interpolasi_geometri.png` | Bentuk kurva sumbu `mix` — **tangga, bukan tanjakan** | `probe` + `pendukung/sumbu_mix.json` |
| 10 | `v10_akumulasi_kv_hop_kv.png` | Berapa konteks diwariskan tiap hop? | `pendukung/agent_trace_perhop.json` |
| 11 | `v11_upaya_perbaikan.png` | Berapa kali `construct` harus mengulang? | `frontend_*.json → repair_attempts` |
| 15 | `v15_cakupan_fungsi_dsl_kv.png` | Apakah `raw` kolaps ke kosakata fungsi yang sempit? | `frontend_*.json` (regex nama fungsi) |
| 17 | `v17_korupsi_cjk_posisi.png` | Di posisi mana aksara CJK muncul di jawaban? | `bench/bench_*_kv_s0.json` |
| 18 | `v18_panjang_repetisi_jawaban.png` | Apakah `raw` lebih panjang/berulang? | `bench/bench_*_kv_s0.json` |
| 19 | `v19_kanal_per_arm.png` | Apakah `kv` lossless karena vektor laten atau karena prompt warisan? | `probe/channel_capacity_*_m10.json` |
| 20 | `v20_kedalaman_ekspresi_kv.png` | Kompleksitas struktural ekspresi per formulasi | `frontend_*.json` |
| 21 | `v21_statistik_latent_stop.png` | Apakah early-stop menyala di produksi? | `pendukung/agent_trace_perhop.json` |
| 22 | `v22_kemiripan_semantik_kv.png` | Keragaman gagasan lewat embedding semantik | `W_in` + `frontend_*.json` |
| 23 | `v23_fidelitas_arah.png` | Fidelitas terhadap kalimat arah, per hop | `pendukung/faktor_perhop.json` |
| 24 | `v24_biaya_percobaan.png` | Percobaan construct+repair per JALAN | `pendukung/faktor_perhop.json` |

Catatan penting untuk v15 vs v22: keduanya mengukur keragaman tetapi memberi
jawaban berbeda. Secara **sintaksis** (v15, kosakata fungsi DSL) `raw` tampak
lebih sempit; secara **semantik** (v22, embedding) kelima formulasi nyaris
identik (0,84--0,85). Perbedaan itu bukan kontradiksi melainkan justru
mendukung tafsir disosiasi: gagasan `raw` sama kayanya, penulisannya yang
gagal.

---

## 3. 🔴 Figur kedaluwarsa — JANGAN dipakai bab hasil

| # | berkas | kenapa | penggantinya |
|---|---|---|---|
| 09 | `v09_funnel_fidelitas_kv.png` | tahap "evaluable" & "\|IC\|>0,02" dihitung dari `ic` A-share | Tabel keandalan + Tabel mutu (IDX) |
| 12 | `v12_matriks_fidelitas.png` | kolom \|IC\| A-share | Tabel uji formal (IDX) |
| 13 | `v13_distribusi_ic.png` | sebaran IC A-share | **i02** |
| 14 | `v14_holdout_vs_seleksi.png` | membaca `holdout_*.json` A-share | **i03** |
| 16 | `v16_pareto_faktor_kv.png` | sumbu \|IC\| A-share | **i04** (sumbu diganti laju lolos gate) |

Ambang `|IC| > 0,02` yang dipakai v09/v12 juga tidak lagi bermakna: pada
universe 37 emiten, ambang signifikansi justru **0,0208** pada jendela seleksi,
sehingga kriteria lama yang dirancang untuk 4.343 saham/hari berubah arti
sepenuhnya.

---

## 4. Sengaja TIDAK dibuat

- **Factor lineage / evolution tree.** Rancangan ini *single-pass*, tanpa loop
  evolusi (keputusan sadar, `DESAIN_EKSPERIMEN.md` §6). Tak ada
  hipotesis→mutasi→umpan balik untuk digambar; menggambarnya akan menyiratkan
  proses yang tak pernah berjalan. Padanan jujur yang benar-benar ada di data
  adalah **v24**.
- **Degradasi parse-rate per hop.** Hanya `construct` yang menulis JSON;
  `parsed_ok` milik `proposal` dan `innovate` SELALU `False` *menurut
  rancangan*, bukan karena kegagalan bertahap. **v10** memakai `kv_len`,
  metrik yang benar-benar terukur di ketiga hop.
- **`fidelity_pd`/`fidelity_dc`/`mech_drift`.** Satu-satunya implementasinya
  membaca format artefak lama yang sudah dihapus; medan itu tidak pernah terisi
  untuk data produksi mana pun. Dilaporkan apa adanya, bukan ditebak.

---

## 5. Tabel bab hasil (results/idx/tabel/*.tex)

Seluruhnya dibangkitkan `scripts/tabel_idx.py` dari **satu** berkas
`results/idx/analisis_idx.json`, sehingga angka antar-tabel tidak dapat saling
bertentangan.

| berkas | isi |
|---|---|
| `idx_keandalan.tex` | jalan, lolos gate, ekspresi, token, detik per sel |
| `idx_mutu.tex` | hidup / signifikan / rerata \|IC\| di dua jendela |
| `idx_backtest.tex` | Sharpe, imbal tahunan, max DD, perputaran, hit rate |
| `idx_uji.tex` | uji formal: Fisher (validitas) + Mann–Whitney (mutu sinyal) |
| `idx_bench.tex` | akurasi tiga tolok ukur penalaran umum |
| `idx_bench_uji.tex` | gradien nilai-p McNemar: ARC-C → GSM8K → HumanEval+ |
| `idx_stabilitas.tex` | berbalik tanda, bertahan signifikan, klaster sinyal, peluruhan |
| `idx_lantai.tex` | lantai acak (bila `lantai_acak.py` sudah dijalankan) |
| `idx_gatepasar.tex` | kesesuaian keputusan gate antar-pasar |

---

## 6. Palet warna — jangan diubah sepihak

```
raw=#e34948 (merah)  soft=#2a78d6 (biru)  gumbel=#1baf7a (aqua)
sample=#eda100 (kuning)  moi=#4a3aa7 (ungu)
```

Didefinisikan di `scripts/plot_readme_figures.py`, diwarisi apa adanya oleh
`scripts/visual_bab4.py` dan `scripts/visual_idx.py` — supaya pembaca yang
melihat README, figur A-share, dan figur IDX tidak perlu mempelajari tiga
legenda warna untuk lima formulasi yang sama.
