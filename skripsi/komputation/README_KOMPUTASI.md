# `komputation/` — kode dan hasil komputasi skripsi

Direktori ini adalah **salinan lengkap** basis kode eksperimen, disatukan ke
dalam repositori skripsi supaya naskah, kode, dan angka hidup dalam satu
riwayat versi. Satu `git clone` memberi seluruhnya.

## Yang ada di sini

```
backend/          paket kode
  llm/            SUMBU A — lima persamaan langkah laten (methods.py)
  mas/            rantai agen proposal -> innovate -> construct + gate
  dsl/            tata bahasa ekspresi faktor: parser, AST, pustaka fungsi
  gate/           kendali mutu ekspresi (struktural + eksekusi)
  eval/           penilaian: RankIC, backtest, holdout, statistik
  bench/          lengan penalaran umum (GSM8K, ARC-C, HumanEval+)
  factor/         orkestrasi lengan faktor
  hf_data_id/     panel pasar Bursa Efek Indonesia
scripts/          perkakas: unduh data, skor, analisis, tabel, figur
docs/             temuan bertahap (HASIL_TAHAP*.md, HASIL_IDX_LQ45.md)
results/          keluaran eksperimen yang dikutip naskah
reference/        kode rujukan LatentMAS & mixinputs (hanya-baca)
configs/          matriks sel eksperimen
```

## Yang SENGAJA tidak disalin

| tidak ada di sini | ukuran | alasan |
|---|---|---|
| `backend/hf_data/` | 854 MB | panel pasar A-share; penelitian berpindah ke IDX |
| `backend/data/qlib/` | 730 MB | data qlib A-share; hanya dipakai jalur LightGBM yang sudah dibuang |
| `.venv/` | 3,8 GB | lingkungan virtual; dibangun ulang dari `requirements.txt` |
| `results/lampiran_dibaca/` | 72 MB | bahan bacaan, bukan keluaran eksperimen |
| `results/*.tar.gz` | 47 MB | arsip snapshot; isinya sudah terwakili direktori di atas |
| `.git/` lama | 575 MB | riwayat berisi dump `factor_values.csv` yang tak dipakai satu angka pun |

## Cara membangun ulang seluruh angka bab hasil

Tanpa GPU. Tahap pembangkitan ekspresi sudah selesai dan keluarannya tersimpan
di `results/factor/frontend_*.json`; sisanya berjalan di CPU.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. panel pasar (unduh sekali; berkas hasilnya sudah disertakan)
python scripts/ambil_data_idx.py
python scripts/panel_indeks_idx.py --indeks lq45

# 2. penilaian dua jendela
bash scripts/skor_idx.sh

# 3. analisis tambahan
python scripts/lantai_acak.py --n 600
python scripts/uji_gate_lintas_pasar.py

# 4. agregasi, tabel, figur
python scripts/analisis_idx.py
python scripts/tabel_idx.py
python scripts/visual_idx.py

# 5. salin tabel & figur ke naskah
cp results/idx/tabel/*.tex   ../assets/tables/
cp results/visual_idx/*.png  ../assets/images/
```

## Variabel lingkungan yang mengendalikan pasar

Tanpa variabel ini, kode berperilaku persis seperti sebelum perpindahan pasar.

| variabel | arti |
|---|---|
| `LAB_PV_FILE` | panel harga-volume; namanya ikut ke nama berkas cache |
| `LAB_TRADING_DAYS` | konstanta anualisasi (241 IDX, 243 A-share) |
| `LAB_GATE_SAMPLE_START` / `_END` | jendela sampel *execution gate* |

## Hubungan dengan repositori `quantalatent`

Pengembangan kode berlanjut di repositori `quantalatent` yang terpisah.
Direktori ini adalah salinan pada saat naskah disusun, dan merupakan **versi
yang angkanya dikutip skripsi**. Bila keduanya berbeda, yang berlaku untuk
naskah adalah yang ada di sini.
