# Sumbu C (`mix`) — geometri vs kinerja, formulasi DIPEGANG TETAP

Regenerasi: `python scripts/analisis_mix.py`

Titik ujung memakai sel `kv_raw`/`kv_soft`: z(alpha=0) identik dengan
z_raw dan z(alpha=1) dengan z_soft, jadi menjalankannya lagi sebagai sel
sendiri hanya akan menghasilkan angka yang sama.

| alpha | cos terukur | lolos gate (faktor) | ekspr/jalan | mean abs IC | HumanEval+ | jwb ber-CJK | token/jalan |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00 = raw | 0.3120 | 50% (10/20) | 2.95 | 0.01249 | 0.42 | 1 | 803.4 |
| 0.25  | 0.4519 | 55% (11/20) | 3.4 | 0.01825 | 0.28 | 4 | 733.5 |
| 0.50  | 0.7197 | 100% (20/20) | 5.8 | 0.01445 | 0.68 | 0 | 511.4 |
| 0.75  | 0.9244 | 100% (20/20) | 5.95 | 0.01468 | 0.69 | 0 | 502.5 |
| 1.00 = soft | 0.9269 | 100% (20/20) | 5.55 | 0.01606 | 0.69 | 0 | 480.9 |

## Bentuk kurva

### laju_lolos_gate
- nilai: [0.5, 0.55, 1.0, 1.0, 1.0]
- selisih antar titik: [0.05, 0.45, 0.0, 0.0]
- **monoton: YA**
- Spearman vs alpha 0.8944, vs cos terukur 0.8944
- lompatan terbesar 0.25→0.5 (cos 0.4519→0.7197): +0.45 = 90% dari seluruh rentang

### akurasi_humanevalplus
- nilai: [0.42, 0.28, 0.68, 0.69, 0.69]
- selisih antar titik: [-0.14, 0.4, 0.01, 0.0]
- **monoton: TIDAK**
- Spearman vs alpha 0.8721, vs cos terukur 0.8721
- lompatan terbesar 0.25→0.5 (cos 0.4519→0.7197): +0.4 = 98% dari seluruh rentang

### ic_mean_abs
- nilai: [0.01249, 0.01825, 0.01445, 0.01468, 0.01606]
- selisih antar titik: [0.0058, -0.0038, 0.0002, 0.0014]
- **monoton: TIDAK**
- Spearman vs alpha 0.4, vs cos terukur 0.4
- lompatan terbesar 0.0→0.25 (cos 0.312→0.4519): +0.0058 = 101% dari seluruh rentang

### ekspresi_per_jalan
- nilai: [2.95, 3.4, 5.8, 5.95, 5.55]
- selisih antar titik: [0.45, 2.4, 0.15, -0.4]
- **monoton: TIDAK**
- Spearman vs alpha 0.7, vs cos terukur 0.7
- lompatan terbesar 0.25→0.5 (cos 0.4519→0.7197): +2.4 = 80% dari seluruh rentang

### korupsi_cjk_humanevalplus
- nilai: [1, 4, 0, 0, 0]
- selisih antar titik: [3, -4, 0, 0]
- **monoton: TIDAK**
- Spearman vs alpha -0.7826, vs cos terukur -0.7826
- lompatan terbesar 0.25→0.5 (cos 0.4519→0.7197): -4 = 100% dari seluruh rentang

## Uji berpasangan lolos gate (McNemar eksak, dipasangkan lewat arah x seed)

| titik | vs raw (alpha=0) | vs soft (alpha=1) |
|---|---|---|
| alpha=0.00 | — | delta=-0.50, b01=0/b10=10, p=0.001953 * |
| alpha=0.25 | delta=+0.05, b01=5/b10=4, p=1 | delta=-0.45, b01=0/b10=9, p=0.003906 * |
| alpha=0.50 | delta=+0.50, b01=10/b10=0, p=0.001953 * | delta=+0.00, b01=0/b10=0, p=1 |
| alpha=0.75 | delta=+0.50, b01=10/b10=0, p=0.001953 * | delta=+0.00, b01=0/b10=0, p=1 |
| alpha=1.00 | delta=+0.50, b01=10/b10=0, p=0.001953 * | — |

`*` = signifikan pada alpha 0,05. Titik falsifikasi: alpha=0,75 secara
geometri sudah praktis sama dengan `soft` (cos 0,9244 vs 0,9269); kalau
geometri memang penjelasnya, kolom kanannya harus TIDAK signifikan.

## Uji berpasangan HumanEval+ (McNemar eksak, dipasangkan lewat indeks soal)

n=100 soal yang sama di kelima sel, seed sampel sama.

| titik | vs raw (alpha=0) | vs soft (alpha=1) |
|---|---|---|
| alpha=0.00 | — | delta=-0.27, b01=4/b10=31, p=3.465e-06 * |
| alpha=0.25 | delta=-0.14, b01=7/b10=21, p=0.01254 * | delta=-0.41, b01=4/b10=45, p=8.225e-10 * |
| alpha=0.50 | delta=+0.26, b01=28/b10=2, p=8.68e-07 * | delta=-0.01, b01=6/b10=7, p=1 |
| alpha=0.75 | delta=+0.27, b01=32/b10=5, p=7.428e-06 * | delta=+0.00, b01=6/b10=6, p=1 |
| alpha=1.00 | delta=+0.27, b01=31/b10=4, p=3.465e-06 * | — |

