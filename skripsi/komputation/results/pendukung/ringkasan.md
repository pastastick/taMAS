# Data pendukung (regenerasi: `python scripts/kumpulkan_pendukung.py`)

## Pemakaian fungsi DSL (2066 ekspresi)

| fungsi | dipakai di | % |
|---|---:|---:|
| RANK | 807 | 39.1 |
| TS_PCTCHANGE | 676 | 32.7 |
| TS_ZSCORE | 509 | 24.6 |
| TS_ARGMAX | 357 | 17.3 |
| TS_MAD | 311 | 15.1 |
| TS_QUANTILE | 294 | 14.2 |
| TS_ARGMIN | 245 | 11.9 |
| TS_CORR | 236 | 11.4 |
| REGRESI | 226 | 10.9 |
| TS_STD | 195 | 9.4 |
| TS_SKEW | 168 | 8.1 |
| DECAYLINEAR | 94 | 4.5 |
| SEQUENCE | 85 | 4.1 |
| TS_KURT | 82 | 4.0 |
| TS_MEAN | 76 | 3.7 |
| TS_MEDIAN | 63 | 3.0 |
| TS_COVARIANCE | 57 | 2.8 |
| DELAY | 48 | 2.3 |
| TS_RANK | 33 | 1.6 |
| TS_SUM | 31 | 1.5 |

## Efektivitas gate (kebocoran = lolos gate tapi gagal dievaluasi)

| sumber | ekspresi | lolos gate | evaluable | BOCOR |
|---|---:|---:|---:|---:|
| arsip_faktor_6jalan_2026-08-10 | 174 | 123 | 124 | 13 |
| arsip_gate_mati_2026-08-10 | 105 | 7 | 3 | 4 |
| arsip_pecahan_gabung_2026-08-28 | 359 | 268 | 275 | 36 |
| matriks | 1428 | 1067 | 1116 | 101 |

## Alasan penolakan gate

| alasan | n |
|---|---:|
| arity | 208 |
| regulator evaluate failed | 93 |
| semantics | 63 |
| regulator reject | 37 |
| unparsable expression | 35 |
| execution | 33 |
| variable | 5 |
| ParseException | 3 |

## Kenapa ekspresi gagal dievaluasi

| error | n |
|---|---:|
| timeout>900s | 161 |
| NameError | 124 |
| TypeError | 73 |
| ParseException | 32 |
| ValueError | 16 |
| empty after dropna/OOS | 12 |
| SyntaxError | 5 |
| LinAlgError | 5 |
| Exception | 4 |
| RecursionError | 2 |
| AssertionError | 1 |

## Waktu sel lengan faktor (biaya teks vs laten)

| sel | comm | metode | n_run | rerata (dtk) |
|---|---|---|---:|---:|
| frontend_kv_and_text_gumbel | kv_and_text | gumbel | 20 | 203.5 |
| frontend_kv_and_text_moi | kv_and_text | moi | 20 | 135.6 |
| frontend_kv_and_text_raw | kv_and_text | raw | 20 | 295.6 |
| frontend_kv_and_text_sample | kv_and_text | sample | 20 | 197.8 |
| frontend_kv_and_text_soft | kv_and_text | soft | 20 | 185.5 |
| frontend_kv_gumbel | kv | gumbel | 20 | 58.0 |
| frontend_kv_moi | kv | moi | 20 | 55.6 |
| frontend_kv_raw | kv | raw | 20 | 77.3 |
| frontend_kv_sample | kv | sample | 20 | 48.8 |
| frontend_kv_soft | kv | soft | 20 | 51.5 |
| frontend_text | text | raw | 20 | 181.4 |
| frontend_kv_and_text_gumbel.sebelum_gabung_20260828_032843 | kv_and_text | gumbel | 8 | 153.6 |
| frontend_kv_and_text_gumbel_s234 | kv_and_text | gumbel | 12 | 236.8 |
| frontend_kv_and_text_moi.sebelum_gabung_20260828_032843 | kv_and_text | moi | 8 | 102.1 |
| frontend_kv_and_text_moi_s234 | kv_and_text | moi | 12 | 158.0 |
| frontend_kv_and_text_sample.sebelum_gabung_20260828_032843 | kv_and_text | sample | 8 | 139.3 |
| frontend_kv_and_text_sample_s234 | kv_and_text | sample | 12 | 236.8 |
| frontend_kv_mix_a025 | kv | mix | 20 | 63.2 |
| frontend_kv_mix_a05 | kv | mix | 20 | 61.4 |
| frontend_kv_mix_a075 | kv | mix | 20 | 72.2 |

## Lengan benchmark

| sel | tugas | comm | metode | n | akurasi | format | waktu (dtk) |
|---|---|---|---|---:|---:|---:|---:|
| bench_arc_challenge_gumbel_kv_and_text_lampiran | arc_challenge | kv_and_text | gumbel | 5 | 0.8 | 1.0 | 164.9 |
| bench_arc_challenge_gumbel_kv_s0 | arc_challenge | kv | gumbel | 100 | 0.91 | 0.98 | 1591.7 |
| bench_arc_challenge_moi_kv_and_text_lampiran | arc_challenge | kv_and_text | moi | 5 | 0.8 | 1.0 | 162.5 |
| bench_arc_challenge_moi_kv_s0 | arc_challenge | kv | moi | 100 | 0.91 | 0.98 | 1531.5 |
| bench_arc_challenge_raw_baseline_s0 | arc_challenge | kv | raw | 100 | 0.91 | 1.0 | 2027.8 |
| bench_arc_challenge_raw_kv_and_text_lampiran | arc_challenge | kv_and_text | raw | 5 | 0.8 | 1.0 | 214.0 |
| bench_arc_challenge_raw_kv_s0 | arc_challenge | kv | raw | 100 | 0.89 | 1.0 | 2057.9 |
| bench_arc_challenge_raw_text_s0 | arc_challenge | text | raw | 100 | 0.93 | 1.0 | 7595.7 |
| bench_arc_challenge_sample_kv_and_text_lampiran | arc_challenge | kv_and_text | sample | 5 | 0.8 | 1.0 | 169.0 |
| bench_arc_challenge_sample_kv_s0 | arc_challenge | kv | sample | 100 | 0.94 | 1.0 | 1325.9 |
| bench_arc_challenge_soft_kv_and_text_lampiran | arc_challenge | kv_and_text | soft | 5 | 0.8 | 1.0 | 157.3 |
| bench_arc_challenge_soft_kv_s0 | arc_challenge | kv | soft | 100 | 0.94 | 1.0 | 1400.1 |
| bench_gsm8k_gumbel_kv_and_text_lampiran | gsm8k | kv_and_text | gumbel | 5 | 0.4 | 0.8 | 283.6 |
| bench_gsm8k_gumbel_kv_s0 | gsm8k | kv | gumbel | 100 | 0.91 | 1.0 | 2814.5 |
| bench_gsm8k_moi_kv_and_text_lampiran | gsm8k | kv_and_text | moi | 5 | 0.6 | 1.0 | 296.1 |
| bench_gsm8k_moi_kv_s0 | gsm8k | kv | moi | 100 | 0.92 | 1.0 | 1585.4 |
| bench_gsm8k_raw_baseline_s0 | gsm8k | kv | raw | 100 | 0.92 | 1.0 | 1569.9 |
| bench_gsm8k_raw_kv_and_text_lampiran | gsm8k | kv_and_text | raw | 5 | 0.8 | 1.0 | 382.5 |
| bench_gsm8k_raw_kv_s0 | gsm8k | kv | raw | 100 | 0.83 | 0.98 | 1971.8 |
| bench_gsm8k_raw_text_s0 | gsm8k | text | raw | 100 | 0.9 | 1.0 | 4919.1 |
| bench_gsm8k_sample_kv_and_text_lampiran | gsm8k | kv_and_text | sample | 5 | 0.8 | 1.0 | 283.8 |
| bench_gsm8k_sample_kv_s0 | gsm8k | kv | sample | 100 | 0.86 | 0.97 | 1645.9 |
| bench_gsm8k_soft_kv_and_text_lampiran | gsm8k | kv_and_text | soft | 5 | 0.8 | 1.0 | 277.8 |
| bench_gsm8k_soft_kv_s0 | gsm8k | kv | soft | 100 | 0.88 | 1.0 | 1447.3 |
| bench_humanevalplus_gumbel_kv_and_text_lampiran | humanevalplus | kv_and_text | gumbel | 5 | 1.0 | 1.0 | 175.6 |
| bench_humanevalplus_gumbel_kv_s0 | humanevalplus | kv | gumbel | 100 | 0.71 | 0.97 | 1492.2 |
| bench_humanevalplus_mix_kv_s0_a025 | humanevalplus | kv | mix | 100 | 0.28 | 0.99 | 2456.2 |
| bench_humanevalplus_mix_kv_s0_a05 | humanevalplus | kv | mix | 100 | 0.68 | 0.99 | 2146.6 |
| bench_humanevalplus_mix_kv_s0_a075 | humanevalplus | kv | mix | 100 | 0.69 | 0.98 | 1619.7 |
| bench_humanevalplus_moi_kv_and_text_lampiran | humanevalplus | kv_and_text | moi | 5 | 1.0 | 1.0 | 197.6 |
| bench_humanevalplus_moi_kv_s0 | humanevalplus | kv | moi | 100 | 0.72 | 1.0 | 1407.6 |
| bench_humanevalplus_raw_baseline_s0 | humanevalplus | kv | raw | 100 | 0.72 | 1.0 | 1085.8 |
| bench_humanevalplus_raw_kv_and_text_lampiran | humanevalplus | kv_and_text | raw | 5 | 0.2 | 1.0 | 395.3 |
| bench_humanevalplus_raw_kv_s0 | humanevalplus | kv | raw | 100 | 0.42 | 1.0 | 1782.8 |
| bench_humanevalplus_raw_text_s0 | humanevalplus | text | raw | 100 | 0.76 | 1.0 | 6925.1 |
| bench_humanevalplus_sample_kv_and_text_lampiran | humanevalplus | kv_and_text | sample | 5 | 1.0 | 1.0 | 159.5 |
| bench_humanevalplus_sample_kv_s0 | humanevalplus | kv | sample | 100 | 0.74 | 1.0 | 1241.1 |
| bench_humanevalplus_soft_kv_and_text_lampiran | humanevalplus | kv_and_text | soft | 5 | 0.8 | 1.0 | 154.4 |
| bench_humanevalplus_soft_kv_s0 | humanevalplus | kv | soft | 100 | 0.69 | 0.96 | 1141.5 |
