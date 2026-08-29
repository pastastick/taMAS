# Data pendukung (regenerasi: `python scripts/kumpulkan_pendukung.py`)

## Pemakaian fungsi DSL (279 ekspresi)

| fungsi | dipakai di | % |
|---|---:|---:|
| RANK | 111 | 39.8 |
| TS_PCTCHANGE | 93 | 33.3 |
| TS_ZSCORE | 69 | 24.7 |
| TS_ARGMAX | 46 | 16.5 |
| TS_QUANTILE | 36 | 12.9 |
| REGRESI | 34 | 12.2 |
| TS_MAD | 34 | 12.2 |
| TS_SKEW | 29 | 10.4 |
| TS_ARGMIN | 24 | 8.6 |
| TS_CORR | 22 | 7.9 |
| TS_STD | 18 | 6.5 |
| TS_COVARIANCE | 15 | 5.4 |
| SEQUENCE | 12 | 4.3 |
| DECAYLINEAR | 11 | 3.9 |
| TS_KURT | 10 | 3.6 |
| TS_MEDIAN | 7 | 2.5 |
| TS_SUM | 7 | 2.5 |
| TS_MEAN | 6 | 2.2 |
| DELAY | 6 | 2.2 |
| DELTA | 5 | 1.8 |

## Efektivitas gate (kebocoran = lolos gate tapi gagal dievaluasi)

| sumber | ekspresi | lolos gate | evaluable | BOCOR |
|---|---:|---:|---:|---:|
| arsip_gate_mati_2026-08-10 | 105 | 7 | 3 | 4 |
| matriks | 174 | 123 | 124 | 13 |

## Alasan penolakan gate

| alasan | n |
|---|---:|
| arity | 15 |
| regulator evaluate failed | 15 |
| regulator reject | 8 |
| unparsable expression | 7 |
| ParseException | 3 |
| execution | 3 |
| semantics | 1 |
| variable | 1 |

## Kenapa ekspresi gagal dievaluasi

| error | n |
|---|---:|
| timeout>900s | 17 |
| NameError | 16 |
| TypeError | 11 |
| ParseException | 9 |
| Exception | 1 |
| SyntaxError | 1 |
| ValueError | 1 |

## Waktu sel lengan faktor (biaya teks vs laten)

| sel | comm | metode | n_run | rerata (dtk) |
|---|---|---|---:|---:|
| frontend_kv_gumbel | kv | gumbel | 6 | 66.0 |
| frontend_kv_moi | kv | moi | 6 | 55.7 |
| frontend_kv_raw | kv | raw | 6 | 108.4 |
| frontend_text | text | raw | 6 | 218.9 |
| frontend_kv_and_text_gumbel | kv_and_text | gumbel | 6 | 139.9 |
| frontend_kv_and_text_moi | kv_and_text | moi | 6 | 211.7 |
| frontend_kv_and_text_raw | kv_and_text | raw | 6 | 224.9 |
| frontend_kv_and_text_sample | kv_and_text | sample | 6 | 209.6 |
| frontend_kv_and_text_soft | kv_and_text | soft | 6 | 200.2 |
| frontend_kv_sample | kv | sample | 6 | 37.4 |
| frontend_kv_soft | kv | soft | 6 | 65.4 |

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
