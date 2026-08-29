# Mutu ekspresi per sel (Level 5) — 14/14 sel sudah diskor

Regenerasi: `python scripts/analisis_mutu_ic.py`

DESKRIPTIF. Uji formal lengan faktor tetap satu: keluarga R vs `raw` pada
laju lolos gate (unit = jalan). Di sini unitnya ekspresi, dan ekspresi dari
jalan yang sama tidak independen.

| sel | medium | metode | ekspresi | ber-IC | mean abs IC | median | p90 | maks | t>=2 | mean abs Sharpe |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kv_and_text_gumbel | kv_and_text | gumbel | 120 | 78% | 0.01398 | 0.01331 | 0.02483 | 0.04937 | 60% | 0.9299 |
| kv_and_text_moi | kv_and_text | moi | 120 | 78% | 0.01433 | 0.01208 | 0.02541 | 0.04818 | 62% | 0.8774 |
| kv_and_text_raw | kv_and_text | raw | 29 | 45% | 0.01367 | 0.0133 | 0.02148 | 0.02981 | 62% | 0.6995 |
| kv_and_text_sample | kv_and_text | sample | 119 | 74% | 0.01343 | 0.01212 | 0.0238 | 0.04937 | 58% | 0.9378 |
| kv_and_text_soft | kv_and_text | soft | 120 | 78% | 0.01475 | 0.0146 | 0.02606 | 0.04205 | 61% | 0.9114 |
| kv_gumbel | kv | gumbel | 112 | 85% | 0.01566 | 0.01163 | 0.03939 | 0.04937 | 67% | 1.0337 |
| kv_mix_a025 | kv | mix | 68 | 78% | 0.01825 | 0.01594 | 0.03939 | 0.05098 | 74% | 1.16 |
| kv_mix_a05 | kv | mix | 116 | 83% | 0.01445 | 0.0111 | 0.03499 | 0.04937 | 54% | 0.9438 |
| kv_mix_a075 | kv | mix | 119 | 85% | 0.01468 | 0.01094 | 0.03589 | 0.04937 | 54% | 0.9956 |
| kv_moi | kv | moi | 106 | 84% | 0.01607 | 0.01165 | 0.03999 | 0.04937 | 55% | 0.9896 |
| kv_raw | kv | raw | 59 | 44% | 0.01249 | 0.01017 | 0.02397 | 0.03253 | 65% | 0.9763 |
| kv_sample | kv | sample | 109 | 87% | 0.01625 | 0.01196 | 0.04143 | 0.04937 | 60% | 0.8902 |
| kv_soft | kv | soft | 111 | 87% | 0.01606 | 0.0145 | 0.03253 | 0.04937 | 63% | 1.1141 |
| text | text | raw | 120 | 68% | 0.0146 | 0.0115 | 0.02781 | 0.05089 | 59% | 0.9425 |

## Keluarga R vs `raw` pada |IC| (deskriptif)

### medium `kv`
- R: 376 ekspresi, mean |IC| 0.01601, median 0.01198
- raw: 26 ekspresi, mean |IC| 0.01249, median 0.01017
- rasio mean R/raw: **1.282×**
- Mann-Whitney U=5386, p=0.3857 (DESKRIPTIF), rank-biserial 0.1018

### medium `kv_and_text`
- R: 369 ekspresi, mean |IC| 0.01413, median 0.01312
- raw: 13 ekspresi, mean |IC| 0.01367, median 0.0133
- rasio mean R/raw: **1.034×**
- Mann-Whitney U=2388, p=0.9806 (DESKRIPTIF), rank-biserial -0.0042

