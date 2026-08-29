# Profil waktu fungsi DSL

Korpus penuh `eval/ic.py` (~4.370 saham x 243 hari OOS) — beban yang
sama dengan skoring produksi. Batas 120 dtk per fungsi; anggaran
skoring produksi 90 dtk, jadi apa pun yang mendekati/melewatinya akan
hilang sebagai `timeout>90s` tergantung beban mesin.

| fungsi | detik | status |
|---|---:|---|
| TS_SKEW | 120.0 | >120s |
| TS_KURT | 120.0 | >120s |
| REGRESI | 80.69 | ok |
| REGBETA | 72.56 | ok |
| TS_MAD | 69.61 | ok |
| TS_CORR | 13.68 | ok |
| TS_COVARIANCE | 13.63 | ok |
| WMA | 12.41 | ok |
| DECAYLINEAR | 7.0 | ok |
| TS_ZSCORE | 4.22 | ok |
| LOWDAY | 3.9 | ok |
| HIGHDAY | 3.86 | ok |
| TS_ARGMAX | 3.61 | ok |
| TS_ARGMIN | 3.6 | ok |
| TS_MEDIAN | 3.17 | ok |
| TS_QUANTILE | 3.13 | ok |
| TS_RANK | 3.11 | ok |
| TS_PCTCHANGE | 2.49 | ok |
| TS_STD | 2.07 | ok |
| TS_VAR | 2.06 | ok |
| TS_MIN | 2.04 | ok |
| TS_SUM | 2.03 | ok |
| TS_MEAN | 2.02 | ok |
| TS_MAX | 1.99 | ok |
| DELTA | 1.7 | ok |
| DELAY | 1.69 | ok |
| ABS | 1.58 | ok |
| RANK | 1.25 | ok |
| SIGN | 0.01 | ok |
| SEQUENCE | 0.01 | KeyError |
| LOG | 0.0 | ok |
