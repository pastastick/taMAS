# Results

Every number reported in the thesis is traceable to a file in this directory,
which is why `results/` is tracked in git. Large or regenerable artifacts are
excluded by [`.gitignore`](.gitignore), which documents how to rebuild each one.

## Layout

| path | contents |
|---|---|
| `bench/bench_<task>_<mode>_<medium>_s0.json` | one benchmark cell: `_meta` (full run config), `summary` (n, accuracy, format rate), `results` (per-question outcome) |
| `bench/analisis.json` | paired analysis over all cells: exact McNemar and bootstrap CIs for every pair, grouped by task with a question fingerprint. Multiplicity correction (Holm, Benjamini–Hochberg), Cochran $Q$, and the family-vs-`raw` contrasts are computed downstream in the thesis analysis scripts, not here |
| `factor/` | alpha-factor arm: generated DSL expressions, gate verdicts, IC and backtest scores |
| `probe/realign_probe_*.json` | realignment matrix geometry per backbone (including whether embeddings are tied) |
| `probe/b7_probe_*.json` | latent-step geometry: cosine to the nearest token embedding, per formulation |
| `probe/channel_capacity_*.json` | symbolic channel capacity: recall of a function-name payload at `m` latent steps |
| `pendukung/` | cross-cell aggregates: token usage, token corruption, gate effectiveness, DSL function usage |
| `arsip_gate_mati_2026-08-10/` | expressions produced while the quality gate was misconfigured, kept for the leakage analysis |

## Reading a cell name

```
bench_humanevalplus_gumbel_kv_s0
      └─ task    └─ latent mode └─ medium └─ sample seed
```

Cells ending in `_baseline_s0` are the single-agent reference. Cells ending in
`_lampiran` ran on 5 questions as transcript suppliers for the thesis appendix,
on a *different* question subsample — they are excluded from every table and
test, and analysis scripts filter them out by `n`.

## Regenerating the aggregates

```bash
python backend/bench/compare.py --out results/bench/analisis.json
python scripts/hitung_token.py --out results/pendukung/token_bench.json
python scripts/kumpulkan_pendukung.py
PYTHONPATH=backend python backend/eval/rescore_all.py
```
