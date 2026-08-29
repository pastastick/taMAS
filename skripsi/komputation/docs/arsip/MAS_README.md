# latent_mas — LatentMAS × QuantaAlpha (rewrite)

Branch: `feat/latentmas-rework`. Penulisan ulang pipeline alpha-mining dengan
kolaborasi laten murni (LatentMAS) + evolusi trajectory (QuantaAlpha).

## Ide inti

Front-end **sequential** (semua kv_only kecuali judger):

```
seed → proposal → construct → consistency → judger ──→ HYPOTHESIS + EXPRESSION
        (kv)       (kv)         (kv)          (kv+text)        │
                                  └ kv_consist (baseline)      ▼
                                                          quality gate (AST/arity)
                                                               │ gagal
                                                               ▼
                                                          repair ×N (dari kv_consist)
```

Evolution:

- **Mutation** = sequential 2-step: `mutation_reflection` (diagnosa step gagal) →
  `mutation_judger` (tulis ulang step itu). Mengikuti lokalisasi-revisi QuantaAlpha.
- **Crossover** = hierarchical: `kv_concat` beberapa KV parent → `crossover_judger`.
  Ini pemetaan langsung hierarchical-LatentMAS (concat working memory) ke
  rekombinasi trajectory QuantaAlpha.

## Distribusi KV — aturan emas

`LocalLLMBackend.run()` **memutasi `past_key_values` in-place**. Maka KV yang
dibaca > 1 konsumen WAJIB di-`kv_ops.kv_deepcopy` dulu. Lihat
[`pipeline.py`](pipeline.py) — judger, tiap repair attempt, dan feedback semuanya
menerima clone dari `kv_consist`, bukan objek yang sama.

| Agent | mode | KV input | catatan |
|---|---|---|---|
| proposal | kv_only | seed/None | extend in-place (linear) |
| construct | kv_only | proposal kv | extend in-place |
| consistency | kv_only | construct kv | → `kv_consist` (baseline) |
| judger | kv_and_text | deepcopy(kv_consist) | output final |
| repair | kv_and_text | deepcopy(kv_consist) | baseline sama tiap attempt |
| feedback | kv_and_text | deepcopy(kv_consist) | anti-bias (bukan kv_judger) |
| mutation_reflection | kv_and_text | deepcopy(kv_feedback) | diagnosa |
| mutation_judger | kv_and_text | kv_reflect | lanjut sequential |
| crossover_judger | kv_and_text | kv_concat(parents) | hierarchical |

## Menjalankan satu agent (debug prompt / KV)

```bash
# proposal → simpan KV
python experiments/run_agent.py proposal \
    --var direction="overnight gap reversal" --save-kv runs/kv_prop.pt

# construct dari KV proposal → simpan
python experiments/run_agent.py construct --load-kv runs/kv_prop.pt \
    --save-kv runs/kv_con.pt

# judger dari KV consistency → lihat teks + parse
python experiments/run_agent.py judger --load-kv runs/kv_consist.pt

# probe: apa yang sebenarnya ada di sebuah KV?
python experiments/inspect_kv.py runs/kv_consist.pt
```

`--show-prompt` mencetak prompt yang dirender. Edit prompt di
[`prompts.yaml`](prompts.yaml) — tidak perlu sentuh kode.

## Logging

`runlog.RunLogger`: console tenang (default hanya WARNING+), detail penuh ke
`latent_runs/<timestamp>/{run.log,events.jsonl,summary.json}`. Timing untuk
SEMUA step (termasuk backtest) lewat `with rl.step("nama"): ...`.
Set `LATENTMAS_CONSOLE_LEVEL=INFO` untuk verbose sementara.

## Status

Selesai: kv_ops, runlog, agent base + prompts, pipeline (front-end + evolution),
harness standalone. **Belum disambung** ke `pipeline/loop.py` lama & backtest Qlib —
itu langkah migrasi berikutnya (sengaja dipisah agar repo tidak rusak).
```
