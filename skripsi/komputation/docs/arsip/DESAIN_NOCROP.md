# QuantaLatent — Production Pipeline (branch `prod/quantalatent-v1`)

> Status: **PLAN — belum diimplementasi.** Dokumen ini untuk direview dulu.
> Keputusan kunci yang sudah diambil:
> 1. Transfer front-end = **NO-CROP** (token jawaban asli tetap di KV; murni laten/KV, tanpa injeksi teks).
> 2. Bangun bertahap: branch + design ini dulu → review → implementasi penuh.

Branch ini adalah pipeline **produksi** yang bersih, terpisah dari `try/promptbench`
(eksperimen) dan dari harness fan-out `runners/v4_eval.py`. Tujuannya menjalankan
loop evolusi alpha-mining nyata, bukan benchmarking medium.

---

## 1. Diagnosis yang melandasi (ringkas)

Fenomena: hipotesis **bermutasi di tiap hop** `proposal→design→construct`, padahal
`feedback→mutation` konsisten (mutation menyebut ulang weakness breakdown nyaris
verbatim).

**Akar:** di `kv_and_text`, jawaban yang di-generate sebuah agent **dibuang dari
KV sebelum di-chain** ke agent berikut:

- `latent_pass(add_generation_prompt=False)` → KV = `[KV-hulu]+[system]+[user]+[N laten]`,
  panjang disimpan `_latent_kv_len` ([client.py:1433](../backend/llm/client.py#L1433)).
- `generate_from_kv(...)` → token jawaban ter-append in-place ([client.py:1457](../backend/llm/client.py#L1457)).
- `kv.crop(_latent_kv_len)` → **token jawaban dihapus lagi** ([client.py:1464](../backend/llm/client.py#L1464)).

Akibatnya downstream hanya mewarisi `[system+user]` hulu + ~N vektor laten (lossy);
**`HYPOTHESIS:` proposal tidak pernah masuk KV design** → design mengarang ulang.

**Kenapa feedback→mutation selamat:** sumber-kebenaran feedback ada di **USER
prompt** (factor_block + backtest) yang **tidak pernah di-crop**; mutation membaca
ulang data yang sama dan menurunkan kelemahan yang sama (konvergensi, bukan recall).
Sumber-kebenaran proposal/design ada di **output mereka** yang justru di-crop.

> Aturan inti: **konten di USER prompt selamat di KV; konten di GENERATED output
> dibuang.** Feedback menaruh payload di user prompt; proposal/design menaruh
> payload di output.

Faktor sekunder (diperbaiki di prompts produksi, bukan akar):
- `design` Step 1 "**Restate** the mechanism" mengundang parafrase ([redesign_v4.yaml:258](../try/promptbench/variants/_authored/redesign_v4.yaml#L258)).
- `construct` system prompt me-reward divergensi ("Maximize exploration", "originality and diversity") ([redesign_v4.yaml:300-313](../try/promptbench/variants/_authored/redesign_v4.yaml#L300-L313)).
- Korupsi decode di hop dalam (token sampah `,T 10`) — isu realigner/long-context, **terpisah** dari drift (lihat §6).

---

## 2. Keputusan fix: NO-CROP front-end

Alih-alih membuang jawaban, **biarkan token jawaban tetap di KV** sehingga agent
berikut membaca output asli agent sebelumnya dari KV (bukan cuma vektor laten).
Latent steps tetap jalan → kontribusi "silent thinking" LatentMAS tetap ada.
Tidak ada injeksi teks ke prompt → handoff tetap **murni via KV** (sesuai pilihan).

### 2.1 Perubahan kode (proposal, belum diterapkan)

`crop` dibuat **kondisional**, default tetap crop (backward-compatible dgn eksperimen):

- `AgentSpec` + field baru `keep_answer_in_kv: bool = False` ([agent.py](../backend/latent_mas/agent.py)).
- Thread flag: `LatentAgent.run` → `build_messages_and_run(..., crop_after_generate=not keep_answer_in_kv)` → `run()` ([client.py:1268](../backend/llm/client.py#L1268)).
- Di blok `kv_and_text`, bungkus crop:
  ```python
  if crop_after_generate:
      try: kv.crop(_latent_kv_len)
      except AttributeError: pass
  ```

### 2.2 Detail chat-template (PENTING untuk no-crop)

Saat tidak di-crop, KV jadi `...[user]+[N laten]+[assistant_prefix]+[answer]` lalu
agent berikut meng-append `[<|im_start|>system ...]`. Turn asisten **belum ditutup
`<|im_end|>`** → model melihat asisten menggantung lalu system baru. Solusi: pada
path no-crop, **append satu token `<|im_end|>` (+`\n`) ke KV** setelah generate,
agar turn tertutup rapi sebelum agent berikut menambah turn baru. Ini bagian dari
implementasi `transfer.py` (lihat §4) dan akan dites di dry-run.

### 2.3 Agent mana yang no-crop?

No-crop dipakai pada setiap hop yang **output-nya = payload hop berikut**:
`proposal`, `design` (front-end), dan di loop evolusi `feedback`, `mutation`,
`crossover` (agar direction/diagnosis asli terbaca, bukan cuma laten).
`construct` = terminal → crop tidak relevan. Default produksi: **no-crop di semua
non-terminal**, dengan opsi mematikan per-agent bila terbukti memicu contamination
(lihat §6).

---

## 3. Arsitektur pipeline (loop produksi)

Topologi yang diminta (beda signifikan hanya di front-end):

```
command(user) ─▶ proposal ─▶ design ─▶ construct ─▶ runner{gate, repair, backtest}
                    ▲                                      │
                    │                                      ▼
              mutation / crossover ◀───────────────────  feedback
                    ▲                                      │
                    └──────────── (DIRECTION) ◀────────────┘   (loop n generasi)
```

Per hop (medium = KV, semua agent decode untuk audit + no-crop default):

| Hop | transfer | decode | catatan |
|---|---|---|---|
| command→proposal | seed direction (gen-0) / chain (gen-n) | ya | gen-0 pakai market_context |
| proposal→design | **chain (no-crop)** | ya | design baca HYPOTHESIS asli dari KV |
| design→construct | **chain (no-crop)** | ya (terminal) | construct baca palette asli dari KV |
| construct→runner | — | — | gate/repair/backtest (deterministik, non-LLM) |
| runner→feedback | teks (backtest+factor) | ya | entry loop; payload di USER prompt |
| feedback→mutation | chain (no-crop) | ya | exploitation, 1 parent |
| feedback×k→crossover | **concat (no-crop)** | ya | exploration, ≥2 parent (`kv_concat`) |
| mutation/crossover→proposal | chain (no-crop) | ya | DIRECTION untuk generasi berikut |

KV ops reuse: `kv_deepcopy` (isolasi sebelum distribusi), `kv_concat` (crossover,
LatentMAS Eq.4) dari [kv_ops.py](../backend/latent_mas/kv_ops.py).

---

## 4. Struktur folder & tanggung jawab modul

```
prod/
  DESIGN.md          ← dokumen ini
  README.md          ← orientasi singkat
  config.py          ← config terpusat (model, latent_steps, paths, seeds, n_gen)
  prompts.yaml       ← prompts kanonik (promosi dari redesign_v4.yaml + fix §1 sekunder)
  transfer.py        ← KV transfer policy: no-crop, kv_close_turn, chain/concat wrapper
  agents.py          ← thin wrapper di atas latent_mas.agent (load_agent + keep_answer_in_kv)
  pipeline.py        ← orkestrator loop evolusi (topologi §3)
  runner.py          ← gate + repair + backtest (reuse scoring/ + factors/ existing)
  run.py             ← CLI entry (argparse): --generations --latent-steps --dry-run
  results/           ← output run produksi (artifacts per generasi)
```

Reuse (tidak ditulis ulang): `backend/llm/client.py` (LocalLLMBackend),
`backend/latent_mas/{agent,kv_ops,parsers}.py`, scoring & DSL gate dari
`try/promptbench/scoring` (akan dimigrasi/di-import bersih ke `prod/runner.py`).

**Perubahan inti yang menyentuh kode bersama:** hanya `agent.py` (+field) dan
`client.py` (crop kondisional) seperti §2.1 — minimal & backward-compatible.

---

## 5. Pijakan ke repo referensi

- **LatentMAS** (README + port lokal [latent_mas_hybrid.py](../backend/core/latent/latent_mas_hybrid.py)):
  agent berkomunikasi via *latent thoughts*, transfer = realignment **Eq.8**
  (`transfer_via_realignment`, [latent_mas_hybrid.py:18](../backend/core/latent/latent_mas_hybrid.py#L18))
  atas hidden states, + working-memory concat ([latent_mas_hybrid.py:369-431](../backend/core/latent/latent_mas_hybrid.py#L369)).
  Penting: referensi **mengakumulasi `cumulative_prompts` sebagai teks** dan
  re-encode saat decode ([latent_mas_hybrid.py:425](../backend/core/latent/latent_mas_hybrid.py#L425)).
  → Keputusan no-crop kita = analog: jawaban diskret tetap tersedia bagi hop
  berikut (lewat KV, bukan re-encode teks), sambil latent steps tetap berjalan.
- **QuantaAlpha**: pipeline alpha-mining (proposal/design/construct + feedback loop)
  mengoper hipotesis/faktor sebagai **teks/JSON**. Kita memetakannya ke transfer
  KV: artefak yang di QuantaAlpha berupa teks, di sini hidup sebagai token nyata
  dalam KV (no-crop) + laten.

---

## 6. Risiko terbuka & mitigasi (untuk dites)

1. **Korupsi decode pada KV panjang** (token sampah `,T 10` di design). No-crop
   **memperpanjang** KV → bisa memperparah. Mitigasi kandidat: turunkan
   `latent_steps` di hop dalam; aktifkan `kv_knn_filter` ([kv_ops.py](../backend/latent_mas/kv_ops.py))
   untuk prune KV; cap panjang KV (`kv_truncate`). **Bukan** disebabkan no-crop —
   harus diukur terpisah (lihat §7).
2. **Schema contamination** (alasan crop dibuat): construct mungkin meniru schema
   JSON palette design alih-alih schema faktornya. Mitigasi: prompt construct tegas
   soal schema output; bila perlu, no-crop hanya di `proposal→design`, dan
   `design→construct` pakai crop+laten. Diputuskan dari hasil dry/real-run.
3. **Memori KV / OOM**: KV tumbuh tiap hop × generasi. Mitigasi: prune/truncate,
   `torch.cuda.empty_cache()` per generasi (sudah ada pola di v4_eval).
4. **Turn asisten tak tertutup** (§2.2): wajib `kv_close_turn`.

---

## 7. Rencana validasi

- **Dry-run**: render tiap node, cek var lengkap + wiring transfer (chain/concat,
  no-crop flag, decode) tanpa GPU.
- **Metrik fidelity hipotesis (baru)**: ukur kemiripan `HYPOTHESIS:` antar
  `proposal→design→construct` (mis. token-overlap / embedding-sim). Target:
  no-crop menaikkan fidelity vs baseline crop.
- **Korupsi**: hitung rasio token non-ASCII/garbage per node sebelum vs sesudah
  no-crop, untuk memisahkan efek no-crop dari efek realigner.
- **Skor**: parser_ok + gate_pass + score (reuse scoring) per construct terminal.
- Bandingkan A/B: `crop` (lama) vs `no-crop` (baru), latent_steps ∈ {0,10,20}.

---

## 8. Fase implementasi

- **F1 — SELESAI ✅** Patch inti (backward-compatible): `keep_answer_in_kv` di
  `AgentSpec` (default True=no-crop) + crop kondisional `crop_after_generate` di
  `client.py` + `_CoreEngine._close_open_turn` (tutup turn asisten). Compile OK.
  Catatan: `try/promptbench` v4_eval kini ikut no-crop (set `keep_answer_in_kv:
  false` di redesign_v4.yaml bila ingin perilaku crop lama).
- **F2 — SELESAI ✅** `prod/` terimplementasi: `config`, `transfer`, `agents`,
  `pipeline` (topologi segmented, KV restart di feedback), `run.py`, `prompts.yaml`
  (promosi + fix: design carry VERBATIM, construct fidelity-first, feedback
  CARRY-FORWARD restate hipotesis+DSL+metrik). **Dry-run terverifikasi** untuk
  mode kv / text / crossover (wiring benar, missing_vars=0, no_crop=True).
- **F3 — SELESAI ✅ (kecuali real-backtest & GPU)** `runner.py` ter-wire ke
  `pipeline._score_and_prepare_feedback` + tracking SOTA RankIC lintas generasi:
  - **gate** = regulator ASLI `latent_mas.pipeline._build_regulator_gate` (impor
    langsung, BUKAN dari `try/`); rejection reason di-LOG.
  - **repair = AGENT** (`prompts.yaml: repair`, standalone/non-chained): memperbaiki
    SATU ekspresi ilegal agar lolos gate TANPA mengubah intent — mengganti fungsi
    agar arity cocok, menyederhanakan argumen, menamai ulang fungsi, dll. Memakai
    `explanation` per-faktor dari construct (intent) + reason gate + library DSL
    (`config.FUNCTION_LIB` via `{{ function_lib }}`). Alur: gate → repair kurung
    murah (gratis) → AGENT repair (sadar-intent) → re-gate. Disuntik pipeline via
    `_make_repair_fn` (butuh backend). Teruji standalone dgn stub: arity-reject→
    agent-fix, unbalanced→bracket-fix, trivial→tetap reject.
  - **backtest** dua mode: `mock` (metrik deterministik hash → loop+log penuh
    tanpa Qlib) dan `real` (adapter ke `factors.QlibFactorRunner._compute_factor_ic`
    — BELUM diuji, butuh env Qlib).
  - **Logging terminal** (`runlog.py`, default on): per-agent durasi
    berpikir(latent)+generate+total, durasi transfer KV, token in/out, kv_len,
    latent_steps; alasan gate menolak; durasi backtest; SOTA/replace-best.
    Backend ditambah `LLMResult.latent_s/gen_s` + propagasi ke `AgentResult`.
  - Sisa F3: jalankan GPU + wire `real` backtest ke data Qlib + populasi ≥2
    lineage untuk crossover sejati.
- **F4 — HARNESS SELESAI ✅ (A/B numbers butuh GPU)** `prod/analyze.py` (tanpa GPU):
  mengukur dari artifacts (prod ATAU legacy) per rantai proposal→design→construct:
  `fidelity_pd/dc` (Jaccard kata-isi hipotesis antar hop), `mech_pd/dc` (overlap
  kata-mekanisme), `drift` flag, `corruption_hits` (heuristik token sampah),
  `gate_pass_frac` (via runner). CLI multi-dir → tabel A/B + `analysis.json`.
  - **Tervalidasi** pada legacy `v4_eval/kv_text/ls10/rep0` (crop baseline):
    `drift_rate=1.0`, `fidelity_pd=0.112`, `corruption=12`, `gate=0.417` —
    mengkuantifikasi drift yang didiagnosis.
  - **Caveat**: heuristik korupsi kasar (false-positive pada output verbose);
    A/B bermakna = run **prod** `kv` vs `text` (prompt & format identik), bukan
    legacy (prompt lama + format beda → angka confounded).
  - Sisa: jalankan prod GPU `kv` (no-crop) & `text` (ls=0) lalu
    `python -m prod.analyze prod/results/kv/ls10 prod/results/text/ls0`.
