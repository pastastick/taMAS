# Latent Pipeline Integration — Factor Mining Components

**Tanggal**: 2026-04-07
**Scope**: `factors/experiment.py`, `factors/proposal.py`, `factors/feedback.py`, `factors/latent_proposal.py`
**Komponen yang diubah**: `scen`, `hypothesis_gen`, `hypothesis2experiment`

---

## Konteks

Setelah infrastruktur latent selesai (llm/, pipeline/, pipeline/evolution/),
sekarang menerapkan latent communication + reasoning ke komponen factor mining
yang direferensikan oleh `AlphaAgentFactorBasePropSetting.component_class_paths`.

Prinsip utama:
- **kv_and_text** — semua step menghasilkan teks DAN KV-cache
  - Teks: untuk trajectory pool, user monitoring, downstream parsing
  - KV-cache: untuk inter-agent latent communication (chain antar step)
- KV-cache encoding terjadi di LLM backend, BUKAN di Scenario/Proposal/Feedback
- Scenario mengontrol KONTEN teks yang masuk ke prompt (dan otomatis ter-encode ke KV)

---

## 1. `scen` — QlibAlphaAgentScenario (`factors/experiment.py`)

### Masalah yang ditemukan

**A. `filtered_tag` di rdagent adalah no-op**

Ditemukan bahwa `QlibFactorScenario.get_scenario_all_desc()` dari rdagent
(Microsoft RD-Agent) **mengabaikan parameter `filtered_tag`** sepenuhnya.
Padahal kode kita memanggil:
```python
# proposal.py:351 (propose step)
self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
# evolving_strategy.py:119 (coder step)
self.scen.get_scenario_all_desc(target_task, filtered_tag="feature")
```

**Dampak**: Setiap step mendapat SELURUH scenario dump (~ratusan token)
padahal hanya butuh sebagian. Ini membuang kapasitas KV-cache.

**Proses berpikir**: Untuk memverifikasi ini, saya fetch source code rdagent
dari GitHub dan menemukan bahwa `get_scenario_all_desc()` hanya punya branch
untuk `simple_background=True` dan default — `filtered_tag` diterima tapi
tidak pernah digunakan di body method.

**B. `_strategy` dan `_experiment_setting` tidak terexpose**

`QlibAlphaAgentScenario.__init__` men-set `self._strategy` dan
`self._experiment_setting`, tapi:
- rdagent's `get_scenario_all_desc()` TIDAK menyertakan keduanya di output
- `_strategy` tidak punya property accessor (hidden attribute)

**C. Dependency pada rdagent property implementation**

Properties seperti `background`, `interface`, dll. di-inherit dari
`QlibFactorScenario` (rdagent). Jika rdagent berubah, kode kita bisa break
tanpa warning.

### Keputusan

1. **Override `get_scenario_all_desc()` dengan proper filtering** —
   implementasi sendiri yang benar-benar mem-filter berdasarkan `filtered_tag`:
   - `"hypothesis_and_experiment"` → background + strategy + experiment_setting
   - `"feature"` → background + source_data + interface + output_format
   - `None` → full (semua section, termasuk strategy + experiment_setting)

2. **Tambah `get_compact_desc(step)`** — method baru untuk latent mode.
   Per-step compact description yang menyisakan lebih banyak kapasitas KV-cache:
   - `"propose"` → background + strategy (APA yang di-explore)
   - `"construct"` → interface + output_format (BAGAIMANA menulis formula)
   - `"feedback"` → background + experiment_setting (BAGAIMANA mengevaluasi)

3. **Self-contained properties** — semua property didefinisikan langsung,
   tidak bergantung pada rdagent. Termasuk `strategy` (property baru).

**Mengapa get_compact_desc() pakai XML-like section markers?**

```xml
<scenario_background>...</scenario_background>
<scenario_strategy>...</scenario_strategy>
```

Section markers membantu attention mechanism model membedakan boundaries
antar bagian skenario. Saat model melakukan latent reasoning (virtual tokens
via `_CoreEngine.latent_pass()`), representasi internal perlu "tahu" dimana
satu section berakhir dan yang lain mulai. Tanpa markers, boundary ini ambigu
di embedding space.

**Mengapa Scenario tetap text-only (tidak langsung encode ke KV)?**

Sempat dipertimbangkan apakah Scenario bisa langsung meng-encode deskripsinya
ke KV-cache. Keputusan: TIDAK, karena:
1. Scenario tidak punya akses ke model LLM (beda layer)
2. KV-cache bergantung pada SELURUH prompt (system + user), bukan hanya scenario
3. Tiap step pakai bagian scenario yang berbeda
4. Encoding ke KV sudah terjadi otomatis di LLM backend saat prompt diproses

Scenario mengontrol KONTEN, LLM backend mengontrol ENCODING.

---

## 2. `hypothesis_gen` — AlphaAgentHypothesisGen (`factors/proposal.py`)

### Masalah yang ditemukan

**A. `gen()` hardcode `get_scenario_all_desc()` di 2 tempat**

```python
# line 351 (dalam retry loop)
scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
# line 385 (final attempt)
scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
```

`LatentHypothesisGen` mewarisi `gen()` tanpa override (method ~60 baris
dengan retry logic). Tidak ada cara bagi latent subclass untuk mengganti
scenario description tanpa menduplikasi seluruh `gen()`.

**B. Bug stale KV-cache di `_call_llm()` text-only path**

```python
def _call_llm(self, user_prompt, system_prompt, json_mode=False):
    if self._past_kv is not None and ...:
        # latent path
        self.last_result = result   # ← set
        return result.text
    # text-only path
    return self.llm_backend.build_messages_and_create_chat_completion(...)
    # ← last_result TIDAK di-reset!
```

**Skenario bug**:
1. Iterasi 1: `_past_kv` ada → latent path → `last_result = LLMResult(kv_cache=X)`
2. Iterasi 2: `_past_kv` jadi None → text-only path → `last_result` MASIH dari iter 1
3. `last_kv` property return `X` (KV stale dari konteks yang salah)
4. Construct step pakai KV ini → reasoning berdasarkan konteks yang salah

**Proses berpikir**: Bug ini ditemukan saat menelusuri alur `last_kv` property:
```python
@property
def last_kv(self):
    if self.last_result is not None:
        return getattr(self.last_result, "kv_cache", None)
    return None
```
Jika `last_result` tidak di-reset saat text-only path, property ini return
nilai dari panggilan latent SEBELUMNYA. Ini bisa terjadi jika `_pipeline_kv`
berubah antara iterasi (misal: None setelah truncation error).

### Keputusan

1. **Tambah `_get_scenario_desc()` helper** — method yang bisa di-override:
   ```python
   # Base class
   def _get_scenario_desc(self) -> str:
       return self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
   ```
   `gen()` memanggil ini sekali, simpan di variabel `scenario_desc`, pakai
   di kedua tempat. Latent subclass override tanpa duplikasi `gen()`.

2. **`LatentHypothesisGen._get_scenario_desc()` → `get_compact_desc("propose")`** —
   hanya background + strategy. Dengan fallback ke `get_scenario_all_desc()`
   jika scenario bukan `QlibAlphaAgentScenario` (`hasattr` check).

3. **Fix stale KV: `self.last_result = None` di text-only path** —
   diterapkan di TIGA class yang punya pattern yang sama:
   - `AlphaAgentHypothesisGen._call_llm()`
   - `AlphaAgentHypothesis2FactorExpression._call_llm()`
   - `AlphaAgentQlibFactorHypothesisExperiment2Feedback._call_llm()`

**Mengapa `_get_scenario_desc()` tanpa parameter (tidak terima `trace`)?**

`AlphaAgentHypothesisGen` punya `self.scen` (di-set oleh `HypothesisGen.__init__`
via inheritance chain). Jadi tidak perlu `trace` — langsung akses `self.scen`.

---

## 3. `hypothesis2experiment` — AlphaAgentHypothesis2FactorExpression (`factors/proposal.py`)

### Masalah yang ditemukan

**A. Sama seperti hypothesis_gen: hardcode scenario, tidak bisa di-override**

```python
# _convert_with_history_limit line 587
scenario=trace.scen.background,
```

**B. Pattern berbeda: akses scenario via `trace.scen`, bukan `self.scen`**

`Hypothesis2Experiment` ABC (dan turunannya) TIDAK menerima `scen` di `__init__`.
Scenario diakses via `trace.scen` di method `convert()`/`prepare_context()`.

Ini berarti `_get_scenario_desc()` harus terima `trace` sebagai argumen —
berbeda dari hypothesis_gen yang pakai `self.scen`.

**C. `prepare_context()` menghitung scenario yang tidak dipakai**

```python
def prepare_context(self, hypothesis, trace, history_limit):
    scenario = trace.scen.get_scenario_all_desc()  # ← dihitung
    return {
        "scenario": scenario,  # ← masuk dict
        ...
    }, True
```

Tapi `_convert_with_history_limit()` TIDAK menggunakan `context["scenario"]` —
langsung pakai `trace.scen.background`. Ini dead code untuk AlphaAgent path
(parent class `QlibFactorHypothesis2Experiment.convert()` menggunakannya,
tapi AlphaAgent override `convert()`).

**Keputusan**: Tidak menghapus `context["scenario"]` karena bisa break
parent class path. Fokus pada `_get_scenario_desc(trace)` pattern.

### Keputusan

1. **`_get_scenario_desc(self, trace)` dengan parameter `trace`** —
   signature berbeda dari hypothesis_gen karena class ini tidak punya `self.scen`.
   Default: `trace.scen.background` (menjaga perilaku lama).

2. **`LatentHypothesis2Experiment._get_scenario_desc(trace)`** →
   `trace.scen.get_compact_desc("construct")` yang return interface + output_format.

   **Mengapa tidak menyertakan background di compact construct?**
   Background sudah ada di KV-cache dari propose step yang di-chain via `_past_kv`.
   Model sudah "tahu" konteks market dari latent reasoning di propose step.
   Construct hanya butuh syntax rules (interface + output_format) untuk
   menghasilkan formula yang presisi dan parseable.

3. **Dokumentasi retry loop KV behavior** — ditambahkan di docstring
   `_convert_with_history_limit()`:
   - Setiap retry pakai `_past_kv` yang SAMA (dari propose step)
   - Retry TIDAK chain KV antar attempt
   - Feedback duplikasi dikirim via teks (user_prompt), bukan latent
   - `last_result` di-update setiap retry → `last_kv` = attempt terakhir

   **Mengapa tidak chain KV antar retry?**
   Dipertimbangkan: retry 2 pakai KV dari retry 1 (yang gagal) agar model
   "belajar" dari kesalahan via latent space. Keputusan: TIDAK, karena:
   - Expression duplication feedback sudah ada di teks user_prompt
   - KV dari attempt gagal bisa akumulasi noise (formula yang buruk)
   - Starting fresh dari propose_kv memberikan starting point yang bersih
   - Lebih predictable dan debuggable

---

## Ringkasan Perubahan per File

### `factors/experiment.py`
- Override `get_scenario_all_desc()` — proper `filtered_tag` support
- Tambah `get_compact_desc(step)` — per-step compact desc untuk latent
- Self-contained properties (background, interface, strategy, dll.)

### `factors/proposal.py`
- `AlphaAgentHypothesisGen`: tambah `_get_scenario_desc()`, fix stale KV
- `AlphaAgentHypothesis2FactorExpression`: tambah `_get_scenario_desc(trace)`, fix stale KV

### `factors/feedback.py`
- Fix stale KV bug di `_call_llm()` text-only path

### `factors/latent_proposal.py`
- `LatentHypothesisGen`: override `_get_scenario_desc()` → `get_compact_desc("propose")`
- `LatentHypothesis2Experiment`: override `_get_scenario_desc(trace)` → `get_compact_desc("construct")`, update docstring dual-output

---

## KV-Cache Flow (setelah perubahan)

```
External/Planning KV
  ↓
AlphaAgentLoop._pipeline_kv = past_kv
  ↓
factor_propose:
  set_past_kv(_pipeline_kv)
  scenario: get_compact_desc("propose")  ← background + strategy
  _call_llm → build_messages_and_run(mode=kv_and_text, role="propose")
  output: hypothesis TEXT + kv_propose
  ↓
factor_construct:
  set_past_kv(kv_propose)                ← chain dari propose
  scenario: get_compact_desc("construct") ← interface + output_format
  _call_llm → build_messages_and_run(mode=kv_and_text, role="construct",
              temperature=0.3)
  output: factor expression TEXT + kv_construct
  [retry loop jika validation gagal → same kv_propose, updated user_prompt]
  ↓
factor_calculate + factor_backtest: (no LLM, no KV)
  ↓
feedback:
  set_past_kv(kv_construct)              ← chain dari construct
  scenario: get_scenario_all_desc()       ← full (belum diubah ke compact)
  _call_llm → build_messages_and_run(mode=kv_and_text, role="feedback")
  output: feedback TEXT + kv_feedback
  → kv_truncate(kv_feedback, kv_max_tokens) → _pipeline_kv (next iteration)
  ↓
_get_trajectory_data() → {"pipeline_kv": _pipeline_kv}
  → StrategyTrajectory.kv_cache (evolution)
```

---

## Status Komponen

| Komponen | Class Path | Status | Keterangan |
|----------|-----------|--------|-----------|
| `scen` | `factors.experiment.QlibAlphaAgentScenario` | ✅ Done | Override get_scenario_all_desc + get_compact_desc |
| `hypothesis_gen` | `factors.proposal.AlphaAgentHypothesisGen` | ✅ Done | _get_scenario_desc helper + stale KV fix |
| `hypothesis2experiment` | `factors.proposal.AlphaAgentHypothesis2FactorExpression` | ✅ Done | _get_scenario_desc(trace) helper + stale KV fix |
| `summarizer` | `factors.feedback.AlphaAgentQlibFactorHypothesisExperiment2Feedback` | ✅ Done | _get_scenario_desc helper + stale KV fix |
| `coder` | `factors.qlib_coder.QlibFactorParser` | ⚠ N/A | Lihat penjelasan di bawah |
| `runner` | `factors.runner.QlibFactorRunner` | ⚠ N/A | Tidak ada LLM call |

---

## 4. `summarizer` — AlphaAgentQlibFactorHypothesisExperiment2Feedback (`factors/feedback.py`)

### Perubahan

1. **Tambah `_get_scenario_desc()` helper** — method yang bisa di-override:
   ```python
   def _get_scenario_desc(self) -> str:
       return self.scen.get_scenario_all_desc()
   ```
   `generate_feedback()` memanggil ini sekali, simpan di `scenario_desc`.

2. **`LatentFeedback._get_scenario_desc()` → `get_compact_desc("feedback")`** —
   hanya background + experiment_setting dengan section markers.

   **Mengapa background tetap disertakan di compact feedback?**
   Berbeda dari construct step (yang menghilangkan background karena sudah ada di KV),
   feedback perlu menyatakan ulang konteks market secara eksplisit. Evaluator
   perlu konteks "domain ini menilai faktor trading" untuk menilai apakah backtest
   result itu baik atau buruk. Construct hanya butuh syntax rules, feedback butuh
   judgment context.

---

## 5. `coder` — QlibFactorParser/FactorParsingStrategy (`factors/coder/evolving_strategy.py`)

### Mengapa tidak diintegrasikan ke latent KV-cache

`FactorParsingStrategy` secara arsitektur **tidak kompatibel** dengan latent pipeline:

1. **Direct instantiation**: `LocalLLMBackend()` di-instantiate baru setiap panggilan
   (bukan `self.llm_backend`), tidak ada channel untuk inject KV-cache dari luar.

2. **Multiprocessing**: `multiprocessing_wrapper` menjalankan `implement_one_task`
   secara parallel. KV-cache adalah GPU tensor — tidak bisa di-pickle untuk
   lintas proses. Ini constraint keras dari PyTorch/CUDA, bukan pilihan desain.

3. **Purpose berbeda**: Coder step adalah code syntax fixing (ekspresi matematika
   → Python yang bisa dieksekusi), bukan semantic reasoning tentang market hypothesis.
   KV-cache dari propose/construct step tidak relevan untuk debugging parse error.

4. **Flow sudah benar**: Dalam KV-cache flow, coder step sudah dilabeli
   "factor_calculate: (parser, no LLM, no KV)" — ini intentional by design.

### Manfaat tidak langsung dari sesi ini

Meskipun tidak diintegrasikan ke latent, coder **sudah mendapat manfaat**:
- `filtered_tag="feature"` di line 278 evolving_strategy.py sekarang bekerja
  benar karena `QlibAlphaAgentScenario.get_scenario_all_desc()` sudah di-override.
- Sebelumnya rdagent mengabaikan `filtered_tag` → coder mendapat full scenario dump.
- Sekarang coder mendapat: background + source_data + interface + output_format
  (tepat untuk code generation context).

---

## 6. `runner` — QlibFactorRunner (`factors/runner.py`)

### Mengapa tidak diintegrasikan

`QlibFactorRunner` adalah backtest executor murni:
- Tidak ada LLM call sama sekali
- Menjalankan faktor Python code, mengumpulkan DataFrame, menjalankan Qlib backtest
- Output: metrik trading (IC, return, drawdown)
- Tidak ada interaksi dengan LLM backend → tidak ada KV-cache yang relevan
