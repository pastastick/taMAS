# Panduan Operasional QuantaLatent

> Dokumen ini adalah panduan menjalankan repo dalam bahasa Indonesia:
> peta repo, setup RunPod, cara menjalankan kedua lengan eksperimen, dan
> masalah yang sering muncul. Ringkasan berbahasa Inggris beserta hasil
> utamanya ada di [`../README.md`](../README.md).
>
> **Pertanyaan penelitiannya** (dirumuskan ulang 2026-08-27, empat butir):
>
> 1. Bagaimana berbagai formulasi pembentukan representasi laten dapat
>    dinyatakan dalam satu kerangka matematis yang seragam?
> 2. Bagaimana pengaruh formulasi itu terhadap penalaran dan fidelitas
>    simbolik, dan apakah pengaruhnya berubah menurut tuntutan presisi tugas?
> 3. Apakah komunikasi lewat KV-cache mempertahankan kinerja pada biaya token
>    dan waktu yang lebih rendah, dan bagaimana formulasi memengaruhi efisiensi
>    itu?
> 4. Sejauh mana representasi laten mempertahankan informasi simbolik pada
>    generasi ekspresi faktor terstruktur?
>
> **Baca dulu**: [`DESAIN_EKSPERIMEN.md`](DESAIN_EKSPERIMEN.md) (apa yang
> diukur dan kenapa), [`TEORI.md`](TEORI.md) (asumsi + bukti matematis di balik
> kerangka dan sumbu interpolasi), lalu [`HASIL_TAHAP0.md`](HASIL_TAHAP0.md)
> (angka yang mendasari rancangan ini).

---

## 0. Yang WAJIB dipahami sebelum menjalankan apa pun

Enam keputusan desain yang menentukan apakah sebuah run sah atau terbuang.
Semuanya sudah pernah salah sekali, dan tiap kesalahan memakan jam GPU.

**(a) Ada dua himpunan formulasi, bukan satu daftar datar.**

$$\mathcal M = \{\texttt{raw}, \texttt{soft}, \texttt{sample}, \texttt{gumbel}, \texttt{moi}\},
\qquad \mathcal R = \mathcal M \setminus \{\texttt{raw}\}$$

$\mathcal R$ = keluarga relaksasi diskret; tiap anggotanya membentuk langkah
laten sebagai kombinasi konveks baris $W_\text{in}$, jadi hasilnya selalu di
dalam convex hull embedding. `raw` satu-satunya di luar. Seluruh klaim skripsi
berbentuk "$\mathcal R$ versus `raw`", **bukan** "varian X terbaik".

> ⚠️ Dokumen lama menyebut `soft` sebagai "kontrol, bukan salah satu dari
> empat". Label itu **dibatalkan** 2026-08-27: seluruh analisis yang terbit
> memperlakukan `soft` sebagai anggota penuh $\mathcal R$ (kontras keluarga
> merata-ratakan empat anggota; Cochran $Q$ dijalankan $k=5$).

**(b) Dua lengan SETARA, dan lengan faktor yang lebih menuntut.**
Bench menanyakan "apakah agen masih bisa bernalar"; lengan faktor menanyakan
"apakah agen masih bisa membawa struktur yang harus tepat". Lengan faktor bukan
lampiran, dan bukti di sana **berjenjang enam level** (parse → evaluable →
fidelitas → keberagaman → RankIC → holdout), bukan satu uji hipotesis. Dengan
20 jalan per sel, uji inferensial formal tak akan punya daya — jangan
memaksakannya.

**(c) `comm_mode=text` tidak punya langkah laten.**
Karena itu kelima nilai Sumbu A menghasilkan sel yang **identik**. `text`
dijalankan **sekali per tugas**, bukan lima kali. Menjalankannya lima kali
membakar GPU untuk lima salinan angka yang sama, dan lebih buruk: lima salinan
itu akan terbaca sebagai lima pengamatan independen di tabel.
`scripts/gen_perintah.py` menegakkan aturan ini — **turunkan perintah dari
sana, jangan mengetiknya tangan**.

**(d) `--sample-seed` harus sama di semua sel bench.**
Itu yang membuat semua metode melihat soal yang sama dan uji berpasangannya
sah. `bench/compare.py` memverifikasi lewat sidik jari dan **mengeluarkan** sel
yang tak cocok — mengubahnya di tengah matriks berarti membuang sel yang sudah
jadi.

**(e) Mode `mix` adalah alat ukur, bukan usulan metode.**

$$z(\alpha) = \frac{(1-\alpha)\,z_\texttt{raw} + \alpha\,z_\texttt{soft}}
{\lVert(1-\alpha)\,z_\texttt{raw} + \alpha\,z_\texttt{soft}\rVert}\cdot\rho$$

Kelima formulasi memberi lima titik terpisah, sehingga hubungan geometri↔kinerja
hanya bisa dibaca "searah". `mix` mengisi jaraknya supaya **bentuk** hubungan
itu yang diuji. Pada $\alpha=0$ ia mereduksi persis ke `raw`, pada $\alpha=1$
persis ke `soft` (diverifikasi numerik), jadi **kedua titik ujung TIDAK
dijalankan ulang** — kurvanya menyambung ke sel yang sudah ada.

> Hipotesisnya sengaja **tidak berarah**. Monoton, ber-ambang, dan tak berpola
> ketiganya temuan; yang ketiga berarti klaim mekanistik Bab IV harus
> dilemahkan. Jangan menulis "semakin dekat embedding semakin baik" sebelum
> datanya ada.

**(f) Guided decoding TIDAK PERNAH aktif.**
`configs/matriks.yaml` menulis `guided_decoding: true` dan
`backend/prompts/factor.yaml` menulis `json_schema: latent_construct`, tetapi
`agent._resolve_json_schema` mensyaratkan env `LATENTMAS_GUIDED=1` yang dulu
di-set `pipeline/loop.py` — modul yang terhapus bersama pipeline evolusi.
Buktinya: tak satu pun keluaran memuat field `reasoning` yang diwajibkan schema
itu. Seluruh angka yang terbit dihasilkan **tanpa** guided decoding. Kalau mau
menyalakannya, set env-nya secara eksplisit dan sadari biayanya (+10–20%
latensi per token) — dan seluruh lengan faktor harus dijalankan ulang.

---

## 1. Peta repo

Basis kode ini berasal dari **perombakan total** `exp/alt3-gumbel-fidelitas`
(`99721ec`) yang dikerjakan di branch `exp/empat-metode-v1` lalu dijadikan isi
`main`: `lab/` dilebur ke `backend/`, seluruh warisan RD-Agent/QuantaAlpha yang
tak terpakai dihapus (pipeline evolusi, CoSTEER, `core/`, agen eksternal,
loader dokumen — ±95 berkas Python), dan dua lengan eksperimen baru dibangun.
**Tidak ada kode yang hilang**: semuanya tetap ada di branch lama dan di
riwayat `main`.

```
backend/
  llm/        mesin LLM: model, KV-cache, latent_pass, _latent_step_vec   ← SUMBU A
  mas/        agen + operasi KV + pipeline rantai faktor                  ← SUMBU B
  bench/      lengan replikasi LatentMAS  (data · scoring · pipeline · run_bench · compare)
  factor/     lengan faktor alpha         (run_factor.py)
  dsl/        parser ekspresi · AST · pustaka fungsi (71 fungsi)
  gate/       gate mutu ekspresi: regulator, arity, redundansi, kompleksitas
  eval/       ic.py · backtest.py · stats.py · fidelity.py · channel_capacity.py
              compare_modes.py · realign_probe.py · b7_probe.py · rescore_all.py
              skor_holdout.py — skor korpus di jendela yang tak pernah dipakai menyaring
  prompts/    factor.yaml (QuantaLatent) · bench.yaml (port LatentMAS)
  paths.py    jalur kanonik + bootstrap sys.path      qlog.py  logger (loguru)
configs/      matriks.yaml — daftar sel eksperimen (sumber kebenaran tunggal)
scripts/      gen_perintah.py (turunkan perintah dari matriks.yaml) ·
              jalankan_matriks.py (runner antrean bergerbang VRAM) ·
              rakit_transkrip.py (satu Markdown terbaca per sel) ·
              hitung_token.py · kumpulkan_pendukung.py · plot_readme_figures.py
reference/    LatentMAS @9a9e4d3 · mixinputs @7aef34b (rujukan, READ-ONLY)
docs/         PANDUAN.md (berkas ini) · TEORI.md · DESAIN_EKSPERIMEN.md ·
              HASIL_TAHAP0.md · HASIL_TAHAP4.md · AUDIT_KRITIS.md
results/      keluaran run — probe/ (artefak Tahap 0) · bench/ · factor/ · pendukung/
assets/       gambar README, dibangkitkan ulang oleh scripts/plot_readme_figures.py
```

Skrip analisis yang membangkitkan tabel skripsi ada **di luar repo ini**, di
`../analisis/` (satu direktori di atas), karena keluarannya `.tex` untuk
`../skripsi/`. Yang relevan: `04_tabel_tex.py` (tabel lengan bench),
`09_faktor_perhop.py` (per-hop + biaya lengan faktor), `10_tabel_faktor.py`
(tabel enam level bukti).

Aturan import: `backend/` adalah root paket. Jalankan apa pun dengan
`PYTHONPATH=backend`, atau panggil skrip langsung — tiap skrip CLI memanggil
`paths.bootstrap()` sendiri.

---

## 2. Setup RunPod

Di RunPod **hanya `/workspace` yang persisten**; `/root` hilang saat pod
restart. Semua artefak (uv, `.venv`, cache HF, model) harus di bawah
`/workspace`.

```bash
# sekali per session SSH baru
source /workspace/runpod_env.sh          # salinan ada di repo: runpod_env.sh
cd /workspace/project/multi-agent-system
uv sync                                   # torch 2.6.0+cu124 dari index cu124
source .venv/bin/activate
```

Spesifikasi pod: A40 46 GB, volume disk ≥ 100 GB, CUDA ≥ 12.1. **Maksimal dua
sel paralel** — lihat §3 untuk angka VRAM terukur per lengan.

### `.env`

```env
HF_TOKEN=hf_...            # opsional; Qwen3 publik bisa diakses anonim
HF_HOME=/workspace/.cache/huggingface
HF_LOCAL_ONLY=0            # 0 = boleh unduh; 1 = paksa offline
```

> Jangan pernah menaruh token asli di berkas yang di-track git. Insiden token
> bocor di `runpod_env.sh` tercatat di `HASIL_TAHAP0.md` §2.

### Data pasar (hanya untuk lengan faktor)

```bash
cd /workspace/project/multi-agent-system/backend
hf download QuantaAlpha/qlib_csi300 --repo-type dataset --local-dir ./hf_data
python -c "import zipfile; zipfile.ZipFile('hf_data/cn_data.zip').extractall('data/qlib/')"
```

`backend/hf_data/daily_pv.h5` adalah satu-satunya berkas yang dibutuhkan
`eval/ic.py`; `data/qlib/cn_data/` dipakai kalau backtest Qlib penuh
dihidupkan lagi. Keduanya gitignored.

### Model

```bash
hf download Qwen/Qwen3-8B     # ~16 GB, atau biarkan terunduh saat run pertama
```

---

## 3. Menjalankan

Satu proses = **satu sel** matriks. Ini disengaja: satu sel tidak menghabiskan
GPU sendirian, jadi beberapa sel dijalankan bersamaan.

Pemakaian VRAM **terukur** di A40 46 GB (`nvidia-smi --query-compute-apps`,
10 Agustus 2026):

| lengan | VRAM per proses | slot paralel aman |
|---|---:|---|
| bench (`bench/run_bench.py`, `--max-new-tokens 2048`) | ~18 GB | 2 |
| faktor (`factor/run_factor.py`, `--max-new-tokens 4096`, rantai 3 agen) | ~21 GB | 2 (= 93% VRAM, tanpa margin) |

> ⚠️ `HASIL_TAHAP0.md` §8.7 menulis "~16 GB → 2–3 run paralel". Angka itu
> berasal dari probe/bench pendek dan **tidak berlaku untuk lengan faktor**;
> menyetel 3 sel faktor paralel akan OOM. Pakai `scripts/jalankan_matriks.py`
> yang punya gerbang VRAM (`--vram-bebas-min`, satuan MiB) dan antre-ulang
> otomatis untuk sel yang OOM (`--ulang-maks`).

### Lengan 1 — benchmark ala LatentMAS

```bash
PYTHONPATH=backend python backend/bench/run_bench.py \
    --task gsm8k --latent-mode gumbel --comm-mode kv \
    --limit 100 --sample-seed 0 --seed 0
```

`--task` ∈ `gsm8k` (math) · `arc_challenge` (commonsense) · `humanevalplus` (code)
`--latent-mode` ∈ `raw` `soft` `sample` `gumbel` `moi` (+ `mix`, lihat §3.4)
`--comm-mode` ∈ `kv` `kv_and_text` `text`; `--baseline` = agen tunggal

> `--sample-seed` HARUS sama di semua sel — lihat §0(d).

### Lengan 2 — faktor alpha (simbolik/DSL)

```bash
PYTHONPATH=backend python backend/factor/run_factor.py \
    --comm-mode kv --latent-mode gumbel --latent-steps 10 \
    --seeds 0,1,2,3,4 --directions d0,d1,opp_mom,opp_rev --tag kv_gumbel
```

Empat arah × lima seed = **20 jalan per sel** (naik dari 6 pada run 2026-08-10).
Dua arah tambahan `opp_mom`/`opp_rev` sudah lama ada di
`run_factor.py::DIRECTIONS` tapi belum pernah dijalankan: `d0` dan `d1`
sama-sama keluarga mean-reversion, sehingga keluaran yang mirip bisa berarti
"arahnya memang mirip" alih-alih "sistem mengabaikan arah". Pasangan tambahan
itu berlawanan pada tiga sumbu sekaligus, jadi keragaman yang terukur tak lagi
bisa dituduh artefak arah.

### 3.4 Sumbu C — interpolasi (`mix`)

```bash
PYTHONPATH=backend python backend/bench/run_bench.py \
    --task humanevalplus --comm-mode kv \
    --latent-mode mix --latent-alpha 0.5 --limit 100 \
    --sample-seed 0 --seed 0 --tag s0_a05
```

`--latent-alpha` **hanya** berlaku untuk `--latent-mode mix`. Jalankan hanya
$\alpha \in \{0{,}25;\ 0{,}5;\ 0{,}75\}$ — lihat §0(e) untuk alasan titik
ujungnya dilewati. Tagnya harus memuat nilai $\alpha$ (`s0_a05`), kalau tidak
sel-sel $\alpha$ berbeda akan saling menimpa berkas keluaran.

### Turunkan seluruh matriks dari config

```bash
python scripts/gen_perintah.py --arm bench                 # 36 sel
python scripts/gen_perintah.py --arm factor                # 11 sel
python scripts/gen_perintah.py --arm interpolasi           #  6 sel

# Runner antrean: menjaga jumlah slot, menunggu VRAM bebas sebelum start sel
# baru, dan mengantre ulang sel yang OOM di belakang antrean.
python scripts/jalankan_matriks.py --arm all --slots 2
python scripts/jalankan_matriks.py --arm all --dry-run     # lihat rencananya dulu
```

### Probe geometri (murah, tak ada generasi teks)

```bash
PYTHONPATH=backend python backend/eval/b7_probe.py \
    --model Qwen/Qwen3-8B --steps 10 --alphas 0.25,0.5,0.75
```

Mengukur `max_v cos(z, W_in[v])` untuk kelima formulasi **dan** sepanjang sumbu
$\alpha$ dalam satu jalankan. Hitungan menit: hanya rollout laten, nol token
teks. Keluarannya menyediakan sumbu-x kurva dose–response — tanpa ini, sumbu
itu hanya bisa diasumsikan linier, dan asumsi itu tak diuji.

### Analisis (tanpa GPU)

```bash
python backend/bench/compare.py --out results/bench/analisis.json   # McNemar + CI bootstrap
PYTHONPATH=backend python backend/eval/rescore_all.py --workers 4    # skor ulang korpus faktor (2021)
PYTHONPATH=backend python backend/eval/skor_holdout.py --workers 4   # level 6: jendela 2022-2025
python backend/eval/compare_modes.py                                # probe kapasitas kanal (Tahap 0)
python scripts/rakit_transkrip.py                                   # transkrip -> 1 Markdown per sel
```

> **`--workers`** memparalelkan evaluasi ekspresi. Proses pekerja mewarisi satu
> salinan data pasar lewat `fork` copy-on-write — aman karena jalur skoring
> hanya membaca — jadi RAM tidak berlipat sejumlah pekerja. Anggaran memori:
> data pasar jendela 4 tahun ≈ 1,1 GB dibagi bersama, plus ≈ 0,5–1 GB sementara
> per pekerja untuk ekspresi rolling-bersarang. Di mesin 8 GB, `--workers 3`
> aman; `--workers 4` kalau ≥ 12 GB. `skor_holdout` auto-memilih `min(4, ncpu)`;
> `rescore_all` default serial (naikkan manual untuk korpus besar). Set
> `LAB_MAX_WORKERS=1` bila memakai `--workers` — joblib di dalam pekerja
> otomatis turun ke satu utas, jadi tak ada fan-out bersarang.

> `rescore_all.py` **menimpa** field `ic` di `frontend_*.json` (itu memang
> tujuannya: memverifikasi angka dokumen bisa direproduksi di CPU).
> `skor_holdout.py` **tidak** — ia bekerja pada salinan dan menulis ke berkas
> sendiri, karena menimpa angka seleksi 2021 dengan angka holdout akan
> menghancurkan seluruh dasar Bab IV. Jangan menyatukan keduanya.

---

## 4. Kenapa GPU, dan seperti apa keluaran yang benar

### 4.1 Apa yang butuh GPU dan apa yang tidak

Yang dijalankan di GPU **hanya pembangkitan langkah laten dan teks oleh LLM** —
tiap langkah laten adalah satu *forward pass* Qwen3-8B, dan tiap agen teks
adalah satu `model.generate`. Itu saja. Semua penilaian dilakukan di CPU,
sengaja, supaya jam GPU tidak terbakar untuk aritmetika:

| kerja | di mana | kenapa |
|---|---|---|
| rollout $m$ langkah laten + generasi teks agen | **GPU** | forward pass model 8 B; tak ada jalan lain |
| geometri $M$ (‖M−I‖, cos(h,hM)) | CPU | `realign_probe.py` cuma membaca `safetensors`, tak menjalankan model |
| geometri vektor laten (`b7_probe`) | **GPU** | butuh vektor laten *produksi* yang hanya ada saat model berjalan |
| parsing ekspresi DSL, RankIC, backtest | CPU | `eval/ic.py` + `eval/backtest.py`, tervalidasi identik 7 desimal vs produksi |
| uji statistik (McNemar, Cochran, bootstrap, Holm) | CPU | `bench/compare.py`, murni pandas/numpy |
| skor holdout 2022–2025 | CPU | `skor_holdout.py`; lihat §3 soal `--workers` |
| rakit transkrip, tabel LaTeX | CPU | `scripts/`, `../analisis/` |

**Konsekuensi praktis:** sesi cloud dengan GPU sebaiknya menjalankan **hanya**
sel matriks (`run_bench.py`, `run_factor.py`) dan `b7_probe.py`, lalu
mematikan GPU. Skoring (`rescore_all.py`, `skor_holdout.py`, `compare.py`) bisa
— dan sebaiknya — dijalankan setelahnya di mesin CPU biasa.

`scripts/jalankan_matriks.py` sudah menegakkan pemisahan ini untuk lengan
faktor: ia menambahkan `--skip-score` ke tiap sel (GPU menulis
`frontend_<tag>.json` lalu langsung keluar, kartu bebas untuk sel berikutnya),
lalu menjalankan satu lewatan `--score-only` di CPU setelah semua sel GPU
selesai. Kalau menjalankan `run_factor.py` sendiri tanpa runner, tambahkan
`--skip-score` manual bila ingin perilaku yang sama; tanpanya ia menskor
in-line di proses yang sama (menahan GPU beberapa menit per sel).

### 4.2 Berkas yang dihasilkan tiap jenis run

| perintah | menulis | isi inti |
|---|---|---|
| `run_bench.py --task T --latent-mode M --comm-mode C --tag s0` | `results/bench/bench_T_M_C_s0.json` + `results/bench/llm_outputs/bench_T_M_C_s0/session_*/` | ringkasan sel + hasil per soal + transkrip tiap panggilan LLM |
| `run_bench.py ... --latent-mode mix --latent-alpha A --tag s0_aXX` | `results/bench/bench_T_mix_C_s0_aXX.json` | sama; `_meta.latent_alpha` terisi |
| `run_factor.py --comm-mode C --latent-mode M --tag C_M` | `results/factor/frontend_C_M.json` + `results/factor/llm_outputs/C_M/session_*/` | `args` + daftar `runs` (satu per arah×seed), tiap run berisi ekspresi mentah, verdikt gate, `agent_trace` |
| `b7_probe.py --alphas ...` | `results/probe/b7_probe_Qwen_Qwen3-8B.json` | `inertness` (uji `use_realign`), `geometry` (5 mode), `geometry_mix` (kurva $\alpha$) |
| `eval/skor_holdout.py` | `results/factor/holdout_<awal>_<akhir>.json` + `results/.cache/holdout_cache_*.json` | `per_tag` + `per_ekspresi` dengan ic seleksi vs holdout berdampingan |
| `eval/rescore_all.py` | **menimpa** `results/factor/frontend_*.json` + `results/factor/icseries_*.parquet` | mengisi field `ic`/`bt_*` tiap ekspresi + deret IC harian |

### 4.3 Skema ringkas keluaran

**Sel bench** — `{_meta, summary, results}`:

```
_meta    : task, latent_mode, latent_alpha, comm_mode, chain, model, limit,
           sample_seed, seed, total_time_s, ...            (konfigurasi lengkap run)
summary  : {n, n_correct, accuracy, format_rate}
results  : [ {index, question, gold, answer_text, prediction, correct,
              format_ok, duration_s, pipeline_error, error}, ... ]   (satu per soal)
```

**Sel faktor** — `{args, runs}`:

```
args : seluruh argumen CLI + latent_alpha
runs : [ {
    direction, seed, comm_mode, latent_mode,
    hypothesis,                          # hipotesis yang dibawa agen proposal
    factors : [ {name, expression, explanation,
                 sem_ok, flags,          # cacat semantik statis
                 ic, icir, tstat, n_days, coverage, n_unique, eval_error,
                 bt_ann_return, bt_sharpe, bt_max_drawdown, bt_turnover, ...,
                 passed_gate} ],
    passing : [ekspresi yang lolos gate],
    gate_log, gate_error, repaired, repair_attempts,
    construct_text_head,                 # 200 char pertama keluaran construct — cek prefiks
    agent_trace : [ {agent, mode, s, latent_s, gen_s, kv_len,
                     n_in_tok, n_out_tok, text_len, rep_ratio, parsed_ok, text} ]
  }, ... ]
```

Sebelum `rescore_all.py`/`skor_holdout.py` dijalankan, field `ic`/`bt_*` pada
`factors` **belum ada** — itu normal, bukan run yang gagal.

### 4.4 Bentuk keluaran yang SEHAT vs RUSAK

Ini yang paling sering salah baca oleh sesi baru. Sebuah run bisa "selesai
tanpa error" tetapi menghasilkan data yang tak berguna.

**Sel bench sehat:**
- `summary.n` **sama persis** dengan `--limit` (100). Kurang dari itu = pipeline
  jatuh di tengah; cek `results[*].pipeline_error`.
- `summary.format_rate` ≥ 0,95. Di bawah itu, model gagal menghasilkan format
  yang bisa dinilai — biasanya kerusakan langkah laten.
- `summary.accuracy` dalam rentang yang masuk akal: GSM8K/ARC-C 0,83–0,94,
  HumanEval+ 0,69–0,76 untuk keluarga relaksasi. **`raw` di HumanEval+ ~0,42
  adalah temuan, bukan bug** — jangan "perbaiki".
- Sebar buka 2–3 `results[*].answer_text`: harus kalimat Inggris wajar. Aksara
  Tionghoa yang menyusup, spasi hilang, suku kata berulang = korupsi token
  (diharapkan sesekali pada `raw`, alarm bila pada mode lain).

**Sel faktor sehat:**
- Tiap `runs[*].agent_trace` punya 3 entri agen (`proposal`, `innovate`,
  `construct`) + mungkin `repair`. `construct` harus `parsed_ok: true` pada
  mayoritas jalan.
- `runs[*].construct_text_head` **dimulai dengan `{`**. Kalau dimulai dengan
  `"` atau langsung nama field, prefill tidak bekerja — cek `prefill:` di
  `backend/prompts/factor.yaml`.
- `gate_error: "no expression from construct"` pada **semua** jalan satu sel =
  sel itu gagal total (inilah gejala `kv_and_text` sebelum perbaikan prefill).
  Pada beberapa jalan saja = wajar.
- Setelah skoring: sel keluarga relaksasi menghasilkan ~15–30 ekspresi ber-`ic`
  per 20 jalan; `raw` jauh lebih sedikit (itu temuan).

**`b7_probe` sehat:**
- `geometry` memuat **lima** kunci: `raw, soft, gumbel, sample, moi`. Kalau
  `moi` hilang, `ALL_MODES` belum diperbarui.
- `geometry.sample.max_cos_embed_mean` = **1,000 persis** — uji kewarasan
  pipeline; kalau bukan 1, probenya rusak, bukan modelnya.
- `geometry.raw.max_cos_embed_mean` ≈ 0,31; keluarga relaksasi 0,93–1,00.
- `geometry_mix` memuat kurva dari `0.0` (= `raw`) sampai `1.0` (= `soft`).

**`skor_holdout` sehat:**
- `per_tag[*].berpasangan` > 0 untuk sel yang punya ekspresi hidup (artinya ada
  ekspresi yang bisa dibandingkan seleksi↔holdout).
- `n_ekspresi_unik` mendekati jumlah ekspresi unik di korpus (≈150 untuk run
  6-jalan; lebih banyak untuk 20-jalan).

### 4.5 Verifikasi cepat pasca-run

```bash
# Bench: satu sel
python -c "import json; d=json.load(open('results/bench/bench_gsm8k_gumbel_kv_s0.json'));
print('n', d['summary']['n'], '| acc', d['summary']['accuracy'],
      '| fmt', d['summary']['format_rate'],
      '| errors', sum(1 for r in d['results'] if r['pipeline_error']))"

# Faktor: semua sel sekaligus
python -c "
import json, glob
for f in sorted(glob.glob('results/factor/frontend_*.json')):
    d=json.load(open(f)); runs=d['runs']
    head_ok=sum(1 for r in runs if (r.get('construct_text_head') or '').lstrip().startswith('{'))
    parsed=sum(1 for r in runs for t in (r.get('agent_trace') or [])
               if t['agent']=='construct' and t.get('parsed_ok'))
    print(f'{f.split(\"/\")[-1]:34s} runs={len(runs):2d} head-{{={head_ok:2d} construct-parsed={parsed:2d}')
"

# compare.py sudah punya verifikasi bawaan: ia MENGELUARKAN sel yang sidik
# jari soalnya tak cocok dan mencetak alasannya.
python backend/bench/compare.py --out results/bench/analisis.json
```

---

## 5. Status matriks dan urutan menjalankan berikutnya

Diperbarui **2026-08-27**. Sebuah sesi baru harus membaca bagian ini sebelum
menyalakan GPU, supaya tidak menjalankan ulang sel yang sudah ada atau
melewatkan sel yang belum.

| bagian | status | catatan |
|---|---|---|
| bench, 21 sel utama (3 tugas × 7 perlakuan, 100 soal) | **selesai** 2026-08-10 | tak perlu diulang |
| bench `kv_and_text`, 15 sel | selesai, 5 soal | pemasok transkrip lampiran; di luar semua tabel/uji |
| faktor, 11 sel × 6 jalan | selesai 2026-08-10 | **akan digantikan** run 20 jalan |
| probe geometri (`b7_probe`) | selesai TAPI **tanpa `moi`** | wajib diulang |
| probe kapasitas kanal (`compare_modes`) | selesai, `moi` ikut | tak perlu diulang |
| skor holdout 2022–2025 | selesai 2026-08-27 (CPU) | `results/factor/holdout_*.json` |
| interpolasi (`mix`) | **belum pernah dijalankan** | 6 sel |

**Urutan yang disarankan** — menaik menurut biaya, dan tiap tahap menghasilkan
angka yang berguna sendiri walau tahap berikutnya batal:

```bash
# 1. Probe geometri — menit. Menutup lubang `moi` DAN memberi sumbu-x kurva.
PYTHONPATH=backend python backend/eval/b7_probe.py \
    --model Qwen/Qwen3-8B --steps 10 --alphas 0.25,0.5,0.75

# 2. Lengan faktor 20 jalan/sel — ±8,5 jam waktu-sel, ±4,5 jam dinding @2 slot.
python scripts/gen_perintah.py --arm factor > /tmp/faktor.sh
python scripts/jalankan_matriks.py --arm factor --slots 2

# 3. Skoring CPU untuk hasil baru (tak butuh GPU — boleh di mesin lain).
#    Korpus 20 jalan/sel jauh lebih besar — pakai --workers.
LAB_MAX_WORKERS=1 PYTHONPATH=backend python backend/eval/rescore_all.py --workers 4
LAB_MAX_WORKERS=1 PYTHONPATH=backend python backend/eval/skor_holdout.py --workers 4

# 4. Interpolasi — ±2-3 jam dinding.
python scripts/jalankan_matriks.py --arm interpolasi --slots 2
```

**Yang sengaja TIDAK dijalankan**, dan alasannya, supaya tak ada yang
menambahkannya kembali karena mengira terlewat:

- *Replikasi multi-seed lengan bench.* Mahal, dan hasilnya hanya mempersempit
  selang kepercayaan pada selisih yang sudah dinyatakan tak signifikan. Ragam
  antar-seed tetap dilaporkan sebagai batas di Bab IV, bukan ditutupi.
- *Lengan faktor 40 jalan/sel.* 20 sudah lebih dari tiga kali lipat run
  sebelumnya; naik ke 40 menggandakan biaya untuk perbaikan presisi yang tak
  mengubah satu pun kesimpulan berjenjang.
- *`kv_and_text` sebagai sel uji penuh di lengan bench.* Ia pemasok transkrip
  (5 soal, subsampel berbeda) dan harus tetap di luar tabel.

### Perubahan perilaku yang memengaruhi perbandingan dengan run lama

Sejak 2026-08-27 agen `construct` memakai **prefill** (`prefill:` di
`backend/prompts/factor.yaml`): pembuka objek JSON dikirim sebagai bagian
giliran asisten. Sebabnya, pada `kv_and_text` agen itu mewarisi KV yang berakhir
dengan objek JSON utuh dari agen `innovate` (±4000 karakter) lalu melanjutkan
seolah masih di dalamnya — keluarannya mulai dari nilai hipotesis, tanpa
`{`, dan gagal diurai. Empat dari lima sel `kv_and_text` menghasilkan **nol**
ekspresi karena ini.

Itu **bukan** pemotongan oleh harness: `text_len` yang tercatat sama persis
dengan panjang yang dibangkitkan model. Diverifikasi ulang terhadap 146
keluaran `construct` yang tersimpan — pemulihannya menaikkan yang terurai dari
54 menjadi 104, dan **tidak mengubah satu pun sel yang sudah sehat** (penjaga di
`_CoreEngine._join_prefill` membuang prefill bila lanjutan model sudah dibuka
`{` sendiri).

> Konsekuensinya: sel faktor lama (6 jalan) dan sel faktor baru (20 jalan)
> **tidak dibangkitkan dengan prosedur yang sama**. Jangan menggabungkan
> keduanya dalam satu tabel. Run baru menggantikan yang lama, tidak menambahnya.

---

## 6. Verifikasi setup (CPU, tanpa GPU)

```bash
PYTHONPATH=backend python -c "
import llm.client, mas.pipeline, bench.pipeline, gate, dsl.expr_parser, eval.ic
print('import ok')
from eval.ic import Lab; lab = Lab(mode='fast')
print(lab.ic('RANK(\$volume)'))         # ~ <IC=-0.04493 t=-6.72 n=243 ...>
"
```

Jalur ini diverifikasi setelah perombakan: 243 hari OOS, ~4370 saham/hari,
IC identik dengan angka produksi lama.

---

## 7. Ganti model untuk VRAM terbatas

| Model | Unduh | VRAM bobot |
|---|---|---|
| `Qwen/Qwen3-4B` | ~8 GB | ~8 GB |
| **`Qwen/Qwen3-8B`** (dipakai skripsi) | ~16 GB | ~16 GB |
| `Qwen/Qwen3-14B` | ~28 GB | ~28 GB |

Ganti lewat `--model` di kedua runner dan `model:` di `configs/matriks.yaml`.
**Seluruh angka skripsi dipatok Qwen3-8B** — mencampur backbone membuat sel tak
sebanding.

---

## 8. Masalah yang sering muncul

**`uv` / `.venv` / model HF hilang setelah pod restart** — semuanya di `/root`
yang ephemeral. Pastikan `source /workspace/runpod_env.sh` dijalankan SEBELUM
`uv sync`, dan `XDG_*`/`UV_CACHE_DIR`/`HF_HOME` menunjuk ke `/workspace`.

**`We couldn't connect to 'https://huggingface.co'`** — set `HF_LOCAL_ONLY=0`
di `.env`, atau pre-download modelnya. Kalau `HF_TOKEN` di-set tapi sudah
kedaluwarsa, error 401 membuat transformers gagal total alih-alih jatuh ke
akses anonim — hapus tokennya.

**CUDA OOM** — turunkan `--max-new-tokens`, kurangi proses paralel, atau turun
ke Qwen3-4B. Cek `--empty-cache-every` di `run_bench.py`.

**`ModuleNotFoundError`** — jalankan dengan `PYTHONPATH=backend` dari root repo,
bukan dari dalam `backend/`.

**Skoring korpus faktor lambat / kena OOM** — skoring ekspresi DSL adalah beban
CPU dominan (rolling bersarang atas jutaan baris; jendela holdout 4 tahun ≈ 4x
data jendela seleksi). Dua kenop:

- **Kecepatan:** `--workers N` (lihat catatan di §3 "Analisis"). Terukur di
  mesin 16-core / 8 GB: jendela holdout, `--workers 3` memberi ≈ 2,4x terhadap
  serial, RAM puncak ≈ 4,6 GB (data pasar ≈ 1,1 GB dibagi bersama lewat fork
  COW + ≈ 1,1 GB per pekerja).
- **Memori:** `eval/ic.py` membatasi worker joblib internal lewat
  `LAB_MAX_WORKERS` (default 3); set `=1` saat memakai `--workers` (joblib di
  dalam pekerja `multiprocessing` otomatis turun ke satu utas — ada peringatan
  Loky yang tidak berbahaya).

Yang mahal sekali di awal: **membangun cache data pasar** untuk jendela baru —
`pd.read_hdf` memuat seluruh 14,2 juta baris sekaligus (puncak ≈ 0,8 GB), lalu
mengirisnya. Cache-nya ditulis sekali ke
`results/.cache/pv_fast_<awal>_<akhir>.parquet` dan dipakai ulang seterusnya.

**Sel `kv_and_text` lengan faktor menghasilkan nol ekspresi** — itu gejala yang
sudah dijelaskan dan diperbaiki; lihat §5 "Perubahan perilaku" dan §4.4. Kalau muncul
lagi setelah perbaikan, periksa bahwa `prefill:` masih ada di
`backend/prompts/factor.yaml` pada agen `construct`, dan bahwa
`_CoreEngine._join_prefill` tidak membuangnya karena keluaran model kebetulan
diawali `{`.

---

## 9. Apa yang dihapus saat perombakan

Semuanya masih ada di `exp/alt3-gumbel-fidelitas`, branch `prod/*`, dan di
riwayat `main` sebelum commit perombakan.

| dihapus | alasan |
|---|---|
| `backend/pipeline/` (evolusi, loop, planning, factor_mining) | lengan faktor kini single-pass; evolusi menambah variabel perancu |
| `backend/coder/costeer/`, `backend/core/` | kerangka RD-Agent; hanya `core/conf.py` yang tersisa → `backend/conf.py` |
| `backend/log/` (wrapper `rdagent.log`) | diganti `backend/qlog.py` (loguru polos) — repo tak lagi butuh RD-Agent |
| `backend/factors/` selain DSL + regulator + template | proposal/runner/feedback/qlib terikat ke pipeline lama |
| `backend/eksternal/`, `app/`, `components/`, `debug/`, `experiments/` | agen makro/berita & harness lama, di luar pertanyaan branch ini |
| `backend/runs/`, `backend/log/<ts>/`, `try/`, `books/` | artefak run lama (±440 MB) |
| `configs/experiment*.yaml`, `backtest.yaml` | dibaca pipeline evolusi yang dihapus; diganti `configs/matriks.yaml` |
| `launcher.py`, `backend/cli.py`, `main.py` | CLI RD-Agent; diganti dua runner + `scripts/gen_perintah.py` |
