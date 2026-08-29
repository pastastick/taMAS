# Desain Eksperimen — Empat Persamaan Langkah Laten pada Kolaborasi Multi-Agen Sekuensial

> Branch `exp/empat-metode-v1`, dibuat 2026-08-10 dari `exp/alt3-gumbel-fidelitas`
> (`99721ec`). Dokumen ini menetapkan APA yang diukur dan KENAPA. Cara
> menjalankannya ada di `../README.md`; angka Tahap 0 yang mendasari branch ini
> ada di `HASIL_TAHAP0.md`.

## 1. Pertanyaan penelitian

Tahap 0 (`HASIL_TAHAP0.md`) menjawab satu pertanyaan sempit di satu domain:
pada kanal laten murni yang membawa **muatan simbolik** (nama fungsi DSL),
persamaan langkah laten resmi LatentMAS (`raw`, ridge $W_a$) memberi recall
**0,000 mutlak**, sementara empat varian yang memproyeksikan balik ke convex
hull embedding (`soft`, `gumbel`, `sample`, `moi`) semuanya mendarat di
0,34–0,38 (m=10) dan 0,72–0,87 (m=40). Perbedaan antar keempat varian itu
sendiri **tidak signifikan** pada n=20.

Branch ini membawa temuan itu keluar dari domainnya:

> **Apakah keunggulan keluarga relaksasi diskret atas ridge $W_a$ bertahan di
> luar tugas simbolik — yaitu pada benchmark penalaran umum tempat LatentMAS
> asli dievaluasi — atau ia spesifik untuk muatan simbolik?**

Dua kemungkinan hasil, dan **keduanya adalah temuan yang bisa dipublikasikan**:

- **Bertahan.** Keputusan "proyeksikan langkah laten ke ruang embedding" adalah
  perbaikan umum atas mekanisme LatentMAS, bukan tambalan domain.
- **Tidak bertahan.** Maka ada **disosiasi**: kanal laten LatentMAS memadai
  untuk membawa *gist* (rencana, arah penalaran) tetapi gagal membawa *simbol*
  (nama fungsi, identifier, ekspresi) — dan gagalnya total, bukan bertahap. Ini
  klaim yang lebih tajam dan persis mengisi celah yang disebut survei
  arXiv:2606.05711 §7.4 dan audit kausal arXiv:2607.26773.

Perhatikan bahwa hipotesis ini **tidak bisa dipatahkan dengan mengganti
varian**, karena yang diuji adalah keluarganya, bukan satu algoritma unggulan.
Itu keputusan yang diambil setelah Tahap 0B menunjukkan tak ada varian yang
terbukti menang (`HASIL_TAHAP0.md` §8.4d).

## 2. Dua sumbu

### Sumbu A — persamaan langkah laten ("latent realignment")

Pemetaan hidden state → vektor yang diumpankan sebagai `inputs_embeds`.
Implementasi: `backend/llm/client.py::_CoreEngine._latent_step_vec`.

| mode | persamaan | asal |
|---|---|---|
| `raw` | $z = M h$ (ridge $W_a$) | **LatentMAS**, Teorema A.1 (arXiv:2510.04646) |
| `gumbel` | $z = \mathrm{softmax}((\log\pi + g)/\tau)\,W_\text{in}$, $g\sim$Gumbel | **Stochastic Soft Thinking**, arXiv:2508.03440 Eq. 4 |
| `moi` | $w = [H p + (\beta{+}1{-}H)\,\mathbb{1}_y]/(\beta{+}1)$, $z = w W_\text{in}$ | **Mixture of Inputs**, arXiv:2505.14827 (NeurIPS 2025) |
| `sample` | $y\sim\mathrm{Cat}(\mathrm{softmax}(\ell/T))$, $z = W_\text{in}[y]$ | dekode kategoris standar (baseline di ketiga paper) |
| `soft` | $z = \mathrm{softmax}(\ell/T)\,W_\text{in}$ | **Soft Thinking** |

Himpunan yang dibandingkan dinyatakan sebagai

$$\mathcal M = \{\texttt{raw},\ \texttt{soft},\ \texttt{sample},\ \texttt{gumbel},\ \texttt{moi}\},
\qquad
\mathcal R = \mathcal M \setminus \{\texttt{raw}\}.$$

$\mathcal R$ adalah **keluarga relaksasi diskret**: setiap anggotanya membentuk
$z$ sebagai kombinasi konveks baris $W_\text{in}$, sehingga hasilnya selalu di
dalam convex hull embedding. `raw` satu-satunya yang di luar, karena fungsi
objektif ridge tak memaksa koefisiennya taknegatif dan berjumlah satu.

> **Perubahan 2026-08-27.** Sampai tanggal itu `soft` dicatat sebagai
> "kontrol, bukan salah satu dari empat". Label itu dibatalkan karena
> bertentangan dengan analisis yang sudah terbit: kontras keluarga-vs-`raw`
> merata-ratakan keempat anggota $\mathcal R$ termasuk `soft`, dan uji Cochran
> $Q$ dijalankan dengan $k=5$. Peran `soft` sebagai pemisah dua efek — "berada
> di manifold embedding" (dimiliki `soft`) versus "entropi tambahan" (dimiliki
> `gumbel`) — tetap berlaku sebagai **cara membaca hasil**, bukan sebagai
> status desain yang berbeda.

Catatan penting dari Tahap 0 yang harus diulang di skripsi: flag `use_realign`
**inert** di luar mode `raw`, dan bahkan di dalam `raw` ia tak mengubah apa pun
— `raw` dan `raw(M=I)` identik bit-per-bit. Karena itu realignment **bukan**
faktor terpisah dalam desain ini; ia adalah Sumbu A itu sendiri.

### Sumbu B — medium komunikasi antar-agen

Rantai selalu **sekuensial**. Yang berubah hanya apa yang mengalir antar-agen.
Implementasi: `backend/bench/pipeline.py` dan `backend/mas/pipeline.py`, dengan
semantik `_agent_mode` yang identik di keduanya.

| medium | handoff | agen perantara | langkah laten |
|---|---|---|---|
| `text` | teks keluaran agen hulu | emit teks | **tidak ada** |
| `kv_and_text` | KV-cache | emit teks | ada |
| `kv` | KV-cache | laten murni (tak emit teks) | ada |
| `baseline` | — (satu agen) | — | tidak ada |

**Konsekuensi desain yang mudah salah dan mahal:** pada `text` tidak ada
langkah laten sama sekali, jadi keempat nilai Sumbu A menghasilkan sel yang
**identik**. `text` karena itu dijalankan **sekali**, bukan empat kali. Empat
salinan angka yang sama akan terbaca sebagai empat pengamatan independen di
tabel dan membuat uji statistiknya salah. `scripts/gen_perintah.py` menegakkan
aturan ini.

## 3. Dua lengan yang setara

Eksperimen berdiri di atas **dua lengan**, bukan satu lengan utama plus satu
tambahan. Keduanya memakai mesin laten yang sama persis, sehingga selisih yang
terukur tak bisa dituduh berasal dari implementasi yang berbeda.

| lengan | pertanyaan yang dijawab | rantai |
|---|---|---|
| **bench** (replikasi LatentMAS) | apakah agen masih bisa **bernalar** setelah teks dihapus dari handoff | planner → critic → refiner → judger |
| **faktor** (generasi DSL) | apakah agen masih bisa **membawa struktur yang harus tepat** | proposal → innovate → construct |

Lengan bench:

| kategori | benchmark | n | metrik utama |
|---|---|---:|---|
| math & science reasoning | **GSM8K** (`openai/gsm8k`, test) | 1319 | exact-match `\boxed{}` |
| commonsense reasoning | **ARC-Challenge** (`allenai/ai2_arc`, test) | 1172 | exact-match `\boxed{}` |
| code generation | **HumanEval+** (`evalplus/humanevalplus`, test) | 164 | pass@1 (eksekusi tes) |

Lengan faktor adalah **stress test** yang membedakan skripsi ini dari
replikasi: ia menguji medium yang sama pada muatan yang seluruhnya simbolik,
tempat Tahap 0 sudah menunjukkan kanal laten resmi gagal total. Ia juga lengan
dengan data terkaya — `agent_trace` merekam per-agen (`kv_len`, token masuk /
keluar, waktu laten / generasi, `parsed_ok`), sehingga kerusakan bisa dilacak
sampai ke hop mana ia muncul. Lengan bench tak menyimpan itu.

> **Perubahan 2026-08-27.** Dokumen ini sebelumnya menyebut lengan faktor
> sebagai "tugas keempat". Penyebutan itu diganti: ia lengan uji tersendiri
> dengan enam level bukti (lihat §4d), bukan satu tugas di antara empat.

Subsample `--limit 200` dengan `--sample-seed` yang sama di semua sel. Soal
yang sama untuk semua metode adalah syarat uji berpasangan; `bench/compare.py`
**memverifikasi** kesamaan itu lewat sidik jari, tidak mengasumsikannya.

## 4. Metrik

Tiga lapis, sengaja tidak diringkas jadi satu angka.

**(a) Metrik paper LatentMAS** — `accuracy` (exact-match / pass@1) dan
`total_time_s`.

**(b) Metrik proyek ini** (dibawa dari Tahap 0–4, alat sudah ada dan tervalidasi):

| metrik | alat | apa yang dijawab |
|---|---|---|
| `format_rate` | `bench/scoring.py` | keandalan FORMAT, terpisah dari mutu jawaban |
| recall / exact kanal | `eval/channel_capacity.py` | berapa muatan simbolik lolos kanal laten |
| fidelitas hop | `eval/fidelity.py` | apakah hipotesis bertahan utuh proposal→…→construct |
| geometri realignment | `eval/realign_probe.py`, `eval/b7_probe.py` | apakah $M$ benar-benar memutar hidden state |

`format_rate` menjawab disosiasi yang dicatat `HASIL_TAHAP4.md` §B2: mode
langkah laten memperbaiki **keandalan** (lolos gate 54%→91%) tanpa memperbaiki
**kekuatan sinyal** ($t=-0{,}27$). Digabung ke satu angka akurasi, dua efek itu
tak bisa dipisahkan lagi.

**(c) Metrik backtest** (lengan faktor) — `eval/ic.py` + `eval/backtest.py`:
RankIC, ICIR, t-stat, lalu return tahunan, Sharpe, max drawdown, turnover,
hit-rate dari portofolio desil long–short dollar-neutral.

> **Batas yang harus dinyatakan di skripsi.** Backtest ini kasar: tanpa
> slippage, tanpa batas likuiditas, tanpa aturan suspensi, rebalance harian
> penuh. `turnover` dilaporkan justru supaya besarnya biaya yang diabaikan
> terlihat. Perbandingan antar-metode tetap sah karena semua metode dinilai
> pipeline yang sama persis — yang tidak sah adalah membaca angkanya sebagai
> ramalan keuntungan.

**(d) Enam level bukti lengan faktor.** Dengan 20 jalan per sel, uji hipotesis
formal tak akan punya daya. Yang dipakai adalah bukti berjenjang yang saling
menguatkan, disusun dari yang paling tak bergantung data pasar ke yang paling
bergantung — sehingga temuan intinya tidak bisa dituduh bergantung pada mutu
backtest:

| level | ukuran | alat |
|---|---|---|
| 1 keandalan | `parse_rate` keluaran agen `construct` | `analisis/` |
| 2 eksekusi | `evaluable_rate` (ekspresi bisa dievaluasi) | `eval/ic.py` |
| 3 fidelitas | lolos gate, korupsi token, fidelitas hop | `gate/`, `eval/fidelity.py` |
| 4 keberagaman | ekspresi unik, cakupan fungsi DSL, klaster sinyal | `eval/rescore_all.py` |
| 5 mutu prediktif | RankIC, ICIR, t-stat pada jendela seleksi 2021 | `eval/ic.py` |
| 6 ketahanan | RankIC pada holdout 2022–2025, berbalik tanda | `eval/skor_holdout.py` |

Level 6 memakai jendela yang **tak pernah** dipakai menyaring ekspresi mana
pun. Tanpa level itu, pembaca berhak bertanya apakah ekspresi yang dilaporkan
membawa sinyal atau hanya kebetulan cocok pada 2021.

**(e) Efisiensi per formulasi.** Token keluaran dan waktu dilaporkan **per
formulasi**, bukan hanya sebagai rentang sel terbaik. Rentang agregat menjawab
"apakah laten lebih murah dari teks"; ia tidak menjawab "formulasi mana yang
murahnya berbeda", padahal biaya berhubungan langsung dengan berapa banyak
teks yang harus dihasilkan ulang ketika keluaran rusak.

## 5. Uji statistik

Semua perbandingan **berpasangan**, lewat `backend/eval/stats.py` yang dipakai
kedua lengan (satu implementasi, bukan dua yang bisa melenceng):

- biner (benar/salah per soal, exact-match) → **McNemar eksak**
- kontinu/ordinal (recall, IC) → **Wilcoxon signed-rank**
- selalu dilaporkan bersama **CI bootstrap 95%**, karena pada n kecil nilai p
  sendirian menyesatkan ke dua arah

## 6. Yang TIDAK dilakukan, dan kenapa

- **Tidak ada loop evolusi.** Lengan faktor single-pass. Loop evolusi menambah
  variabel perancu (seleksi induk, tekanan keragaman, memori negatif) yang tak
  ada padanannya di lengan benchmark, sehingga kedua lengan tak lagi sebanding.
  Kodenya ada di branch `exp/alt3-gumbel-fidelitas`.
- **Tidak mengklaim satu varian menang.** Tahap 0B: tak satu pun keunggulan
  antar-varian mencapai signifikansi pada n=20 (`HASIL_TAHAP0.md` §8.3–8.4).
- **Tidak mengklaim menemukan Gumbel untuk langkah laten.** Sudah ada
  (arXiv:2508.03440), untuk single-model. Yang baru: memindahkannya ke
  kolaborasi laten multi-agen, dan alat ukur kapasitas kanal simboliknya.
- **Satu backbone (Qwen3-8B).** Generalisasi lintas ukuran model tidak diuji
  dan tidak boleh diklaim.

## 7. Peta kode

```
backend/
  llm/        mesin: model, KV, latent_pass, _latent_step_vec  ← SUMBU A ada di sini
  mas/        agen + KV ops + pipeline faktor                  ← SUMBU B
  bench/      lengan replikasi LatentMAS (data/scoring/pipeline/run_bench/compare)
  factor/     lengan faktor alpha (run_factor.py)
  dsl/        parser ekspresi, AST, pustaka fungsi
  gate/       gate mutu ekspresi (regulator, arity, redundansi)
  eval/       ic.py backtest.py stats.py fidelity.py channel_capacity.py …
  prompts/    factor.yaml (QuantaLatent) · bench.yaml (port LatentMAS)
reference/    LatentMAS @9a9e4d3 · mixinputs @7aef34b (rujukan, read-only)
```
