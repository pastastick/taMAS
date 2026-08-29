# Pengalaman Sebelumnya: Membangun Sistem Multi-Agent (QuantaLatent)

> **Status per 25 Agustus 2026 — dibaca ulang dan sebagian sudah usang.**
> Yang masih berlaku: seluruh bagian *kekuatan* dan *kelemahan pola kerja* di
> bawah — itu penilaian perilaku, dan masih akurat.
> Yang sudah TIDAK berlaku: deskripsi teknisnya. Sejak dokumen ini ditulis,
> backbone pindah Qwen3-4B → **Qwen3-8B**, lapisan evolusi
> (mutation/crossover/feedback) **dilepas seluruhnya**, dan arah penelitian
> berganti dari "KV vs teks untuk faktor alpha" menjadi **perbandingan lima
> formulasi langkah laten** pada tiga benchmark + satu domain simbolik.
> Arsitektur `prod/backend-v1` yang disebut di bawah sudah digantikan isi
> `main`. Keadaan sekarang: `quantalatent/README.md` dan
> `quantalatent/docs/DESAIN_EKSPERIMEN.md`.
>
> Disimpan karena bagian perilakunya masih dipakai; empat berkas lain di
> direktori ini dihapus 25 Agustus 2026 karena isinya sudah tak berlaku sama
> sekali.

> Ditulis 2026-07-14 sebagai konteks lintas-proyek. Proyek baru yang saya (user) kerjakan
> setelah ini **bisa jadi berbeda total** dari yang dijelaskan di bawah — anggap ini sebagai
> *pengalaman yang saya bawa*, bukan requirement untuk proyek baru. Tapi pola kerja, kekuatan,
> dan kelemahan saya sebagai pembangun sistem multi-agent kemungkinan besar terulang.

## Apa yang dibangun

**QuantaLatent** = fusi dua paper riset ke satu sistem: **QuantaAlpha** (evolutionary
multi-agent LLM untuk mining faktor alpha saham — hipotesis→ekspresi simbolik→backtest→
evolusi mutation/crossover) dan **LatentMAS** (kolaborasi antar-agent lewat KV-cache/latent
memory, bukan lewat teks, memakai realignment ridge-regression W_a antar hidden-state).
Model lokal kecil (Qwen3-4B, HuggingFace murni via `inputs_embeds`, tanpa vLLM). Proyek ini
juga menjadi skripsi TA1 Statistika UGM — jantungnya adalah studi banding empiris: **apakah
hand-off antar-agent lewat KV-cache (laten) menghasilkan faktor alpha yang lebih baik/setara
dibanding hand-off lewat teks, pada model kecil?**

Arsitektur akhir (branch `prod/backend-v1`): pipeline depan `proposal→design→construct`
(agent `design`/Architect ditambahkan belakangan) → gate regulator (9 gate keras, bukan skor
kontinu) → repair agent → backtest (hybrid: RankIC per-faktor standalone + LightGBM combined
untuk portofolio) → feedback (evaluatif murni) → evolusi `mutation`(exploitation)/
`crossover`(exploration, pakai `kv_concat` sebagai operator rekombinasi hierarkis).

## Kekuatan yang terlihat dari proyek ini

1. **Debugging root-cause yang tekun dan jujur.** Ketika hasil terlihat buruk ("KV
   collapse"), tidak langsung menerima kesimpulan permukaan ("KV laten memang rusak untuk
   simbolik") — digali sampai ketemu bahwa itu **bug artefak** (dangling assistant turn di
   `client.py`, crop yang membuang jawaban sebelum di-chain ke agent berikutnya). Setelah
   di-fix, narasi dikoreksi ke penjelasan yang lebih tepat (diversity collapse karena
   `latent_pass` deterministik, bukan "lossy secara inheren"). Ini pola yang bagus dan harus
   dipertahankan: **curiga pada kesimpulan yang terlalu rapi, cari akar mekanistik sebelum
   menulis klaim final.**
2. **Disiplin invarian sistem non-trivial.** Paham betul bahwa `DynamicCache` dari HuggingFace
   itu mutable in-place — setiap KV yang dibaca >1 consumer wajib di-`deepcopy` dulu. Ini jenis
   bug kelas "silent corruption" yang mudah terlewat tapi ditangani sebagai aturan desain
   eksplisit sejak awal.
3. **Bersedia membongkar ulang saat premis salah**, bukan menambal di atas fondasi yang goyah
   (redesign v2→v3→v4 pipeline, evolution operator judger-only→guidance-reentry→
   judger-only lagi — tiap pivot didasari bukti GPU nyata, bukan tebakan).
4. **Menjaga jujur soal keterbatasan klaim**: berulang kali secara eksplisit menahan diri dari
   overclaim saat n=1 seed ("JANGAN overclaim, butuh ≥3 seed"), dan memisahkan temuan "faktor
   masih lemah secara absolut" dari pertanyaan riset utama "KV vs TEXT" agar tidak
   tercampur.

## Kelemahan & pola yang perlu diwaspadai di proyek baru

1. **Pivot lebih cepat daripada validasi selesai.** Log riwayat proyek ini menunjukkan pola
   berulang: desain baru dibangun di atas kertas + verifikasi statis (py_compile, dry-run,
   parser round-trip) sampai matang, lalu **redesign berikutnya dimulai sebelum GPU-test
   yang sebelumnya kelar**. Kalimat "BELUM GPU-tested" muncul berkali-kali di riwayat commit,
   dan versi pipeline berganti (v1→v2→v3→v4→judger-only→guidance-reentry→judger-only lagi)
   tanpa siklus closure yang konsisten. Risiko: energi habis di desain ulang, bukan di
   pembuktian empiris. **Di proyek baru: tahan godaan redesign sampai eksperimen berjalan
   yang sedang dikerjakan benar-benar diuji end-to-end**, bahkan kalau hasilnya jelek —
   hasil jelek yang terukur lebih berharga daripada desain baru yang belum diuji.
2. **Churn nama branch/eksperimen tinggi.** 12+ branch (`feat/latentmas-rework`,
   `experiment/prompt-bench`, `exp/v4-eval`, `newEvol`, `prod/quantalatent-v1`,
   `prod/backend-v1`, dst.), beberapa di-rename di tengah jalan, submodule pointer yang sering
   tertinggal dari branch aktifnya. Enak untuk eksplorasi cepat, tapi butuh effort ekstra untuk
   tahu "versi mana yang sebenarnya current" — pernah butuh baca ulang riwayat panjang hanya
   untuk menjawab itu. **Mitigasi yang sudah mulai dipakai dan terbukti membantu**: satu file
   "source of truth" per topik (mis. `MONITORING_NOTES.md`, `GUIDE.md` di promptbench) yang
   dibaca duluan di sesi baru — pertahankan kebiasaan ini dari awal proyek baru, jangan
   tunggu sampai kebingungan dulu baru dibuat.
3. **Ambang GPU sebagai bottleneck berulang.** Banyak pekerjaan besar (prompt redesign,
   evolution rewrite, hybrid backtest) diselesaikan secara statis lalu masuk antrian
   "SISA: run GPU" — dan antrian itu menumpuk sebelum sempat dieksekusi karena keburu
   pivot desain lagi (lihat poin 1). Kalau proyek baru juga bergantung pada resource mahal/
   terbatas (GPU, API berbayar, environment eksternal), rencanakan validasi *lebih dini dan
   lebih sering* dalam siklus kecil, bukan menumpuk banyak perubahan lalu validasi sekali di
   akhir.
4. **Kecenderungan menambah kompleksitas arsitektur (agent baru, mode baru, layer evolusi
   baru) sebelum versi sebelumnya benar-benar dibuktikan bekerja.** Contoh: agent `design`
   ditambahkan, `comm_mode` tiga-nilai (kv/kv_and_text/text) ditambahkan, fan-out 22-node
   dibangun — semua sebelum baseline sebelumnya lulus uji GPU penuh. Powerful untuk riset
   eksploratif, tapi mahal kalau tujuannya adalah *shipping* sesuatu yang stabil.
5. **Klaim awal yang harus dikoreksi setelah observasi lebih lanjut** ("KV lossy untuk
   simbolik" → ternyata artefak bug) menunjukkan risiko: kesimpulan ditulis (termasuk ke
   dokumen skripsi) sebelum akar masalah benar-benar dikonfirmasi. Sudah dikoreksi dengan
   baik, tapi pelajarannya: **tulis kesimpulan besar paling akhir, setelah bug-bug jelas
   disingkirkan**, bukan di tengah proses debugging.

## Kesalahan teknis konkret yang berulang di proyek ini (pola, bukan detail sekali pakai)

- Bug "hilangnya konten yang baru digenerate saat di-crop dari KV sebelum di-chain ke agent
  berikutnya" — pelajaran umum: **kalau ada operasi "ringkas/pangkas" state sebelum
  diteruskan ke consumer berikutnya, pastikan konten yang penting untuk konsumer itu
  benar-benar bertahan**, jangan asumsikan.
- Format tax pada model kecil: instruksi output JSON menurunkan akurasi dibanding format
  kata kunci sederhana (NAME/DESC/VARS/EXPR) pada Qwen3-4B. Pelajaran umum untuk model kecil:
  **format output yang "mahal secara sintaks" (JSON bersarang, escape karakter) punya biaya
  akurasi nyata** — pilih format paling sederhana yang masih bisa di-parse dengan andal.
  Ini kemungkinan tidak berlaku sama untuk model besar (GPT/Claude tier atas) — perlu
  diuji ulang, jangan diasumsikan berlaku universal di proyek baru.
- Mode collapse dari contoh di system prompt: SEMUA hipotesis lintas sesi ternyata jadi
  parafrase satu contoh yang ada di prompt (anchoring), bukan eksplorasi asli. Pelajaran
  umum: **contoh konkret di prompt itu pisau bermata dua** — membantu format, tapi model
  kecil cenderung meniru verbatim isi contoh, bukan hanya strukturnya. Perlu daftar keluarga
  mekanisme/kategori (bukan satu contoh tunggal) atau larangan eksplisit "jangan tiru pola
  X" untuk mencegah ini.
- Sampling deterministik di jalur laten (tidak ada temperature saat "berpikir", hanya saat
  emit teks) menyebabkan hasil antar-trajectory nyaris identik → operasi evolusi (crossover)
  jadi mendaur ulang faktor yang sama alih-alih menjelajah. Pelajaran umum: **kalau
  arsitektur butuh diversitas (evolutionary search, ensemble, multi-agent exploration),
  cek eksplisit apakah tiap jalur benar-benar stokastik** — determinisme yang tak disengaja
  di satu titik bisa membunuh diversitas di seluruh sistem tanpa gejala yang jelas.

## Cara kerja preferensi (relevan untuk kolaborasi di proyek baru)

- Suka penjelasan **depth-first, per-bagian, walkthrough kode baris-per-baris** dalam Bahasa
  Indonesia, lompat ke file yang di-import sebelum lanjut — bukan ringkasan tingkat tinggi
  duluan.
- Nyaman dengan siklus **hipotesis→temuan→koreksi terbuka** dan senang saat asisten
  menantang klaim yang terlalu kuat, bukan cuma menyetujui.
- Terbiasa membangun harness verifikasi statis dulu (py_compile, dry-run, parser round-trip,
  validasi YAML) sebelum menyentuh GPU/resource mahal — kebiasaan bagus, pertahankan.
- Menjaga satu file "source of truth" per topik yang harus dibaca di awal sesi baru — pola
  ini terbukti berguna dan sepadan diterapkan sejak awal proyek baru (bukan ditambahkan
  belakangan setelah riwayat sudah rumit).

## Ringkasan satu-paragraf (kalau butuh versi pendek)

Saya (user) sudah pernah membangun sistem multi-agent LLM nyata yang cukup kompleks
(QuantaLatent: fusi QuantaAlpha + LatentMAS, KV-cache latent collaboration, evolutionary
factor mining, Qwen3-4B lokal) dan cukup kuat di debugging root-cause serta kejujuran
epistemik (mengoreksi klaim sendiri saat bukti baru muncul). Kelemahan utama saya adalah
**kecenderungan mendesain ulang sebelum desain sebelumnya selesai divalidasi end-to-end**,
churn branch/nama yang tinggi, dan menumpuk pekerjaan besar sebelum sempat divalidasi pada
resource yang mahal (GPU). Di proyek baru, tolong bantu saya menahan diri dari pivot
prematur dan dorong closure/validasi lebih sering dalam siklus yang lebih kecil.
