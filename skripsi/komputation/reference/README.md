# reference/ — kode rujukan, READ-ONLY

Salinan kode dua paper yang dibandingkan skripsi ini, dipatok pada commit
tertentu supaya klaim "algoritma kami setia pada paper X" bisa diperiksa tanpa
bergantung pada repo hulu yang bisa berubah.

| direktori | repo | commit | diambil |
|---|---|---|---|
| `LatentMAS/` | https://github.com/Gen-Verse/LatentMAS | `9a9e4d331eb11430bd9e64754c6b252b06d73031` | 2026-08-10 |
| `mixinputs/` | https://github.com/EvanZhuang/mixinputs | `7aef34b8113cfaa56a73fb06df68b68dfb67485e` | 2026-08-10 |

**Kode di sini tidak pernah diimpor saat run.** Ia rujukan pembacaan dan
pembanding. Yang benar-benar dijalankan ada di `backend/`; kesetiaannya
terhadap kedua paper didokumentasikan di `docs/HASIL_TAHAP0.md` §9 (rumus
per-mode) dan `docs/DESAIN_EKSPERIMEN.md` §2.

Yang **tidak** ikut disalin (dan kenapa): `LatentMAS/example_logs/` (21 MB log
run), `LatentMAS/assets/` dan `mixinputs/assets/` (gambar paper),
`mixinputs/example/` (4,7 MB notebook contoh). Semuanya tak menyentuh
algoritma; totalnya 34 MB dari 42 MB kedua repo. Ambil dari GitHub bila perlu.

Yang **ikut** disalin dan penting:

- `LatentMAS/methods/{latent_mas,text_mas,baseline}.py` — tiga metode yang
  jadi acuan Sumbu B.
- `LatentMAS/prompts.py` — sumber `backend/prompts/bench.yaml`. Teks
  instruksinya diambil apa adanya; hanya bentuknya (f-string → Jinja) yang
  berubah.
- `LatentMAS/data.py` — sumber `backend/bench/data.py` (format pertanyaan,
  pemetaan label ARC, pembungkus HumanEval+).
- `LatentMAS/utils.py` — sumber `backend/bench/scoring.py` (ekstraksi
  `\boxed{}`, eksekusi kode bertimeout).
- `LatentMAS/data/medqa.json` — satu-satunya dataset yang ter-vendor di repo
  hulu. Tidak dipakai skripsi ini (kategori sains diwakili GSM8K), disimpan
  kalau-kalau lengan MedQA ditambahkan.
- `mixinputs/mixinputs/` — implementasi MoI rujukan untuk memverifikasi
  persamaan di `_latent_step_vec` mode `"moi"`. Diverifikasi 2026-08-10:
  rumus **aljabar-identik** (dibuktikan lewat substitusi, lihat
  `docs/HASIL_TAHAP0.md` §9.2), tapi entropi $H$ dinormalisasi berbeda —
  rujukan pakai $\log(20)$ (top-k slice API logprobs vLLM), harness ini pakai
  $\log(V)$ (vocab penuh, karena logit diakses langsung via HF). Bukan bug;
  keterbatasan rekayasa vLLM di kode rujukan, bukan definisi paper.

Memperbarui pinning: clone ulang, salin bagian yang sama, perbarui tabel di
atas. Jangan menyunting isi `reference/` — kalau ada yang perlu berubah,
tempatnya di `backend/`.
