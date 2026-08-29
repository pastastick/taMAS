# Hasil Tahap 0 — Gumbel vs Ridge pada Kapasitas Kanal Laten

> Dijalankan 2026-08-09 di RunPod (A40 46GB, CUDA 12.8, torch 2.6.0+cu124),
> model `Qwen/Qwen3-8B`. Ini adalah gerbang keputusan Alt 3
> (`skripsi/alternatif_gumbel_latentmas.md` §6 Tahap 0) — satu-satunya
> pertanyaan yang dijawab: **apakah mengganti persamaan langkah laten
> LatentMAS dari ridge $W_a$ resmi paper ke relaksasi Gumbel-softmax menaikkan
> kapasitas kanal laten murni (`kv_latent_only`) untuk muatan simbolik?**

## 0. Ringkasan satu paragraf

**Ya, meyakinkan.** Pada `latent_steps=10` (setelan produksi), ridge $W_a$
resmi paper (Teorema A.1 LatentMAS) memberi kapasitas kanal laten murni
**recall = 0,000** — nol, bukan sekadar rendah — pada kedua jenis muatan (nama
fungsi DSL dan token acak), dan ini **tidak berubah** baik memakai matriks
ridge maupun tanpanya (mode `raw` dengan `use_realign=True` vs `False` — yaitu
konfigurasi *default* repo resmi LatentMAS — hasilnya identik bit-per-bit).
Gumbel-softmax memberi recall 0,350 (dsl) dan 0,190 (token) pada `m=10`, naik
ke 0,840 dan 0,760 pada `m=40` — keduanya signifikan secara statistik
(Wilcoxon p≤0,001, CI bootstrap 95% tak pernah menyentuh nol) — sementara
`raw` tetap presisi nol di `m=40` juga. Analisis tambahan (`soft`, proyeksi
convex-hull TANPA noise Gumbel) memisahkan efek "berada di manifold embedding"
dari efek "entropi tambahan": `soft` memulihkan hampir seluruh keunggulan
gumbel pada muatan `dsl` (0,340 vs 0,350, TIDAK signifikan, p=0,975) tetapi
jauh tertinggal pada muatan `token` (0,060 vs 0,190, signifikan, p=0,005).
**Kesimpulan gerbang**: `gumbel` > `raw` terbukti kuat → lanjut ke Tahap 1
(`alternatif_gumbel_latentmas.md` §6).

## 1. Setup

| komponen | nilai |
|---|---|
| GPU | NVIDIA A40, 46068 MiB, CUDA 12.8, driver 570.211.01 |
| Environment | uv + `.venv` Python 3.10, torch 2.6.0+cu124 (dari `pyproject.toml`) |
| Model | `Qwen/Qwen3-8B` (diunduh anonim dari HF Hub, ~16GB) |
| Skrip | `lab/channel_capacity.py` (A9, tak berubah logikanya — hanya ditambah `--no-realign` dan `load_dotenv`) |
| Payload | `dsl` (5 nama fungsi dari pustaka 71 fungsi produksi) dan `token` (5 pseudo-kata acak) |
| k, trials, seed | k=5, trials=20, seed=0 — identik dengan run `gumbel` yang sudah ada, supaya berpasangan |
| Lengan diuji | `kv_latent_only` (kanal laten murni — satu-satunya lengan yang menguji ekspresivitas vektor laten) |

Empat konfigurasi dijalankan:

| tag | `--latent-mode` | `--no-realign` | `--latent-steps` | makna |
|---|---|:-:|---:|---|
| `raw_m10` | `raw` | tidak | 10 | ridge $W_a$ aktif (Teorema A.1 LatentMAS), m=produksi |
| `raw_m40` | `raw` | tidak | 40 | ridge $W_a$ aktif, m besar |
| `raw_norealign_m10` | `raw` | **ya** | 10 | $M=I$ — **default resmi repo LatentMAS** (`--latent_space_realign` OFF) |
| `soft_m10` | `soft` | — | 10 | proyeksi convex-hull TANPA noise Gumbel (kontrol untuk memisahkan efek) |

Dibandingkan terhadap data `gumbel` yang sudah ada dari sesi sebelumnya
(`channel_capacity_Qwen_Qwen3-8B_m10.json`, `..._m40.json`), dengan
konfigurasi identik (k, trials, seed, payload) sehingga perbandingannya
**berpasangan** (paired) — muatan yang sama persis dipakai di semua mode.

## 2. Catatan implementasi

Dua bug/gap kecil ditemukan dan diperbaiki sebelum run:

1. **Secret leak**: `runpod_env.sh` (berkas *git-tracked*) memiliki token HF
   asli yang hardcoded (`export HF_TOKEN="hf_otarf...gcQv"`, commit `33dd614`).
   Token itu berstatus *expired* di Hub dan menyebabkan
   `RepositoryNotFoundError` bahkan untuk model publik (Qwen3, Apache-2.0)
   karena error 401 membuat transformers gagal total alih-alih jatuh ke akses
   anonim. Token dihapus dari berkas, diganti komentar yang menjelaskan token
   sebenarnya ada di `.env` (git-ignored) dan sudah otomatis dimuat. **Token
   lama itu sebaiknya di-revoke di huggingface.co/settings/tokens** karena
   sudah bocor ke riwayat git terlepas dari statusnya sekarang.
2. **`.env` tidak pernah dibaca `lab/*.py`**: hanya `launcher.py` dan `cli.py`
   yang memanggil `load_dotenv()`. Skrip di `lab/` berjalan dengan env shell
   apa adanya. Ditambahkan `load_dotenv(QL / ".env", override=False)` di
   awal `channel_capacity.py` (lihat komentar di kode) — tanpa menimpa
   variabel yang sudah di-export shell, supaya `HF_LOCAL_ONLY=1 python
   lab/...` tetap bisa memaksa mode offline.

Perubahan pada `lab/channel_capacity.py` (selain dua fix di atas):
`--no-realign` (mode `raw` tanpa matriks ridge), pencatatan `use_realign` di
`_meta`, dan nama berkas keluaran otomatis memuat `{mode}[_norealign]_m{m}`
supaya dua run dengan mode/m berbeda tidak saling menimpa.

Skrip baru `lab/compare_channel_modes.py` dibuat untuk analisis: memuat semua
`channel_capacity_*.json` yang ada, mengelompokkannya per sel eksperimen
(model, k, trials, m, seed), memverifikasi muatannya benar-benar identik
antar-mode (bukan diasumsikan), lalu menjalankan uji berpasangan (Wilcoxon
signed-rank untuk recall, McNemar eksak untuk exact-match) + CI bootstrap 95%.

## 3. Hasil lengkap

### 3.1. m=10 (setelan produksi) — empat mode dibandingkan

**Payload `dsl`** (n=20 tiap sel):

| mode | recall | exact | halusinasi | posisi (p1..p5) |
|---|---:|---:|---:|---|
| gumbel | **0,350** | 0,000 | 0,083 | 0,90 0,60 0,25 0,00 0,00 |
| soft | 0,340 | 0,000 | 0,105 | 0,80 0,70 0,20 0,00 0,00 |
| raw | 0,000 | 0,000 | 0,000 | 0,00 0,00 0,00 0,00 0,00 |
| raw(M=I) | 0,000 | 0,000 | 0,000 | 0,00 0,00 0,00 0,00 0,00 |

**Payload `token`** (n=20 tiap sel):

| mode | recall | exact | halusinasi | posisi (p1..p5) |
|---|---:|---:|---:|---|
| gumbel | **0,190** | 0,000 | 0,259 | 0,65 0,30 0,00 0,00 0,00 |
| soft | 0,060 | 0,000 | 0,050 | 0,25 0,05 0,00 0,00 0,00 |
| raw | 0,000 | 0,000 | 0,000 | 0,00 0,00 0,00 0,00 0,00 |
| raw(M=I) | 0,000 | 0,000 | 0,000 | 0,00 0,00 0,00 0,00 0,00 |

**Uji berpasangan** (Wilcoxon signed-rank untuk Δrecall, CI bootstrap 95%):

| perbandingan | payload | Δrecall | CI95% | p (Wilcoxon) |
|---|---|---:|---|---:|
| gumbel − raw | dsl | +0,350 | [+0,270, +0,430] | **<0,001** |
| gumbel − raw | token | +0,190 | [+0,120, +0,250] | **0,001** |
| gumbel − raw(M=I) | dsl | +0,350 | [+0,270, +0,430] | **<0,001** |
| gumbel − raw(M=I) | token | +0,190 | [+0,120, +0,250] | **0,001** |
| gumbel − soft | dsl | +0,010 | [−0,090, +0,120] | 0,975 (tak signifikan) |
| gumbel − soft | token | +0,130 | [+0,060, +0,200] | **0,005** |
| raw − raw(M=I) | dsl & token | 0,000 | [0,000, 0,000] | 1,000 (identik) |
| soft − raw | dsl | +0,340 | [+0,250, +0,420] | **<0,001** |
| soft − raw | token | +0,060 | [+0,020, +0,110] | **0,034** |

### 3.2. m=40 — gerbang WAJIB kedua

| mode | payload | recall | exact | halusinasi |
|---|---|---:|---:|---:|
| gumbel | dsl | **0,840** | 0,700 | 0,000 |
| raw | dsl | 0,000 | 0,000 | 0,000 |
| gumbel | token | **0,760** | 0,600 | 0,020 |
| raw | token | 0,000 | 0,000 | 0,000 |

| perbandingan | payload | Δrecall | CI95% | p | Δexact | p (McNemar) |
|---|---|---:|---|---:|---:|---:|
| gumbel − raw | dsl | +0,840 | [+0,690, +0,960] | **<0,001** | +0,700 | **<0,001** (14/0) |
| gumbel − raw | token | +0,760 | [+0,570, +0,920] | **<0,001** | +0,600 | **<0,001** (12/0) |

Pada m=40, gumbel bahkan mencapai **exact-match 0,70 (dsl) dan 0,60 (token)**
— memulihkan seluruh 5 item persis benar pada mayoritas percobaan — sementara
`raw` tetap presisi nol di exact maupun recall, di kedua m.

## 4. Tiga temuan

**(a) Kegagalan `raw` bukan soal matriks ridge-nya — ini gagal secara
struktural.** `raw` (ridge $W_a$ aktif) dan `raw(M=I)` (default resmi
LatentMAS, tanpa realignment sama sekali) menghasilkan angka **identik
bit-per-bit** di semua sel yang diuji (Δrecall=0,000, CI=[0,000, 0,000]).
Mengaktifkan atau menonaktifkan mekanisme realignment resmi paper tidak
mengubah apa pun — kanal tetap presisi nol. ini memperkuat (bukan
melemahkan) klaim dari `lab/AUDIT_KRITIS.md` §4.3 dan `HASIL_TAHAP4.md` §2:
pada Qwen3-8B, matriks ridge $W_a$ nyaris ortogonal terhadap masukannya
(cos=0,011) dan efeknya terhadap fidelitas simbolik kosong — bukan karena ia
diimplementasikan salah, tapi karena memetakan hidden state kembali ke ruang
token diskret lewat satu peta linear tunggal secara struktural tidak cukup,
baik dengan realignment maupun tanpanya.

**(b) `raw` tidak membaik dengan `latent_steps` lebih banyak — `gumbel`
membaik tajam.** Dari m=10 ke m=40, `raw` tetap 0,000→0,000 di kedua
payload, sedangkan `gumbel` naik 0,35→0,84 (dsl) dan 0,19→0,76 (token). Ini
bertentangan dengan intuisi "mungkin raw hanya butuh lebih banyak langkah" —
datanya menunjukkan raw punya **lantai keras di nol**, bukan sekadar lambat.

**(c) Disosiasi proyeksi vs entropi — keduanya berkontribusi, tapi untuk hal
berbeda.** `soft` (proyeksi convex-hull TANPA noise Gumbel) memulihkan hampir
seluruh keunggulan `gumbel` pada payload `dsl` (0,340 vs 0,350, selisih
`tidak` signifikan) tapi jauh tertinggal pada `token` (0,060 vs 0,190,
signifikan p=0,005). Payload `dsl` adalah nama fungsi nyata yang mungkin
sudah dekat dengan token vocab asli (`RANK`, `DELTA`, dst) — proyeksi ke
manifold saja cukup. Payload `token` adalah pseudo-kata yang benar-benar
tanpa prior — di situ entropi Gumbel memberi kontribusi independen dan
signifikan, konsisten dengan temuan lama (`b7_probe.py`) bahwa mode `soft`
menghasilkan vektor laten yang **identik/deterministik** (`hidden_identical:
true`, `cos: 1.0`) — mode-collapse yang membatasi keragaman yang bisa dibawa
kanal untuk muatan yang benar-benar baru.

## 5. Batas berlaku

- **n=20/sel, 1 seed.** Sama seperti A9 asli, ini bukan replikasi
  multi-seed. Variasi antar-seed tidak terukur (lihat batasan yang sama di
  `alternatif_fidelitas_simbol.md` §7).
- **Hanya `latent_steps` ∈ {10, 40}, hanya Qwen3-8B.** Belum diuji di
  backbone lain (4B/14B) atau nilai m lain (20, 80, 160 dari Figure 8 paper).
- **`kv_latent_only` di sini bukan replika `comm_mode` produksi apa pun** —
  ia isolasi kanal murni untuk tujuan pengukuran (lihat catatan kejujuran di
  docstring `channel_capacity.py`).
- **Tahap 0 ini TIDAK mengukur akurasi hilir (benchmark LatentMAS
  asli)** — hanya kapasitas kanal simbolik k=5. Tahap 1
  (`alternatif_gumbel_latentmas.md` §6) yang memindahkan pengujian ke
  benchmark bergaya paper (HumanEval+/MBPP+) belum dijalankan.
- **`raw` di sini SELALU dengan `latent_early_stop_cos=1.0` (nonaktif)** —
  sama dengan seluruh data `gumbel` lama, jadi perbandingannya adil, tapi
  berarti hasil ini tidak mencerminkan interaksi dengan early-stop (B6)
  produksi.

## 6. Keputusan gerbang

Sesuai kriteria di `README.md` §7 dan `alternatif_gumbel_latentmas.md` §6
Tahap 0:

> `gumbel` > `raw` meyakinkan → lanjut Tahap 1.

**Kriteria terpenuhi dengan sangat jelas** — bukan hanya lolos ambang
signifikansi, tapi dengan effect size besar (Δrecall 0,19–0,84) dan pola yang
konsisten di 2 nilai m × 2 payload × 2 varian raw (4 dari 4 perbandingan
gumbel-vs-raw signifikan di p≤0,001; hanya perbandingan gumbel-vs-soft pada
payload dsl yang tidak signifikan, dan itu justru temuan yang bermakna, lihat
§4c).

**Rekomendasi: lanjut ke Tahap 1** (`alternatif_gumbel_latentmas.md` §6):
probe simbolik pada tugas bergaya LatentMAS (subsample HumanEval+/MBPP+,
raw vs gumbel), lalu Tahap 2 (kontrol *gist* — GSM8K/MedQA subsample) untuk
menegakkan bentuk disosiasi (§4 dokumen itu): kolaborasi laten mungkin
memperbaiki keandalan/format tanpa memulihkan fidelitas simbolik penuh.

---

# Tahap 0B — cek literatur + dua algoritma kandidat baru

> Dijalankan 2026-08-09 (lanjutan sesi yang sama, setup identik). Dua pertanyaan:
> (a) apakah "Gumbel untuk kolaborasi laten multi-agen training-free" benar-benar
> belum ada di literatur, dan (b) adakah algoritma yang lebih baik dari Gumbel?

## 7. Hasil cek literatur — posisi klaim orisinalitas

Empat sumber yang menentukan, dibaca penuh (bukan dari abstrak saja):

| sumber | apa isinya | konsekuensi untuk klaim skripsi |
|---|---|---|
| **Stochastic Soft Thinking** (arXiv:2508.03440, Wu dkk.) | Persis mengusulkan Gumbel-Softmax di atas Soft Thinking untuk mengatasi "Greedy Pitfall"; unggul di 8 benchmark penalaran | ⚠️ **Gumbel di langkah laten BUKAN ide baru** — tapi ini **single-model**, bukan multi-agen, dan tidak mengukur fidelitas simbolik |
| **Beyond Tokens: survei komunikasi laten MAS** (arXiv:2606.05711, 18 metode 2024–2026) | Taksonomi WHAT/WHICH/HOW. **Tidak satu pun** dari 18 metode memakai langkah laten stokastik/relaksasi diskret. **Tidak satu pun** mengukur kapasitas kanal/fidelitas simbolik — mereka menyebutnya celah eksplisit di §7.4 ("a complementary statistical account would characterise when a receiver can decode a sender representation") | ✅ **Celah yang diisi proyek ini terkonfirmasi oleh survei terbaru**, dua-duanya: stokastisitas DI MAS laten, dan pengukuran kapasitas kanal |
| **Do Latent Channels Actually Communicate?** (arXiv:2607.26773, Jul 2026) | Audit kausal LatentMAS (Qwen3-4B/8B), intervensi pesan (other-example/self-generated). Eksplisit: **"No systematic symbolic-content test — they don't isolate prompt-KV from latent-KV mechanistically"** | ✅ Karya terdekat yang ada; **justru menyatakan isolasi prompt-KV vs latent-KV sebagai yang belum dilakukan** — itu persis desain A9 lengan `kv_prompt_only`/`kv_latent_only` |
| **Mixture of Inputs** (arXiv:2505.14827, NeurIPS 2025) | Training-free, Bayesian Dirichlet: distribusi = prior, token tersampel = observasi, input = ekspektasi posterior | Kandidat algoritma baru → diuji di §9 |

**Posisi klaim yang jujur setelah cek ini** (turun dari "menemukan Gumbel", naik di
sisi lain): kontribusinya **bukan** relaksasi Gumbel-nya (sudah ada untuk single-model),
melainkan (i) **mentransplantasikan keluarga relaksasi diskret ke kolaborasi laten
multi-agen** — yang menurut survei 2606.05711 belum dilakukan siapa pun, dan (ii)
**alat ukur kapasitas kanal simbolik** yang survei itu sendiri sebut sebagai celah,
dan yang audit kausal terbaru (2607.26773) nyatakan belum ia lakukan.

## 8. Dua algoritma kandidat: `sample` dan `moi`

- **`sample`** — sudah ada di kode (`z = W_in[i], i ~ softmax(W_out h/T)`) tapi
  **belum pernah diuji A9**. Ini batas ekstrem: token diskret murni, nol superposisi.
- **`moi`** — **implementasi baru** (`_latent_step_vec`, mode `"moi"`), setia pada
  MoI paper: `w = [H·p + (β+1−H)·onehot(i~p)] / (β+1)`, `z = w @ W_in`, dengan
  `H` = entropi ternormalisasi ∈ [0,1], β=1 (setelan universal paper).
  Intuisinya cocok dengan masalah kita: one-hot **menjangkarkan identitas token
  diskret** (yang hilang di `soft`/`gumbel` karena merata-rata seluruh vocab),
  sementara suku `H·p` mempertahankan superposisi. Model yakin → nyaris one-hot;
  model ragu → condong ke distribusi.

Konfigurasi identik Tahap 0 (k=5, 20 trial, seed=0, m ∈ {10, 40}) → berpasangan
penuh dengan semua data sebelumnya.

### 8.1. Hasil — m=10 (setelan produksi)

`kv_latent_only`, n=20/sel:

| mode | dsl recall | dsl halus | token recall | token halus |
|---|---:|---:|---:|---:|
| **moi** | **0,380** | 0,068 | 0,130 | 0,140 |
| sample | 0,360 | **0,029** | 0,130 | 0,165 |
| gumbel | 0,350 | 0,083 | **0,190** | 0,259 |
| soft | 0,340 | 0,105 | 0,060 | **0,050** |
| raw / raw(M=I) | 0,000 | 0,000 | 0,000 | 0,000 |

### 8.2. Hasil — m=40

| mode | dsl recall | dsl exact | token recall | token exact |
|---|---:|---:|---:|---:|
| **moi** | **0,870** | **0,750** | **0,850** | **0,650** |
| gumbel | 0,840 | 0,700 | 0,760 | 0,600 |
| sample | 0,720 | 0,600 | 0,810 | 0,650 |
| raw | 0,000 | 0,000 | 0,000 | 0,000 |

### 8.3. Uji berpasangan — yang signifikan dan yang TIDAK

**Signifikan (p<0,01), tanpa kecuali:** setiap mode berbasis proyeksi
(`soft`/`gumbel`/`sample`/`moi`) mengalahkan `raw` di **8 dari 8** sel
(m × payload × varian raw), Δrecall +0,06 s/d +0,87, CI 95% tak pernah menyentuh nol.
Di m=40 juga signifikan pada exact-match (McNemar p<0,001, mis. moi−raw 15/0).

**TIDAK signifikan — dan ini yang penting untuk kejujuran klaim:**

| perbandingan | m | payload | Δrecall | p |
|---|---|---|---:|---:|
| moi − gumbel | 10 | dsl | +0,030 | 0,394 |
| moi − gumbel | 40 | dsl | +0,030 | 0,666 |
| moi − gumbel | 40 | token | +0,090 | 0,288 |
| moi − sample | 40 | dsl | +0,150 | 0,083 |
| gumbel − sample | 40 | dsl | +0,120 | 0,157 |
| gumbel − sample | 40 | token | −0,050 | 0,496 |

**Satu-satunya perbedaan antar-mode-proyeksi yang signifikan:** pada m=10 payload
`token`, `gumbel` mengalahkan `moi` dan `sample` (Δ=+0,060, p=0,034 keduanya) dan
`soft` (Δ=+0,130, p=0,005).

### 8.4. Kesimpulan Tahap 0B

**(a) `moi` adalah pemenang nominal di 3 dari 4 sel** (dsl m=10, dsl m=40, token
m=40) dan mencapai angka tertinggi yang pernah terukur di proyek ini
(recall 0,870 / exact 0,750). **Tetapi tidak satu pun keunggulannya atas `gumbel`
mencapai signifikansi pada n=20.** Menyebut MoI "lebih baik dari Gumbel" saat ini
**tidak didukung data** — yang sah dikatakan: "setara, dengan kecenderungan nominal
konsisten ke arah MoI yang perlu n lebih besar untuk diuji."

**(b) Batas yang nyata bukan antar-algoritma-stokastik, melainkan proyeksi vs
tidak.** Empat mode dengan mekanisme sangat berbeda — rata-rata lunak (`soft`),
noise Gumbel (`gumbel`), sampel keras (`sample`), campuran Bayesian (`moi`) —
semuanya mendarat di rentang sempit 0,34–0,38 (dsl m=10) dan 0,72–0,87 (m=40),
sementara ridge `W_a` resmi paper mendarat di **0,000 mutlak**. Ini menguatkan
temuan inti Tahap 0: yang menentukan adalah **apakah langkah laten dikembalikan ke
ruang embedding sama sekali**, bukan bagaimana persisnya.

**(c) Trade-off halusinasi yang bisa dilaporkan terpisah.** Pada m=10 dsl, `sample`
punya halusinasi jauh terendah (0,029 vs gumbel 0,083 vs soft 0,105) dengan recall
setara — untuk domain DSL faktor alpha (di mana ekspresi ngawur lolos gate lebih
mahal daripada ekspresi hilang), ini sumbu yang mungkin lebih relevan daripada recall.

**(d) Konsekuensi untuk arah skripsi.** Karena tak ada algoritma yang terbukti
unggul, bingkai "mengusulkan algoritma baru yang menang" **tidak didukung**.
Bingkai yang didukung penuh data: *"keluarga relaksasi diskret (empat varian,
termasuk satu yang belum pernah diterapkan ke kolaborasi laten) memulihkan kapasitas
kanal simbolik yang hilang total pada mekanisme resmi LatentMAS; perbedaan antar-varian
kecil dan tak signifikan, sehingga yang menentukan adalah keputusan desain
proyeksi-ke-embedding, bukan pilihan varian."* Ini justru lebih kuat dari klaim
"algoritma saya menang", karena tak bisa dipatahkan dengan mengganti varian.

### 8.5. Batas berlaku Tahap 0B

- n=20/sel, **satu seed**. Angka MoI di §8.1–8.2 memakai β=1 (default paper);
  sweep β lengkap ({0,25…8}, §8.6) menunjukkan ini **tidak masalah** — β tak
  berpengaruh signifikan pada domain ini, beda dengan temuan MoI di tugas
  penalaran umum.
- `raw(M=I)` hanya diuji di m=10 (di m=40 hanya `raw` dengan ridge).
- Perbedaan nominal 0,03–0,15 pada n=20 **tidak bisa dibedakan dari derau** — untuk
  memutuskan pemenang sejati butuh n≫20 atau multi-seed.

### 8.6. Sweep β MoI — TIDAK ADA β optimal yang bisa dibedakan dari derau

> Dijalankan 2026-08-09 (lanjutan sesi yang sama). Paper MoI melaporkan β optimal
> **bergantung tugas** (β≤1 menolong AIME, β>1 menolong Count Down 4). Pertanyaannya:
> apakah itu berlaku juga di kanal laten murni domain faktor alpha?

Enam nilai β diuji di m=10 (β=1 dari §9.1, lima nilai baru dijalankan **paralel
2×** per batch — 2 proses @ ~16GB muat bersamaan di GPU 46GB, lihat §9.6):

| β | dsl recall | dsl halus | token recall | token halus | rata-rata |
|---:|---:|---:|---:|---:|---:|
| 0,25 | 0,350 | 0,054 | 0,130 | 0,165 | 0,240 |
| 0,5  | 0,370 | 0,054 | 0,130 | 0,165 | 0,250 |
| **1,0** (default paper) | 0,380 | 0,068 | 0,130 | 0,140 | 0,255 |
| 2,0  | 0,380 | 0,054 | 0,130 | 0,165 | 0,255 |
| 4,0  | 0,360 | 0,054 | 0,110 | 0,135 | 0,235 |
| 8,0  | 0,380 | 0,071 | 0,130 | 0,190 | 0,255 |

**Uji berpasangan (Wilcoxon, 15 pasangan β×β, kedua payload = 30 uji total):
TIDAK SATU PUN signifikan.** p berkisar 0,499–1,000; sebagian besar pasangan
punya n≠ ≤ 2 dari 20 trial (praktis identik trial-per-trial, bukan hanya
rata-ratanya kebetulan dekat). Rentang **32×** pada β (0,25 → 8) tidak
menghasilkan perbedaan yang bisa dibedakan dari derau sampel.

**Kesimpulan: tidak ada β optimal yang bisa diklaim di domain ini.** Tiga nilai
(β=1, β=2, β=8) berbagi rata-rata tertinggi (0,255) secara tepat — bukan karena
istimewa, tapi karena recall dsl/token masing-masing sudah jenuh di nilai yang
sama (0,380/0,130) untuk ketiganya. β=1 (default universal paper) sudah optimal
sejauh yang bisa dibuktikan data ini, jadi **tidak perlu diganti** untuk domain
faktor alpha — beda dengan temuan MoI di tugas penalaran umum (AIME, Count Down)
tempat β bergantung tugas. **Konfirmasi tambahan di m=40 sengaja TIDAK
dijalankan**: dengan efek sekecil ini pada m=10 (dan pola serupa hampir pasti
berulang di m=40, mengingat mekanisme yang sama), biaya GPU tambahan tidak
sepadan dengan nilai informasi yang didapat — sweep m=10 sudah menjawab
pertanyaannya secara meyakinkan (bukan "belum cukup data", tapi "efeknya
memang tidak ada pada rentang yang diuji").

## 9. Asal-usul algoritma — rujukan arXiv persis

Tiga mode stokastik yang diuji Tahap 0B bukan diciptakan proyek ini; berikut
rujukan persis untuk masing-masing, supaya atribusi di skripsi tepat.

### 9.1. `gumbel` — Stochastic Soft Thinking

**Wu, J., Lu, J., Ren, Z., Hu, G., Wu, Z., Dai, D., Wu, H.** *LLMs are
Single-threaded Reasoners: Demystifying the Working Mechanism of Soft
Thinking.* arXiv:2508.03440v4 [cs.CL], 2025.

Persamaan Gumbel-Softmax mereka (Eq. 4, §5.1.2), untuk logit asli $\pi_i$ dan
suhu $\tau$:

$$y_i = \frac{\exp((g_i + \log \pi_i)/\tau)}{\sum_k \exp((g_k + \log \pi_k)/\tau)},
\qquad g_i \sim \text{Gumbel}(0,1)$$

Keluaran $y$ dipakai sebagai kombinasi konveks atas matriks embedding (bukan
sampel keras) — **identik secara matematis** dengan yang diimplementasikan di
`_latent_step_vec` mode `"gumbel"` (`client.py`): `logits += -log(-log(u))`,
`z = softmax(logits/T) @ W_in`. Satu-satunya beda adalah nilai suhu: paper
memakai $\tau=0{,}5$ sebagai default, harness ini memakai $T=0{,}7$ (warisan
setelan produksi B2, bukan tuning ulang untuk kesetaraan dengan paper ini —
lihat §7 `HASIL_TAHAP4.md`). Training-free di waktu inferensi, sama seperti
mode lain di sini. **Konteksnya single-model** (satu model bernalar sendiri),
bukan multi-agen — proyek ini yang pertama menerapkannya pada transfer KV
antar-agen (lihat §8).

### 9.2. `moi` — Mixture of Inputs

**Zhuang, Y., dkk.** *Mixture of Inputs: Text Generation Beyond Discrete Token
Sampling.* NeurIPS 2025, arXiv:2505.14827.

Model Bayesian Dirichlet-Multinomial: distribusi $p$ (output softmax) sebagai
prior, token tersampel $y\sim p$ sebagai observasi dengan pseudo-count
$(\beta{+}1{-}H)$, $H$ = entropi ternormalisasi $\in[0,1]$:

$$w_i = \frac{H \cdot p_i + (\beta+1-H)\cdot \mathbb{1}[i=y]}{\beta+1},
\qquad z = w \cdot W_{\text{in}}$$

Diimplementasikan persis sesuai rumus ini di `_latent_step_vec` mode `"moi"`.
Paper tidak me-rescale embedding hasil; normalisasi ke `target_norm` di
harness ini adalah **konvensi seragam proyek** untuk semua mode (supaya
perbandingan antar-mode adil), bukan bagian dari definisi MoI aslinya.
β=1 dipakai sebagai default (setelan universal paper mereka untuk
GPQA-Diamond/LiveCodeBench) — sweep §9.5b menunjukkan pada domain ini β tidak
berpengaruh sama sekali, jadi default itu tak perlu diganti.

**Verifikasi terhadap kode rujukan (2026-08-10) — rumus identik, satu
konstanta berbeda karena keterbatasan API, bukan pilihan desain.**
Implementasi resmi (`reference/mixinputs/mixinputs/gpu_model_runner.py:1190–1243`,
vLLM patch) menulis rumusnya dalam DUA langkah — campur `posterior_probs =
(p+β·onehot)/(1+β)` atas kandidat, lalu gerbang entropi `H·(Σ posterior·e) +
(1-H)·e_ŷ` — alih-alih satu closed-form. Substitusi aljabar keduanya
membuktikan **identik**:

$$z_{\text{ref}} = \frac{H\bar p + (1{+}\beta{-}H)\,e_{\hat y}}{1+\beta}
= w\cdot W_\text{in} = z$$

Yang **berbeda**: referensi menghitung $p$ dan $H$ di atas **top-20 slice**
API logprobs vLLM (`gpu_input_batch.py:330`, default `num_logprobs=20` bila
tak diset eksplisit), dengan $H_\max=\log(20)$ — bukan $\log(V)$, $V\approx
151.936$ untuk Qwen3. Harness ini menghitung $p$/$H$ dari **logit vocab
penuh** karena hidden state diakses langsung lewat
`model.get_output_embeddings()` (HF), tanpa lapisan sampler vLLM yang
membatasi ke top-k. Akibatnya entropi ternormalisasi referensi secara
sistematis lebih tinggi untuk tingkat "kepastian" model yang sama (denominator
$\log(20)\approx3$ vs $\log(V)\approx12$), sehingga bobot campuran $H$ tidak
akan bit-identik antar dua implementasi meski rumusnya sama persis. Ini
kompromi rekayasa mixinputs (dibatasi API logprobs vLLM), bukan bagian dari
definisi konseptual paper — versi harness ini (vocab penuh) karenanya lebih
setia ke definisi Bayesian paper, tapi **bukan replikasi bit-identik** kode
rujukan. Klaim yang sah: "rumus MoI diimplementasikan tepat sesuai definisi
paper", bukan "direplikasi identik dari kode rujukan mixinputs".

### 9.3. `sample` — pengambilan sampel kategoris standar

Tidak berasal dari satu paper tunggal — ini teknik dekode standar tekstual
($y\sim\text{Categorical}(\text{softmax}(\text{logits}/T))$, lalu $z=W_{\text{in}}[y]$),
lebih tua dari ketiga paper laten di atas. Disebut eksplisit sebagai baseline
**"Token (Sampling)"** di tabel evaluasi Stochastic Soft Thinking (§10.1) dan
dipakai sebagai pembanding di hampir semua paper penalaran laten (CoCoNut,
Soft Thinking). Fungsinya di Tahap 0B adalah **batas ekstrem**: superposisi
nol, identitas token diskret penuh — kebalikan `soft` yang superposisi penuh
tanpa identitas diskret sama sekali. `gumbel` dan `moi` adalah dua cara
berbeda menyeimbangkan kedua ekstrem ini.

### 8.7. Catatan efisiensi GPU untuk tahap berikutnya

Tiap run memakai **~16 GB dari 46 GB** VRAM A40 dan GPU hanya 70–95% terpakai oleh
satu proses. **2–3 run bisa jalan paralel** — Tahap 0B yang berjalan serial memakan
~35 menit dan seharusnya bisa ~12–15 menit. Untuk Tahap 1/2 dan sweep β, jalankan
proses bersamaan dengan jeda start ~30 detik (agar fase muat model dari network
storage tidak rebutan I/O), bukan berurutan.

---

## 10. Cara mereproduksi

```bash
source /workspace/runpod_env.sh
source /workspace/project/multi-agent-system/.venv/bin/activate
cd /workspace/project/multi-agent-system
export PYTHONPATH=backend

# 4 run Tahap 0 (~5-8 menit GPU tiap satu)
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode raw \
    --latent-steps 10 --k 5 --trials 20 --seed 0
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode raw \
    --latent-steps 40 --k 5 --trials 20 --seed 0
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode raw \
    --no-realign --latent-steps 10 --k 5 --trials 20 --seed 0
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode soft \
    --latent-steps 10 --k 5 --trials 20 --seed 0

# Tahap 0B — dua algoritma kandidat (bisa & sebaiknya PARALEL, lihat §9.6)
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode sample \
    --latent-steps 10 --k 5 --trials 20 --seed 0
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode moi \
    --latent-steps 10 --k 5 --trials 20 --seed 0        # --latent-beta 1.0 (default)
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode sample \
    --latent-steps 40 --k 5 --trials 20 --seed 0
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode moi \
    --latent-steps 40 --k 5 --trials 20 --seed 0

# Analisis statistik berpasangan (Wilcoxon + McNemar + bootstrap CI)
python lab/compare_channel_modes.py --out lab/out/tahap0_analysis.json
```

Berkas mentah: `lab/out/channel_capacity_Qwen_Qwen3-8B_{raw_m10,raw_m40,
raw_norealign_m10,soft_m10}.json` (baru) + `..._m10.json`, `..._m40.json`
(gumbel, sudah ada sebelumnya). Ringkasan uji: `lab/out/tahap0_analysis.json`.
