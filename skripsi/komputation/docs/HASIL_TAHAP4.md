# Hasil Tahap 4 (B6, B7) + sumbu A9 + riset B10

> Dijalankan 2026-08-07 di A40 46 GB, backbone **Qwen3-8B**, branch
> `exp/rencana-perbaikan`. Semua angka di dokumen ini diukur pada **jalur
> produksi** (`_CoreEngine.latent_pass` lewat `LocalLLMBackend`), bukan pada
> replika — kecuali di tempat yang disebutkan eksplisit.
>
> Rujukan rencana: `lab/RENCANA_PERBAIKAN.md` §B6, §B7, §A9, §B10.

## Ringkasan satu paragraf

B6 (early-stop) dipasang dan **terbukti tidak menyentuh konfigurasi produksi**:
ia menyala 9/9 pada mode lama `raw` (menghemat 47% langkah) dan **0/9** pada
`gumbel`/`soft`/`sample`. B7 (ganti persamaan realignment) diubah permanen di
kode; buktinya matriks ridge `M` pada Qwen3-8B memutar hidden state sampai
`cos(h, hM) = 0,011` — praktis ortogonal — dan konsekuensi yang harus ikut
dilaporkan adalah bahwa flag `use_realign` **inert** di produksi (dibuktikan
identik bit-per-bit). A9 (alat baru) mengukur kapasitas kanal secara langsung
dan menemukan bahwa mode `kv` produksi lossless **bukan karena vektor latennya
ekspresif, melainkan karena token prompt hulu ikut diwariskan verbatim**;
kanal laten murni hanya memulihkan 19–35% muatan pada `latent_steps=10`.
B10 diteliti dan diprototipekan, hasilnya di §4.

---

## 1. B6 — early-stop adaptif pada rollout laten

**Perubahan.** `_CoreEngine.latent_pass` berhenti bila
`cos(h_k, h_{k−1}) > latent_early_stop_cos` (default 0,999, ambang yang sama
dengan definisi "titik tetap" di `lab/latent_dynamics.py`). `latent_steps`
karenanya berubah makna dari **target** menjadi **batas atas**.

Berkas yang berubah: `backend/llm/client.py` (`_CoreEngine.__init__`,
`latent_pass`, `LLMResult`, `LocalLLMBackend.__init__`),
`backend/pipeline/settings.py`, `backend/pipeline/factor_mining.py`
(kunci YAML `latent.early_stop_cos`), `backend/latent_mas/agent.py`
(`AgentResult.n_latent_steps` / `.latent_stop`), 5 berkas `configs/*.yaml`.

Langkah yang benar-benar berjalan sekarang tercatat per-agen
(`n_latent_steps`, `latent_stop`), sehingga penghematan bisa dibaca dari log
run — bukan disimpulkan dari durasi.

**Verifikasi** (`lab/early_stop_probe.py`, anggaran 60 langkah, 3 prompt ×
3 seed = 9 run per mode):

| mode langkah laten | early-stop menyala | langkah terpakai | hemat |
|---|---:|---:|---:|
| `raw` @T0,7 (mode LAMA) | **9/9** | 31,7 [17–45] / 60 | **47,2%** |
| `soft` @T1,0 | 0/9 | 60,0 / 60 | 0% |
| **`gumbel` @T0,7 (PRODUKSI)** | **0/9** | 60,0 / 60 | 0% |
| `sample` @T1,0 | 0/9 | 60,0 / 60 | 0% |

Berkas: `lab/out/early_stop_Qwen_Qwen3-8B_b6_budget60.json`.

**Baca hati-hati.** B6 **tidak** memperbaiki apa pun di konfigurasi produksi
sekarang, dan itu memang hasil yang benar, bukan kegagalan. Rollout `gumbel`
stokastik dan tidak pernah membeku (konsisten dengan A4: `fixed_point_step` =
"tak pernah" untuk gumbel), jadi tak ada salinan untuk dipotong. Nilai B6 ada
di tempat lain: ia membuat sistem **tahan terhadap salah-setel**. Kalau
`latent_steps` dinaikkan lagi atau `step_mode` dikembalikan ke `raw` — dua hal
yang sudah pernah terjadi dan sudah pernah meruntuhkan produksi (G2: 0/6 run
pada ls=60) — biayanya kini terpotong otomatis 47%.

Karena B6 nol-efek pada rantai produksi, **tidak ada lengan A1/A2 tambahan yang
dijalankan untuk B6**: membandingkan dua lengan yang menjalankan komputasi
identik hanya akan menghasilkan selisih derau. Klaim yang didaftarkan adalah
"tak berefek di produksi", dan tabel di atas sudah membuktikannya di tingkat
mekanisme.

**Uji asap end-to-end** (rantai produksi penuh sesudah B6+B7, `comm_mode=kv`,
ls=10, gumbel, `proposal→innovate→construct`, 1 run):

| agen | mode | anggaran → langkah nyata | sebab berhenti | KV |
|---|---|---|---|---:|
| `proposal` | kv_only | 10 → **10** | `budget` | 890 |
| `innovate` | kv_only | 10 → **10** | `budget` | 2 469 |
| `construct` | kv_and_text | 10 → **10** | `budget` | 4 531 |

54 s, **6 ekspresi, 6 lolos gate, 0 cacat semantik**; skoring CPU: 5 ber-IC,
5 hidup, mean \|IC\| = 0,0130, maks \|IC\| = 0,0394. Dua hal terkonfirmasi
sekaligus: (i) B6/B7 tidak merusak jalur produksi, dan (ii) pembukuan B6
memang menyala di produksi dan melaporkan `stop=budget` di ketiga agen —
early-stop tidak pernah memotong, persis seperti yang diramalkan tabel
mekanisme di atas. Panjang KV construct (4 531) juga cocok dengan A5 (4 624).

Satu ekspresi dari enam tak ber-IC: `REGRESI($volume, SEQUENCE(20), 20) -
$volume` kena `TimeoutError` pada anggaran 90 s/ekspresi. Ini artefak
pengukuran yang SUDAH diketahui dan sudah dimitigasi (RENCANA §B16 risiko
kedua: REGBETA/REGRESI bisa memakan belasan menit per ekspresi, karena itu
skoring CPU diberi anggaran waktu) — bukan regresi dari B6/B7.

Angka \|IC\| dari SATU run tidak boleh dibandingkan dengan rujukan 6-run di
HASIL_A8 (\|IC\|/run 0,0182); uji ini menguji **keutuhan jalur**, bukan mutu.
Berkas: `lab/out/frontend_tahap4_sanity.json`.

---

## 2. B7 — ganti persamaan realignment, permanen

**Perubahan.** Default `latent_step_mode` di `_CoreEngine` diubah
`"raw"` → `"gumbel"`. Sebelum ini default **kode** dan default **produksi**
berbeda (settings.py sudah `gumbel` sejak B2, kode masih `raw`), sehingga jalur
mana pun yang membangun `LocalLLMBackend` tanpa lewat `Settings` diam-diam
memakai persamaan yang sudah ditinggalkan. `_CoreEngine` kini juga mencetak
**persamaan yang berlaku**, bukan sekadar nama modenya.

Persamaan produksi:

    z = softmax((W_out h + g) / T) @ W_in      (gumbel, T = 0,7)

menggantikan

    z = (h @ M_ridge) dinormalkan,   M = (W_outᵀ W_out + λI)⁻¹ W_outᵀ W_in

### 2.1. Bukti — ridge M praktis ortogonal pada Qwen3-8B

`lab/realign_probe.py` (CPU, membaca safetensors saja):

| besaran | nilai |
|---|---:|
| `tie_word_embeddings` | **False** (ada `lm_head.weight` terpisah) |
| simpangan `M` dari identitas (relatif Frobenius) | **104,0%** |
| `cos(h, hM)` rata-rata atas 512 vektor acak | **0,011** |
| `cos(h, hM)` minimum | −0,048 |
| rasio norma \|\|hM\|\|/\|\|h\|\| | 0,306 |
| `cos(W_in[i], W_out[i])` rata-rata | 0,004 |

Baris terakhir menjelaskan sisanya: pada Qwen3-8B ruang masukan dan ruang
keluaran memang **hampir ortogonal per-token** (cos 0,004), jadi peta linear
tunggal yang memetakan salah satunya ke yang lain hanya bisa dipenuhi dalam
arti kuadrat-terkecil — bukan dalam arti "arah vektornya dipertahankan".
Berkas: `lab/out/realign_probe_Qwen_Qwen3-8B.json`.

### 2.2. Bukti — geometri vektor laten yang benar-benar diumpankan

`lab/b7_probe.py`, `max_v cos(z_k, W_in[v])` pada rollout produksi 10 langkah:

| mode | cos ke embedding terdekat (rata-rata) | rentang |
|---|---:|---|
| `raw` (ridge M) | **0,312** | [0,164 – 0,385] |
| `soft` | 0,927 | [0,784 – 1,000] |
| **`gumbel` (produksi)** | **0,985** | [0,863 – 1,000] |
| `sample` | 1,000 | [1,000 – 1,000] |

Artinya vektor yang disuntikkan `raw` tidak menyerupai embedding token mana pun
— ia berada di daerah ruang yang tak pernah ditemui model saat dilatih. Proyeksi
convex-hull menaruhnya kembali di dalam manifold embedding.

### 2.3. Konsekuensi yang WAJIB dilaporkan, bukan disembunyikan

**`use_realign` sekarang inert di produksi.** Diuji deterministik (mode `soft`,
prompt & seed sama, `use_realign` True vs False):

| mode | hidden akhir identik? | max \|selisih\| | cos |
|---|---|---:|---:|
| `raw` | tidak | 55,6 | 0,242 |
| `soft` | **ya (bit-per-bit)** | **0,0** | 1,000000 |

Sebabnya struktural: `_latent_step_vec` hanya memanggil `realigner.apply()` di
mode `raw`; di mode lain ia hanya meminjam `target_norm`. Maka:

- hasil ablasi **G6** (`use_realign` ON vs OFF) **hanya berlaku untuk mode
  `raw`**, dan tidak boleh dikutip sebagai temuan umum;
- klaim Bab 4 §Realignment Laten harus dinyatakan sebagai **sejarah mekanisme**
  ("ridge M adalah rancangan awal; diukur, ditemukan ortogonal, diganti"),
  bukan sebagai deskripsi sistem yang berjalan.

Ini persis risiko yang didaftarkan di RENCANA §B7 ("mengubah klaim Bab 4 tentang
mekanisme inti; harus dilaporkan sebagai temuan"). Dokumen ini adalah
pelaporannya. `LatentRealigner` **tidak dihapus** karena (i) `target_norm`
dipakai semua mode dan (ii) mode `raw` masih harus bisa dijalankan untuk
mereplikasi baseline G1/G3/G6 yang sudah dilaporkan.

---

## 3. A9 — kapasitas kanal laten (alat BARU)

Alat: `lab/channel_capacity.py`. Titipkan muatan yang **diketahui** (k = 5 item)
ke agen hulu, minta agen hilir menyebutkannya kembali, ukur akurasi
rekonstruksi. Hasilnya tak bergantung pada mutu faktor sama sekali.

Lima lengan; yang berbeda **hanya apa yang dioper** (komputasi hulu identik):

| lengan | yang diwariskan ke hilir |
|---|---|
| `text` | teks yang ditulis hulu (konteks hilir bersih) |
| `kv_full` | SELURUH KV hulu = token prompt + m vektor laten (= `comm_mode=kv`) |
| `kv_prompt_only` | hanya token prompt hulu (blok laten dibuang) — **kontrol positif** |
| `kv_latent_only` | hanya m vektor laten (prompt dipotong, RoPE di-re-rotasi/B8) |
| `none` | tidak ada apa-apa — **lantai tebakan** |

Dua jenis muatan: `dsl` (nama fungsi dari pustaka, in-domain) dan `token`
(pseudo-kata acak `qoc43`, tanpa prior semantik). 20 percobaan per sel, muatan
sama untuk semua lengan (berpasangan).

### 3.1. Hasil

recall = fraksi item yang benar dipulihkan (urutan diabaikan).

| lengan | m=10 `dsl` | m=10 `token` | m=40 `dsl` | m=40 `token` |
|---|---:|---:|---:|---:|
| `text` | 0,840 | 0,970 | 0,860 | 0,710 |
| `kv_full` | **1,000** | **1,000** | **1,000** | **1,000** |
| `kv_prompt_only` | 1,000 | 0,990 | 1,000 | 0,990 |
| `kv_latent_only` | **0,350** | **0,190** | **0,840** | **0,760** |
| `none` (lantai) | 0,000 | 0,000 | 0,000 | 0,000 |

Berkas: `lab/out/channel_capacity_Qwen_Qwen3-8B_m10.json`, `..._m40.json`.

### 3.2. Tiga temuan

**(a) Mode `kv` lossless — tetapi bukan karena vektor latennya.**
`kv_full` = 1,000 dan `kv_prompt_only` = 1,000 sedangkan `kv_latent_only` =
0,350. Jadi yang membuat handoff KV tak kehilangan muatan simbolik adalah
**token prompt hulu yang ikut diwariskan verbatim**, bukan ekspresivitas
"latent thoughts". Ini menjawab pertanyaan yang selama ini hanya bisa ditafsir
dari mutu faktor. Perlu dicatat: ini **tidak** membantah Teorema 3.3 LatentMAS
(kesetaraan KV-transfer dengan menyuapkan keluaran hulu) — justru
mengonfirmasinya. Yang tidak terkonfirmasi adalah tafsiran populer bahwa
vektor latennya sendiri yang membawa muatan.

**(b) Kanal laten meluruh sepanjang urutan, tidak gagal acak.**
Recall per posisi muatan (m=10):

| lengan | p1 | p2 | p3 | p4 | p5 |
|---|---:|---:|---:|---:|---:|
| `kv_latent_only` (`dsl`) | 0,90 | 0,60 | 0,25 | 0,00 | 0,00 |
| `kv_latent_only` (`token`) | 0,65 | 0,30 | 0,00 | 0,00 | 0,00 |
| `kv_prompt_only` (`dsl`) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |

Kegagalannya berbentuk **konfabulasi yang masuk akal**, bukan keluaran kosong:
`TS_COVARIANCE` → `TS_COVARIABILIDAD`; `sul17` → `sul12` (konsonan-vokal
selamat, digitnya tidak). Kanal ini menyimpan sesuatu yang **kabur**, bukan
tidak menyimpan apa-apa.

**(c) Kapasitas kanal laten naik tajam dengan `latent_steps`.**
Dari m=10 ke m=40: `dsl` 0,350 → 0,840, `token` 0,190 → 0,760 — jadi pada
setelan produksi kanal laten hanya membawa **0,42× (dsl) dan 0,25× (token)**
dari apa yang ia bawa di m=40.

Ini berkonsekuensi langsung pada **B1**, yang menurunkan `latent_steps` 60 → 10
demi keandalan produksi (G2: 0/6 run pada ls=60 vs 6/6 pada ls=10):
**B1 membeli keandalan dengan menyempitkan kanal laten.** Kedua fakta itu benar
dan harus dilaporkan bersama; menyebut B1 sebagai perbaikan murni akan
menyesatkan.

*Presisi yang harus dijaga*: yang diukur adalah m=10 vs **m=40**, bukan m=60.
Besar kerugian B1 yang sebenarnya (60 → 10) BELUM diukur, dan tidak boleh
diekstrapolasi dari dua titik — A9 §3.1 tidak memuat kolom m=60. Yang sah
dikatakan hanyalah arah dan besaran pada rentang yang diukur.

Arah lanjutan yang wajar (bukan untuk sesi ini): cari `latent_steps` yang
memaksimalkan kapasitas kanal DENGAN keandalan tetap 6/6 — A9 sekarang membuat
sumbu itu bisa diukur, dan `lab/early_stop_probe.py` + B6 membuat biayanya
tak lagi linear terhadap anggaran.

### 3.3. Batas berlaku

- Lengan `text` di sini **bukan** replika persis `comm_mode="text"` produksi
  (di sana `latent_steps=0`; di sini hulu tetap berpikir laten agar
  perbandingannya berpasangan). Jangan kutip 0,840 sebagai angka produksi.
- Muatan simbolik k=5 adalah kasus **terburuk yang wajar** untuk kanal kontinu:
  ia menuntut pemulihan diskret eksak. Kanal yang sama bisa jadi jauh lebih
  baik untuk muatan yang memang kontinu (arah, sikap, penekanan) — A9 tidak
  mengukur itu dan tidak boleh dipakai untuk menyimpulkannya.

---

## 4. B10 — *latent bottleneck*: diteliti, diprototipekan, **tidak diadopsi**

RENCANA §B10 mengusulkan meringkas SELURUH KV hulu jadi `m ≪ L` vektor
(attention pooling), supaya konteks emitter berhenti tumbuh linear terhadap
jumlah hop. Ini didaftarkan sebagai "risiko TINGGI, ini riset". Di bawah:
apa kata literatur, apa kata pengukuran kita, dan apa keputusannya.

### 4.1. Posisi di literatur (per Agustus 2026)

| pendekatan | mengurangi apa | training-free? | rasio dilaporkan |
|---|---|---|---|
| **STILL** (arXiv:2606.07878) — kompaktor Perceiver per-layer, latent queries cross-attend KV penuh | **jumlah pasangan KV** | **TIDAK** — dilatih dgn KL vs model konteks-penuh (~120k item @ 88k konteks) | 8×–200× |
| **Q-KVComm** (arXiv:2512.17914) — kuantisasi adaptif per-layer utk komunikasi antar-agen | **bit per pasangan** | tak dinyatakan eksplisit di makalah; mekanismenya (kuantisasi) tak menuntut pelatihan | 5–6×, degradasi < 5% |
| **KVCOMM** (arXiv:2510.12872, NeurIPS'25) — penyelarasan offset utk MENGGUNAKAN ULANG cache konteks yang tumpang tindih | tak mengurangi; **reuse** | ya | reuse > 70%, prefill 7,8× |
| **Cache-to-Cache** (arXiv:2510.03215) — proyektor terlatih antar-model heterogen | — (transfer) | TIDAK | akurasi +8,5–10,5% |

Polanya konsisten dan menentukan: **pengurangan agresif JUMLAH pasangan KV
hanya tercapai dengan modul yang DILATIH.** Metode training-free yang berhasil
menyerang sumbu lain — lebar bit per pasangan (Q-KVComm), atau penggunaan ulang
prefix (KVCOMM) — bukan `L → m`. Survei taksonomi komunikasi laten
(arXiv:2606.05711, 18 metode 2024–2026) tidak menyebut satu pun peringkas
`L → m` yang training-free.

Ada alasan matematis, bukan sekadar empiris. Attention memakai
`softmax(q·k)`, dan **rata-rata key BUKAN key dari token rata-rata** — pooling
linear pada key adalah hampiran yang softmax perbesar. Ditambah lagi RoPE tidak
distributif terhadap penjumlahan: `α·R_j k_j + (1−α)·R_j' k_j'` tak bisa
ditulis sebagai `R_φ(α k_j + (1−α) k_j')` untuk rotasi `R_φ` mana pun, sehingga
merata-ratakan key ter-rotasi mencampur fase yang tak sepadan ("aggregation
barrier"). Prototipe di bawah **menghindari** perangkap kedua ini (key
di-UN-rotasi ke posisi 0 dulu, baru dipool, lalu di-re-rotasi ke `[0, m)` —
prosedur yang sama dengan STILL), jadi kegagalannya bukan kegagalan
implementasi yang naif.

### 4.2. Prototipe & pengukuran

Alat: `lab/latent_bottleneck.py`. Alat ukurnya A9 (muatan diketahui k=5),
tetapi konteks hulu dibuat sepanjang produksi — prompt `construct` ASLI,
**L ≈ 2 613 token** (bandingkan A5: KV construct produksi 4 624 token). Empat
keluarga bottleneck, semuanya training-free, tiga anggaran, 10 percobaan/sel.

| keluarga | apa yang dilakukan |
|---|---|
| `pool_uniform` | L posisi dibagi m segmen kontigu; K & V dirata-rata per segmen (usulan B10 apa adanya) |
| `pool_vnorm` | sama, rata-rata berbobot ‖V_i‖₂ — proksi attention pooling tanpa forward tambahan |
| `select_recent` | simpan m token terakhir (= `kv_truncate`, jalur B8) |
| `select_knn` | simpan m token paling mirip query hilir (= `kv_knn_filter`) |

**Hasil (recall muatan; `full` = tanpa kompresi = 1,000; `none` = lantai = 0,000):**

| keluarga | 16 slot (163×) | 64 slot (41×) | 256 slot (10×) |
|---|---:|---:|---:|
| *muatan di TENGAH konteks (`--position mid`)* ||||
| `pool_uniform` | 0,000 | 0,000 | 0,000 |
| `pool_vnorm` | 0,000 | 0,000 | 0,000 |
| `select_recent` | 0,060 | 0,060 | 0,060 |
| `select_knn` | 0,000 | 0,000 | 0,020 |
| *muatan di AKHIR konteks (`--position end`)* ||||
| `pool_uniform` | 0,000 | 0,000 | 0,000 |
| `pool_vnorm` | 0,000 | 0,000 | 0,000 |
| `select_recent` | 0,120 | **0,920** | **0,960** |
| `select_knn` | 0,000 | 0,000 | 0,040 |

Berkas: `lab/out/latent_bottleneck_Qwen_Qwen3-8B_{mid,end}.json`.

### 4.3. Tiga bacaan dari tabel itu

**(a) Pooling gagal total, di semua anggaran, di semua posisi.** Bahkan pada
kompresi yang sangat longgar (256 slot = hanya 10×), `pool_uniform` dan
`pool_vnorm` memulihkan **nol** item. Dan ini bukan kegagalan RoPE yang naif —
prototipe sudah meng-UN-rotasi key sebelum pooling.

**(b) Cache-nya tidak rusak; ia kosong.** Ini pembedaan yang menentukan.
Keluaran hilir pada `pool_uniform@256` bukan sampah, melainkan koheren dan
patuh-format: `"12345"`, `"1,2,3,4,5"`, `"payload,items,comma,separator,nothing"`
— sedangkan lengan `none` (tanpa cache sama sekali) menjawab
`"item1,item2,item3,item4,item5"`. Jadi KV yang dipool **menyisakan cukup
untuk membuat model tetap waras, tetapi tidak menyisakan muatannya**; secara
fungsional ia setara dengan tidak mengirim apa pun. Kalau keluarannya sampah,
kesimpulannya akan berbeda (implementasi rusak) — ia tidak.

**(c) `select_recent` menang HANYA karena eksperimen menaruh jawabannya di
tempat yang ia simpan.** Bandingkan barisnya: 0,060 saat muatan di tengah,
0,920 saat muatan di akhir — anggaran, model, dan kode identik. Inilah kenapa
posisi muatan dijadikan variabel eksplisit di alat ini. Kalau lengan `end` saja
yang dilaporkan, `select_recent@64` akan terlihat seperti "kompresi 41× nyaris
lossless", padahal yang terukur adalah bias resensi, bukan kompresi. Di produksi
informasi yang dibutuhkan hilir tersebar di sepanjang konteks hulu, jadi baris
`mid` yang lebih mewakili.

### 4.4. Keputusan: **B10 TIDAK diadopsi**; alasannya, dan apa yang menggantikannya

Tiga garis bukti menunjuk arah yang sama.

1. **Literatur** (§4.1): pengurangan agresif jumlah pasangan KV hanya
   tercapai dengan modul terlatih (STILL). Yang training-free menyerang sumbu
   lain. B10 sebagaimana ditulis = training-free `L → m` = tepat di kotak yang
   kosong.
2. **Pengukuran kita** (§4.2): empat keluarga training-free, tiga anggaran,
   dua posisi muatan — pooling nol di semua sel.
3. **A9** (§3) mencabut premis B10 dari arah lain, dan ini yang paling
   menentukan. B10 ingin **membuang** token prompt hulu dan menyisakan ringkasan
   padat. Tetapi A9 menunjukkan justru **token prompt itulah kanal yang
   bekerja** (`kv_prompt_only` = 1,000) sementara blok laten sudah lossy
   (`kv_latent_only` = 0,350). Jadi B10 mengusulkan mengompres bagian yang
   berfungsi dan mempertahankan bagian yang sudah bocor.

Menjalankan B10 sebagai perubahan produksi karena itu **tidak dibenarkan oleh
bukti mana pun yang kita punya**. Itu bukan kegagalan rencana: RENCANA §B10
sendiri menetapkan gerbang "jangan dikerjakan sebelum … A8 menunjukkan rantai
agennya layak", dan yang terjadi di sini adalah gerbang tambahan (A9) menutup
lebih dulu — persis fungsi sebuah gerbang.

**Yang tetap valid dari motivasi B10.** Keluhan aslinya benar: konteks emitter
tumbuh linear terhadap jumlah hop (A5: KV construct 4 624 token). Yang keliru
adalah obatnya. Tiga jalan yang konsisten dengan bukti:

- **B4/B5 (prompt ringkas)** — sudah terbukti: redundansi construct 59,5% →
  2,5%, KV total −22%, tanpa menyentuh mekanisme. Ini menyerang pertumbuhan
  konteks pada sumbernya, bukan dengan mengompres akibatnya.
- **B14 (konteks segar + ringkasan terstruktur)** — meringkas dalam ruang
  TEKS, bukan ruang KV. A9 memberinya dasar baru: ringkasan teks memulihkan
  0,84–0,97 sementara pooling KV memulihkan 0,00.
- **Kompaktor terlatih ala STILL** — satu-satunya jalur yang literaturnya
  mendukung untuk `L → m`, tetapi ia melanggar sifat *training-free* yang
  menjadi justru inti klaim LatentMAS dan jangkar skripsi ini. Layak
  disebut di Bab 5 sebagai arah lanjutan, bukan dikerjakan sekarang.

**Nilai B10 bagi skripsi tetap ada, dan bukan nilai negatif.** Bab 5 kini bisa
menyatakan, dengan angka dan bukan dengan dugaan: *bottleneck laten
training-free pada skala konteks produksi tidak mempertahankan muatan simbolik;
kanal yang sesungguhnya bekerja pada mode `kv` adalah token prompt yang
diwariskan, bukan vektor latennya.* Itu klaim yang bisa dipertahankan di
sidang, dan ia lahir dari eksperimen yang dirancang untuk bisa gagal.

### 4.5. Batas berlaku §4

- Diuji pada **satu** jenis muatan (simbolik diskret, k=5) dan **satu** L
  (≈2 613). Pooling mungkin memadai untuk muatan yang memang kontinu — arah
  riset, penekanan, sikap — dan §4 tidak mengukur itu.
- Empat keluarga adalah bottleneck training-free yang **paling wajar**, bukan
  keseluruhan ruangnya. Varian yang belum diuji: pooling per-layer dengan
  anggaran berbeda tiap layer (pola piramidal), pooling yang mempertahankan
  token attention-sink di awal urutan, dan kombinasi seleksi + pooling.
- Semua angka `latent_steps=10` (produksi). Mengingat temuan A9 §3.2(c)
  (kapasitas kanal naik tajam dengan m), bottleneck pada m yang lebih besar
  belum tentu berperilaku sama.

---

## 5. Cara mereproduksi

```bash
# B6 — kapan early-stop menyala, per mode langkah laten (~2 menit GPU)
python lab/early_stop_probe.py --model Qwen/Qwen3-8B --budget 60 --seeds 0,1,2 \
    --tag b6_budget60

# B7 — geometri ridge M (CPU, hanya baca safetensors)
python lab/realign_probe.py --model Qwen/Qwen3-8B
# B7 — inertness use_realign + geometri vektor laten produksi (~3 menit GPU)
python lab/b7_probe.py --model Qwen/Qwen3-8B --steps 10

# A9 — kapasitas kanal (~8 menit GPU per nilai m)
python lab/channel_capacity.py --model Qwen/Qwen3-8B --k 5 --trials 20 \
    --payload dsl,token --latent-steps 10 --tag m10
python lab/channel_capacity.py --model Qwen/Qwen3-8B --k 5 --trials 20 \
    --payload dsl,token --latent-steps 40 --tag m40

# B10 — bottleneck training-free (~6 menit GPU per posisi)
python lab/latent_bottleneck.py --trials 10 --budgets 16,64,256 --position mid --tag mid
python lab/latent_bottleneck.py --trials 10 --budgets 16,64,256 --position end --tag end
```

Mematikan B6 untuk mereplikasi baseline pra-Tahap-4: `--early-stop-cos 1.0`
pada `lab/frontend_probe.py`, atau `latent.early_stop_cos: 1.0` di YAML.
Mengembalikan persamaan realignment lama: `latent.step_mode: "raw"`.

## 6. Rujukan literatur yang dipakai §4

| ref | relevansi |
|---|---|
| STILL — *Amortized KV Cache Compaction in a Single Forward Pass*, arXiv:2606.07878 | kompaktor Perceiver per-layer, 8×–200×, **dilatih**; sumber prosedur un-rotasi/re-rotasi RoPE yang dipakai prototipe kita |
| Q-KVComm — *Efficient Multi-Agent Communication via Adaptive KV Cache Compression*, arXiv:2512.17914 | 5–6× lewat kuantisasi **bit per pasangan**, bukan pengurangan jumlah pasangan |
| KVCOMM, NeurIPS'25, arXiv:2510.12872 | training-free, tetapi soal **penggunaan ulang** prefix yang tumpang tindih (offset alignment), bukan `L → m` |
| Cache-to-Cache, arXiv:2510.03215 | proyektor **terlatih** antar-model heterogen |
| *Beyond tokens: a unified framework for latent communication in LLM-based MAS*, arXiv:2606.05711 | taksonomi WHAT/WHICH/HOW atas 18 metode 2024–2026 — rujukan Bab 2/5 |
| LatentMAS, arXiv:2511.20639 | kerangka yang direplikasi repo ini; Teorema 3.1 (ekspresivitas) & 3.3 (kesetaraan transfer KV) yang diuji A9 |

## 7. Yang belum dikerjakan sesudah ini

- **A10** (sensitivitas arah) dan **A11** (stabilitas jangka panjang) — alatnya
  belum ada.
- **Tahap 5** (`mutation`, `crossover`, `feedback`) — belum disentuh sejak awal
  proyek. A9 kini menyediakan alat yang tepat untuk pertanyaan intinya: apakah
  `guidance_kv` benar-benar memindahkan arah dari Director ke front-end, atau
  hanya tampak begitu.
- **`latent_steps` optimal** — A9 §3.2(c) menunjukkan kapasitas kanal naik
  tajam dari m=10 ke m=40, sedangkan B1 menurunkannya ke 10 demi keandalan.
  Titik yang memaksimalkan kapasitas DENGAN keandalan 6/6 belum dicari, dan
  sekarang bisa dicari karena kedua sumbunya sudah terukur.
