# Landasan Matematis dan Statistis

> Dokumen ini memuat dasar teori di balik Bab III skripsi, ditambah teori sumbu
> interpolasi yang belum ada di sana. Susunannya sengaja seragam: tiap bagian
> dimulai dari **logika awal dan asumsinya** — apa yang sedang diandaikan, dan
> kenapa pengandaian itu masuk akal — baru kemudian **bukti matematisnya**.
> Urutan itu penting karena hampir semua kekeliruan pada penelitian semacam ini
> bukan salah hitung, melainkan asumsi yang tak pernah dinyatakan.
>
> Notasi mengikuti Bab III. Cara menjalankan eksperimennya ada di
> [`PANDUAN.md`](PANDUAN.md); apa yang diukur dan kenapa ada di
> [`DESAIN_EKSPERIMEN.md`](DESAIN_EKSPERIMEN.md).

## Notasi

| lambang | arti |
|---|---|
| $d$ | dimensi hidden state |
| $V$ | ukuran kosakata |
| $W_\text{in} \in \mathbb{R}^{V\times d}$ | matriks embedding masukan |
| $W_\text{out} \in \mathbb{R}^{V\times d}$ | matriks *language-model head* |
| $h \in \mathbb{R}^{d}$ | hidden state lapisan terakhir |
| $\ell = W_\text{out} h \in \mathbb{R}^{V}$ | logit |
| $p = \operatorname{softmax}(\ell/T)$ | distribusi token, $T = 0{,}7$ |
| $e_y$ | vektor satuan pada indeks $y$ |
| $\Delta^{V-1}$ | simpleks probabilitas atas $V$ token |
| $\Phi$ | fungsi langkah laten, $z = \Phi(h)$ |
| $\rho$ | magnitudo rata-rata baris $W_\text{in}$ |

---

# Bagian 1 — Kenapa dibutuhkan sebuah persamaan sama sekali

## 1.1 Logika awal

Pada pembangkitan token biasa, keluaran satu langkah adalah **token diskret**,
dan token itu punya barisnya sendiri di $W_\text{in}$ yang siap dipakai sebagai
masukan langkah berikutnya. Lingkarannya tertutup dengan sendirinya.

Penalaran laten memutus lingkaran itu. Satu langkah laten sengaja **tidak**
memilih token; yang tersedia hanya $h \in \mathbb{R}^d$, keluaran aliran
Transformer. Tetapi masukan model harus berupa vektor di ruang embedding
**masukan**. Jadi ada celah yang harus dijembatani:

$$h \in \text{ruang keluaran} \quad\longrightarrow\quad z \in \text{ruang masukan}.$$

**Asumsi yang sedang dibuat, dan jarang dinyatakan:** bahwa kedua ruang itu
*bisa* dijembatani oleh sebuah fungsi tetap $\Phi$, dan bahwa pilihan $\Phi$
tidak banyak berpengaruh. Asumsi kedua itulah yang diuji seluruh penelitian ini.
Kalau $\Phi$ ternyata menentukan, maka setiap sistem kolaborasi laten bertumpu
pada satu komponen yang tak pernah dibandingkan terhadap alternatif.

Iterasinya:

$$z_t = \Phi(h_t), \qquad h_{t+1} = f_\theta\big(z_t \mid \mathrm{KV}_{1:t}\big),
\qquad t = 1,\dots,m, \quad m = 10.$$

## 1.2 Penyelesaian LatentMAS: regresi ridge

LatentMAS menjembataninya dengan satu pemetaan linier $M \in \mathbb{R}^{d\times d}$
yang dilatih agar baris $W_\text{out}$ terpetakan ke baris $W_\text{in}$:

$$M = \arg\min_{M} \Big\{ \lVert W_\text{out} M - W_\text{in}\rVert_F^2 + \lambda \lVert M \rVert_F^2 \Big\}, \qquad \lambda = 10^{-5}.$$

**Bukti bentuk tertutup.** Tulis $J(M) = \lVert W_\text{out}M - W_\text{in}\rVert_F^2 + \lambda\lVert M\rVert_F^2$.
Karena $\lVert A \rVert_F^2 = \operatorname{tr}(A^\top A)$,

$$J(M) = \operatorname{tr}\big[(W_\text{out}M - W_\text{in})^\top(W_\text{out}M - W_\text{in})\big] + \lambda\operatorname{tr}(M^\top M).$$

Turunan terhadap $M$ (memakai $\partial_M \operatorname{tr}(M^\top A M) = 2AM$ dan
$\partial_M \operatorname{tr}(M^\top B) = B$):

$$\frac{\partial J}{\partial M} = 2W_\text{out}^\top W_\text{out} M - 2W_\text{out}^\top W_\text{in} + 2\lambda M.$$

Disamakan nol:

$$\big(W_\text{out}^\top W_\text{out} + \lambda I\big) M = W_\text{out}^\top W_\text{in}
\;\Longrightarrow\;
\boxed{M = \big(W_\text{out}^\top W_\text{out} + \lambda I\big)^{-1} W_\text{out}^\top W_\text{in}.}$$

Suku $\lambda I$ menjamin matriks yang dibalik bernilai definit positif,
sehingga penyelesaiannya tunggal dan ada meskipun $W_\text{out}^\top W_\text{out}$
singular. $J$ konveks tegas dalam $M$ untuk $\lambda>0$, jadi titik stasioner
itu minimum global — bukan sekadar titik kritis.

Hasilnya dinormalisasi ke magnitudo rata-rata embedding masukan,
$\rho = \tfrac1V\sum_i \lVert W_\text{in}[i]\rVert$:

$$\Phi_\text{raw}(h) = \rho\,\frac{hM}{\lVert hM\rVert}.$$

## 1.3 Celah antara yang dioptimalkan dan yang dipakai

Inilah bagian yang paling penting, dan yang memotivasi seluruh penelitian.

Fungsi objektif di atas mengoptimalkan $M$ pada **baris-baris $W_\text{out}$**.
Tetapi saat dipakai, argumen $\Phi_\text{raw}$ bukan baris $W_\text{out}$ —
melainkan **hidden state aktual** $h$ yang dihasilkan aliran Transformer. Dua
pernyataan berikut berbeda, dan yang pertama tidak menyiratkan yang kedua:

1. $M$ merekonstruksi $W_\text{in}$ dari baris $W_\text{out}$ dengan galat kecil;
2. $hM$ mendarat di wilayah yang berperilaku seperti embedding masukan, untuk
   $h$ yang benar-benar muncul saat inferensi.

Selama $h$ berada di rentang baris $W_\text{out}$, pernyataan (2) mengikuti (1).
Tidak ada yang menjamin itu. Dan pada Qwen3-8B, pengukurannya menunjukkan
justru sebaliknya:

$$\frac{\lVert M - I\rVert_F}{\lVert I \rVert_F} = 1{,}04,
\qquad \overline{\cos(h, hM)} = 0{,}011,
\qquad \overline{\cos(W_\text{in}[i], W_\text{out}[i])} = 0{,}0041.$$

Angka ketiga menjelaskan dua yang pertama: pada model tanpa *tied embeddings*,
baris ke-$i$ dari kedua matriks nyaris ortogonal, sehingga $M$ memang **harus**
memutar jauh. Akibatnya $hM$ tak lagi menyerupai embedding token mana pun —
terukur $\max_i \cos(\Phi_\text{raw}(h), W_\text{in}[i]) \approx 0{,}31$.

> **Batas klaim.** Ini bukan bukti bahwa KV-cache kehilangan informasi. Ia
> memisahkan dua hal yang mudah tertukar: *pembentukan* representasi laten
> (yang diuji di sini) dan *pengangkutan* KV antar-agen (yang tidak dibantah
> data mana pun di penelitian ini).

---

# Bagian 2 — Kerangka penyatuan

## 2.1 Logika awal

Tiga metode dari literatur penalaran laten model tunggal — *Soft Thinking*,
*Stochastic Soft Thinking* (Gumbel), dan *Mixture of Inputs* — dibahas di paper
yang berbeda dengan notasi yang berbeda, seolah tiga mekanisme terpisah.

Pengamatan yang menyatukannya: ketiganya, ditambah pengambilan sampel
kategoris biasa, mengerjakan **dua langkah yang sama**. Pertama, ubah $h$
menjadi bobot atas kosakata. Kedua, campur baris $W_\text{in}$ dengan bobot itu.
Yang berbeda hanya **aturan pembentukan bobotnya**.

Kalau pengamatan itu benar, perbandingan antar-metode menjadi perbandingan
antar-aturan-bobot pada ruang yang sama — bukan perbandingan antar-arsitektur.
Itu jauh lebih terkendali, dan memungkinkan pertanyaan yang lebih tajam:
apakah yang menentukan adalah *varian mana yang dipilih*, atau *keanggotaan
pada keluarga itu sendiri*?

## 2.2 Definisi

Simpleks probabilitas:

$$\Delta^{V-1} = \Big\{ w \in \mathbb{R}^V \;\Big|\; w_i \ge 0,\ \textstyle\sum_{i=1}^V w_i = 1 \Big\}.$$

Sebuah metode termasuk **keluarga relaksasi diskret** $\mathcal R$ bila
representasinya (sebelum normalisasi) dapat ditulis

$$\tilde z = \sum_{i=1}^{V} w_i\, W_\text{in}[i] = w^\top W_\text{in},
\qquad w \in \Delta^{V-1}. \tag{$\star$}$$

Keempat anggotanya:

| metode | bobot $w$ |
|---|---|
| `soft` | $w = p$ |
| `sample` | $w = e_y$, $\ y \sim \operatorname{Cat}(p)$ |
| `gumbel` | $w = \operatorname{softmax}\big((\ell + g)/\tau\big)$, $\ g_i \overset{\text{iid}}{\sim} \operatorname{Gumbel}(0,1)$ |
| `moi` | $w = \dfrac{H p + (\beta + 1 - H)\,e_y}{\beta + 1}$ |

dengan $H$ entropi ternormalisasi $H = -\sum_i p_i \log p_i / \log V \in [0,1]$
dan $\beta \ge 0$ ($\beta = 1$ = setelan baku paper MoI).

**Arti geometris $(\star)$.** Himpunan semua $w^\top W_\text{in}$ dengan
$w \in \Delta^{V-1}$ persis adalah **lambung konveks** baris-baris
$W_\text{in}$ — menurut definisi lambung konveks sebagai himpunan semua
kombinasi konveks. Jadi keanggotaan $\mathcal R$ setara dengan pernyataan
"langkah laten selalu mendarat di dalam lambung konveks embedding token",
yakni di dalam wilayah yang bisa dinyatakan sebagai campuran token-token yang
memang pernah dilihat model.

## 2.3 Proposisi 1 — MoI adalah kombinasi konveks `soft` dan `sample`

**Logika awal.** MoI diperkenalkan dengan bahasa Bayes: distribusi $p$ sebagai
*prior* Dirichlet, token tersampel sebagai *observasi*, masukan berikutnya
sebagai ekspektasi *posterior*. Bahasa itu menyembunyikan hal sederhana — bahwa
hasilnya cuma rata-rata berbobot dua metode lain.

**Pernyataan.** Untuk setiap $h$, $\beta \ge 0$, dan $y$ yang tersampel,

$$w_\text{moi} = \alpha\, w_\text{soft} + (1-\alpha)\, w_\text{sample},
\qquad \alpha = \frac{H}{\beta+1} \in [0,1],$$

dan karenanya $w_\text{moi} \in \Delta^{V-1}$, yakni MoI $\in \mathcal R$.

**Bukti.** Dari definisi,

$$w_\text{moi} = \frac{H p + (\beta+1-H) e_y}{\beta+1}
= \frac{H}{\beta+1}\,p + \frac{\beta+1-H}{\beta+1}\,e_y.$$

Tetapkan $\alpha = H/(\beta+1)$. Maka koefisien kedua adalah

$$\frac{\beta+1-H}{\beta+1} = 1 - \frac{H}{\beta+1} = 1-\alpha,$$

sehingga $w_\text{moi} = \alpha p + (1-\alpha)e_y = \alpha w_\text{soft} + (1-\alpha) w_\text{sample}$.

Selanjutnya $H \in [0,1]$ karena entropi Shannon atas $V$ hasil terbatas oleh
$\log V$ (maksimum pada distribusi seragam), dan ternormalisasi oleh $\log V$.
Dengan $\beta \ge 0$ maka $\beta + 1 \ge 1 > 0$, sehingga

$$0 \le \alpha = \frac{H}{\beta+1} \le \frac{1}{1} = 1.$$

Karena $\Delta^{V-1}$ konveks (irisan setengah-ruang $w_i \ge 0$ dengan
hiperbidang $\sum_i w_i = 1$, keduanya konveks) dan $w_\text{soft}, w_\text{sample} \in \Delta^{V-1}$,
maka kombinasi konveks keduanya juga di $\Delta^{V-1}$. $\blacksquare$

**Akibat yang tidak sepele.** Pada $\beta = 1$ berlaku $\alpha = H/2 \le \tfrac12$.
Jadi kontribusi distribusi $p$ **tak pernah melebihi separuh** bobot: pada
setelan bakunya, MoI secara struktural berlabuh lebih dekat ke token diskret
daripada ke distribusi penuh. Ini prediksi yang bisa diperiksa langsung lewat
probe geometri — dan yang membuat baris `moi` di tabel geometri wajib ada.

## 2.4 Proposisi 2 — Gumbel terhubung kontinu ke `sample`

**Logika awal.** *Soft Thinking* deterministik cenderung terkunci pada token
berpeluang tertinggi. Derau Gumbel ditambahkan untuk melonggarkannya. Pertanyaan
teoretisnya: apakah penambahan derau itu memindahkan metode ke tempat lain, atau
ia hanya bergerak di sepanjang garis yang sudah ada antara `soft` dan `sample`?

**Pernyataan (Gumbel-Max).** Bila $g_i \overset{\text{iid}}{\sim}\operatorname{Gumbel}(0,1)$, maka

$$\arg\max_i \,(\ell_i + g_i) \;\sim\; \operatorname{Cat}\big(\operatorname{softmax}(\ell)\big).$$

**Bukti.** Gumbel baku punya CDF $F(x) = \exp(-e^{-x})$ dan pdf
$f(x) = e^{-x}\exp(-e^{-x})$. Peluang indeks $k$ menjadi pemenang:

$$\Pr[k = \arg\max] = \int_{-\infty}^{\infty} f(x-\ell_k) \prod_{i\ne k} F(x - \ell_i)\,dx
= \int_{-\infty}^{\infty} e^{-(x-\ell_k)} \exp\!\Big(-\sum_{i} e^{-(x-\ell_i)}\Big) dx.$$

Tulis $S = \sum_i e^{\ell_i}$ dan substitusikan $u = e^{-x}S$, sehingga
$du = -e^{-x} S\,dx$:

$$= \int_{0}^{\infty} \frac{e^{\ell_k}}{S}\,e^{-u}\,du = \frac{e^{\ell_k}}{S} = \operatorname{softmax}(\ell)_k. \qquad\blacksquare$$

**Akibat.** Gumbel-Softmax adalah relaksasi kontinu dari argmax itu:

$$w_\text{gumbel} = \operatorname{softmax}\big((\ell+g)/\tau\big).$$

Untuk $\tau \to 0^+$, softmax memusat pada koordinat terbesar, sehingga
$w_\text{gumbel} \to e_{y}$ dengan $y = \arg\max_i(\ell_i+g_i) \sim \operatorname{Cat}(\operatorname{softmax}(\ell))$,
yakni persis `sample`. Untuk $\tau$ membesar, $(\ell+g)/\tau \to 0$ dan
$w_\text{gumbel}$ menuju distribusi seragam.

Jadi `gumbel` menempuh interpolasi antara dua ekstrem yang **sama** dengan yang
dijembatani MoI, hanya lewat jalur berbeda: MoI mencampur eksplisit di simpleks,
Gumbel lewat suhu di ruang logit. Ketiga metode itu bukan tiga mekanisme
terpisah, melainkan tiga cara menempatkan diri pada ruang yang sama.

## 2.5 Proposisi 3 — `raw` berada di luar keluarga

**Logika awal.** Godaan yang harus dihindari: menyimpulkan bahwa beda `raw`
dengan yang lain adalah "linear versus taklinear". Itu keliru — `soft` pun
linear dalam $p$, dan `raw` pun linear dalam $h$. Yang membedakan bukan
linearitas melainkan **kendala**.

**Pernyataan.** Fungsi objektif pembentuk $M$ tidak memuat satu pun suku yang
memaksa representasi $\tilde z_\text{raw} = hM$ ditulis sebagai kombinasi
konveks baris $W_\text{in}$. Karena itu `raw` $\notin \mathcal R$.

**Bukti.** Andaikan `raw` anggota $\mathcal R$. Maka untuk setiap $h$ ada
$w(h) \in \Delta^{V-1}$ dengan $hM = w(h)^\top W_\text{in}$. Dua akibat langsung:

1. **Kendala tanda.** Setiap $\tilde z \in \operatorname{conv}(W_\text{in})$
   terletak di lambung konveks — himpunan terbatas. Sementara $h \mapsto hM$
   linear dan surjektif ke ruang bagian $\operatorname{row}(M)$, sehingga untuk
   $h$ berskala besar $\lVert hM \rVert$ tumbuh tanpa batas. Lambung konveks
   dari himpunan terbatas $\{W_\text{in}[i]\}$ juga terbatas, jadi tak mungkin
   memuat seluruh citra itu.
2. **Kendala normalisasi.** $\sum_i w_i = 1$ menetapkan sebuah hiperbidang
   afin; tidak ada suku pada $J(M)$ yang mensyaratkannya.

Karena itu pengandaian tersebut gugur. $\blacksquare$

Yang tersisa bukan kepastian bahwa $hM$ *selalu* di luar lambung, melainkan
bahwa **tidak ada yang menjaminnya berada di dalam** — dan pengukuran
$\max_i \cos = 0{,}31$ menunjukkan pada praktiknya ia memang jauh di luar,
sementara anggota $\mathcal R$ berada di $0{,}93$–$1{,}00$.

> Nilai tepat $1{,}000$ pada `sample` bukan kebetulan melainkan konsekuensi
> definisi: $w = e_y$ memberi $\tilde z = W_\text{in}[y]$, satu baris embedding
> persis, sehingga kosinus terhadap tetangga terdekatnya adalah $1$ menurut
> konstruksi. Angka itu berfungsi sebagai **uji kewarasan** pipeline
> pengukuran: kalau ia bukan $1$, yang rusak adalah probenya.

## 2.6 Konvensi normalisasi

**Logika awal.** Kalau metode-metode dibandingkan tanpa penyeragaman panjang
vektor, selisih yang terukur bisa berasal dari magnitudo semata — dan magnitudo
adalah properti yang paling mudah diubah tanpa mengubah isi informasi.

Karena itu semua metode memakai konvensi yang sama:

$$\Phi(h) = \rho\,\frac{\tilde z}{\lVert \tilde z\rVert}.$$

**Akibat geometris.** Seluruh metode dibandingkan pada permukaan bola
berjari-jari $\rho$ di $\mathbb{R}^d$. Yang tersisa sebagai variabel adalah
**arah**, bukan panjang. Ukuran $\max_i \cos(z, W_\text{in}[i])$ pun murni
ukuran arah, sehingga sumbu geometris dan sumbu eksperimen sejalan.

> Ini keputusan *harness*, bukan sifat asli tiap paper sumber. Paper MoI,
> misalnya, tidak menskalakan ulang embedding. Konsekuensinya harus dinyatakan:
> hasil penelitian ini berlaku **pada kondisi normalisasi tersebut**, dan
> pertanyaan "apakah magnitudo juga berpengaruh" tidak dijawab di sini.

---

# Bagian 3 — Teori sumbu interpolasi

## 3.1 Logika awal: kenapa lima titik tidak cukup

Setelah Bagian 2, kita punya lima formulasi dan dua ukuran: kedekatan geometris
ke embedding token, dan kinerja tugas. Terukur bahwa keduanya bergerak searah —
`raw` rendah pada keduanya, anggota $\mathcal R$ tinggi pada keduanya.

Tetapi lima titik yang berasal dari lima **mekanisme berbeda** hanya bisa
mendukung pernyataan "keduanya berkorelasi". Ia tak bisa memisahkan dua
penjelasan yang sangat berbeda:

- **(A)** Geometri itulah yang menentukan. Semakin jauh langkah laten dari
  lambung konveks embedding, semakin rusak identitas token yang dibawanya.
- **(B)** Geometri hanya penanda. Yang sebenarnya menentukan adalah sesuatu
  yang kebetulan menyertai `raw` — misalnya bahwa ia satu-satunya yang tidak
  melewati distribusi $p$ sama sekali.

Membedakan (A) dari (B) menuntut **memvariasikan geometri sambil menahan hal
lain tetap**. Itulah yang tidak bisa dilakukan dengan lima metode yang berbeda
di banyak hal sekaligus, dan itulah yang bisa dilakukan satu keluarga berparameter.

**Karena itu `mix` bukan usulan metode baru.** Ia alat ukur. Tak ada klaim
bahwa seseorang sebaiknya memakai $\alpha = 0{,}5$ di sistem produksi.

## 3.2 Definisi

Untuk $\alpha \in [0,1]$,

$$\Phi_\text{mix}^{(\alpha)}(h) \;=\; \rho\,\frac{u_\alpha}{\lVert u_\alpha\rVert},
\qquad u_\alpha = (1-\alpha)\,\Phi_\text{raw}(h) + \alpha\,\Phi_\text{soft}(h).$$

Perhatikan bahwa yang dicampur adalah **keluaran akhir kedua metode** — masing-masing
sudah ternormalisasi ke $\rho$ — bukan bobot atau logitnya. Pilihan itu disengaja
dan dibuktikan konsekuensinya di §3.3–3.4.

## 3.3 Proposisi 4 — kedua titik ujung mereduksi persis

**Pernyataan.** $\Phi_\text{mix}^{(0)} = \Phi_\text{raw}$ dan $\Phi_\text{mix}^{(1)} = \Phi_\text{soft}$, sebagai fungsi, bukan hanya secara hampiran.

**Bukti.** Untuk $\alpha = 0$: $u_0 = \Phi_\text{raw}(h)$, yang menurut definisi
sudah bernorma $\rho$. Maka

$$\Phi_\text{mix}^{(0)}(h) = \rho\,\frac{\Phi_\text{raw}(h)}{\lVert\Phi_\text{raw}(h)\rVert} = \rho\,\frac{\Phi_\text{raw}(h)}{\rho} = \Phi_\text{raw}(h).$$

Serupa untuk $\alpha=1$. $\blacksquare$

**Kenapa ini penting secara praktis, bukan hanya rapi.** Karena kedua ujungnya
*identik* dengan metode yang selnya sudah dijalankan, kurva dose–response
menyambung langsung ke matriks eksperimen yang ada: hanya tiga nilai tengah
yang perlu GPU, dan titik ujungnya tidak menambah derau sampling pada dua titik
yang seharusnya berimpit. Sifat ini diverifikasi numerik (galat maksimum
$0{,}0\times10^{0}$, yakni identik bit-per-bit).

## 3.4 Proposisi 5 — sepanjang $\alpha$ hanya arah yang berubah

**Pernyataan.** $\lVert \Phi_\text{mix}^{(\alpha)}(h)\rVert = \rho$ untuk semua
$\alpha$, dan lintasan $\alpha \mapsto \Phi_\text{mix}^{(\alpha)}(h)$ menelusuri
busur lingkaran besar pada bola berjari-jari $\rho$, dari $\Phi_\text{raw}(h)$
ke $\Phi_\text{soft}(h)$.

**Bukti.** Norma konstan langsung dari normalisasi akhir, asalkan $u_\alpha \ne 0$
— yang berlaku kecuali kedua ujungnya persis berlawanan arah, keadaan berukuran
nol yang tak pernah teramati (terukur $\cos \approx 0{,}095$, jauh dari $-1$).

Untuk bentuk lintasannya: tulis $a = \Phi_\text{raw}(h)$ dan $b = \Phi_\text{soft}(h)$,
keduanya bernorma $\rho$. Maka $u_\alpha = (1-\alpha)a + \alpha b$ terletak pada
ruas garis yang menghubungkan $a$ dan $b$, yaitu tali busur. Seluruh ruas itu
terletak di bidang dua dimensi $\operatorname{span}\{a,b\}$. Memproyeksikan tiap
titiknya kembali ke bola berjari-jari $\rho$ menghasilkan kurva pada irisan
bola dengan bidang tersebut — yakni lingkaran besar. Karena $u_\alpha$ bergerak
monoton sepanjang tali busur, proyeksinya bergerak monoton sepanjang busur
pendek antara $a$ dan $b$. $\blacksquare$

Verifikasi numeriknya (vokabulari sintetis, $d=16$):

| $\alpha$ | $\lVert z\rVert$ | $\cos(z, z_\text{raw})$ | $\cos(z, z_\text{soft})$ |
|---:|---:|---:|---:|
| 0,00 | 3,7000 | 1,0000 | 0,0954 |
| 0,25 | 3,7000 | 0,9520 | 0,3956 |
| 0,50 | 3,7000 | 0,7401 | 0,7401 |
| 0,75 | 3,7000 | 0,3956 | 0,9520 |
| 1,00 | 3,7000 | 0,0954 | 1,0000 |

Kesimetrian sempurna di $\alpha = 0{,}5$ adalah tanda bahwa lintasannya memang
busur, bukan campuran yang bias ke salah satu ujung.

## 3.5 Apa yang kurva ini bisa dan tidak bisa buktikan

**Yang bisa.** Bila ukuran geometris $s(\alpha) = \max_i\cos(\Phi^{(\alpha)}_\text{mix}(h), W_\text{in}[i])$
dan ukuran kinerja $y(\alpha)$ bergerak bersama secara monoton, maka penjelasan
(B) di §3.1 menjadi jauh lebih sulit dipertahankan: sepanjang sumbu ini,
*hanya* geometri yang berubah — mekanismenya satu, keluarganya satu, dan tak
ada perbedaan lain yang bisa dituduh.

**Yang tidak bisa.** Ini tetap bukan eksperimen kausal dalam arti ketat. $\alpha$
mengubah arah vektor, dan arah itu memengaruhi banyak hal hilir sekaligus.
Yang diperoleh adalah **hubungan dosis–respons**, yang di banyak bidang
diperlakukan sebagai bukti kuat tapi bukan bukti mutlak.

**Tiga bentuk hasil, dan ketiganya temuan:**

| bentuk | pembacaan |
|---|---|
| monoton | geometri memprediksi kinerja; klaim mekanistik Bab IV menguat |
| ber-ambang | ada batas jarak dari lambung konveks; di bawahnya runtuh, di atasnya datar |
| tak berpola | geometri **bukan** penjelasnya; klaim mekanistik harus dilemahkan |

> **Karena itu hipotesisnya sengaja tidak berarah.** Menuliskan "semakin dekat
> semakin baik" sebelum data ada akan mengubah eksperimen ini dari pengujian
> menjadi pencarian pembenaran.

## 3.6 Kenapa sumbu-$x$ harus diukur, bukan diasumsikan

Godaan berikutnya: memplot kinerja terhadap $\alpha$ langsung. Itu keliru,
karena $\alpha$ adalah bobot pencampuran, **bukan** jarak ke lambung konveks.
Hubungan $\alpha \mapsto s(\alpha)$ tidak linear — dari tabel §3.4 terlihat
$\cos$ bergerak cepat di tengah dan lambat di ujung.

Karena itu probe geometri (`b7_probe.py --alphas`) mengukur $s(\alpha)$ secara
langsung untuk tiap $\alpha$ yang dijalankan, dan kurva dose–response diplot
sebagai $y$ terhadap $s$, bukan $y$ terhadap $\alpha$. Biayanya nyaris nol —
tak ada pembangkitan teks sama sekali, hanya rollout laten.

---

# Bagian 4 — Landasan statistik

## 4.1 Logika awal: kenapa berpasangan

Setiap kondisi eksperimen mengerjakan **soal yang sama persis**. Rancangan itu
dipilih karena ragam antar-soal jauh lebih besar daripada selisih antar-metode
yang ingin dideteksi: soal GSM8K yang sulit akan salah pada hampir semua metode.
Rancangan berpasangan membuang ragam itu dari galat baku, karena tiap soal
menjadi kontrolnya sendiri.

Syaratnya keras: kalau himpunan soalnya berbeda, uji berpasangan **tidak sah**.
Karena itu kesamaannya diverifikasi mekanis lewat sidik jari indeks soal, bukan
diasumsikan dari pemakaian *seed* yang sama; `bench/compare.py` mengeluarkan sel
yang sidik jarinya tak cocok.

## 4.2 Uji McNemar eksak

**Asumsi.** Untuk dua kondisi $A$ dan $B$ pada $n$ soal berpasangan, susun

| | $B$ benar | $B$ salah |
|---|---:|---:|
| **$A$ benar** | $a$ | $b$ |
| **$A$ salah** | $c$ | $d$ |

Hipotesis nolnya adalah **simetri marginal**: $\Pr[A \text{ benar}] = \Pr[B\text{ benar}]$.

**Turunan.** Pasangan konkordan ($a$ dan $d$) tidak membawa informasi tentang
selisih marginal: keduanya menambah cacah yang sama pada kedua sisi. Yang
informatif hanya pasangan diskordan. Di bawah $H_0$,

$$\Pr[A\text{ benar}, B\text{ salah}] = \Pr[A\text{ salah}, B\text{ benar}],$$

sehingga, **dikondisikan** pada jumlah diskordan $n_d = b+c$, cacah $b$
mengikuti

$$b \mid n_d \;\sim\; \operatorname{Binomial}\!\big(n_d, \tfrac12\big).$$

Nilai $p$ dua sisi eksaknya adalah peluang ekor binomial itu:

$$p = 2\min\Big\{ \Pr[X \le \min(b,c)],\ \tfrac12 \Big\},
\qquad X \sim \operatorname{Binomial}(n_d, \tfrac12),$$

dipotong di 1. Versi eksak dipakai — bukan hampiran $\chi^2$ dengan statistik
$(b-c)^2/(b+c)$ — karena $n_d$ bisa kecil, dan pada $n_d$ kecil hampiran
$\chi^2$ memberi nilai $p$ yang terlalu kecil (uji jadi terlalu sering menolak).

## 4.3 Selang kepercayaan bootstrap

Nilai $p$ menjawab "apakah selisihnya bisa dibedakan dari nol", tidak menjawab
"seberapa besar". Karena itu tiap perbandingan disertai selang kepercayaan
persentil bootstrap 95%: sampel ulang $n$ **pasangan** (bukan pengamatan
individual — pasangannya yang merupakan unit acak) dengan pengembalian sebanyak
$B$ kali, hitung selisih akurasinya di tiap replikasi, ambil persentil 2,5 dan
97,5.

Bootstrap dipilih karena statistiknya adalah selisih dua proporsi berkorelasi,
yang distribusi sampelnya tidak berbentuk sederhana; menyampel ulang pasangan
mempertahankan korelasi itu secara otomatis.

## 4.4 Uji Cochran $Q$

**Logika awal.** Uji berpasangan menjawab pertanyaan pasangan-demi-pasangan.
Pertanyaan yang lebih pokok — apakah **kelima** persamaan berbeda sama sekali —
menuntut satu uji menyeluruh, kalau tidak, tiap pasangan diuji lalu yang paling
ekstrem dilaporkan, dan itu bentuk lain dari mengintip data.

**Pernyataan.** Untuk $k$ kondisi berpasangan dengan hasil biner atas $n$ blok
(di sini: soal), dengan $X_{ij}\in\{0,1\}$, $C_j = \sum_i X_{ij}$ total kolom
$j$, dan $R_i = \sum_j X_{ij}$ total baris $i$:

$$Q = \frac{(k-1)\Big[k\sum_{j=1}^{k} C_j^2 - \big(\sum_j C_j\big)^2\Big]}
{k\sum_{i=1}^{n} R_i - \sum_{i=1}^{n} R_i^2}.$$

Di bawah $H_0$ (semua kondisi punya peluang sukses sama), $Q \xrightarrow{d} \chi^2_{k-1}$.

**Intuisi turunannya.** Pembilang mengukur ragam antar-total-kolom — besar bila
kondisi berbeda. Penyebut adalah faktor penormalisasi yang hanya bergantung pada
total baris, yaitu bagian yang **tetap** di bawah permutasi label kondisi di
dalam tiap blok. Baris yang seluruhnya 0 atau seluruhnya 1 menyumbang nol pada
penyebut — konsisten dengan gagasan bahwa soal yang dijawab sama oleh semua
kondisi tak membawa informasi, persis seperti pasangan konkordan pada McNemar.

## 4.5 Koreksi multiplisitas

**Logika awal.** Dengan 7 perlakuan ada $\binom{7}{2}=21$ pasangan per tugas,
dan tiga tugas memberi 63 pengujian. Bila tiap uji dibaca pada $\alpha = 0{,}05$,
maka di bawah $H_0$ menyeluruh peluang munculnya **setidaknya satu** temuan
palsu adalah $1-(1-0{,}05)^{63} \approx 0{,}96$. Membaca hasilnya tanpa koreksi
karena itu praktis dijamin menghasilkan temuan palsu.

**Holm (mengendalikan FWER).** Urutkan $p_{(1)}\le\dots\le p_{(m)}$. Tolak
$H_{(i)}$ selama $p_{(i)} \le \alpha/(m-i+1)$, berhenti pada pelanggaran
pertama. Prosedur ini mengendalikan peluang **setidaknya satu** penolakan salah
pada tingkat $\alpha$, tanpa asumsi struktur ketergantungan.

**Benjamini–Hochberg (mengendalikan FDR).** Cari $i$ terbesar dengan
$p_{(i)} \le \tfrac{i}{m}\alpha$, tolak semua hipotesis sampai peringkat itu.
Ia mengendalikan **proporsi harapan** penolakan yang salah, lebih longgar dari
Holm, sehingga dilaporkan berdampingan: bila keduanya memberi himpunan yang
sama — sebagaimana yang terjadi — kesimpulannya tidak bergantung pada pilihan
prosedur.

## 4.6 Kenapa lengan faktor tidak diuji seperti lengan bench

**Logika awal.** Godaannya adalah memperlakukan kedua lengan sama karena
keduanya setara kedudukannya. Tetapi kesetaraan kedudukan bukan kesetaraan
ukuran sampel: lengan bench punya 100 soal berpasangan per sel, lengan faktor
20 jalan.

**Perhitungan daya.** Untuk uji McNemar eksak dua sisi pada $\alpha=0{,}05$,
dengan skenario paling menguntungkan (seluruh ketaksesuaian searah), daya untuk
mendeteksi selisih proporsi $d$:

| $n$ jalan | $d = 0{,}15$ | $d = 0{,}25$ | $d = 0{,}35$ | $d = 0{,}50$ |
|---:|---:|---:|---:|---:|
| 6 | 0,000 | 0,000 | 0,002 | 0,016 |
| 20 | 0,067 | 0,383 | 0,755 | 0,979 |
| 40 | 0,567 | 0,957 | 0,999 | 1,000 |
| 100 | 0,998 | 1,000 | 1,000 | 1,000 |

Angka ini menghasilkan kesimpulan yang **lebih tajam** daripada "jangan diuji":

- Pada $n=6$ (run 2026-08-10), daya praktis nol bahkan untuk selisih 50 poin.
  Uji formal di sana bukan sekadar lemah — ia tak bermakna. Keputusan lama
  melaporkan lengan faktor secara deskriptif karena itu **benar**.
- Pada $n=20$, kontras yang besar **layak diuji**. Selisih lolos gate antara
  `raw` (17%) dan anggota $\mathcal R$ (67–100%) berada di $d \approx 0{,}5$–$0{,}83$,
  wilayah dengan daya $\ge 0{,}98$.
- Pada $n=20$, kontras **antar-anggota** $\mathcal R$ (misal gumbel 83% vs
  sample 67%, $d\approx0{,}17$) punya daya $\approx 0{,}07$. Menguji itu lalu
  melaporkan "tidak signifikan" akan menyesatkan: ketidaksignifikanannya
  ditentukan rancangan, bukan data.

**Kebijakan yang mengikuti**, dan yang harus dinyatakan di skripsi: pada lengan
faktor, **satu** kontras diuji secara formal — keluarga $\mathcal R$ terhadap
`raw`, karena hanya itu yang berdaya — sementara seluruh sisanya dilaporkan
lewat enam level bukti secara deskriptif. Ini bukan kompromi: ia justru cermin
struktur H1, yang memang memprediksi keanggotaan keluarga yang menentukan dan
bukan varian di dalamnya.

## 4.7 Prinsip interpretasi

Hasil yang tidak signifikan dilaporkan sebagai **tidak signifikan**. Ungkapan
seperti "cenderung lebih baik" tidak dipakai untuk menggantikan hasil yang tak
mencapai ambang.

Prinsip itu penting khusus untuk H1, yang **memprediksi** kemiripan antar-anggota
$\mathcal R$. Tidak ditemukannya perbedaan antar-varian karena itu bukan
kegagalan eksperimen melainkan salah satu kemungkinan hasil yang relevan — asal
disertai keterangan daya seperti §4.6, supaya "tak ada beda" tidak tertukar
dengan "tak sanggup mendeteksi beda".

Prinsip pendampingnya adalah **ketelitian ruang lingkup**. Pernyataan bahwa
suatu mekanisme "mempertahankan informasi" harus selalu menyebut *informasi
apa*, *pada tahap mana*, dan *dalam arti apa*. Penelitian ini karena itu
memisahkan secara eksplisit:

1. **fidelitas pengangkutan KV** — kemampuan KV-cache meneruskan representasi
   yang sudah tersimpan; tidak dibantah data mana pun di sini;
2. **fidelitas pembentukan** — kemampuan $\Phi$ menghasilkan representasi yang
   masih bisa diproses agen berikutnya; inilah yang diuji.

Menukar keduanya menghasilkan kalimat yang faktanya benar tetapi kesimpulannya
menyesatkan.
