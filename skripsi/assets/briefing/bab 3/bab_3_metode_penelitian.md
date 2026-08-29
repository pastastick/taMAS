# BAB III
# METODE PENELITIAN

Bab ini membahas metode yang digunakan dalam penelitian untuk membandingkan persamaan langkah laten pada kolaborasi multi-agen berbasis *large language model* (LLM) [cite: 1]. Berbeda dengan pendekatan pada umumnya, penelitian ini berfokus pada evaluasi persamaan langkah laten ($\Phi$) sebagai objek utama yang menentukan keberhasilan transfer informasi laten [cite: 2]. Pembuktian dan penurunan matematis dari masing-masing metode—seperti *ridge regression* pada LatentMAS, *Gumbel-Max trick* pada Stochastic Soft Thinking, dan inferensi Bayesian pada Mixture of Inputs—telah diuraikan secara mendalam pada Bab II. Oleh karena itu, bab ini difokuskan pada formulasi akhir persamaan, penyatuannya ke dalam bentuk umum kombinasi konveks, serta rancangan eksperimen dan pengujian hipotesis [cite: 1, 2].

## 3.1 Formulasi Langkah Laten

LatentMAS merupakan kerangka kolaborasi multi-agen yang memungkinkan agen melakukan penalaran dan pertukaran informasi dalam ruang laten tanpa menghasilkan teks pada setiap langkah penalaran [cite: 1]. Pada setiap langkah, model menghasilkan representasi pada lapisan terakhir yang kemudian digunakan kembali sebagai masukan pada langkah berikutnya [cite: 1].

Misalkan suatu model bahasa memiliki dimensi tersembunyi sebesar $d$ [cite: 1]. Setelah model menerima suatu konteks, diperoleh *hidden state* terakhir
\[ h_t\in\mathbb{R}^{d}. \]
Vektor tersebut kemudian dipetakan menggunakan suatu fungsi transformasi $\Phi$ sehingga diperoleh representasi laten
\[ z_t=\Phi(h_t). \tag{3.1} \]
Representasi $z_t$ selanjutnya diberikan kembali kepada model sebagai *input embedding* pada langkah berikutnya [cite: 1]. Jika $f_\theta$ menyatakan fungsi Transformer dengan parameter $\theta$, maka proses pembangkitan langkah laten dapat dituliskan sebagai
\[ h_{t+1} = f_\theta \left( z_t\mid \mathrm{KV}_{1:t} \right), \tag{3.2} \]
dengan $\mathrm{KV}_{1:t}$ menyatakan KV-cache yang telah terbentuk hingga langkah ke-$t$ [cite: 1].

Proses pada Persamaan (3.1) dan (3.2) dilakukan secara berulang sebanyak $m=10$ langkah laten [cite: 1]. Berbeda dengan pembangkitan autoregresif biasa, $z_t$ langsung diberikan melalui parameter `inputs_embeds` sehingga tidak menghasilkan token diskret, melainkan representasi kontinu [cite: 1]. Fungsi $\Phi$ inilah yang menjadi objek utama yang dibandingkan dalam penelitian ini, dengan menjaga struktur agen dan prosedur komunikasi tetap [cite: 1, 2].

---

## 3.2 Persamaan Langkah Laten LatentMAS (Metode *Raw*)

Pada model bahasa, matriks embedding masukan $W_{\mathrm{in}}\in\mathbb{R}^{V\times d}$ dan matriks *language model head* keluaran $W_{\mathrm{out}}\in\mathbb{R}^{V\times d}$ tidak selalu memiliki representasi geometris yang sama [cite: 1]. Untuk menghubungkan kedua ruang tersebut, LatentMAS menggunakan matriks *realignment* $W_a$ [cite: 1].

Penurunan solusi $W_a$ melalui *ridge regression* untuk meminimalkan galat rekonstruksi telah dibahas secara rinci pada Bab II [cite: 1, 2]. Persamaan akhir matriks *realignment* tersebut adalah
\[ W_a = \left( W_{\mathrm{out}}^\top W_{\mathrm{out}} + \lambda I \right)^{-1} W_{\mathrm{out}}^\top W_{\mathrm{in}}, \tag{3.3} \]
dengan $\lambda=10^{-5}$ sebagai parameter regularisasi [cite: 1].

Setelah *hidden state* dipetakan menggunakan $W_a$, diperoleh representasi awal $\tilde z_{\mathrm{raw}} = hW_a$ [cite: 1]. Untuk menjaga skala representasi laten agar sebanding dengan embedding masukan, digunakan normalisasi magnitudo seragam $\rho = \frac{1}{V} \sum_{i=1}^{V} \|W_{\mathrm{in}}[i]\|_2$ [cite: 1, 2]. Representasi laten metode *raw* dirumuskan sebagai:
\[ \Phi_{\mathrm{raw}}(h) = \rho \frac{hW_a}{\|hW_a\|_2}. \tag{3.4} \]

Memori kerja laten yang terbentuk dari proses ini, yakni kumpulan KV-cache $\mathcal{M}_{A} = \left\{ \left( K^{(l)}_{A}, V^{(l)}_{A} \right) \right\}_{l=1}^{L}$, kemudian ditransfer ke agen berikutnya tanpa proses dekode teks [cite: 1, 2].

---

## 3.3 Bentuk Umum Keluarga Relaksasi Diskret

Pendekatan alternatif selain pemetaan linier langsung (metode *raw*) adalah memanfaatkan distribusi probabilitas token [cite: 1]. Metode-metode turunan relaksasi diskret dapat dituliskan menggunakan satu bentuk umum berdasarkan kombinasi konveks baris matriks embedding [cite: 2].

Misalkan logit untuk seluruh token dinyatakan sebagai $\ell = W_{\mathrm{out}}h$ [cite: 1, 2]. Distribusi token dasar dihitung menggunakan *temperature scaling* $T=0{,}7$:
\[ p = \operatorname{softmax}\left(\frac{\ell}{T}\right). \tag{3.5} \]

Sebuah langkah laten dikatakan berada dalam keluarga relaksasi diskret apabila dapat dituliskan sebagai kombinasi konveks [cite: 2]:
\[ \tilde z(w) = \sum_{i=1}^{V} w_iW_{\mathrm{in}}[i], \tag{3.6} \]
dengan bobot $w$ berada pada simpleks probabilitas:
\[ w \in \Delta^{V-1} = \left\{ w\in\mathbb{R}^{V} : w_i\geq0,\, \sum_{i=1}^{V}w_i=1 \right\}. \tag{3.7} \]

Representasi tersebut kemudian dinormalisasi seragam untuk memfokuskan perbandingan murni pada arah vektor [cite: 2]:
\[ \Phi_w(h) = \rho \frac{\tilde z(w)}{\|\tilde z(w)\|_2}. \tag{3.8} \]
Perbedaan antar-metode dalam keluarga ini murni terletak pada aturan pembentukan bobot $w$ [cite: 1, 2].

---

## 3.4 Varian Persamaan Langkah Laten

Berdasarkan bentuk umum pada Persamaan (3.6), penelitian ini mengevaluasi empat varian pembentukan bobot $w$ [cite: 1, 2].

### 3.4.1 Soft Thinking (Metode *Soft*)
Metode *Soft Thinking* menggunakan distribusi probabilitas secara langsung sebagai bobot [cite: 1, 2]:
\[ w_{\mathrm{soft}} = p. \tag{3.9} \]

### 3.4.2 Sampling Kategoris (Metode *Sample*)
Sebagai pembanding diskret, metode *sample* memilih satu token secara kategoris $Y \sim \operatorname{Categorical}(p)$ dan menggunakan vektor satuan $e_y$ [cite: 1, 2]:
\[ w_{\mathrm{sample}} = e_y. \tag{3.10} \]

### 3.4.3 Gumbel-Softmax (Metode *Gumbel*)
*Stochastic Soft Thinking* menggunakan derau Gumbel standar $g_i \sim \operatorname{Gumbel}(0,1)$ dan parameter temperatur $\tau=0{,}5$ untuk menghasilkan bobot stokastik [cite: 1, 2]:
\[ w_{\mathrm{gumbel}} = \operatorname{softmax}\left(\frac{\ell+g}{\tau}\right). \tag{3.11} \]
Penurunan metode ini dari *Gumbel-Max trick* serta penyesuaian perhitungan suhunya telah diuraikan pada Bab II [cite: 1, 2].

### 3.4.4 Mixture of Inputs (Metode *MoI*)
Mixture of Inputs menggabungkan distribusi token dengan token hasil sampling $Y \sim \operatorname{Categorical}(p)$ melalui estimasi *posterior mean* [cite: 1, 2]. Dengan $\beta=1$ dan entropi ternormalisasi $H \in [0,1]$, bobot MoI dirumuskan sebagai:
\[ w_{\mathrm{moi}} = \frac{Hp + (\beta+1-H)e_y}{\beta+1}. \tag{3.12} \]
Rincian inferensi Bayesian pembentuk bobot ini telah dibahas penuh pada Bab II [cite: 1, 2]. Pada implementasi ini, normalisasi entropi $H$ dihitung menggunakan $\log V$ (kosakata penuh) untuk presisi teoretis representasi ruang logit [cite: 2].

---

## 3.5 Analisis Teoretis Penyatuan Metode

### 3.5.1 Proposisi 3.1: Keanggotaan pada Simpleks Probabilitas
**Proposisi 3.1.** Metode *soft*, *sample*, *gumbel*, dan *moi* menghasilkan representasi laten yang seluruhnya merupakan kombinasi konveks dari embedding token [cite: 1, 2].
**Bukti.** Distribusi $p$ (*soft*), vektor satu-hot $e_y$ (*sample*), hasil *softmax* (*gumbel*), dan rata-rata terbobot taknegatif dari nilai yang berjumlah satu (*moi*) secara matematis merupakan anggota simpleks probabilitas $\Delta^{V-1}$. Karena itu, representasi $\tilde z$ selalu berada di dalam lambung konveks (convex hull) dari matriks $W_{\mathrm{in}}$ [cite: 1, 2]. $\square$

### 3.5.2 Proposisi 3.2: MoI sebagai Interpolasi Adaptif
**Proposisi 3.2.** MoI dapat dinyatakan sebagai kombinasi konveks antara metode *Soft Thinking* dan sampling kategoris [cite: 1, 2].
**Bukti.** Dengan mendefinisikan $\lambda = \frac{H}{\beta+1}$, Persamaan (3.12) dapat direstrukturisasi menjadi $w_{\mathrm{moi}} = \lambda w_{\mathrm{soft}} + (1-\lambda) w_{\mathrm{sample}}$ [cite: 1, 2].
Karena $H \in [0,1]$ dan $\beta=1$, maka $\lambda \leq 0{,}5$. Hal ini membuktikan bahwa MoI secara struktural berlabuh pada token diskret, dan hanya berinterpolasi adaptif ke arah representasi *soft* berdasarkan peningkatan tingkat keraguan model (entropi $H$) [cite: 2]. $\square$

### 3.5.3 Proposisi 3.3: Limit Gumbel-Softmax
**Proposisi 3.3.** Gumbel-Softmax mendekati sampling kategoris ketika temperatur $\tau$ mendekati nol [cite: 1].
**Bukti.** Sifat asimtotik fungsi *softmax* menjamin bahwa ketika $\tau \rightarrow 0^+$, komponen logit dominan yang ditambahkan derau Gumbel akan dieksponensiasi menjadi mendekati nilai 1, sehingga distribusi bobot $w(\tau)$ mendekati vektor satu-hot $e_y$ hasil *Gumbel-Max trick* [cite: 1]. $\square$

---

## 3.6 Perbedaan Metode *Raw* dan Keluarga Relaksasi Diskret

Berdasarkan penyatuan di atas, terdapat perbedaan mendasar secara geometris. Keempat metode dalam keluarga relaksasi diskret dijamin selalu menghasilkan vektor di dalam lambung konveks embedding token karena bobotnya ditahan pada ruang simpleks $\Delta^{V-1}$ [cite: 2]. Sebaliknya, metode *raw* ($\tilde z_{\mathrm{raw}} = hW_a$) memetakan *hidden state* ke ruang $\mathbb{R}^d$ tanpa kendala proyektif (unconstrained projection) [cite: 2].

Ketiadaan kendala ini menyebabkan vektor laten *raw* mendarat di luar *manifold* embedding yang dipelajari model selama proses pelatihan [cite: 2]. Disosiasi geometris inilah yang melandasi perbandingan dalam pengujian empiris sistem [cite: 2].

---

## 3.7 Rancangan Eksperimen

Eksperimen dirancang melintasi dua sumbu pengujian untuk mengisolasi efek bentuk persamaan dari efek medium komunikasi [cite: 1, 2].

**Sumbu A: Persamaan Langkah Laten.** Dievaluasi lima kondisi independen: *raw*, *soft*, *sample*, *gumbel*, dan *moi* [cite: 1, 2].
**Sumbu B: Medium Komunikasi Antar-Agen.** Dievaluasi tiga formasi medium untuk mentransfer informasi:
1. `text`: Keluaran teks murni (tanpa langkah laten) [cite: 1, 2].
2. `kv`: Pertukaran memori kerja murni (KV-cache) dengan langkah laten aktif [cite: 1, 2].
3. `kv_and_text`: Pertukaran gabungan KV-cache dan teks [cite: 1, 2].

*Aturan Validitas Statistik:* Pada medium `text`, persamaan $\Phi$ tidak tereksekusi karena tiadanya siklus penalaran laten. Oleh karena itu, pengujian `text` hanya dijalankan **satu kali** sebagai *baseline* kontrol (bukan disalin ulang untuk setiap varian Sumbu A) guna menjaga keabsahan asumsi independensi observasi pada pengujian statistik berpasangan [cite: 2]. Seluruh kondisi beroperasi pada topologi agen sekuensial yang konsisten: $\mathrm{Planner} \rightarrow \mathrm{Critic} \rightarrow \mathrm{Refiner} \rightarrow \mathrm{Judger}$ [cite: 1].

---

## 3.8 Hipotesis Penelitian

Rancangan eksperimen secara khusus ditujukan untuk menguji tiga rumusan hipotesis yang dapat dipatahkan (falsifiable) secara terukur [cite: 2]:

1. **H1 (Struktur Keluarga Relaksasi):** Keluarga relaksasi diskret (*soft*, *sample*, *gumbel*, *moi*) akan mengungguli performa metode *raw* secara universal karena mampu menahan representasi laten di dalam batasan lambung konveks embedding token asli [cite: 2].
2. **H2 (Disosiasi Simbolik vs Semantik):** Penurunan performa pada metode *raw* berkorelasi secara langsung dengan intensitas tuntutan presisi simbolik tugas yang diuji. Urutan keparahan kejatuhan akurasi diramalkan sebagai: $\text{GSM8K} < \text{ARC-C} < \text{HumanEval+} < \text{Tugas Ekspresi Simbolik (QuantaAlpha)}$ [cite: 2].
3. **H3 (Efisiensi Medium):** Modus komunikasi medium `kv` mampu mereduksi jumlah penggunaan token keluaran secara masif tanpa memicu kerugian kompresi informasi, dengan syarat persamaan $\Phi$ yang dipilih termasuk dalam keluarga relaksasi diskret [cite: 2].

---

## 3.9 Model dan Parameter Eksperimen

Model dasar (backbone) yang mendasari penelitian ini adalah arsitektur **Qwen3-8B** [cite: 1, 2]. Spesifikasi model ini dipilih karena pemisahan struktural matriks masukan dan keluarannya (untied embeddings), yang menghasilkan matriks *realignment* $W_a$ dengan transformasi aktual yang berjarak dari matriks identitas ($M \not\approx I$) [cite: 1, 2].

| Parameter | Nilai | Keterangan |
| :--- | :---: | :--- |
| Model Dasar | Qwen3-8B | Untied embeddings [cite: 1, 2] |
| Jumlah langkah laten ($m$) | 10 | Standar per agen [cite: 1] |
| Suhu distribusi ($T$) | 0,7 | Penyesuaian suhu logit dasar [cite: 1, 2] |
| Regularisasi ridge ($\lambda$) | $10^{-5}$ | Kondisi batas optimasi matriks $W_a$ [cite: 1] |
| Suhu Gumbel ($\tau$) | 0,5 | Pengendali kelonggaran distribusi stokastik [cite: 1] |
| Parameter MoI ($\beta$) | 1 | Penyeimbang laju prior Dirichlet [cite: 1, 2] |
| Magnitudo Normalisasi ($\rho$) | $\frac{1}{V}\sum \|W_{\mathrm{in}}[i]\|_2$ | Diaplikasikan seragam pada kelima persamaan laten [cite: 1, 2] |

---

## 3.10 Data dan Tugas Evaluasi

Evaluasi ketahanan sistem diukur menggunakan empat domain tugas analitis untuk meninjau gradasi kebutuhan presisi struktur, yang akan memverifikasi Hipotesis H2 [cite: 1, 2]:
1. **GSM8K:** Penalaran hitungan aritmetika dasar dengan toleransi struktur logika fleksibel [cite: 1, 2].
2. **ARC-Challenge:** Pemahaman relasi logis teks dan sains dalam penyelesaian soal pilihan ganda [cite: 1, 2].
3. **HumanEval-Plus:** Generasi kode logika pemrograman Python (*full unit-test pass rate*) [cite: 1, 2].
4. **Faktor Simbolik (QuantaAlpha):** Generasi perumusan matematika DSL (*Domain Specific Language*) untuk kerangka *alpha mining*, merepresentasikan beban uji murni simbolik tinggi [cite: 1, 2].

---

## 3.11 Prosedur Eksperimen

Tahapan eksperimen dijalankan melintasi pipa evaluasi dengan regulasi ketat terhadap desain *paired-sample* [cite: 1, 2]:
1. Inisialisasi model Qwen3-8B serta pengekstraksian matriks $W_{\mathrm{in}}$ dan $W_{\mathrm{out}}$ [cite: 1].
2. Perhitungan matematis pembentukan matriks *realignment* $W_a$ [cite: 1].
3. Penarikan instansiasi data *subsampel* yang dikunci secara deterministik menggunakan *seed* identik [cite: 1, 2]. Integritas kesamaan soal divalidasi penuh melalui pencocokan sidik jari data [cite: 2].
4. Pelaksanaan iterasi eksekusi sistem agen untuk setiap kondisi medium dan varian metode $\Phi$ [cite: 1].
5. Pembangkitan siklus $m=10$ langkah laten di lingkungan internal agen (kecuali medium `text`) [cite: 1].
6. Transmisi muatan informasi KV-Cache antar agen yang bersangkutan sesuai peruntukan medium [cite: 1].
7. Penerjemahan teks hasil agregasi akhir (*Judger*) untuk pengukuran performa [cite: 1].

---

## 3.12 Metrik Evaluasi

Struktur analitis dievaluasi dalam kompartemen berlapis untuk mengantisipasi ambiguitas sumber perbaikan/penurunan mutu generatif [cite: 2]:
1. **Accuracy (Exact Match / Pass@1):** Verifikasi objektivitas kelulusan numerik, kategoris, atau kelulusan instrumen verifikasi *unit test* [cite: 1, 2].
2. **Format Rate:** Konfirmasi laju keabsahan struktural dan *parsing* format atas hasil keluaran model [cite: 1, 2].
3. **Symbolic Fidelity:** Pemetaan laju korupsi token (seperti insersi aksara intrusif lintas batas bahasa/CJK), ditujukan eksklusif untuk perumusan kode dan matematika [cite: 1, 2].
4. **Token Usage:** Perbandingan agregat token yang dihasilkan sebagai indikator kapasitas dan reduksi operasional [cite: 1, 2].
5. **Inference Time:** Durasi aktual resolusi tunggal dari tahap awal masukan ke tahap penyerahan jawaban akhir [cite: 1].

---

## 3.13 Pengujian Statistik

Perbandingan performa divalidasi melintasi instansiasi berpasangan dengan kerangka uji statistik rigor [cite: 1, 2]:
1. **Uji McNemar Eksak:** Menguji nilai kebermaknaan parameter tipe klasifikasi biner, seperti uji status benar/salah (*Accuracy*) dan *Format Rate* [cite: 1, 2].
2. **Uji Wilcoxon Signed-Rank:** Menguji perbandingan non-parametrik berskala interval atau ordinal seperti kuantitas token kompresi dan selisih temporal inferensi [cite: 1, 2].
3. **Interval Kepercayaan (CI) Bootstrap 95%:** Diaplikasikan guna menggambarkan proyeksi ketidakpastian secara presisi pada ukuran perbedaan efek perlakuan [cite: 1, 2].
4. **Koreksi Multiplisitas (Holm / Benjamini-Hochberg):** Modifikasi atas laju uji signifikansi $p$-value untuk meminimalisasi pembengkakan Kesalahan Tipe I akibat agregasi pengujian puluhan kombinasi hipotesis (*false discoveries*) [cite: 1, 2]. Deviasi margin yang gagal mencapai signifikansi diposisikan secara jelas sebagai kesetaraan, bukan sekadar nilai tren [cite: 2].

---

## 3.14 Alur Penelitian

Urutan tahapan keseluruhan kerangka penyusunan dan pengujian metode dirumuskan sebagai berikut [cite: 1]:
\[ \boxed{\text{Persiapan Model \& Data}} \rightarrow \boxed{\text{Perhitungan }W_a} \rightarrow \boxed{\text{Formulasi } \Phi \text{ (5 Varian)}} \rightarrow \boxed{\text{Penalaran Laten}} \rightarrow \boxed{\text{Transfer KV-Cache}} \rightarrow \boxed{\text{Evaluasi Berlapis}} \rightarrow \boxed{\text{Analisis Statistik}}. \tag{3.13} \]
