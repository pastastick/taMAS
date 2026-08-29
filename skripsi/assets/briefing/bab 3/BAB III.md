# BAB III  
# METODOLOGI PENELITIAN

## 3.1 Desain Penelitian

Penelitian ini menggunakan pendekatan eksperimen kuantitatif untuk menguji pengaruh formulasi persamaan langkah laten terhadap kinerja kolaborasi multi-agen berbasis *large language model* (LLM). Fokus penelitian tidak diletakkan pada perubahan arsitektur multi-agen, perancangan *prompt*, maupun perubahan mekanisme komunikasi antar-agen secara umum, melainkan pada fungsi pemetaan yang digunakan untuk mengubah *hidden state* keluaran model menjadi representasi yang dapat digunakan kembali sebagai masukan pada langkah laten berikutnya.

Objek utama penelitian dinyatakan sebagai fungsi

\[
\Phi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d},
\]

dengan \(d\) menyatakan dimensi *hidden state* model. Fungsi tersebut menerima *hidden state* terakhir pada suatu langkah laten dan menghasilkan vektor yang diberikan kepada model melalui parameter `inputs_embeds`. Dengan demikian, penelitian mengisolasi persamaan langkah laten sebagai variabel utama yang dibandingkan, sementara komponen lain dari sistem dijaga tetap.

Rancangan penelitian dibangun berdasarkan dua sumbu eksperimen. Sumbu pertama mengatur **cara pembentukan representasi laten**, sedangkan sumbu kedua mengatur **medium komunikasi antar-agen**. Pemisahan kedua sumbu tersebut diperlukan agar pengaruh formulasi \(\Phi\) dapat dibedakan dari pengaruh penggunaan komunikasi laten melalui KV-cache.

Secara umum, alur penelitian terdiri atas:

1. menentukan *backbone* LLM yang digunakan;
2. membentuk persamaan langkah laten untuk setiap metode yang dibandingkan;
3. menjalankan rangkaian agen dengan medium komunikasi yang telah ditentukan;
4. mengevaluasi keluaran pada beberapa jenis tugas dengan tuntutan presisi simbolik yang berbeda;
5. mengukur akurasi, keandalan keluaran, fidelitas simbolik, dan biaya komputasi;
6. melakukan pengujian statistik berpasangan terhadap hasil eksperimen; dan
7. menarik kesimpulan berdasarkan hipotesis yang telah ditentukan sebelum evaluasi utama.

Rancangan ini tidak ditujukan untuk menunjukkan bahwa satu metode tertentu selalu lebih baik daripada metode lainnya. Sebaliknya, pengujian dirancang untuk menentukan apakah terdapat perbedaan yang secara sistematis dapat dikaitkan dengan **keanggotaan representasi laten pada lambung konveks ruang embedding**, sebagaimana dirumuskan pada bagian berikut.

---

## 3.2 Objek Penelitian dan Formulasi Langkah Laten

### 3.2.1 Proses Pembangkitan Langkah Laten

Pada suatu langkah laten, model telah memproses konteks dan menghasilkan *hidden state* terakhir

\[
h_t\in\mathbb{R}^{d}.
\]

Berbeda dengan pembangkitan token autoregresif biasa yang langsung memilih sebuah token dari distribusi probabilitas, penelitian ini membentuk suatu representasi laten

\[
z_t=\Phi(h_t),
\]

kemudian menggunakan \(z_t\) sebagai *input embedding* untuk *forward pass* berikutnya:

\[
h_{t+1}
=
f_\theta
\left(
z_t\mid \mathrm{KV}_{1:t}
\right).
\]

Proses tersebut diulangi sebanyak \(m\) langkah. Dalam penelitian ini digunakan

\[
m=10.
\]

Representasi yang dihasilkan pada setiap langkah tidak diterjemahkan menjadi token teks. Sebaliknya, representasi tersebut langsung digunakan sebagai masukan kontinu pada langkah berikutnya. Dengan demikian, satu langkah laten tidak identik dengan satu token teks.

Mekanisme ini mengikuti prinsip dasar LatentMAS, yaitu melakukan *latent thought generation* menggunakan representasi lapisan terakhir dan menyimpan hasil pemrosesan tersebut dalam KV-cache yang dapat diteruskan kepada agen berikutnya. LatentMAS menggunakan KV-cache sebagai memori kerja yang menyimpan representasi *key* dan *value* pada setiap lapisan Transformer.

Dalam implementasi penelitian, setiap *latent step* terdiri atas empat operasi utama, yaitu:

1. mengambil *last hidden state* \(h_t\);
2. menerapkan fungsi \(\Phi\) untuk menghasilkan \(z_t\);
3. memasukkan \(z_t\) sebagai *virtual token* melalui `inputs_embeds`; dan
4. menjalankan *forward pass* untuk memperoleh \(h_{t+1}\).

Dengan demikian, fungsi \(\Phi\) merupakan titik yang membedakan metode-metode yang dibandingkan dalam penelitian ini.

---

## 3.3 Persamaan Langkah Laten Resmi LatentMAS

### 3.3.1 Input–Output Alignment

LatentMAS menggunakan pemetaan linier untuk mengubah *hidden state* keluaran model ke ruang embedding masukan. Misalkan

\[
W_{\mathrm{in}}\in\mathbb{R}^{V\times d}
\]

merupakan matriks embedding masukan dan

\[
W_{\mathrm{out}}\in\mathbb{R}^{V\times d}
\]

merupakan matriks pada *language-model head*, dengan \(V\) merupakan ukuran kosakata.

Matriks pemetaan \(M\in\mathbb{R}^{d\times d}\) diperoleh melalui permasalahan *ridge regression*

\[
M
=
\arg\min_M
\left\{
\left\|
W_{\mathrm{out}}M-W_{\mathrm{in}}
\right\|_F^2
+
\lambda\|M\|_F^2
\right\},
\]

dengan \(\lambda=10^{-5}\). Bentuk tertutupnya adalah

\[
M=
\left(
W_{\mathrm{out}}^\top W_{\mathrm{out}}
+
\lambda I
\right)^{-1}
W_{\mathrm{out}}^\top W_{\mathrm{in}}.
\]

Persamaan tersebut merupakan formulasi *realignment matrix* yang digunakan LatentMAS untuk menghubungkan ruang keluaran dan ruang masukan model.

Hasil pemetaan kemudian dinormalisasi terhadap magnitudo rata-rata embedding masukan. Dengan

\[
\rho=
\frac{1}{V}
\sum_{i=1}^{V}
\left\|
W_{\mathrm{in}}[i]
\right\|,
\]

fungsi langkah laten yang digunakan sebagai *baseline* dirumuskan sebagai

\[
\Phi_{\mathrm{raw}}(h)
=
\rho
\frac{hM}{\|hM\|}.
\]

Pada penelitian ini formulasi tersebut disebut sebagai metode **raw**.

### 3.3.2 Batas Jaminan Persamaan Raw

Persamaan *ridge* tersebut mengoptimalkan kedekatan

\[
W_{\mathrm{out}}M\approx W_{\mathrm{in}}.
\]

Namun, objektif tersebut secara langsung mengoptimalkan pemetaan pada baris-baris \(W_{\mathrm{out}}\). Sementara itu, argumen aktual dari \(\Phi_{\mathrm{raw}}\) merupakan *hidden state* yang dihasilkan oleh aliran komputasi Transformer.

Oleh karena itu, terdapat perbedaan antara:

1. kemampuan \(M\) merekonstruksi embedding masukan dari baris \(W_{\mathrm{out}}\); dan
2. kemampuan \(M\) menghasilkan representasi yang tetap berada pada wilayah representasi masukan ketika diberikan *hidden state* aktual sebagai input.

Perbedaan tersebut menjadi dasar pengujian geometris penelitian. Pada Qwen3-8B, pengukuran awal menunjukkan

\[
\frac{\|M-I\|_F}{\|I\|_F}=1{,}04,
\]

sedangkan rata-rata kemiripan kosinus antara \(h\) dan \(hM\) adalah sekitar \(0{,}011\). Selain itu, nilai rata-rata kemiripan kosinus antara \(W_{\mathrm{in}}[i]\) dan \(W_{\mathrm{out}}[i]\) adalah sekitar \(0{,}0041\). Pengukuran terhadap kedekatan keluaran \(\Phi_{\mathrm{raw}}\) dengan embedding token menghasilkan rata-rata maksimum kemiripan kosinus sekitar \(0{,}31\).

Hasil tersebut tidak digunakan sebagai bukti bahwa KV-cache kehilangan informasi. Sebaliknya, interpretasi metodologis penelitian ini adalah bahwa potensi kehilangan informasi perlu dipisahkan antara **mekanisme pembentukan representasi laten** dan **mekanisme pengiriman KV-cache**. Kerangka penelitian secara eksplisit menempatkan \(\Phi\), bukan KV-cache, sebagai komponen yang diuji.

---

## 3.4 Reformulasi Langkah Laten sebagai Relaksasi Diskret

### 3.4.1 Bentuk Umum

Untuk memperoleh perbandingan yang terkendali antara beberapa metode, penelitian ini merumuskan langkah laten sebagai pembentukan distribusi bobot pada kosakata.

Dari *hidden state* \(h\), diperoleh logit

\[
\ell=W_{\mathrm{out}}h,
\]

kemudian distribusi probabilitas token dihitung sebagai

\[
p
=
\operatorname{softmax}
\left(
\frac{\ell}{T}
\right),
\]

dengan suhu

\[
T=0{,}7.
\]

Selanjutnya didefinisikan simpleks probabilitas

\[
\Delta^{V-1}
=
\left\{
w\in\mathbb{R}^{V}
\mid
w_i\geq0,\;
\sum_{i=1}^{V}w_i=1
\right\}.
\]

Sebuah metode langkah laten dimasukkan ke dalam keluarga relaksasi diskret apabila representasinya dapat ditulis sebagai

\[
\tilde z
=
\sum_{i=1}^{V}
w_iW_{\mathrm{in}}[i],
\qquad
w\in\Delta^{V-1}.
\]

Setelah pembentukan \(\tilde z\), seluruh metode dinormalisasi menggunakan konvensi yang sama:

\[
\Phi(h)
=
\rho
\frac{\tilde z}{\|\tilde z\|}.
\]

Dengan formulasi tersebut, perbedaan antar-metode tidak lagi terutama terletak pada ruang representasi yang digunakan, melainkan pada **aturan pembentukan bobot \(w\)**.

---

## 3.5 Metode yang Dibandingkan

Empat metode dalam keluarga relaksasi diskret digunakan bersama satu metode *baseline* di luar keluarga tersebut.

### 3.5.1 Soft

Pada metode *soft*, bobot langsung menggunakan distribusi probabilitas model:

\[
w_{\mathrm{soft}}=p.
\]

Representasi laten menjadi

\[
\tilde z_{\mathrm{soft}}
=
\sum_{i=1}^{V}
p_iW_{\mathrm{in}}[i].
\]

Formulasi ini mengikuti definisi *Soft Thinking*, yaitu menggunakan distribusi probabilitas token sebagai *soft token* dan menghasilkan *soft input* melalui kombinasi berbobot embedding token. Karena seluruh bobot tidak negatif dan berjumlah satu, representasi tersebut merupakan kombinasi konveks embedding token.

### 3.5.2 Sample

Metode *sample* digunakan sebagai kontrol diskret. Token

\[
y\sim\operatorname{Cat}(p)
\]

dipilih dari distribusi \(p\), kemudian bobot direpresentasikan menggunakan vektor satuan

\[
w_{\mathrm{sample}}=e_y.
\]

Dengan demikian,

\[
\tilde z_{\mathrm{sample}}
=
W_{\mathrm{in}}[y].
\]

Metode ini merepresentasikan pembentukan input sebagaimana proses autoregresif diskret, yaitu memilih satu token dan menggunakan embedding token tersebut sebagai masukan berikutnya.

### 3.5.3 Gumbel

Metode *gumbel* menggunakan derau Gumbel independen

\[
g_i\overset{\mathrm{iid}}{\sim}
\operatorname{Gumbel}(0,1)
\]

dan membentuk

\[
w_{\mathrm{gumbel}}
=
\operatorname{softmax}
\left(
\frac{\ell+g}{\tau}
\right).
\]

Formulasi tersebut berasal dari Stochastic Soft Thinking yang menggunakan Gumbel-Softmax untuk memasukkan stokastisitas ke dalam proses *soft reasoning*. Paper tersebut menunjukkan bahwa pendekatan *soft thinking* deterministik dapat didominasi oleh sinyal token dengan probabilitas tertinggi dan mengusulkan stokastisitas untuk mengurangi perilaku *greedy*. 
Dalam konteks penelitian ini, Gumbel tidak diperkenalkan sebagai metode baru. Kontribusinya adalah menempatkan mekanisme tersebut sebagai salah satu aturan pembentukan \(w\) dalam kerangka langkah laten multi-agen yang sama.

### 3.5.4 Mixture of Inputs

Metode *Mixture of Inputs* (MoI) menggabungkan informasi distribusi \(p\) dengan token yang diperoleh melalui proses sampling.

Misalkan

\[
y\sim\operatorname{Cat}(p)
\]

dan \(H\) merupakan entropi ternormalisasi:

\[
H
=
-\frac{
\sum_{i=1}^{V}p_i\log p_i
}{
\log V
}.
\]

Dengan parameter \(\beta\), bobot MoI dirumuskan sebagai

\[
w_{\mathrm{moi}}
=
\frac{
Hp+(\beta+1-H)e_y
}{
\beta+1
}.
\]

Pada penelitian ini digunakan

\[
\beta=1.
\]

MoI pada sumber aslinya dibangun dengan interpretasi Bayesian, yaitu distribusi keluaran dipandang sebagai prior sedangkan token hasil sampling dipandang sebagai observasi. Ekspektasi posterior kemudian digunakan untuk membentuk embedding campuran.

---

## 3.6 Proposisi Matematis untuk Penyatuan Metode

### 3.6.1 Proposisi 1: MoI sebagai Kombinasi Konveks Soft dan Sample

Bobot MoI dapat ditulis ulang sebagai

\[
w_{\mathrm{moi}}
=
\frac{H}{\beta+1}p
+
\frac{\beta+1-H}{\beta+1}e_y.
\]

Dengan mendefinisikan

\[
\alpha=\frac{H}{\beta+1},
\]

diperoleh

\[
w_{\mathrm{moi}}
=
\alpha w_{\mathrm{soft}}
+
(1-\alpha)w_{\mathrm{sample}}.
\]

Karena

\[
0\leq H\leq1
\]

dan \(\beta\geq0\), maka

\[
0\leq\alpha\leq1.
\]

Selanjutnya, karena \(w_{\mathrm{soft}}\) dan \(w_{\mathrm{sample}}\) merupakan anggota \(\Delta^{V-1}\), serta simpleks probabilitas bersifat konveks, maka

\[
w_{\mathrm{moi}}\in\Delta^{V-1}.
\]

Dengan demikian, MoI merupakan kombinasi konveks antara representasi *soft* dan representasi *sample*.

Pada \(\beta=1\),

\[
\alpha=\frac{H}{2},
\]

sehingga kontribusi distribusi \(p\) tidak pernah melebihi satu setengah dari total bobot. Hal ini menunjukkan bahwa MoI tetap lebih dekat secara struktural kepada token hasil sampling dibandingkan kepada *soft distribution* penuh.

### 3.6.2 Proposisi 2: Hubungan Gumbel dengan Sampling Diskret

Gumbel-Max Trick menyatakan bahwa

\[
\arg\max_i(\ell_i+g_i)
\]

menghasilkan sampel dari distribusi kategorikal yang berasal dari *softmax* logit. Oleh sebab itu, ketika temperatur Gumbel-Softmax mendekati nol,

\[
\tau\rightarrow0,
\]

maka

\[
w_{\mathrm{gumbel}}
\rightarrow e_y,
\]

dengan

\[
y\sim\operatorname{Cat}(\operatorname{softmax}(\ell)).
\]

Dengan demikian, *gumbel* memiliki hubungan kontinu terhadap *sample*. Pada temperatur yang semakin tinggi, distribusi menjadi semakin halus.

Proposisi ini menunjukkan bahwa *soft*, *sample*, dan *gumbel* tidak perlu dipandang sebagai tiga mekanisme yang sepenuhnya terpisah. Ketiganya dapat ditempatkan dalam ruang yang sama berdasarkan bentuk bobot \(w\).

### 3.6.3 Proposisi 3: Raw Berada di Luar Keluarga Relaksasi Diskret

Untuk metode *raw*, representasi sebelum normalisasi adalah

\[
\tilde z_{\mathrm{raw}}=hM.
\]

Berbeda dengan metode yang menggunakan \(w\in\Delta^{V-1}\), tidak terdapat kendala bahwa koefisien representasi terhadap embedding token harus tidak negatif dan berjumlah satu.

Dengan demikian,

\[
\tilde z_{\mathrm{raw}}
\not\equiv
\sum_iw_iW_{\mathrm{in}}[i],
\qquad
w\in\Delta^{V-1},
\]

dalam arti bahwa fungsi objektif pembentukan \(M\) tidak memaksakan representasi tersebut sebagai kombinasi konveks embedding token.

Perbedaan mendasar antara *raw* dan empat metode lainnya bukan terletak pada linearitas, karena keduanya dapat melibatkan operasi linear. Perbedaannya terletak pada **kendala konveksitas**. Metode *soft*, *sample*, *gumbel*, dan *moi* menghasilkan titik pada lambung konveks embedding masukan, sedangkan *raw* tidak memiliki kendala tersebut.

---

## 3.7 Konvensi Normalisasi

Agar perbandingan antar-metode tidak dipengaruhi oleh perbedaan magnitudo vektor, seluruh representasi laten dinormalisasi menggunakan target magnitudo yang sama:

\[
\Phi(h)
=
\rho
\frac{\tilde z}{\|\tilde z\|}.
\]

Dengan demikian, perbedaan eksperimen diarahkan pada **arah representasi laten**, bukan panjang vektornya.

Secara geometris, seluruh metode dibandingkan pada bola berjari-jari \(\rho\). Pada empat metode keluarga relaksasi diskret, arah vektor dibatasi oleh kombinasi konveks embedding token, sedangkan metode *raw* tidak memiliki batas konveks tersebut.

Konvensi ini merupakan keputusan *experimental harness* penelitian, bukan karakteristik asli seluruh paper sumber. Oleh karena itu, interpretasi hasil akan dibatasi pada kondisi normalisasi yang telah ditetapkan dalam eksperimen.

---

## 3.8 Implementasi MoI dan Perbedaan dengan Kode Rujukan

MoI dalam penelitian ini diimplementasikan berdasarkan persamaan matematis metode sebagaimana didefinisikan pada paper sumber. Namun, implementasi penelitian tidak diklaim sebagai reproduksi bit-identik terhadap kode rujukan MoI.

Perbedaan utama terdapat pada normalisasi entropi. Pada penelitian ini, entropi menggunakan ukuran kosakata penuh,

\[
\log V,
\]

karena distribusi logit diperoleh langsung dari *output embedding* model. Sementara itu, kode rujukan MoI menggunakan distribusi terbatas yang tersedia melalui *top-20 log probabilities*, sehingga normalisasinya menggunakan \(\log20\). Dengan demikian, persamaan yang digunakan tetap mengikuti definisi matematis MoI, tetapi detail implementasinya berbeda.

Perbedaan tersebut dicatat secara eksplisit untuk mencegah interpretasi yang berlebihan terhadap hasil. Penelitian dapat menyatakan bahwa **rumus MoI diimplementasikan sesuai definisi matematis paper**, tetapi tidak menyatakan bahwa kode yang digunakan identik dengan implementasi referensi.

---

## 3.9 Rancangan Komunikasi Multi-Agen

### 3.9.1 Struktur Agen

Sistem multi-agen menggunakan rantai sekuensial dengan empat peran:

\[
\text{Planner}
\rightarrow
\text{Critic}
\rightarrow
\text{Refiner}
\rightarrow
\text{Judger}.
\]

Struktur tersebut dipertahankan sama pada setiap kondisi eksperimen. Perubahan hanya dilakukan pada representasi yang digunakan untuk meneruskan informasi antar-agen.

Pendekatan ini mengikuti prinsip LatentMAS yang menggunakan struktur *sequential multi-agent system* untuk menguji pertukaran informasi antar-agen. Dalam LatentMAS, komunikasi laten dilakukan dengan meneruskan KV-cache dari agen sebelumnya kepada agen berikutnya.

### 3.9.2 Sumbu Medium Komunikasi

Eksperimen menggunakan tiga medium utama dan satu kontrol:

| Medium | Informasi yang diteruskan | Langkah laten | Fungsi utama |
|---|---|---:|---|
| `text` | teks keluaran agen sebelumnya | tidak ada | kontrol komunikasi tekstual |
| `kv_and_text` | KV-cache dan teks | ada | komunikasi gabungan |
| `kv` | KV-cache | ada | komunikasi laten |
| `baseline` | tidak ada agen perantara | tidak ada | kontrol satu agen |

Pada medium `text`, agen menghasilkan teks dan teks tersebut digunakan sebagai masukan agen berikutnya. Tidak terdapat proses langkah laten sehingga \(\Phi\) tidak digunakan.

Hal ini menghasilkan konsekuensi metodologis penting. Karena Sumbu A tidak aktif pada medium `text`, empat varian \(\Phi\) tidak menghasilkan kondisi eksperimen yang berbeda. Oleh karena itu, kondisi `text` dijalankan **satu kali**, bukan empat kali. Pengulangan hasil yang identik sebagai empat observasi akan menciptakan pseudo-replikasi dan menyebabkan analisis statistik memperlakukan satu kondisi sebagai empat pengamatan independen.

Pada medium `kv`, agen perantara tidak menghasilkan keluaran teks untuk diteruskan. Informasi diteruskan melalui KV-cache yang memuat representasi konteks dan langkah laten. LatentMAS mendefinisikan memori kerja agen sebagai kumpulan KV-cache pada seluruh lapisan Transformer dan meneruskannya dengan menggabungkan cache agen sebelumnya ke cache agen berikutnya.

---

## 3.10 Model dan Lingkungan Eksperimen

Eksperimen utama menggunakan **Qwen3-8B** sebagai *backbone* LLM. Pemilihan model ini didasarkan pada kebutuhan penelitian untuk menguji fungsi realignment pada kondisi ketika \(W_{\mathrm{in}}\) dan \(W_{\mathrm{out}}\) tidak *tied*. Pada Qwen3-4B, kedua matriks tersebut berbagi bobot sehingga matriks realignment mendekati identitas dan formulasi *raw* menjadi hampir tidak memberikan perubahan. Oleh sebab itu, Qwen3-8B dipilih sebagai *backbone* penelitian.

Penggunaan satu *backbone* dilakukan secara sengaja. Tujuannya adalah mengisolasi pengaruh persamaan langkah laten tanpa memasukkan variasi tambahan akibat perbedaan keluarga atau ukuran model.

Dengan demikian, penelitian ini **tidak** dirancang untuk menguji generalisasi metode lintas ukuran maupun lintas keluarga LLM.

---

## 3.11 Data dan Tugas Evaluasi

Evaluasi dirancang untuk mencakup tugas dengan tuntutan presisi simbolik yang berbeda. Pemilihan tugas tidak hanya bertujuan mengukur kemampuan penalaran secara umum, tetapi juga menguji apakah efek formulasi \(\Phi\) berubah ketika representasi harus mempertahankan informasi simbolik secara lebih presisi.

### 3.11.1 GSM8K

GSM8K digunakan sebagai tugas penalaran matematika dengan jawaban numerik. Dataset tersebut berisi persoalan matematika berbentuk soal cerita yang membutuhkan penalaran multi-langkah.

Karakteristik utama GSM8K dalam penelitian ini adalah keluaran akhir relatif pendek dan dapat dievaluasi melalui kesetaraan numerik.

### 3.11.2 ARC-Challenge

ARC-Challenge digunakan untuk menguji pertanyaan pilihan ganda yang membutuhkan penalaran dan pemilihan jawaban yang tepat. Evaluasi dilakukan berdasarkan kecocokan pilihan jawaban.

### 3.11.3 HumanEval-Plus

HumanEval-Plus digunakan sebagai tugas generasi kode. Berbeda dari jawaban numerik atau pilihan ganda, keluaran kode harus memenuhi aturan sintaksis dan perilaku fungsional yang diuji melalui *unit test*. Oleh karena itu, kesalahan pada satu komponen simbolik tertentu dapat menyebabkan keseluruhan keluaran gagal memenuhi pengujian.

LatentMAS sendiri mengevaluasi tugas kode dengan menjalankan kode yang dihasilkan terhadap *unit tests* dalam lingkungan terisolasi. Sampel dinyatakan benar apabila seluruh pengujian berhasil tanpa *runtime error*.

### 3.11.4 Generasi Faktor Simbolik

Sebagai domain tambahan, penelitian menggunakan tugas generasi ekspresi faktor simbolik yang terinspirasi dari alur *alpha mining*. QuantaAlpha memformulasikan *alpha mining* sebagai pencarian fungsi faktor \(f\) dari ruang ekspresi yang menghasilkan sinyal prediktif terhadap *future return*.

Dalam konteks penelitian ini, domain tersebut digunakan sebagai pengujian eksternal terhadap kemampuan kanal mempertahankan muatan simbolik. Ekspresi faktor dibangun menggunakan domain-specific language (DSL), sehingga kesalahan pada operator, argumen, urutan fungsi, atau struktur ekspresi dapat menyebabkan ekspresi tidak dapat dievaluasi.

Domain ini tidak digunakan untuk menyatakan bahwa penelitian mengembangkan kembali algoritma QuantaAlpha. QuantaAlpha digunakan sebagai sumber konteks domain dan sebagai motivasi penggunaan ekspresi faktor simbolik; fokus penelitian tetap pada persamaan langkah laten dan komunikasi multi-agen.

---

## 3.12 Protokol Pengambilan Sampel dan Pairing

Seluruh perbandingan dirancang secara berpasangan. Setiap kondisi eksperimen menghadapi himpunan soal yang sama sehingga perbedaan performa dapat dibandingkan pada tingkat item, bukan hanya pada tingkat agregat dataset.

Untuk setiap tugas digunakan *subsample* dengan `sample_seed` yang sama pada seluruh kondisi yang dibandingkan. Kesamaan tersebut diverifikasi melalui *fingerprint* data, bukan hanya diasumsikan dari penggunaan *seed* yang sama.

Apabila *fingerprint* antar-kondisi tidak sesuai, pasangan tersebut dikeluarkan dari analisis perbandingan.

Rancangan berpasangan memiliki dua keuntungan utama. Pertama, variasi tingkat kesulitan soal dikontrol karena kondisi yang dibandingkan memperoleh soal yang sama. Kedua, analisis statistik dapat memanfaatkan informasi mengenai apakah perubahan performa terjadi pada item yang sama.

---

## 3.13 Hipotesis Penelitian

### 3.13.1 Hipotesis H1: Pengaruh Keanggotaan Keluarga Relaksasi Diskret

Hipotesis pertama menyatakan bahwa performa langkah laten lebih berkaitan dengan apakah representasi yang dihasilkan berada dalam keluarga

\[
\tilde z
=
\sum_iw_iW_{\mathrm{in}}[i],
\qquad
w\in\Delta^{V-1},
\]

daripada dengan pilihan spesifik aturan pembentukan \(w\).

Dengan demikian, dirumuskan:

> **H1:** Metode yang menghasilkan representasi pada lambung konveks embedding memiliki karakteristik performa yang berbeda dari metode *raw* yang tidak dibatasi oleh lambung konveks.

Prediksi yang telah ditentukan adalah bahwa *raw* akan tertinggal dibandingkan anggota keluarga, sedangkan perbedaan antaranggota keluarga relatif kecil dan tidak signifikan secara statistik.

Hipotesis ini secara sengaja bukan hipotesis bahwa salah satu dari *soft*, *sample*, *gumbel*, atau *moi* merupakan metode terbaik.

### 3.13.2 Hipotesis H2: Presisi Simbolik

Hipotesis kedua menyatakan bahwa pengaruh representasi yang berada di luar wilayah embedding token akan semakin terlihat ketika tugas semakin bergantung pada presisi simbolik.

Urutan prediksi yang ditetapkan sebelum evaluasi utama adalah

\[
\mathrm{GSM8K}
<
\mathrm{ARC\text{-}Challenge}
<
\mathrm{HumanEval+}.
\]

GSM8K digunakan sebagai tugas dengan jawaban numerik pendek, ARC-Challenge menggunakan simbol pilihan jawaban, sedangkan HumanEval-Plus menuntut keluaran kode yang harus memenuhi pengujian eksekusi.

Hipotesis ini memisahkan dua konsep yang sering digabungkan dalam evaluasi LLM, yaitu kemampuan mempertahankan **gagasan umum (*gist*)** dan kemampuan mempertahankan **identitas simbol secara tepat**.

### 3.13.3 Hipotesis H3: Medium Komunikasi

Hipotesis ketiga menyatakan bahwa komunikasi melalui KV-cache dapat mengurangi token keluaran secara substansial tanpa kehilangan akurasi yang berarti apabila formulasi \(\Phi\) berada dalam keluarga relaksasi diskret.

> **H3:** Medium `kv` memberikan pengurangan biaya token dibandingkan medium `text`, sementara akurasi tetap berada pada tingkat yang sebanding selama persamaan langkah laten berada dalam keluarga relaksasi diskret.

Hipotesis ini dapat ditolak apabila penggunaan `kv` menghasilkan penurunan akurasi yang bermakna dibandingkan `text`.

---

## 3.14 Variabel Penelitian

Variabel penelitian dapat dirangkum sebagai berikut.

| Jenis | Variabel | Operasionalisasi |
|---|---|---|
| Independen | formulasi \(\Phi\) | `raw`, `soft`, `sample`, `gumbel`, `moi` |
| Independen | medium komunikasi | `text`, `kv_and_text`, `kv` |
| Independen | jenis tugas | GSM8K, ARC-Challenge, HumanEval-Plus, faktor simbolik |
| Konstan | backbone | Qwen3-8B |
| Konstan | latent steps | \(m=10\) |
| Konstan | suhu distribusi | \(T=0{,}7\) |
| Konstan | parameter MoI | \(\beta=1\) |
| Konstan | regularisasi raw | \(\lambda=10^{-5}\) |
| Dependen | kinerja jawaban | exact-match / pass@1 |
| Dependen | keandalan | format rate |
| Dependen | fidelitas simbolik | recall, exact match, error/korupsi simbol |
| Dependen | biaya | token keluaran dan waktu inferensi |
| Dependen | faktor simbolik | validitas ekspresi dan RankIC |

---

## 3.15 Metrik Evaluasi

Evaluasi tidak hanya menggunakan akurasi karena satu nilai akurasi dapat menggabungkan beberapa mekanisme kegagalan yang berbeda.

### 3.15.1 Akurasi Jawaban

Untuk tugas dengan keluaran yang dapat dibandingkan secara langsung digunakan *exact match*. Untuk tugas kode digunakan keberhasilan pengujian sebagai ukuran *pass@1* atau ukuran yang ekuivalen dengan satu sampel keluaran.

Pada GSM8K, jawaban dianggap benar apabila nilai numeriknya sama dengan jawaban referensi. Pada tugas pilihan ganda, jawaban dibandingkan berdasarkan pilihan yang benar. Prosedur tersebut mengikuti protokol evaluasi LatentMAS.

### 3.15.2 Format Rate

`format_rate` digunakan untuk membedakan kegagalan akibat keluaran yang tidak memiliki format yang dapat dievaluasi dari kegagalan akibat jawaban yang salah.

Pemisahan tersebut penting karena peningkatan format yang valid tidak identik dengan peningkatan kemampuan penalaran.

### 3.15.3 Fidelitas Simbolik

Fidelitas simbolik digunakan untuk mengukur apakah informasi yang seharusnya dipertahankan secara simbolik tetap dapat diterima pada tahap berikutnya.

Metrik yang digunakan mencakup:

- *recall* simbol;
- *exact match*;
- tingkat halusinasi simbol;
- tingkat korupsi token; dan
- pada kasus lintas-bahasa, kemunculan karakter yang tidak diharapkan pada keluaran.

Pengukuran ini dimaksudkan untuk memberikan bukti langsung mengenai jenis informasi yang hilang, bukan sekadar menunjukkan bahwa skor akhir berubah.

### 3.15.4 Biaya Komputasi

Efisiensi diukur melalui:

1. jumlah token keluaran per soal; dan
2. waktu inferensi per soal.

Pengukuran token merupakan metrik penting karena salah satu motivasi utama LatentMAS adalah menghindari pertukaran *chain-of-thought* tekstual antar-agen. LatentMAS melaporkan pengurangan penggunaan token yang besar dibandingkan komunikasi berbasis teks.

---

## 3.16 Pengujian Geometris

Selain evaluasi tugas, dilakukan analisis geometris terhadap representasi yang dihasilkan oleh setiap metode.

Ukuran utama adalah kedekatan representasi laten terhadap embedding token, yang dapat dinyatakan melalui

\[
s(z)
=
\max_i
\cos
\left(
z,W_{\mathrm{in}}[i]
\right).
\]

Nilai tersebut digunakan untuk mengukur apakah representasi laten memiliki arah yang dekat dengan setidaknya satu embedding token.

Pada pengukuran awal Qwen3-8B, nilai rata-rata untuk metode *raw* berada sekitar \(0{,}31\), sedangkan *soft* sekitar \(0{,}93\) dan *gumbel* sekitar \(0{,}98\).

Analisis geometris ini tidak digunakan untuk menyimpulkan bahwa kedekatan terhadap token tertentu secara otomatis menyebabkan jawaban benar. Fungsinya adalah menyediakan ukuran mekanistik yang menghubungkan formulasi matematis dengan perilaku empiris.

Dengan demikian, penelitian membedakan dua jenis bukti:

1. **bukti geometris**, yaitu lokasi representasi pada ruang embedding; dan
2. **bukti fungsional**, yaitu pengaruh lokasi tersebut terhadap hasil tugas.

---

## 3.17 Analisis Statistik

Karena setiap kondisi menerima soal yang sama, analisis statistik dilakukan menggunakan pendekatan berpasangan.

### 3.17.1 Data Biner

Untuk data biner seperti benar/salah dan format valid/tidak valid digunakan **uji McNemar eksak**.

Uji tersebut hanya mempertimbangkan pasangan yang memberikan hasil berbeda antar-kondisi. Pendekatan eksak digunakan karena jumlah pasangan diskordan dapat relatif kecil.

### 3.17.2 Data Kontinu atau Ordinal

Untuk metrik kontinu atau ordinal seperti *recall* dan RankIC digunakan **Wilcoxon signed-rank test**. Pengujian ini dipilih karena tidak memerlukan asumsi distribusi normal dan sesuai dengan struktur data berpasangan.

### 3.17.3 Interval Kepercayaan

Setiap perbandingan dilengkapi dengan **interval kepercayaan bootstrap 95%**. Nilai \(p\) tidak digunakan sebagai satu-satunya dasar interpretasi karena pada ukuran sampel yang kecil nilai tersebut dapat memberikan gambaran yang tidak lengkap mengenai besarnya efek.

### 3.17.4 Koreksi Multipisitas

Karena terdapat banyak perbandingan antar-metode dan tugas, pengujian dilakukan dengan koreksi terhadap *multiple comparisons*. Kerangka penelitian menetapkan penggunaan prosedur seperti Holm atau Benjamini–Hochberg.

Secara khusus, terdapat 21 pasangan perbandingan pada tiga tugas utama sehingga dapat muncul hingga 63 pengujian. Tanpa koreksi, probabilitas munculnya temuan signifikan secara kebetulan akan meningkat.

### 3.17.5 Prinsip Interpretasi

Hasil yang tidak signifikan dilaporkan sebagai **tidak signifikan**. Istilah seperti "cenderung lebih baik" tidak digunakan untuk menggantikan hasil yang gagal mencapai tingkat signifikansi yang ditentukan.

Prinsip tersebut penting terutama untuk H1. Hipotesis H1 justru memprediksi bahwa beberapa varian dalam keluarga relaksasi diskret dapat memberikan performa yang serupa. Oleh karena itu, tidak ditemukannya perbedaan signifikan antar-varian bukan merupakan kegagalan eksperimen, tetapi merupakan salah satu kemungkinan hasil yang relevan terhadap hipotesis.

---

## 3.18 Prinsip Ketelitian Klaim

Penelitian ini menerapkan prinsip bahwa kebenaran suatu pernyataan harus dipahami berdasarkan **ruang lingkup rujukannya**, bukan hanya berdasarkan kebenaran literal kalimat.

Sebagai ilustrasi, pernyataan "cokelat itu manis" dapat benar apabila yang dimaksud adalah produk cokelat yang telah diberi gula. Namun, pernyataan yang sama dapat menghasilkan pemahaman yang salah apabila konteks yang dimaksud adalah buah kakao dalam keadaan alaminya. Dengan kata lain, sebuah pernyataan dapat mengandung fakta yang benar tetapi tetap menghasilkan kesimpulan yang menyesatkan apabila konteks yang diperlukan tidak dinyatakan.

Prinsip yang sama diterapkan dalam penelitian ini. Sebagai contoh, pernyataan bahwa suatu mekanisme "mempertahankan informasi" harus selalu disertai keterangan mengenai **informasi apa**, **pada tahap mana**, dan **dalam arti apa** informasi tersebut dipertahankan.

Oleh karena itu, penelitian ini membedakan:

1. **fidelitas komunikasi KV**, yaitu kemampuan KV-cache meneruskan representasi yang telah tersimpan;
2. **fidelitas pembentukan langkah laten**, yaitu kemampuan \(\Phi\) menghasilkan representasi yang sesuai dengan ruang input model; dan
3. **fidelitas tugas**, yaitu kemampuan sistem mempertahankan informasi yang relevan hingga menghasilkan keluaran yang benar.

LatentMAS memberikan dasar teoritis bahwa transfer working memory melalui KV-cache dapat mempertahankan representasi yang diterima agen sebelumnya ketika kondisi model dan cache sesuai. Karena itu, penelitian ini tidak menyebut KV-cache sebagai mekanisme yang "lossy" tanpa kualifikasi. Fokus pengujian adalah apakah proses pembentukan \(z_t=\Phi(h_t)\) sebelum representasi tersebut masuk ke KV-cache menghasilkan distorsi yang relevan terhadap tugas.

Dengan prinsip ini, klaim penelitian dibatasi pada kondisi eksperimen yang benar-benar diuji.

---

## 3.19 Batasan Rancangan Penelitian

Beberapa batasan ditetapkan secara sengaja.

Pertama, eksperimen utama hanya menggunakan Qwen3-8B. Oleh karena itu, hasil tidak digunakan untuk menyatakan generalisasi terhadap seluruh ukuran atau keluarga LLM.

Kedua, penelitian tidak menguji evolusi agen, *reinforcement learning*, maupun pelatihan parameter model. Fokus penelitian dibatasi pada intervensi *training-free* terhadap persamaan langkah laten.

Ketiga, penelitian tidak menganggap Gumbel-Softmax, Soft Thinking, atau MoI sebagai penemuan baru. Ketiga pendekatan tersebut telah diperkenalkan dalam penelitian terdahulu. Kontribusi penelitian terletak pada penyatuan formulasi matematisnya dan pemindahan mekanisme tersebut ke konteks kolaborasi laten multi-agen. 
Keempat, medium `kv_and_text` digunakan sebagai kondisi tambahan dan memiliki ukuran sampel yang lebih kecil dibandingkan kondisi utama. Oleh karena itu, kesimpulan utama mengenai perbandingan medium tidak diperluas melampaui kondisi yang memiliki dukungan data memadai.

Kelima, hasil pada domain faktor simbolik tidak ditafsirkan sebagai prediksi keuntungan investasi. QuantaAlpha sendiri menempatkan *alpha mining* sebagai proses pencarian faktor dan mengevaluasinya melalui metrik seperti IC, ARR, dan MDD. Penelitian ini hanya menggunakan domain tersebut sebagai pengujian kemampuan mempertahankan ekspresi simbolik.

---

## 3.20 Alur Pelaksanaan Penelitian

Secara keseluruhan, penelitian dilaksanakan melalui tahapan berikut:

\[
\boxed{
\text{Pemilihan Backbone}
}
\rightarrow
\boxed{
\text{Pembentukan }\Phi
}
\rightarrow
\boxed{
\text{Normalisasi}
}
\rightarrow
\boxed{
\text{Latent Reasoning}
}
\rightarrow
\boxed{
\text{Komunikasi Antar-Agen}
}
\rightarrow
\boxed{
\text{Evaluasi Tugas}
}
\rightarrow
\boxed{
\text{Analisis Statistik}
}
\rightarrow
\boxed{
\text{Interpretasi}
}
\]

Tahap pertama menetapkan Qwen3-8B sebagai *backbone*. Tahap kedua membentuk lima kondisi persamaan langkah laten, yaitu *raw*, *soft*, *sample*, *gumbel*, dan *moi*. Tahap ketiga menerapkan normalisasi magnitudo yang sama. Tahap keempat menjalankan sepuluh langkah laten pada setiap agen yang menggunakan komunikasi laten. Tahap kelima meneruskan informasi melalui medium `text`, `kv_and_text`, atau `kv`.

Selanjutnya, keluaran sistem dievaluasi menggunakan metrik akurasi, keandalan format, fidelitas simbolik, dan biaya. Hasil setiap kondisi kemudian dipasangkan berdasarkan soal yang sama dan dianalisis menggunakan pengujian statistik yang sesuai.

Interpretasi akhir dilakukan dengan menghubungkan tiga tingkat bukti, yaitu:

\[
\text{formulasi matematis}
\rightarrow
\text{geometri representasi}
\rightarrow
\text{kinerja tugas}.
\]

Dengan demikian, apabila suatu perbedaan performa ditemukan, penelitian tidak hanya menanyakan **metode mana yang memperoleh skor lebih tinggi**, tetapi juga **apakah perbedaan tersebut konsisten dengan perbedaan struktur matematis representasi laten yang digunakan**.