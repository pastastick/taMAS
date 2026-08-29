# BAB III  
# METODE PENELITIAN

Bab ini membahas metode yang digunakan dalam penelitian untuk membandingkan persamaan langkah laten pada kolaborasi multi-agen berbasis *large language model* (LLM). Pembahasan diawali dengan formulasi langkah laten pada LatentMAS, kemudian dilanjutkan dengan metode *Soft Thinking*, *Gumbel-Softmax*, dan *Mixture of Inputs* (MoI). Selanjutnya, metode-metode tersebut dirumuskan dalam suatu bentuk umum berdasarkan kombinasi konveks embedding token. Pada bagian akhir dibahas rancangan eksperimen, data yang digunakan, metrik evaluasi, dan metode pengujian statistik.

## 3.1. Formulasi Langkah Laten

LatentMAS merupakan kerangka kolaborasi multi-agen yang memungkinkan agen melakukan penalaran dan pertukaran informasi dalam ruang laten tanpa menghasilkan teks pada setiap langkah penalaran. Pada setiap langkah, model menghasilkan representasi pada lapisan terakhir yang kemudian digunakan kembali sebagai masukan pada langkah berikutnya.

Misalkan suatu model bahasa memiliki dimensi tersembunyi sebesar \(d\). Setelah model menerima suatu konteks, diperoleh *hidden state* terakhir

\[
h_t\in\mathbb{R}^{d}.
\]

Vektor tersebut kemudian dipetakan menggunakan suatu fungsi transformasi \(\Phi\) sehingga diperoleh representasi laten

\[
z_t=\Phi(h_t).
\tag{3.1}
\]

Representasi \(z_t\) selanjutnya diberikan kembali kepada model sebagai *input embedding* pada langkah berikutnya. Jika \(f_\theta\) menyatakan fungsi Transformer dengan parameter \(\theta\), maka proses pembangkitan langkah laten dapat dituliskan sebagai

\[
h_{t+1}
=
f_\theta
\left(
z_t\mid \mathrm{KV}_{1:t}
\right),
\tag{3.2}
\]

dengan \(\mathrm{KV}_{1:t}\) menyatakan KV-cache yang telah terbentuk hingga langkah ke-\(t\).

Proses pada Persamaan (3.1) dan (3.2) dilakukan secara berulang sebanyak \(m\) langkah. Pada penelitian ini digunakan

\[
m=10.
\]

Berbeda dengan pembangkitan autoregresif biasa, \(z_t\) tidak terlebih dahulu diubah menjadi token diskrit. Vektor tersebut langsung diberikan melalui parameter `inputs_embeds`. Dengan demikian, satu langkah laten tidak menghasilkan satu token teks, tetapi menghasilkan satu representasi kontinu yang kemudian diproses kembali oleh model.

Secara umum, apabila \(h_1\) merupakan *hidden state* awal, maka urutan langkah laten dapat dituliskan sebagai

\[
h_1
\overset{\Phi}{\longrightarrow}
z_1
\overset{f_\theta}{\longrightarrow}
h_2
\overset{\Phi}{\longrightarrow}
z_2
\overset{f_\theta}{\longrightarrow}
\cdots
\overset{\Phi}{\longrightarrow}
z_m
\overset{f_\theta}{\longrightarrow}
h_{m+1}.
\tag{3.3}
\]

Dalam penelitian ini, fungsi \(\Phi\) merupakan objek yang dibandingkan. Struktur agen, model dasar, jumlah langkah laten, dan prosedur komunikasi dijaga tetap sehingga perbedaan hasil dapat dikaitkan dengan perbedaan formulasi \(\Phi\).

---

## 3.2. Persamaan Langkah Laten LatentMAS

### 3.2.1. Input Embedding dan Output Embedding

Pada model bahasa autoregresif, token masukan terlebih dahulu dipetakan menjadi vektor embedding. Misalkan \(V\) menyatakan ukuran kosakata dan \(d\) menyatakan dimensi *hidden state*. Matriks embedding masukan dapat dituliskan sebagai

\[
W_{\mathrm{in}}\in\mathbb{R}^{V\times d}.
\]

Baris ke-\(i\) dari matriks tersebut, yaitu

\[
W_{\mathrm{in}}[i]\in\mathbb{R}^{d},
\]

merupakan embedding dari token ke-\(i\).

Pada bagian keluaran, model menggunakan *language model head* yang dinyatakan dengan

\[
W_{\mathrm{out}}\in\mathbb{R}^{V\times d}.
\]

Jika \(h\in\mathbb{R}^{d}\) merupakan *hidden state* terakhir, maka logit untuk seluruh token dapat dituliskan sebagai

\[
\ell=W_{\mathrm{out}}h,
\tag{3.4}
\]

dengan

\[
\ell\in\mathbb{R}^{V}.
\]

Distribusi probabilitas token selanjutnya diperoleh menggunakan fungsi *softmax*,

\[
p_i
=
\frac{\exp(\ell_i)}
{\displaystyle\sum_{j=1}^{V}\exp(\ell_j)},
\qquad i=1,2,\ldots,V.
\tag{3.5}
\]

Secara vektor,

\[
p=\operatorname{softmax}(\ell).
\tag{3.6}
\]

Matriks \(W_{\mathrm{in}}\) dan \(W_{\mathrm{out}}\) memiliki dimensi yang sesuai untuk dihubungkan melalui transformasi linear, tetapi keduanya tidak selalu memiliki representasi geometris yang sama. Oleh karena itu, diperlukan suatu transformasi yang memetakan *hidden state* dari ruang keluaran ke ruang embedding masukan.

### 3.2.2. Realignment Matrix

LatentMAS menggunakan matriks *realignment* \(W_a\) untuk memetakan *hidden state* \(h\) ke ruang embedding masukan. Pemetaan tersebut dinyatakan sebagai

\[
e=hW_a.
\tag{3.7}
\]

Matriks \(W_a\) dicari sedemikian rupa sehingga hasil pemetaan keluaran model mendekati embedding masukan. Permasalahan tersebut dirumuskan sebagai

\[
\min_{W_a}
\left\|
W_{\mathrm{out}}W_a-W_{\mathrm{in}}
\right\|_F^2.
\tag{3.8}
\]

Norma Frobenius suatu matriks \(A=[a_{ij}]\) didefinisikan sebagai

\[
\|A\|_F^2
=
\sum_i\sum_j a_{ij}^2.
\tag{3.9}
\]

Dengan demikian, Persamaan (3.8) meminimalkan jumlah kuadrat selisih antara \(W_{\mathrm{out}}W_a\) dan \(W_{\mathrm{in}}\).

### 3.2.3. Penurunan Solusi Least Squares

Untuk memperoleh solusi Persamaan (3.8), fungsi objektif dapat dituliskan sebagai

\[
J(W_a)
=
\left\|
W_{\mathrm{out}}W_a-W_{\mathrm{in}}
\right\|_F^2.
\tag{3.10}
\]

Dengan menggunakan sifat norma Frobenius,

\[
J(W_a)
=
\operatorname{tr}
\left[
(W_{\mathrm{out}}W_a-W_{\mathrm{in}})^\top
(W_{\mathrm{out}}W_a-W_{\mathrm{in}})
\right].
\tag{3.11}
\]

Turunan terhadap \(W_a\) menghasilkan

\[
\frac{\partial J(W_a)}
{\partial W_a}
=
2W_{\mathrm{out}}^\top
(W_{\mathrm{out}}W_a-W_{\mathrm{in}}).
\tag{3.12}
\]

Pada titik optimum, turunan pertama sama dengan nol sehingga

\[
W_{\mathrm{out}}^\top
(W_{\mathrm{out}}W_a-W_{\mathrm{in}})
=0.
\tag{3.13}
\]

Dengan demikian diperoleh *normal equation*

\[
W_{\mathrm{out}}^\top W_{\mathrm{out}}W_a
=
W_{\mathrm{out}}^\top W_{\mathrm{in}}.
\tag{3.14}
\]

Apabila \(W_{\mathrm{out}}^\top W_{\mathrm{out}}\) invertibel, solusi least squares dapat dituliskan sebagai

\[
W_a
=
\left(
W_{\mathrm{out}}^\top W_{\mathrm{out}}
\right)^{-1}
W_{\mathrm{out}}^\top W_{\mathrm{in}}.
\tag{3.15}
\]

Namun, dalam praktiknya matriks tersebut dapat memiliki kondisi numerik yang kurang baik. Oleh karena itu, LatentMAS menggunakan regularisasi *ridge*.

### 3.2.4. Realignment Matrix dengan Ridge Regression

Fungsi objektif dengan regularisasi dirumuskan sebagai

\[
J(W_a)
=
\left\|
W_{\mathrm{out}}W_a-W_{\mathrm{in}}
\right\|_F^2
+
\lambda
\|W_a\|_F^2,
\tag{3.16}
\]

dengan \(\lambda>0\) merupakan parameter regularisasi.

Turunan Persamaan (3.16) terhadap \(W_a\) adalah

\[
\frac{\partial J(W_a)}
{\partial W_a}
=
2W_{\mathrm{out}}^\top
(W_{\mathrm{out}}W_a-W_{\mathrm{in}})
+
2\lambda W_a.
\tag{3.17}
\]

Dengan menyamakan turunan terhadap nol diperoleh

\[
W_{\mathrm{out}}^\top
(W_{\mathrm{out}}W_a-W_{\mathrm{in}})
+
\lambda W_a
=0.
\tag{3.18}
\]

Persamaan tersebut dapat disusun menjadi

\[
W_{\mathrm{out}}^\top W_{\mathrm{out}}W_a
+
\lambda W_a
=
W_{\mathrm{out}}^\top W_{\mathrm{in}},
\tag{3.19}
\]

sehingga

\[
\left(
W_{\mathrm{out}}^\top W_{\mathrm{out}}
+
\lambda I
\right)W_a
=
W_{\mathrm{out}}^\top W_{\mathrm{in}}.
\tag{3.20}
\]

Dengan asumsi matriks pada ruas kiri invertibel, diperoleh

\[
\boxed{
W_a
=
\left(
W_{\mathrm{out}}^\top W_{\mathrm{out}}
+
\lambda I
\right)^{-1}
W_{\mathrm{out}}^\top W_{\mathrm{in}}
}
\tag{3.21}
\]

yang merupakan persamaan *realignment matrix* yang digunakan pada LatentMAS. Paper LatentMAS juga menurunkan solusi tersebut dari permasalahan *ridge regression*.

Pada penelitian ini digunakan

\[
\lambda=10^{-5}.
\]

### 3.2.5. Normalisasi Representasi LatentMAS

Setelah *hidden state* dipetakan menggunakan \(W_a\), diperoleh

\[
\tilde z_{\mathrm{raw}}
=
hW_a.
\tag{3.22}
\]

Untuk menjaga skala representasi laten agar berada pada magnitudo yang sebanding dengan embedding masukan, digunakan normalisasi.

Misalkan

\[
\rho
=
\frac{1}{V}
\sum_{i=1}^{V}
\left\|
W_{\mathrm{in}}[i]
\right\|_2
\tag{3.23}
\]

merupakan rata-rata norma embedding masukan. Representasi laten kemudian dirumuskan sebagai

\[
\boxed{
\Phi_{\mathrm{raw}}(h)
=
\rho
\frac{hW_a}
{\|hW_a\|_2}
}
\tag{3.24}
\]

Persamaan (3.24) merupakan persamaan langkah laten yang digunakan sebagai metode *raw* pada penelitian ini.

### 3.2.6. Latent Thought Generation

Dengan menggunakan Persamaan (3.24), langkah laten ke-\(t\) dapat dituliskan sebagai

\[
z_t
=
\rho
\frac{h_tW_a}
{\|h_tW_a\|_2}.
\tag{3.25}
\]

Vektor \(z_t\) selanjutnya diberikan kepada model sebagai *input embedding*. Dengan demikian,

\[
h_{t+1}
=
f_\theta
\left(
z_t\mid \mathrm{KV}_{1:t}
\right).
\tag{3.26}
\]

Persamaan tersebut dilakukan berulang hingga \(m\) langkah. Pada akhir proses, KV-cache yang terbentuk selama langkah laten digunakan sebagai memori kerja laten.

### 3.2.7. Transfer Working Memory melalui KV-cache

Untuk setiap lapisan Transformer ke-\(l\), KV-cache dapat dinyatakan sebagai pasangan

\[
\mathrm{KV}^{(l)}
=
\left(
K^{(l)},V^{(l)}
\right).
\tag{3.27}
\]

Jika suatu agen \(A_1\) telah menyelesaikan proses penalaran, memori kerja latennya dapat dinyatakan sebagai

\[
\mathcal{M}_{A_1}
=
\left\{
\left(
K^{(l)}_{A_1},
V^{(l)}_{A_1}
\right)
\right\}_{l=1}^{L},
\tag{3.28}
\]

dengan \(L\) menyatakan jumlah lapisan Transformer.

KV-cache tersebut kemudian diberikan kepada agen berikutnya. Apabila agen \(A_2\) menghasilkan KV-cache baru

\[
\mathcal{M}_{A_2}^{\mathrm{new}},
\]

maka memori yang digunakan pada tahap berikutnya dapat dibentuk dengan menggabungkan KV-cache sebelumnya dan KV-cache baru pada setiap lapisan.

LatentMAS menggunakan mekanisme tersebut untuk meneruskan memori kerja antar-agen tanpa melakukan dekode dan enkode ulang dalam bentuk teks.

---

## 3.3. Representasi Laten Berbasis Distribusi Token

Metode *raw* menggunakan transformasi linear langsung terhadap *hidden state*. Pendekatan lain dapat dibentuk dengan memanfaatkan distribusi probabilitas token yang dihasilkan oleh *language model head*.

Pada penelitian ini, logit diperoleh berdasarkan Persamaan (3.4), yaitu

\[
\ell=W_{\mathrm{out}}h.
\]

Distribusi token kemudian dihitung menggunakan *temperature scaling*,

\[
p_i
=
\frac{\exp(\ell_i/T)}
{\displaystyle\sum_{j=1}^{V}\exp(\ell_j/T)},
\tag{3.29}
\]

dengan

\[
T=0{,}7.
\]

Bentuk umum tersebut menghasilkan

\[
p
=
\operatorname{softmax}
\left(
\frac{\ell}{T}
\right).
\tag{3.30}
\]

Distribusi \(p\) digunakan sebagai dasar pembentukan representasi laten pada metode *soft*, *sample*, *gumbel*, dan *moi*.

---

## 3.4. Metode Soft Thinking

### 3.4.1. Soft Token

Pada pembangkitan token konvensional, model memilih satu token dari distribusi \(p\). Sebaliknya, *Soft Thinking* mempertahankan seluruh distribusi probabilitas tersebut.

Misalkan

\[
p=(p_1,p_2,\ldots,p_V),
\]

dengan

\[
p_i\geq0,
\qquad
\sum_{i=1}^{V}p_i=1.
\]

Distribusi tersebut digunakan sebagai bobot embedding sehingga diperoleh

\[
\tilde z_{\mathrm{soft}}
=
\sum_{i=1}^{V}
p_iW_{\mathrm{in}}[i].
\tag{3.31}
\]

Karena \(p\) merupakan distribusi probabilitas, maka \(p\) merupakan anggota simpleks probabilitas

\[
\Delta^{V-1}
=
\left\{
w\in\mathbb{R}^{V}
:
w_i\geq0,\,
\sum_{i=1}^{V}w_i=1
\right\}.
\tag{3.32}
\]

Oleh karena itu, \(\tilde z_{\mathrm{soft}}\) merupakan kombinasi konveks dari baris-baris \(W_{\mathrm{in}}\).

Representasi yang digunakan dalam penelitian kemudian dinormalisasi menjadi

\[
\Phi_{\mathrm{soft}}(h)
=
\rho
\frac{\tilde z_{\mathrm{soft}}}
{\|\tilde z_{\mathrm{soft}}\|_2}.
\tag{3.33}
\]

---

## 3.5. Metode Sampling Kategoris

Sebagai pembanding terhadap *Soft Thinking*, digunakan pembangkitan token secara kategoris.

Misalkan

\[
Y\sim\operatorname{Categorical}(p).
\tag{3.34}
\]

Jika token yang terpilih adalah \(y\), didefinisikan vektor satuan

\[
e_y=(0,\ldots,0,1,0,\ldots,0)^\top,
\tag{3.35}
\]

dengan nilai satu berada pada indeks \(y\).

Representasi embedding yang dihasilkan adalah

\[
\tilde z_{\mathrm{sample}}
=
\sum_{i=1}^{V}(e_y)_iW_{\mathrm{in}}[i]
=
W_{\mathrm{in}}[y].
\tag{3.36}
\]

Karena \(e_y\in\Delta^{V-1}\), representasi tersebut juga merupakan kombinasi konveks embedding token, tetapi hanya menggunakan satu titik pada simpleks.

Setelah normalisasi,

\[
\Phi_{\mathrm{sample}}(h)
=
\rho
\frac{W_{\mathrm{in}}[y]}
{\|W_{\mathrm{in}}[y]\|_2}.
\tag{3.37}
\]

---

## 3.6. Metode Gumbel-Softmax

### 3.6.1. Distribusi Gumbel

Distribusi Gumbel digunakan untuk menghasilkan derau acak yang dapat digunakan dalam proses pemilihan kategori. Variabel random \(G\) dikatakan mengikuti distribusi Gumbel dengan parameter lokasi \(\mu\) dan skala \(\beta>0\), dituliskan sebagai

\[
G\sim\operatorname{Gumbel}(\mu,\beta),
\]

apabila fungsi distribusi kumulatifnya adalah

\[
F_G(g)
=
\exp
\left[
-\exp
\left(
-\frac{g-\mu}{\beta}
\right)
\right].
\tag{3.38}
\]

Dalam penelitian ini digunakan distribusi Gumbel standar,

\[
G\sim\operatorname{Gumbel}(0,1).
\tag{3.39}
\]

### 3.6.2. Gumbel-Max Trick

Misalkan terdapat \(V\) kategori dengan probabilitas

\[
p_i>0,
\qquad
\sum_{i=1}^{V}p_i=1.
\]

Untuk setiap kategori dihasilkan derau independen

\[
g_i\overset{\mathrm{iid}}{\sim}
\operatorname{Gumbel}(0,1).
\tag{3.40}
\]

Gumbel-Max Trick memilih kategori

\[
Y
=
\arg\max_i
\left(
\log p_i+g_i
\right).
\tag{3.41}
\]

Metode tersebut menghasilkan

\[
P(Y=i)=p_i.
\tag{3.42}
\]

Dengan demikian, Gumbel-Max dapat menghasilkan sampel kategoris melalui operasi *argmax* pada log-probabilitas yang telah ditambahkan derau Gumbel. Stochastic Soft Thinking menggunakan prinsip tersebut sebagai dasar pembentukan *stochastic soft token*.

### 3.6.3. Relaksasi Gumbel-Max

Operasi \(\arg\max\) pada Persamaan (3.41) menghasilkan vektor diskrit sehingga tidak dapat digunakan secara langsung sebagai representasi kontinu.

Untuk memperoleh representasi yang dapat berubah secara halus, operasi tersebut digantikan oleh *softmax*. Misalkan \(\ell_i\) merupakan logit dan \(g_i\) merupakan derau Gumbel. Dengan temperatur \(\tau>0\), bobot Gumbel-Softmax didefinisikan sebagai

\[
w_i^{\mathrm{gumbel}}
=
\frac{
\exp\left((\ell_i+g_i)/\tau\right)
}{
\displaystyle
\sum_{j=1}^{V}
\exp\left((\ell_j+g_j)/\tau\right)
}.
\tag{3.43}
\]

Secara vektor,

\[
w_{\mathrm{gumbel}}
=
\operatorname{softmax}
\left(
\frac{\ell+g}{\tau}
\right).
\tag{3.44}
\]

Karena fungsi *softmax* menghasilkan nilai nonnegatif dengan jumlah satu, maka

\[
w_{\mathrm{gumbel}}\in\Delta^{V-1}.
\tag{3.45}
\]

Representasi embedding kemudian diperoleh melalui

\[
\tilde z_{\mathrm{gumbel}}
=
\sum_{i=1}^{V}
w_i^{\mathrm{gumbel}}
W_{\mathrm{in}}[i].
\tag{3.46}
\]

Setelah normalisasi,

\[
\Phi_{\mathrm{gumbel}}(h)
=
\rho
\frac{
\tilde z_{\mathrm{gumbel}}
}{
\|\tilde z_{\mathrm{gumbel}}\|_2
}.
\tag{3.47}
\]

### 3.6.4. Pengaruh Temperatur

Parameter \(\tau\) mengatur tingkat kehalusan distribusi Gumbel-Softmax.

Untuk \(\tau\rightarrow0\),

\[
w_{\mathrm{gumbel}}
\rightarrow e_y,
\tag{3.48}
\]

sehingga distribusi semakin mendekati representasi one-hot yang dihasilkan oleh Gumbel-Max.

Sebaliknya, ketika \(\tau\) meningkat, distribusi menjadi lebih menyebar pada beberapa kategori. Dengan demikian, parameter \(\tau\) mengatur hubungan antara representasi diskrit dan representasi kontinu.

Stochastic Soft Thinking menggunakan Gumbel-Softmax sebagai metode untuk menghasilkan *stochastic soft token* dengan tingkat *softness* yang dapat dikendalikan melalui temperatur.

Pada penelitian ini digunakan temperatur Gumbel

\[
\tau=0{,}5.
\]

---

## 3.7. Metode Mixture of Inputs

### 3.7.1. Konsep Mixture of Inputs

Mixture of Inputs (MoI) merupakan metode yang menggabungkan token hasil sampling dengan informasi distribusi token. Misalkan \(e_i\in\mathbb{R}^{d}\) merupakan embedding token ke-\(i\), distribusi token dinyatakan dengan \(p\), dan token hasil sampling dinyatakan dengan \(y\).

MoI membentuk representasi

\[
\tilde z_{\mathrm{moi}}
=
\sum_{i=1}^{V}
w_i e_i,
\tag{3.49}
\]

dengan

\[
w_i\geq0,
\qquad
\sum_{i=1}^{V}w_i=1.
\tag{3.50}
\]

Berbeda dengan *Soft Thinking* yang menggunakan \(w_i=p_i\), MoI menentukan \(w_i\) berdasarkan proses inferensi Bayesian.

### 3.7.2. Entropi Ternormalisasi

Entropi distribusi \(p\) didefinisikan sebagai

\[
H(p)
=
-\sum_{i=1}^{V}
p_i\log p_i.
\tag{3.51}
\]

Nilai maksimum entropi distribusi kategoris dengan \(V\) kategori adalah

\[
H_{\max}=\log V.
\tag{3.52}
\]

Oleh karena itu, digunakan entropi ternormalisasi

\[
\boxed{
H
=
-\frac{1}{\log V}
\sum_{i=1}^{V}
p_i\log p_i
}
\tag{3.53}
\]

sehingga

\[
0\leq H\leq1.
\tag{3.54}
\]

Nilai \(H\) yang mendekati nol menunjukkan distribusi yang terkonsentrasi pada beberapa token, sedangkan nilai \(H\) yang mendekati satu menunjukkan distribusi yang lebih menyebar.

### 3.7.3. Prior Dirichlet

MoI memodelkan bobot campuran \(w\) sebagai variabel random pada simpleks probabilitas. Distribusi prior yang digunakan adalah distribusi Dirichlet,

\[
w\sim\operatorname{Dir}(\alpha),
\tag{3.55}
\]

dengan parameter

\[
\alpha=Hp.
\tag{3.56}
\]

Dengan demikian,

\[
\alpha_i=Hp_i.
\tag{3.57}
\]

Jumlah parameter konsentrasi adalah

\[
\sum_{i=1}^{V}\alpha_i
=
H\sum_{i=1}^{V}p_i
=
H.
\tag{3.58}
\]

Karena \(H\in[0,1]\), jumlah konsentrasi prior dikendalikan oleh tingkat ketidakpastian distribusi token.

### 3.7.4. Informasi Token Hasil Sampling

Setelah distribusi \(p\) diperoleh, satu token dipilih berdasarkan

\[
Y\sim\operatorname{Categorical}(p).
\tag{3.59}
\]

Hasil sampling tersebut dinyatakan menggunakan vektor satu-hot

\[
y_i
=
\begin{cases}
1,&i=Y,\\
0,&i\neq Y.
\end{cases}
\tag{3.60}
\]

MoI memperlakukan token hasil sampling sebagai satu observasi atau *pseudo-count*. Dengan parameter \(\beta\), *pseudo-count* didefinisikan sebagai

\[
c_i
=
(\beta+1-H)y_i.
\tag{3.61}
\]

Jumlah seluruh *pseudo-count* adalah

\[
N
=
\sum_{i=1}^{V}c_i
=
\beta+1-H.
\tag{3.62}
\]

### 3.7.5. Distribusi Posterior

Distribusi Dirichlet memiliki sifat konjugat terhadap distribusi Multinomial. Oleh karena itu, apabila

\[
w\sim\operatorname{Dir}(\alpha)
\]

dan observasi memberikan *count* \(c\), distribusi posterior adalah

\[
w\mid c
\sim
\operatorname{Dir}(\alpha+c).
\tag{3.63}
\]

Pada MoI diperoleh

\[
w\mid y
\sim
\operatorname{Dir}
\left(
\alpha+c
\right).
\tag{3.64}
\]

Dengan Persamaan (3.57) dan (3.61),

\[
w\mid y
\sim
\operatorname{Dir}
\left(
Hp+
(\beta+1-H)y
\right).
\tag{3.65}
\]

### 3.7.6. Posterior Mean

Untuk variabel random

\[
W\sim\operatorname{Dir}(\alpha_1,\ldots,\alpha_V),
\]

nilai harapannya pada komponen ke-\(i\) adalah

\[
E[W_i]
=
\frac{\alpha_i}
{\displaystyle\sum_{j=1}^{V}\alpha_j}.
\tag{3.66}
\]

Berdasarkan Persamaan (3.65), parameter posterior pada komponen ke-\(i\) adalah

\[
\alpha_i+c_i
=
Hp_i+
(\beta+1-H)y_i.
\tag{3.67}
\]

Jumlah parameter posterior adalah

\[
\sum_{i=1}^{V}
(\alpha_i+c_i)
=
H+\beta+1-H
=
\beta+1.
\tag{3.68}
\]

Oleh karena itu, posterior mean dapat dituliskan sebagai

\[
\boxed{
w_i
=
\frac{
Hp_i+
(\beta+1-H)y_i
}{
\beta+1
}
}
\tag{3.69}
\]

atau dalam bentuk vektor,

\[
\boxed{
w_{\mathrm{moi}}
=
\frac{
Hp+
(\beta+1-H)e_y
}{
\beta+1
}
}.
\tag{3.70}
\]

Persamaan tersebut merupakan persamaan utama Mixture of Inputs yang diperoleh melalui estimasi posterior mean pada model Dirichlet-Multinomial.

### 3.7.7. Representasi Embedding MoI

Setelah bobot \(w_{\mathrm{moi}}\) diperoleh, embedding campuran dihitung menggunakan

\[
\tilde z_{\mathrm{moi}}
=
\sum_{i=1}^{V}
w_{\mathrm{moi},i}
W_{\mathrm{in}}[i].
\tag{3.71}
\]

Dengan menggunakan Persamaan (3.70), diperoleh

\[
\tilde z_{\mathrm{moi}}
=
\sum_{i=1}^{V}
\left[
\frac{
Hp_i+
(\beta+1-H)(e_y)_i
}{
\beta+1
}
\right]
W_{\mathrm{in}}[i].
\tag{3.72}
\]

Karena

\[
\sum_{i=1}^{V}
p_iW_{\mathrm{in}}[i]
=
\tilde z_{\mathrm{soft}}
\]

dan

\[
\sum_{i=1}^{V}
(e_y)_iW_{\mathrm{in}}[i]
=
W_{\mathrm{in}}[y],
\]

maka

\[
\boxed{
\tilde z_{\mathrm{moi}}
=
\frac{H}{\beta+1}
\tilde z_{\mathrm{soft}}
+
\frac{\beta+1-H}{\beta+1}
W_{\mathrm{in}}[y]
}.
\tag{3.73}
\]

Persamaan (3.73) menunjukkan bahwa MoI merupakan perpaduan antara representasi *soft* dan representasi hasil sampling.

Setelah normalisasi, diperoleh

\[
\Phi_{\mathrm{moi}}(h)
=
\rho
\frac{
\tilde z_{\mathrm{moi}}
}{
\|\tilde z_{\mathrm{moi}}\|_2
}.
\tag{3.74}
\]

Pada penelitian ini digunakan

\[
\beta=1.
\]

---

## 3.8. Penyatuan Persamaan Langkah Laten

Keempat metode berbasis distribusi token dapat dituliskan menggunakan satu bentuk umum.

Misalkan

\[
w\in\Delta^{V-1}.
\]

Definisikan

\[
\tilde z(w)
=
\sum_{i=1}^{V}
w_iW_{\mathrm{in}}[i].
\tag{3.75}
\]

Representasi langkah laten kemudian dinormalisasi sebagai

\[
\boxed{
\Phi_w(h)
=
\rho
\frac{
\tilde z(w)
}{
\|\tilde z(w)\|_2
}
}.
\tag{3.76}
\]

Dengan demikian, perbedaan metode terletak pada cara menentukan \(w\).

### 3.8.1. Proposisi 3.1

**Proposisi 3.1.** Metode *Soft Thinking*, sampling kategoris, Gumbel-Softmax, dan Mixture of Inputs menghasilkan representasi laten yang merupakan kombinasi konveks dari embedding token.

**Bukti.**

Untuk metode *Soft Thinking* berlaku

\[
w_{\mathrm{soft}}=p.
\]

Karena \(p\) merupakan distribusi probabilitas,

\[
p_i\geq0,
\qquad
\sum_i p_i=1,
\]

maka

\[
w_{\mathrm{soft}}\in\Delta^{V-1}.
\]

Untuk sampling kategoris,

\[
w_{\mathrm{sample}}=e_y.
\]

Vektor satu-hot memenuhi

\[
(e_y)_i\geq0,
\qquad
\sum_i(e_y)_i=1,
\]

sehingga

\[
w_{\mathrm{sample}}\in\Delta^{V-1}.
\]

Untuk Gumbel-Softmax,

\[
w_{\mathrm{gumbel}}
=
\operatorname{softmax}
\left(
\frac{\ell+g}{\tau}
\right).
\]

Fungsi *softmax* menghasilkan komponen nonnegatif dengan jumlah satu sehingga

\[
w_{\mathrm{gumbel}}\in\Delta^{V-1}.
\]

Untuk MoI,

\[
w_{\mathrm{moi}}
=
\frac{
Hp+
(\beta+1-H)e_y
}{
\beta+1}.
\]

Karena \(0\leq H\leq1\) dan \(\beta\geq0\), diperoleh

\[
H\geq0
\]

dan

\[
\beta+1-H\geq0.
\]

Selain itu,

\[
\sum_i w_{\mathrm{moi},i}
=
\frac{
H\sum_i p_i+
(\beta+1-H)\sum_i(e_y)_i
}{
\beta+1}
=1.
\]

Dengan demikian,

\[
w_{\mathrm{moi}}\in\Delta^{V-1}.
\]

Karena keempat bobot tersebut merupakan anggota \(\Delta^{V-1}\), maka masing-masing representasi

\[
\tilde z
=
\sum_iw_iW_{\mathrm{in}}[i]
\]

merupakan kombinasi konveks dari baris matriks \(W_{\mathrm{in}}\). \(\square\)

### 3.8.2. Proposisi 3.2

**Proposisi 3.2.** MoI dapat dinyatakan sebagai kombinasi konveks antara metode *Soft Thinking* dan sampling kategoris.

**Bukti.**

Persamaan MoI dapat ditulis sebagai

\[
w_{\mathrm{moi}}
=
\frac{H}{\beta+1}p
+
\frac{\beta+1-H}{\beta+1}e_y.
\tag{3.77}
\]

Definisikan

\[
\lambda
=
\frac{H}{\beta+1}.
\tag{3.78}
\]

Maka

\[
1-\lambda
=
\frac{\beta+1-H}{\beta+1}.
\tag{3.79}
\]

Sehingga

\[
w_{\mathrm{moi}}
=
\lambda p
+
(1-\lambda)e_y.
\tag{3.80}
\]

Karena \(0\leq H\leq1\) dan \(\beta\geq0\), maka

\[
0\leq\lambda\leq1.
\]

Dengan demikian, \(w_{\mathrm{moi}}\) merupakan kombinasi konveks dari \(p\) dan \(e_y\). Karena

\[
w_{\mathrm{soft}}=p
\]

dan

\[
w_{\mathrm{sample}}=e_y,
\]

maka

\[
\boxed{
w_{\mathrm{moi}}
=
\lambda w_{\mathrm{soft}}
+
(1-\lambda)w_{\mathrm{sample}}
}.
\tag{3.81}
\]

\(\square\)

Dari Persamaan (3.81), apabila entropi meningkat, kontribusi distribusi *soft* meningkat secara relatif. Sebaliknya, ketika entropi menurun, kontribusi token hasil sampling menjadi relatif lebih besar.

### 3.8.3. Proposisi 3.3

**Proposisi 3.3.** Gumbel-Softmax mendekati sampling kategoris ketika temperatur \(\tau\) mendekati nol.

**Bukti.**

Misalkan

\[
a_i=\ell_i+g_i.
\]

Maka

\[
w_i(\tau)
=
\frac{\exp(a_i/\tau)}
{\sum_j\exp(a_j/\tau)}.
\tag{3.82}
\]

Misalkan \(k=\arg\max_i a_i\) merupakan indeks maksimum unik. Untuk \(i\neq k\),

\[
\frac{w_i(\tau)}{w_k(\tau)}
=
\exp
\left(
\frac{a_i-a_k}{\tau}
\right).
\tag{3.83}
\]

Karena

\[
a_i-a_k<0,
\]

maka ketika

\[
\tau\rightarrow0^+,
\]

diperoleh

\[
\exp
\left(
\frac{a_i-a_k}{\tau}
\right)
\rightarrow0.
\]

Akibatnya,

\[
w_k(\tau)\rightarrow1
\]

dan

\[
w_i(\tau)\rightarrow0,
\qquad i\neq k.
\]

Dengan demikian,

\[
\boxed{
w(\tau)
\rightarrow e_k
}
\qquad
\text{ketika }\tau\rightarrow0^+.
\tag{3.84}
\]

Karena \(k\) dipilih berdasarkan Gumbel-Max Trick, \(e_k\) memiliki distribusi kategoris yang sama dengan sampling dari \(p\). \(\square\)

---

## 3.9. Perbedaan Metode Raw dan Keluarga Relaksasi Diskret

Metode *raw* menggunakan

\[
\tilde z_{\mathrm{raw}}
=
hW_a,
\tag{3.85}
\]

sedangkan empat metode lainnya menggunakan

\[
\tilde z
=
\sum_{i=1}^{V}
w_iW_{\mathrm{in}}[i],
\qquad
w\in\Delta^{V-1}.
\tag{3.86}
\]

Dengan demikian, *raw* tidak menentukan representasi laten melalui bobot probabilitas pada simpleks.

Sebaliknya, metode *soft*, *sample*, *gumbel*, dan *moi* menentukan suatu titik pada simpleks probabilitas terlebih dahulu, kemudian memetakan titik tersebut ke ruang embedding melalui

\[
w
\mapsto
\sum_iw_iW_{\mathrm{in}}[i].
\tag{3.87}
\]

Bentuk umum tersebut digunakan sebagai dasar perbandingan dalam penelitian.

---

## 3.10. Rancangan Eksperimen

Rancangan eksperimen terdiri atas dua faktor utama, yaitu persamaan langkah laten dan medium komunikasi antar-agen.

### 3.10.1. Persamaan Langkah Laten

Persamaan langkah laten yang dibandingkan terdiri atas lima kondisi, yaitu:

1. *raw*;
2. *soft*;
3. *sample*;
4. *gumbel*; dan
5. *moi*.

Metode *raw* merupakan formulasi berbasis *realignment matrix*, sedangkan empat metode lainnya menggunakan representasi berbasis kombinasi embedding token.

### 3.10.2. Medium Komunikasi

Medium komunikasi yang digunakan terdiri atas

1. `text`;
2. `kv_and_text`; dan
3. `kv`.

Pada medium `text`, informasi antar-agen diteruskan melalui keluaran teks. Pada medium `kv`, informasi diteruskan menggunakan KV-cache. Pada medium `kv_and_text`, kedua bentuk informasi digunakan secara bersamaan.

Struktur agen yang digunakan adalah

\[
\mathrm{Planner}
\rightarrow
\mathrm{Critic}
\rightarrow
\mathrm{Refiner}
\rightarrow
\mathrm{Judger}.
\tag{3.88}
\]

Setiap kondisi menggunakan struktur agen yang sama.

---

## 3.11. Model dan Parameter Eksperimen

Model dasar yang digunakan dalam penelitian adalah Qwen3-8B. Pemilihan model dilakukan karena model tersebut memiliki matriks embedding masukan dan keluaran yang tidak *tied*, sehingga pemetaan \(W_a\) memberikan transformasi yang dapat diuji.

Parameter utama penelitian ditunjukkan pada Tabel 3.1.

| Parameter | Nilai |
|---|---:|
| Model | Qwen3-8B |
| Jumlah latent step \(m\) | 10 |
| Temperature distribusi \(T\) | 0,7 |
| Regularisasi ridge \(\lambda\) | \(10^{-5}\) |
| Temperature Gumbel \(\tau\) | 0,5 |
| Parameter MoI \(\beta\) | 1 |

---

## 3.12. Data dan Tugas Evaluasi

Evaluasi dilakukan pada beberapa tugas yang memiliki karakteristik keluaran berbeda. Tugas yang digunakan meliputi GSM8K, ARC-Challenge, HumanEval-Plus, serta tugas generasi ekspresi faktor simbolik.

GSM8K digunakan untuk mengevaluasi kemampuan penyelesaian soal matematika berbentuk cerita. ARC-Challenge digunakan untuk mengevaluasi kemampuan menjawab pertanyaan pilihan ganda. HumanEval-Plus digunakan untuk mengevaluasi kemampuan menghasilkan program yang memenuhi *unit test*. LatentMAS sendiri menggunakan GSM8K, ARC-Challenge, dan HumanEval-Plus sebagai bagian dari evaluasi kemampuan penalaran dan generasi kode.

Tugas faktor simbolik digunakan untuk mengevaluasi kemampuan sistem dalam menghasilkan ekspresi yang memiliki struktur simbolik tertentu. Domain tersebut berkaitan dengan formulasi *alpha mining* pada QuantaAlpha.

---

## 3.13. Prosedur Eksperimen

Tahapan eksperimen dilakukan sebagai berikut.

1. Menyiapkan model Qwen3-8B dan tokenizer.
2. Mengambil matriks \(W_{\mathrm{in}}\) dan \(W_{\mathrm{out}}\).
3. Menghitung matriks *realignment* \(W_a\) menggunakan Persamaan (3.21).
4. Menyiapkan dataset dan menentukan *subsample* menggunakan *seed* yang sama.
5. Menjalankan sistem multi-agen pada setiap kondisi eksperimen.
6. Pada kondisi laten, menghasilkan \(m=10\) langkah laten.
7. Menghitung \(z_t\) berdasarkan metode yang digunakan.
8. Menyimpan representasi tersebut dalam KV-cache melalui proses *forward* berikutnya.
9. Meneruskan informasi kepada agen berikutnya sesuai medium komunikasi.
10. Menghasilkan jawaban akhir pada agen *Judger*.
11. Menghitung metrik evaluasi.
12. Membandingkan hasil antar-kondisi menggunakan pengujian statistik berpasangan.

---

## 3.14. Metrik Evaluasi

Kinerja sistem dievaluasi berdasarkan beberapa metrik.

### 3.14.1. Accuracy

Untuk tugas dengan jawaban diskrit, akurasi dihitung sebagai

\[
\mathrm{Accuracy}
=
\frac{N_{\mathrm{correct}}}{N_{\mathrm{total}}}.
\tag{3.89}
\]

### 3.14.2. Format Rate

Format rate dihitung sebagai

\[
\mathrm{Format\ Rate}
=
\frac{N_{\mathrm{valid\ format}}}
{N_{\mathrm{total}}}.
\tag{3.90}
\]

Metrik tersebut digunakan untuk mengetahui proporsi keluaran yang dapat diproses menggunakan prosedur evaluasi yang telah ditentukan.

### 3.14.3. Symbolic Fidelity

Fidelitas simbolik dihitung berdasarkan kesesuaian simbol yang dihasilkan dengan simbol yang diperlukan pada keluaran. Ukuran yang digunakan dapat berupa *symbol recall*, *exact match*, serta tingkat kesalahan struktur ekspresi.

### 3.14.4. Token Usage

Jumlah token keluaran digunakan untuk mengukur efisiensi komunikasi berbasis teks. Untuk setiap soal,

\[
N_{\mathrm{token}}
=
\sum_{a=1}^{A}
N_{\mathrm{token},a},
\tag{3.91}
\]

dengan \(A\) menyatakan jumlah agen yang menghasilkan keluaran teks.

Pada komunikasi laten, langkah internal tidak dihitung sebagai token keluaran karena representasi tersebut tidak didekode menjadi token.

### 3.14.5. Waktu Inferensi

Waktu inferensi diukur sebagai waktu yang diperlukan sistem untuk menyelesaikan satu proses evaluasi dari masukan hingga keluaran akhir. Nilai yang digunakan adalah waktu eksekusi aktual pada lingkungan eksperimen.

---

## 3.15. Pengujian Statistik

Karena setiap metode dievaluasi pada kumpulan soal yang sama, perbandingan dilakukan secara berpasangan.

Untuk data biner, seperti benar atau salah, digunakan uji McNemar. Untuk metrik kontinu atau ordinal digunakan uji Wilcoxon *signed-rank*. Selain nilai \(p\), digunakan interval kepercayaan bootstrap sebesar 95% untuk menggambarkan ketidakpastian estimasi perbedaan.

Apabila dilakukan beberapa pengujian secara bersamaan, digunakan koreksi *multiple comparisons* untuk mengendalikan peningkatan peluang kesalahan tipe I.

---

## 3.16. Alur Penelitian

Secara keseluruhan, alur penelitian dapat diringkas sebagai berikut:

\[
\boxed{\text{Persiapan Model}}
\rightarrow
\boxed{\text{Perhitungan }W_a}
\rightarrow
\boxed{\text{Pembentukan }\Phi}
\rightarrow
\boxed{\text{Latent Reasoning}}
\rightarrow
\boxed{\text{Komunikasi Antar-Agen}}
\rightarrow
\boxed{\text{Evaluasi}}
\rightarrow
\boxed{\text{Analisis Statistik}}.
\tag{3.92}
\]

Pada tahap pembentukan \(\Phi\), lima metode yang telah dijelaskan digunakan secara bergantian. Untuk metode *soft*, *sample*, *gumbel*, dan *moi*, representasi laten dibentuk melalui bobot pada simpleks probabilitas. Sementara itu, metode *raw* menggunakan matriks *realignment* LatentMAS.

Hasil dari masing-masing kondisi kemudian dibandingkan berdasarkan akurasi, format keluaran, fidelitas simbolik, jumlah token, dan waktu inferensi.