# BAB II
# LANDASAN TEORI

Bab ini menyajikan konsep dan teori dasar yang menjadi landasan penelitian. Pembahasan meliputi probabilitas, distribusi relevan, teori informasi, inferensi Bayesian, aljabar linear, pembelajaran mesin, model bahasa, Transformer, penalaran dengan *Large Language Model* (LLM), *Soft Thinking*, *Mixture of Inputs* (MoI), *Stochastic Soft Thinking*, sistem multi-agen, LatentMAS, hubungan matematis antar metode, serta metode pengujian statistik. Setiap subbab dilengkapi dengan definisi, notasi, serta penurunan rumus utama yang digunakan dalam penelitian.

---

## 2.1 Probabilitas dan Variabel Acak

### 2.1.1 Ruang Sampel dan *Event*
Ruang sampel \(\Omega\) adalah himpunan semua hasil yang mungkin dari suatu percobaan acak. *Event* \(A\) adalah subset dari \(\Omega\), \(A \subseteq \Omega\). Probabilitas \(P(A)\) memenuhi aksioma: \(P(\Omega)=1\), \(0 \le P(A) \le 1\), dan untuk *event* saling lepas berlaku aditivitas.

### 2.1.2 Variabel Acak
Variabel acak \(X\) adalah fungsi \(X:\Omega \rightarrow \mathbb{R}\) yang memetakan setiap hasil ke bilangan real. Variabel acak dapat bersifat diskrit atau kontinu.

### 2.1.3 Distribusi Probabilitas
Distribusi probabilitas variabel acak \(X\) menggambarkan peluang \(X\) mengambil nilai-nilai tertentu. Untuk kasus diskrit dinyatakan dengan *probability mass function* (PMF), untuk kontinu dengan *probability density function* (PDF).

### 2.1.4 *Probability Mass Function*
Untuk variabel diskrit, PMF \(p(x)\) didefinisikan sebagai
\[
p(x) = P(X = x), \qquad \sum_{x} p(x) = 1.
\]

### 2.1.5 *Probability Density Function*
Untuk variabel kontinu, PDF \(f(x)\) memenuhi \(f(x) \ge 0\) dan \(\int_{-\infty}^{\infty} f(x)\,dx = 1\). Probabilitas pada interval \([a,b]\) dihitung sebagai \(\int_a^b f(x)\,dx\).

### 2.1.6 *Conditional Probability*
Probabilitas bersyarat \(A\) diberikan \(B\) didefinisikan sebagai
\[
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0.
\]

### 2.1.7 *Joint Probability*
Probabilitas gabungan dua variabel acak \(X\) dan \(Y\) dinyatakan dengan \(P(X=x, Y=y)\) (diskrit) atau \(f_{X,Y}(x,y)\) (kontinu).

### 2.1.8 *Marginal Probability*
Distribusi marginal diperoleh dengan menjumlahkan (diskrit) atau mengintegralkan (kontinu) distribusi gabungan:
\[
P(X=x) = \sum_y P(X=x, Y=y), \qquad f_X(x) = \int f_{X,Y}(x,y)\,dy.
\]

### 2.1.9 *Independence*
Dua variabel acak \(X\) dan \(Y\) dikatakan saling bebas jika
\[
P(X=x, Y=y) = P(X=x)P(Y=y) \quad \text{atau} \quad f_{X,Y}(x,y)=f_X(x)f_Y(y).
\]

### 2.1.10 *Expectation*
Nilai harapan \(E[X]\) dihitung sebagai
\[
E[X] = \sum_x x\,p(x) \quad \text{atau} \quad E[X] = \int x f(x)\,dx.
\]

### 2.1.11 *Variance* dan *Covariance*
Variansi \( \text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2\).
Kovariansi \( \text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])]\).

---

## 2.2 Distribusi Probabilitas yang Relevan

### 2.2.1 Bernoulli
Variabel acak \(X \sim \text{Bernoulli}(p)\) mengambil nilai 1 dengan peluang \(p\) dan 0 dengan peluang \(1-p\). PMF: \(P(X=x) = p^x (1-p)^{1-x}, \; x \in \{0,1\}\).

### 2.2.2 Categorical
Distribusi kategoris dengan \(V\) kategori memiliki parameter \(p = (p_1,\dots,p_V)\), \(p_i \ge 0\), \(\sum_i p_i = 1\). PMF: \(P(X = i) = p_i\). Vektor indikator \(y\) one-hot dengan \(y_i = 1\) dan lainnya 0 dapat dinyatakan sebagai \(Y \sim \text{Categorical}(p)\).

### 2.2.3 Multinomial
Distribusi Multinomial\(\text{Mult}(n, p)\) menghitung frekuensi \(x_i\) dari setiap kategori dalam \(n\) percobaan independen. PMF:
\[
P(X_1=x_1,\dots,X_V=x_V) = \frac{n!}{x_1!\cdots x_V!} \prod_{i=1}^V p_i^{x_i}, \quad \sum_i x_i = n.
\]

### 2.2.4 Gaussian
Distribusi Normal \(\mathcal{N}(\mu, \sigma^2)\) dengan PDF
\[
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
\]

### 2.2.5 Gumbel
Variabel acak \(G \sim \text{Gumbel}(\mu, \beta)\) memiliki fungsi distribusi kumulatif
\[
F_G(g) = \exp\left[-\exp\left(-\frac{g-\mu}{\beta}\right)\right], \quad \beta>0.
\]
Distribusi Gumbel standar (\(\mu=0,\beta=1\)) mempunyai sifat penting untuk *Gumbel-Max Trick*.

**Gumbel-Max Trick.** Diberikan probabilitas kategori \(p_i>0\), \(\sum_i p_i = 1\). Bangkitkan \(g_i \overset{\text{iid}}{\sim} \text{Gumbel}(0,1)\). Maka
\[
Y = \arg\max_i \left(\log p_i + g_i\right) \sim \text{Categorical}(p).
\]
*Bukti.* Misalkan \(k\) adalah indeks tertentu. \(P(Y=k) = P(\log p_k + g_k \ge \log p_i + g_i \; \forall i \neq k)\). Kondisikan pada \(g_k\), peluang tersebut adalah \(\prod_{i \neq k} P(g_i \le \log p_k - \log p_i + g_k)\). Untuk \(G \sim \text{Gumbel}(0,1)\), berlaku \(P(G \le t) = \exp(-e^{-t})\). Maka
\[
P(Y=k \mid g_k) = \prod_{i \neq k} \exp\left(-e^{-(\log p_k - \log p_i + g_k)}\right) = \exp\left(-\sum_{i \neq k} \frac{p_i}{p_k} e^{-g_k}\right).
\]
Integralkan terhadap distribusi \(g_k\) dengan PDF \(f(g_k) = e^{-g_k} \exp(-e^{-g_k})\):
\[
P(Y=k) = \int_{-\infty}^\infty \exp\left(-\frac{1-p_k}{p_k} e^{-g_k}\right) e^{-g_k} e^{-e^{-g_k}}\, dg_k.
\]
Substitusi \(u = e^{-g_k}\), \(du = -e^{-g_k} dg_k\) menghasilkan
\[
P(Y=k) = \int_0^\infty \exp\left(-\frac{u}{p_k}\right) u \, du = p_k.
\]
Dengan demikian, Gumbel-Max menghasilkan sampel sesuai distribusi \(p\). ∎

### 2.2.6 Dirichlet
Distribusi Dirichlet dengan parameter konsentrasi \(\alpha = (\alpha_1,\dots,\alpha_V)\), \(\alpha_i>0\), didefinisikan pada simpleks \(\Delta^{V-1}\). PDF:
\[
f(w;\alpha) = \frac{1}{B(\alpha)} \prod_{i=1}^V w_i^{\alpha_i - 1}, \quad w_i\ge0,\; \sum_i w_i = 1,
\]
dengan \(B(\alpha)\) fungsi beta multinomial. Nilai harapan: \(E[W_i] = \alpha_i / \sum_j \alpha_j\). Sifat konjugat terhadap Multinomial dibahas pada 2.4.6.

---

## 2.3 Teori Informasi

### 2.3.1 *Information*
Informasi diri (*self-information*) dari *event* dengan probabilitas \(p\) didefinisikan sebagai \(I(p) = -\log p\).

### 2.3.2 *Entropy*
Entropi distribusi diskrit \(p\) adalah \(H(p) = -\sum_i p_i \log p_i\), mengukur ketidakpastian rata-rata.

### 2.3.3 *Normalized Entropy*
Entropi ternormalisasi terhadap nilai maksimum \(\log V\):
\[
H = -\frac{1}{\log V} \sum_i p_i \log p_i, \quad 0 \le H \le 1.
\]

### 2.3.4 *Cross-Entropy*
*Cross-entropy* antara distribusi sebenarnya \(p\) dan distribusi aproksimasi \(q\) didefinisikan sebagai \(H(p,q) = -\sum_i p_i \log q_i\).

### 2.3.5 *KL Divergence*
Divergensi Kullback–Leibler mengukur perbedaan dua distribusi:
\[
D_{\mathrm{KL}}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}.
\]

### 2.3.6 *Jensen-Shannon Divergence*
Divergensi Jensen-Shannon merupakan simetrisasi KL:
\[
\mathrm{JSD}(p \| q) = \frac{1}{2} D_{\mathrm{KL}}(p \| m) + \frac{1}{2} D_{\mathrm{KL}}(q \| m), \quad m = \frac{p+q}{2}.
\]

---

## 2.4 *Bayesian Inference*

### 2.4.1 *Prior*
Distribusi *prior* \(P(\theta)\) menyatakan keyakinan awal tentang parameter \(\theta\).

### 2.4.2 *Likelihood*
*Likelihood* \(P(\mathcal{D}|\theta)\) adalah probabilitas data \(\mathcal{D}\) jika parameter adalah \(\theta\).

### 2.4.3 *Evidence*
*Evidence* atau *marginal likelihood* \(P(\mathcal{D}) = \int P(\mathcal{D}|\theta) P(\theta) d\theta\).

### 2.4.4 *Posterior*
Distribusi *posterior* diperoleh melalui teorema Bayes:
\[
P(\theta|\mathcal{D}) = \frac{P(\mathcal{D}|\theta) P(\theta)}{P(\mathcal{D})}.
\]

### 2.4.5 *Bayesian Updating*
Pembaruan pengetahuan dilakukan dengan menggunakan *posterior* sebagai *prior* baru ketika data tambahan diterima.

### 2.4.6 *Conjugate Prior*
Distribusi *prior* disebut konjugat terhadap *likelihood* jika *posterior* berada dalam keluarga distribusi yang sama. Contoh utama: distribusi Dirichlet adalah konjugat bagi Multinomial.

Misalkan \(w \sim \text{Dir}(\alpha)\) dan observasi \(c = (c_1,\dots,c_V)\) dengan \(n = \sum_i c_i\) berasal dari \(\text{Mult}(n, w)\). Maka
\[
P(c|w) \propto \prod_i w_i^{c_i}.
\]
*Posterior*:
\[
P(w|c) \propto \prod_i w_i^{\alpha_i + c_i - 1} \sim \text{Dir}(\alpha + c).
\]

### 2.4.7 *Posterior Predictive Distribution*
Distribusi prediktif *posterior* untuk observasi baru \(\tilde{x}\) adalah \(P(\tilde{x}|\mathcal{D}) = \int P(\tilde{x}|\theta) P(\theta|\mathcal{D}) d\theta\).

---

## 2.5 *Linear Algebra* dan Representasi Vektor

### 2.5.1 *Scalar, Vector, Matrix*
Skalar \(a\), vektor \(\mathbf{v} \in \mathbb{R}^n\), matriks \(A \in \mathbb{R}^{m \times n}\).

### 2.5.2 *Dot Product*
Hasil kali titik \(\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i = \mathbf{u}^\top \mathbf{v}\).

### 2.5.3 *Matrix Multiplication*
Perkalian matriks \(C = AB\) dengan \(C_{ij} = \sum_k A_{ik} B_{kj}\).

### 2.5.4 *Norm*
Norma vektor \(\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}\). Norma Frobenius matriks \(A\):
\[
\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)}.
\]

### 2.5.5 *Linear Combination*
Kombinasi linear vektor \(\{\mathbf{v}_1,\dots,\mathbf{v}_K\}\) adalah \(\sum_k c_k \mathbf{v}_k\) dengan koefisien \(c_k \in \mathbb{R}\).

### 2.5.6 *Convex Combination*
Kombinasi konveks mensyaratkan \(c_k \ge 0\) dan \(\sum_k c_k = 1\). Himpunan semua kombinasi konveks dari titik-titik tersebut membentuk *convex hull*. Simpleks probabilitas \(\Delta^{V-1}\) adalah himpunan semua vektor \(w\) dengan \(w_i \ge 0,\ \sum_i w_i = 1\), sehingga setiap titik di dalamnya merupakan kombinasi konveks dari basis one-hot.

### 2.5.7 *Vector Space*
Ruang vektor dilengkapi dengan operasi penjumlahan dan perkalian skalar yang memenuhi aksioma tertentu.

### 2.5.8 *Embedding Space*
Ruang embedding adalah ruang vektor kontinu berdimensi \(d\) tempat representasi token (kata) dipetakan melalui matriks embedding \(W_{\mathrm{in}}\). Setiap baris \(W_{\mathrm{in}}[i]\) adalah vektor embedding token ke-\(i\).

---

## 2.6 *Machine Learning* dan *Neural Network*

### 2.6.1 Model Parametrik
Model parametrik dinyatakan sebagai fungsi \(f_\theta\) dengan parameter \(\theta\) yang dipelajari dari data.

### 2.6.2 Parameter dan *Hyperparameter*
Parameter dipelajari selama pelatihan; *hyperparameter* ditentukan sebelum pelatihan (misalnya laju pembelajaran, koefisien regularisasi).

### 2.6.3 *Forward Pass*
Komputasi maju dari masukan hingga keluaran melalui lapisan-lapisan jaringan.

### 2.6.4 *Loss Function*
Fungsi kerugian \(L(\theta)\) mengukur kesalahan model; diminimalkan saat pelatihan.

### 2.6.5 *Training vs Inference*
Pelatihan menyesuaikan \(\theta\) menggunakan data; inferensi menggunakan model tetap untuk menghasilkan prediksi.

### 2.6.6 *Autoregressive Modeling*
Model autoregresif memprediksi token berikutnya berdasarkan token-token sebelumnya:
\[
P(x_1, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t}).
\]

---

## 2.7 *Language Model*

### 2.7.1 Token dan *Vocabulary*
Teks dipecah menjadi token (kata, subkata). *Vocabulary* \(V\) adalah himpunan semua token yang dikenal model.

### 2.7.2 *Tokenization*
Proses pemecahan teks menjadi deretan token sesuai *vocabulary*.

### 2.7.3 *Next-Token Prediction*
Model bahasa menghasilkan distribusi \(P(x_t \mid x_{<t})\) untuk token berikutnya.

### 2.7.4 *Language Model Probability*
Probabilitas keseluruhan teks diberikan oleh rantai autoregresif di atas.

### 2.7.5 *Logits*
*Hidden state* terakhir \(h \in \mathbb{R}^d\) dipetakan menjadi vektor logit \(\ell = W_{\mathrm{out}} h \in \mathbb{R}^V\).

### 2.7.6 *Softmax*
Distribusi token diperoleh dengan fungsi *softmax*:
\[
p_i = \frac{e^{\ell_i}}{\sum_j e^{\ell_j}}, \quad p = \operatorname{softmax}(\ell).
\]

### 2.7.7 *Temperature*
Temperature \(T>0\) digunakan untuk mengontrol ketajaman distribusi:
\[
p_i = \frac{e^{\ell_i / T}}{\sum_j e^{\ell_j / T}}.
\]
\(T \to 0\) membuat distribusi mendekati one-hot pada argmax; \(T \to \infty\) mendekati seragam.

### 2.7.8 *Greedy Decoding*
Memilih token dengan probabilitas tertinggi setiap langkah (\(T=0\)).

### 2.7.9 *Sampling*
Mengambil sampel token dari distribusi \(p\) dengan \(T\) tertentu.

### 2.7.10 *Top-k*
Hanya mempertimbangkan \(k\) token dengan probabilitas tertinggi; probabilitas lainnya dinolkan lalu dinormalisasi ulang.

### 2.7.11 *Top-p* (*Nucleus Sampling*)
Memilih himpunan token terkecil yang massa probabilitas kumulatifnya \(\ge p\); sisanya dinolkan.

---

## 2.8 *Transformer*

### 2.8.1 Arsitektur Transformer
Transformer terdiri dari lapisan *self-attention* dan *feed-forward network* yang disusun secara berulang, dengan normalisasi dan koneksi residual.

### 2.8.2 *Embedding*
Token masukan dipetakan menjadi vektor embedding melalui \(W_{\mathrm{in}}\).

### 2.8.3 *Positional Encoding*
Informasi posisi ditambahkan ke embedding agar model mengenali urutan token.

### 2.8.4 *Self-Attention*
Mekanisme atensi yang mengaitkan setiap token dengan seluruh token dalam sekuens.

### 2.8.5 *Query, Key, Value*
Setiap token menghasilkan vektor \(Q, K, V\) melalui proyeksi linear dari *hidden state*.

### 2.8.6 *Scaled Dot-Product Attention*
\[
\text{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
\]

### 2.8.7 *Multi-Head Attention*
Mekanisme atensi dijalankan secara paralel dengan beberapa *head*, lalu hasilnya digabung.

### 2.8.8 *Feed-Forward Network*
Setiap posisi diproses dengan MLP dua lapis (umumnya dengan aktivasi nonlinier).

### 2.8.9 *Residual Connection*
Menambahkan masukan lapisan ke keluarannya: \(\text{Output} = \text{Layer}(x) + x\).

### 2.8.10 *Layer Normalization*
Normalisasi pada dimensi fitur untuk menstabilkan pelatihan.

### 2.8.11 *Decoder-only Transformer*
Arsitektur yang hanya menggunakan dekoder Transformer; kausal mask memastikan prediksi hanya bergantung pada token sebelumnya.

### 2.8.12 *Causal Masking*
Masker segitiga bawah mencegah setiap token memperhatikan token di depannya pada *self-attention*.

### 2.8.13 *KV Cache*
*Key* dan *Value* dari langkah sebelumnya disimpan dalam cache untuk menghindari komputasi ulang saat pembangkitan autoregresif. Untuk lapisan \(l\), KV-cache adalah pasangan \((K^{(l)}, V^{(l)})\).

---

## 2.9 *Large Language Model* dan *Reasoning*

### 2.9.1 LLM
Model bahasa berskala besar dengan miliaran parameter, dilatih pada korpus teks masif.

### 2.9.2 *Chain-of-Thought*
Teknik penalaran dengan menghasilkan langkah-langkah perantara dalam bahasa alami sebelum jawaban akhir.

### 2.9.3 *Discrete Token Reasoning*
Penalaran yang setiap langkahnya direpresentasikan sebagai token diskrit yang didekode.

### 2.9.4 *Continuous Reasoning*
Penalaran yang menggunakan representasi kontinu (misalnya *hidden state* atau embedding) tanpa menghasilkan token eksplisit.

### 2.9.5 *Latent Representation*
Representasi tersembunyi dalam ruang kontinu yang menyandikan informasi konteks.

### 2.9.6 *Hidden State*
Vektor \(h_t \in \mathbb{R}^d\) pada lapisan terakhir Transformer setelah memproses token ke-\(t\).

### 2.9.7 *Latent Thought*
Proses berpikir dalam ruang laten kontinu, di mana setiap langkah menggunakan representasi laten sebagai masukan.

---

## 2.10 *Soft Thinking*

### 2.10.1 *Discrete Token Thinking*
Model konvensional memilih satu token diskrit pada setiap langkah; informasi distribusi penuh dibuang.

### 2.10.2 *Soft Token*
*Soft token* mempertahankan seluruh distribusi probabilitas \(p = \operatorname{softmax}(\ell/T)\) di atas *vocabulary*. Vektor \(p\) berada pada simpleks probabilitas \(\Delta^{V-1}\).

### 2.10.3 *Soft Input*
Alih-alih memberikan embedding satu token, *soft input* menggunakan ekspektasi embedding terhadap distribusi \(p\):
\[
\tilde z_{\mathrm{soft}} = \sum_{i=1}^V p_i \, W_{\mathrm{in}}[i].
\]
Karena \(p \in \Delta^{V-1}\), \(\tilde z_{\mathrm{soft}}\) merupakan kombinasi konveks dari baris-baris \(W_{\mathrm{in}}\).

### 2.10.4 *Probability Simplex*
Simpleks \(\Delta^{V-1}\) adalah himpunan titik dengan koordinat nonnegatif yang berjumlah satu.

### 2.10.5 *Weighted Embedding*
Setiap token diberi bobot sesuai probabilitasnya untuk membentuk representasi kontinu yang kaya informasi.

### 2.10.6 *Top-k/Top-p Truncation*
Distribusi dapat dipotong dengan Top-k atau Top-p sebelum menghitung *soft token*.

### 2.10.7 *Greedy Pitfall*
Pendekodean *greedy* dapat mengabaikan alternatif bernilai tinggi, menyebabkan teks repetitif atau kurang kreatif.

---

## 2.11 *Mixture of Inputs*

### 2.11.1 Motivasi MoI
MoI menggabungkan token hasil sampling dengan informasi distribusi penuh untuk mendapatkan representasi yang seimbang antara eksplorasi dan eksploitasi ketidakpastian.

### 2.11.2 *Entropy-scaled Prior*
Misalkan \(p\) adalah distribusi token. Entropi ternormalisasi \(H = -\frac{1}{\log V}\sum_i p_i \log p_i \in [0,1]\) mengukur ketidakpastian.

### 2.11.3 *Dirichlet Prior*
Parameter prior Dirichlet untuk bobot campuran \(w\) ditetapkan sebagai \(\alpha = H p\). Jumlah konsentrasi \(\sum_i \alpha_i = H\) mencerminkan tingkat kepercayaan pada distribusi \(p\): ketidakpastian tinggi memberikan konsentrasi lebih besar.

### 2.11.4 *Multinomial Observation*
Satu token sampel \(Y \sim \text{Categorical}(p)\) dianggap sebagai observasi *pseudo-count* \(c = (\beta + 1 - H) e_Y\), dengan \(e_Y\) vektor one-hot dan \(\beta \ge 0\) parameter *pseudo-count*.

### 2.11.5 *Posterior Distribution*
Karena Dirichlet konjugat terhadap Multinomial, *posterior* setelah mengamati \(c\) adalah
\[
w \mid y \sim \text{Dir}\big(\alpha + c\big) = \text{Dir}\big(Hp + (\beta+1-H)e_Y\big).
\]

### 2.11.6 *Posterior Mean*
Nilai harapan *posterior* untuk setiap komponen \(i\) adalah
\[
E[w_i \mid y] = \frac{\alpha_i + c_i}{\sum_j (\alpha_j + c_j)}.
\]
Jumlah parameter *posterior*: \(\sum_j (\alpha_j + c_j) = H + (\beta+1-H) = \beta+1\). Sehingga
\[
\boxed{ w_i = \frac{H p_i + (\beta+1-H) (e_Y)_i}{\beta+1} }.
\]
Dalam notasi vektor,
\[
\boxed{ w_{\mathrm{moi}} = \frac{H p + (\beta+1-H) e_Y}{\beta+1} }.
\tag{2.1}
\]

### 2.11.7 *Mixing Weight*
Persamaan (2.1) menunjukkan bahwa \(w_{\mathrm{moi}}\) adalah kombinasi konveks dari distribusi *soft* \(p\) dan sampel one-hot \(e_Y\):
\[
w_{\mathrm{moi}} = \lambda p + (1-\lambda) e_Y, \quad \lambda = \frac{H}{\beta+1}.
\]
Karena \(0\le H \le 1\) dan \(\beta\ge 0\), maka \(0\le \lambda \le 1\). Ini membuktikan bahwa MoI menginterpolasi antara *soft* dan *discrete sample*.

### 2.11.8 *Mixture Embedding*
Representasi embedding MoI diperoleh sebagai kombinasi konveks embedding token:
\[
\tilde z_{\mathrm{moi}} = \sum_i w_{\mathrm{moi},i} W_{\mathrm{in}}[i] = \lambda \tilde z_{\mathrm{soft}} + (1-\lambda) W_{\mathrm{in}}[Y].
\tag{2.2}
\]

---

## 2.12 *Stochastic Soft Thinking*

### 2.12.1 *Randomness* dan *Exploration*
Penyisipan keacakan membantu eksplorasi representasi laten, menghindari jalan buntu deterministik.

### 2.12.2 *Stochastic Soft Token*
Token lunak yang diperoleh dari distribusi acak yang parameterisasinya berasal dari logit model.

### 2.12.3 *Dirichlet Sampling*
Membangkitkan titik pada simpleks dengan distribusi Dirichlet.

### 2.12.4 *Gumbel Distribution*
Telah dijelaskan pada 2.2.5. Sifat Gumbel-Max digunakan untuk sampling kategoris terdiferensialkan.

### 2.12.5 *Gumbel-Max Trick*
Sudah diuraikan beserta buktinya di 2.2.5: \(\arg\max_i(\log p_i + g_i) \sim \text{Categorical}(p)\).

### 2.12.6 *Gumbel-Softmax*
Operasi \(\arg\max\) tidak terdiferensialkan, maka digantikan oleh *softmax* dengan temperatur \(\tau>0\):
\[
w_i^{\mathrm{gumbel}} = \frac{\exp((\log p_i + g_i)/\tau)}{\sum_j \exp((\log p_j + g_j)/\tau)}.
\tag{2.3}
\]
Dengan \(p = \operatorname{softmax}(\ell)\), bentuk di atas ekuivalen dengan \(\operatorname{softmax}((\ell + g)/\tau)\) karena \(\log p_i = \ell_i - \log\sum_j e^{\ell_j}\), dan konstanta penormalan saling menghilangkan.

**Konvergensi ke one-hot.** Misalkan \(k = \arg\max_i (\log p_i + g_i)\) unik. Maka untuk \(i \neq k\),
\[
\frac{w_i}{w_k} = \exp\left( \frac{(\log p_i + g_i) - (\log p_k + g_k)}{\tau} \right) \xrightarrow{\tau \to 0^+} 0.
\]
Akibatnya \(w_k \to 1\) dan \(w_i \to 0\) untuk \(i \neq k\). Jadi \(w^{\mathrm{gumbel}} \to e_k\), yang secara distribusi setara dengan sampel dari \(p\) berdasarkan Gumbel-Max. Dengan demikian Gumbel-Softmax menghasilkan aproksimasi kontinu dari sampling kategoris.

### 2.12.7 *Temperature*
Temperatur \(\tau\) mengontrol tingkat *softness*: kecil mendekati diskrit, besar mendekati seragam.

### 2.12.8 *Softness vs Randomness*
*Softness* (kehalusan distribusi) dan *randomness* (keacakan) dapat divariasikan secara independen melalui \(T\) dan \(\tau\).

### 2.12.9 *Luce’s Choice Axiom*
Aksioma pemilihan Luce menyatakan bahwa rasio peluang dua alternatif tidak bergantung pada alternatif lain; dipenuhi oleh *softmax*.

### 2.12.10 *Sampling Without Replacement*
Teknik mengambil beberapa token tanpa pengembalian, relevan untuk eksplorasi.

### 2.12.11 *Test-time Scaling*
Meningkatkan komputasi saat inferensi (misalnya jumlah langkah laten) untuk meningkatkan performa.

### 2.12.12 *Pass@k*
Metrik keberhasilan jika jawaban benar muncul dalam \(k\) sampel pertama.

---

## 2.13 *Multi-Agent System*

### 2.13.1 *Agent*
Entitas yang memproses informasi, memiliki memori, dan menghasilkan aksi (teks atau representasi laten).

### 2.13.2 *Multi-Agent System* (MAS)
Sistem yang terdiri dari beberapa agen yang berinteraksi untuk menyelesaikan tugas.

### 2.13.3 *Sequential MAS*
Agen bekerja secara berurutan; keluaran agen sebelumnya menjadi masukan agen berikutnya.

### 2.13.4 *Hierarchical MAS*
Agen disusun dalam hierarki, misalnya perencana, kritikus, penyuling.

### 2.13.5 *Agent Communication*
Pertukaran informasi antar agen melalui medium tertentu (teks, representasi laten, atau kombinasi).

### 2.13.6 *Text-based Communication*
Agen berkomunikasi dengan menghasilkan teks dalam bahasa alami.

### 2.13.7 *Latent Communication*
Agen berkomunikasi dengan mentransfer representasi laten (misalnya *hidden state* atau KV-cache) tanpa dekoding ke teks.

---

## 2.14 LatentMAS

### 2.14.1 *Autoregressive Latent Thought*
LatentMAS menjalankan langkah penalaran laten secara autoregresif: setiap langkah menghasilkan vektor laten yang diumpankan kembali sebagai masukan.

### 2.14.2 *Hidden-State Recurrence*
Misalkan \(h_t\) adalah *hidden state* setelah langkah ke-\(t\). Fungsi transformasi \(\Phi\) memetakan \(h_t\) menjadi representasi laten \(z_t = \Phi(h_t)\) yang digunakan sebagai *input embedding* langkah berikutnya:
\[
h_{t+1} = f_\theta(z_t \mid \mathrm{KV}_{1:t}).
\]

### 2.14.3 *Latent Working Memory*
Seluruh rangkaian langkah laten membentuk memori kerja laten yang disimpan dalam KV-cache.

### 2.14.4 *KV-cache Transfer*
KV-cache dari agen sebelumnya dapat diteruskan ke agen berikutnya sebagai memori kerja, sehingga komunikasi laten terjadi tanpa teks.

### 2.14.5 *Input-Output Alignment*
Agar *hidden state* dapat digunakan sebagai *input embedding*, diperlukan pemetaan dari ruang keluaran (logit) ke ruang embedding masukan.

### 2.14.6 *Alignment Matrix*
Dicari matriks \(W_a \in \mathbb{R}^{d \times d}\) sehingga \(h W_a\) mendekati representasi yang sesuai di ruang embedding.

### 2.14.7 *Ridge Regression*
Masalah *alignment* dirumuskan sebagai regresi linear dengan regularisasi *ridge*:
\[
J(W_a) = \| W_{\mathrm{out}} W_a - W_{\mathrm{in}} \|_F^2 + \lambda \|W_a\|_F^2, \quad \lambda>0.
\tag{2.4}
\]
**Penurunan solusi.** Turunan terhadap \(W_a\):
\[
\frac{\partial J}{\partial W_a} = 2W_{\mathrm{out}}^\top (W_{\mathrm{out}} W_a - W_{\mathrm{in}}) + 2\lambda W_a.
\]
Menyamakan dengan nol menghasilkan persamaan normal teregularisasi:
\[
(W_{\mathrm{out}}^\top W_{\mathrm{out}} + \lambda I) W_a = W_{\mathrm{out}}^\top W_{\mathrm{in}}.
\]
Dengan asumsi matriks dalam kurung invertibel, solusi *ridge*:
\[
\boxed{ W_a = \big( W_{\mathrm{out}}^\top W_{\mathrm{out}} + \lambda I \big)^{-1} W_{\mathrm{out}}^\top W_{\mathrm{in}} }.
\tag{2.5}
\]
Matriks ini meminimalkan kesalahan rekonstruksi embedding dari *hidden state*.

### 2.14.8 *Wasserstein Distance*
Digunakan untuk mengukur kesamaan distribusi laten dalam beberapa analisis teoretis.

### 2.14.9 *Expressiveness Theorem*
Teorema dalam LatentMAS yang menyatakan bahwa representasi laten mampu mengekspresikan sembarang distribusi token.

### 2.14.10 *Lossless Communication Theorem*
Teorema yang menjamin bahwa komunikasi berbasis KV-cache tidak kehilangan informasi dibandingkan komunikasi teks.

### 2.14.11 *Complexity Analysis*
Analisis kompleksitas komputasi menunjukkan penghematan token dan inferensi lebih efisien dibandingkan rantai pemikiran tekstual.

**Representasi Laten LatentMAS.** Setelah memperoleh \(W_a\), *hidden state* diproyeksikan:
\[
\tilde z_{\mathrm{raw}} = h W_a.
\]
Untuk menjaga magnitudo setara dengan embedding, dilakukan normalisasi dengan rata-rata norma embedding \(\rho = \frac{1}{V}\sum_i \|W_{\mathrm{in}}[i]\|_2\):
\[
\boxed{ z_{\mathrm{raw}} = \rho \frac{h W_a}{\|h W_a\|_2} }.
\tag{2.6}
\]
Inilah representasi langkah laten yang digunakan dalam metode *raw*.

---

## 2.15 Konsep Matematis yang Menghubungkan Ketiga Metode

### 2.15.1 Discrete → Probability Distribution
Token diskrit yang dipilih secara *greedy* atau *sampling* mengabaikan distribusi penuh \(p\). Mempertahankan \(p\) memungkinkan representasi yang lebih informatif.

### 2.15.2 Probability Distribution → Embedding
Distribusi \(p \in \Delta^{V-1}\) dipetakan ke ruang embedding melalui kombinasi konveks \(\tilde z = \sum_i p_i W_{\mathrm{in}}[i]\). Ini menghasilkan titik di *convex hull* embedding token.

### 2.15.3 Embedding → Hidden Representation
Embedding tersebut diumpankan ke Transformer untuk menghasilkan *hidden state* baru.

### 2.15.4 Hidden Representation → Latent Thought
*Hidden state* diproses oleh \(\Phi\) menjadi representasi laten \(z_t\), membentuk siklus penalaran kontinu.

### 2.15.5 Stochasticity → Exploration
Penambahan keacakan melalui Gumbel atau sampling Dirichlet menciptakan eksplorasi dalam ruang representasi.

### 2.15.6 Latent Representation → Agent Communication
Representasi laten (KV-cache atau embedding) dapat ditransfer antar agen, memungkinkan komunikasi efisien tanpa dekoding teks.

**Keterkaitan metode:** Semua metode *soft, sample, gumbel, moi* menghasilkan representasi sebagai kombinasi konveks dari embedding token, karena vektor bobot \(w\) selalu berada pada simpleks \(\Delta^{V-1}\). Metode *raw* (LatentMAS) menggunakan pemetaan linear langsung dari *hidden state* tanpa melalui simpleks. Dengan demikian, perbedaan fundamental terletak pada cara menghasilkan bobot \(w\) (untuk keluarga relaksasi diskret) atau transformasi linear \(W_a\) (untuk *raw*). Formulasi ini menyatukan pandangan teoretis terhadap langkah laten.

---

## 2.16 Pengujian Statistik

Untuk membandingkan performa metode secara berpasangan pada data yang sama, digunakan beberapa uji statistik yang sesuai dengan jenis data dan desain eksperimen.

### 2.16.1 Uji McNemar
Uji McNemar digunakan untuk data biner berpasangan (benar/salah) dari dua metode pada sampel yang identik. Misalkan \(n_{01}\) = banyaknya soal yang dijawab benar oleh metode A tetapi salah oleh B, dan \(n_{10}\) sebaliknya. Statistik uji (dengan koreksi kontinuitas):
\[
\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} \sim \chi^2(1) \text{ di bawah } H_0.
\]
Nilai \(p\) kecil menunjukkan perbedaan signifikan.

### 2.16.2 Uji Wilcoxon *Signed-Rank*
Untuk metrik kontinu atau ordinal (misalnya skor, jumlah token), uji Wilcoxon Signed-Rank menguji median selisih nol. Langkah: hitung selisih \(d_i\) antara dua metode untuk setiap subjek, buang nol, peringkat nilai mutlaknya, lalu jumlahkan peringkat bertanda:
\[
W = \sum_{i=1}^n \operatorname{sgn}(d_i) R_i.
\]
Di bawah \(H_0\), distribusi \(W\) simetris di sekitar nol; untuk sampel besar digunakan aproksimasi normal.

### 2.16.3 Interval Kepercayaan *Bootstrap*
Untuk memperoleh interval kepercayaan perbedaan metrik tanpa asumsi distribusi, digunakan *bootstrap* persentil: ambil \(B\) sampel ulang dengan pengembalian, hitung statistik \(\hat{\theta}^*_b\), lalu interval kepercayaan \(100(1-\alpha)\%\) adalah persentil \(\alpha/2\) dan \(1-\alpha/2\) dari distribusi bootstrap.

### 2.16.4 Koreksi Perbandingan Berganda
Jika banyak uji dilakukan bersamaan, probabilitas kesalahan tipe I meningkat. Koreksi Bonferroni membagi tingkat signifikansi \(\alpha\) dengan jumlah uji \(m\): \(\alpha_{\mathrm{adj}} = \alpha / m\). Alternatif lain meliputi metode Holm atau Benjamini-Hochberg.

---

Dengan landasan teori ini, metode, derivasi, dan alat evaluasi yang digunakan dalam penelitian telah diuraikan secara menyeluruh, siap mendukung perumusan metode pada bab selanjutnya.
