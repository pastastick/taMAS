"""methods.py — SUMBU A skripsi: lima persamaan langkah laten.

Fungsi murni yang memetakan satu hidden state ke satu vektor yang diumpankan
balik sebagai `inputs_embeds` — inilah yang dibandingkan `docs/DESAIN_EKSPERIMEN.md`
§2 antar M = {`raw`, `soft`, `sample`, `gumbel`, `moi`}, dengan keluarga
relaksasi diskret R = M \\ {`raw`}. Diekstrak dari
`engine.py::_CoreEngine._latent_step_vec` ke modul berdiri sendiri supaya
matematikanya bisa dibaca, disitir, dan (bila perlu) diuji terisolasi tanpa
menyisir mesin KV-cache/generate di sekitarnya.

Kesetiaan terhadap paper diverifikasi 2026-08-10 terhadap
`reference/LatentMAS` dan `reference/mixinputs` — rinciannya di
`docs/HASIL_TAHAP0.md` §9 (rujukan arXiv persis per mode + hasil verifikasi
aljabar `moi`, termasuk satu divergensi implementasi yang terdokumentasi di
sana, bukan di sini, supaya penjelasannya tak terpisah dari angkanya).

    from llm.methods import latent_step_vec, LATENT_STEP_MODES
"""
from __future__ import annotations

from typing import Any

import torch

from llm._shared import LatentRealigner

# Mode langkah laten yang sah.
# "moi" = Mixture of Inputs (Zhuang dkk., NeurIPS 2025, arXiv:2505.14827):
# sampel token diskret sebagai OBSERVASI, distribusi sebagai PRIOR Dirichlet,
# input berikutnya = ekspektasi posterior (campuran one-hot + distribusi).
# Training-free seperti mode lain; ditambahkan 2026-08-09 untuk Tahap 0 lanjutan
# (kandidat literatur yang menjembatani `sample` dan `soft`).
# "mix" = keluarga interpolasi raw<->soft, ditambahkan 2026-08-27. Ia BUKAN
# metode dari literatur dan bukan usulan metode baru: ia sumbu ukur. Kelima
# mode di atas memberi lima titik terpisah, sehingga hubungan antara geometri
# representasi dan kinerja hanya bisa dibaca sebagai "searah". `mix` mengisi
# jarak di antaranya secara kontinu, sehingga BENTUK hubungan itu yang diuji —
# monoton, ber-ambang, atau tidak berpola sama sekali. Ketiganya temuan.
LATENT_STEP_MODES = ("raw", "soft", "gumbel", "sample", "moi", "mix")


def latent_step_vec(
    last_hidden: "torch.Tensor",
    *,
    mode: str,
    model: Any,
    realigner: LatentRealigner,
    temp: float,
    beta: float,
    alpha: float = 1.0,
) -> "torch.Tensor":
    """Petakan hidden state ke vektor yang diumpankan sebagai inputs_embeds.

    `raw` mempertahankan jalur produksi lama (realigner, ridge $W_a$ resmi
    LatentMAS). Mode lain memproyeksikan lewat distribusi token dulu sehingga
    hasilnya adalah kombinasi konveks baris W_in — selalu di dalam convex hull
    embedding nyata. Normalisasi akhir ke target_norm sama untuk semua mode.

    Args:
        last_hidden: hidden state [B, d] dari langkah laten sebelumnya.
        mode: salah satu dari `LATENT_STEP_MODES`.
        model: model HF (untuk `get_input_embeddings`/`get_output_embeddings`).
        realigner: `LatentRealigner` — dipakai penuh di mode `raw`; di mode
            lain hanya dipinjam `target_norm`-nya (matriks M tak pernah dipakai).
        temp: suhu softmax T (di-clamp minimal 1e-6).
        beta: parameter β mode `moi` (tak dipakai mode lain).
        alpha: parameter α mode `mix` (tak dipakai mode lain). α=0 memberi
            `raw` persis, α=1 memberi `soft` persis.
    """
    if mode == "raw":
        return realigner.apply(last_hidden, model)

    if mode == "mix":
        # z(α) = normalisasi( (1-α)·z_raw + α·z_soft ).
        #
        # Kedua ujung dihitung lebih dulu SECARA PENUH lewat jalur mode
        # aslinya, bukan disusun ulang di sini. Itu yang menjamin α=0 dan α=1
        # menghasilkan vektor yang identik dengan sel `raw` dan `soft` yang
        # sudah dijalankan — sehingga kedua titik ujung kurva tak perlu
        # dijalankan ulang di GPU, dan kurvanya tersambung ke matriks
        # eksperimen yang sudah ada alih-alih berdiri sendiri.
        #
        # Keduanya sudah ternormalisasi ke `target_norm` yang sama, jadi
        # campurannya adalah titik pada tali busur antara dua vektor
        # sepanjang itu; normalisasi ulang mengembalikannya ke sfera yang
        # sama. Akibatnya yang berubah sepanjang α murni ARAH — dan arah
        # itulah yang diukur `max_i cos(z, W_in[i])`, sehingga jarak ke
        # convex hull embedding bergerak kontinu tanpa dicampuri perubahan
        # panjang vektor.
        a = min(max(float(alpha), 0.0), 1.0)
        z_raw = realigner.apply(last_hidden, model).float()
        z_soft = latent_step_vec(last_hidden, mode="soft", model=model,
                                 realigner=realigner, temp=temp,
                                 beta=beta).float()
        if a <= 0.0:
            return z_raw.to(last_hidden.dtype)
        if a >= 1.0:
            return z_soft.to(last_hidden.dtype)
        z = (1.0 - a) * z_raw + a * z_soft
        tn = realigner._ensure_matrix(model)[1].to(last_hidden.device)
        z = z * (tn / z.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return z.to(last_hidden.dtype)

    W_in = model.get_input_embeddings().weight                     # [V, d]
    target_norm = realigner._ensure_matrix(model)[1].to(last_hidden.device)
    T = max(temp, 1e-6)

    out_emb = model.get_output_embeddings()
    logits = (out_emb(last_hidden) if out_emb is not None
              else last_hidden @ W_in.T).float()                    # [B, V]

    if mode == "sample":
        idx = torch.multinomial(torch.softmax(logits / T, dim=-1), 1)  # [B,1]
        z = W_in[idx.squeeze(-1)].float()
    elif mode == "moi":
        # Mixture of Inputs (arXiv:2505.14827, training-free). Distribusi
        # p adalah PRIOR Dirichlet (α_i = H·p_i, H = entropi ternormalisasi
        # ∈ [0,1]); token tersampel adalah OBSERVASI dengan pseudo-count
        # (β+1−H); input berikutnya = ekspektasi posterior:
        #   w_i = [H·p_i + (β+1−H)·1[i=y]] / (β+1),  z = w @ W_in.
        # Intuisi untuk kanal simbolik: one-hot menjangkarkan identitas
        # token diskret (yang hilang di `soft`/`gumbel` karena rata-rata
        # seluruh vocab), sementara suku H·p mempertahankan superposisi.
        # H tinggi (model ragu) → campuran condong ke distribusi; H rendah
        # (model yakin) → nyaris one-hot murni. β=1 = default paper.
        # Catatan: paper tidak me-rescale embedding; normalisasi ke
        # target_norm di bawah adalah konvensi SERAGAM harness ini untuk
        # semua mode, dipertahankan agar perbandingan antar-mode adil.
        #
        # Verifikasi vs kode rujukan (docs/HASIL_TAHAP0.md §9.2): rumus di
        # bawah aljabar-identik dgn reference/mixinputs (dibuktikan lewat
        # substitusi), TAPI H di sini dinormalisasi log(V) — VOCAB PENUH,
        # karena logits diakses langsung (get_output_embeddings). Kode
        # rujukan menormalisasi log(20) — TOP-20 SLICE API logprobs vLLM
        # (gpu_input_batch.py num_logprobs default 20), keterbatasan
        # rekayasa vLLM, bukan definisi paper. Akibatnya H di sini secara
        # sistematis lebih rendah (lebih "yakin") drpd kode rujukan pada
        # tingkat kepastian model yang sama — bukan bug, tapi bukan
        # replikasi bit-identik. Jangan klaim "identik kode rujukan";
        # klaim yang sah = "identik definisi paper".
        probs = torch.softmax(logits / T, dim=-1)                  # [B, V]
        idx = torch.multinomial(probs, 1)                          # [B, 1]
        onehot = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        H = (-(probs * probs.clamp_min(1e-12).log()).sum(-1, keepdim=True)
             / torch.log(torch.tensor(float(probs.shape[-1]),
                                      device=probs.device)))       # [B, 1]
        w = (H * probs + (beta + 1.0 - H) * onehot) / (beta + 1.0)
        z = (w.to(W_in.dtype) @ W_in).float()                      # [B, d]
    else:
        if mode == "gumbel":
            u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
            logits = logits + (-torch.log(-torch.log(u)))
        probs = torch.softmax(logits / T, dim=-1)
        z = (probs.to(W_in.dtype) @ W_in).float()              # [B, d]

    z = z * (target_norm / z.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    return z.to(last_hidden.dtype)
