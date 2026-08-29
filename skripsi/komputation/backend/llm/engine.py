"""engine.py — mesin internal: model HuggingFace + tokenizer + rollout laten.

Diekstrak dari bekas `client.py` BAGIAN 5 (`LLMResult`) + BAGIAN 6
(`_CoreEngine`). `_CoreEngine` tidak dipakai langsung dari luar paket `llm/` —
diakses lewat `backend.LocalLLMBackend`, yang membungkus lock, snapshot debug,
dan API publik di atasnya.

Matematika SUMBU A (empat persamaan langkah laten) TIDAK ada di sini —
`_latent_step_vec` di bawah hanya memanggil `llm.methods.latent_step_vec`.
Lihat modul itu untuk rumusnya, dan `docs/HASIL_TAHAP0.md` §9 untuk verifikasi
kesetiaan terhadap paper.
"""
from __future__ import annotations

import os
import re
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm._shared import (
    KVCache,
    OutputMode,
    LatentRealigner,
    _ensure_pad_token,
    _past_length,
    kv_knn_filter,
)
from llm.methods import LATENT_STEP_MODES, latent_step_vec

# Cache-only model load by default. Set HF_LOCAL_ONLY=0 to allow HF Hub fetches
# (e.g. first download or to refresh an outdated snapshot).
_HF_LOCAL_ONLY = os.environ.get("HF_LOCAL_ONLY", "1") not in ("0", "false", "False")

# HuggingFace token for downloading gated/private models (Qwen3, etc.)
_HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or None


# =============================================================================
# RESULT DATACLASS (output fleksibel)
# =============================================================================

@dataclass
class LLMResult:
    """
    Output fleksibel dari LocalLLMBackend.

    Tiga mode:
        "kv_only"     : kv_cache ada, text=None
                        Untuk agen yang hanya membangun konteks (propose, construct).
        "text_only"   : text ada, kv_cache=None
                        Untuk agen yang butuh teks (judger, feedback).
        "kv_and_text" : keduanya ada
                        Untuk agen yang butuh output teks DAN meneruskan KV.
    """
    text        : Optional[str]           = None
    kv_cache    : Optional[KVCache]       = None
    input_ids   : Optional[torch.Tensor]  = None   # [1, seq_len]
    output_ids  : Optional[torch.Tensor]  = None   # [1, gen_len]
    hidden_last : Optional[torch.Tensor]  = None   # [1, d]
    latent_vecs : Optional[torch.Tensor]  = None   # [steps, d]
    mode        : OutputMode              = "text_only"
    latent_s    : float                   = 0.0    # durasi latent_pass ("berpikir")
    gen_s       : float                   = 0.0    # durasi generate teks
    # B6: langkah laten yang BENAR-BENAR dijalankan (≤ latent_steps bila
    # early-stop menyala) + sebabnya ("budget" | "early_stop" | "off").
    # Dicatat supaya klaim "early-stop menghemat langkah" bisa diverifikasi
    # dari log run, bukan disimpulkan dari durasi.
    n_latent_steps: int                   = 0
    latent_stop   : str                   = "off"

    @property
    def has_text(self) -> bool:
        return self.text is not None

    @property
    def has_kv(self) -> bool:
        return self.kv_cache is not None


# =============================================================================
# CORE ENGINE
# =============================================================================

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# ── Global LLM output log state ───────────────────────────────────────────────
# Semua LocalLLMBackend instance berbagi satu session dir dan satu counter
# atomik, sehingga semua output LLM (dari pipeline, evaluator, mutation, dsb.)
# terkumpul di satu session folder — terorganisir per run proses.
#
# Layout:
#   debug/llm_outputs/
#     session_20260520_060618/
#       0001_propose_kv_and_text.md
#       0002_construct_kv_and_text.md
#       0003_coder_text_only.md
#       ...
#       index.jsonl
#
# _GLOBAL_OUTPUT_RUN_TS diambil sekali saat modul pertama kali di-import
# dan dipakai sebagai nama session subfolder.
_GLOBAL_OUTPUT_LOG_DIR:  Optional[Path] = None
_GLOBAL_OUTPUT_INDEX:    Optional[Path] = None
_GLOBAL_CALL_COUNTER:    int = 0
_GLOBAL_OUTPUT_LOCK:     threading.Lock = threading.Lock()
_GLOBAL_OUTPUT_RUN_TS:   str = time.strftime("%Y%m%d_%H%M%S")


def _init_global_output_dir(output_log_dir: str) -> None:
    """Inisialisasi session dir output global (idempoten — hanya sekali per proses)."""
    global _GLOBAL_OUTPUT_LOG_DIR, _GLOBAL_OUTPUT_INDEX
    with _GLOBAL_OUTPUT_LOCK:
        if _GLOBAL_OUTPUT_LOG_DIR is not None:
            return
        try:
            base = Path(output_log_dir)
            # Buat session subfolder dengan timestamp run
            session_dir = base / f"session_{_GLOBAL_OUTPUT_RUN_TS}"
            session_dir.mkdir(parents=True, exist_ok=True)
            _GLOBAL_OUTPUT_LOG_DIR = session_dir
            _GLOBAL_OUTPUT_INDEX   = session_dir / "index.jsonl"
        except Exception:
            pass


def _load_or_get_cached_model(
    model_name: str,
    device: torch.device,
) -> Tuple[Any, Any]:
    """Return shared (model, tokenizer) for (model_name, device).

    Why: Multiple LocalLLMBackend instances on the same GPU previously
    each reloaded ~8 GB of Qwen3-4B weights, causing OOM. This cache
    keeps a single copy per (model, device) and hands it to every
    _CoreEngine that asks for the same pair.
    """
    cache_key = (model_name, str(device))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            print(f"[CoreEngine] Reusing cached {model_name} on {device}")
            return cached

        print(
            f"[CoreEngine] Loading {model_name} on {device} "
            f"(local_files_only={_HF_LOCAL_ONLY}) ..."
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=_HF_LOCAL_ONLY,
            token=_HF_TOKEN,
        )
        _ensure_pad_token(tokenizer)

        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if torch.cuda.is_available()
                             else torch.float32,
                local_files_only=_HF_LOCAL_ONLY,
                token=_HF_TOKEN,
            )

        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

        model.to(device).eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        _MODEL_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer


class _CoreEngine:
    """
    Engine internal: model HuggingFace + tokenizer + realigner.
    Tidak dipakai langsung dari luar -- diakses lewat LocalLLMBackend.

    Khusus Qwen3-4B:
      - Mode thinking: Qwen3 punya chain-of-thought bawaan (<think>...</think>).
        Untuk pipeline QuantaAlpha yang parse JSON output, thinking harus dimatikan
        dengan menambahkan '/no_think' di system prompt (instruksi resmi Qwen3).
      - Chat template Qwen3 sudah proper -- tidak perlu fallback manual.

    Model+tokenizer di-share via _MODEL_CACHE: instance kedua dengan
    (model_name, device) yang sama tidak reload bobot dari disk/GPU.
    """

    QWEN3_NOTHINK_SUFFIX = "/no_think"

    def __init__(
        self,
        model_name     : str,
        device         : torch.device,
        latent_steps   : int   = 0,
        use_realign    : bool  = False,
        enable_thinking: bool  = False,
        knn_enabled    : bool  = False,
        knn_percentage : float = 0.8,
        knn_min_keep   : int   = 5,
        knn_strategy   : str   = "top",
        latent_step_mode: Optional[str]   = None,
        latent_step_temp: Optional[float] = None,
        latent_step_beta: Optional[float] = None,
        latent_step_alpha: Optional[float] = None,
        latent_early_stop_cos: Optional[float] = None,
    ) -> None:
        self.model_name      = model_name
        self.device          = device
        self.latent_steps    = latent_steps
        self.enable_thinking = enable_thinking

        # KNN-based KV-cache filtering (diterapkan pada past_kv dari step sebelumnya)
        # GUARD: KNN tidak kompatibel dengan latent virtual tokens. Saat
        # latent_steps > 0, setiap agent kv_only menambah N virtual token ke KV.
        # KNN lalu memfilter campuran real+virtual token; re-rotasi RoPE
        # (commit 53f7397) memulihkan posisi token NYATA tapi TIDAK posisi
        # relatif virtual token → distorsi terakumulasi tiap pass → collapse
        # output (judger EOS dini, evolution agent degenerate "1 1 1..."). Lihat
        # konteks.txt Temuan #5. Matikan otomatis agar kombinasi ini tak
        # tersulut tanpa sengaja.
        if latent_steps > 0 and knn_enabled:
            warnings.warn(
                "KNN dinonaktifkan otomatis: tidak kompatibel dengan "
                "latent_steps > 0 (RoPE virtual-token desync, lihat "
                "konteks.txt Temuan #5).",
                RuntimeWarning, stacklevel=2,
            )
            print("[CoreEngine] KNN auto-disabled (latent_steps > 0; "
                  "incompatible with virtual tokens)")
            knn_enabled = False
        self.knn_enabled    = knn_enabled
        self.knn_percentage = knn_percentage
        self.knn_min_keep   = knn_min_keep
        self.knn_strategy   = knn_strategy

        self.model, self.tokenizer = _load_or_get_cached_model(model_name, device)

        # Cap wall-clock per panggilan generate (permintaan user 2026-07-06:
        # "inferensi agent max 5 menit"). Generasi sehat selesai <40s; yang
        # tersentuh cap hanyalah episode degenerasi/rambling (MONITORING_NOTES B2)
        # yang outputnya unparseable juga. Override: env LATENT_MAX_GEN_SECONDS.
        self.max_gen_seconds = float(os.environ.get("LATENT_MAX_GEN_SECONDS", "300"))

        self.realigner: Optional[LatentRealigner] = None
        if latent_steps > 0:
            self.realigner = LatentRealigner(self.model, device,
                                             use_realign=use_realign)
            print(f"[CoreEngine] Realigner built (use_realign={use_realign})")

        # ── Mode langkah laten (AUDIT_KRITIS §7 / G3) ──────────────────────
        # "raw" = perilaku lama: z = realign(h) lalu dinormalkan ke target_norm.
        # Terukur di lab/latent_dynamics.py: vektornya berada DI LUAR manifold
        # embedding (kosinus ke embedding terdekat negatif) dan rollout-nya
        # deterministik → entropi jalur laten nol → pencarian evolusioner
        # kehilangan sumber variansnya (B14).
        # Alternatif memakai kombinasi konveks embedding NYATA, sehingga vektor
        # laten selalu in-distribution, dan T menjadi knob entropi:
        #   soft    z = softmax(W_out h / T) @ W_in            (deterministik)
        #   gumbel  z = softmax((W_out h + g) / T) @ W_in      (stokastik, kontinu)
        #   sample  z = W_in[i], i ~ softmax(W_out h / T)      (batas diskretnya)
        # Semua varian tetap dinormalkan ke target_norm, jadi hanya ARAH vektor
        # yang berubah — sisa pipeline (KV, chat template, parser) tak tersentuh.
        # Argumen eksplisit MENANG atas env var: dengan B2 dipromosikan jadi
        # konfigurasi produksi, mode langkah laten harus terbaca dari
        # configs/experiment.yaml (satu file = satu eksperimen reproducible).
        # Env var dipertahankan sebagai default supaya lab/gpu_suite.py — yang
        # men-set LATENT_STEP_MODE sebelum membangun backend — tetap bekerja.
        # B7 (2026-08-07): default kode diubah "raw" → "gumbel", PERMANEN.
        # Alasannya bukan preferensi melainkan pengukuran: pada Qwen3-8B
        # (tie_word_embeddings=False) matriks ridge M memutar hidden state
        # sampai cos(h, hM) = 0,011 — praktis ORTOGONAL terhadap masukannya
        # (lab/out/realign_probe_Qwen_Qwen3-8B.json). Vektor yang dihasilkan
        # `raw` juga di luar manifold embedding (cos ke embedding terdekat
        # 0,275 vs 0,940 pada gumbel — RENCANA_PERBAIKAN §A4). Jadi `raw`
        # adalah ekstrapolasi linear ke daerah yang tak pernah difit, sedangkan
        # softmax(W_out h / T) @ W_in adalah proyeksi ke convex hull embedding
        # NYATA. Sebelum ini default kode dan default produksi berbeda
        # (settings.py sudah "gumbel" sejak B2, kode masih "raw") — jalur mana
        # pun yang membangun backend tanpa lewat Settings diam-diam memakai
        # persamaan yang sudah ditinggalkan.
        self.latent_step_mode = (
            latent_step_mode
            if latent_step_mode is not None
            else os.environ.get("LATENT_STEP_MODE", "gumbel")
        ).strip().lower()
        self.latent_step_temp = float(
            latent_step_temp
            if latent_step_temp is not None
            else os.environ.get("LATENT_STEP_TEMP", "0.7")
        )
        # β hanya dipakai mode "moi" (konsentrasi pseudo-count observasi;
        # paper MoI: β=1 sebagai setelan universal, sweep {0.25..8} per-task).
        self.latent_step_beta = float(
            latent_step_beta
            if latent_step_beta is not None
            else os.environ.get("LATENT_STEP_BETA", "1.0")
        )
        # α hanya dipakai mode "mix" (sumbu interpolasi raw<->soft).
        # Default 1.0 = ujung `soft`, dipilih supaya salah setel tak
        # diam-diam menghasilkan campuran yang tak dimaksudkan siapa pun.
        self.latent_step_alpha = float(
            latent_step_alpha
            if latent_step_alpha is not None
            else os.environ.get("LATENT_STEP_ALPHA", "1.0")
        )
        if self.latent_step_mode not in LATENT_STEP_MODES:
            raise ValueError(
                f"LATENT_STEP_MODE={self.latent_step_mode!r} tidak dikenal; "
                f"pilih salah satu dari {sorted(LATENT_STEP_MODES)}"
            )
        # Cetak PERSAMAAN yang benar-benar berlaku, bukan cuma nama modenya.
        # `use_realign` hanya berpengaruh pada mode "raw": di mode lain matriks
        # M tidak pernah dipakai (lihat llm.methods.latent_step_vec —
        # realigner hanya dimintai `target_norm`), sehingga ablasi use_realign
        # G6 TIDAK berlaku untuk konfigurasi produksi sekarang. Menuliskannya
        # di log mencegah kekeliruan itu terbawa ke analisis.
        if latent_steps > 0:
            _EQ = {
                "raw":    "z = (h @ M_ridge) dinormalkan   [M ridge DIPAKAI: "
                          f"use_realign={use_realign}]",
                "soft":   "z = softmax(W_out h / T) @ W_in   [M ridge TIDAK dipakai]",
                "gumbel": "z = softmax((W_out h + g) / T) @ W_in   [M ridge TIDAK dipakai]",
                "sample": "z = W_in[i], i ~ softmax(W_out h / T)   [M ridge TIDAK dipakai]",
                "moi":    "z = [(H·p + (β+1−H)·onehot(i~p)) / (β+1)] @ W_in, "
                          f"p = softmax(W_out h / T), β={self.latent_step_beta}"
                          "   [MoI arXiv:2505.14827; M ridge TIDAK dipakai]",
                # Sumbu C (interpolasi). Entri ini WAJIB ada: `_EQ` diindeks
                # langsung oleh mode, jadi mode yang hilang di sini membuat
                # constructor melempar KeyError — bukan sekadar log yang
                # kurang. `mix` sudah lama ada di `llm.methods` (α=0 → `raw`
                # persis, α=1 → `soft` persis), tapi tak pernah dijalankan
                # lewat engine sampai 2026-08-27, sehingga celah ini baru
                # muncul saat b7_probe menghitung kurva α.
                "mix":    "z = normalisasi((1−α)·z_raw + α·z_soft)   "
                          f"[α={self.latent_step_alpha}; sumbu ukur, bukan "
                          "metode: M ridge DIPAKAI lewat suku z_raw bila α<1]",
            }
            print(f"[CoreEngine] latent step mode={self.latent_step_mode} "
                  f"T={self.latent_step_temp} → {_EQ[self.latent_step_mode]}")

        # ── B6: early-stop adaptif pada rollout laten ──────────────────────
        # [G1: jalur laten `raw` mencapai titik tetap di langkah 12 (4B) / 34
        # (8B) — setiap langkah setelah itu menyalin vektor yang sama dan
        # mendesak konteks; G7: pengaruh kanal laten TURUN 4× saat ls 10→60]
        # Berhenti bila cos(h_k, h_{k−1}) > ambang: langkah berikutnya akan
        # menghasilkan vektor yang praktis sama, jadi ia hanya menambah token
        # ke KV tanpa menambah informasi. Ini mengubah makna `latent_steps`
        # dari TARGET menjadi BATAS ATAS.
        # Ambang mengikuti definisi "titik tetap" di lab/latent_dynamics.py
        # (0,999) supaya angka di RENCANA_PERBAIKAN §A4 dan perilaku produksi
        # memakai kriteria yang sama persis.
        # Nonaktifkan dengan nilai ≥ 1.0 (mis. 1.0) — bukan dengan 0, karena
        # 0 justru berarti "berhenti begitu arahnya tak berlawanan".
        _es_raw = (latent_early_stop_cos
                   if latent_early_stop_cos is not None
                   else os.environ.get("LATENT_EARLY_STOP_COS", "0.999"))
        try:
            _es = float(_es_raw)
        except (TypeError, ValueError):
            _es = 0.999
        self.latent_early_stop_cos: Optional[float] = None if _es >= 1.0 else _es

        # Diagnostik langkah laten terakhir (dibaca LocalLLMBackend.run →
        # LLMResult.n_latent_steps). Bukan state semantik: hanya pembukuan.
        self.last_latent_steps_run: int = 0
        self.last_latent_stop: str = "off"

        if latent_steps > 0:
            print(f"[CoreEngine] latent early-stop cos>"
                  f"{self.latent_early_stop_cos} (None = nonaktif)")
        print(f"[CoreEngine] Ready. latent_steps={latent_steps}")

    # ── Satu langkah laten: hidden state → vektor token virtual ─────────────
    def _latent_step_vec(self, last_hidden: "torch.Tensor") -> "torch.Tensor":
        """Petakan hidden state ke vektor yang diumpankan sebagai inputs_embeds.

        Rumus keempat mode (SUMBU A skripsi) ada di `llm.methods` — modul
        berdiri sendiri supaya matematikanya bisa dibaca terpisah dari mesin
        KV/generate di kelas ini. Method ini hanya mengoper state engine.
        """
        return latent_step_vec(
            last_hidden,
            mode=self.latent_step_mode,
            model=self.model,
            realigner=self.realigner,
            temp=self.latent_step_temp,
            beta=self.latent_step_beta,
            alpha=self.latent_step_alpha,
        )

    # ── Chat formatting ────────────────────────────────────────────────────
    # [terjawab — skripsi Bab 4 §Pemrosesan Prompt dan Penggunaan-Ulang KV]:
    #   chat template Qwen3; enable_thinking=False menyisipkan <think></think> kosong;
    #   add_generation_prompt mengontrol prefiks asisten.
    def format_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Render messages ke prompt string.

        Untuk Qwen3 dengan thinking=False:
          '/no_think' ditambahkan ke konten system message.
          Ini adalah cara resmi mematikan thinking di Qwen3.

        Args:
            add_generation_prompt: Jika True, tambahkan prefix assistant
                (e.g. ``<|im_start|>assistant\\n``) di akhir prompt.
                Set False saat latent_pass di kv_and_text mode agar
                prefix hanya ditambahkan saat generate_from_kv().
        """
        if not self.enable_thinking:
            msgs = []
            for m in messages:
                if m["role"] == "system":
                    msgs.append({
                        "role": "system",
                        "content": m["content"].rstrip() + "\n" + self.QWEN3_NOTHINK_SUFFIX
                    })
                else:
                    msgs.append(m)
        else:
            msgs = messages

        if getattr(self.tokenizer, "chat_template", None):
            # Hard switch resmi Qwen3: enable_thinking=False membuat template
            # menyisipkan blok <think></think> KOSONG di assistant prefix, jadi
            # model langsung menulis jawaban. Soft switch '/no_think' saja masih
            # membiarkan model memutuskan sendiri — kadang ia emit blok think
            # kosong lalu EOS (n_out_tok~30, text_len=0 di snapshot repair).
            # Template non-Qwen mengabaikan kwarg ini.
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.enable_thinking,
            )

        # Fallback (tidak diharapkan untuk Qwen3)
        parts = []
        for m in msgs:
            parts.append(f"<|{m['role']}|>\n{m.get('content','')}\n<|/{m['role']}|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def tokenize(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids [1, L], attention_mask [1, L])."""
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @staticmethod
    def _extend_mask(mask: torch.Tensor, past_len: int) -> torch.Tensor:
        """Perluas attention mask untuk token yang sudah di-cache."""
        if past_len == 0:
            return mask
        past_ones = torch.ones(
            (mask.shape[0], past_len), dtype=mask.dtype, device=mask.device
        )
        return torch.cat([past_ones, mask], dim=-1)

    # ── Forward pass ────────────────────────────────────────────────────────
    # [terjawab — skripsi Bab 4 §Mekanisme Penalaran Laten + §Operasi KV-Cache]:
    #   realignment (ridge) + iterasi latent_pass; DynamicCache = wadah KV
    #   (transfer via deepcopy/concat/truncate, bukan operasi matematis tersendiri).
    @torch.no_grad()
    def _forward(
        self,
        input_ids  : torch.Tensor,
        attn_mask  : torch.Tensor,
        past_kv    : Optional[KVCache],
        need_hidden: bool = True,
    ) -> Tuple[KVCache, Optional[torch.Tensor]]:
        """Satu forward pass. Return (past_kv_baru, last_hidden atau None).

        past_kv diasumsikan sudah DynamicCache (dinormalisasi di LocalLLMBackend.run
        entry). Engine internal bekerja penuh dalam format DynamicCache; konversi
        ke tuple hanya dilakukan sekali di run() exit.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        last_hidden = None
        if need_hidden:
            # [B, seq, d] -> ambil posisi terakhir -> [B, d]
            last_hidden = out.hidden_states[-1][:, -1, :]
        return out.past_key_values, last_hidden

    # [terjawab — skripsi Bab 4]: Qwen3 = transformer decoder-only standar
    #   (rujuk Vaswani et al., sudah disitir di Bab 2). _close_open_turn menutup
    #   turn asisten dengan <|im_end|> pada jalur NO-CROP agar struktur chat valid.
    @torch.no_grad()
    def _close_open_turn(self, past_kv: KVCache) -> None:
        """Tutup turn asisten yang menggantung dengan <|im_end|>\\n (path NO-CROP).

        model.generate berhenti DI token EOS (<|im_end|>) tanpa men-cache K/V-nya,
        jadi setelah generate tanpa-crop turn asisten tidak punya penutup di KV.
        Mem-forward <|im_end|>\\n menjaga struktur chat tetap valid untuk agent
        berikutnya yang di-chain (DESIGN.md prod/ §2.2). KV (DynamicCache) dimutasi
        in-place."""
        end_ids, _ = self.tokenize("<|im_end|>\n")
        if end_ids is None or end_ids.shape[-1] == 0:
            return
        past_len = _past_length(past_kv)
        mask = torch.ones(
            (end_ids.shape[0], past_len + end_ids.shape[-1]),
            dtype=torch.long, device=self.device,
        )
        self._forward(end_ids, mask, past_kv, need_hidden=False)

    # ── Latent pass ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def latent_pass(
        self,
        messages    : List[Dict[str, str]],
        past_kv     : Optional[KVCache] = None,
        record_vecs : bool = True,
        latent_steps: Optional[int] = None,
        add_generation_prompt: bool = True,
    ) -> Tuple[KVCache, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass + N latent steps. Tidak menghasilkan teks.

        Setiap latent step:
            1. ambil last_hidden [B, d]
            2. realign: latent_vec = apply_realignment(last_hidden)
            3. buat virtual token: embed = latent_vec.unsqueeze(1)  [B, 1, d]
            4. forward pass dengan inputs_embeds=embed (bukan input_ids)
            5. dapat last_hidden baru

        Virtual token ini tidak punya representasi teks.
        Model "berpikir diam" -- memperbarui KV-cache tanpa menulis token.

        B6 (early-stop): rollout berhenti lebih awal saat
        ``cos(h_k, h_{k-1}) > latent_early_stop_cos``, sehingga `latent_steps`
        adalah BATAS ATAS, bukan target. Jumlah langkah yang benar-benar
        dijalankan ada di ``self.last_latent_steps_run`` dan sebab berhentinya
        di ``self.last_latent_stop``.

        Args:
            add_generation_prompt: Teruskan ke format_messages().
                Untuk kv_and_text mode, set False agar assistant prefix
                ditambahkan hanya saat generate_from_kv().

        Returns:
            (updated_kv, last_hidden [B, d], latent_vecs [steps, d] atau None)
        """
        prompt = self.format_messages(messages, add_generation_prompt=add_generation_prompt)
        ids, mask = self.tokenize(prompt)

        # ── KNN filter past_kv dari step sebelumnya ─────────────
        # Hitung cosine similarity antara input embeddings prompt saat ini
        # dengan key vectors di middle layer KV-cache, lalu pertahankan
        # hanya token yang paling relevan. Ini mengurangi noise dari
        # step sebelumnya dan fokuskan konteks latent.
        if self.knn_enabled and past_kv is not None and _past_length(past_kv) > 0:
            query_embeds = self.model.get_input_embeddings()(ids)
            query_hidden = query_embeds.mean(dim=1)  # [B, hidden_dim]
            past_kv = kv_knn_filter(
                past_kv, query_hidden,
                percentage=self.knn_percentage,
                min_keep=self.knn_min_keep,
                strategy=self.knn_strategy,
                model=self.model,
            )

        past_len = _past_length(past_kv)
        ext_mask = self._extend_mask(mask, past_len)

        # Forward pass pertama: encode seluruh prompt
        past, last_hidden = self._forward(ids, ext_mask, past_kv, need_hidden=True)

        _n_steps = latent_steps if latent_steps is not None else self.latent_steps
        self.last_latent_steps_run = 0
        self.last_latent_stop = "off"
        if _n_steps == 0 or self.realigner is None:
            return past, last_hidden, None

        vecs: List[torch.Tensor] = []
        _early = self.latent_early_stop_cos      # B6; None = nonaktif
        self.last_latent_stop = "budget"

        for _k in range(_n_steps):
            latent_vec   = self._latent_step_vec(last_hidden)  # [B, d]
            latent_embed = latent_vec.unsqueeze(1)             # [B, 1, d]

            if record_vecs:
                vecs.append(latent_vec.detach().cpu())

            past_len    = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long, device=self.device,
            )

            out = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = out.past_key_values
            prev_hidden = last_hidden
            last_hidden = out.hidden_states[-1][:, -1, :]
            self.last_latent_steps_run = _k + 1

            # B6: titik tetap tercapai → langkah berikutnya hanya menyalin.
            # Perbandingan pada HIDDEN STATE (bukan pada vektor laten z), sama
            # dengan metrik `cos_prev` di lab/latent_dynamics.py. float32 dipakai
            # karena cosine bf16 hanya punya ~3 digit signifikan — tak cukup
            # untuk membedakan 0,999 dari 1,000.
            if _early is not None:
                cos = torch.nn.functional.cosine_similarity(
                    last_hidden.float(), prev_hidden.float(), dim=-1
                ).min()
                if float(cos) > _early:
                    self.last_latent_stop = "early_stop"
                    break

        latent_tensor = None
        if record_vecs and vecs:
            # [steps, d], ambil batch index 0
            latent_tensor = torch.stack([v[0] for v in vecs], dim=0)

        return past, last_hidden, latent_tensor

    # ── Generate teks ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_text(
        self,
        messages      : List[Dict[str, str]],
        past_kv       : Optional[KVCache] = None,
        max_new_tokens : int   = 2048,
        temperature    : float = 0.6,
        top_p          : float = 0.95,
        top_k          : int   = 20,
        repetition_penalty: float = 1.05,
        return_kv      : bool  = False,
        prefix_allowed_tokens_fn: Optional[Any] = None,
        prefill        : str   = "",
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[KVCache]]:
        """
        Generate teks. Opsional kembalikan KV-cache sesudah generate.

        Args:
            prefill: teks yang mengisi awal giliran asisten (lihat
                `generate_from_kv`). Dipasang di sini juga — bukan hanya di
                jalur KV yang bermasalah — supaya prosedur pembangkitan
                IDENTIK di ketiga medium. Perbedaan prosedur antar-medium
                justru akan mencemari perbandingan medium yang menjadi salah
                satu pertanyaan penelitian.
            prefix_allowed_tokens_fn: Callable (batch_id, input_ids) -> list[int]
                yang membatasi token boleh-keluar di setiap step dekoder.
                Diteruskan ke model.generate() untuk guided decoding
                (contoh: enforce JSON schema via lm-format-enforcer).

        Returns:
            (text, input_ids [1,L], output_ids [1,G], kv atau None)
        """
        # prefill disambung sebagai STRING sebelum tokenisasi supaya BPE merge
        # di batas prefix-asisten/prefill sama dengan yang dilihat model saat
        # dilatih; menokenisasinya terpisah lalu menyambung id bisa memecah
        # merge itu.
        prompt = self.format_messages(messages) + prefill
        ids, mask = self.tokenize(prompt)
        prompt_len = int(mask.sum())

        # ── KNN filter past_kv (sama seperti di latent_pass) ────
        if self.knn_enabled and past_kv is not None and _past_length(past_kv) > 0:
            query_embeds = self.model.get_input_embeddings()(ids)
            query_hidden = query_embeds.mean(dim=1)
            past_kv = kv_knn_filter(
                past_kv, query_hidden,
                percentage=self.knn_percentage,
                min_keep=self.knn_min_keep,
                strategy=self.knn_strategy,
                model=self.model,
            )

        past_len      = _past_length(past_kv)
        ext_mask      = self._extend_mask(mask, past_len)

        out = self.model.generate(
            input_ids=ids,
            attention_mask=ext_mask,
            max_new_tokens=max_new_tokens,
            max_time=self.max_gen_seconds,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_kv,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        generated_ids = out.sequences[0, prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not self.enable_thinking:
            text = self._strip_thinking(text)
        text = self._join_prefill(prefill, text)

        kv_out = out.past_key_values if return_kv else None
        return text, ids, generated_ids.unsqueeze(0), kv_out

    # ── Generation from existing KV (kv_and_text mode) ──────────────────
    # [terjawab — skripsi Bab 4 §Pemrosesan Prompt]:
    #   prefix_ids = tok_with[len(tok_without):] (selisih tokenisasi full-text).
    @staticmethod
    def _join_prefill(prefill: str, text: str) -> str:
        """Sambung kembali prefill ke lanjutan yang dibangkitkan model.

        Prefill ikut dikirim sebagai input, sehingga ia TIDAK ada di
        `generated_ids` dan harus dipasang lagi di depan teks.

        Satu penjagaan, dan sengaja yang paling sederhana: bila lanjutan itu
        sendiri sudah dibuka `{`, model mengabaikan prefill dan menulis
        objeknya sendiri dari awal — menempelkan prefill di depannya justru
        akan MERUSAK keluaran yang sehat (`{"hypothesis": {"hypothesis": …`).
        Dalam kasus itu prefill dibuang dan teks model dipakai apa adanya.

        Akibatnya perubahan ini no-op untuk sel yang keluarannya memang sudah
        utuh, dan hanya menutup kasus keluaran yang mulai dari tengah objek.
        """
        if not prefill or text.lstrip().startswith("{"):
            return text
        return prefill + text

    def _get_generation_prefix_ids(
        self, messages: List[Dict[str, str]], prefill: str = "",
    ) -> torch.Tensor:
        """
        Ekstrak token IDs untuk assistant generation prefix.

        Cara kerja:
            1. Tokenize prompt DENGAN add_generation_prompt=True (+ prefill)
            2. Tokenize prompt TANPA add_generation_prompt
            3. Selisih = token prefix assistant (misal ``<|im_start|>assistant\\n``)

        Menggunakan tokenisasi full-text untuk menjaga BPE merge di batas —
        termasuk batas antara prefix asisten dan `prefill`, yang karena itu
        disambung sebagai STRING sebelum ditokenisasi, bukan ditokenisasi
        terpisah lalu digabung sebagai id.
        """
        prompt_with = self.format_messages(messages, add_generation_prompt=True) + prefill
        prompt_without = self.format_messages(messages, add_generation_prompt=False)

        ids_with = self.tokenizer(
            prompt_with, add_special_tokens=False
        )["input_ids"]
        ids_without = self.tokenizer(
            prompt_without, add_special_tokens=False
        )["input_ids"]

        prefix_ids = ids_with[len(ids_without):]
        if not prefix_ids:
            # Fallback: tokenize newline sebagai trigger minimal
            prefix_ids = self.tokenizer(
                "\n", add_special_tokens=False
            )["input_ids"][-1:]

        return torch.tensor([prefix_ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate_from_kv(
        self,
        past_kv        : KVCache,
        messages       : List[Dict[str, str]],
        max_new_tokens : int   = 2048,
        temperature    : float = 0.6,
        top_p          : float = 0.95,
        top_k          : int   = 20,
        repetition_penalty: float = 1.05,
        return_kv      : bool  = True,
        prefix_allowed_tokens_fn: Optional[Any] = None,
        prefill        : str   = "",
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[KVCache]]:
        """
        Generate teks dari KV-cache yang sudah ada TANPA re-encode pesan.

        Dipakai setelah latent_pass() di kv_and_text mode.

        `prefill` mengisi awal giliran asisten dengan teks yang SUDAH pasti,
        lalu memasangnya kembali ke hasil dekode. Ini yang menutup artefak
        `kv_and_text` pada lengan faktor: agen yang mewarisi KV berisi objek
        JSON utuh dari agen hulu cenderung melanjutkan seolah masih berada di
        dalam objek itu, sehingga keluarannya dimulai dari nilai — tanpa `{`
        pembuka — dan gagal diurai. Bukan pemotongan oleh harness: panjang
        teks yang tercatat sama persis dengan yang dibangkitkan model.

        Flow:
            latent_pass(messages, add_generation_prompt=False)
                → KV berisi [prompt tanpa assistant prefix + latent steps]
            generate_from_kv(kv, messages)
                → Kirim HANYA assistant prefix tokens sebagai input_ids
                → Model generate response dari konteks latent

        Ini menghindari double-encoding: prompt hanya diproses sekali
        (di latent_pass), dan generate hanya menambahkan trigger minimal.

        Returns:
            (text, prefix_ids [1, P], output_ids [1, G], kv atau None)
        """
        prefix_ids = self._get_generation_prefix_ids(messages, prefill)
        prefix_len = prefix_ids.shape[-1]

        past_len = _past_length(past_kv)
        mask = torch.ones(
            (1, past_len + prefix_len), dtype=torch.long, device=self.device,
        )

        out = self.model.generate(
            input_ids=prefix_ids,
            attention_mask=mask,
            past_key_values=past_kv,
            max_new_tokens=max_new_tokens,
            max_time=self.max_gen_seconds,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        # out.sequences = [prefix_ids + generated_ids]
        generated_ids = out.sequences[0, prefix_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not self.enable_thinking:
            text = self._strip_thinking(text)
        # Prefill dipasang SESUDAH strip thinking: ia bukan bagian dari
        # lanjutan yang dibangkitkan model, jadi ia tak boleh ikut tersapu
        # regex penghapus blok <think>.
        text = self._join_prefill(prefill, text)

        kv_out = out.past_key_values if return_kv else None
        return text, prefix_ids, generated_ids.unsqueeze(0), kv_out

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Hapus blok <think>...</think> dari output Qwen3.

        Menangani tiga kasus:
        - Closed: <think>...</think> → dihapus seluruhnya
        - Unclosed: <think>... tanpa </think> (model stuck) → dihapus sampai akhir
        - Orphan: </think> tanpa <think> pembuka → tag yatim dibuang
        """
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"</?think>", "", text)
        return text.strip()
