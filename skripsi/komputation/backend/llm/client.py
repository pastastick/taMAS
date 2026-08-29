"""client.py — permukaan publik paket llm/, sengaja tipis.

Sampai 2026-08-10 berkas ini adalah SATU file 2335 baris berisi seluruh mesin
LLM (ConvManager, KVCacheStore, `_CoreEngine`, `LocalLLMBackend`,
`LocalChatSession`, factory, utilitas embedding). Isinya dipecah jadi modul
fokus di bawah supaya tiap bagian bisa dibaca berdiri sendiri; berkas ini
dipertahankan sebagai TITIK IMPOR TUNGGAL yang sama seperti sebelumnya
(`from llm.client import LocalLLMBackend, ...`) — sehingga tak satu pun dari
lusinan pemanggil di `mas/`, `bench/`, `eval/`, `factor/`, `gate/` perlu
berubah.

Peta pemecahan:
    llm/methods.py   SUMBU A skripsi — 4 formula langkah laten, murni
    llm/debug_log.py     ConvRecord, TensorConvManager (riwayat tensor debug)
    llm/kv_store.py       KVCacheStore (persistensi KV .pt + indeks SQLite)
    llm/engine.py         LLMResult, _CoreEngine, cache model bersama
    llm/backend.py        LocalLLMBackend (API publik utama) + get_local_backend
    llm/session.py         LocalChatSession
    llm/embeddings.py      utilitas embedding (kompat APIBackend lama)

Untuk kode BARU, impor langsung dari modul spesifik di atas dianjurkan
(lebih jelas asalnya); `llm.client` tetap berfungsi penuh untuk kode lama.
"""
from __future__ import annotations

# ── Shared imports from _shared.py (single source of truth) ────────────────
from llm._shared import (
    KVCache,
    OutputMode,
    LatentRealigner,
    _past_length,
    _ensure_pad_token,
    _kv_to_cpu,
    _kv_to_device,
    kv_truncate,
    kv_knn_filter,
    kv_size_bytes,
    robust_json_parse,
    md5_hash,
)

# ── Sumbu A: empat persamaan langkah laten ──────────────────────────────────
from llm.methods import LATENT_STEP_MODES as _LATENT_STEP_MODES, latent_step_vec

# ── Debug/persistensi ────────────────────────────────────────────────────────
from llm.debug_log import ConvRecord, TensorConvManager
from llm.kv_store import KVCacheStore

# ── Mesin inti ────────────────────────────────────────────────────────────
from llm.engine import LLMResult, _CoreEngine

# ── API publik utama ─────────────────────────────────────────────────────
from llm.backend import LocalLLMBackend, get_local_backend
from llm.session import LocalChatSession

# ── Embedding (kompat APIBackend) ────────────────────────────────────────
from llm.embeddings import (
    calculate_embedding_distance_between_str_list,
    _get_embedding_backend,
)

__all__ = [
    # _shared re-exports
    "KVCache", "OutputMode", "LatentRealigner", "_past_length",
    "_ensure_pad_token", "_kv_to_cpu", "_kv_to_device", "kv_truncate",
    "kv_knn_filter", "kv_size_bytes", "robust_json_parse", "md5_hash",
    # methods
    "_LATENT_STEP_MODES", "latent_step_vec",
    # debug_log / kv_store
    "ConvRecord", "TensorConvManager", "KVCacheStore",
    # engine
    "LLMResult", "_CoreEngine",
    # backend / session
    "LocalLLMBackend", "get_local_backend", "LocalChatSession",
    # embeddings
    "calculate_embedding_distance_between_str_list", "_get_embedding_backend",
]
