"""
llm package
===========
Public API untuk mesin LLM lokal (Qwen3 via HuggingFace) + empat persamaan
langkah laten yang dibandingkan skripsi ini.

Module structure:
    _shared.py       - KVCache, LatentRealigner, helper KV lintas-modul
    methods.py       - SUMBU A: 4 formula langkah laten (raw/gumbel/moi/sample)
    engine.py        - _CoreEngine (model+tokenizer, latent_pass, generate)
    backend.py       - LocalLLMBackend (API publik utama)
    session.py       - LocalChatSession (sesi multi-turn)
    embeddings.py    - utilitas embedding (kompat APIBackend lama)
    debug_log.py     - ConvRecord/TensorConvManager (tensor debug)
    kv_store.py      - KVCacheStore (persistensi KV .pt + SQLite)
    config.py        - LLM_SETTINGS
    guided_decoding.py - constrained decoding (construct step)
    client.py        - facade impor tunggal, re-export semua di atas
                        (kode baru dianjurkan impor modul spesifik langsung)
"""

# Shared types and utilities
from llm._shared import (
    KVCache,
    OutputMode,
    LatentRealigner,
    _past_length,
    _kv_to_cpu,
    _kv_to_device,
    kv_truncate,
    kv_knn_filter,
    kv_size_bytes,
    robust_json_parse,
    md5_hash,
)

# Primary backend
from llm.client import (
    LocalLLMBackend,
    LLMResult,
    KVCacheStore,
    TensorConvManager,
    LocalChatSession,
    get_local_backend,
    calculate_embedding_distance_between_str_list,
)

# Guided JSON decoding (untuk construct step khususnya)
from llm.guided_decoding import (
    CONSTRUCT_FACTOR_JSON_SCHEMA,
    build_guided_json_prefix_fn,
)

__all__ = [
    # Types
    "KVCache",
    "OutputMode",
    # Shared utilities
    "LatentRealigner",
    "_past_length",
    "_kv_to_cpu",
    "_kv_to_device",
    "kv_truncate",
    "kv_knn_filter",
    "kv_size_bytes",
    "robust_json_parse",
    "md5_hash",
    # Primary backend
    "LocalLLMBackend",
    "LLMResult",
    "KVCacheStore",
    "TensorConvManager",
    "LocalChatSession",
    "get_local_backend",
    "calculate_embedding_distance_between_str_list",
    # Guided decoding
    "CONSTRUCT_FACTOR_JSON_SCHEMA",
    "build_guided_json_prefix_fn",
]
