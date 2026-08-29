"""
llm/config.py
=============
LLM configuration settings used across the pipeline.
"""

from conf import ExtendedBaseSettings


class LLMConfig(ExtendedBaseSettings):
    """Settings for LLM backends. Values can be overridden via env vars or config file."""

    # [terjawab]: ini setting INFRA umum, BUKAN hyperparameter laten (yang laten ada di
    #   pipeline/settings.py: latent_steps, kv_max_tokens, knn_*).
    #   - chat_token_limit       : batas token konteks chat completion.
    #   - factor_mining_timeout  : batas waktu satu run mining (detik) sebelum dihentikan.
    #   - init_chat_cache_seed   : seed untuk cache prompt-LLM (reproducibility).
    #   - embedding_max_str_num  : batas jumlah string per batch saat embedding.
    #   Banding LatentMAS: batas maksimum sisi laten (kedalaman/ukuran KV) diatur di settings.py.
    # Token limits
    chat_token_limit: int = 8192

    # Timeout
    factor_mining_timeout: int = 3600  # seconds

    # Cache
    init_chat_cache_seed: int = 42

    # Embedding
    embedding_max_str_num: int = 50


LLM_SETTINGS = LLMConfig()
