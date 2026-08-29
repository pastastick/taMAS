"""embeddings.py — utilitas embedding, kompatibilitas APIBackend lama.

Diekstrak dari bekas `client.py` BAGIAN 10. Bukan jalur laten (tidak
menyentuh `methods.py`/`engine.py`) — hanya mean-pool input embeddings
dari satu `LocalLLMBackend` singleton (`latent_steps=0`), dipakai untuk
kesamaan teks (mis. dedup faktor) di tempat lain di codebase.
"""
from __future__ import annotations

from typing import List, Optional

from llm.backend import LocalLLMBackend

# Singleton backend untuk embedding operations (lazy-init)
_embedding_backend: Optional[LocalLLMBackend] = None


def _get_embedding_backend() -> LocalLLMBackend:
    """Dapatkan/buat singleton backend untuk embedding utilities."""
    global _embedding_backend
    if _embedding_backend is None:
        _embedding_backend = LocalLLMBackend(latent_steps=0)
    return _embedding_backend


def calculate_embedding_distance_between_str_list(
    source_list: List[str],
    target_list: List[str],
) -> List[List[float]]:
    """
    Hitung cosine similarity antara dua list string.
    Compatible dengan fungsi lama di client.py.

    Returns:
        Matrix [len(source), len(target)] berisi cosine similarity scores.
    """
    backend = _get_embedding_backend()
    src_emb = backend.create_embedding(source_list)
    tgt_emb = backend.create_embedding(target_list)

    # Cosine similarity
    result: List[List[float]] = []
    for s in src_emb:
        row: List[float] = []
        s_norm = sum(x * x for x in s) ** 0.5
        for t in tgt_emb:
            t_norm = sum(x * x for x in t) ** 0.5
            if s_norm == 0 or t_norm == 0:
                row.append(0.0)
            else:
                dot = sum(a * b for a, b in zip(s, t))
                row.append(dot / (s_norm * t_norm))
        result.append(row)
    return result
