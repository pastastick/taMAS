"""
mas/kv_ops.py
====================
Operasi KV-cache yang BENAR untuk LatentMAS: copy, distribusi, concat, save/load.

Latar belakang masalah (kenapa modul ini ada)
----------------------------------------------
`LocalLLMBackend.run()` mengembalikan `result.kv_cache` sebagai **DynamicCache di
GPU** — objek yang **dimutasi in-place** oleh `model.generate()` / `latent_pass()`.
Konsekuensinya:

  1. Kalau objek KV yang sama dioper ke DUA agent (mis. hierarchical/crossover),
     agent pertama akan memutasi KV itu → agent kedua menerima KV yang sudah
     ter-append. Ini bug senyap.
  2. Untuk retry (repair) kita butuh "baseline" yang tidak berubah antar attempt.
  3. Untuk menyimpan KV ke disk lalu memakainya lagi setelah production, kita
     butuh snapshot CPU yang independen.

Modul ini menyediakan primitive yang menjamin **isolasi**:

  - `kv_deepcopy(kv)`   → snapshot independen (tensor di-clone). WAJIB sebelum
                          mendistribusikan satu KV ke beberapa agent.
  - `kv_concat([...])`  → gabung beberapa KV layer-wise sepanjang dim sekuens.
                          Ini implementasi "latent working memory transfer"
                          hierarchical dari paper LatentMAS (Eq. 4): prepend
                          K/V tiap agent ke agent berikutnya.
  - `kv_save / kv_load` → persist ke disk (selalu via CPU) dan muat kembali.
  - `kv_clone_to_device`, `kv_seq_len`, `kv_size_bytes`, `kv_truncate`,
    `kv_knn_filter` → helper format-agnostik (tuple lama maupun DynamicCache).

Semua fungsi menerima dan mengembalikan **DynamicCache** bila inputnya DynamicCache,
atau tuple bila inputnya tuple — sehingga hasilnya bisa langsung dipakai sebagai
`past_key_values` di `LocalLLMBackend.run()`.

Format yang didukung
--------------------
- DynamicCache  (transformers >= 4.36 / 5.x; ini yang dipakai pipeline)
- Legacy tuple  : Tuple[Tuple[Tensor key, Tensor value], ...]  (untuk file lama)

Bentuk tensor per layer: key/value = [batch, n_heads, seq_len, head_dim].
Concat & truncate beroperasi pada dim=-2 (seq_len).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch
# [terjawab — skripsi Bab 4 §Operasi KV-Cache]: deepcopy (isolasi/clone), kv_concat
#   (konkatenasi hierarkis sepanjang dim sekuens), truncate (n token terakhir),
#   kv_knn_filter (seleksi kosinus). Tiap operasi punya rumusan di Bab 4.
# Bangun di atas helper single-source-of-truth di llm/_shared.py.
from llm._shared import (
    KVCache,
    _is_dynamic_cache,
    _kv_pairs,
    _kv_from_pairs,
    _kv_to_cpu,
    _kv_to_device,
    _past_length,
    kv_size_bytes,
    kv_truncate,
    kv_knn_filter,
)

__all__ = [
    "kv_seq_len",
    "kv_size_bytes",
    "kv_deepcopy",
    "kv_clone_to_device",
    "kv_concat",
    "kv_truncate",
    "kv_knn_filter",
    "kv_save",
    "kv_load",
    "kv_describe",
    "kv_distribute",
]


# ─────────────────────────────────────────────────────────────────────────────
# Inspeksi
# ─────────────────────────────────────────────────────────────────────────────

def kv_seq_len(kv: Optional[KVCache]) -> int:
    """Jumlah token yang tersimpan di KV-cache (0 bila None/kosong)."""
    return _past_length(kv) if kv is not None else 0


def kv_describe(kv: Optional[KVCache]) -> dict:
    """Ringkasan KV untuk logging/debug: n_layers, seq_len, head_dim, device, MB."""
    if kv is None:
        return {"present": False}
    pairs = _kv_pairs(kv)
    if not pairs:
        return {"present": True, "n_layers": 0, "seq_len": 0}
    k0 = pairs[0][0]
    return {
        "present": True,
        "format": "DynamicCache" if _is_dynamic_cache(kv) else "tuple",
        "n_layers": len(pairs),
        "seq_len": int(k0.shape[-2]),
        "n_heads": int(k0.shape[1]),
        "head_dim": int(k0.shape[-1]),
        "device": str(k0.device),
        "dtype": str(k0.dtype),
        "size_mb": round(kv_size_bytes(kv) / (1024 ** 2), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Copy / isolasi
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def kv_deepcopy(kv: Optional[KVCache]) -> Optional[KVCache]:
    """Snapshot KV-cache yang BENAR-BENAR independen (tensor di-clone).

    Gunakan ini SEBELUM:
      - mendistribusikan satu KV ke beberapa agent (hierarchical/crossover),
      - menyimpan baseline sebelum retry/repair,
      - menyimpan KV "judger" yang masih akan dipakai beberapa cabang.

    Tanpa clone, mutasi in-place oleh `model.generate()` pada satu cabang akan
    bocor ke cabang lain. Format & device dipertahankan.
    """
    if kv is None:
        return None
    pairs = [(k.clone(), v.clone()) for k, v in _kv_pairs(kv)]
    return _kv_from_pairs(pairs, kv)


@torch.no_grad()
def kv_clone_to_device(
    kv: Optional[KVCache], device: torch.device | str
) -> Optional[KVCache]:
    """Pindahkan salinan KV ke device target (clone, bukan view)."""
    if kv is None:
        return None
    dev = torch.device(device)
    pairs = [(k.to(dev), v.to(dev)) for k, v in _kv_pairs(kv)]
    return _kv_from_pairs(pairs, kv)


# ─────────────────────────────────────────────────────────────────────────────
# Concat hierarchical (LatentMAS Eq. 4)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def kv_concat(kvs: Sequence[Optional[KVCache]]) -> Optional[KVCache]:
    """Gabung beberapa KV-cache layer-wise sepanjang sekuens (dim=-2).

    Ini primitive untuk **hierarchical latent working-memory transfer**
    (LatentMAS Figure 3 / Eq. 4): KV dari beberapa agent independen
    (mis. k parent pada crossover) di-concat menjadi satu working memory
    sebelum agent berikutnya (summarizer / crossover-judger) membacanya.

    Urutan penting: kvs[0] menempati posisi sekuens paling awal, kvs[-1]
    paling akhir (paling dekat dengan token yang akan digenerate). Untuk
    crossover, taruh parent yang paling ingin "didengar" model di posisi
    akhir.

    Semua KV harus punya n_layers, n_heads, dan head_dim yang sama (model
    yang sama). Entri None dilewati. Mengembalikan None bila semua None.

    Catatan device/dtype: semua di-pindah ke device & dtype milik KV pertama
    yang valid agar `torch.cat` tidak gagal lintas device.
    """
    valid = [kv for kv in kvs if kv is not None and _past_length(kv) > 0]
    if not valid:
        return None
    if len(valid) == 1:
        return kv_deepcopy(valid[0])

    per_kv_pairs = [_kv_pairs(kv) for kv in valid]

    n_layers = len(per_kv_pairs[0])
    for i, pairs in enumerate(per_kv_pairs):
        if len(pairs) != n_layers:
            raise ValueError(
                f"kv_concat: KV ke-{i} punya {len(pairs)} layer, "
                f"berbeda dari {n_layers}. Pastikan semua dari model yang sama."
            )

    ref_k = per_kv_pairs[0][0][0]
    ref_device, ref_dtype = ref_k.device, ref_k.dtype

    merged_pairs = []
    for layer_idx in range(n_layers):
        keys, vals = [], []
        for pairs in per_kv_pairs:
            k, v = pairs[layer_idx]
            keys.append(k.to(device=ref_device, dtype=ref_dtype))
            vals.append(v.to(device=ref_device, dtype=ref_dtype))
        merged_pairs.append((
            torch.cat(keys, dim=-2),
            torch.cat(vals, dim=-2),
        ))

    # Rebuild dalam format KV pertama (DynamicCache bila aslinya DynamicCache).
    return _kv_from_pairs(merged_pairs, valid[0])


# ─────────────────────────────────────────────────────────────────────────────
# Distribusi (helper orkestrasi)
# ─────────────────────────────────────────────────────────────────────────────

def kv_distribute(kv: Optional[KVCache], n: int) -> List[Optional[KVCache]]:
    """Hasilkan `n` salinan independen dari satu KV.

    Dipakai orkestrator saat satu KV (mis. kv_consist) harus dikirim ke
    beberapa agent yang berjalan dari baseline yang sama tanpa saling
    mengontaminasi. Salinan pertama bisa saja objek asli? TIDAK — semua
    di-clone agar tidak ada satupun cabang yang memutasi sumber.
    """
    if kv is None or n <= 0:
        return [None] * max(n, 0)
    return [kv_deepcopy(kv) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Persist ke disk
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def kv_save(kv: Optional[KVCache], path: str | Path, metadata: Optional[dict] = None) -> Path:
    """Simpan KV-cache ke file `.pt` (selalu dipindah ke CPU dulu).

    Disimpan sebagai tuple-of-(key,value) + metadata. Bisa dimuat ulang
    dengan `kv_load()` lalu dipakai sebagai `past_key_values` setelah run
    produksi — mendukung eksperimen "ambil KV agent X dari run kemarin".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kv": _kv_to_cpu(kv) if kv is not None else None,
        "seq_len": kv_seq_len(kv),
        "describe": kv_describe(kv),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


@torch.no_grad()
def kv_load(
    path: str | Path,
    device: Optional[torch.device | str] = None,
    as_dynamic: bool = True,
) -> Optional[KVCache]:
    """Muat KV-cache dari file `.pt`.

    Args:
        device     : pindahkan ke device ini setelah load (None = tetap CPU).
        as_dynamic : bila True, bungkus menjadi DynamicCache agar siap dipakai
                     sebagai `past_key_values` (default — ini yang biasanya mau).
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    kv = payload["kv"] if isinstance(payload, dict) else payload
    if kv is None:
        return None
    if device is not None:
        kv = _kv_to_device(kv, torch.device(device))
    if as_dynamic:
        from transformers import DynamicCache
        # kv di sini adalah tuple-of-(k,v); bungkus jadi DynamicCache.
        pairs = [(k, v) for k, v in kv]
        return DynamicCache(ddp_cache_data=pairs)
    return kv
