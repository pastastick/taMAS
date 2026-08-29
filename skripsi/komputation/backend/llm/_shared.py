"""
llm/_shared.py
==============
Shared utilities for all LLM modules.

Extracted from xmodels.py, models.py, new_client.py, and client.py
to eliminate code duplication across the llm package.

Provides:
    - Type aliases (KVCache, OutputMode)
    - KV-cache helpers (_past_length, _kv_to_cpu, _kv_to_device)
    - Tokenizer helpers (_ensure_pad_token)
    - JSON utilities (robust_json_parse, md5_hash)
    - LatentRealigner class (realignment matrix for latent reasoning)
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

KVCache    = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]   # standard HF KV-cache
OutputMode = Literal["kv_only", "text_only", "kv_and_text"]  # mode output fleksibel


# ─────────────────────────────────────────────────────────────────────────────
# KV-cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _past_length(past_key_values) -> int:
    """Berapa token yang sudah ada di KV-cache.

    Mendukung dua format:
    - Legacy tuple: Tuple[Tuple[Tensor, Tensor], ...] (format lama HF)
    - DynamicCache: objek Cache dari transformers >= 4.36 (Qwen3, dll.)
    """
    if past_key_values is None:
        return 0
    # New transformers Cache API (DynamicCache, StaticCache, etc.)
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values.get_seq_length()
    # Legacy tuple format
    if not past_key_values:
        return 0
    return past_key_values[0][0].shape[-2]


def _is_dynamic_cache(kv) -> bool:
    """True jika kv adalah transformers 5.x DynamicCache (punya .layers list)."""
    return kv is not None and hasattr(kv, 'layers') and isinstance(kv.layers, list)


def _kv_to_cpu(kv: KVCache) -> KVCache:
    """Pindahkan KV-cache ke CPU untuk disimpan.

    transformers 5.x: DynamicCache.__iter__ yield (keys, values, optional_sliding)
    where optional_sliding is None for full-attention layers. Kita hanya simpan
    (keys, values) per layer yang sudah diinisialisasi.
    """
    if kv is None:
        return None
    if _is_dynamic_cache(kv):
        return tuple(
            (layer.keys.cpu(), layer.values.cpu())
            for layer in kv.layers
            if getattr(layer, 'is_initialized', False) and layer.keys is not None
        )
    # Legacy tuple format: Tuple[Tuple[Tensor, ...], ...]
    return tuple(
        tuple(t.cpu() for t in layer if t is not None)
        for layer in kv
    )


def _kv_to_device(kv: KVCache, device: torch.device) -> KVCache:
    """Pindahkan KV-cache dari CPU ke device target."""
    if kv is None:
        return None
    if _is_dynamic_cache(kv):
        return tuple(
            (layer.keys.to(device), layer.values.to(device))
            for layer in kv.layers
            if getattr(layer, 'is_initialized', False) and layer.keys is not None
        )
    return tuple(
        tuple(t.to(device) for t in layer if t is not None)
        for layer in kv
    )


def kv_seq_len(kv: KVCache) -> int:
    """Alias for _past_length — total sequence length in KV-cache."""
    return _past_length(kv)


def kv_size_bytes(kv: KVCache) -> int:
    """Estimate total bytes used by a KV-cache (all layers, CPU or GPU)."""
    if kv is None:
        return 0
    if _is_dynamic_cache(kv):
        total = 0
        for layer in kv.layers:
            if getattr(layer, 'is_initialized', False) and layer.keys is not None:
                total += layer.keys.nelement() * layer.keys.element_size()
                total += layer.values.nelement() * layer.values.element_size()
        return total
    total = 0
    for layer in kv:
        for t in layer:
            if t is not None:
                total += t.nelement() * t.element_size()
    return total


def kv_truncate(kv: KVCache, max_tokens: int, model=None) -> KVCache:
    """
    Truncate KV-cache to keep only the last `max_tokens` tokens.

    KV shape per layer: (key, value) each [batch, n_heads, seq_len, head_dim]
    Slices on dim=-2 (seq_len).

    transformers 5.x DynamicCache: dimodifikasi in-place (layer tensors di-slice)
    dan dikembalikan sebagai DynamicCache agar bisa langsung dipakai sebagai
    past_key_values di model.generate() / model.forward() berikutnya.

    PEMBUKUAN RoPE (B8 — HASIL_GPU §8.2). Memotong ekor cache menyisakan token
    yang KEY-nya masih membawa fase RoPE dari posisi ASLI [N−k, N), sementara
    panjang fisik cache menyusut jadi k. Token berikutnya diberi posisi mulai
    dari k oleh HF → seluruh konteks lama tampak "bergeser" sejauh N−k posisi.
    Ini persis kelas galat yang sudah ditangani `kv_knn_filter` lewat
    `_rerotate_keys_contiguous`, tetapi jalur truncate tak pernah memakainya.

    Karena itu `model` diminta di sini: dengan model tersedia, key di-re-rotasi
    dari posisi asli ke posisi kontigu [0, k) sehingga cache tampak persis
    seperti cache normal panjang k. Tanpa `model` fungsi ini FAIL-OPEN ke
    perilaku lama (slice saja) supaya pemanggil lama tidak pecah — tetapi
    jalur produksi WAJIB mengoper model, kalau tidak menyalakan anggaran KV
    (B9) hanya menukar satu bug dengan bug lain.
    """
    seq_len = _past_length(kv)
    if seq_len <= max_tokens:
        return kv
    if _is_dynamic_cache(kv):
        for layer in kv.layers:
            if getattr(layer, 'is_initialized', False) and layer.keys is not None:
                layer.keys = layer.keys[..., -max_tokens:, :]
                layer.values = layer.values[..., -max_tokens:, :]
        out = kv
    else:
        # Legacy tuple format
        out = tuple(
            tuple(t[..., -max_tokens:, :] for t in layer if t is not None)
            for layer in kv
        )
    if model is None:
        return out
    pairs = _kv_pairs(out)
    if not pairs:
        return out
    batch = pairs[0][0].shape[0]
    device = pairs[0][0].device
    orig = torch.arange(seq_len - max_tokens, seq_len, device=device)
    orig = orig.unsqueeze(0).expand(batch, -1)
    return _rerotate_keys_contiguous(out, orig, model)


# ─────────────────────────────────────────────────────────────────────────────
# KNN-based KV-cache selective filtering
# ─────────────────────────────────────────────────────────────────────────────

def _kv_pairs(kv) -> list:
    """Extract list of (key, value) tensor pairs from any KV cache format.

    transformers 5.x DynamicCache: iterate over .layers and extract tensors.
    Legacy tuple format: each item is (key, value[, ...]).
    """
    if _is_dynamic_cache(kv):
        return [
            (layer.keys, layer.values)
            for layer in kv.layers
            if getattr(layer, 'is_initialized', False) and layer.keys is not None
        ]
    return [(item[0], item[1]) for item in kv]


def _kv_from_pairs(pairs: list, original_kv) -> KVCache:
    """Reconstruct KV cache from (key, value) pairs.

    If original was DynamicCache, returns a new DynamicCache via ddp_cache_data.
    Otherwise returns a tuple.
    """
    if _is_dynamic_cache(original_kv):
        from transformers import DynamicCache
        return DynamicCache(ddp_cache_data=pairs)
    return tuple(pairs)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotasi separuh dimensi terakhir (konvensi RoPE HF)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _get_rotary_emb(model):
    """Ambil modul rotary embedding model (Qwen3 & sejenis). None bila tak ada."""
    inner = getattr(model, "model", model)
    return getattr(inner, "rotary_emb", None) or getattr(model, "rotary_emb", None)


@torch.no_grad()
def _rerotate_keys_contiguous(
    kv: KVCache,
    orig_positions: torch.Tensor,
    model,
) -> KVCache:
    """Re-rotasi key cache dari posisi RoPE ASLI ke posisi KONTIGU [0..k-1].

    Setelah `kv_knn_filter` menyeleksi subset token (posisi asli berlubang,
    mis. [3, 7, 12, 400, 401]), key yang tersimpan masih membawa fase RoPE dari
    posisi aslinya — sementara panjang fisik cache menyusut jadi k. Token baru
    yang ditambahkan setelahnya akan diberi posisi mulai dari k oleh HF →
    desync RoPE → output degenerate.

    Karena RoPE additif (R(a)·R(b) = R(a+b)), kita terapkan R(new−old) pada tiap
    key sehingga key yang semula ter-rotasi di `old` menjadi seakan ter-rotasi di
    `new` (kontigu). Hasilnya cache tampak persis seperti cache normal panjang k:
    semua forward/generate berikutnya memakai penomoran posisi default tanpa
    perlu thread position_ids, dan invarian "panjang == posisi+1" terjaga lintas
    agent. Hanya KEY yang dirotasi (value tak terkena RoPE).

    Args:
        kv             : cache hasil filter (DynamicCache atau tuple).
        orig_positions : [B, k] posisi asli tiap token yang dipertahankan
                         (urut menaik, dari `selected` di kv_knn_filter).
        model          : model HF (untuk mengakses rotary_emb).

    Returns:
        KV-cache dengan key ter-rotasi ulang; bila rotary_emb tak tersedia,
        kembalikan kv apa adanya (fail-open, perilaku lama).
    """
    rotary = _get_rotary_emb(model)
    if rotary is None:
        print("[KNN] rotary_emb tak ditemukan — re-rotasi dilewati (RoPE mungkin desync)")
        return kv

    pairs = _kv_pairs(kv)
    if not pairs:
        return kv
    ref_key = pairs[0][0]                       # [B, H, k, D]
    device = ref_key.device
    batch, _, k, _ = ref_key.shape

    new_positions = torch.arange(k, device=device).unsqueeze(0).expand(batch, -1)
    delta = (new_positions - orig_positions.to(device)).to(torch.float32)  # [B, k] ≤ 0

    # cos/sin untuk sudut delta. Pakai ref float32 agar presisi rotasi terjaga.
    # Asumsi: attention_scaling == 1.0 (RoPE standar Qwen3-4B), sehingga rotasi
    # murni dan komposisi R(a)·R(b)=R(a+b) eksak. Untuk varian long-context
    # (YaRN/linear scaling, scaling≠1) komposisi ini hanya hampiran.
    ref_f32 = torch.zeros(1, dtype=torch.float32, device=device)
    cos, sin = rotary(ref_f32, delta)           # [B, k, D]
    cos = cos.unsqueeze(1)                       # [B, 1, k, D] broadcast atas heads
    sin = sin.unsqueeze(1)

    rotated = []
    for key, value in pairs:
        key_f = key.float()
        key_rot = key_f * cos + _rotate_half(key_f) * sin
        rotated.append((key_rot.to(key.dtype), value))
    return _kv_from_pairs(rotated, kv)


@torch.no_grad()
def kv_knn_filter(
    kv: KVCache,
    query_hidden: torch.Tensor,
    percentage: float = 0.8,
    min_keep: int = 5,
    strategy: str = "top",
    model=None,
    rerotate: bool = True,
) -> KVCache:
    """
    KNN-based KV-cache selective filtering.

    Alih-alih memotong N token terakhir secara buta (kv_truncate),
    fungsi ini memilih token berdasarkan relevansi: menghitung cosine
    similarity antara query_hidden dan key vectors di middle layer,
    lalu mempertahankan top-k% token paling relevan ditambah min_keep
    token terbaru.  Urutan temporal dipertahankan.

    Diadaptasi dari LatentMASMethod._knn_filter_kv_cache
    (core/latent/latent_mas_knn.py).

    Komplementer dengan kv_truncate:
      - kv_truncate: cap ukuran agar tidak OOM (hard limit)
      - kv_knn_filter: pilih token RELEVAN dari yang tersisa (quality)

    Pipeline flow:
      feedback_kv → kv_truncate(max_tokens) → simpan ke _pipeline_kv
      → engine.latent_pass() → kv_knn_filter(query=current prompt) → forward

    Args:
        kv            : Standard HF KV-cache tuple.
                        Shape per layer: (key[B, H, S, D], value[B, H, S, D])
        query_hidden  : [batch, hidden_dim] atau [batch, seq, hidden_dim].
                        Biasanya mean dari input embeddings prompt saat ini.
        percentage    : Fraksi token yang dipertahankan (0.0–1.0).
        min_keep      : Minimum token terbaru yang selalu dipertahankan
                        terlepas dari skor similarity.
        strategy      : "top" (paling mirip), "bottom" (paling beda),
                        "random" (baseline acak).
        model         : model HF. Bila diberikan + rerotate, key yang
                        dipertahankan di-rotasi ulang ke posisi RoPE kontigu
                        (lihat _rerotate_keys_contiguous) sehingga tidak ada
                        desync posisi saat token baru ditambahkan. Bila None,
                        perilaku lama (RoPE bisa desync) dipertahankan.
        rerotate      : Aktifkan re-rotasi (hanya berlaku bila model != None).

    Returns:
        Filtered KV-cache dalam format yang sama; key sudah dire-rotasi ke
        posisi kontigu bila model diberikan.
    """
    seq_len = _past_length(kv)
    if seq_len == 0:
        return kv

    # Hitung k (total token yang dipertahankan)
    k = max(int(seq_len * percentage), min_keep)
    k = min(k, seq_len)

    # Jika mempertahankan semua, tidak perlu filter
    if k >= seq_len:
        return kv

    # Normalize ke list of (key, value) pairs untuk komputasi seragam
    pairs = _kv_pairs(kv)

    # Gunakan keys dari middle layer untuk similarity computation.
    # Middle layer dipilih karena representasinya paling seimbang
    # antara low-level (awal) dan high-level (akhir) features.
    num_layers = len(pairs)
    mid_idx = num_layers // 2
    keys = pairs[mid_idx][0]  # [batch, n_heads, seq_len, head_dim]
    batch_size, num_heads, _, head_dim = keys.shape

    # Rata-rata keys across heads: [batch, seq_len, head_dim]
    keys_avg = keys.mean(dim=1)

    # Siapkan query: pastikan [batch, hidden_dim]
    if query_hidden.dim() == 3:
        query_hidden = query_hidden.mean(dim=1)

    # Proyeksi query ke head_dim jika dimensi berbeda.
    # Input embedding dim = num_heads * head_dim (full hidden_dim),
    # sedangkan keys per head = head_dim.
    hidden_dim = query_hidden.shape[-1]
    if hidden_dim != head_dim:
        if hidden_dim == num_heads * head_dim:
            # Perfect split ke heads lalu average
            query_hidden = query_hidden.view(batch_size, num_heads, head_dim).mean(dim=1)
        else:
            # Reshape apapun ke kelipatan head_dim lalu average
            query_hidden = query_hidden.reshape(batch_size, -1, head_dim).mean(dim=1)

    # Cosine similarity: [batch, seq_len]
    keys_norm = torch.nn.functional.normalize(keys_avg, p=2, dim=-1)
    query_norm = torch.nn.functional.normalize(
        query_hidden.unsqueeze(1), p=2, dim=-1
    )
    similarity = torch.matmul(
        keys_norm, query_norm.transpose(-2, -1)
    ).squeeze(-1)

    # Selalu pertahankan min_keep token terbaru (konteks paling segar)
    _min_keep = min(min_keep, seq_len)
    k_selective = k - _min_keep

    device = keys.device

    if k_selective > 0:
        # Pilih dari token awal (sebelum min_keep terakhir)
        early_seq_len = seq_len - _min_keep
        early_sim = similarity[:, :early_seq_len]

        if strategy == "top":
            _, topk_idx = torch.topk(
                early_sim,
                k=min(k_selective, early_seq_len),
                dim=1, largest=True,
            )
        elif strategy == "bottom":
            _, topk_idx = torch.topk(
                early_sim,
                k=min(k_selective, early_seq_len),
                dim=1, largest=False,
            )
        elif strategy == "random":
            k_rand = min(k_selective, early_seq_len)
            topk_idx = torch.stack([
                torch.randperm(early_seq_len, device=device)[:k_rand]
                for _ in range(batch_size)
            ])
        else:
            raise ValueError(
                f"Invalid knn_strategy: {strategy!r}. "
                f"Must be 'top', 'bottom', or 'random'."
            )

        # Sort agar urutan temporal terjaga
        topk_idx_sorted, _ = torch.sort(topk_idx, dim=1)

        # Gabungkan dengan token terbaru
        recent_idx = torch.arange(
            seq_len - _min_keep, seq_len, device=device,
        ).unsqueeze(0).expand(batch_size, -1)
        selected = torch.cat([topk_idx_sorted, recent_idx], dim=1)
    else:
        # k <= min_keep: cukup ambil token terbaru saja
        selected = torch.arange(
            seq_len - k, seq_len, device=device,
        ).unsqueeze(0).expand(batch_size, -1)

    filtered = _kv_select_indices(kv, pairs, selected)

    # Re-rotasi key dari posisi asli (berlubang) ke posisi kontigu [0..k-1]
    # agar token baru berikutnya tidak mengalami desync RoPE.
    if model is not None and rerotate:
        filtered = _rerotate_keys_contiguous(filtered, selected, model)

    return filtered


def _kv_select_indices(kv: KVCache, pairs: list, indices: torch.Tensor) -> KVCache:
    """
    Pilih posisi tertentu dari setiap layer KV-cache.

    Args:
        kv      : KV-cache asli (untuk deteksi format dan rekonstruksi).
        pairs   : List of (key, value) tensors dari _kv_pairs(kv).
        indices : [batch, k] indeks posisi yang akan dipertahankan.

    Returns:
        Filtered KV-cache dalam format yang sama dengan kv.
    """
    k = indices.shape[1]
    filtered = []
    for key, value in pairs:
        batch_size, num_heads, _, head_dim = key.shape
        idx_exp = indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, k, head_dim,
        )
        filtered.append((
            torch.gather(key, dim=2, index=idx_exp),
            torch.gather(value, dim=2, index=idx_exp),
        ))
    return _kv_from_pairs(filtered, kv)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


# ─────────────────────────────────────────────────────────────────────────────
# JSON utilities
# ─────────────────────────────────────────────────────────────────────────────

def md5_hash(s: str) -> str:
    return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()


def _regex_extract_fields(text: str) -> dict:
    """
    Last-resort extraction for structurally broken JSON.
    Finds "key": "value" pairs using lookahead at the next "key": pattern.
    Handles missing closing quotes caused by LLM generating multi-line values.
    """
    result = {}
    for m in re.finditer(
        r'"([^"\n]+?)"\s*:\s*"(.*?)(?="[^"\n]*?"\s*:|\s*\}|\Z)',
        text, re.DOTALL
    ):
        key = m.group(1).strip()
        val = m.group(2).strip().rstrip(',').rstrip('"').strip()
        if key:
            result[key] = val
    # extract non-string values (true/false/null/numbers)
    for m in re.finditer(r'"([^"\n]+?)"\s*:\s*(true|false|null|-?\d+(?:\.\d+)?)', text):
        key = m.group(1).strip()
        if key in result:
            continue
        val_str = m.group(2)
        if val_str == 'true':
            result[key] = True
        elif val_str == 'false':
            result[key] = False
        elif val_str == 'null':
            result[key] = None
        else:
            try:
                result[key] = float(val_str) if '.' in val_str else int(val_str)
            except ValueError:
                pass
    return result


def _sanitize_json_string_values(text: str) -> str:
    """Escape literal newlines/carriage-returns/tabs inside JSON string values."""
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            result.append(ch)
            continue
        if ch == '\\':
            escape_next = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            if ch == '\n':
                result.append('\\n')
                continue
            if ch == '\r':
                result.append('\\r')
                continue
            if ch == '\t':
                result.append('\\t')
                continue
        result.append(ch)
    return ''.join(result)


def robust_json_parse(text: str, max_retries: int = 3) -> dict:
    """
    Robust JSON parser: handles extra data, LaTeX escapes, markdown-wrapped JSON,
    and literal newlines inside string values.
    Raises json.JSONDecodeError if all strategies fail.
    """
    original_text = text

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract JSON code block (raw + sanitized)
    json_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(json_block_pattern, text)
    if matches:
        for match in matches:
            candidate = match.strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            try:
                return json.loads(_sanitize_json_string_values(candidate))
            except json.JSONDecodeError:
                continue

    # Strategy 3: find first complete JSON object (extra data)
    brace_count = 0
    start_idx = -1
    end_idx = -1
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                end_idx = i
                break

    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Strategy 4: sanitize literal newlines in string values
        try:
            return json.loads(_sanitize_json_string_values(json_str))
        except json.JSONDecodeError:
            pass

        # Strategy 5: fix LaTeX escapes
        fixed_str = json_str
        latex_commands = ['text', 'frac', 'left', 'right', 'times', 'cdot', 'sqrt',
                          'sum', 'prod', 'int', 'alpha', 'beta', 'gamma', 'delta']
        for cmd in latex_commands:
            fixed_str = re.sub(r'(?<!\\)\\(' + cmd + r')', r'\\\\\1', fixed_str)
        fixed_str = re.sub(r'(?<!\\)\\([_\{\}\[\]])', r'\\\\\1', fixed_str)
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError:
            pass

    # Strategy 6: looser JSON extraction
    potential_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    for pj in potential_jsons:
        try:
            result = json.loads(pj)
            if isinstance(result, dict) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            continue

    # Strategy 7: regex field extraction for structurally broken JSON
    # (e.g. missing closing quote on multi-line LLM string values)
    extracted = _regex_extract_fields(text)
    if extracted:
        return extracted

    raise json.JSONDecodeError(
        f"Could not parse JSON; original text length: {len(original_text)}",
        original_text,
        0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LatentRealigner
# ─────────────────────────────────────────────────────────────────────────────
# [terjawab — skripsi Bab 4 §Realignment Laten]: M = (Wout^T Wout + lambda I)^-1
#   Wout^T Win (ridge), lalu normalisasi ke magnitudo rata-rata embedding masukan.
#
# ── STATUS SEJAK B7 (2026-08-07) ─────────────────────────────────────────────
# Matriks ridge M BUKAN LAGI jalur produksi. `_CoreEngine.latent_step_mode`
# default "gumbel", dan pada semua mode selain "raw" M tidak pernah diterapkan —
# kelas ini hanya dimintai `target_norm`. Konsekuensi yang harus ikut dilaporkan
# di Bab 4, bukan disembunyikan:
#   (a) ablasi `use_realign` (G6) hanya bermakna pada mode "raw";
#   (b) pada Qwen3-8B, M memutar hidden state sampai cos(h, hM) = 0,011 —
#       praktis ortogonal (lab/out/realign_probe_Qwen_Qwen3-8B.json), dan
#       hasilnya berada di luar manifold embedding (cos ke embedding terdekat
#       0,275 vs 0,940 pada gumbel).
# Kelas ini DIPERTAHANKAN karena (i) `target_norm` dipakai semua mode, dan
# (ii) mode "raw" masih harus bisa dijalankan untuk mereplikasi baseline
# G1/G3/G6 yang sudah dilaporkan.
class LatentRealigner:
    """
    Membangun dan menerapkan matriks realignment untuk latent reasoning.

    CATATAN: sejak B7 hanya dipakai penuh oleh mode langkah laten "raw"
    (baseline lama). Mode produksi memakai proyeksi convex-hull embedding
    dan hanya meminjam `target_norm` dari sini.

    Masalah:
        last_hidden_state h ada di "output space" (setelah semua transformer layer).
        Input token embedding e ada di "input space" (sebelum layer pertama).
        Kedua space berbeda meskipun dimensi sama.
        Mengumpankan h langsung sebagai input akan membingungkan model.

    Solusi: cari matriks M yang meminimumkan rekonstruksi error:

        min_M  ||Wout @ M - Win||^2_F

    dimana:
        Win  in R^{V x d}  = input embedding matrix
        Wout in R^{V x d}  = output embedding matrix (lm_head)

    Solusi normal equations:
        Gram = Wout^T Wout + eps*I      (regularisasi untuk stabilitas numerik)
        M    = Gram^{-1} (Wout^T Win)

    Setelah proyeksi, normalisasi ke magnitude rata-rata input embedding:
        target_norm = mean_i(||Win[i]||)
        aligned     = h @ M
        aligned     = aligned * (target_norm / ||aligned||)

    Jika use_realign=False: M = I (identity), hanya normalisasi magnitude.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        device: torch.device,
        use_realign: bool = True,
        reg_lambda: float = 1e-5,
    ) -> None:
        self.device = torch.device(device)
        self.use_realign = use_realign
        self.reg_lambda = reg_lambda

        # cache: { (model_id, device) : (matrix, target_norm) }
        self._cache: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

        # debug hook
        self.pre_aligned: Optional[torch.Tensor] = None

        # build initial
        self._ensure_matrix(model)

    def _build_matrix(
        self,
        model: AutoModelForCausalLM,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        input_embeds = (
            model.get_input_embeddings()
            if hasattr(model, "get_input_embeddings")
            else None
        )
        output_embeds = (
            model.get_output_embeddings()
            if hasattr(model, "get_output_embeddings")
            else None
        )

        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)

        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot access embedding weights for realignment.")

        # float32 for numerical stability
        W_in = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        W_out = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

        # target norm (mean embedding norm)
        target_norm = W_in.norm(dim=1).mean().detach()

        if not self.use_realign:
            d = W_in.shape[1]
            return torch.eye(d, device=device, dtype=torch.float32), target_norm

        # === Ridge Regression ===
        # Solve: (W_out^T W_out + lambda*I) M = W_out^T W_in

        gram = W_out.T @ W_out

        reg = self.reg_lambda * torch.eye(
            gram.shape[0], device=device, dtype=torch.float32
        )

        gram = gram + reg
        rhs = W_out.T @ W_in

        try:
            M = torch.linalg.solve(gram, rhs)
        except RuntimeError:
            # fallback for ill-conditioned matrices
            M = torch.linalg.lstsq(gram, rhs).solution

        return M, target_norm

    def _ensure_matrix(
        self,
        model: AutoModelForCausalLM,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        key = (id(model), self.device)

        if key not in self._cache:
            matrix, target_norm = self._build_matrix(model, self.device)
            self._cache[key] = (matrix, target_norm)
        else:
            matrix, target_norm = self._cache[key]

        return matrix, target_norm

    def apply(
        self,
        hidden: torch.Tensor,
        model: AutoModelForCausalLM,
    ) -> torch.Tensor:
        """
        Apply latent realignment.
        Args:
            hidden : [..., d] hidden states
        Returns:
            aligned hidden states
        """
        device = hidden.device
        key = (id(model), device)

        if key not in self._cache:
            matrix, target_norm = self._build_matrix(model, device)
            self._cache[key] = (matrix, target_norm)
        else:
            matrix, target_norm = self._cache[key]

        # ensure correct dtype/device
        matrix = matrix.to(device=device, dtype=torch.float32)
        target_norm = target_norm.to(device=device, dtype=torch.float32)

        # projection
        hidden_fp32 = hidden.to(torch.float32)
        aligned = hidden_fp32 @ matrix

        # debug hook
        self.pre_aligned = aligned.detach().clone()

        # norm alignment
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm / aligned_norm)

        return aligned.to(hidden.dtype)
