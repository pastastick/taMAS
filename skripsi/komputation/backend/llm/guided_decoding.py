"""
llm/guided_decoding.py
======================
Guided JSON decoding via `lm-format-enforcer`'s JsonSchemaParser.

Dipakai khususnya untuk construct step pada pipeline factor mining, di mana
model kecil (<~70B) sering gagal mengikuti schema nested
`{factor_name → {description, variables, formulation, expression}}`
karena anchoring pada token-token `$close`/`TS_MEAN`/dst yang dominan di prompt.

Dengan prefix_allowed_tokens_fn dari lm-format-enforcer, di setiap langkah
dekoder hanya token yang melanjutkan parse JSON yang valid yang boleh keluar.
Struktur output DIPAKSA valid — bukan sekadar dijinjit via prompt.

Runtime overhead ~10–20% latency untuk schema nested dalam (token-by-token
grammar check), namun correctness gain jauh lebih penting untuk model 4B.

Catatan kompatibilitas:
    lm-format-enforcer >=0.10 impor `PreTrainedTokenizerBase` dari
    `transformers.tokenization_utils`, tapi di transformers terbaru class
    tersebut pindah ke `transformers.tokenization_utils_base`. Integrasi
    bawaan (`lmformatenforcer.integrations.transformers`) karena itu gagal
    di-load. Kita mereplikasi 3 fungsi inti dari integrasi tersebut inline
    di sini, hanya menggunakan core API `lmformatenforcer` yang stabil.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# Schema: construct output (per factor_experiment_output_format di prompts.yaml)
# ─────────────────────────────────────────────────────────────────────────────
#
# Shape yang dipaksakan:
#
#   {
#     "<factor_name>": {
#       "description": "...",
#       "variables":   { "<symbol>": "<meaning>", ... },
#       "formulation": "...",
#       "expression":  "..."
#     },
#     ...
#   }
#
# `additionalProperties` pada outer object bertipe object → outer key adalah
# nama factor bebas (bukan enum), value-nya WAJIB object 4-field.
# `required` mem-paksa keempat field ada sebelum object bisa ditutup.
# `minProperties: 1` menjamin setidaknya satu factor ter-generate.

# Schema yang DIPAKAI jalur mas (B11). Bentuknya mengikuti
# `parsers.parse_construct_json`: {hypothesis, factors:[{name, expression,
# explanation}]}. Schema lama di bawah (CONSTRUCT_FACTOR_JSON_SCHEMA) melayani
# format rdagent yang berbeda dan dipertahankan untuk jalur itu.
#
# minItems 1 menjamin setidaknya satu faktor; maxItems 6 mengikuti kontrak prompt
# ("3 sampai 6 ekspresi"). Grammar hanya memaksa STRUKTUR — ia tidak menjamin ISI
# ekspresinya bermakna; itu tetap tugas gate.
#
# `reasoning` ada di sini dengan sengaja. Guided decoding memaksa SELURUH output
# menjadi JSON, sehingga prosa penalaran yang diminta prompt ("Step 1 … Step 5")
# tak punya tempat lagi — dan menghapus penalaran dari model kecil biasanya
# menurunkan mutu, bukan menaikkannya. Dengan menaruhnya sebagai field string
# PERTAMA, model tetap berpikir dulu baru menulis faktornya, tanpa satu pun token
# keluar dari grammar JSON. Field ini tidak dibaca parser; ia ada untuk model.
LATENT_CONSTRUCT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "hypothesis": {"type": "string"},
        "factors": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "expression": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["name", "expression", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reasoning", "hypothesis", "factors"],
    "additionalProperties": False,
}


CONSTRUCT_FACTOR_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "variables": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "formulation": {"type": "string"},
            "expression": {"type": "string"},
        },
        "required": ["description", "variables", "formulation", "expression"],
        "additionalProperties": False,
    },
    "minProperties": 1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Replika lokal dari lmformatenforcer.integrations.transformers
# ─────────────────────────────────────────────────────────────────────────────
#
# Kenapa replika: integrasi bawaan import `PreTrainedTokenizerBase` dari
# `transformers.tokenization_utils` yang sudah tidak ada di versi transformers
# terbaru. Kita hanya pakai core API `lmformatenforcer` di sini — tanpa
# import `transformers.tokenization_utils` sama sekali → kompatibel lintas
# versi transformers.

def _build_regular_tokens_list(
    tokenizer: Any, vocab_size: int,
) -> List[Tuple[int, str, bool]]:
    """Enumerasi token reguler: (token_id, decoded_with_space_trick, is_word_start).

    Trik "prepend token 0 lalu strip karakter pertama" dipakai oleh
    lm-format-enforcer untuk mendeteksi apakah sebuah token menandai awal kata
    (mulai dengan spasi) tanpa bergantung pada informasi vocab BPE internal.
    """
    token_0 = tokenizer.encode("0")[-1]
    regular_tokens: List[Tuple[int, str, bool]] = []
    for token_idx in range(vocab_size):
        if token_idx in tokenizer.all_special_ids:
            continue
        decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
        decoded_regular = tokenizer.decode([token_idx])
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
    return regular_tokens


def _decode_function(tokenizer: Any, tokens: List[int]) -> str:
    """Decode token IDs ke string, buang karakter replacement (U+FFFD) di ekor."""
    decoded = tokenizer.decode(tokens)
    return decoded.rstrip("�")


# Membangun tokenizer-data berarti men-decode SETIAP token di vocab (±150 k untuk
# Qwen3) — beberapa detik. Tanpa cache, biaya itu dibayar ulang setiap kali agen
# dipanggil dan akan tampak sebagai "guided decoding lambat" padahal yang lambat
# adalah persiapannya. Di-key oleh id(tokenizer) karena tokenizer di-share lewat
# _MODEL_CACHE di client.py, jadi satu entri per model per proses.
_TOKENIZER_DATA_CACHE: Dict[int, Any] = {}


def _build_token_enforcer_tokenizer_data(
    tokenizer: Any,
    use_bitmask: bool = False,
    vocab_size: Optional[int] = None,
) -> Any:
    """Bangun TokenEnforcerTokenizerData dari tokenizer HuggingFace (di-cache)."""
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

    key = id(tokenizer)
    cached = _TOKENIZER_DATA_CACHE.get(key)
    if cached is not None:
        return cached
    vocab_size = vocab_size or len(tokenizer)
    regular_tokens = _build_regular_tokens_list(tokenizer, vocab_size)
    decode_fn = functools.partial(_decode_function, tokenizer)
    # Signature TokenEnforcerTokenizerData berbeda antar-versi lm-format-enforcer:
    # 0.10.12 menerima (regular_tokens, decoder, eos_token_id) saja, sedangkan
    # versi lain juga menerima (use_bitmask, vocab_size). Replika lama di file ini
    # ditulis untuk varian 5-argumen, sehingga di lingkungan ini SETIAP panggilan
    # guided decoding meledak dengan TypeError — dan karena jalur ini tak pernah
    # dipakai siapa pun (B11 kode mati), tak ada yang pernah tahu. Coba bentuk
    # panjang dulu, jatuh ke bentuk pendek bila ditolak.
    try:
        data = TokenEnforcerTokenizerData(
            regular_tokens, decode_fn, tokenizer.eos_token_id,
            use_bitmask, vocab_size,
        )
    except TypeError:
        data = TokenEnforcerTokenizerData(
            regular_tokens, decode_fn, tokenizer.eos_token_id,
        )
    _TOKENIZER_DATA_CACHE[key] = data
    return data


class _TransformersPrefixAllowedTokensFn:
    """Wrapper yang men-expose TokenEnforcer sebagai `prefix_allowed_tokens_fn`
    transformers-compatible: `fn(batch_id, input_ids_tensor) -> List[int]`."""

    def __init__(self, token_enforcer: Any) -> None:
        self.token_enforcer = token_enforcer

    def __call__(self, batch_id: int, sent: Any) -> List[int]:
        token_sequence = sent.tolist()
        allowed = self.token_enforcer.get_allowed_tokens(token_sequence)
        # 0.10.12 mengembalikan List[int] langsung; versi lain membungkusnya
        # dalam objek ber-atribut `.allowed_tokens`.
        return getattr(allowed, "allowed_tokens", allowed)


# ─────────────────────────────────────────────────────────────────────────────
# Builder: prefix_allowed_tokens_fn
# ─────────────────────────────────────────────────────────────────────────────
# [terjawab — investigasi]: TIDAK aktif. Dirangkai di client.run() hanya bila
#   `json_schema` dioper, namun tak ada satu pun pemanggil yang mengopernya;
#   construct memakai `json_mode` (ekstraksi pasca-generasi). Dead path — kandidat
#   hapus, atau pertahankan sebagai opsi constrained decoding di masa depan.
def build_guided_json_prefix_fn(
    tokenizer: Any,
    schema: Dict[str, Any],
) -> Callable[[int, Any], List[int]]:
    """
    Bangun `prefix_allowed_tokens_fn` yang meng-enforce `schema` pada output model.

    Fungsi kembalian di-pass ke `model.generate(prefix_allowed_tokens_fn=...)`.
    Di setiap step dekoder, ia dipanggil dengan (batch_id, generated_ids) dan
    mengembalikan daftar token_id yang boleh dipilih. Token di luar daftar ini
    ditolak via logits mask → secara struktural mustahil menghasilkan JSON
    yang menyimpang dari schema.

    Implementasi memakai core API `lmformatenforcer` (JsonSchemaParser +
    TokenEnforcer) tanpa melalui integrasi transformers bawaan (yang di versi
    transformers terbaru gagal import PreTrainedTokenizerBase).
    """
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.tokenenforcer import TokenEnforcer

    parser = JsonSchemaParser(schema)
    tokenizer_data = _build_token_enforcer_tokenizer_data(tokenizer)
    token_enforcer = TokenEnforcer(tokenizer_data, parser)
    return _TransformersPrefixAllowedTokensFn(token_enforcer)


# Registry nama → schema, supaya prompts.yaml bisa menunjuk schema lewat STRING
# (`json_schema: latent_construct`) tanpa menyalin JSON schema ke dalam YAML.
JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "latent_construct": LATENT_CONSTRUCT_JSON_SCHEMA,
    "rdagent_construct": CONSTRUCT_FACTOR_JSON_SCHEMA,
}


__all__ = [
    "CONSTRUCT_FACTOR_JSON_SCHEMA",
    "LATENT_CONSTRUCT_JSON_SCHEMA",
    "JSON_SCHEMAS",
    "build_guided_json_prefix_fn",
]
