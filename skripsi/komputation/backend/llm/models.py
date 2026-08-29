"""
llm/models.py — ModelWrapper for Latent-MAS benchmark (core/latent/)
====================================================================

Peran file ini:
    Menyediakan ``ModelWrapper`` yang digunakan HANYA oleh
    ``core/latent/latent_method.py`` untuk menjalankan benchmark
    Latent-MAS (GSM8K, AIME, HumanEval, dll).

    ModelWrapper beroperasi di level TENSOR (input_ids, attention_mask)
    dan mendukung BATCH processing — cocok untuk benchmark yang
    memproses banyak soal sekaligus.

Hubungan dengan file lain di llm/:
    client.py   → ``LocalLLMBackend`` + ``_CoreEngine``
                  Dipakai oleh pipeline QuantaAlpha (factor mining).
                  Beroperasi di level MESSAGES (list[dict]) dan SINGLE
                  sequence. Mendukung kv_and_text, text_only, kv_only mode.

    _shared.py  → Utilitas bersama (LatentRealigner, KVCache helpers, dll).
                  Dipakai oleh KEDUA models.py dan client.py.

    models.py   → File ini. Hanya untuk benchmark Latent-MAS.

Shared logic (realignment, KV-cache helpers) didelegasikan ke _shared.py
untuk menghindari duplikasi.
"""

import os
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cache-only model load by default. Set HF_LOCAL_ONLY=0 to allow HF Hub fetches.
_HF_LOCAL_ONLY = os.environ.get("HF_LOCAL_ONLY", "1") not in ("0", "false", "False")

# ── Shared utilities from _shared.py (single source of truth) ─────────
from llm._shared import (
    _ensure_pad_token,
    _past_length,
    LatentRealigner,
)

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


class ModelWrapper:
    """
    Model wrapper untuk Latent-MAS benchmark.

    Dipakai oleh core/latent/latent_method.py untuk:
        - run_batch()      : multi-agent latent reasoning (HuggingFace)
        - run_batch_vllm() : multi-agent dengan vLLM decoding

    BUKAN untuk pipeline QuantaAlpha — gunakan LocalLLMBackend dari client.py.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        use_vllm: bool = False,
        args=None,
    ):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.args = args

        # Dual-device support for vLLM path (HF on device2, vLLM on device)
        self.HF_device = getattr(args, "device2", device) if args else device

        use_realign = bool(getattr(args, "latent_space_realign", False)) if args else False

        # Load model + tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=_HF_LOCAL_ONLY,
        )
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                local_files_only=_HF_LOCAL_ONLY,
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device).eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        # Realigner — didelegasikan ke LatentRealigner dari _shared.py
        self.realigner: Optional[LatentRealigner] = None
        if use_realign:
            self.realigner = LatentRealigner(
                self.model, device, use_realign=True,
            )

        # Embedding layer reference (for vLLM path)
        self.embedding_layer = self.model.get_input_embeddings()

    # ── Chat formatting ────────────────────────────────────────────────

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(self.device)

    # ── Realignment (delegasi ke LatentRealigner) ──────────────────────

    def _apply_latent_realignment(
        self, hidden: torch.Tensor, model: torch.nn.Module,
    ) -> torch.Tensor:
        """Apply latent realignment. Delegasi ke LatentRealigner dari _shared.py."""
        if self.realigner is None:
            # No realignment configured — just normalize magnitude
            # (sama dengan LatentRealigner(use_realign=False))
            return hidden
        return self.realigner.apply(hidden, model)

    @property
    def pre_aligned(self) -> Optional[torch.Tensor]:
        """Debug hook: pre-normalized aligned tensor dari realignment terakhir."""
        if self.realigner is not None:
            return self.realigner.pre_aligned
        return None

    # ── Text generation (batch) ────────────────────────────────────────

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len, past_len + input_ids.shape[-1],
                dtype=torch.long, device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    # ── Latent pass (batch) — inti Latent-MAS ──────────────────────────

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """
        Forward pass + N latent steps (batch).
        Tidak menghasilkan teks — hanya membangun KV-cache.

        Dipakai oleh LatentMASMethod.run_batch() untuk non-judger agents.

        Returns:
            Updated KV-cache (past_key_values).
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

        for _step in range(latent_steps):
            latent_vec = self._apply_latent_realignment(last_hidden, self.model)
            latent_embed = latent_vec.unsqueeze(1)  # [B, 1, D]

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long, device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[Tuple, torch.Tensor]:
        """
        Sama seperti generate_latent_batch, tapi juga mengembalikan
        hidden states dari semua token yang diproses (prompt + latent).

        Dipakai oleh LatentMASMethod.run_batch_vllm() untuk menyiapkan
        embeddings yang diteruskan ke vLLM engine.

        Returns:
            (updated_kv, hidden_embeddings [B, L_total, D])
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        # Hidden states dari prompt: [B, seq_len, D]
        prompt_hidden = outputs.hidden_states[-1]
        all_hidden = [prompt_hidden]

        last_hidden = prompt_hidden[:, -1, :]  # [B, D]

        for _step in range(latent_steps):
            latent_vec = self._apply_latent_realignment(last_hidden, self.model)
            latent_embed = latent_vec.unsqueeze(1)  # [B, 1, D]

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long, device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            all_hidden.append(outputs.hidden_states[-1])  # [B, 1, D]

        # Concat: [B, seq_len + latent_steps, D]
        combined_hidden = torch.cat(all_hidden, dim=1)
        return past, combined_hidden
