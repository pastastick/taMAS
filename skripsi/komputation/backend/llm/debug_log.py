"""debug_log.py — riwayat percakapan sebagai tensor mentah (.pt), untuk debug.

Diekstrak apa adanya dari bekas `client.py` BAGIAN 2 ("CONV MANAGER"). Terpisah
dari `kv_store.py`: modul ini menyimpan `input_ids`/`output_ids`/`hidden_last`/
`latent_vecs` per langkah pipeline untuk decode-ulang manusiawi
(`TensorConvManager.decode_step`), sedangkan `kv_store.py` menyimpan KV-cache
itu sendiri untuk resume/transfer antar-agen — dua kebutuhan berbeda yang
kebetulan sama-sama berformat `.pt`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer


@dataclass
class ConvRecord:
    """
    Satu record percakapan yang disimpan ke disk.
    Menyimpan tensor mentah agar bisa di-decode ulang untuk debugging.

    Fields:
        conv_id       : ID unik percakapan
        step          : urutan langkah dalam pipeline (propose=0, construct=1, dst)
        role          : nama agen
        input_ids     : token ID prompt yang dikirim ke model  [1, seq_len]
        output_ids    : token ID yang dihasilkan model         [1, gen_len]
        hidden_last   : last hidden state setelah forward pass [1, hidden_dim]
                        (opsional, hanya jika output_hidden_states=True)
        latent_vecs   : semua latent vectors selama latent steps [steps, hidden_dim]
                        (opsional, hanya jika ada latent pass)
        metadata      : dict bebas untuk info tambahan
    """
    conv_id     : str
    step        : int
    role        : str
    input_ids   : torch.Tensor
    output_ids  : Optional[torch.Tensor] = None
    hidden_last : Optional[torch.Tensor] = None
    latent_vecs : Optional[torch.Tensor] = None
    metadata    : Dict[str, Any] = field(default_factory=dict)


class TensorConvManager:
    """
    Menyimpan riwayat percakapan dalam format tensor (.pt) ke direktori.

    Struktur direktori:
        conv_dir/
          {conv_id}/
            step_{n:03d}_{role}.pt      <- ConvRecord tersimpan sebagai dict tensor
            index.json                  <- metadata ringan (bisa dibaca tanpa load tensor)
    """

    def __init__(self, conv_dir: str = "./debug/conv_logs") -> None:
        self.conv_dir = Path(conv_dir)
        self.conv_dir.mkdir(parents=True, exist_ok=True)

    def _get_conv_path(self, conv_id: str) -> Path:
        p = self.conv_dir / conv_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, record: ConvRecord) -> Path:
        """Simpan satu ConvRecord ke disk. Return path file."""
        conv_path = self._get_conv_path(record.conv_id)
        filename  = f"step_{record.step:03d}_{record.role}.pt"
        filepath  = conv_path / filename

        payload = {
            "conv_id"    : record.conv_id,
            "step"       : record.step,
            "role"       : record.role,
            "input_ids"  : record.input_ids.cpu() if record.input_ids is not None else None,
            "output_ids" : record.output_ids.cpu() if record.output_ids is not None else None,
            "hidden_last": record.hidden_last.cpu() if record.hidden_last is not None else None,
            "latent_vecs": record.latent_vecs.cpu() if record.latent_vecs is not None else None,
            "metadata"   : record.metadata,
        }
        torch.save(payload, filepath)

        # Update index ringan (JSON)
        index_path = conv_path / "index.json"
        index = self._load_index(index_path)
        index.append({
            "step"       : record.step,
            "role"       : record.role,
            "file"       : filename,
            "has_text"   : record.output_ids is not None,
            "has_hidden" : record.hidden_last is not None,
            "has_latent" : record.latent_vecs is not None,
            "metadata"   : record.metadata,
        })
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        return filepath

    def load(self, conv_id: str, step: int, role: str) -> Optional[Dict[str, Any]]:
        """Load ConvRecord dari disk. Return dict atau None."""
        filepath = self._get_conv_path(conv_id) / f"step_{step:03d}_{role}.pt"
        if not filepath.exists():
            return None
        return torch.load(filepath, map_location="cpu")

    def list_steps(self, conv_id: str) -> List[Dict[str, Any]]:
        """Daftar langkah yang sudah disimpan untuk conv_id ini."""
        index_path = self._get_conv_path(conv_id) / "index.json"
        return self._load_index(index_path)

    @staticmethod
    def _load_index(index_path: Path) -> List[Dict[str, Any]]:
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        return []

    def decode_step(self, conv_id: str, step: int, role: str,
                    tokenizer: AutoTokenizer) -> Dict[str, str]:
        """
        Helper debug: decode tensor ke teks yang bisa dibaca manusia.

        Return dict berisi:
            input_text   : teks prompt
            output_text  : teks output (jika ada)
            hidden_norm  : norm dari last hidden state
            latent_norms : list norm tiap latent step
        """
        record = self.load(conv_id, step, role)
        if record is None:
            return {"error": f"Tidak ditemukan: {conv_id}/step_{step}_{role}"}

        result: Dict[str, str] = {}

        if record["input_ids"] is not None:
            result["input_text"] = tokenizer.decode(
                record["input_ids"][0], skip_special_tokens=True
            )
        if record["output_ids"] is not None:
            result["output_text"] = tokenizer.decode(
                record["output_ids"][0], skip_special_tokens=True
            )
        if record["hidden_last"] is not None:
            norm = record["hidden_last"].float().norm().item()
            result["hidden_norm"] = f"{norm:.4f}"
        if record["latent_vecs"] is not None:
            norms = record["latent_vecs"].float().norm(dim=-1).tolist()
            result["latent_norms"] = str([f"{n:.4f}" for n in norms])

        return result
