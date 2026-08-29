"""Apakah matriks realignment ridge benar-benar melakukan sesuatu?

LatentRealigner (backend/llm/_shared.py:696) menyelesaikan
    M = (W_out^T W_out + λI)^{-1} W_out^T W_in

Bila backbone punya `tie_word_embeddings: true` (Qwen3-4B, GPT-2) maka
W_out IS W_in → secara aljabar
    M = (W^T W + λI)^{-1} W^T W = V diag(σ²/(σ²+λ)) V^T  ≈  I,
sehingga ablasi `use_realign` BUKAN ablasi (kedua cabang identik).

Bila backbone TIDAK tied (Qwen3-8B/14B: ada `lm_head.weight` terpisah) maka M
betul-betul memetakan dua ruang berbeda dan ablasi itu bermakna. Skrip ini
mengukur mana yang berlaku untuk model yang diberikan.

Ia memuat HANYA matriks embedding dari safetensors (tanpa memuat model penuh).

    PYTHONPATH=backend python backend/eval/realign_probe.py --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ensure_out, OUT_PROBE as _OUT

OUT = ensure_out(_OUT)
LAMBDA = 1e-5  # backend/llm/config default reg_lambda

EMBED_KEY = "model.embed_tokens.weight"
HEAD_KEY = "lm_head.weight"


def _hub_root() -> Path:
    for env in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        if os.environ.get(env):
            return Path(os.environ[env])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache/huggingface/hub"


def _snapshot(model_id: str) -> Path:
    root = _hub_root() / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not root.is_dir():
        raise SystemExit(f"snapshot tidak ditemukan: {root}")
    return sorted(root.iterdir())[-1]


def load_matrices(model_id: str) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Kembalikan (W_in, W_out, tied). W_out = W_in bila checkpoint tied."""
    from safetensors import safe_open

    snap = _snapshot(model_id)
    index = snap / "model.safetensors.index.json"
    if index.exists():
        wmap = json.loads(index.read_text())["weight_map"]
    else:  # checkpoint satu shard
        single = next(snap.glob("*.safetensors"))
        with safe_open(single, framework="pt") as f:
            wmap = {k: single.name for k in f.keys()}

    def get(key: str) -> torch.Tensor:
        with safe_open(snap / wmap[key], framework="pt") as f:
            return f.get_tensor(key)

    cfg = json.loads((snap / "config.json").read_text())
    tied_flag = bool(cfg.get("tie_word_embeddings", False))
    W_in = get(EMBED_KEY)
    has_head = HEAD_KEY in wmap
    print(f"[probe] {model_id}: {EMBED_KEY} {tuple(W_in.shape)} {W_in.dtype}")
    print(f"[probe] config tie_word_embeddings={tied_flag} | "
          f"{HEAD_KEY} ada di checkpoint? {has_head}")
    if tied_flag or not has_head:
        return W_in, W_in, True
    return W_in, get(HEAD_KEY), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    W_in_raw, W_out_raw, tied = load_matrices(args.model)
    W_in = W_in_raw.to(torch.float32)
    W_out = W_in if tied else W_out_raw.to(torch.float32)
    V, d = W_in.shape

    gram = W_out.T @ W_out                  # [d, d]
    rhs = W_out.T @ W_in                    # = gram bila tied
    M = torch.linalg.solve(gram + LAMBDA * torch.eye(d), rhs)

    eye = torch.eye(d)
    dev_f = (M - eye).norm().item()
    rel = dev_f / eye.norm().item()
    # sudut yang dibuat M pada vektor acak berdistribusi seperti hidden state
    torch.manual_seed(0)
    h = torch.randn(512, d)
    h = h / h.norm(dim=1, keepdim=True)
    hm = h @ M
    cos = torch.nn.functional.cosine_similarity(h, hm, dim=1)
    scale = hm.norm(dim=1) / h.norm(dim=1)

    evals = torch.linalg.eigvalsh(gram)     # σ² dari W_out
    shrink = evals / (evals + LAMBDA)

    target_norm = W_in.norm(dim=1).mean().item()

    res = {
        "model": args.model, "tied": tied,
        "vocab": V, "d_h": d, "reg_lambda": LAMBDA,
        "frobenius_M_minus_I": dev_f,
        "relative_deviation": rel,
        "cos(h, hM)_mean": cos.mean().item(),
        "cos(h, hM)_min": cos.min().item(),
        "norm_ratio_mean": scale.mean().item(),
        "gram_eig_min": evals.min().item(),
        "gram_eig_max": evals.max().item(),
        "shrink_min": shrink.min().item(),
        "shrink_mean": shrink.mean().item(),
        "target_norm(mean ||W_in[i]||)": target_norm,
        "embed_norm_std": W_in.norm(dim=1).std().item(),
    }
    if not tied:
        # seberapa mirip kedua ruang sebelum dipetakan (kalau W_out ≈ W_in,
        # M akan tetap ≈ I meski checkpoint menyimpan dua matriks terpisah)
        cos_rows = torch.nn.functional.cosine_similarity(W_in, W_out, dim=1)
        res["cos(W_in_row, W_out_row)_mean"] = cos_rows.mean().item()
        res["cos(W_in_row, W_out_row)_median"] = cos_rows.median().item()

    print(json.dumps(res, indent=2))
    tag = args.model.replace("/", "_")
    (OUT / f"realign_probe_{tag}.json").write_text(json.dumps(res, indent=2))
    print("\nKESIMPULAN:")
    print(f"  M menyimpang dari identitas sebesar {rel*100:.4f}% (relatif Frobenius).")
    print(f"  Vektor acak diputar rata-rata cos={cos.mean():.6f} (1.0 = tidak diputar).")
    if rel < 1e-3:
        print("  → realignment ridge = identitas + penskalaan norma; "
              "ablasi use_realign TIDAK bermakna pada backbone ini.")
    else:
        print("  → realignment ridge betul-betul memetakan output→input space; "
              "ablasi use_realign BERMAKNA pada backbone ini.")


if __name__ == "__main__":
    main()
