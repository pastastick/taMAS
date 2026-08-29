#!/usr/bin/env bash
# /workspace/runpod_env.sh — sumber file ini di awal setiap session
# Di-copy ke /workspace/ oleh setup_runpod.sh

# uv & cargo binary location (uv installer default ke ~/.local/bin → ephemeral)
export XDG_DATA_HOME=/workspace/.local/share
export XDG_CONFIG_HOME=/workspace/.config
export XDG_CACHE_HOME=/workspace/.cache
export PATH=/workspace/.local/bin:$PATH

# uv cache & virtualenv
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.local/share/uv/python
export UV_TOOL_DIR=/workspace/.local/share/uv/tools

# pip cache (untuk fallback jika tidak pakai uv)
export PIP_CACHE_DIR=/workspace/.cache/pip

# HuggingFace model & dataset cache (default ~/.cache/huggingface → ephemeral)
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub

# HuggingFace token — SENGAJA TIDAK di-hardcode di sini lagi (2026-08-09).
# File ini di-track git; commit 33dd614 pernah menaruh token asli langsung di
# baris ini (`export HF_TOKEN="hf_otarf...gcQv"`) — token itu sekarang berstatus
# expired ("skripsi" is expired di Hub) dan sudah bocor ke riwayat git terlepas
# dari itu. Sebaiknya di-revoke di https://huggingface.co/settings/tokens kalau
# belum. Token yang masih berlaku ("skripsi2") ada di `.env` (git-ignored) dan
# sudah otomatis dimuat oleh launcher.py / lab/*.py via load_dotenv — TIDAK
# perlu diulang di sini. Kalau shell butuh HF_TOKEN untuk perintah `hf` manual,
# source .env langsung: `set -a; source .env; set +a`.
#
# Qwen3 (Apache-2.0) dan repo publik lain tetap bisa diunduh tanpa token sama
# sekali (akses anonim, lebih lambat/rate-limited) — jadi tidak fatal bila
# HF_TOKEN kosong di shell ini.

# Torch hub & inductor cache
export TORCH_HOME=/workspace/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor

# Izinkan transformers download model dari HF saat run pertama.
# Default kode adalah local_files_only=True (offline) — set 0 agar model
# Qwen3 ter-download otomatis jika belum ada di cache HF.
export HF_LOCAL_ONLY=0

# Project-specific
export PYTHONPATH=/workspace/project/multi-agent-system/backend
