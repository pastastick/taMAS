#!/usr/bin/env bash
# Rantai GPU tanpa jeda: selesaikan lengan FAKTOR dulu, lalu lengan BENCH.
#
# Urutan ini disengaja. Sisa lengan faktor hanya ~30 menit sedangkan lengan
# bench ~16 jam; mendahulukan yang pendek berarti satu lengan sudah UTUH di
# tangan dalam setengah jam, alih-alih dua lengan yang sama-sama setengah jadi
# kalau pod mati di tengah malam. Total waktu GPU-nya identik.
#
# `--tanpa-skor-cpu` di kedua lengan: skoring/backtest tidak butuh GPU, dan
# sewa RunPod ditagih per WAKTU HIDUP instance — menahan pod hidup untuk
# pekerjaan CPU berarti membayar harga GPU untuk sesuatu yang gratis di laptop.
# Skoring dijalankan belakangan di mesin sendiri:
#     PYTHONPATH=backend python backend/eval/rescore_all.py --budget 900
#
#     bash scripts/jalankan_semua.sh
set -u

cd /workspace/project/multi-agent-system || exit 1
source /workspace/runpod_env.sh
set -a; source .env 2>/dev/null; set +a
PY=.venv/bin/python
ARGS="--slots 2 --vram-bebas-min 24000 --ulang-maks 1 --tanpa-skor-cpu"

echo "=========================================================="
echo " [1/2] LENGAN FAKTOR — sisa 5 sel (kv_and_text x 5 metode)"
echo "=========================================================="
$PY scripts/jalankan_matriks.py --arm factor $ARGS

echo
echo "=========================================================="
echo " [2/2] LENGAN BENCH — 36 sel, limit 100"
echo "=========================================================="
$PY scripts/jalankan_matriks.py --arm bench $ARGS

echo
echo "=========================================================="
echo " SELURUH PEKERJAAN GPU SELESAI — pod boleh dimatikan."
echo " Sisa pekerjaan (CPU, gratis di lokal):"
echo "   PYTHONPATH=backend python backend/eval/rescore_all.py --budget 900"
echo "   python backend/bench/compare.py --out results/bench/analisis.json"
echo "   python scripts/kumpulkan_pendukung.py"
echo "=========================================================="
echo "SEMUA-GPU-SELESAI"
