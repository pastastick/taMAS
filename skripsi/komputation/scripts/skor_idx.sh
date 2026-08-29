#!/usr/bin/env bash
# Skoring korpus faktor di panel IDX (LQ45) — CPU, dua jendela.
#
# Tidak menyentuh frontend_*.json sama sekali (itu angka A-share yang menopang
# Bab IV sekarang); `skor_holdout.py` bekerja pada SALINAN dan menulis
# berkasnya sendiri: results/factor/holdout_<pasar>_<awal>_<akhir>_q<q>.json
set -eu
AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$AKAR"

export LAB_PV_FILE="${LAB_PV_FILE:-backend/hf_data_id/daily_pv_idx_lq45.h5}"
export LAB_TRADING_DAYS="${LAB_TRADING_DAYS:-241}"
export PYTHONPATH=backend
PY=.venv/bin/python
Q="${Q:-0.2}"
W="${W:-3}"

for JENDELA in "2021-01-01,2021-12-31" "2022-01-01,2025-12-26"; do
    echo "=============================================================="
    echo "jendela $JENDELA · panel $LAB_PV_FILE · kuantil $Q"
    echo "=============================================================="
    "$PY" backend/eval/skor_holdout.py --window "$JENDELA" \
          --quantile "$Q" --budget 900 --workers "$W"
done
echo "SEMUA SELESAI"
