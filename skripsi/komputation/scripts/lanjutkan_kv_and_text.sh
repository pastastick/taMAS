#!/usr/bin/env bash
# Naikkan tiga sel `kv_and_text` dari 8 jalan (seed 0,1) ke 20 jalan (seed 0-4).
#
# KENAPA CARA INI, BUKAN `--seeds 0,1,2,3,4` KE TAG ASLI. `run_factor.py`
# menulis `frontend_<tag>.json` sekali di akhir dan tidak punya resume: tag yang
# sama akan DITIMPA. Menjalankan ulang seed 0-1 berarti (a) membuang 24 jalan
# GPU yang sudah jadi, dan (b) membangkitkannya ulang dengan `--temperature 0.8`
# + batching vLLM yang tak menjamin reproduksi bit-per-bit — sehingga angka yang
# sudah terdokumentasi di docs/HASIL_TAHAP5.md bisa bergeser tanpa alasan
# ilmiah. Jadi: hanya seed 2,3,4 yang dijalankan, ke tag SEMENTARA, lalu
# digabung dengan scripts/gabung_jalan.py.
#
# Kenapa harus dinaikkan sama sekali: pada n=8 dengan efek sempurna
# (b01=5, b10=0), p terkecil yang MUNGKIN dari McNemar eksak adalah
# 2 x 0,5^5 = 0,0625 > 0,05. Ketiga sel itu tidak bisa signifikan berapa pun
# bagusnya hasilnya — itu batas resolusi rancangan, bukan sifat data. Lihat
# docs/HASIL_TAHAP5.md §1.1.
#
#   bash scripts/lanjutkan_kv_and_text.sh --dry-run   # lihat perintahnya saja
#   bash scripts/lanjutkan_kv_and_text.sh             # jalankan (GPU)
#   bash scripts/lanjutkan_kv_and_text.sh --gabung    # gabungkan setelah selesai
#
# Estimasi: 36 jalan. Rerata sel saat ini 102-154 dtk/jalan => ~79 menit
# serial, ~45 menit pada 2 slot. Skoring IC-nya CPU dan TIDAK dijalankan di
# sini (--skip-score) supaya kartu tidak ditahan pekerjaan non-GPU.
set -euo pipefail

AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AKAR"

# Interpreter: SAMA seperti skrip lain di repo (setup_pod.sh, skor_cpu.sh,
# jalankan_interpolasi.sh) — `.venv/bin/python` kalau ada. `python` telanjang
# TIDAK boleh dipakai: di pod, PATH bisa menunjuk venv lain (mis.
# /workspace/.dlvenv milik pengunduh HF) yang tak punya torch, dan selnya mati
# di baris impor setelah slot GPU terlanjur dipakai.
PY="${PY:-$AKAR/.venv/bin/python}"
[ -x "$PY" ] || PY=python

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SEEDS_KURANG="${SEEDS_KURANG:-2,3,4}"
SUFIKS="${SUFIKS:-_s234}"
SLOT="${SLOT:-2}"                 # sel GPU serentak; sel faktor ~21GB, A40 46GB
STAGGER="${STAGGER:-30}"          # jeda antar-start: fase muat model rebutan I/O
VRAM_MIN="${VRAM_MIN:-24000}"     # MiB bebas yang harus ada sebelum sel baru
LOGS="results/logs"

MODE=""
[[ "${1:-}" == "--dry-run" ]] && MODE="dry"
[[ "${1:-}" == "--gabung" ]] && MODE="gabung"

# latent_mode yang macet di 8 jalan. `soft` TIDAK ada di sini: ia sudah 20.
MODES=(sample gumbel moi)

perintah() {  # $1 = latent_mode, $2 = tag keluaran
    echo "PYTHONPATH=backend $PY backend/factor/run_factor.py \
--model $MODEL --comm-mode kv_and_text --latent-mode $1 \
--latent-steps 10 --latent-temp 0.7 \
--seeds $SEEDS_KURANG --directions d0,d1,opp_mom,opp_rev \
--chain proposal,innovate,construct --max-repair 3 \
--tag $2 --skip-score"
}

if [[ "$MODE" == "gabung" ]]; then
    for m in "${MODES[@]}"; do
        asli="kv_and_text_${m}"
        "$PY" scripts/gabung_jalan.py --keluaran "$asli" \
            --dari "$asli" "${asli}${SUFIKS}"
        echo
    done
    # Pecahan `_s234` harus KELUAR dari results/factor/ begitu tergabung.
    # Selama ia di sana, ia terhitung sebagai sel matriks tersendiri oleh setiap
    # pembaca korpus (kumpulkan_pendukung.py, agregasi_agent_trace.py,
    # faktor_perhop.py, eval/skor_holdout.py, eval/rescore_all.py) — padahal
    # jalan-jalannya sudah ikut masuk ke sel gabungan, jadi korpus tercacah dua
    # kali tanpa peringatan. Terjadi 2026-08-28: 36 jalan pecahan + 24 jalan
    # cadangan terbaca sebagai matriks.
    ARSIP="results/arsip_pecahan_gabung_$(date +%Y-%m-%d)"
    mkdir -p "$ARSIP"
    for m in "${MODES[@]}"; do
        for f in "results/factor/frontend_kv_and_text_${m}${SUFIKS}.json" \
                 "results/factor/icseries_kv_and_text_${m}${SUFIKS}.parquet"; do
            [[ -e "$f" ]] && mv "$f" "$ARSIP"/ && echo "arsip → $ARSIP/$(basename "$f")"
        done
    done
    echo

    echo "Setelah SEMUA tergabung, skor ulang ketiganya (CPU, boleh paralel):"
    for m in "${MODES[@]}"; do
        echo "  PYTHONPATH=backend $PY backend/factor/run_factor.py \
--score-only --tag kv_and_text_${m} --budget 900"
    done
    echo "Lalu: $PY scripts/kekuatan_uji_faktor.py --comm-mode kv_and_text"
    exit 0
fi

if [[ "$MODE" == "dry" ]]; then
    for m in "${MODES[@]}"; do
        echo "# sel: kv_and_text_${m}  (8 -> 20 jalan)"
        perintah "$m" "kv_and_text_${m}${SUFIKS}"
        echo
    done
    echo "# lalu gabungkan:"
    echo "bash scripts/lanjutkan_kv_and_text.sh --gabung"
    exit 0
fi

mkdir -p "$LOGS"

# Gerbang VRAM, bukan sekadar hitungan slot: sel faktor memakai ~21GB
# (max_new_tokens 4096 x rantai 3 agen), jadi 3 sel serentak = OOM di A40 46GB.
# Terukur 2026-08-10, docs/HASIL_TAHAP5.md + memory vram-sel-faktor-21gb.
tunggu_slot() {
    while true; do
        aktif=$(jobs -rp | wc -l)
        if (( aktif < SLOT )); then
            bebas=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 999999)
            if (( bebas >= VRAM_MIN )); then return 0; fi
            echo "[tunggu] VRAM bebas ${bebas}MiB < ${VRAM_MIN}MiB"
        fi
        sleep 20
    done
}

mulai=$(date +%s)
for m in "${MODES[@]}"; do
    tag="kv_and_text_${m}${SUFIKS}"
    keluaran="results/factor/frontend_${tag}.json"
    if [[ -f "$keluaran" ]]; then
        echo "[lewati] $tag sudah ada -> $keluaran"
        continue
    fi
    tunggu_slot
    log="$LOGS/frontend_${tag}.log"
    echo "[mulai ] $tag  (log: $log)"
    eval "$(perintah "$m" "$tag")" >"$log" 2>&1 &
    sleep "$STAGGER"
done

wait
echo "[selesai] ${#MODES[@]} sel dalam $(( ($(date +%s) - mulai) / 60 )) menit"

echo
echo "Periksa dulu sebelum menggabung — tiap sel harus 12 jalan (3 seed x 4 arah):"
for m in "${MODES[@]}"; do
    tag="kv_and_text_${m}${SUFIKS}"
    "$PY" - "$tag" <<'PY'
import json, sys
from pathlib import Path
tag = sys.argv[1]
p = Path("results/factor") / f"frontend_{tag}.json"
if not p.exists():
    print(f"  {tag:32} TIDAK ADA — sel gagal, lihat results/logs/")
    raise SystemExit
runs = json.loads(p.read_text())["runs"]
lolos = sum(1 for r in runs if r.get("passing"))
err = sum(1 for r in runs if r.get("error"))
print(f"  {tag:32} {len(runs):2d} jalan  lolos {lolos}  error {err}  "
      f"seed={sorted({r['seed'] for r in runs})}")
PY
done
echo
echo "Kalau ketiganya 12 jalan tanpa error:  bash scripts/lanjutkan_kv_and_text.sh --gabung"
