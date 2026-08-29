#!/usr/bin/env bash
# Skoring IC korpus faktor di CPU — paralel, ber-checkpoint, bisa dilanjutkan.
#
# MASALAH YANG DIPECAHKAN. Skoring korpus butuh berjam-jam di CPU sementara
# GPU mengerjakan sel berikutnya, jadi ia praktis SELALU mati bersama pod.
# Sebelum 2026-08-28, `run_factor.py --score-only` (a) berjalan SERIAL di satu
# core sementara 15 core menganggur, dan (b) menulis hasilnya ATOMIK DI AKHIR —
# sehingga proses yang dibunuh di jam ke-3 kehilangan SELURUH pekerjaannya.
# Persis itu yang terjadi 2026-08-27: `text` (3 jam 16 mnt) dan `kv_gumbel`
# (2 jam 27 mnt) hilang total. Skrip ini memakai jalur `rescore_all.py` yang:
#   - paralel lewat fork copy-on-write (1 salinan data pasar untuk N pekerja;
#     terukur 2,4x lebih cepat di 3 pekerja),
#   - menyimpan cache ke disk setiap `--checkpoint` detik, DAN
#   - menangkap SIGTERM/SIGINT: saat pod dimatikan ia menyimpan progres lalu
#     menulis tag yang sudah lengkap, bukan mati membawa semuanya.
# Menjalankan perintah yang sama lagi MELANJUTKAN dari checkpoint terakhir.
#
#     bash scripts/skor_cpu.sh                 # latar belakang, sel belum diskor
#     bash scripts/skor_cpu.sh --status        # progres
#     bash scripts/skor_cpu.sh --stop          # berhenti RAPI (progres tersimpan)
#     bash scripts/skor_cpu.sh --tunggu        # depan (tidak background)
#     bash scripts/skor_cpu.sh --tags kv_gumbel,text
#     bash scripts/skor_cpu.sh --semua         # TERMASUK sel yang sudah diskor
set -u

AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AKAR"

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python
WORKERS="${WORKERS:-3}"        # 3 aman di mesin 8GB; 4 kalau >=12GB RAM
BUDGET="${BUDGET:-900}"        # WAJIB 900, bukan default 90 — lihat catatan bawah
CHECKPOINT="${CHECKPOINT:-180}"
LOG="results/logs/skor_cpu.log"
PIDF="results/logs/skor_cpu.pid"
TAGS=""; SEMUA=0; DEPAN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --status)
            if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
                echo "JALAN (pid $(cat "$PIDF"))"
            else
                echo "tidak jalan"
            fi
            echo "--- 15 baris terakhir $LOG ---"
            tail -15 "$LOG" 2>/dev/null || echo "(belum ada log)"
            exit 0 ;;
        --stop)
            if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
                # SIGTERM, BUKAN SIGKILL: penangan di rescore_all.py yang
                # menyimpan progres hanya berjalan kalau prosesnya diberi
                # kesempatan. `kill -9` membuang semuanya.
                kill -TERM "$(cat "$PIDF")"
                echo "SIGTERM terkirim — tunggu sampai log berkata 'selesai'."
                echo "Progres tersimpan; jalankan lagi untuk melanjutkan."
            else
                echo "tidak ada proses yang jalan"
            fi
            exit 0 ;;
        --tags)   TAGS="$2"; shift 2 ;;
        --semua)  SEMUA=1; shift ;;
        --tunggu) DEPAN=1; shift ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "argumen tak dikenal: $1"; exit 1 ;;
    esac
done

mkdir -p results/logs

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "SUDAH JALAN (pid $(cat "$PIDF")). Pakai --status atau --stop."
    exit 1
fi

# ── pilih tag ──────────────────────────────────────────────────────────────
# Default HANYA sel yang belum punya icseries_<tag>.parquet. Ini bukan
# kerapian: `rescore_all.py` MENIMPA field `ic` di frontend_<tag>.json untuk
# tag yang diprosesnya. Menimpa sel yang angkanya sudah dikutip dokumen
# (kv_soft/kv_sample/kv_raw) tanpa alasan berarti mempertaruhkan dasar Bab IV
# demi pekerjaan yang tak perlu.
if [ -z "$TAGS" ]; then
    TAGS=$("$PY" - "$SEMUA" <<'PY'
import pathlib, sys
semua = sys.argv[1] == "1"
d = pathlib.Path("results/factor")
out = []
for p in sorted(d.glob("frontend_*.json")):
    tag = p.stem[len("frontend_"):]
    if semua or not (d / f"icseries_{tag}.parquet").exists():
        out.append(tag)
print(",".join(out))
PY
)
fi

if [ -z "$TAGS" ]; then
    echo "Tidak ada sel yang perlu diskor. (Pakai --semua untuk memaksa skor ulang.)"
    exit 0
fi

echo "tag       : $TAGS"
echo "pekerja   : $WORKERS   anggaran: ${BUDGET}s/ekspresi   checkpoint: ${CHECKPOINT}s"
echo "log       : $LOG"
# --budget 900 dan bukan 90: pada anggaran ketat, ekspresi rolling berat
# (TS_SKEW/TS_KURT/TS_MAD/REGRESI) berada di bibir batas, sehingga ekspresi
# yang SAMA bisa ber-IC atau `ic=None` tergantung beban mesin — perbandingan
# antar-metode jadi bergantung pada hal yang tak ada kaitannya dengan metode.
if [ "$BUDGET" -lt 900 ]; then
    echo "⚠️  BUDGET=$BUDGET < 900. Hasilnya tidak sebanding dengan matriks yang ada."
fi

PERINTAH=("$PY" backend/eval/rescore_all.py --tags "$TAGS"
          --workers "$WORKERS" --budget "$BUDGET" --checkpoint-detik "$CHECKPOINT")

export PYTHONPATH=backend
if [ "$DEPAN" -eq 1 ]; then
    exec "${PERINTAH[@]}" 2>&1 | tee -a "$LOG"
fi

# Latar belakang. Perintahnya SEDERHANA (bukan rantai `cd ... && ...`) supaya
# `$!` benar-benar PID python. Kalau dibungkus rantai, `$!` adalah PID subshell
# bash dan `--stop` akan mengirim sinyal ke proses yang salah — pythonnya lalu
# mati tanpa sempat menyimpan progres. Terverifikasi 2026-08-28.
nohup "${PERINTAH[@]}" >>"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "dimulai di latar belakang, pid $(cat "$PIDF")"
echo
echo "  pantau   : bash scripts/skor_cpu.sh --status"
echo "  hentikan : bash scripts/skor_cpu.sh --stop     (progres TETAP tersimpan)"
