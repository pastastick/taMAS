#!/usr/bin/env bash
# Kemas hasil eksperimen jadi arsip TERPISAH per lengan, untuk diunduh sebelum
# pod RunPod dihapus.
#
# Kenapa terpisah. Kedua lengan selesai pada waktu yang berbeda (faktor lebih
# dulu, bench belasan jam kemudian) dan diperiksa secara terpisah pula. Satu
# arsip gabungan memaksa menunggu keduanya selesai, dan setiap pemeriksaan
# ulang berarti mengunduh puluhan MB yang sudah dimiliki.
#
# Kenapa ini perlu padahal repo sudah di-push ke GitHub: `llm_outputs/`
# SENGAJA tidak di-track git (puluhan MB dan terus tumbuh), padahal isinya
# transkrip agen dari run yang sudah lewat — satu-satunya salinan. Begitu pod
# dihapus, ia hilang permanen.
#
#     bash scripts/kemas_hasil.sh faktor      # lengan faktor + pendukung
#     bash scripts/kemas_hasil.sh bench       # lengan bench + pendukung
#     bash scripts/kemas_hasil.sh semua       # dua arsip sekaligus
#     bash scripts/kemas_hasil.sh faktor --ringkas   # tanpa transkrip llm_outputs
set -u

# Akar repo diturunkan dari lokasi skrip, bukan dipatok ke
# /workspace/project/multi-agent-system: pod baru sering di-clone ke
# jalur lain dan patokan lama membuat skrip mengemas direktori yang salah
# (atau gagal senyap) tepat saat pod hendak dihapus.
AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AKAR" || exit 1
# Arsip ditaruh di induk repo kalau /workspace tak ada (mis. mesin lokal).
TUJUAN="/workspace"; [ -d "$TUJUAN" ] || TUJUAN="$(dirname "$AKAR")"
STAMP=$(date +%Y%m%d_%H%M)
APA="${1:-semua}"
RINGKAS=0
[ "${2:-}" = "--ringkas" ] && RINGKAS=1

# Yang dikecualikan hanyalah cache data pasar `pv_fast_*.parquet` (~58 MB):
# itu memang dibangkitkan ulang otomatis dari `backend/hf_data/daily_pv.h5`,
# jadi membawanya hanya memperbesar unduhan.
#
# BUKAN seluruh `.cache/`. Sejak rework checkpoint 2026-08-28, direktori yang
# sama juga menampung `rescore_cache.json` + `rescore_series.parquet` —
# 1004 ekspresi terskor, ~4,2 jam CPU pada 16 pekerja, dan TIDAK dibangkitkan
# ulang dari daily_pv.h5. `--exclude=.cache` yang lama membuangnya diam-diam
# tepat saat arsip dibuat untuk menyelamatkan isi pod.
EXCL="--exclude=pv_fast_*.parquet"
[ $RINGKAS -eq 1 ] && EXCL="$EXCL --exclude=llm_outputs"

kemas() {
    local nama="$1"; shift
    local out="${TUJUAN}/hasil_${nama}_${STAMP}.tar.gz"
    local ada=0 p
    for p in "$@"; do [ -e "$p" ] && ada=1; done
    if [ $ada -eq 0 ]; then
        echo "  (lewati $nama — belum ada isinya)"
        return
    fi
    tar czf "$out" $EXCL "$@" 2>/dev/null
    echo "  $out  ($(du -h "$out" | cut -f1))"
    tar tzf "$out" | awk -F/ '{print "     "$1"/"$2}' | sort -u | head -8
}

# Konteks yang menyertai KEDUA arsip: tanpa ini, angka di dalamnya tak bisa
# ditafsirkan ulang di kemudian hari (setelan matriks, dokumen desain, dan
# tabel pendukung yang diregenerasi dari artefak run).
KONTEKS="configs/matriks.yaml docs results/pendukung"

# Turunan + arsip yang HARUS ikut. results/ tidak di-track git, jadi arsip ini
# satu-satunya salinan begitu pod dihapus (§4 PANDUAN). Sebelum 2026-08-28,
# `semua` melewatkan results/visual/ — `finalisasi.sh --kemas` membangkitkan
# 24 figur Bab IV lalu tidak memasukkannya ke arsip mana pun. Arsip run lama
# juga tertinggal, padahal justru itu yang menjelaskan kenapa sel tertentu
# tidak ikut matriks.
TURUNAN="results/visual"
ARSIP=$(ls -d results/arsip_* 2>/dev/null | tr '\n' ' ')

echo "Mengemas (ringkas=$RINGKAS) ..."
case "$APA" in
    faktor|factor)
        kemas faktor results/factor $ARSIP $TURUNAN $KONTEKS ;;
    bench)
        kemas bench results/bench results/logs $KONTEKS ;;
    semua|all)
        kemas faktor results/factor $ARSIP $TURUNAN $KONTEKS
        kemas bench  results/bench results/logs $KONTEKS
        kemas probe  results/probe results/.cache ;;
    *)
        echo "pemakaian: bash scripts/kemas_hasil.sh {faktor|bench|semua} [--ringkas]"
        exit 1 ;;
esac

echo
echo "Unduh dari mesin lokalmu:"
echo "  scp -P <PORT> root@<HOST>:${TUJUAN}/hasil_*.tar.gz ."
echo "  atau lewat file browser RunPod di ${TUJUAN}/"
