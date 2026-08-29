#!/usr/bin/env bash
# Regenerasi SELURUH analisis + 24 figur Bab IV, lalu kemas hasil.
#
# Dijalankan SETELAH data GPU final dan skoring IC selesai. Urutannya penting:
# figur membaca berkas pendukung, dan berkas pendukung membaca korpus. Menjalankan
# visual_bab4.py lebih dulu akan menghasilkan figur dari angka lama tanpa keluhan.
#
# KENAPA INI TIDAK BOLEH DILEWATI. Berkas di results/pendukung/ adalah turunan,
# dan turunan yang basi TIDAK memberi tanda apa pun. 2026-08-27 delapan berkas
# di sana masih memerikan korpus 174 ekspresi padahal matriksnya sudah 910 —
# angka yang salah, tanpa peringatan, siap masuk naskah. Skrip ini membangkitkan
# ulang semuanya dari korpus yang ada sekarang.
#
#     bash scripts/finalisasi.sh              # analisis + figur
#     bash scripts/finalisasi.sh --kemas      # + arsip tar.gz untuk diunduh
set -u

AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AKAR"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python
export PYTHONPATH=backend

gagal=0
jalankan() {
    local label="$1"; shift
    printf '\n=== %s ===\n' "$label"
    if "$@"; then :; else
        echo "!! GAGAL: $label"; gagal=$((gagal+1))
    fi
}

# ── 1. peringatan kalau datanya belum siap ────────────────────────────────
"$PY" - <<'PY'
import json, pathlib, sys
d = pathlib.Path("results/factor")
kurang, belum = [], []
for p in sorted(d.glob("frontend_*.json")):
    tag = p.stem[len("frontend_"):]
    if len(json.loads(p.read_text())["runs"]) < 20:
        kurang.append(tag)
    if not (d / f"icseries_{tag}.parquet").exists():
        belum.append(tag)
if kurang:
    print(f"⚠️  {len(kurang)} sel masih < 20 jalan: {kurang}")
    print("    Uji per-anggota di medium itu TIDAK BISA signifikan (n=8 -> p_min 0,0625).")
if belum:
    print(f"⚠️  {len(belum)} sel belum punya IC: {belum}")
    print("    Figur v13/v14/v16 dan Level 5-6 akan tidak lengkap.")
if not kurang and not belum:
    print("✓ korpus lengkap: semua sel 20 jalan & sudah diskor")
PY

# ── 2. analisis ───────────────────────────────────────────────────────────
# kumpulkan_pendukung.py dijalankan PERTAMA: ia yang membangkitkan ulang
# waktu_sel/gate_efektivitas/pemakaian_fungsi/ringkasan yang paling mudah basi.
jalankan "data pendukung"        "$PY" scripts/kumpulkan_pendukung.py
jalankan "per-hop lengan faktor" "$PY" scripts/faktor_perhop.py
jalankan "agregasi agent_trace"  "$PY" scripts/agregasi_agent_trace.py
jalankan "uji formal kv"         "$PY" scripts/kekuatan_uji_faktor.py --comm-mode kv
jalankan "uji formal kv_and_text" "$PY" scripts/kekuatan_uji_faktor.py --comm-mode kv_and_text
jalankan "geometri vs kinerja"   "$PY" scripts/analisis_geometri_kinerja.py

# Holdout hanya kalau korpusnya sudah diskor — ia berangkat dari ekspresi yang
# punya IC seleksi. Menjalankannya di korpus setengah jadi menghasilkan angka
# Level 6 yang tak sebanding dengan Level 5.
if [ "${LEWATI_HOLDOUT:-0}" != "1" ]; then
    jalankan "skor holdout 2022-2025" "$PY" backend/eval/skor_holdout.py --budget 900
fi

# ── 3. figur ──────────────────────────────────────────────────────────────
jalankan "24 figur Bab IV" "$PY" scripts/visual_bab4.py --all

echo
echo "=== figur terbangkit ==="
ls results/visual/*.png 2>/dev/null | wc -l | xargs printf '  %s dari 24\n'
for f in $(seq -w 1 24); do
    ls results/visual/v${f}_*.png >/dev/null 2>&1 || echo "  HILANG: v${f}"
done

# ── 4. kemas ──────────────────────────────────────────────────────────────
if [ "${1:-}" = "--kemas" ]; then
    printf '\n=== mengemas ===\n'
    bash scripts/kemas_hasil.sh semua
    echo
    echo "⚠️  results/ TIDAK di-track git. Arsip ini satu-satunya salinan"
    echo "    begitu pod dihapus — scp keluar SEBELUM menghapus pod."
fi

echo
[ "$gagal" -eq 0 ] && echo "SELESAI tanpa kegagalan." \
                   || echo "SELESAI dengan $gagal langkah GAGAL — baca di atas."
exit $gagal
