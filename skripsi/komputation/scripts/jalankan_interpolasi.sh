#!/usr/bin/env bash
# Sumbu C — interpolasi `mix` antara `raw` dan `soft`. 6 sel GPU.
#
# APA YANG DIUKUR. Lima formulasi memberi lima titik TERPISAH, jadi hubungan
# geometri-representasi <-> kinerja hanya bisa dibaca sebagai "searah". Mode
# `mix` mengisi jaraknya:  z(a) = normalisasi((1-a)*z_raw + a*z_soft).
# Yang diuji adalah BENTUK hubungan itu, dan hipotesisnya SENGAJA tak berarah:
# monoton, ber-ambang, atau tak berpola — ketiganya temuan, dan yang ketiga
# berarti klaim mekanistik Bab IV harus dilemahkan. Jangan menulis
# "makin dekat embedding makin baik" sebelum datanya ada.
#
# Titik ujung TIDAK dijalankan: a=0 identik bit-per-bit dengan `raw` dan a=1
# dengan `soft` (kedua ujung dihitung penuh lewat jalur mode aslinya di
# backend/llm/methods.py), keduanya sudah ada di matriks. Jadi hanya 3 nilai
# tengah yang makan GPU: 0,25 / 0,5 / 0,75.
#
# ⚠️ SUMBU-X SUDAH DIUKUR DAN TIDAK LINIER. Kurva geometri `mix`
# (results/probe/b7_probe_Qwen_Qwen3-8B.json -> geometry_mix):
#     a    0      0,25     0,5      0,75     1
#     cos  0,3120 0,4519   0,7197   0,9244   0,9269
# Ia JENUH: paling curam di 0,25->0,5, lalu praktis datar di 0,75->1. Artinya
# memplot kinerja terhadap `a` akan menyesatkan — sumbu-x harus cos terukur
# ini. Dan a=0,75 yang secara geometri sudah ~`soft` justru titik falsifikasi
# terkuat: kalau geometri memang penjelasnya, ia HARUS berkinerja seperti
# `soft`; kalau tidak, penjelasan mekanistiknya gugur.
#
#     bash scripts/jalankan_interpolasi.sh --dry-run
#     bash scripts/jalankan_interpolasi.sh
#
# Estimasi (dari durasi sel yang sudah ada): 3 sel faktor 20 jalan ~63 mnt +
# 3 sel bench humanevalplus limit 100 ~73 mnt = ~2 jam 15 mnt serial,
# ~1,5-2 jam pada 2 slot termasuk muat model 6x.
set -u

AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AKAR"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python
SLOTS="${SLOTS:-2}"

if [ "${1:-}" = "--dry-run" ]; then
    exec "$PY" scripts/jalankan_matriks.py --arm interpolasi --slots "$SLOTS" --dry-run
fi

# --tanpa-skor-cpu: skoring IC diserahkan SELURUHNYA ke scripts/skor_cpu.sh.
# Dua alasan. (1) Sewa GPU ditagih per waktu hidup instance; menahan pod untuk
# pekerjaan CPU berarti membayar harga GPU untuk kerja non-GPU. (2) Jalur
# skoring di dalam jalankan_matriks.py TIDAK ber-checkpoint — kalau pod mati di
# tengahnya, progresnya hilang. skor_cpu.sh ber-checkpoint dan bisa dilanjutkan.
echo "6 sel sumbu interpolasi, $SLOTS slot. Skoring CPU TIDAK dijalankan di sini."
echo "Jalankan  bash scripts/skor_cpu.sh  di terminal lain supaya CPU terpakai"
echo "selama GPU bekerja."
echo
"$PY" scripts/jalankan_matriks.py --arm interpolasi --slots "$SLOTS" --tanpa-skor-cpu
rc=$?

echo
echo "== hasil sel interpolasi =="
"$PY" - <<'PY'
import json, pathlib
f = pathlib.Path("results/factor"); b = pathlib.Path("results/bench")
for tag in ("kv_mix_a025", "kv_mix_a05", "kv_mix_a075"):
    p = f / f"frontend_{tag}.json"
    if not p.exists():
        print(f"  {tag:34} TIDAK ADA — lihat results/logs/"); continue
    runs = json.loads(p.read_text())["runs"]
    lolos = sum(1 for r in runs if r.get("passing"))
    print(f"  {tag:34} {len(runs):2d} jalan  lolos {lolos}/{len(runs)}")
for a in ("a025", "a05", "a075"):
    p = b / f"bench_humanevalplus_mix_kv_s0_{a}.json"
    if not p.exists():
        print(f"  bench humanevalplus {a:5}          TIDAK ADA"); continue
    s = json.loads(p.read_text())["summary"]
    print(f"  bench humanevalplus {a:5}          akurasi {s.get('accuracy')} "
          f"(n={s.get('n')})")
PY
echo
echo "Berikutnya: bash scripts/skor_cpu.sh   (3 sel faktor baru perlu IC)"
exit $rc
