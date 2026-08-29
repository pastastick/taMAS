#!/usr/bin/env bash
# Periksa (dan siapkan) lingkungan pod sebelum menjalankan apa pun.
#
# Dipakai sebagai langkah PERTAMA di sesi cloud yang tidak punya konteks
# sebelumnya. Ia tidak mengubah apa-apa selain membuat venv & memasang
# dependensi kalau belum ada; sisanya hanya melapor. Setiap baris "TIDAK ADA"
# di keluarannya adalah alasan untuk berhenti dan bertanya, bukan untuk
# menjalankan eksperimen dan berharap.
#
#     bash scripts/setup_pod.sh
set -u

AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AKAR"
echo "repo      : $AKAR"
echo "commit    : $(git rev-parse --short HEAD 2>/dev/null || echo '(bukan repo git)')"
echo "cabang    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
echo

gagal=0
lapor() { # $1 = label, $2 = ok/tidak, $3 = detail
    if [ "$2" = "ok" ]; then printf '  ✓ %-26s %s\n' "$1" "$3"
    else printf '  ✗ %-26s %s\n' "$1" "$3"; gagal=$((gagal+1)); fi
}

# ── 1. venv ────────────────────────────────────────────────────────────────
if [ ! -x .venv/bin/python ]; then
    echo "[setup] .venv belum ada — membuatnya"
    python3 -m venv .venv || exit 1
    .venv/bin/pip install -q -U pip
    if [ -f requirements.txt ]; then
        .venv/bin/pip install -q -r requirements.txt
    else
        echo "  ⚠️  requirements.txt tidak ada — pasang dependensi manual"
    fi
fi
PY=.venv/bin/python
echo "== lingkungan =="
lapor "python" ok "$($PY -V 2>&1)"

for m in torch transformers pandas numpy pyarrow scipy; do
    v=$($PY -c "import $m;print(getattr($m,'__version__','?'))" 2>/dev/null)
    [ -n "$v" ] && lapor "$m" ok "$v" || lapor "$m" tidak "TIDAK TERPASANG"
done

# ── 2. GPU ─────────────────────────────────────────────────────────────────
echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader \
        | sed 's/^/  /'
    bebas=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    # Sel faktor memakai ~21GB (max_new_tokens 4096 x rantai 3 agen); 2 slot
    # aman di A40 46GB, 3 slot = OOM. Terukur 2026-08-10.
    if [ "$bebas" -ge 45000 ]; then lapor "slot faktor muat" ok "2 slot (~21GB/sel)"
    elif [ "$bebas" -ge 24000 ]; then lapor "slot faktor muat" ok "1 slot saja"
    else lapor "VRAM bebas" tidak "${bebas}MiB — terlalu sedikit untuk 1 sel"; fi
else
    lapor "nvidia-smi" tidak "TIDAK ADA — lengan GPU tak bisa dijalankan"
fi

# ── 3. data & model ────────────────────────────────────────────────────────
echo
echo "== data =="
[ -f backend/hf_data/daily_pv.h5 ] \
    && lapor "daily_pv.h5" ok "$(du -h backend/hf_data/daily_pv.h5 | cut -f1)" \
    || lapor "daily_pv.h5" tidak "TIDAK ADA — skoring IC mustahil"
n_cache=$(ls results/.cache/pv_fast_*.parquet 2>/dev/null | wc -l)
[ "$n_cache" -gt 0 ] \
    && lapor "cache data pasar" ok "$n_cache jendela (hemat ~10 mnt/jendela)" \
    || lapor "cache data pasar" ok "belum ada (akan dibangun otomatis, lambat sekali di awal)"

hf="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -d "$hf" ]; then
    lapor "cache HF" ok "$hf ($(du -sh "$hf" 2>/dev/null | cut -f1))"
else
    lapor "cache HF" ok "kosong — Qwen3-8B (~16GB) akan diunduh saat sel pertama"
fi

# ── 4. status matriks ──────────────────────────────────────────────────────
echo
echo "== status lengan faktor =="
if [ -d results/factor ]; then
    $PY - <<'PY'
import json, pathlib
d = pathlib.Path("results/factor")
baris = []
for p in sorted(d.glob("frontend_*.json")):
    tag = p.stem[len("frontend_"):]
    try:
        runs = json.loads(p.read_text())["runs"]
    except Exception as e:
        baris.append((tag, -1, "RUSAK: " + type(e).__name__, "")); continue
    skor = (d / f"icseries_{tag}.parquet").exists()
    baris.append((tag, len(runs), "20 ✓" if len(runs) >= 20 else f"{len(runs)} ⚠️",
                  "IC ✓" if skor else "IC —"))
for tag, n, jalan, ic in baris:
    print(f"  {tag:26} {jalan:>6}  {ic}")
kurang = [b[0] for b in baris if 0 <= b[1] < 20]
belum  = [b[0] for b in baris if b[3] == "IC —"]
print(f"\n  sel < 20 jalan : {len(kurang)}  {kurang}")
print(f"  sel belum diskor: {len(belum)}  {belum}")
PY
else
    lapor "results/factor" tidak "TIDAK ADA — apakah results/ sudah di-scp masuk?"
fi

echo
if [ "$gagal" -eq 0 ]; then
    echo "SIAP. Lanjutkan ke scripts/PANDUAN.md §3."
else
    echo "ADA $gagal MASALAH di atas — baca scripts/PANDUAN.md §2 sebelum lanjut."
fi
exit 0
