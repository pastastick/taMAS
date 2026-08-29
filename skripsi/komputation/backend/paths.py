"""Jalur kanonik proyek + bootstrap sys.path untuk skrip yang dijalankan langsung.

Sebelumnya tiap skrip di `lab/` menghitung sendiri `QL = __file__.parent.parent`
dan menaruh keluaran di `lab/out`. Setelah `lab/` dilebur ke `backend/`,
kedalaman berkas berubah per-modul sehingga perhitungan itu jadi sumber bug
diam-diam. Semua jalur sekarang berasal dari satu tempat: modul ini.

Layout:
    <QL_ROOT>/backend/      paket kode (harus ada di sys.path)
    <QL_ROOT>/results/      SEMUA keluaran run (gitignored kecuali probe/)
    <QL_ROOT>/reference/    kode rujukan LatentMAS & mixinputs (read-only)
    <QL_ROOT>/docs/         temuan (.md)

Pemakaian di skrip CLI (baris pertama sesudah import stdlib):

    from paths import bootstrap, RESULTS
    bootstrap()
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent            # <QL_ROOT>/backend
QL_ROOT = BACKEND.parent                             # <QL_ROOT>
RESULTS = QL_ROOT / "results"
REFERENCE = QL_ROOT / "reference"
DOCS = QL_ROOT / "docs"
CONFIGS = QL_ROOT / "configs"

PROMPTS_DIR = BACKEND / "prompts"
FACTOR_PROMPTS = PROMPTS_DIR / "factor.yaml"
BENCH_PROMPTS = PROMPTS_DIR / "bench.yaml"

# Sub-direktori keluaran per lengan eksperimen.
OUT_PROBE = RESULTS / "probe"      # kapasitas kanal, realign, b7 (artefak lama ikut di sini)
OUT_FACTOR = RESULTS / "factor"    # lengan faktor alpha
OUT_BENCH = RESULTS / "bench"      # lengan replikasi LatentMAS
CACHE = RESULTS / ".cache"


def bootstrap() -> None:
    """Pastikan `backend/` bisa di-import saat skrip dijalankan sebagai berkas.

    Direktori skrip sendiri dibuang dari sys.path lebih dulu: menjalankan
    `python backend/eval/channel_capacity.py` menaruh `backend/eval/` di posisi
    0, sehingga `ic.py`/`compare_modes.py` di sana bisa membayangi modul lain
    dengan nama sama. Yang tersisa hanya `backend/` dan root proyek.
    """
    here = str(Path(sys.argv[0]).resolve().parent) if sys.argv and sys.argv[0] else ""
    sys.path[:] = [p for p in sys.path if p not in ("", ".", here)]
    for p in (str(QL_ROOT), str(BACKEND)):
        if p not in sys.path:
            sys.path.insert(0, p)


def ensure_out(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
