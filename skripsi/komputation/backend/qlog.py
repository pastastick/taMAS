"""qlog — logger tunggal proyek.

Menggantikan paket `log/` lama yang membungkus `rdagent.log` (RD-Agent). Modul
lama itu memasang sink file, storage pickle, dan `trace_path` milik RD-Agent —
seluruhnya ikut mati bersama pipeline evolusi yang dihapus di branch ini. Yang
benar-benar dipakai kode yang tersisa hanya `logger.{info,warning,error,debug}`
(diperiksa dengan grep sebelum penghapusan), jadi loguru polos sudah cukup dan
proyek ini tidak lagi butuh RD-Agent terpasang sama sekali.

Sink file opsional: set `QUANTA_RUN_DIR` sebelum import mana pun, dan seluruh
baris log ikut ditulis ke `$QUANTA_RUN_DIR/console.log` dengan format default
loguru — sama persis dengan yang tampil di terminal.

    from qlog import logger
    logger.info("...")
"""
from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

_run_dir = os.getenv("QUANTA_RUN_DIR")
if _run_dir:
    try:
        Path(_run_dir).mkdir(parents=True, exist_ok=True)
        logger.add(str(Path(_run_dir) / "console.log"), level="DEBUG", enqueue=True)
    except OSError:  # direktori tak bisa dibuat → cukup log ke stderr
        pass

__all__ = ["logger"]
