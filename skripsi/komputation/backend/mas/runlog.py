"""
mas/runlog.py
====================
Logging berbasis-file + timing per-step, dengan console yang TENANG.

Kenapa modul ini ada
--------------------
Keluhan nyata pada debug lama:
  1. Terlalu banyak log di CLI saat running → membingungkan.
  2. Timing hanya terasa untuk generate teks; waktu antar-step / backtest
     tidak kelihatan.
  3. Susah membaca ulang setelah run selesai.

`RunLogger` menyelesaikan ketiganya:
  - **Console tenang**: default hanya WARNING/ERROR + ringkasan step yang
    muncul ke layar. Detail lengkap tetap ditulis ke file.
  - **Timing semua step**: `with rl.step("backtest"): ...` membungkus blok
    APAPUN (generate teks, backtest, parsing, dll.) dan mencatat durasinya.
  - **Bisa dibaca ulang**: tiap run punya folder sendiri berisi
        run.log         — log human-readable kronologis
        events.jsonl    — event terstruktur (mudah di-parse)
        summary.json    — ringkasan timing per-step + bottleneck (saat finalize)

Pemakaian
---------
    from mas.runlog import get_run_logger

    rl = get_run_logger(run_name="exp_judger")     # buat / ambil singleton

    with rl.step("propose", direction_id=0):
        kv = proposal.run(...)

    with rl.step("backtest"):
        exp = runner.develop(...)

    rl.info("hypothesis generated", text_len=len(text))   # → file; console diam
    rl.warn("collapse detected, retrying")                # → file + console

    rl.finalize()   # tulis + cetak summary timing
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_LEVELS = {"DEBUG": 10, "INFO": 20, "STEP": 25, "WARNING": 30, "ERROR": 40}

# Anchor default run dir ke backend/ (parent dari mas/), bukan cwd —
# supaya `python -m pipeline.factor_mining` dari root project tetap menulis
# log di dalam backend, bukan di luar folder.
_DEFAULT_RUN_DIR = Path(__file__).resolve().parent.parent / "latent_runs"


@dataclass
class _StepStat:
    name: str
    count: int = 0
    total_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0
    errors: int = 0

    def add(self, dur: float, error: bool) -> None:
        self.count += 1
        self.total_s += dur
        self.min_s = min(self.min_s, dur)
        self.max_s = max(self.max_s, dur)
        if error:
            self.errors += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total_s": round(self.total_s, 3),
            "avg_s": round(self.total_s / self.count, 3) if self.count else 0.0,
            "min_s": round(self.min_s, 3) if self.count else 0.0,
            "max_s": round(self.max_s, 3),
            "errors": self.errors,
        }


class RunLogger:
    """Logger per-run: file detail + console ringkas + timing per-step."""

    def __init__(
        self,
        run_dir: str | Path = _DEFAULT_RUN_DIR,
        run_name: Optional[str] = None,
        console_level: str = "WARNING",
        tee_console_steps: bool = True,
        tee_logger: bool = False,
        nest_timestamp: bool = True,
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{run_name}" if run_name else ts
        # nest_timestamp=False → tulis langsung di run_dir/<run_name> (tanpa awalan
        # timestamp) supaya bisa di-anchor ke run dir terpadu (backend/runs/<id>/latent).
        self.dir = Path(run_dir) / (run_name or ts) if not nest_timestamp else Path(run_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)

        self._console_level = _LEVELS.get(console_level.upper(), 30)
        self._tee_steps = tee_console_steps
        # tee_logger: teruskan baris human-readable ke rdagent logger (loguru) →
        # masuk ke console.log run dir terpadu. Jadi SATU log lengkap (regulator +
        # pipeline + keputusan gate + repair) di console.log; events.jsonl tetap
        # untuk parsing terstruktur. Soft-import agar mas tetap standalone.
        self._rd = None
        if tee_logger:
            try:
                from qlog import logger as _rd  # rdagent logger wrapper
                self._rd = _rd
            except Exception:
                self._rd = None
        self._lock = threading.Lock()
        self._t0 = time.time()

        self._run_log = open(self.dir / "run.log", "a", buffering=1)
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)
        self._stats: Dict[str, _StepStat] = {}
        self._step_depth = 0

        self.info("run started", run_dir=str(self.dir), pid=os.getpid())

    # ── core writers ────────────────────────────────────────────────────────

    def _safe_write(self, attr: str, filename: str, text: str) -> None:
        """Tulis ke file log; JANGAN PERNAH melempar exception ke pipeline.

        Errno 5 (I/O error) pada volume network membuat file handle rusak
        permanen — insiden 2026-07-06 03:04: satu write runlog gagal →
        3 task evolution mati beruntun (MONITORING_NOTES B12). Di sini:
        coba tulis; bila gagal, reopen handle sekali; bila masih gagal,
        buang baris ini (logging best-effort, pipeline jalan terus).
        """
        try:
            getattr(self, attr).write(text)
            return
        except (OSError, ValueError):  # ValueError = write ke handle tertutup
            pass
        try:
            getattr(self, attr).close()
        except Exception:
            pass
        try:
            setattr(self, attr, open(self.dir / filename, "a", buffering=1))
            getattr(self, attr).write(text)
        except Exception:
            print(f"[runlog] gagal tulis {filename} (I/O); baris dibuang",
                  file=sys.stderr)

    def event(self, kind: str, **fields: Any) -> None:
        """Tulis satu event terstruktur ke events.jsonl."""
        rec = {"t": round(time.time() - self._t0, 4), "kind": kind, **fields}
        with self._lock:
            self._safe_write("_events", "events.jsonl",
                             json.dumps(rec, default=str) + "\n")

    def log(self, level: str, msg: str, **fields: Any) -> None:
        lvl = _LEVELS.get(level.upper(), 20)
        suffix = ("  " + " ".join(f"{k}={v}" for k, v in fields.items())) if fields else ""
        line = f"[{round(time.time() - self._t0, 2):>8.2f}s] {level:<7} {msg}{suffix}"
        with self._lock:
            self._safe_write("_run_log", "run.log", line + "\n")
            teed = False
            if self._rd is not None:
                # Teruskan ke rdagent logger → console.log terpadu (juga ke stderr
                # via sink konsol rdagent). Hindari double-print stderr di bawah.
                try:
                    meth = {"WARNING": "warning", "ERROR": "error",
                            "DEBUG": "debug"}.get(level.upper(), "info")
                    getattr(self._rd, meth)(f"[latent] {msg}{suffix}")
                    teed = True
                except Exception:
                    teed = False
            if not teed and lvl >= self._console_level:
                print(line, file=sys.stderr)
        self.event("log", level=level, msg=msg, **fields)

    def debug(self, msg: str, **f: Any) -> None: self.log("DEBUG", msg, **f)
    def info(self, msg: str, **f: Any) -> None: self.log("INFO", msg, **f)
    def warn(self, msg: str, **f: Any) -> None: self.log("WARNING", msg, **f)
    def error(self, msg: str, **f: Any) -> None: self.log("ERROR", msg, **f)

    # ── step timing ─────────────────────────────────────────────────────────

    @contextmanager
    def step(self, name: str, **ctx: Any):
        """Bungkus blok kode apapun; catat durasi + error.

        Berlaku untuk generate teks, backtest, parsing, IO — apapun. Inilah
        cara melihat "waktu antar step" yang sebelumnya tidak terukur.
        """
        self.event("step_start", step=name, **ctx)
        if self._tee_steps:
            indent = "  " * self._step_depth
            print(f"{indent}▶ {name} ...", file=sys.stderr)
        self._step_depth += 1
        t0 = time.time()
        err: Optional[BaseException] = None
        try:
            yield
        except BaseException as e:  # noqa: BLE001 — re-raised below
            err = e
            raise
        finally:
            dur = time.time() - t0
            self._step_depth -= 1
            self._stats.setdefault(name, _StepStat(name)).add(dur, err is not None)
            self.event(
                "step_end", step=name, duration_s=round(dur, 4),
                error=(repr(err) if err else None), **ctx,
            )
            if self._tee_steps:
                indent = "  " * self._step_depth
                tag = "✗" if err else "✓"
                print(f"{indent}{tag} {name}  {dur:.2f}s", file=sys.stderr)

    # ── summary / finalize ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        steps = {n: s.as_dict() for n, s in self._stats.items()}
        total = sum(s["total_s"] for s in steps.values())
        bottleneck = None
        if steps:
            bn = max(steps.items(), key=lambda kv: kv[1]["total_s"])
            bottleneck = {
                "step": bn[0],
                "total_s": bn[1]["total_s"],
                "pct": round(bn[1]["total_s"] / max(total, 1e-9) * 100, 1),
            }
        return {
            "wall_clock_s": round(time.time() - self._t0, 3),
            "summed_step_s": round(total, 3),
            "steps": steps,
            "bottleneck": bottleneck,
        }

    def finalize(self, print_table: bool = True) -> Dict[str, Any]:
        s = self.summary()
        (self.dir / "summary.json").write_text(json.dumps(s, indent=2))
        if print_table:
            self._print_table(s)
        self.info("run finished", wall_clock_s=s["wall_clock_s"])
        with self._lock:
            self._run_log.flush()
            self._events.flush()
        return s

    def _print_table(self, s: Dict[str, Any]) -> None:
        rows = sorted(s["steps"].items(), key=lambda kv: -kv[1]["total_s"])
        print("\n── timing summary " + "─" * 50, file=sys.stderr)
        print(f"{'step':<26}{'n':>4}{'total_s':>10}{'avg_s':>9}{'max_s':>9}{'err':>5}",
              file=sys.stderr)
        for name, st in rows:
            print(f"{name:<26}{st['count']:>4}{st['total_s']:>10.2f}"
                  f"{st['avg_s']:>9.2f}{st['max_s']:>9.2f}{st['errors']:>5}",
                  file=sys.stderr)
        bn = s["bottleneck"]
        if bn:
            print(f"\nbottleneck: {bn['step']} ({bn['total_s']:.1f}s, {bn['pct']}%)",
                  file=sys.stderr)
        print(f"wall clock: {s['wall_clock_s']:.1f}s   "
              f"(summed steps: {s['summed_step_s']:.1f}s)\n", file=sys.stderr)

    def close(self) -> None:
        with self._lock:
            self._run_log.close()
            self._events.close()


# ── singleton ────────────────────────────────────────────────────────────────

_GLOBAL: Optional[RunLogger] = None
_GLOBAL_LOCK = threading.Lock()


def get_run_logger(
    run_name: Optional[str] = None,
    run_dir: str | Path = _DEFAULT_RUN_DIR,
    console_level: Optional[str] = None,
    tee_logger: bool = False,
    nest_timestamp: bool = True,
) -> RunLogger:
    """Ambil RunLogger global (buat sekali). `console_level` bisa di-override
    via env LATENTMAS_CONSOLE_LEVEL (DEBUG/INFO/WARNING/ERROR). Bila pipeline sudah
    menyemai singleton lebih dulu (set_run_logger di output dir terpadu), call ini
    mengembalikannya — run_name di sini diabaikan."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            lvl = console_level or os.environ.get("LATENTMAS_CONSOLE_LEVEL", "WARNING")
            _GLOBAL = RunLogger(run_dir=run_dir, run_name=run_name, console_level=lvl,
                                tee_logger=tee_logger, nest_timestamp=nest_timestamp)
        return _GLOBAL


def set_run_logger(logger: Optional[RunLogger]) -> None:
    global _GLOBAL
    with _GLOBAL_LOCK:
        _GLOBAL = logger
