"""Rantai multi-agen SEKUENSIAL untuk lengan benchmark: planner → critic → refiner → judger.

Ini padanan `mas/pipeline.py` untuk tugas benchmark. Rantainya dipisah, bukan
dipakai ulang, karena `FrontEndPipeline` terikat erat ke tugas faktor —
regulator gate, repair ekspresi, parser JSON konstruksi, alpha-zoo. Tak satu pun
punya arti di GSM8K. Yang DIPAKAI ULANG adalah lapisan yang memang umum:
`mas.agent.LatentAgent` (render Jinja + panggil backend + KV), `mas.kv_ops`
(deepcopy KV), dan `llm.client.LocalLLMBackend` (mesin + empat persamaan
langkah laten). Dengan begitu kedua lengan skripsi memakai mesin laten yang
sama persis, sehingga perbedaan hasil tidak bisa dituduh berasal dari
implementasi yang berbeda.

Semantik medium (comm_mode) dijaga IDENTIK dengan `FrontEndPipeline._agent_mode`
supaya kedua lengan bisa dibaca di satu tabel:

    text        semua agen text_only; handoff = teks agen hulu; past_kv selalu None
    kv_and_text semua agen kv_and_text; handoff = KV, teks tetap dikeluarkan
    kv          planner/critic/refiner kv_only (laten murni, tak emit teks),
                judger kv_and_text — judger WAJIB mengeluarkan teks karena
                jawabannya yang dinilai

`baseline` bukan comm_mode melainkan susunan rantai: hanya judger, tanpa agen
hulu — lantai pembanding "agen tunggal" seperti `methods/baseline.py` di repo
LatentMAS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mas.agent import AgentResult, LatentAgent, _load_specs

COMM_MODES = ("text", "kv_and_text", "kv")
DEFAULT_CHAIN = ("planner", "critic", "refiner", "judger")
BASELINE_CHAIN = ("judger",)

# Mode NATIF tiap agen sebelum comm_mode diterapkan. Judger sengaja tidak
# pernah kv_only — lihat docstring modul.
_NATIVE_MODE = {
    "planner": "kv_and_text",
    "critic": "kv_and_text",
    "refiner": "kv_and_text",
    "judger": "kv_and_text",
}
_KV_ONLY_CAPABLE = ("planner", "critic", "refiner")


@dataclass
class BenchOutput:
    """Hasil satu soal dilewatkan satu rantai."""
    answer_text: str
    agents: List[Dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer_text.strip())


class BenchPipeline:
    """Rantai sekuensial satu-soal untuk benchmark ala LatentMAS."""

    def __init__(
        self,
        backend: Any,
        *,
        comm_mode: str = "kv",
        chain: tuple = DEFAULT_CHAIN,
        prompts_path: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        text_context_chars: int = 8000,
        runlog: Any = None,
    ) -> None:
        if comm_mode not in COMM_MODES:
            raise ValueError(f"comm_mode harus salah satu dari {COMM_MODES}, dapat {comm_mode!r}")
        from paths import BENCH_PROMPTS

        path = prompts_path or BENCH_PROMPTS
        specs = _load_specs(path)
        unknown = [c for c in chain if c not in specs]
        if unknown:
            raise KeyError(f"agen {unknown} tak ada di {path}; tersedia {sorted(specs)}")
        if chain[-1] != "judger":
            raise ValueError("agen terakhir rantai harus 'judger' — dialah yang menjawab")

        self.comm_mode = comm_mode
        self.chain = tuple(chain)
        # `LatentAgent.run(**vars)` meneruskan kwarg tak dikenal ke Jinja, BUKAN
        # ke backend — anggaran token hanya bisa disetel lewat spec. Judger yang
        # dipotong di tengah kehilangan `\boxed{}`-nya dan dinilai salah padahal
        # penalarannya benar, jadi batasnya disetel di sini, sekali, di spec.
        if max_new_tokens is not None:
            for spec in specs.values():
                spec.max_new_tokens = max_new_tokens
        self.max_new_tokens = max_new_tokens
        # Batas panjang konteks teks: `text_mas` di repo rujukan memotong konteks
        # ke `--text_mas_context_length` (default 8000 karakter). Tanpa batas ini
        # mode `text` mengoper seluruh riwayat dan panjang promptnya meledak di
        # akhir rantai — bukan karena desainnya, tapi karena lupa dipotong.
        self.text_context_chars = text_context_chars
        self.runlog = runlog
        self.agents: Dict[str, LatentAgent] = {
            name: LatentAgent(specs[name], backend, strict_vars=False, runlog=runlog)
            for name in self.chain
        }

    # ── medium ──────────────────────────────────────────────────────────────
    @property
    def _is_text(self) -> bool:
        return self.comm_mode == "text"

    def _agent_mode(self, name: str) -> str:
        if self._is_text:
            return "text_only"
        if self.comm_mode == "kv" and name in _KV_ONLY_CAPABLE:
            return "kv_only"
        return _NATIVE_MODE.get(name, "kv_and_text")

    # ── eksekusi ────────────────────────────────────────────────────────────
    def run_item(self, item: Dict[str, Any]) -> BenchOutput:
        """Jalankan satu soal melalui rantai; kembalikan jawaban judger."""
        t0 = time.time()
        question = item["question"]
        family = item.get("task_family", "math")
        parts: List[str] = []
        prev_kv = None
        traces: List[Dict[str, Any]] = []
        res: Optional[AgentResult] = None

        try:
            for i, name in enumerate(self.chain):
                # Agen pertama tak punya hulu → selalu membaca soal sebagai teks.
                eff_handoff = "text" if (i == 0 or self._is_text) else "kv"
                context = ""
                if self._is_text and parts:
                    context = "\n\n".join(parts)[: self.text_context_chars]
                kw: Dict[str, Any] = {
                    "question": question,
                    "context": context,
                    "handoff": eff_handoff,
                    "task_family": family,
                }
                res = self.agents[name].run(
                    past_kv=None if self._is_text else prev_kv,
                    mode_override=self._agent_mode(name),
                    **kw,
                )
                prev_kv = res.kv_cache
                traces.append(res.describe())
                text = (res.text or "").strip()
                if text and name != "judger":
                    parts.append(f"[{name}]:\n{text}")
        except Exception as e:  # noqa: BLE001 — satu soal gagal ≠ seluruh run gagal
            return BenchOutput("", traces, time.time() - t0,
                               error=f"{type(e).__name__}: {e}")

        assert res is not None
        return BenchOutput((res.text or "").strip(), traces, time.time() - t0)
