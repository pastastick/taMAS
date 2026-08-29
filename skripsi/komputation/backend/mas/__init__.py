"""
mas — LatentMAS × QuantaAlpha (rewrite, branch feat/latentmas-rework)
============================================================================
Pipeline alpha-mining berbasis kolaborasi laten murni (LatentMAS) dengan
evolusi trajectory (QuantaAlpha).

Modul:
  kv_ops    — operasi KV-cache yang benar (deepcopy/concat/save/load).  [butuh torch]
  runlog    — logging file + timing per-step, console tenang.            [tanpa torch]
  parsers   — parser output teks tiap agent.                            [tanpa torch]
  agent     — LatentAgent modular (bisa dijalankan standalone).         [butuh torch]
  pipeline  — orkestrator: front-end sequential + evolution.            [butuh torch]
  prompts.yaml — prompt semua agent.

Impor bersifat LAZY (PEP 562): `from mas import parse_repair` tidak akan
menarik torch. Hanya nama dari modul torch-dependent yang memicu impor torch.

Entry standalone:
  experiments/run_agent.py   — jalankan satu agent saja, dump teks + simpan KV.
  experiments/inspect_kv.py  — muat KV tersimpan, jalankan probe introspeksi.
"""

import importlib
from typing import Any

# nama publik → modul asalnya
_EXPORTS = {
    "kv_ops": "mas.kv_ops",
    "get_run_logger": "mas.runlog",
    "RunLogger": "mas.runlog",
    "LatentAgent": "mas.agent",
    "AgentSpec": "mas.agent",
    "AgentResult": "mas.agent",
    "load_agent": "mas.agent",
    "load_all_agents": "mas.agent",
    "FrontEndPipeline": "mas.pipeline",
    "FrontEndOutput": "mas.pipeline",
    "default_quality_gate": "mas.pipeline",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    mod_path = _EXPORTS.get(name)
    if mod_path is None:
        raise AttributeError(f"module 'mas' has no attribute {name!r}")
    mod = importlib.import_module(mod_path)
    return mod if name == "kv_ops" else getattr(mod, name)
