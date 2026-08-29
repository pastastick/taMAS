"""Loader tiga benchmark lengan replikasi LatentMAS.

Satu benchmark per kategori, sesuai keputusan desain skripsi (lihat
`docs/DESAIN_EKSPERIMEN.md` §2):

    math      GSM8K              openai/gsm8k, config "main", split test (1319 soal)
    choice    ARC-Challenge      allenai/ai2_arc, split test (1172 soal)
    code      HumanEval+         evalplus/humanevalplus, split test (164 soal)

Diport dari `reference/LatentMAS/data.py` @ 9a9e4d3 dan SENGAJA dibuat identik
di bagian yang mempengaruhi angka: format pertanyaan, pemetaan label pilihan
ARC (1/2/3/4 → a/b/c/d), pembungkus prompt HumanEval+, dan cara `gold`
diturunkan. Yang berbeda hanya hal yang tak menyentuh angka: nama repo dataset
gsm8k dinaikkan ke `openai/gsm8k` (alias lama `gsm8k` sudah usang di Hub) dan
tiap item membawa `task_family` supaya prompt judger tahu format jawabannya.

`limit` + `seed` memberi subsample acak yang STABIL: benchmark penuh × 4 metode
× 3 medium × 3 benchmark adalah anggaran GPU yang tak realistis untuk skripsi
ini, jadi subsample-nya harus sama persis di seluruh sel supaya perbandingan
tetap BERPASANGAN (soal yang sama untuk semua metode) — sama seperti desain
berpasangan Tahap 0 di `docs/HASIL_TAHAP0.md`.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

# (nama HF, config, split, keluarga tugas)
TASKS = {
    "gsm8k":         ("openai/gsm8k",           "main",          "test", "math"),
    "arc_challenge": ("allenai/ai2_arc",        "ARC-Challenge", "test", "choice"),
    "humanevalplus": ("evalplus/humanevalplus", None,            "test", "code"),
}

_ARC_LABEL_MAP = {"1": "a", "2": "b", "3": "c", "4": "d"}

_CODE_PREAMBLE = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:\n"
)


def _normalize(ans: Optional[str]) -> Optional[str]:
    return None if ans is None else ans.strip().lower()


def _extract_gold_gsm8k(solution: str) -> Optional[str]:
    """Angka sesudah '####' di jawaban referensi GSM8K."""
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", solution)
    return m.group(1) if m else None


def _map_arc_label(label: str) -> str:
    s = str(label).strip()
    return _ARC_LABEL_MAP.get(s, s.lower())


def _load_gsm8k(cache_dir: Optional[str]) -> List[Dict]:
    from datasets import load_dataset

    name, config, split, family = TASKS["gsm8k"]
    ds = load_dataset(name, config, split=split, cache_dir=cache_dir)
    return [
        {
            "question": it["question"].strip(),
            "solution": it["answer"],
            "gold": _normalize(_extract_gold_gsm8k(it["answer"])),
            "task_family": family,
        }
        for it in ds
    ]


def _load_arc_challenge(cache_dir: Optional[str]) -> List[Dict]:
    from datasets import load_dataset

    name, config, split, family = TASKS["arc_challenge"]
    ds = load_dataset(name, config, split=split, cache_dir=cache_dir)
    out = []
    for it in ds:
        labels = it["choices"]["label"]
        texts = it["choices"]["text"]
        mapped = [_map_arc_label(lab) for lab in labels]
        lines = [f"{lab}: {txt.strip()}" for lab, txt in zip(mapped, texts)]
        answer = _map_arc_label(it.get("answerKey", "").strip()) if it.get("answerKey") else ""
        out.append({
            "question": it["question"].strip() + "\n" + "\n".join(lines),
            "solution": answer,
            "gold": _normalize(answer),
            "task_family": family,
        })
    return out


def _load_humanevalplus(cache_dir: Optional[str]) -> List[Dict]:
    from datasets import load_dataset

    name, config, split, family = TASKS["humanevalplus"]
    ds = load_dataset(name, config, split=split, cache_dir=cache_dir)
    out = []
    for it in ds:
        # `gold` di sini BUKAN jawaban, melainkan harness tes yang akan
        # dieksekusi setelah kode model ditempel di depannya (lihat
        # bench/scoring.py). Ini persis konvensi repo LatentMAS.
        test = str(it["test"]).replace("candidate", it["entry_point"])
        test += f'\n\ncheck({it["entry_point"]})'
        out.append({
            "question": _CODE_PREAMBLE + it["prompt"] + "\n",
            "solution": test,
            "gold": test,
            "task_family": family,
        })
    return out


_LOADERS = {
    "gsm8k": _load_gsm8k,
    "arc_challenge": _load_arc_challenge,
    "humanevalplus": _load_humanevalplus,
}


def load_task(
    task: str,
    *,
    limit: Optional[int] = None,
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """Muat satu benchmark; `limit` mengambil subsample acak yang reproducible.

    Subsample dipilih dengan RNG terpisah yang HANYA bergantung pada
    (task, seed, limit) — bukan pada `random` global — supaya seed generasi LLM
    boleh diubah tanpa menggeser soal yang dipilih. Itu syarat perbandingan
    berpasangan antar-metode.
    """
    if task not in _LOADERS:
        raise ValueError(f"task harus salah satu dari {sorted(_LOADERS)}, dapat {task!r}")
    items = _LOADERS[task](cache_dir)
    if limit is not None and 0 < limit < len(items):
        rng = random.Random(f"{task}:{seed}:{limit}")
        idx = sorted(rng.sample(range(len(items)), limit))
        items = [items[i] for i in idx]
    for i, it in enumerate(items):
        it["index"] = i
    return items


def task_family(task: str) -> str:
    return TASKS[task][3]
