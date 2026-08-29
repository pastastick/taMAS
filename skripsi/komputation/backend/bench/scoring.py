"""Penilaian jawaban benchmark — exact-match \\boxed{} dan pass@1 eksekusi kode.

Diport dari `reference/LatentMAS/{utils,methods/text_mas}.py` @ 9a9e4d3 dengan
logika penilaian dipertahankan persis: ambil isi `\\boxed{}` TERAKHIR, jatuh ke
angka terakhir bila tak ada kotak, normalisasi lowercase+strip, lalu bandingkan
string. Untuk kode: ambil blok ```python TERAKHIR, tempel harness tes di
belakangnya, jalankan di proses terpisah dengan timeout.

Yang ditambahkan (bukan di repo asli, tapi perlu untuk skripsi):
`format_ok` — apakah jawaban muncul dalam FORMAT yang diminta (ada `\\boxed{}`
atau ada blok kode) terlepas dari benar-salahnya. Temuan `docs/HASIL_TAHAP4.md`
§B2 adalah bahwa mode langkah laten memperbaiki KEANDALAN FORMAT tanpa
memperbaiki mutu jawaban; tanpa metrik ini, dua efek itu tercampur di satu
angka akurasi dan disosiasinya tak bisa ditunjukkan.

CATATAN KEAMANAN. `score_code` menjalankan kode yang ditulis LLM. Ia berjalan di
proses anak dengan timeout, tapi TIDAK di-sandbox: tak ada batas filesystem,
jaringan, atau syscall. Jalankan lengan `humanevalplus` hanya di mesin yang
memang disediakan untuk eksperimen ini (pod RunPod), bukan di mesin kerja
dengan kredensial. Ini batasan yang sama dengan harness EvalPlus dan LatentMAS
asli, dicatat di sini supaya bukan kejutan.
"""
from __future__ import annotations

import re
import traceback
from multiprocessing import Manager, Process
from typing import Any, Dict, Optional, Tuple

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_PYBLOCK_RE = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    return None if ans is None else ans.strip().lower()


def extract_boxed_answer(text: str) -> Optional[str]:
    """Isi \\boxed{} terakhir; bila kosong, angka terakhir di seluruh teks."""
    boxes = _BOXED_RE.findall(text)
    if boxes:
        num = _NUMBER_RE.search(boxes[-1])
        return num.group(0) if num else boxes[-1].strip()
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1] if numbers else None


def extract_python_block(text: str) -> Optional[str]:
    """Blok ```python terakhir."""
    blocks = _PYBLOCK_RE.findall(text)
    return blocks[-1].strip() if blocks else None


def _worker(ns, code: str) -> None:
    try:
        exec(code, {})  # noqa: S102 — memang harus mengeksekusi kode kandidat
        ns["ok"], ns["error"] = True, None
    except BaseException:  # noqa: BLE001 — termasuk SystemExit dari kode LLM
        ns["ok"], ns["error"] = False, traceback.format_exc()


def run_with_timeout(code: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Jalankan `code` di proses terpisah; True bila selesai tanpa exception."""
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=_worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return False, f"TimeoutError: eksekusi melewati {timeout} detik"
        return bool(ns.get("ok", False)), ns.get("error")


def score_item(item: Dict[str, Any], answer_text: str, *, timeout: int = 10) -> Dict[str, Any]:
    """Nilai satu jawaban judger terhadap satu soal.

    Return: {correct, prediction, format_ok, error}
      correct    — benar menurut kriteria benchmark-nya
      format_ok  — jawaban keluar dalam format yang diminta (boxed / blok kode)
      prediction — jawaban yang diekstrak (None bila format gagal)
    """
    family = item.get("task_family", "math")
    text = answer_text or ""

    if family == "code":
        pred = extract_python_block(text)
        if pred is None:
            return {"correct": False, "prediction": None, "format_ok": False,
                    "error": "tidak ada blok ```python di keluaran"}
        ok, err = run_with_timeout(pred + "\n" + str(item.get("gold", "")), timeout)
        return {"correct": ok, "prediction": pred, "format_ok": True, "error": err}

    # math & choice: keduanya dinilai lewat \boxed{}
    format_ok = bool(_BOXED_RE.search(text))
    pred = normalize_answer(extract_boxed_answer(text))
    gold = normalize_answer(str(item.get("gold", "")) if item.get("gold") is not None else None)
    correct = bool(pred and gold and pred == gold)
    return {"correct": correct, "prediction": pred, "format_ok": format_ok, "error": None}


def summarize(results: list) -> Dict[str, Any]:
    """Agregat satu sel eksperimen (satu metode × medium × benchmark)."""
    n = len(results)
    if n == 0:
        return {"n": 0, "accuracy": None, "format_rate": None, "n_correct": 0}
    n_correct = sum(1 for r in results if r.get("correct"))
    n_format = sum(1 for r in results if r.get("format_ok"))
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n,
        "format_rate": n_format / n,
    }
