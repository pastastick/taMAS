#!/usr/bin/env python3
"""Kumpulkan data pendukung skripsi dari artefak run — bukan dari catatan tangan.

Kenapa skrip, bukan tabel yang diketik manual. Angka lampiran harus bisa
DIREGENERASI dari berkas hasil; kalau ia hanya ada sebagai teks yang pernah
disalin dari layar, tak ada cara memeriksanya kembali saat ditanya di sidang,
dan tiap kali data bertambah tabelnya jadi usang diam-diam. Semua yang ditulis
di sini diturunkan dari `results/**/*.json` yang memang dihasilkan runner.

Keluaran → `results/pendukung/`:
    pemakaian_fungsi.json   frekuensi tiap fungsi DSL di korpus ekspresi
    gate_efektivitas.json   kebocoran gate: lolos-gate vs benar-benar evaluable
    alasan_tolak_gate.json  distribusi alasan penolakan gate
    ragam_eval_error.json   kenapa ekspresi gagal dievaluasi
    waktu_sel.json          durasi tiap sel (sumbu biaya teks vs laten)
    ringkasan.md            semuanya dalam bentuk tabel siap-baca

    python scripts/kumpulkan_pendukung.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "pendukung"
_FUNC_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s*\(")


def korpus_faktor() -> list[tuple[str, dict, dict]]:
    """(sumber, run, faktor) untuk seluruh frontend_*.json, termasuk arsip.

    Arsip ikut dibaca DAN ditandai sumbernya: sel yang dibuang dari matriks
    tetap sah sebagai bukti (mis. kebocoran gate saat gate mati), asal tidak
    tercampur dengan data eksperimen. Kolom `sumber` yang menjaganya.
    """
    rows = []
    for p in sorted(RESULTS.glob("**/frontend_*.json")):
        sumber = "matriks" if p.parent.name == "factor" else p.parent.name
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 — berkas separuh tertulis saat run jalan
            continue
        for r in d.get("runs", []):
            for f in (r.get("factors") or []):
                rows.append((f"{sumber}/{p.stem}", r, f))
    return rows


def pemakaian_fungsi(rows) -> dict:
    c = collections.Counter()
    n = 0
    for _, _, f in rows:
        e = f.get("expression") or ""
        if not e:
            continue
        n += 1
        for fn in set(_FUNC_RE.findall(e)):
            c[fn] += 1
    return {"n_ekspresi": n,
            "frekuensi": dict(c.most_common()),
            "persen": {k: round(100 * v / n, 1) for k, v in c.most_common()} if n else {}}


def gate_efektivitas(rows) -> dict:
    """Kebocoran gate = ekspresi yang LOLOS gate tapi gagal dievaluasi.

    Ini ukuran yang membuat metrik `lolos_gate` bisa dipercaya. Tanpa ia,
    "lolos gate 8/11" terdengar seperti 73% berhasil padahal separuhnya
    meledak begitu dijalankan.
    """
    per = collections.defaultdict(lambda: collections.Counter())
    for sumber, _, f in rows:
        k = per[sumber.split("/")[0]]
        k["ekspresi"] += 1
        lolos = bool(f.get("passed_gate"))
        hidup = f.get("ic") is not None
        k["lolos_gate"] += lolos
        k["bisa_dievaluasi"] += hidup
        if lolos and not hidup:
            k["BOCOR_lolos_tapi_mati"] += 1
    return {k: dict(v) for k, v in per.items()}


def alasan_tolak(rows_raw) -> dict:
    c = collections.Counter()
    for p in sorted(RESULTS.glob("**/frontend_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        for r in d.get("runs", []):
            for g in (r.get("gate_log") or []):
                if not g.get("ok"):
                    c[str(g.get("reason") or "(kosong)").split(":")[0][:48]] += 1
    return dict(c.most_common())


def eval_error(rows) -> dict:
    c = collections.Counter()
    for _, _, f in rows:
        err = f.get("eval_error")
        if err:
            c[str(err).split(":")[0][:48]] += 1
    return dict(c.most_common())


def waktu_sel() -> dict:
    """Durasi per sel. Untuk lengan faktor ini SUMBU BIAYA teks vs laten:
    handoff teks mengoper seluruh konteks sebagai token, jadi wajar lebih lama
    — itu bagian dari yang dibandingkan, bukan gangguan."""
    out = {"faktor": {}, "bench": {}}
    for p in sorted(RESULTS.glob("**/frontend_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        a = d.get("args", {})
        durs = [r.get("duration_s") for r in d.get("runs", []) if r.get("duration_s")]
        if durs:
            out["faktor"][p.stem] = {
                "comm_mode": a.get("comm_mode"), "latent_mode": a.get("latent_mode"),
                "n_run": len(durs), "total_s": round(sum(durs), 1),
                "rerata_s": round(sum(durs) / len(durs), 1),
            }
    for p in sorted((RESULTS / "bench").glob("bench_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        m, s = d.get("_meta", {}), d.get("summary", {})
        out["bench"][p.stem] = {
            "task": m.get("task"), "comm_mode": m.get("comm_mode"),
            "latent_mode": m.get("latent_mode"), "baseline": m.get("baseline"),
            "n": s.get("n"), "akurasi": s.get("accuracy"),
            "format_rate": s.get("format_rate"), "total_time_s": m.get("total_time_s"),
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = korpus_faktor()

    data = {
        "pemakaian_fungsi.json": pemakaian_fungsi(rows),
        "gate_efektivitas.json": gate_efektivitas(rows),
        "alasan_tolak_gate.json": alasan_tolak(rows),
        "ragam_eval_error.json": eval_error(rows),
        "waktu_sel.json": waktu_sel(),
    }
    for nama, isi in data.items():
        (OUT / nama).write_text(json.dumps(isi, indent=2, ensure_ascii=False))

    L = ["# Data pendukung (regenerasi: `python scripts/kumpulkan_pendukung.py`)", ""]
    pf = data["pemakaian_fungsi.json"]
    L += [f"## Pemakaian fungsi DSL ({pf['n_ekspresi']} ekspresi)", "",
          "| fungsi | dipakai di | % |", "|---|---:|---:|"]
    for k, v in list(pf["frekuensi"].items())[:20]:
        L.append(f"| {k} | {v} | {pf['persen'][k]} |")

    L += ["", "## Efektivitas gate (kebocoran = lolos gate tapi gagal dievaluasi)", "",
          "| sumber | ekspresi | lolos gate | evaluable | BOCOR |", "|---|---:|---:|---:|---:|"]
    for k, v in data["gate_efektivitas.json"].items():
        L.append(f"| {k} | {v.get('ekspresi',0)} | {v.get('lolos_gate',0)} | "
                 f"{v.get('bisa_dievaluasi',0)} | {v.get('BOCOR_lolos_tapi_mati',0)} |")

    L += ["", "## Alasan penolakan gate", "", "| alasan | n |", "|---|---:|"]
    for k, v in data["alasan_tolak_gate.json"].items():
        L.append(f"| {k} | {v} |")

    L += ["", "## Kenapa ekspresi gagal dievaluasi", "", "| error | n |", "|---|---:|"]
    for k, v in data["ragam_eval_error.json"].items():
        L.append(f"| {k} | {v} |")

    w = data["waktu_sel.json"]
    if w["faktor"]:
        L += ["", "## Waktu sel lengan faktor (biaya teks vs laten)", "",
              "| sel | comm | metode | n_run | rerata (dtk) |", "|---|---|---|---:|---:|"]
        for k, v in w["faktor"].items():
            L.append(f"| {k} | {v['comm_mode']} | {v['latent_mode']} | "
                     f"{v['n_run']} | {v['rerata_s']} |")
    if w["bench"]:
        L += ["", "## Lengan benchmark", "",
              "| sel | tugas | comm | metode | n | akurasi | format | waktu (dtk) |",
              "|---|---|---|---|---:|---:|---:|---:|"]
        for k, v in w["bench"].items():
            L.append(f"| {k} | {v['task']} | {v['comm_mode']} | {v['latent_mode']} | "
                     f"{v['n']} | {v['akurasi']} | {v['format_rate']} | {v['total_time_s']} |")

    (OUT / "ringkasan.md").write_text("\n".join(L) + "\n")
    print(f"ditulis ke {OUT}/  ({len(rows)} faktor dibaca)")
    for nama in data:
        print(f"  {nama}")
    print("  ringkasan.md")


if __name__ == "__main__":
    main()
