#!/usr/bin/env python3
"""Agregasi `agent_trace` per-hop — biaya & keandalan tiap agen, bukan tiap sel.

KENAPA ADA. `frontend_*.json` adalah SATU-SATUNYA tempat di seluruh proyek
yang merekam data per-agen (`kv_len`, token masuk/keluar, waktu laten vs
generasi, `parsed_ok`) — README §"Repository layout" bahkan menyebutnya
sebagai pembeda lengan faktor dari lengan bench. Tapi tak ada satu skrip pun
DI REPO INI yang mengagregasinya; pemecahan per-hop yang ada
(`09_faktor_perhop.py`) hidup di `../analisis/`, di luar repo, dan keluarannya
`.tex` untuk skripsi — bukan artefak yang bisa dibaca ulang di sini.

Skrip ini mengisi celah itu: agregasi per (comm_mode, latent_mode, agent),
median + IQR (bukan mean saja — durasi long-tail karena REGRESI/REGBETA
rolling-nested pada `construct`). Dijalankan CPU-only, dan resumable seperti
skrip lain: memakai berkas yang sudah ada di `results/factor/`, arsip
diikutkan dan ditandai `sumber` (sama seperti `kumpulkan_pendukung.py`).

    python scripts/agregasi_agent_trace.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "pendukung"
NUMERIC = ["s", "latent_s", "gen_s", "kv_len", "n_in_tok", "n_out_tok",
          "text_len", "rep_ratio"]


def _iqr_summary(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    n = len(vals)
    return {
        "n": n, "median": st.median(vals),
        "p25": vals[int(0.25 * (n - 1))], "p75": vals[int(0.75 * (n - 1))],
        "mean": st.mean(vals), "min": vals[0], "max": vals[-1],
    }


def korpus() -> list[tuple[str, dict]]:
    """(sumber, run) untuk seluruh frontend_*.json termasuk arsip — sama
    aturan sumber seperti kumpulkan_pendukung.py::korpus_faktor()."""
    out = []
    for p in sorted(RESULTS.glob("**/frontend_*.json")):
        sumber = "matriks" if p.parent.name == "factor" else p.parent.name
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 — berkas separuh tertulis
            continue
        tag = p.stem.replace("frontend_", "")
        for r in d.get("runs", []):
            r = dict(r)
            r["_sumber"] = sumber
            r["_tag"] = tag
            out.append((sumber, r))
    return out


def main() -> None:
    rows = korpus()
    print(f"{len(rows)} trajectory ditemukan lintas frontend_*.json (termasuk arsip).")

    # kelompok: (sumber, tag, agent) -> {metrik: [nilai...]}, + parsed_ok terpisah
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    parsed: dict[tuple, list[bool]] = defaultdict(list)
    stop_reason: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sumber, r in rows:
        for t in (r.get("agent_trace") or []):
            key = (sumber, r["_tag"], t.get("agent"), t.get("mode"))
            for k in NUMERIC:
                v = t.get(k)
                if v is not None:
                    groups[key][k].append(v)
            if "parsed_ok" in t:
                parsed[key].append(bool(t["parsed_ok"]))
            if t.get("latent_stop"):
                stop_reason[key][t["latent_stop"]] += 1

    out = {}
    lines = ["# Agregasi agent_trace — biaya & keandalan per-hop (median + IQR)",
             "", "Regenerasi: `python scripts/agregasi_agent_trace.py`",
             "", "Unit: satu baris = satu (sumber, tag, agen, mode KV). "
             "`s` = waktu total hop (detik); `latent_s`/`gen_s` pecahannya.",
             "", "| sumber | tag | agen | mode | n | kv_len (med) | n_in_tok (med) | "
             "n_out_tok (med) | latent_s (med) | gen_s (med) | parsed_ok% |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]

    for key in sorted(groups):
        sumber, tag, agent, mode = key
        m = {k: _iqr_summary(v) for k, v in groups[key].items()}
        p_ok = parsed.get(key, [])
        p_rate = (sum(p_ok) / len(p_ok)) if p_ok else None
        entry = {
            "sumber": sumber, "tag": tag, "agent": agent, "mode": mode,
            "metrik": m, "parsed_ok_rate": p_rate,
            "latent_stop_reasons": dict(stop_reason.get(key, {})),
        }
        out.setdefault(sumber, {}).setdefault(tag, []).append(entry)

        def _med(k):
            return m.get(k, {}).get("median")
        n = m.get("s", {}).get("n", 0)
        row = (f"| {sumber} | {tag} | {agent} | {mode} | {n} | "
               f"{(_med('kv_len') or 0):.0f} | {(_med('n_in_tok') or 0):.0f} | "
               f"{(_med('n_out_tok') or 0):.0f} | {(_med('latent_s') or 0):.2f} | "
               f"{(_med('gen_s') or 0):.2f} | "
               f"{f'{100*p_rate:.0f}' if p_rate is not None else '-'} |")
        lines.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agent_trace_perhop.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    (OUT / "agent_trace_perhop.md").write_text("\n".join(lines) + "\n")
    print(f"[tersimpan] {OUT / 'agent_trace_perhop.json'} "
          f"({sum(len(v) for tags in out.values() for v in tags.values())} baris)")


if __name__ == "__main__":
    main()
