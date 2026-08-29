#!/usr/bin/env python3
"""Bangun korpus ekspresi lengan faktor, pulihkan artefak kv_and_text, lalu skor.

DUA KORPUS, sengaja dilaporkan berdampingan:

  korpus TERCATAT   — dari results/factor/frontend_<sel>.json. Ini yang benar-benar
                      dihasilkan sistem ujung-ke-ujung (sesudah agen repair), dan
                      sudah bernilai IC + backtest.
  korpus DIPULIHKAN — dari snapshot llm_outputs/<sel>/*/**construct*.md, dengan satu
                      aturan pemulihan deterministik (lihat PULIHKAN di bawah).

KENAPA ADA PEMULIHAN. Pada medium `kv_and_text`, keluaran agen construct
kehilangan tepat prefiks pembuka objek JSON (`{\\n  "hypothesis": `); sisa
keluarannya utuh dan valid. Akibatnya parser sistem menolak SELURUH keluaran dan
sel itu tercatat menghasilkan nol ekspresi — padahal modelnya berhasil. Aturan
pemulihan hanya menambahkan kembali prefiks yang hilang, dan secara empiris TAK
PERNAH aktif pada sel yang keluarannya sudah normal (0 kasus dari 62 panggilan
construct di sel `kv`/`text`), sehingga ia tidak dapat mengubah sel yang sehat.

Keluaran:
  analisis/faktor_korpus.csv    satu baris = satu ekspresi unik per sel
  analisis/faktor_ringkas.json  agregat per sel untuk kedua korpus
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QL = ROOT / "quantalatent"
sys.path.insert(0, str(QL / "backend"))

SNAP = Path("/tmp/results/factor/llm_outputs")
FRONTEND = QL / "results" / "factor"
OUT = ROOT / "analisis"
OUT.mkdir(exist_ok=True)

PREFIKS = '{\n  "hypothesis": '


# ── pembacaan snapshot ──────────────────────────────────────────────────────
def baca_response(p: Path) -> str | None:
    t = p.read_text(encoding="utf-8", errors="replace")
    i = t.find("## Response")
    if i < 0:
        return None
    m = re.search(r"```text\n(.*?)\n```", t[i:], re.S)
    return m.group(1) if m else None


def parse_atau_pulihkan(txt: str) -> tuple[str, dict | None]:
    """(status, objek). status ∈ {ok, pulih, gagal}."""
    c = txt.strip()
    i, j = c.find("{"), c.rfind("}")
    if i >= 0 and j > i:
        try:
            o = json.loads(c[i:j + 1])
            if isinstance(o, dict) and "factors" in o:
                return "ok", o
        except Exception:
            pass
    j = c.rfind("}")
    if j > 0:
        try:
            o = json.loads(PREFIKS + c[:j + 1])
            if isinstance(o, dict) and "factors" in o:
                return "pulih", o
        except Exception:
            pass
    return "gagal", None


def korpus_snapshot() -> tuple[dict, dict]:
    """{sel: {ekspresi: nama}}, {sel: Counter(status panggilan)}"""
    korpus: dict[str, dict[str, str]] = defaultdict(dict)
    status: dict[str, Counter] = defaultdict(Counter)
    for p in sorted(SNAP.glob("*/*/*construct*.md")):
        sel = p.relative_to(SNAP).parts[0]
        r = baca_response(p)
        if r is None:
            status[sel]["tanpa_response"] += 1
            continue
        st, o = parse_atau_pulihkan(r)
        status[sel][st] += 1
        for f in (o or {}).get("factors") or []:
            e = (f.get("expression") or "").strip()
            if e:
                korpus[sel].setdefault(e, f.get("name") or "")
    return korpus, status


# ── gate ────────────────────────────────────────────────────────────────────
def bangun_gate():
    """Gate yang sama dengan yang dipakai pipeline (FrontEndPipeline)."""
    import dsl.factor_ast  # noqa: F401  (memutus circular import)
    from mas.pipeline import FrontEndPipeline
    gate, _reg = FrontEndPipeline._build_regulator_gate(None)
    return gate


# ── utama ───────────────────────────────────────────────────────────────────
def main() -> None:
    import pandas as pd

    korpus, status = korpus_snapshot()
    print(f"[snapshot] {len(korpus)} sel, "
          f"{sum(len(v) for v in korpus.values())} ekspresi (belum unik-lintas-sel)",
          flush=True)
    for sel in sorted(status):
        c = status[sel]
        print(f"  {sel:22s} ok={c['ok']:3d} pulih={c['pulih']:3d} "
              f"gagal={c['gagal']:3d} ekspresi={len(korpus.get(sel, {})):3d}", flush=True)

    # korpus tercatat (sudah bernilai IC + backtest)
    tercatat: dict[str, dict[str, dict]] = {}
    for p in sorted(FRONTEND.glob("frontend_*.json")):
        sel = p.stem[len("frontend_"):]
        d = json.loads(p.read_text())
        m: dict[str, dict] = {}
        for r in d["runs"]:
            for f in r.get("factors") or []:
                e = (f.get("expression") or "").strip()
                if e:
                    m[e] = f
        tercatat[sel] = m
    print(f"[tercatat] {sum(len(v) for v in tercatat.values())} ekspresi", flush=True)

    gate = bangun_gate()

    # skoring: satu Lab, satu cache lintas sel
    from eval.ic import Lab
    from factor.run_factor import score_expressions
    lab = Lab(mode="fast")
    cache: dict = {}
    series: dict = {}

    # seed cache dari nilai yang SUDAH dihitung (hemat waktu, angka identik)
    KUNCI = ("sem_ok", "sem_errors", "flags", "ic", "icir", "tstat", "n_days",
             "coverage", "n_unique", "eval_error")
    for m in tercatat.values():
        for e, f in m.items():
            if "ic" in f and e not in cache:
                cache[e] = {k: f[k] for k in KUNCI if k in f}
                cache[e].update({k: v for k, v in f.items() if k.startswith("bt_")})
    print(f"[cache] {len(cache)} ekspresi dipulihkan dari skoring sebelumnya", flush=True)

    baris = []
    for sel in sorted(set(korpus) | set(tercatat)):
        ekspresi = sorted(set(korpus.get(sel, {})) | set(tercatat.get(sel, {})))
        runs = [{"factors": [{"expression": e} for e in ekspresi], "passing": []}]
        n_baru = sum(1 for e in ekspresi if e not in cache)
        if n_baru:
            print(f"[skor] {sel}: {n_baru} ekspresi baru dari {len(ekspresi)}", flush=True)
        score_expressions(runs, budget_s=60, cache=cache, lab=lab, series_cache=series)
        for f in runs[0]["factors"]:
            e = f["expression"]
            try:
                lolos = bool(gate(e)[0])
            except Exception:
                lolos = False
            b = {
                "sel": sel,
                "medium": ("text" if sel == "text"
                           else "kv_and_text" if sel.startswith("kv_and_text") else "kv"),
                "metode": ("raw" if sel == "text" else sel.split("_")[-1]),
                "ekspresi": e,
                "di_tercatat": e in tercatat.get(sel, {}),
                "di_snapshot": e in korpus.get(sel, {}),
                "lolos_gate": lolos,
            }
            for k in ("ic", "icir", "tstat", "n_days", "n_unique", "eval_error",
                      "sem_ok"):
                b[k] = f.get(k)
            for k in ("bt_ann_return", "bt_ann_vol", "bt_sharpe", "bt_max_drawdown",
                      "bt_turnover", "bt_hit_rate"):
                b[k] = f.get(k)
            b["hidup"] = (b["ic"] is not None and (b["n_unique"] or 0) > 2)
            baris.append(b)

    df = pd.DataFrame(baris)
    df.to_csv(OUT / "faktor_korpus.csv", index=False)
    print(f"\n[tulis] {OUT/'faktor_korpus.csv'}  ({len(df)} baris)", flush=True)

    ring = {"status_panggilan": {k: dict(v) for k, v in status.items()}, "per_sel": []}
    for sel, g in df.groupby("sel"):
        t, r = g[g.di_tercatat], g
        def blok(x):
            h = x[x.hidup]
            return {
                "ekspresi": int(len(x)),
                "lolos_gate": int(x.lolos_gate.sum()),
                "hidup": int(len(h)),
                "mean_abs_ic": float(h.ic.abs().mean()) if len(h) else None,
                "n_ic_signifikan": int((h.tstat.abs() >= 1.96).sum()) if len(h) else 0,
                "mean_sharpe": float(h.bt_sharpe.mean()) if len(h) else None,
                "mean_ann_return": float(h.bt_ann_return.mean()) if len(h) else None,
                "mean_mdd": float(h.bt_max_drawdown.mean()) if len(h) else None,
                "mean_turnover": float(h.bt_turnover.mean()) if len(h) else None,
            }
        ring["per_sel"].append({"sel": sel,
                                "medium": g.medium.iloc[0], "metode": g.metode.iloc[0],
                                "tercatat": blok(t), "dipulihkan": blok(r)})
    (OUT / "faktor_ringkas.json").write_text(json.dumps(ring, indent=2))
    print(f"[tulis] {OUT/'faktor_ringkas.json'}", flush=True)


if __name__ == "__main__":
    main()
