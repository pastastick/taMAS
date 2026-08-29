"""prod/analyze.py — F4 harness validasi (TANPA GPU).

Mengukur, dari artifacts hasil run (prod ATAU legacy v4_eval), apakah hipotesis
BERTAHAN utuh lintas hop proposal->design->construct (inti klaim no-crop), plus
korupsi decode & skor construct. Dipakai untuk A/B:
  - kv (no-crop)  vs  text (full-text, ls=0)        [butuh run prod GPU]
  - DAN sebagai BASELINE: legacy results/v4_eval/* (crop lama) bisa dianalisis
    sekarang tanpa GPU untuk mengkuantifikasi drift yang didiagnosis.

Metrik per rantai (chain proposal->design->construct):
  - fidelity_pd / fidelity_dc : Jaccard kata-isi hipotesis antar hop (1=identik).
  - mech_pd / mech_dc         : overlap kata-mekanisme (momentum/reversal/volume/...).
  - drift_flag                : True bila mekanisme berubah antar hop.
Per node:
  - corruption_hits           : heuristik token sampah (mis. ',T 10', 'TTo').
Per construct terminal:
  - parse_ok, n_legal/n_total (gate), via runner (opsional).

CLI:
  python -m prod.analyze <run_dir> [<run_dir2> ...]
  # contoh baseline (crop lama):
  python -m prod.analyze try/promptbench/results/v4_eval/kv_text/ls10/rep0
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── ekstraksi artifact ───────────────────────────────────────────────────────
_STAGE_RE = re.compile(r"(proposal|design|construct)_(\w+)\.txt$")
_HYP_RE = re.compile(r"HYPOTHESIS:\s*(.+)", re.I)
_RESP_RE = re.compile(r"^=+\s*$\nRESPONSE\n=+\s*$", re.M)


def _response_section(text: str) -> str:
    """Ambil teks setelah header RESPONSE (sebelum SCORE DETAIL bila ada)."""
    idx = text.rfind("\nRESPONSE\n")
    body = text[idx + len("\nRESPONSE\n"):] if idx != -1 else text
    cut = body.find("\nSCORE DETAIL\n")
    if cut != -1:
        body = body[:cut]
    # buang baris bar '====' yang menempel tepat setelah header
    return body.lstrip("=\n ").strip()


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e < s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def extract_hypothesis(response: str) -> Optional[str]:
    """Hipotesis dari RESPONSE, berlapis: (1) field JSON 'hypothesis'
    (design/construct), (2) baris 'HYPOTHESIS:' terakhir (proposal compliant),
    (3) fallback: baris non-kosong terakhir yang cukup panjang (proposal yang lupa
    prefix). Lapis (3) menjaga metrik tetap terhitung untuk output non-compliant."""
    obj = _extract_json(response)
    if obj and isinstance(obj.get("hypothesis"), str) and obj["hypothesis"].strip():
        return obj["hypothesis"].strip()
    hits = _HYP_RE.findall(response)
    if hits:
        return hits[-1].strip()
    for line in reversed([ln.strip() for ln in (response or "").splitlines()]):
        if line and not line.startswith(("{", "}", "=", "#")) and len(line.split()) >= 4:
            return line
    return None


# ── metrik teks ──────────────────────────────────────────────────────────────
_STOP = set("a an the of to in on and or for with is are be it its as that this when "
            "then which whats what how into over more less than next period returns "
            "return cross sectional stock stocks market price prices day days".split())
_MECH = {
    "momentum": r"momentum|trend|persist|continu|drift|follow",
    "reversal": r"revers|mean.?revert|overbought|oversold|rebound|over.?react|snap.?back",
    "volatility": r"volatil|turbulen|dispersion|\brisk\b",
    "volume": r"volume|liquid|turnover|trading activity|flow",
    "correlation": r"correlat|covar|co.?move|comove|lead.?lag|\bbeta\b",
    "stability": r"stable|stabilit|calm|narrow range|low.?vol",
}


def _words(text: str) -> set:
    toks = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return {t for t in toks if t not in _STOP}


def jaccard(a: str, b: str) -> Optional[float]:
    if not a or not b:
        return None
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return None
    return round(len(wa & wb) / len(wa | wb), 3)


def mech_set(text: str) -> set:
    t = (text or "").lower()
    return {m for m, pat in _MECH.items() if re.search(pat, t)}


def mech_overlap(a: str, b: str) -> Optional[float]:
    if not a or not b:
        return None
    ma, mb = mech_set(a), mech_set(b)
    if not (ma or mb):
        return None
    return round(len(ma & mb) / len(ma | mb), 3)


# heuristik korupsi decode: token sampah yg teramati (',T 10', 'TTo', 'Td10', ...)
_GARBAGE = [
    re.compile(r"[,(]\s*[A-Z]{1,3}\s+\d"),          # ", T  10"
    re.compile(r"\b[A-Z][a-z][A-Z]\w*"),            # "TTo", "ToTd"
    re.compile(r"\b[A-Z]{1,3}\d"),                  # "Td10"
    re.compile(r"[^\x00-\x7F]"),                     # non-ASCII
]


def corruption_hits(text: str) -> int:
    return sum(len(p.findall(text or "")) for p in _GARBAGE)


# ── analisis satu run dir ────────────────────────────────────────────────────

def analyze_run(run_dir: Path) -> Dict[str, Any]:
    chains: Dict[str, Dict[str, Any]] = {}
    for f in sorted(run_dir.glob("*.txt")):
        m = _STAGE_RE.search(f.name)
        if not m:
            continue
        stage, chain = m.group(1), m.group(2)
        resp = _response_section(f.read_text(encoding="utf-8", errors="replace"))
        chains.setdefault(chain, {})[stage] = {
            "hyp": extract_hypothesis(resp), "corruption": corruption_hits(resp),
            "resp": resp,
        }

    rows = []
    for chain, st in sorted(chains.items()):
        p, d, c = st.get("proposal"), st.get("design"), st.get("construct")
        hp = p["hyp"] if p else None
        hd = d["hyp"] if d else None
        hc = c["hyp"] if c else None
        f_pd, f_dc = jaccard(hp, hd), jaccard(hd, hc)
        m_pd, m_dc = mech_overlap(hp, hd), mech_overlap(hd, hc)
        drift = bool((m_pd is not None and m_pd < 1.0) or (m_dc is not None and m_dc < 1.0))
        score = _score_construct(c["resp"]) if c else None
        rows.append({
            "chain": chain, "fidelity_pd": f_pd, "fidelity_dc": f_dc,
            "mech_pd": m_pd, "mech_dc": m_dc, "drift": drift,
            "corruption": {k: st[k]["corruption"] for k in ("proposal", "design", "construct") if k in st},
            "mech_chain": [sorted(mech_set(h)) if h else None for h in (hp, hd, hc)],
            "hyps": {"proposal": hp, "design": hd, "construct": hc},
            "construct_score": score,
        })
    return {"run_dir": str(run_dir), "n_chains": len(rows), "chains": rows,
            "summary": _summarize(rows)}


def _score_construct(resp: str) -> Optional[dict]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from prod import runner as R
        hyp, facs = R.parse_construct(resp)
        graded = R.gate_and_repair(facs)  # tanpa agent (deterministik) → baseline gate
        n_legal = sum(int(g["ok"]) for g in graded)
        return {"parse_ok": bool(facs), "n_total": len(facs), "n_legal": n_legal,
                "gate_pass_frac": round(n_legal / len(facs), 3) if facs else 0.0}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _summarize(rows: List[dict]) -> dict:
    def _mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.fmean(vals), 3) if vals else None
    corr = [sum(r["corruption"].values()) for r in rows if r.get("corruption")]
    legal = [r["construct_score"]["gate_pass_frac"] for r in rows
             if r.get("construct_score") and "gate_pass_frac" in r["construct_score"]]
    return {
        "fidelity_pd_mean": _mean("fidelity_pd"),
        "fidelity_dc_mean": _mean("fidelity_dc"),
        "mech_pd_mean": _mean("mech_pd"),
        "mech_dc_mean": _mean("mech_dc"),
        "drift_rate": round(sum(int(r["drift"]) for r in rows) / len(rows), 3) if rows else None,
        "corruption_total": sum(corr),
        "gate_pass_frac_mean": round(statistics.fmean(legal), 3) if legal else None,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m prod.analyze <run_dir> [<run_dir2> ...]")
        sys.exit(1)
    results = []
    for d in sys.argv[1:]:
        rd = Path(d)
        if not rd.is_dir():
            print(f"[skip] bukan direktori: {d}")
            continue
        res = analyze_run(rd)
        results.append(res)
        print(f"\n=== {res['run_dir']}  (chains={res['n_chains']}) ===")
        print(f"{'chain':<6} {'fid_pd':>7} {'fid_dc':>7} {'mech_pd':>8} {'mech_dc':>8} "
              f"{'drift':>6} {'corrupt':>8} {'gate':>6}")
        for r in res["chains"]:
            sc = r.get("construct_score") or {}
            print(f"{r['chain']:<6} {_f(r['fidelity_pd']):>7} {_f(r['fidelity_dc']):>7} "
                  f"{_f(r['mech_pd']):>8} {_f(r['mech_dc']):>8} {str(r['drift']):>6} "
                  f"{sum(r['corruption'].values()):>8} {_f(sc.get('gate_pass_frac')):>6}")
        s = res["summary"]
        print(f"SUMMARY fidelity_pd={s['fidelity_pd_mean']} fidelity_dc={s['fidelity_dc_mean']} "
              f"mech_pd={s['mech_pd_mean']} mech_dc={s['mech_dc_mean']} drift_rate={s['drift_rate']} "
              f"corruption_total={s['corruption_total']} gate_pass={s['gate_pass_frac_mean']}")
        rd_out = rd / "analysis.json"
        rd_out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[analysis] -> {rd_out}")

    if len(results) > 1:
        print("\n=== A/B SUMMARY ===")
        print(f"{'run':<50} {'fid_pd':>7} {'fid_dc':>7} {'drift':>6} {'corrupt':>8} {'gate':>6}")
        for res in results:
            s = res["summary"]
            print(f"{res['run_dir'][-48:]:<50} {_f(s['fidelity_pd_mean']):>7} "
                  f"{_f(s['fidelity_dc_mean']):>7} {_f(s['drift_rate']):>6} "
                  f"{s['corruption_total']:>8} {_f(s['gate_pass_frac_mean']):>6}")


def _f(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
