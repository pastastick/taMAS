#!/usr/bin/env python3
"""Lengan faktor simbolik sebagai kondisi PELENGKAP.

Sesuai cakupan yang ditetapkan pada Bab~III, lengan faktor tidak diperlakukan
sebagai sel eksperimen penuh. Skrip ini menghasilkan tiga hal saja:

  1. laju kegagalan parsing keluaran agen construct per sel (bukti fidelitas
     simbolik yang tidak memerlukan data pasar sama sekali);
  2. agregat korpus TERCATAT yang sudah bernilai IC dan backtest;
  3. beberapa kartu faktor konkret sebagai bukti kualitatif.

Ekspresi hasil pemulihan artefak `kv_and_text` hanya diskor untuk sebagian
kecil sampel — cukup untuk menunjukkan bahwa keluarannya memang ekspresi yang
sah, bukan untuk membandingkan medium.

Keluaran → analisis/faktor_pelengkap.json, skripsi/assets/tables/faktor_*.tex
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
AN = ROOT / "analisis"
TAB = ROOT / "skripsi" / "assets" / "tables"
PREFIKS = '{\n  "hypothesis": '
N_SKOR_PER_SEL = 2          # ekspresi hasil pemulihan yang diskor per sel

URUT = ["text", "kv_raw", "kv_soft", "kv_gumbel", "kv_sample", "kv_moi",
        "kv_and_text_raw", "kv_and_text_soft", "kv_and_text_gumbel",
        "kv_and_text_sample", "kv_and_text_moi"]
NAMA = {"text": ("teks", "raw"), "kv_raw": ("kv", "raw"),
        "kv_soft": ("kv", "soft"), "kv_gumbel": ("kv", "gumbel"),
        "kv_sample": ("kv", "sample"), "kv_moi": ("kv", "moi")}


def baca_response(p: Path) -> str | None:
    t = p.read_text(encoding="utf-8", errors="replace")
    i = t.find("## Response")
    if i < 0:
        return None
    m = re.search(r"```text\n(.*?)\n```", t[i:], re.S)
    return m.group(1) if m else None


def parse_atau_pulihkan(txt: str):
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


def main() -> None:
    # ── 1. status parsing per sel ───────────────────────────────────────
    status: dict[str, Counter] = defaultdict(Counter)
    pulihan: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(SNAP.glob("*/*/*construct*.md")):
        sel = p.relative_to(SNAP).parts[0]
        r = baca_response(p)
        if r is None:
            continue
        st, o = parse_atau_pulihkan(r)
        status[sel][st] += 1
        if st == "pulih":
            for f in (o or {}).get("factors") or []:
                e = (f.get("expression") or "").strip()
                if e and e not in [x["ekspresi"] for x in pulihan[sel]]:
                    pulihan[sel].append({"ekspresi": e, "nama": f.get("name", "")})

    # ── 2. korpus tercatat (sudah bernilai IC + backtest) ───────────────
    tercatat = {}
    for p in sorted(FRONTEND.glob("frontend_*.json")):
        sel = p.stem[len("frontend_"):]
        d = json.loads(p.read_text())
        fs, hip = [], []
        for r in d["runs"]:
            if r.get("hypothesis"):
                hip.append(r["hypothesis"])
            for f in r.get("factors") or []:
                if f.get("expression"):
                    fs.append(f | {"passed_gate": f.get("expression")
                                   in (r.get("passing") or [])})
        hidup = [f for f in fs if f.get("ic") is not None and (f.get("n_unique") or 0) > 2]
        tercatat[sel] = {
            "ekspresi": len(fs),
            "lolos_gate": sum(1 for f in fs if f["passed_gate"]),
            "hidup": len(hidup),
            "mean_abs_ic": (sum(abs(f["ic"]) for f in hidup) / len(hidup)) if hidup else None,
            "ic_signifikan": sum(1 for f in hidup if abs(f.get("tstat") or 0) >= 1.96),
            "faktor": fs, "hipotesis": hip,
        }

    # ── 3. gate atas ekspresi hasil pemulihan ───────────────────────────
    import dsl.factor_ast  # noqa: F401
    from mas.pipeline import FrontEndPipeline
    gate, _ = FrontEndPipeline._build_regulator_gate(None)
    for sel, lst in pulihan.items():
        for x in lst:
            try:
                x["lolos_gate"] = bool(gate(x["ekspresi"])[0])
            except Exception:
                x["lolos_gate"] = False

    # ── 4. skor sampel kecil ekspresi pulihan ───────────────────────────
    contoh = []
    if "--cepat" in sys.argv:
        lama = AN / "faktor_pelengkap.json"
        if lama.exists():
            contoh = json.loads(lama.read_text()).get("contoh_pulihan", [])
    else:
        from eval.ic import Lab
        from factor.run_factor import score_expressions
        lab, cache = Lab(mode="fast"), {}
    for sel in (sorted(pulihan) if "--cepat" not in sys.argv else []):
        lolos = [x for x in pulihan[sel] if x["lolos_gate"]][:N_SKOR_PER_SEL]
        if not lolos:
            continue
        runs = [{"factors": [{"expression": x["ekspresi"]} for x in lolos],
                 "passing": []}]
        print(f"[skor] {sel}: {len(lolos)} ekspresi", flush=True)
        score_expressions(runs, budget_s=45, cache=cache, lab=lab)
        for x, f in zip(lolos, runs[0]["factors"]):
            contoh.append({"sel": sel, "nama": x["nama"], **f})

    hasil = {
        "status_parsing": {k: dict(v) for k, v in status.items()},
        "pulihan_per_sel": {k: {"ekspresi": len(v),
                                "lolos_gate": sum(1 for x in v if x["lolos_gate"])}
                            for k, v in pulihan.items()},
        "tercatat": {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("faktor", "hipotesis")}
                     for k, v in tercatat.items()},
        "contoh_pulihan": contoh,
    }
    (AN / "faktor_pelengkap.json").write_text(json.dumps(hasil, indent=2, default=str))

    # ── 5. tabel LaTeX ──────────────────────────────────────────────────
    tulis_tabel_parsing(status, tercatat)
    tulis_kartu(tercatat)
    print("[tulis] analisis/faktor_pelengkap.json + tabel LaTeX")


def tulis_tabel_parsing(status, tercatat) -> None:
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Keluaran agen \textit{construct} pada lengan faktor "
         r"simbolik. Kolom \textsc{Gagal} adalah panggilan yang keluarannya "
         r"tidak dapat diurai menjadi objek JSON yang sah; kolom "
         r"\textsc{Pulih} adalah panggilan yang hanya kehilangan prefiks "
         r"pembuka objek dan dapat dipulihkan secara deterministik.}",
         r"\label{tab:faktor-parsing}", r"\small",
         r"\begin{tabular}{llrrrrr}", r"\hline",
         r"Medium & Metode & Panggilan & Utuh & Pulih & Gagal & Gagal (\%) \\",
         r"\hline"]
    for sel in URUT:
        c = status.get(sel)
        if not c:
            continue
        tot = c["ok"] + c["pulih"] + c["gagal"]
        med, met = (NAMA.get(sel) or
                    ("kv+teks", sel.replace("kv_and_text_", "")))
        b.append(f"{med} & \\texttt{{{met}}} & {tot} & {c['ok']} & {c['pulih']}"
                 f" & {c['gagal']} & "
                 + (f"{c['gagal'] / tot * 100:.0f}".replace(".", ",") if tot else "---")
                 + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TAB / "faktor_parsing.tex").write_text("\n".join(b) + "\n")

    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Korpus ekspresi yang benar-benar diserahkan sistem "
         r"ujung-ke-ujung pada medium \texttt{text} dan \texttt{kv} (enam "
         r"jalannya per sel). Ekspresi disebut hidup bila RankIC-nya "
         r"terdefinisi dan skornya tidak konstan.}",
         r"\label{tab:faktor-korpus}", r"\small",
         r"\begin{tabular}{llrrrr}", r"\hline",
         r"Medium & Metode & Ekspresi & Dapat dievaluasi & "
         r"$\overline{|\text{RankIC}|}$ & $|t|\ge1{,}96$ \\", r"\hline"]
    for sel in URUT:
        if sel not in NAMA or sel not in tercatat:
            continue
        v = tercatat[sel]
        med, met = NAMA[sel]
        ic = ("---" if v["mean_abs_ic"] is None
              else f"{v['mean_abs_ic']:.4f}".replace(".", ","))
        b.append(f"{med} & \\texttt{{{met}}} & {v['ekspresi']} & {v['hidup']}"
                 f" & {ic} & {v['ic_signifikan']} \\\\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TAB / "faktor_korpus.tex").write_text("\n".join(b) + "\n")


def tulis_kartu(tercatat) -> None:
    """Beberapa kartu faktor konkret, gaya Lampiran C QuantaAlpha."""
    kandidat = []
    for sel in ("text", "kv_gumbel", "kv_moi", "kv_sample", "kv_soft"):
        v = tercatat.get(sel)
        if not v:
            continue
        hidup = [f for f in v["faktor"]
                 if f.get("ic") is not None and (f.get("n_unique") or 0) > 2
                 and f.get("bt_sharpe") is not None]
        for f in sorted(hidup, key=lambda f: -abs(f.get("tstat") or 0)):
            if f["expression"] not in {k[1]["expression"] for k in kandidat}:
                kandidat.append((sel, f,
                                 v["hipotesis"][0] if v["hipotesis"] else ""))
                break
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Faktor dengan statistik-$t$ RankIC terbesar pada tiap sel "
         r"medium yang menghasilkan ekspresi hidup, beserta metrik "
         r"\textit{backtest} portofolio desil \textit{long--short}. Angka "
         r"\textit{backtest} tidak boleh dibaca sebagai ramalan keuntungan; "
         r"lihat batas berlaku.}",
         r"\label{tab:kartu-faktor}", r"\small",
         r"\begin{tabular}{llrrrrr}", r"\hline",
         r"Sel & Ekspresi & RankIC & $t$ & ARR & Sharpe & MDD \\", r"\hline"]
    for sel, f, _ in kandidat:
        med, met = NAMA[sel]
        e = f["expression"]
        if len(e) > 44:
            # potong di TENGAH: dua ekspresi yang berbagi awalan panjang akan
            # tampak identik bila dipotong di ekor saja
            e = e[:26] + " ... " + e[-13:]
        e = e.replace("_", r"\_").replace("$", r"\$")

        def g(k, p=3):
            v = f.get(k)
            return "---" if v is None else f"{v:+.{p}f}".replace(".", ",")
        b.append(f"{med}/\\texttt{{{met}}} & \\texttt{{\\scriptsize {e}}} & "
                 f"{g('ic', 4)} & {g('tstat', 2)} & {g('bt_ann_return')} & "
                 f"{g('bt_sharpe', 2)} & {g('bt_max_drawdown')} \\\\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (TAB / "faktor_kartu.tex").write_text("\n".join(b) + "\n")


if __name__ == "__main__":
    main()
