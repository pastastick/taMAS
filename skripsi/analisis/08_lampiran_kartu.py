#!/usr/bin/env python3
"""Lampiran D — kartu faktor, meniru format Lampiran C QuantaAlpha.

Tiap kartu memuat satu lintasan nyata: hipotesis yang dibangkitkan agen
pertama, ekspresi yang disusun agen terakhir, status gate, lalu metrik RankIC
dan backtest. Data diambil apa adanya dari korpus hasil, tidak dikarang.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "quantalatent" / "results" / "factor"
DST = ROOT / "skripsi" / "bab" / "lampiran_d_kartu.tex"

PILIH = [("text", "teks", "raw"), ("kv_gumbel", "kv", "gumbel"),
         ("kv_raw", "kv", "raw")]


def esc(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return "".join(c if ord(c) < 128 else f"[U+{ord(c):04X}]" for c in s)


def angka(v, p=4):
    return "---" if v is None else f"{v:+.{p}f}".replace(".", ",")


def kartu(sel: str, medium: str, metode: str) -> list[str]:
    d = json.loads((FRONTEND / f"frontend_{sel}.json").read_text())
    terbaik, run_terbaik = None, None
    for r in d["runs"]:
        for f in r.get("factors") or []:
            if f.get("ic") is None or (f.get("n_unique") or 0) <= 2:
                continue
            if terbaik is None or abs(f.get("tstat") or 0) > abs(terbaik.get("tstat") or 0):
                terbaik, run_terbaik = f, r
    b = [rf"\begin{{kartufaktor}}{{Medium {medium} $\cdot$ langkah laten "
         rf"\texttt{{{metode}}}}}"]
    if terbaik is None:
        # sel yang tidak menghasilkan satu pun ekspresi yang dapat dinilai
        r = next((r for r in d["runs"] if r.get("gate_error")), d["runs"][0])
        b += [r"\barisfield{Hipotesis}{" +
              (esc(r.get("hypothesis") or "") or r"\textit{tidak terbentuk}") + "}",
              r"\barisfield{Status}{\textit{tidak ada ekspresi yang dapat "
              r"dievaluasi pada sel ini}}",
              r"\barisfield{Galat gate}{\code{" + esc(str(r.get("gate_error") or "-")) + "}}",
              r"\barisfield{Keluaran mentah}{}",
              r"\begin{kotakekspresi}",
              esc((r.get("construct_text_head") or "")[:300]),
              r"\end{kotakekspresi}"]
        b.append(r"\end{kartufaktor}")
        return b

    b += [r"\barisfield{Hipotesis}{" + esc(run_terbaik.get("hypothesis") or "-") + "}",
          r"\barisfield{Nama faktor}{" + esc(terbaik.get("name") or "-") + "}",
          r"\barisfield{Ekspresi}{}",
          r"\begin{kotakekspresi}", esc(terbaik["expression"]), r"\end{kotakekspresi}",
          r"\barisfield{Maksud}{" + esc((terbaik.get("explanation") or "-")[:260]) + "}",
          r"\barisfield{Lolos \textit{gate}}{" +
          ("ya" if terbaik.get("passed_gate") or terbaik["expression"]
           in (run_terbaik.get("passing") or []) else "tidak") + "}",
          r"\barisfield{RankIC}{" + angka(terbaik.get("ic")) +
          r" \quad ICIR " + angka(terbaik.get("icir"), 3) +
          r" \quad $t$ " + angka(terbaik.get("tstat"), 2) + "}",
          r"\barisfield{Hari perdagangan}{" + str(terbaik.get("n_days") or "-") + "}",
          r"\barisfield{\textit{Backtest}}{ARR " + angka(terbaik.get("bt_ann_return"), 3) +
          r" \quad Sharpe " + angka(terbaik.get("bt_sharpe"), 2) +
          r" \quad MDD " + angka(terbaik.get("bt_max_drawdown"), 3) +
          r" \quad \textit{turnover} " + angka(terbaik.get("bt_turnover"), 3) + "}",
          r"\end{kartufaktor}"]
    return b


def main() -> None:
    b = [r"\chapter{Kartu Faktor}", r"\label{lamp:kartu}", "",
         r"Tiga lintasan nyata dari lengan faktor simbolik, disajikan dengan "
         r"format yang meniru kartu faktor pada lampiran QuantaAlpha. Kartu "
         r"ketiga sengaja menampilkan sel yang gagal, karena bentuk "
         r"kegagalannya merupakan bagian dari temuan dan bukan data yang "
         r"hilang. Angka \textit{backtest} berasal dari simulasi kasar dan "
         r"tidak boleh dibaca sebagai ramalan keuntungan.", ""]
    for sel, medium, metode in PILIH:
        b += kartu(sel, medium, metode) + [""]
    DST.write_text("\n".join(b) + "\n")
    print(f"[tulis] {DST}")


if __name__ == "__main__":
    main()
