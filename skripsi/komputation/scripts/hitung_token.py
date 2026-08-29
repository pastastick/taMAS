#!/usr/bin/env python3
"""Agregasi pemakaian token per sel bench — metrik utama paper LatentMAS.

KENAPA ADA. `bench/scoring.py::summarize` hanya merekam akurasi, format_rate,
dan waktu. Padahal Tabel 1/2 paper LatentMAS (arXiv:2511.20639) melaporkan TIGA
metrik, dan yang paling menonjol justru **Token** (total output token, turun
63–80%) serta **Speed**. Tanpa angka token, hasil skripsi ini tak bisa
disandingkan langsung dengan klaim utama paper — padahal di sumbu itulah
replikasinya paling mungkin berhasil.

TIDAK PERLU MENJALANKAN ULANG. Tiap panggilan LLM sudah menyimpan
`input_tokens`/`output_tokens` di header snapshot-nya (`llm/backend.py`
menulisnya). Skrip ini hanya menjumlahkan yang sudah ada di disk.

Definisi mengikuti paper: "total output token usage" per run — yaitu jumlah
token yang DIHASILKAN seluruh agen dalam satu soal, dirata-ratakan per soal.
Token masukan dilaporkan terpisah karena bukan yang diklaim paper (dan pada
medium KV, sebagian konteks masuk lewat cache, bukan lewat token teks — justru
itu mekanisme penghematannya).

    python scripts/hitung_token.py
    python scripts/hitung_token.py --out results/pendukung/token_bench.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import OUT_BENCH, RESULTS  # noqa: E402

_IN = re.compile(r"^- input_tokens:\s*(\d+)", re.M)
_OUT = re.compile(r"^- output_tokens:\s*(\d+)", re.M)

# Aksara CJK di jawaban berbahasa Inggris = token dari bahasa lain menyusup.
# Ini FIDELITAS SIMBOLIK yang bisa dilihat mata: langkah laten `raw` (ridge W_a)
# menghasilkan vektor yang tidak berada di manifold embedding nyata, sehingga
# model sesekali men-decode token yang salah sama sekali ("step by步",
# "1. **确定每"). Terukur 2026-08-10 di GSM8K: nol pada medium TANPA laten
# (baseline, text), dan terbanyak justru pada `raw` — metode dgn akurasi
# terendah. `sample`/`gumbel` yang selalu mendarat di embedding token sungguhan
# mencatat nol. Jadi angka ini bukan sekadar keanehan kosmetik; ia bentuk
# kegagalan yang menjelaskan selisih akurasinya.
_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")


def token_sel(dir_sel: Path) -> dict:
    """Jumlahkan token seluruh panggilan LLM di satu sel."""
    n_call = tok_in = tok_out = 0
    for md in dir_sel.rglob("*.md"):
        teks = md.read_text(errors="ignore")
        mi, mo = _IN.search(teks), _OUT.search(teks)
        if mo is None:
            continue
        n_call += 1
        tok_out += int(mo.group(1))
        if mi:
            tok_in += int(mi.group(1))
    return {"n_panggilan": n_call, "token_keluaran": tok_out,
            "token_masukan": tok_in}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(RESULTS / "pendukung" / "token_bench.json"))
    args = ap.parse_args()

    akar = OUT_BENCH / "llm_outputs"
    if not akar.exists():
        print(f"tak ada {akar}")
        return

    baris = []
    for dir_sel in sorted(p for p in akar.iterdir() if p.is_dir()):
        stem = dir_sel.name
        js = OUT_BENCH / f"bench_{stem}.json"
        t = token_sel(dir_sel)
        rec = {"sel": stem, **t}
        if js.exists():
            d = json.loads(js.read_text())
            m, s = d.get("_meta", {}), d.get("summary", {})
            n = s.get("n") or 0
            # NB: jangan pakai nama `t` di sini — `t` menampung hasil
            # `token_sel()` dan masih dibutuhkan di bawah.
            rusak, contoh = 0, None
            for r in d.get("results", []):
                teks = r.get("answer_text") or ""
                mm = _CJK.search(teks)
                if mm:
                    rusak += 1
                    if contoh is None:
                        contoh = teks[max(0, mm.start() - 25):
                                      mm.start() + 4].replace("\n", " ")
            rec.update({
                "n_jawaban_ber_cjk": rusak,
                "laju_korupsi_token": round(rusak / n, 4) if n else None,
                "contoh_korupsi": contoh,
                "task": m.get("task"),
                "medium": "baseline" if m.get("baseline") else m.get("comm_mode"),
                "metode": m.get("latent_mode"),
                "n_soal": n,
                "akurasi": s.get("accuracy"),
                "waktu_total_s": m.get("total_time_s"),
                # Metrik paper: token keluaran & detik PER SOAL.
                "token_per_soal": round(t["token_keluaran"] / n, 1) if n else None,
                "detik_per_soal": (round(m["total_time_s"] / n, 1)
                                   if n and m.get("total_time_s") else None),
            })
        else:
            rec["catatan"] = "sel belum selesai (JSON hasil belum ada)"
        baris.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(baris, indent=2, ensure_ascii=False))

    print(f"{'sel':34s} {'n':>4s} {'akurasi':>8s} {'token/soal':>11s} "
          f"{'dtk/soal':>9s} {'CJK':>4s}")
    print("-" * 78)
    for r in baris:
        if r.get("n_soal"):
            print(f"{r['sel']:34s} {r['n_soal']:4d} {r['akurasi']:8.3f} "
                  f"{r['token_per_soal']:11.1f} {r['detik_per_soal']:9.1f} "
                  f"{r['n_jawaban_ber_cjk']:4d}")
        else:
            print(f"{r['sel']:34s}    - (belum selesai, {r['n_panggilan']} panggilan)")
    print(f"\nditulis → {args.out}")


if __name__ == "__main__":
    main()
