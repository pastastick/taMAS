#!/usr/bin/env python3
"""Rakit transkrip per sel jadi SATU berkas Markdown yang bisa dibaca manusia.

Masalah yang diselesaikan: hasil satu sel tersebar di ratusan berkas —
`results/bench/bench_<tugas>_<mode>_<medium>_s0.json` berisi hasil per soal,
sementara prompt dan jawaban mentah tiap panggilan LLM ada di
`llm_outputs/<sel>/session_*/NNNN_<agen>_<medium>.md`. Membaca satu sel berarti
membuka 100+ berkas, sebagian JSON.

Skrip ini menggabungkannya per sel. Ia MENYALIN, bukan meringkas: seluruh
prompt, seluruh jawaban, seluruh hasil per soal ikut apa adanya. Merangkum akan
membuang justru bagian yang biasanya dicari — jawaban yang rusak, token yang
menyimpang, penggalan yang jadi bukti di skripsi.

Sumber transkrip. `llm_outputs/` tidak di-track git (lihat
`results/.gitignore`) dan hanya tersimpan di arsip `hasil_*.tar.gz`. Skrip ini
membaca langsung dari arsip bila direktori hasil ekstraknya tidak ada, sehingga
tak perlu mengekstrak 3000+ berkas ke pohon kerja lebih dulu.

Pemakaian:
    python scripts/rakit_transkrip.py                    # semua lengan
    python scripts/rakit_transkrip.py --arm factor
    python scripts/rakit_transkrip.py --sel gsm8k_raw_kv_s0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

QL = Path(__file__).resolve().parents[1]
RESULTS = QL / "results"
OUT = RESULTS / "lampiran_dibaca"

_URUT = re.compile(r"^(\d+)_")


# ── sumber transkrip: direktori hasil ekstrak, atau langsung dari arsip ──────

class SumberTranskrip:
    """Menyediakan isi berkas transkrip per sel, dari mana pun ia tersimpan."""

    def __init__(self) -> None:
        self.dari_dir: dict[str, Path] = {}
        self.dari_tar: dict[str, list[tuple[str, tarfile.TarFile, tarfile.TarInfo]]] = {}
        self._tars: list[tarfile.TarFile] = []

    def pindai(self) -> None:
        for d in RESULTS.rglob("llm_outputs"):
            for sel in d.iterdir():
                if sel.is_dir():
                    self.dari_dir.setdefault(sel.name, sel)
        for arsip in sorted(RESULTS.glob("*.tar.gz")):
            try:
                tf = tarfile.open(arsip, "r:gz")
            except tarfile.TarError as e:
                print(f"[lewati] {arsip.name}: {e}", file=sys.stderr)
                continue
            self._tars.append(tf)
            for info in tf.getmembers():
                if not info.isfile() or "/llm_outputs/" not in info.name:
                    continue
                bagian = info.name.split("/llm_outputs/", 1)[1].split("/")
                if len(bagian) < 2:
                    continue
                sel = bagian[0]
                if sel in self.dari_dir:      # yang di disk menang
                    continue
                self.dari_tar.setdefault(sel, []).append((bagian[-1], tf, info))

    def sel_tersedia(self) -> set[str]:
        return set(self.dari_dir) | set(self.dari_tar)

    def berkas(self, sel: str) -> list[tuple[str, str]]:
        """[(nama_berkas, isi)] terurut nomor panggilan."""
        keluar: list[tuple[str, str]] = []
        if sel in self.dari_dir:
            for p in self.dari_dir[sel].rglob("*.md"):
                keluar.append((p.name, p.read_text(encoding="utf8", errors="replace")))
        else:
            for nama, tf, info in self.dari_tar.get(sel, []):
                f = tf.extractfile(info)
                if f is None:
                    continue
                keluar.append((nama, f.read().decode("utf8", errors="replace")))

        def kunci(x):
            m = _URUT.match(x[0])
            return (int(m.group(1)) if m else 10**9, x[0])

        return sorted(keluar, key=kunci)

    def tutup(self) -> None:
        for tf in self._tars:
            tf.close()


# ── penyusun bagian ─────────────────────────────────────────────────────────

def _blok_meta(meta: dict) -> list[str]:
    b = ["## Konfigurasi run", "", "| kunci | nilai |", "|---|---|"]
    for k, v in meta.items():
        if k == "prompts":            # prompt lengkap muncul di transkrip
            v = "(lihat transkrip)"
        b.append(f"| `{k}` | {v} |")
    return b + [""]


def bagian_bench(doc: dict) -> list[str]:
    b: list[str] = []
    if "_meta" in doc:
        b += _blok_meta(doc["_meta"])
    s = doc.get("summary") or {}
    if s:
        b += ["## Ringkasan", "",
              "| n | benar | akurasi | laju format |", "|---:|---:|---:|---:|",
              f"| {s.get('n')} | {s.get('n_correct')} | {s.get('accuracy')} "
              f"| {s.get('format_rate')} |", ""]

    hasil = doc.get("results") or []
    if hasil:
        b += ["## Hasil per soal", ""]
        for r in hasil:
            tanda = "BENAR" if r.get("correct") else "SALAH"
            b += [f"### Soal {r.get('index')} — {tanda}", "",
                  f"- prediksi: `{r.get('prediction')}`  |  kunci: "
                  f"`{r.get('gold')}`  |  format sah: {r.get('format_ok')}"
                  f"  |  {r.get('duration_s')} dtk"]
            if r.get("error") or r.get("pipeline_error"):
                b.append(f"- galat: `{r.get('error') or r.get('pipeline_error')}`")
            b += ["", "**Pertanyaan**", "", "```text",
                  str(r.get("question", "")).rstrip(), "```", "",
                  "**Jawaban**", "", "```text",
                  str(r.get("answer_text", "")).rstrip(), "```", ""]
    return b


def bagian_faktor(doc: dict) -> list[str]:
    b: list[str] = []
    if "args" in doc:
        b += _blok_meta(doc["args"])
    runs = doc.get("runs") or []
    b += ["## Ringkasan jalan", "",
          "| # | arah | seed | detik | ekspresi | lolos gate | repair | galat gate |",
          "|---:|---|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(runs):
        b.append(f"| {i} | {r.get('direction')} | {r.get('seed')} | "
                 f"{r.get('duration_s')} | {len(r.get('factors') or [])} | "
                 f"{len(r.get('passing') or [])} | {r.get('repair_attempts', 0)} | "
                 f"{r.get('gate_error') or '—'} |")
    b.append("")

    for i, r in enumerate(runs):
        b += [f"## Jalan {i} — arah `{r.get('direction')}`, seed {r.get('seed')}", "",
              f"- hipotesis: {r.get('hypothesis') or '(kosong)'}", ""]
        for t in (r.get("agent_trace") or []):
            b += [f"### Agen `{t.get('agent')}` ({t.get('mode')})", "",
                  f"- {t.get('s')} dtk (laten {t.get('latent_s')}, "
                  f"generasi {t.get('gen_s')})  |  KV {t.get('kv_len')} token"
                  f"  |  masuk {t.get('n_in_tok')} / keluar {t.get('n_out_tok')} token"
                  f"  |  rasio ulang {t.get('rep_ratio')}"
                  f"  |  terurai: {t.get('parsed_ok')}"]
            teks = t.get("text") or ""
            if teks:
                b += ["", "```text", teks.rstrip(), "```"]
            b.append("")
        for f in (r.get("factors") or []):
            b += [f"### Faktor `{f.get('name')}`", "", "```text",
                  str(f.get("expression", "")).strip(), "```", "",
                  f"- penjelasan: {f.get('explanation', '')}",
                  f"- lolos gate: {f.get('passed_gate')}  |  IC: {f.get('ic')}"
                  f"  |  t: {f.get('tstat')}  |  n unik: {f.get('n_unique')}",
                  f"- galat evaluasi: {f.get('eval_error') or '—'}", ""]
    return b


def bagian_transkrip(berkas: list[tuple[str, str]]) -> list[str]:
    if not berkas:
        return ["## Transkrip panggilan LLM", "",
                "_Tidak ada transkrip untuk sel ini — `llm_outputs/` tak "
                "tersimpan di disk maupun di arsip `hasil_*.tar.gz`._", ""]
    b = [f"## Transkrip panggilan LLM ({len(berkas)} panggilan)", ""]
    for nama, isi in berkas:
        # Judul level-1 tiap transkrip diturunkan ke level-3 supaya berkas
        # gabungan punya satu hierarki yang utuh dan bisa dilipat per bagian.
        isi = re.sub(r"^# ", "### ", isi, count=1, flags=re.M)
        isi = re.sub(r"^## ", "#### ", isi, flags=re.M)
        b += [isi.rstrip(), "", "---", ""]
    return b


# ── perakit ─────────────────────────────────────────────────────────────────

def nama_sel_bench(path: Path) -> str:
    """`bench_gsm8k_raw_kv_s0.json` -> `gsm8k_raw_kv_s0` (nama dir llm_outputs)."""
    return path.stem[len("bench_"):]


def rakit(lengan: str, path: Path, sumber: SumberTranskrip) -> Path | None:
    doc = json.loads(path.read_text())
    if lengan == "bench":
        sel = nama_sel_bench(path)
        badan = bagian_bench(doc)
    else:
        sel = path.stem[len("frontend_"):]
        badan = bagian_faktor(doc)

    berkas = sumber.berkas(sel)
    isi = ([f"# {sel}", "",
            f"Lengan **{lengan}**. Dirakit dari `{path.relative_to(QL)}`"
            + (f" dan {len(berkas)} transkrip panggilan LLM." if berkas else ".")
            + " Seluruh isi disalin apa adanya, tanpa diringkas.", ""]
           + badan + bagian_transkrip(berkas))

    tujuan = OUT / lengan / f"{sel}.md"
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    tujuan.write_text("\n".join(isi), encoding="utf8")
    return tujuan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["bench", "factor", "all"])
    ap.add_argument("--sel", default="", help="rakit satu sel saja")
    a = ap.parse_args()

    sumber = SumberTranskrip()
    sumber.pindai()
    print(f"[sumber] {len(sumber.dari_dir)} sel di disk, "
          f"{len(sumber.dari_tar)} sel dari arsip", flush=True)

    tugas: list[tuple[str, Path]] = []
    if a.arm in ("bench", "all"):
        tugas += [("bench", p) for p in sorted((RESULTS / "bench").glob("bench_*.json"))]
    if a.arm in ("factor", "all"):
        tugas += [("factor", p) for p in sorted((RESULTS / "factor").glob("frontend_*.json"))]

    total = 0
    for lengan, p in tugas:
        sel = nama_sel_bench(p) if lengan == "bench" else p.stem[len("frontend_"):]
        if a.sel and a.sel != sel:
            continue
        t = rakit(lengan, p, sumber)
        if t:
            kb = t.stat().st_size / 1024
            print(f"  {lengan:6s} {sel:40s} -> {t.name}  ({kb:,.0f} KB)")
            total += 1

    sumber.tutup()
    print(f"\n[selesai] {total} berkas di {OUT}")


if __name__ == "__main__":
    main()
