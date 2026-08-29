"""Lengan replikasi LatentMAS: 4 persamaan langkah laten × 3 medium × 3 benchmark.

Satu pemanggilan = SATU SEL matriks eksperimen (satu metode × satu medium ×
satu benchmark × satu seed). Sengaja begitu, bukan satu proses yang menyapu
seluruh matriks: `docs/HASIL_TAHAP0.md` §8.7 mencatat satu proses hanya memakai
~16 GB dari 46 GB VRAM A40 dan 70–95% GPU, sehingga 2–3 sel bisa jalan PARALEL.
Sweep serial di dalam satu proses justru membuang setengah kartu.

    # satu sel
    PYTHONPATH=backend python backend/bench/run_bench.py \
        --task gsm8k --latent-mode gumbel --comm-mode kv --limit 200 --seed 0

    # tiga sel paralel (beri jeda ~30 dtk agar tak rebutan I/O muat model)
    for m in raw gumbel moi; do
      PYTHONPATH=backend python backend/bench/run_bench.py \
          --task gsm8k --latent-mode $m --comm-mode kv --limit 200 --tag s0 &
      sleep 30
    done; wait

Keluaran: `results/bench/bench_<task>_<mode>_<comm>_<tag>.json` berisi ringkasan
sel + jejak per-soal (pertanyaan, jawaban judger, prediksi, benar/salah,
format_ok, durasi, jejak tiap agen). Analisis berpasangan antar-sel dilakukan
`backend/bench/compare.py`.

CATATAN KESETARAAN dengan angka paper. Model dipatok Qwen3-8B, rantai dan teks
prompt diambil dari repo rujukan (`backend/prompts/bench.yaml`), dan penilaian
diport apa adanya. Yang TIDAK setara dan harus dilaporkan: (a) subsample
`--limit`, bukan benchmark penuh; (b) satu seed per sel kecuali dijalankan
ulang; (c) mesin latennya milik proyek ini (`llm/client.py`), bukan
`reference/LatentMAS/methods/latent_mas.py`. (c) itu disengaja — hanya dengan
mesin yang sama keempat persamaan bisa dibandingkan tanpa perancu implementasi.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import BENCH_PROMPTS, QL_ROOT, bootstrap, ensure_out, OUT_BENCH
bootstrap()

# `.env` dimuat SEBELUM modul backend mana pun: llm/client.py membaca
# HF_LOCAL_ONLY saat modul diimpor, bukan saat dipanggil (alasan yang sama
# didokumentasikan di eval/channel_capacity.py).
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(QL_ROOT / ".env", override=False)
except ImportError:
    pass

OUT = ensure_out(OUT_BENCH)


def build_backend(args: argparse.Namespace):
    """Bangun LocalLLMBackend dengan persamaan langkah laten yang diminta."""
    # Mode langkah laten diteruskan lewat env — jalur yang sama dipakai lengan
    # faktor (`factor/run_factor.py`), jadi kedua lengan menyetel mesin dengan
    # cara yang identik dan tak ada cabang konfigurasi kedua yang bisa melenceng.
    os.environ["LATENT_STEP_MODE"] = args.latent_mode
    os.environ["LATENT_STEP_TEMP"] = str(args.latent_temp)
    if args.latent_beta is not None:
        os.environ["LATENT_STEP_BETA"] = str(args.latent_beta)
    if args.latent_alpha is not None:
        os.environ["LATENT_STEP_ALPHA"] = str(args.latent_alpha)

    from llm.client import get_local_backend

    # Snapshot prompt+keluaran tiap panggilan LLM. TANPA argumen ini,
    # `get_local_backend` memakai defaultnya, `./debug/llm_outputs` — di LUAR
    # `results/`, sehingga transkrip lengan bench tidak ikut `kemas_hasil.sh`
    # DAN tidak ikut git (`debug/` gitignored). Di pod sewaan yang akhirnya
    # dihapus, itu berarti transkripnya hilang permanen tanpa ada yang sadar.
    # Lengan faktor sudah menaruhnya di `results/factor/llm_outputs/<tag>`;
    # di sini disamakan supaya kedua lengan punya satu konvensi.
    # Nama folder dibuat IDENTIK dengan stem berkas keluaran sel (lihat `main`),
    # supaya transkrip dan hasilnya bisa dipasangkan tanpa menebak.
    comm = "baseline" if args.baseline else args.comm_mode
    stem = "_".join(p for p in [args.task, args.latent_mode, comm, args.tag] if p)
    return get_local_backend(
        model_name=args.model,
        latent_steps=args.latent_steps,
        use_realign=not args.no_realign,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=False,
        output_log_dir=str(ensure_out(OUT_BENCH) / "llm_outputs" / stem),
    )


def run_cell(args: argparse.Namespace) -> dict:
    from bench.data import load_task
    from bench.pipeline import BASELINE_CHAIN, DEFAULT_CHAIN, BenchPipeline
    from bench.scoring import score_item, summarize

    items = load_task(args.task, limit=args.limit, seed=args.sample_seed)
    chain = BASELINE_CHAIN if args.baseline else DEFAULT_CHAIN

    backend = build_backend(args)
    pipe = BenchPipeline(
        backend,
        comm_mode=args.comm_mode,
        chain=chain,
        prompts_path=Path(args.prompts) if args.prompts else BENCH_PROMPTS,
        max_new_tokens=args.max_new_tokens,
        text_context_chars=args.text_context_chars,
    )

    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    results = []
    t0 = time.time()
    for n, item in enumerate(items, 1):
        out = pipe.run_item(item)
        score = score_item(item, out.answer_text, timeout=args.code_timeout)
        results.append({
            "index": item["index"],
            "question": item["question"],
            "gold": item.get("gold"),
            "answer_text": out.answer_text,
            "duration_s": round(out.duration_s, 3),
            "pipeline_error": out.error,
            "agents": out.agents if args.save_traces else None,
            **score,
        })
        if n % args.log_every == 0 or n == len(items):
            acc = sum(1 for r in results if r["correct"]) / len(results)
            print(f"[{args.task}/{args.latent_mode}/{args.comm_mode}] "
                  f"{n}/{len(items)} acc={acc:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        # KV satu soal tak pernah diwariskan ke soal berikutnya; bersihkan
        # cache alokator secara berkala agar fragmentasi tak menumpuk sepanjang
        # ratusan soal.
        if args.empty_cache_every and n % args.empty_cache_every == 0:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — CPU-only / torch tak ada
                pass

    return {
        "_meta": {
            "task": args.task,
            "task_family": items[0]["task_family"] if items else None,
            "latent_mode": args.latent_mode,
            "latent_steps": args.latent_steps,
            "latent_temp": args.latent_temp,
            "latent_beta": args.latent_beta,
            "latent_alpha": args.latent_alpha,
            "use_realign": not args.no_realign,
            "comm_mode": args.comm_mode,
            "chain": list(chain),
            "baseline": args.baseline,
            "model": args.model,
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "total_time_s": round(time.time() - t0, 1),
            "prompts": str(Path(args.prompts) if args.prompts else BENCH_PROMPTS),
        },
        "summary": summarize(results),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True,
                    choices=["gsm8k", "arc_challenge", "humanevalplus"])
    # ── SUMBU A: persamaan langkah laten, M = {raw, soft, sample, gumbel, moi} ──
    ap.add_argument("--latent-mode", default="raw",
                    choices=["raw", "gumbel", "moi", "sample", "soft", "mix"],
                    help="raw = ridge W_a resmi LatentMAS; keempat sisanya "
                         "keluarga relaksasi diskret R (proyeksi balik ke "
                         "convex hull embedding)")
    ap.add_argument("--latent-steps", type=int, default=10)
    ap.add_argument("--latent-temp", type=float, default=0.7)
    ap.add_argument("--latent-beta", type=float, default=None,
                    help="hanya mode moi (default paper: 1.0)")
    ap.add_argument("--latent-alpha", type=float, default=None,
                    help="hanya mode mix: 0 = raw persis, 1 = soft persis")
    ap.add_argument("--no-realign", action="store_true",
                    help="use_realign=False; hanya berpengaruh di mode raw")
    # ── SUMBU B: medium komunikasi antar-agen ──────────────────────────────
    ap.add_argument("--comm-mode", default="kv",
                    choices=["kv", "kv_and_text", "text"])
    ap.add_argument("--baseline", action="store_true",
                    help="rantai judger saja (lantai agen-tunggal)")
    # ── model & dekode ─────────────────────────────────────────────────────
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--text-context-chars", type=int, default=8000,
                    help="batas konteks teks di comm_mode=text (repo rujukan: 8000)")
    # ── sampling soal & pelaporan ──────────────────────────────────────────
    ap.add_argument("--limit", type=int, default=200,
                    help="jumlah soal (subsample berpasangan); -1 = seluruh benchmark")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="seed PEMILIHAN soal — samakan di semua sel agar berpasangan")
    ap.add_argument("--seed", type=int, default=0, help="seed generasi LLM")
    ap.add_argument("--code-timeout", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--empty-cache-every", type=int, default=25)
    ap.add_argument("--save-traces", action="store_true",
                    help="simpan jejak per-agen (besar; perlu untuk analisis fidelitas)")
    ap.add_argument("--prompts", default="", help="path bench.yaml alternatif")
    ap.add_argument("--tag", default="", help="sufiks nama berkas keluaran")
    args = ap.parse_args()

    if args.limit is not None and args.limit < 0:
        args.limit = None

    payload = run_cell(args)

    comm = "baseline" if args.baseline else args.comm_mode
    parts = [args.task, args.latent_mode, comm]
    if args.tag:
        parts.append(args.tag)
    path = OUT / ("bench_" + "_".join(parts) + ".json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    s = payload["summary"]
    print(f"\n=== {path.name} ===")
    print(f"n={s['n']}  akurasi={s['accuracy']:.4f}  "
          f"format_ok={s['format_rate']:.4f}  "
          f"waktu={payload['_meta']['total_time_s']}s")


if __name__ == "__main__":
    main()
