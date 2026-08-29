"""Lengan faktor alpha: proposal → innovate → construct → gate → skor, SINGLE-PASS.

Bekas `lab/frontend_probe.py`; dipromosikan jadi salah satu dari dua lengan
eksperimen skripsi (lihat `docs/DESAIN_EKSPERIMEN.md` §3). Inilah tugas
SIMBOLIK — pasangan tugas benchmark di `backend/bench/`, dijalankan dengan
mesin laten yang sama persis sehingga selisihnya tak bisa dituduh berasal dari
implementasi yang berbeda.

Kenapa tanpa backtest Qlib. Satu trajectory produksi ≈ 13 menit, mayoritasnya
backtest LightGBM gabungan — metrik yang oleh `docs/AUDIT_KRITIS.md` §S3/B8
justru dinyatakan TIDAK boleh dipakai membandingkan mode, karena ia mencampur
banyak faktor sehingga kontribusi satu ekspresi tak bisa diisolasi. Metrik yang
jujur (per-factor RankIC OOS) dihitung `eval/ic.py` di CPU dan sudah divalidasi
identik 7 desimal terhadap produksi; metrik portofolio ada di `eval/backtest.py`.
Jadi: jalankan bagian yang butuh GPU (rantai agen) saja, lalu skor di CPU.

Satu "run" = satu (arah × seed) → satu FrontEndOutput. Yang direkam:
  - hipotesis, faktor mentah construct, gate_log per-ekspresi, repair
  - waktu per agen + panjang KV (deteksi degenerasi/rambling)
  - untuk tiap ekspresi: cacat semantik (validate_semantics + static_flags),
    lalu IC/ICIR/t/n_unique OOS dari `eval.ic`

Pemakaian (satu sel = satu proses; 2–3 sel muat paralel di A40):
    PYTHONPATH=backend python backend/factor/run_factor.py \
        --comm-mode kv --latent-mode gumbel --latent-steps 10 \
        --seeds 0,1,2 --directions d0,d1 --tag kv_gumbel
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (FACTOR_PROMPTS, QL_ROOT as QL, bootstrap,
                   ensure_out, OUT_FACTOR as _OUT)
bootstrap()

OUT = ensure_out(_OUT)


# Dua arah eksplorasi yang BENAR-BENAR dipakai batch produksi 2026-07-05
# (backend/runs/prod_text_.../stdout.log baris "Direction 0/1"). Dipakai ulang
# supaya hasil sebanding; planning LLM sengaja tidak dijalankan agar variansnya
# tidak mencemari perbandingan antar-lengan.
DIRECTIONS = {
    "d0": "short-term reversal after abnormally high-volume days in small-cap stocks",
    "d1": "mean-reversion in low-volatility stocks during regime transitions",
    # ── Pasangan BERLAWANAN untuk A10 (sensitivitas arah) ────────────────────
    # d0/d1 di atas TIDAK bisa dipakai untuk A10: keduanya sama-sama keluarga
    # mean-reversion, jadi keluaran yang mirip bisa berarti "arahnya memang
    # mirip", bukan "sistem mengabaikan arah". Pasangan di bawah dibuat
    # berlawanan pada TIGA sumbu sekaligus — tanda efek (lanjut vs balik),
    # horizon (panjang vs sangat pendek), dan kolom pembawa sinyal (tren harga
    # vs rentang intraday) — sehingga sistem yang membaca arahnya TIDAK MUNGKIN
    # menghasilkan himpunan ekspresi yang sama untuk keduanya.
    "opp_mom": ("long-horizon price momentum continuation: stocks that trended "
                "up over 30-60 days keep outperforming, signal carried by "
                "sustained directional drift in close prices"),
    "opp_rev": ("very short-horizon contrarian reversal: stocks with the "
                "largest 1-3 day intraday range expansion snap back and "
                "underperform, signal carried by high-low range spikes"),
}


# ── runlog stub: kumpulkan event tanpa menulis ke volume network ─────────────
class Collector:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def _rec(self, level: str, msg: str, **f):
        self.events.append({"level": level, "msg": msg, **{k: str(v)[:400] for k, v in f.items()}})

    def info(self, msg, **f): self._rec("INFO", msg, **f)
    def warn(self, msg, **f): self._rec("WARNING", msg, **f)
    def error(self, msg, **f): self._rec("ERROR", msg, **f)
    def event(self, kind, **f): self._rec("EVENT", kind, **f)

    def step(self, name, **ctx):
        import contextlib
        return contextlib.nullcontext()


def repetition_ratio(text: str) -> float:
    """Fraksi token yang merupakan pengulangan token sebelumnya (deteksi
    degenerasi 'selection selection selection...' — B2/B10)."""
    w = re.findall(r"\w+", (text or "").lower())
    if len(w) < 20:
        return 0.0
    return 1.0 - len(set(w)) / len(w)


def instrument(pipeline, collector, keep_text: int = 6000):
    """Bungkus tiap LatentAgent.run agar per-agen tercatat (durasi, panjang KV,
    panjang teks, rasio repetisi) tanpa mengubah kode produksi.

    `n_in_tok` dicatat karena sumbu A6 (biaya per faktor diterima) butuh TOKEN
    DIPROSES, bukan hanya token yang di-emit; tanpa ini biaya lengan `text`
    (prompt panjang, KV nol) tak bisa dibandingkan adil dengan lengan `kv`.

    `text` (dipotong `keep_text` char) dicatat karena sumbu A7 butuh membaca
    keluaran ASLI agen hulu: kepatuhan palette hanya bisa dihitung bila palette
    design tersimpan. Nol biaya GPU, ~5 KB per run.
    """
    trace: list[dict] = []
    for name, agent in pipeline.agents.items():
        orig = agent.run

        def wrapped(_orig=orig, _name=name, **kw):
            t0 = time.time()
            res = _orig(**kw)
            trace.append({
                "agent": _name, "mode": res.mode,
                "s": round(time.time() - t0, 2),
                "latent_s": res.latent_s, "gen_s": res.gen_s,
                # B6: anggaran vs langkah yang benar-benar berjalan.
                "n_latent_steps": getattr(res, "n_latent_steps", 0),
                "latent_stop": getattr(res, "latent_stop", "off"),
                "kv_len": res.kv_seq_len, "n_out_tok": res.n_output_tokens,
                "n_in_tok": res.n_input_tokens,
                "text_len": len(res.text or ""),
                "rep_ratio": round(repetition_ratio(res.text or ""), 3),
                "parsed_ok": res.parsed is not None,
                "text": (res.text or "")[:keep_text],
            })
            return res

        agent.run = wrapped
    return trace


def build_backend(args):
    from llm.client import LocalLLMBackend
    return LocalLLMBackend(
        model_name=args.model,
        device="cuda",
        latent_steps=args.latent_steps,
        use_realign=not args.no_realign,
        enable_thinking=False,
        log_tensors=False,
        store_kv=False,
        output_log_dir=str(OUT / "llm_outputs" / (args.tag or "probe")),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95,
        knn_enabled=False,          # auto-disabled saat latent_steps>0 (client.py)
        # B6. Default None → engine memakai env/0,999. Lengan yang ingin
        # mereplikasi baseline pra-B6 memberi 1.0 (mematikan early-stop).
        latent_early_stop_cos=getattr(args, "early_stop_cos", None),
    )


def run_once(backend, args, direction: str, seed: int, prompts_path: Path):
    import torch
    from mas.agent import load_all_agents
    from mas.pipeline import FrontEndPipeline

    torch.manual_seed(seed)
    col = Collector()
    agents = load_all_agents(backend, runlog=col, path=prompts_path)
    chain = getattr(args, "chain", None)
    if isinstance(chain, str):
        chain = tuple(c.strip() for c in chain.split(",") if c.strip()) or None
    pipe = FrontEndPipeline(backend, runlog=col, agents=agents,
                            use_regulator=True, comm_mode=args.comm_mode,
                            max_repair_attempts=args.max_repair,
                            chain=chain,
                            free_form=getattr(args, "free_form", None))
    trace = instrument(pipe, col)

    t0 = time.time()
    err = None
    try:
        fe = pipe.run(direction=direction)
    except Exception:                                # noqa: BLE001
        err = traceback.format_exc()[-2000:]
        fe = None
    dur = round(time.time() - t0, 2)

    if fe is None:
        return {"error": err, "duration_s": dur, "agent_trace": trace,
                "chain": ",".join(pipe.chain), "free_form": pipe.free_form,
                "events": col.events}

    return {
        "duration_s": dur,
        "chain": ",".join(pipe.chain),
        "free_form": pipe.free_form,
        "hypothesis": fe.hypothesis,
        "factors": fe.factors,
        "passing": fe.expressions,
        "gate_log": fe.gate_log,
        "repaired": fe.repaired,
        "repair_attempts": fe.repair_attempts,
        "gate_error": fe.gate_error,
        "construct_text_len": len(fe.judger_text or ""),
        "construct_rep_ratio": round(repetition_ratio(fe.judger_text or ""), 3),
        "construct_text_head": (fe.judger_text or "")[:600],
        "agent_trace": trace,
        "events": [e for e in col.events if e["level"] in ("WARNING", "ERROR")],
    }


# ── skoring CPU ─────────────────────────────────────────────────────────────
class _time_budget:
    """Batas waktu per-ekspresi. Tanpa ini satu operator lambat menyandera
    seluruh sweep: `TS_QUANTILE` rolling dan `REGBETA`/`REGRESI` (joblib
    per-instrumen) bisa memakan belasan menit untuk SATU ekspresi, dan sweep
    yang macet 11 menit di satu faktor terlihat seperti GPU yang menggantung.
    Ekspresi yang lewat batas ditandai `eval_error='timeout'` — dilaporkan apa
    adanya, bukan disamarkan jadi faktor tanpa IC."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        import signal

        def _raise(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"melebihi {self.seconds}s")

        try:
            self._old = signal.signal(signal.SIGALRM, _raise)
            signal.alarm(self.seconds)
        except ValueError:
            self._old = None
        return self

    def __exit__(self, *exc):
        import signal

        if self._old is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False


def _score_one_expr(e: str, lab, budget_s: int, quantile: float, cost_bps: float):
    """Skor SATU ekspresi. Dipisah dari `score_expressions` supaya bisa
    dijalankan di proses pekerja terpisah (paralelisasi skoring korpus).

    Mengembalikan `(entry, series_or_None)` — `entry` adalah dict cache yang
    sama persis dengan yang dulu disusun inline. Tak menyentuh `runs`, `cache`,
    atau state bersama apa pun, sehingga aman dipanggil dari `Pool`.
    """
    from eval.backtest import backtest_values
    from eval.ic import ICResult
    from gate.static_flags import static_flags
    from gate.factor_regulator import validate_semantics

    ok, errs = validate_semantics(e)
    bt = None
    try:
        with _time_budget(budget_s):
            vals = lab.values(e)
            res, ser = lab._ic_core(vals)
            if res.error is not None or res.ic is None:
                ser = None
            bt = backtest_values(lab, vals, quantile=quantile, cost_bps=cost_bps)
    except TimeoutError:
        res, ser = ICResult(None, None, 0, None, 0.0, 0.0,
                            error=f"timeout>{budget_s}s"), None
    except Exception as ex:  # noqa: BLE001 — ekspresi LLM gagal ribuan cara
        res, ser = ICResult(None, None, 0, None, 0.0, 0.0,
                            error=f"{type(ex).__name__}: {ex}"), None

    entry = {
        "sem_ok": bool(ok), "sem_errors": errs,
        "flags": static_flags(e),
        "ic": res.ic, "icir": res.icir, "tstat": res.tstat,
        "n_days": res.n_days, "coverage": res.coverage,
        "n_unique": res.n_unique, "eval_error": res.error,
    }
    hidup = (res.ic is not None and (res.n_unique or 0) > 2)
    if bt is not None and hidup:
        entry.update({f"bt_{k}": v for k, v in bt.as_dict().items()})
    elif bt is not None:
        entry["bt_error"] = (f"tak dilaporkan: ekspresi degenerate "
                             f"(n_unique={res.n_unique}, ic={res.ic})")
    return entry, ser


# ── paralelisasi skoring: pekerja mewarisi `lab` lewat fork (copy-on-write) ──
# `Lab` memegang DataFrame data pasar (~1 GB pada jendela 4 tahun). Ia TIDAK
# dioper lewat argumen Pool — itu akan mem-pickle DataFrame-nya ke tiap pekerja.
# Sebagai gantinya ia ditaruh di global modul SEBELUM Pool dibuat, dan proses
# anak hasil `fork` mewarisinya tanpa menyalin selama hanya dibaca (semua jalur
# `values`/`_ic_core`/`backtest_values` hanya membaca). Ini yang membuat N
# pekerja berbagi SATU salinan data alih-alih N salinan.
_MP_LAB = None


def _mp_init() -> None:
    import os as _os
    import signal as _signal
    # REGBETA/REGRESI memanggil joblib `n_jobs=-1`; tanpa batas ini tiap pekerja
    # men-spawn 16 sub-pekerja lagi → N*16 proses berebut CPU dan meledakkan RAM.
    _os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    # Pekerja hasil `fork` MEWARISI penangan sinyal induk. Kalau induk memasang
    # penangan "berhenti anggun" (eval/rescore_all.py memasangnya supaya
    # skoring berjam-jam yang dibunuh saat pod mati tetap menyimpan progres),
    # pekerja ikut mewarisi penangan itu — dan `Pool.terminate()`, yang bekerja
    # dengan MENGIRIM SIGTERM ke pekerja, jadi tidak mematikan siapa pun.
    # Akibatnya proses menggantung selamanya justru saat diminta berhenti.
    # Terukur 2026-08-28: 4 salinan pesan "sinyal diterima" lalu hang.
    # Pekerja harus kembali ke perilaku default: mati kalau disuruh mati.
    for _s in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(_s, _signal.SIG_DFL)
        except (ValueError, OSError):   # bukan thread utama
            pass


def _mp_score_one(args):
    e, budget_s, quantile, cost_bps = args
    return e, _score_one_expr(e, _MP_LAB, budget_s, quantile, cost_bps)


def score_expressions(runs: list[dict], window=None, series_path: Path | None = None,
                      budget_s: int = 90, cache: dict | None = None,
                      lab=None, series_cache: dict | None = None,
                      quantile: float = 0.1, cost_bps: float = 0.0,
                      workers: int = 1, on_progress=None) -> None:
    """Isi tiap faktor dengan cacat semantik + IC/ICIR + metrik backtest (in-place).

    Deret IC harian juga disimpan (parquet) supaya analisis bisa mengelompokkan
    faktor jadi KLASTER SINYAL — ukuran keragaman pencarian (AUDIT_KRITIS §2.4)
    yang jauh lebih informatif daripada sekadar jumlah ekspresi.

    `cache`/`series_cache`/`lab` boleh dioper dari luar supaya skoring BANYAK
    tag berjalan dalam satu proses tanpa memuat ulang data pasar dan tanpa
    mengevaluasi ulang ekspresi yang sama (631 ekspresi unik dari 805 total
    lintas-tag). Dipakai `eval/rescore_all.py`.

    METRIK SELAIN IC (ditambahkan 2026-08-10). `configs/matriks.yaml` sudah
    lama mendeklarasikan `backtest: [ann_return, sharpe, max_drawdown,
    turnover, hit_rate]` sebagai metrik lengan faktor, dan `eval/backtest.py`
    sudah mengimplementasikannya, tapi tak ada satu pun pemanggil yang
    menyambungkan keduanya — sehingga yang benar-benar tercatat hanya IC.
    Sekarang tersambung. RankIC menjawab "apakah ekspresi ini punya daya
    prediksi lintas-saham"; ia TIDAK menjawab "kalau diperdagangkan, apa
    hasilnya" — return, drawdown, dan biaya perputaran posisi tak muncul di
    korelasi peringkat sama sekali.

    Ketiganya (IC · deret harian · backtest) dihitung dari SATU kali evaluasi
    ekspresi. Itu sengaja: evaluasi DSL adalah bagian termahal skoring korpus,
    dan memanggil `ic_full()` lalu `backtest_expression()` akan mengevaluasi
    ekspresi yang sama dua kali — pada ekspresi rolling bersarang itu berarti
    puluhan detik terbuang per ekspresi.
    """
    import pandas as pd

    from eval.ic import Lab
    # Pre-import factor_ast memutus circular import factor_regulator →
    # coder/__init__ → evaluators → factor_regulator (sama seperti yang
    # dilakukan FrontEndPipeline._build_regulator_gate saat import COLD).
    # Dilakukan di INDUK sebelum fork supaya modulnya sudah termuat di semua
    # proses anak.
    import dsl.factor_ast  # noqa: F401
    import gate.factor_regulator  # noqa: F401

    if lab is None:
        lab = Lab(mode="fast", window=window)
    if cache is None:
        cache = {}
    series_all: dict[str, "pd.Series"] = series_cache if series_cache is not None else {}
    series: dict[str, "pd.Series"] = {}

    # Ekspresi unik yang BELUM ada di cache — inilah kerja mahal sebenarnya.
    perlu = list(dict.fromkeys(
        f["expression"] for r in runs for f in (r.get("factors") or [])
        if f.get("expression") and f["expression"] not in cache
    ))

    if workers > 1 and perlu:
        # Materialkan data pasar di INDUK dulu — `.df` dan `.label` adalah
        # properti malas; menyentuhnya sekarang membuat proses anak hasil fork
        # mewarisi DataFrame yang sudah jadi, bukan memuat ulang HDF5 ~1 GB
        # masing-masing.
        _ = lab.df, lab.label

        global _MP_LAB
        _MP_LAB = lab
        ctx = mp.get_context("fork")
        tugas = [(e, budget_s, quantile, cost_bps) for e in perlu]
        # maxtasksperchild: ekspresi rolling-bersarang patologis bisa
        # meninggalkan memori yang tak dibebaskan pandas; daur ulang pekerja
        # secara berkala mengembalikannya ke OS.
        with ctx.Pool(workers, initializer=_mp_init, maxtasksperchild=12) as pool:
            for i, (e, (entry, ser)) in enumerate(
                    pool.imap_unordered(_mp_score_one, tugas, chunksize=1), 1):
                cache[e] = entry
                if ser is not None:
                    series_all[e] = ser
                if on_progress:
                    on_progress(i, len(perlu), e, entry)
        _MP_LAB = None
    else:
        for i, e in enumerate(perlu, 1):
            entry, ser = _score_one_expr(e, lab, budget_s, quantile, cost_bps)
            cache[e] = entry
            if ser is not None:
                series_all[e] = ser
            if on_progress:
                on_progress(i, len(perlu), e, entry)

    # Tempel hasil ke `runs` (in-place) — murah, tak ada evaluasi di sini.
    for r in runs:
        for f in r.get("factors", []) or []:
            e = f.get("expression", "")
            if not e or e not in cache:
                continue
            if e in series_all:
                series[e] = series_all[e]
            # BUANG metrik backtest LAMA sebelum menempel yang baru.
            #
            # `f` bisa datang dari salinan `frontend_<tag>.json` yang SUDAH
            # berisi `bt_*` hasil penilaian sebelumnya (mis. pasar A-share).
            # Ketika ekspresi ternyata degenerate di pasar/jendela yang baru,
            # `_score_one_expr` sengaja TIDAK melaporkan backtest — entry-nya
            # hanya memuat `bt_error` tanpa satu pun kunci `bt_*` lain.
            # `dict.update()` maka meninggalkan angka lama itu utuh, sehingga
            # baris tersebut membawa Sharpe/turnover dari pasar yang salah
            # sambil field IC-nya sudah dari pasar yang benar. Terukur 7 dari
            # 1.373 baris pada penilaian IDX 2026-08-28 (ketahuan lewat
            # `bt_n_long` = 433 pada universe 37 saham). Menghapusnya lebih
            # dulu membuat "tidak dilaporkan" terbaca sebagai None, bukan
            # sebagai angka pasar lain.
            for _k in [k for k in f if k.startswith("bt_")]:
                del f[_k]
            f.update(cache[e])
            f["passed_gate"] = e in (r.get("passing") or [])

    if series_path is not None and series:
        pd.DataFrame(series).to_parquet(series_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--comm-mode", default="kv",
                    choices=["kv", "kv_and_text", "text", "summary"])
    ap.add_argument("--latent-steps", type=int, default=10)
    # SUMBU A skripsi: persamaan langkah laten, M = {raw, soft, sample, gumbel,
    # moi}. `raw` = ridge W_a resmi LatentMAS (Teorema A.1) dan satu-satunya
    # anggota di luar keluarga relaksasi diskret R; keempat sisanya
    # memproyeksikan balik ke convex hull embedding.
    ap.add_argument("--latent-mode", default="raw",
                    choices=["raw", "gumbel", "moi", "sample", "soft", "mix"],
                    help="persamaan langkah laten (diteruskan via LATENT_STEP_MODE)")
    ap.add_argument("--latent-temp", type=float, default=0.7)
    ap.add_argument("--latent-alpha", type=float, default=None,
                    help="hanya mode mix: 0 = raw persis, 1 = soft persis")
    ap.add_argument("--early-stop-cos", dest="early_stop_cos", type=float,
                    default=None,
                    help="B6: berhenti bila cos(h_k,h_k-1) > nilai ini. "
                         "1.0 = matikan (baseline pra-B6). None = default engine (0,999)")
    ap.add_argument("--no-realign", action="store_true", help="use_realign=False (G6)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--directions", default="d0,d1")
    ap.add_argument("--prompts", default="", help="path prompts.yaml alternatif")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-repair", type=int, default=3)
    ap.add_argument("--holdout", action="store_true",
                    help="skor pada 2022-01-01..2025-12-26 (holdout sejati)")
    ap.add_argument("--chain", default="",
                    help="susunan agen front-end, mis. 'proposal,innovate,construct' "
                         "(kosong = proposal,design,construct)")
    ap.add_argument("--free-form", dest="free_form", default=None,
                    action="store_true",
                    help="lepas klem FIDELITY di construct (default: ikut chain)")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--score-only", action="store_true",
                    help="lewati GPU; skor ulang frontend_<tag>.json yang sudah ada")
    ap.add_argument("--skip-score", action="store_true",
                    help="lewati tahap skoring CPU; tulis frontend_<tag>.json "
                         "lalu keluar. Pasangan dari --score-only: fase GPU "
                         "berhenti begitu ekspresi jadi, dan skoringnya "
                         "dijalankan sebagai proses CPU terpisah yang boleh "
                         "berjalan BERSAMAAN dengan sel GPU berikutnya — atau "
                         "di mesin lain sama sekali (skoring tak butuh GPU).")
    ap.add_argument("--budget", type=int, default=900,
                    help="anggaran detik per ekspresi saat skoring. Dinaikkan "
                         "dari 90 ke 900 (2026-08-10): anggaran ketat itu "
                         "peninggalan saat skoring dikira menahan GPU. Ia "
                         "TIDAK, dan efek sampingnya buruk — beberapa fungsi "
                         "DSL (TS_SKEW/TS_KURT/TS_MAD/REGRESI, semuanya "
                         "`rolling().apply` dengan callback Python) berada di "
                         "bibir 90 dtk, sehingga ekspresi yang sama bisa "
                         "ber-IC atau `ic=None` TERGANTUNG BEBAN MESIN. Itu "
                         "membuat perbandingan antar-metode bergantung pada "
                         "hal yang tak ada kaitannya dengan metode. Terukur: "
                         "5 ekspresi hilang begitu di run 2026-08-10 "
                         "(results/pendukung/ragam_eval_error.json).")
    args = ap.parse_args()

    if args.score_only:
        path = OUT / f"frontend_{args.tag}.json"
        doc = json.loads(path.read_text())
        window = ("2022-01-01", "2025-12-26") if args.holdout else None
        score_expressions(doc["runs"], window=window, budget_s=args.budget,
                          series_path=OUT / f"icseries_{args.tag}.parquet")
        path.write_text(json.dumps(doc, indent=2, default=str))
        print(f"di-skor ulang → {path}")
        return

    # G3: mode langkah laten diteruskan ke client.py lewat env (patch minimal).
    os.environ["LATENT_STEP_MODE"] = args.latent_mode
    os.environ["LATENT_STEP_TEMP"] = str(args.latent_temp)
    if getattr(args, "latent_alpha", None) is not None:
        os.environ["LATENT_STEP_ALPHA"] = str(args.latent_alpha)

    OUT.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts) if args.prompts else FACTOR_PROMPTS

    backend = build_backend(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dirs = [d.strip() for d in args.directions.split(",") if d.strip()]

    runs = []
    for d in dirs:
        for s in seeds:
            print(f"\n=== {args.tag} | dir={d} seed={s} | comm={args.comm_mode} "
                  f"ls={args.latent_steps} latent_mode={args.latent_mode} ===", flush=True)
            r = run_once(backend, args, DIRECTIONS[d], s, prompts_path)
            r.update({"direction": d, "seed": s, "comm_mode": args.comm_mode,
                      "latent_steps": args.latent_steps, "latent_mode": args.latent_mode,
                      "model": args.model, "use_realign": not args.no_realign,
                      "prompts": str(prompts_path)})
            n_fac = len(r.get("factors") or [])
            n_pass = len(r.get("passing") or [])
            print(f"    -> {r['duration_s']}s  n_factors={n_fac} n_pass={n_pass} "
                  f"repaired={r.get('repaired')} err={bool(r.get('error'))}", flush=True)
            runs.append(r)
            # buang KV/tensor sisa antar-run
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()

    # Tulis SEBELUM skoring: kerja GPU tak boleh hilang gara-gara error di
    # tahap CPU (skoring bisa diulang dengan --score-only).
    path = OUT / f"frontend_{args.tag}.json"
    path.write_text(json.dumps({"args": vars(args), "runs": runs}, indent=2, default=str))

    if args.skip_score:
        # GPU sudah selesai; skoring diserahkan ke proses CPU terpisah supaya
        # kartu langsung bisa dipakai sel berikutnya alih-alih menganggur
        # beberapa menit menunggu evaluasi DSL lintas ~4.370 saham × 243 hari.
        print(f"tersimpan (BELUM di-skor) → {path}")
        print(f"skor nanti dengan: --score-only --tag {args.tag}")
        return

    window = ("2022-01-01", "2025-12-26") if args.holdout else None
    print("\n[probe] skoring ekspresi di CPU ...", flush=True)
    score_expressions(runs, window=window, budget_s=args.budget,
                      series_path=OUT / f"icseries_{args.tag}.parquet")
    path.write_text(json.dumps({"args": vars(args), "runs": runs}, indent=2, default=str))
    print(f"tersimpan → {path}")

    # ringkasan cepat
    allf = [f for r in runs for f in (r.get("factors") or [])]
    ok = [f for f in allf if f.get("ic") is not None]
    alive = [f for f in ok if (f.get("n_unique") or 0) > 2]
    sem_bad = [f for f in allf if f.get("sem_ok") is False]
    print(f"\n[{args.tag}] runs={len(runs)} ekspresi={len(allf)} "
          f"lolos-gate={sum(1 for f in allf if f.get('passed_gate'))} "
          f"ber-IC={len(ok)} hidup={len(alive)} cacat-semantik={len(sem_bad)}")
    if ok:
        import statistics as st
        print(f"  mean IC={st.mean(f['ic'] for f in ok):+.5f} "
              f"mean |IC|={st.mean(abs(f['ic']) for f in ok):.5f} "
              f"max |IC|={max(abs(f['ic']) for f in ok):.5f} "
              f"IC>0={sum(1 for f in ok if f['ic'] > 0)}/{len(ok)}")


if __name__ == "__main__":
    main()
