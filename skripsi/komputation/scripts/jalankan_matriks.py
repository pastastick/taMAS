#!/usr/bin/env python3
"""Jalankan matriks eksperimen: antrean GPU + pipeline skoring CPU paralel.

Tiga hal yang tidak dilakukan `gen_perintah.py --parallel N`, dan kenapa
masing-masing penting di sini.

1. ANTREAN, BUKAN BARRIER. Generator mengeluarkan batch `cmd & cmd & wait`:
   batch berikutnya menunggu sel TERLAMBAT di batch sebelumnya. Durasi sel di
   matriks ini timpang (baseline satu agen vs rantai tiga agen; humanevalplus
   164 soal vs gsm8k 200; `text` tanpa langkah laten vs `kv` 10 langkah), jadi
   barrier itu membuat slot GPU menganggur menunggu tetangganya. Di sini slot
   dijaga selalu penuh.

2. SKORING CPU DIPIPELINE. Satu sel faktor = fase GPU (agen menulis ekspresi
   DSL) lalu fase CPU (evaluasi ekspresi + RankIC + backtest lintas ~4.370
   saham × 243 hari). Kalau keduanya dalam satu proses, KARTU MENGANGGUR
   selama fase CPU. Di sini sel faktor dijalankan dengan `--skip-score`, dan
   begitu berkas `frontend_<tag>.json`-nya muncul, skoringnya dilepas sebagai
   proses CPU terpisah yang berjalan BERSAMAAN dengan sel GPU berikutnya.
   Karena itu pula sel faktor diantrikan LEBIH DULU (`--urutan faktor-dulu`,
   default): makin awal ekspresi jadi, makin banyak skoring CPU yang bisa
   bersembunyi di balik waktu GPU lengan benchmark.

3. RESUMABLE. Sel yang keluarannya sudah ada di-SKIP. Pod RunPod bisa mati di
   tengah jalan; menjalankan ulang skrip ini melanjutkan, bukan mengulang dari
   nol. Nama berkas keluaran diturunkan dengan aturan yang sama persis seperti
   `run_bench.py` / `run_factor.py` (lihat `output_path`), jadi deteksi "sudah
   jadi" tidak menebak.

Log tiap sel masuk ke berkasnya sendiri di `results/logs/`, supaya kegagalan
satu sel bisa dibaca tanpa mengurai keluaran belasan proses yang bertumpuk.

    python scripts/jalankan_matriks.py --arm all --slots 2
    python scripts/jalankan_matriks.py --arm all --slots 2 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import OUT_BENCH, OUT_FACTOR, RESULTS  # noqa: E402

LOGS = RESULTS / "logs"


def cells(arm: str) -> list[str]:
    """Daftar perintah dari gen_perintah.py — sumber kebenaran tunggal matriks."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_perintah.py"), "--arm", arm],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )
    return [ln.strip() for ln in out.stdout.splitlines()
            if ln.strip() and not ln.startswith("#")]


def _flag(cmd: str, name: str, default: str = "") -> str:
    m = re.search(rf"--{name}\s+(\S+)", cmd)
    return m.group(1) if m else default


def is_factor(cmd: str) -> bool:
    return "run_factor.py" in cmd


def output_path(cmd: str) -> Path:
    """Berkas keluaran sel — aturan penamaan disalin dari kedua runner.

    run_bench.py : results/bench/bench_<task>_<latent_mode>_<comm|baseline>[_<tag>].json
    run_factor.py: results/factor/frontend_<tag>.json   (BUKAN factor_<tag>.json)
    """
    if is_factor(cmd):
        return OUT_FACTOR / f"frontend_{_flag(cmd, 'tag')}.json"
    task = _flag(cmd, "task")
    mode = _flag(cmd, "latent-mode", "raw")
    comm = "baseline" if "--baseline" in cmd else _flag(cmd, "comm-mode", "kv")
    parts = [task, mode, comm]
    tag = _flag(cmd, "tag")
    if tag:
        parts.append(tag)
    return OUT_BENCH / ("bench_" + "_".join(parts) + ".json")


def score_path(cmd: str) -> Path:
    """Penanda bahwa fase CPU sel faktor sudah selesai (ditulis score_expressions)."""
    return OUT_FACTOR / f"icseries_{_flag(cmd, 'tag')}.parquet"


def log_path(cmd: str, suffix: str = "") -> Path:
    return LOGS / (output_path(cmd).stem + suffix + ".log")


def lock_path(cmd: str) -> Path:
    return LOGS / (output_path(cmd).stem + ".lock")


def sedang_jalan(cmd: str) -> int:
    """PID sel ini kalau sedang dijalankan proses LAIN, 0 kalau tidak.

    Kenapa berkas kunci, bukan sekadar "lewati kalau keluarannya sudah ada".
    Deteksi berbasis keluaran punya lubang yang mahal: sel yang SEDANG BERJALAN
    belum menulis keluaran, sehingga runner kedua menganggapnya belum
    dikerjakan dan meluncurkan DUPLIKATNYA. Yang terjadi 2026-08-10 09:55:
    satu sel `text` yang sudah 50/100 diduplikasi dari nol, dua proses menulis
    ke log yang sama (log aslinya ter-truncate), dan VRAM nyaris jebol karena
    tiga sel 21GB berebut kartu 46GB.

    Kunci berisi PID. Kunci yatim (PID-nya sudah mati, mis. pod restart)
    diabaikan otomatis — jadi tak perlu pembersihan manual yang bisa terlupa.
    """
    p = lock_path(cmd)
    try:
        pid = int(p.read_text().strip())
    except Exception:  # noqa: BLE001 — tak ada kunci / isinya rusak
        return 0
    try:
        os.kill(pid, 0)          # sinyal 0 = cek keberadaan, tidak membunuh
        return pid
    except OSError:
        return 0                 # kunci yatim


def score_cmd(cmd: str) -> str:
    """Perintah fase CPU untuk sel faktor: skor ulang JSON yang barusan ditulis."""
    keep = ["--model", "--tag"]
    parts = [f"PYTHONPATH=backend {sys.executable} backend/factor/run_factor.py",
             "--score-only"]
    for k in keep:
        v = _flag(cmd, k.lstrip("-"))
        if v:
            parts.append(f"{k} {v}")
    if "--holdout" in cmd:
        parts.append("--holdout")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True,
                    choices=["bench", "factor", "interpolasi", "all"],
                    help="'interpolasi' = hanya sel sumbu C (mode `mix`); "
                         "campuran run_bench.py + run_factor.py, keduanya "
                         "ditangani dispatch yang sama")
    ap.add_argument("--slots", type=int, default=2,
                    help="sel GPU serentak; A40 46GB muat 2 aman (~16GB/sel, "
                         "docs/HASIL_TAHAP0.md §8.7)")
    ap.add_argument("--cpu-slots", type=int, default=3,
                    help="proses skoring CPU serentak (tiap proses pakai "
                         "LAB_MAX_WORKERS=3 core joblib)")
    ap.add_argument("--tanpa-skor-cpu", action="store_true",
                    help="JANGAN jalankan tahap skoring CPU sama sekali; sel "
                         "faktor berhenti setelah `frontend_<tag>.json` ditulis. "
                         "Dipakai saat sewa GPU ditagih per WAKTU HIDUP "
                         "instance: skoring tak butuh GPU, jadi menahan pod "
                         "hidup untuk mengerjakannya berarti membayar harga "
                         "GPU untuk pekerjaan CPU. Skor belakangan di mesin "
                         "sendiri: `python backend/eval/rescore_all.py "
                         "--budget 900` (memuat data pasar sekali dan berbagi "
                         "cache antar-tag, jadi lebih murah daripada per-sel).")
    ap.add_argument("--stagger", type=int, default=30,
                    help="jeda detik antar-start GPU; fase muat model rebutan "
                         "I/O network storage kalau serentak")
    ap.add_argument("--vram-bebas-min", type=int, default=24000,
                    help="MiB VRAM bebas yang harus tersedia sebelum sel GPU "
                         "BARU dimulai. `--slots` saja tidak cukup: sel faktor "
                         "memakai ~21GB (max_new_tokens 4096 + rantai 3 agen), "
                         "sel bench ~18GB, jadi batas jumlah yang sama tidak "
                         "aman untuk keduanya. Gerbang ini membuat jumlah sel "
                         "serentak menyesuaikan diri dengan yang NYATA terpakai.")
    ap.add_argument("--ulang-maks", type=int, default=1,
                    help="berapa kali sel yang gagal diantrikan ulang. OOM "
                         "biasanya sembuh sendiri saat sel tetangga selesai, "
                         "jadi sekali ulang menyelamatkan sel tanpa intervensi.")
    ap.add_argument("--urutan", default="faktor-dulu",
                    choices=["faktor-dulu", "generator"],
                    help="faktor-dulu: sel faktor diantrikan lebih dahulu agar "
                         "skoring CPU-nya menumpang waktu GPU lengan benchmark")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun", action="store_true",
                    help="jangan skip sel yang sudah punya keluaran")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    OUT_BENCH.mkdir(parents=True, exist_ok=True)
    OUT_FACTOR.mkdir(parents=True, exist_ok=True)

    all_cells = cells(args.arm)
    if args.urutan == "faktor-dulu":
        all_cells = ([c for c in all_cells if is_factor(c)]
                     + [c for c in all_cells if not is_factor(c)])

    todo, skipped, dikunci = [], [], []
    for c in all_cells:
        if not args.rerun and output_path(c).exists():
            skipped.append(c)
        elif sedang_jalan(c):
            # Sedang dikerjakan proses lain — JANGAN duplikasi.
            dikunci.append(c)
        else:
            todo.append(c)
    for c in dikunci:
        print(f"#   SEDANG JALAN (pid {sedang_jalan(c)}) {output_path(c).name}")

    # Sel faktor yang fase GPU-nya sudah selesai di run sebelumnya tapi fase
    # CPU-nya belum — antrikan skoringnya saja, jangan ulangi GPU-nya.
    pending_score = [] if args.tanpa_skor_cpu else [
        c for c in skipped if is_factor(c) and not score_path(c).exists()]

    print(f"# {len(todo)} sel GPU, {len(skipped)} dilewati, "
          f"{len(pending_score)} skoring CPU tertunggak")
    if args.dry_run:
        for c in todo:
            kind = "FAKTOR(gpu)" if is_factor(c) else "bench"
            print(f"  [{kind}] -> {output_path(c).name}")
        for c in pending_score:
            print(f"  [skor-cpu] -> {score_path(c).name}")
        return

    gpu: list[tuple] = []          # (proc, cmd, t0)
    cpu: list[tuple] = []          # (proc, cmd, t0)
    queue = list(todo)
    cpu_queue = list(pending_score)
    done, failed, scored, score_failed = [], [], [], []
    percobaan: dict[str, int] = {}
    t_start = time.time()

    def vram_bebas() -> int:
        """MiB VRAM bebas; -1 kalau nvidia-smi tak bisa dibaca (jangan blokir)."""
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            used, total = (int(x) for x in r.stdout.strip().split("\n")[0].split(","))
            return total - used
        except Exception:  # noqa: BLE001 — gerbang ini best-effort
            return -1

    def launch(cmd: str, on_gpu: bool):
        if on_gpu:
            shell = cmd.replace("PYTHONPATH=backend python ",
                                f"PYTHONPATH=backend {sys.executable} ")
            # Fase CPU dilepas dari proses GPU — lihat docstring butir 2.
            if is_factor(cmd) and "--skip-score" not in shell:
                shell += " --skip-score"
            lp = log_path(cmd)
        else:
            shell = score_cmd(cmd)
            lp = log_path(cmd, "_skor")
        fh = open(lp, "w")
        p = subprocess.Popen(shell, shell=True, cwd=ROOT,
                             stdout=fh, stderr=subprocess.STDOUT)
        if on_gpu:
            lock_path(cmd).write_text(str(p.pid))
        (gpu if on_gpu else cpu).append((p, cmd, time.time()))
        tag = "GPU" if on_gpu else "CPU"
        print(f"[MULAI {tag}] {output_path(cmd).name}  (log: {lp.name})", flush=True)

    def sel_eksternal() -> int:
        """Sel yang jalan di bawah proses LAIN (kunci hidup, bukan luncuran kita).

        Slot GPU adalah sumber daya BERSAMA satu kartu, bukan milik satu runner.
        Tanpa ikut menghitung sel eksternal, runner yang dijalankan saat masih
        ada sel yatim akan menambah `--slots` sel LAGI di atasnya — 4 sel x 21GB
        di kartu 46GB, yaitu OOM. `dikunci` dihitung ulang tiap iterasi karena
        sel eksternal bisa selesai kapan saja.
        """
        return sum(1 for c in all_cells
                   if c not in [x[1] for x in gpu] and sedang_jalan(c))

    while queue or gpu or cpu_queue or cpu:
        while queue and len(gpu) + sel_eksternal() < args.slots:
            # Gerbang VRAM. Slot pertama selalu boleh jalan (kalau tidak, sel
            # yang butuh lebih dari ambang tak akan pernah dapat giliran dan
            # matriks mandek diam-diam).
            bebas = vram_bebas()
            if gpu and 0 <= bebas < args.vram_bebas_min:
                print(f"[TUNGGU] VRAM bebas {bebas} MiB < {args.vram_bebas_min} "
                      f"— tunda start sel berikutnya ({len(gpu)} sel jalan)",
                      flush=True)
                break
            launch(queue.pop(0), on_gpu=True)
            if queue and len(gpu) < args.slots:
                time.sleep(args.stagger)
        while cpu_queue and len(cpu) < args.cpu_slots:
            launch(cpu_queue.pop(0), on_gpu=False)

        time.sleep(5)

        for entry in list(gpu):
            p, cmd, t0 = entry
            if p.poll() is None:
                continue
            gpu.remove(entry)
            lock_path(cmd).unlink(missing_ok=True)
            dur, name = (time.time() - t0) / 60, output_path(cmd).name
            if p.returncode == 0 and output_path(cmd).exists():
                done.append(cmd)
                print(f"[OK  GPU {len(done)}/{len(todo)}] {name}  {dur:.1f} mnt",
                      flush=True)
                if is_factor(cmd) and not args.tanpa_skor_cpu:
                    cpu_queue.append(cmd)   # fase CPU-nya menyusul, paralel
            else:
                n = percobaan.get(cmd, 0) + 1
                percobaan[cmd] = n
                oom = "OutOfMemoryError" in log_path(cmd).read_text(errors="ignore")[-4000:]
                if n <= args.ulang_maks:
                    # Antrikan ke BELAKANG: beri waktu sel tetangga selesai dan
                    # melepas VRAM sebelum sel ini dicoba lagi.
                    queue.append(cmd)
                    print(f"[ULANG {n}/{args.ulang_maks}] {name} rc={p.returncode}"
                          f"{' (OOM)' if oom else ''} — diantrikan ulang di belakang",
                          flush=True)
                else:
                    failed.append(cmd)
                    print(f"[GAGAL GPU rc={p.returncode}] {name}  {dur:.1f} mnt "
                          f"{'(OOM) ' if oom else ''}-> {log_path(cmd)}", flush=True)

        for entry in list(cpu):
            p, cmd, t0 = entry
            if p.poll() is None:
                continue
            cpu.remove(entry)
            dur = (time.time() - t0) / 60
            if p.returncode == 0:
                scored.append(cmd)
                print(f"[OK  CPU skor] {score_path(cmd).name}  {dur:.1f} mnt",
                      flush=True)
            else:
                score_failed.append(cmd)
                print(f"[GAGAL CPU skor rc={p.returncode}] "
                      f"{output_path(cmd).name} -> {log_path(cmd, '_skor')}",
                      flush=True)

    print(f"\n=== selesai {(time.time()-t_start)/60:.1f} menit: "
          f"{len(done)} sel GPU OK, {len(failed)} gagal; "
          f"{len(scored)} skor CPU OK, {len(score_failed)} gagal ===")
    for c in failed + score_failed:
        print(f"  GAGAL: {output_path(c).name}")
    (RESULTS / "status_matriks.json").write_text(json.dumps({
        "arm": args.arm, "slots": args.slots, "cpu_slots": args.cpu_slots,
        "gpu_ok": [output_path(c).name for c in done],
        "gpu_gagal": [output_path(c).name for c in failed],
        "skor_ok": [score_path(c).name for c in scored],
        "skor_gagal": [output_path(c).name for c in score_failed],
        "dilewati": [output_path(c).name for c in skipped],
        "durasi_menit": round((time.time() - t_start) / 60, 1),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
