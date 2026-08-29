#!/usr/bin/env python3
"""Turunkan daftar perintah run dari `configs/matriks.yaml`.

Kenapa generator, bukan runner yang membaca YAML langsung: satu proses per sel
adalah keputusan sadar (2–3 sel muat paralel di satu A40 — docs/HASIL_TAHAP0.md
§8.7), dan sebuah runner yang menyapu matriks di dalam satu proses justru
membuang setengah kartu. Generator memberi kontrol penjadwalan ke shell tanpa
kehilangan satu sumber kebenaran untuk isi matriksnya.

Ia juga menegakkan satu aturan yang gampang terlewat kalau perintah ditulis
tangan: pada `comm_mode=text` TIDAK ADA langkah laten, jadi keempat nilai
Sumbu A menghasilkan sel yang identik. Menjalankannya empat kali membakar GPU
untuk empat salinan angka yang sama — dan lebih buruk, empat salinan itu akan
terlihat seperti empat pengamatan independen di tabel. Di sini `text`
dijalankan SEKALI.

    python scripts/gen_perintah.py --arm bench             # perintah lengan benchmark
    python scripts/gen_perintah.py --arm factor            # perintah lengan faktor
    python scripts/gen_perintah.py --arm bench --parallel 3 > jalankan.sh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from paths import CONFIGS  # noqa: E402

import yaml  # noqa: E402

PY = "PYTHONPATH=backend python"


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf8"))


def _latent_modes(cfg: dict) -> list[str]:
    """Himpunan M = {raw, soft, sample, gumbel, moi}, satu sumber untuk kedua lengan.

    `kontrol_latent_mode` masih dibaca supaya konfigurasi lama (yang menaruh
    `soft` terpisah sebagai kontrol) tetap menghasilkan daftar sel yang sama.
    Sejak 2026-08-27 kunci itu tak lagi dipakai `matriks.yaml`: `soft` adalah
    anggota penuh, bukan kontrol.
    """
    modes = list(cfg["latent_modes"])
    kontrol = cfg.get("kontrol_latent_mode")
    if kontrol and kontrol not in modes:
        modes.append(kontrol)
    return modes


def bench_commands(cfg: dict) -> list[str]:
    b = cfg["bench"]
    model, ls = cfg["model"], cfg["latent_steps"]
    # `--limit` sengaja TIDAK dimasukkan ke `base`: medium lampiran memakai
    # limit sendiri (lihat di bawah), dan menaruh dua `--limit` di satu
    # perintah lalu mengandalkan argparse mengambil yang terakhir adalah
    # jebakan yang mudah terlewat saat perintahnya dibaca manusia.
    base = (f"{PY} backend/bench/run_bench.py --model {model} "
            f"--latent-steps {ls} --latent-temp {cfg['latent_temp']} "
            f"--sample-seed {b['sample_seed']} "
            f"--temperature {b['temperature']} --top-p {b['top_p']} "
            f"--max-new-tokens {b['max_new_tokens']}")
    lim = f"--limit {b['limit']}"

    # Medium yang dijalankan HANYA sebagai lampiran, bukan lengan uji: sedikit
    # soal, semata untuk memperlihatkan seperti apa keluaran LLM ketika
    # masukannya berupa laten. Diberi limit DAN tag sendiri.
    #
    # Tag terpisah itu pengaman, bukan kosmetik. `bench/compare.py`
    # mengelompokkan sel per `(task, limit, sample_seed, model)` dan
    # mengeluarkan sel yang sidik jarinya beda — jadi sel lampiran memang sudah
    # otomatis terpisah dari analisis. Tapi nama berkas yang berbeda
    # (`..._lampiran.json`, bukan `..._s0.json`) membuat pemisahan itu terlihat
    # oleh MANUSIA yang membaca direktori hasil, bukan hanya oleh skrip.
    #
    # CATATAN PENTING soal apa yang HILANG: dengan `kv_and_text` jadi lampiran,
    # Sumbu B keluar dari analisis kuantitatif. Perbandingan medium yang tersisa
    # adalah `kv` vs `text` vs `baseline` — masih menjawab pertanyaan inti
    # (laten vs teks), tapi tak lagi bisa memisahkan "KV membantu" dari
    # "menghapus teks merugikan". Itu harus dinyatakan di skripsi, bukan
    # didiamkan.
    lampiran_mode = b.get("comm_mode_lampiran")
    lim_lampiran = f"--limit {b.get('limit_lampiran', 5)}"

    cmds = []
    for task in b["tasks"].values():
        for seed in b["seeds"]:
            tag = f"s{seed}"
            # Lantai: agen tunggal. Tak ada rantai → tak ada handoff → satu sel.
            if cfg.get("baseline"):
                cmds.append(f"{base} {lim} --task {task} --seed {seed} --baseline "
                            f"--latent-mode raw --comm-mode kv --tag {tag}")
            # Baseline teks: SATU sel, bukan empat (lihat docstring).
            cmds.append(f"{base} {lim} --task {task} --seed {seed} "
                        f"--comm-mode {cfg['comm_mode_tanpa_laten']} "
                        f"--latent-mode raw --tag {tag}")
            # Matriks penuh Sumbu A × Sumbu B untuk medium ber-KV.
            modes = _latent_modes(cfg)
            for comm in cfg["comm_modes"]:
                lampiran = (comm == lampiran_mode)
                for mode in modes:
                    extra = (f" --latent-beta {cfg['latent_beta']}"
                             if mode == "moi" else "")
                    cmds.append(f"{base} {lim_lampiran if lampiran else lim} "
                                f"--task {task} --seed {seed} "
                                f"--comm-mode {comm} --latent-mode {mode}"
                                f"{extra} "
                                f"--tag {'lampiran' if lampiran else tag}")
    return cmds


def factor_commands(cfg: dict) -> list[str]:
    f = cfg["factor"]
    seeds = ",".join(str(s) for s in f["seeds"])
    dirs = ",".join(f["directions"])
    base = (f"{PY} backend/factor/run_factor.py --model {cfg['model']} "
            f"--latent-steps {cfg['latent_steps']} "
            f"--latent-temp {cfg['latent_temp']} "
            f"--seeds {seeds} --directions {dirs} --chain {f['chain']} "
            f"--max-repair {f['max_repair']}")
    # `guided_decoding` di matriks.yaml bersifat DOKUMENTASI: run_factor.py
    # tidak mengeksposnya sebagai flag — nilainya ditentukan `json_schema:` di
    # backend/prompts/factor.yaml (B11/B16). Dicatat di sini supaya setelan yang
    # berlaku saat run tercatat bersama sel-selnya, bukan supaya diteruskan.
    cmds = [f"{base} --comm-mode {cfg['comm_mode_tanpa_laten']} "
            f"--latent-mode raw --tag text"]
    for comm in cfg["comm_modes"]:
        for mode in _latent_modes(cfg):
            cmds.append(f"{base} --comm-mode {comm} --latent-mode {mode} "
                        f"--tag {comm}_{mode}")
    return cmds


def interpolasi_commands(cfg: dict) -> list[str]:
    """Sumbu C: kurva dose-response lewat mode `mix`.

    Hanya nilai alpha di `alphas_perlu_run` yang dikeluarkan. Titik ujung
    (alpha 0 dan 1) sengaja dilewati karena menghasilkan vektor yang identik
    dengan sel `raw` dan `soft` yang sudah ada — menjalankannya ulang hanya
    membakar GPU untuk angka yang sudah dimiliki, dan hasilnya pun tak akan
    persis sama karena sampling ulang, sehingga kurvanya justru jadi lebih
    sulit dibaca, bukan lebih mudah.
    """
    it = cfg.get("interpolasi")
    if not it:
        return []
    b = cfg["bench"]
    base_bench = (f"{PY} backend/bench/run_bench.py --model {cfg['model']} "
                  f"--latent-steps {cfg['latent_steps']} "
                  f"--latent-temp {cfg['latent_temp']} "
                  f"--sample-seed {b['sample_seed']} "
                  f"--temperature {b['temperature']} --top-p {b['top_p']} "
                  f"--max-new-tokens {b['max_new_tokens']} --limit {b['limit']}")
    f = cfg["factor"]
    base_faktor = (f"{PY} backend/factor/run_factor.py --model {cfg['model']} "
                   f"--latent-steps {cfg['latent_steps']} "
                   f"--latent-temp {cfg['latent_temp']} "
                   f"--seeds {','.join(str(x) for x in f['seeds'])} "
                   f"--directions {','.join(f['directions'])} "
                   f"--chain {f['chain']} --max-repair {f['max_repair']}")

    cmds = []
    for a in it["alphas_perlu_run"]:
        tanda = f"a{a}".replace(".", "")          # 0.25 -> a025
        for task in it["bench_tasks"]:
            for seed in b["seeds"]:
                cmds.append(f"{base_bench} --task {task} --seed {seed} "
                            f"--comm-mode {it['comm_mode']} --latent-mode mix "
                            f"--latent-alpha {a} --tag s{seed}_{tanda}")
        if it.get("jalankan_lengan_faktor"):
            cmds.append(f"{base_faktor} --comm-mode {it['comm_mode']} "
                        f"--latent-mode mix --latent-alpha {a} "
                        f"--tag {it['comm_mode']}_mix_{tanda}")
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True,
                    choices=["bench", "factor", "interpolasi", "all"])
    ap.add_argument("--config", default=str(CONFIGS / "matriks.yaml"))
    ap.add_argument("--parallel", type=int, default=0,
                    help="bila >0, keluarkan skrip shell yang menjalankan N sel "
                         "bersamaan dengan jeda 30 detik antar-start")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    cmds: list[str] = []
    if args.arm in ("bench", "all"):
        cmds += bench_commands(cfg)
    if args.arm in ("factor", "all"):
        cmds += factor_commands(cfg)
    if args.arm in ("interpolasi", "all"):
        cmds += interpolasi_commands(cfg)

    if args.parallel <= 0:
        print(f"# {len(cmds)} sel dari {args.config}")
        for c in cmds:
            print(c)
        return

    print("#!/usr/bin/env bash")
    print("set -u")
    print(f"# {len(cmds)} sel, {args.parallel} proses paralel.")
    print("# Jeda 30 dtk antar-start: fase muat model dari network storage")
    print("# rebutan I/O kalau serentak (docs/HASIL_TAHAP0.md §8.7).")
    for i in range(0, len(cmds), args.parallel):
        batch = cmds[i:i + args.parallel]
        print(f"\n# ── batch {i // args.parallel + 1} ──")
        for j, c in enumerate(batch):
            print(f"{c} &")
            if j < len(batch) - 1:
                print("sleep 30")
        print("wait")


if __name__ == "__main__":
    main()
