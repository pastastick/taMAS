"""Skor ulang korpus ekspresi pada jendela HOLDOUT — tanpa GPU, tanpa merusak angka seleksi.

Kenapa alat ini ada. Seluruh angka IC yang dilaporkan skripsi dihitung pada
segmen `test` konfigurasi (2021), yaitu jendela yang SAMA dengan yang dipakai
sistem untuk menilai dan menyaring ekspresi. Pembaca berhak bertanya apakah
ekspresi itu benar-benar membawa sinyal atau hanya kebetulan cocok pada periode
tersebut. `daily_pv.h5` memuat data sampai 2026-01, jadi pertanyaan itu bisa
dijawab langsung: nilai ulang korpus yang sama pada 2022--2025, periode yang tak
pernah dilihat sistem maupun penulisnya.

Beda dengan `rescore_all.py`, yang MENIMPA field `ic` di dalam
`frontend_*.json` supaya angka dokumen bisa diverifikasi ulang. Di sini
penimpaan itu justru berbahaya: ia akan mengganti angka seleksi 2021 yang
menopang seluruh Bab IV dengan angka holdout. Karena itu skrip ini bekerja pada
SALINAN, menulis ke berkasnya sendiri, dan memakai cache terpisah supaya kedua
jendela tak pernah bercampur di satu kunci ekspresi.

Keluaran `results/factor/holdout_<awal>_<akhir>.json`:
  - `per_ekspresi` : ic/icir/tstat/n_days + metrik backtest pada holdout,
                     berdampingan dengan ic seleksi yang tersimpan (`ic_seleksi`)
  - `per_tag`      : agregat per sel (jumlah hidup, rerata |IC|, berapa yang
                     tetap signifikan, berapa yang berbalik tanda)

Pemakaian:
    PYTHONPATH=backend python backend/eval/skor_holdout.py
    PYTHONPATH=backend python backend/eval/skor_holdout.py --window 2022-01-01,2025-12-26
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import bootstrap, ensure_out, CACHE, OUT_FACTOR
bootstrap()

OUT = ensure_out(OUT_FACTOR)

# Split test QuantaAlpha — periode setelah jendela seleksi 2021, belum pernah
# dipakai untuk menilai maupun menyaring ekspresi mana pun di penelitian ini.
HOLDOUT = ("2022-01-01", "2025-12-26")
BUDGET_S = 90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default=",".join(HOLDOUT),
                    help="AWAL,AKHIR jendela penilaian (format YYYY-MM-DD)")
    ap.add_argument("--tags", default="", help="daftar tag dipisah koma (kosong = semua)")
    ap.add_argument("--budget", type=int, default=BUDGET_S,
                    help="anggaran detik per ekspresi")
    ap.add_argument("--fresh", action="store_true",
                    help="abaikan cache holdout di disk")
    ap.add_argument("--quantile", type=float, default=0.1,
                    help="fraksi universe per sisi portofolio long-short. "
                         "0,1 = desil (bawaan, dipakai panel A-share dengan "
                         "~4.240 saham/hari). Pada universe sempit fraksi ini "
                         "HARUS dinaikkan: LQ45 hanya 37 saham/hari, sehingga "
                         "desil memberi k=3 saham per sisi dan derau portofolio "
                         "sd=sigma*sqrt(2/k) membengkak 1,53x dibanding kuintil "
                         "(k=7). Nilainya ikut masuk label keluaran + cache, "
                         "karena mengubahnya mengubah SEMUA field bt_*.")
    ap.add_argument("--cost-bps", type=float, default=0.0,
                    help="biaya transaksi satu arah (basis poin)")
    ap.add_argument("--simpan-deret", action="store_true",
                    help="tulis deret IC HARIAN per ekspresi ke "
                         "results/factor/icseries_<pasar>_<label>.parquet. "
                         "Dibutuhkan analisis yang bekerja pada dimensi WAKTU "
                         "(peluruhan alpha per tahun, klaster sinyal lewat "
                         "korelasi IC harian antar-ekspresi) — keduanya tidak "
                         "bisa dihitung dari rerata IC saja. Mati secara bawaan "
                         "karena berkasnya besar dan hanya perlu sekali.")
    ap.add_argument("--workers", type=int, default=0,
                    help="proses pekerja untuk skoring ekspresi (0 = auto: "
                         "min(4, ncpu); 1 = serial). Pekerja berbagi satu "
                         "salinan data pasar lewat fork copy-on-write.")
    args = ap.parse_args()

    if args.workers <= 0:
        args.workers = min(4, os.cpu_count() or 1)

    awal, akhir = [s.strip() for s in args.window.split(",")]

    # Penanda PASAR ikut masuk ke label. Tanpa ini, menilai korpus yang sama di
    # panel IDX akan menimpa berkas hasil A-share untuk jendela yang sama —
    # dan cache-nya (berkunci ekspresi) akan mencampur IC dari dua bursa
    # berbeda di satu kunci. Panel bawaan tetap memakai label lama supaya
    # berkas `holdout_2022-01-01_2025-12-26.json` yang sudah dikutip dokumen
    # tidak berpindah nama.
    from eval.ic import pasar_tag, pv_source
    from eval.backtest import TRADING_DAYS
    pasar, pv_file = pasar_tag(), pv_source()
    label = (f"{awal}_{akhir}" if pasar == "daily_pv"
             else f"{pasar}_{awal}_{akhir}")
    if abs(args.quantile - 0.1) > 1e-9:
        label += f"_q{args.quantile:g}"
    print(f"[skor] panel pasar : {pasar}", flush=True)
    print(f"[skor] kuantil     : {args.quantile:g} per sisi", flush=True)

    from eval.ic import Lab
    from factor.run_factor import score_expressions

    paths = sorted(OUT.glob("frontend_*.json"))
    if args.tags:
        ingin = {t.strip() for t in args.tags.split(",") if t.strip()}
        paths = [p for p in paths if p.stem[len("frontend_"):] in ingin]
    if not paths:
        print("tidak ada frontend_*.json yang cocok", file=sys.stderr)
        raise SystemExit(1)

    # Cache DIPISAH per jendela. Kalau ia berbagi berkas dengan `rescore_all`,
    # kunci cache-nya (ekspresi) akan menunjuk ke IC dari jendela yang salah.
    cache_path = ensure_out(CACHE) / f"holdout_cache_{label}.json"
    cache: dict = {}
    if cache_path.exists() and not args.fresh:
        cache = json.loads(cache_path.read_text())
        print(f"[holdout] cache dipulihkan: {len(cache)} ekspresi", flush=True)

    lab = Lab(mode="fast", window=(awal, akhir))
    print(f"[holdout] {len(paths)} tag · jendela {awal}..{akhir} · "
          f"anggaran {args.budget}s/ekspresi · {args.workers} pekerja", flush=True)

    # ── Pra-lewat: skor SELURUH korpus lintas-tag dalam satu kumpulan pekerja.
    # Tanpa ini, tiap tag memanggil score_expressions sendiri dan kumpulan
    # pekerjanya dibongkar-pasang berulang; digabung, satu Pool mengerjakan
    # semua ekspresi unik dan penjadwalannya jauh lebih rapat.
    gabungan = [r for p in paths
                for r in json.loads(p.read_text())["runs"]]
    n_awal = len(cache)
    tick = [time.time()]

    def _progres(i, total, e, entry):
        cache_path.write_text(json.dumps(cache, indent=1, default=str))
        now = time.time()
        if now - tick[0] >= 15 or i == total:
            tick[0] = now
            ic = entry.get("ic")
            print(f"[holdout]   {i:3d}/{total}  ic={ic if ic is None else f'{ic:+.4f}'}"
                  f"  {e[:60]}", flush=True)

    from factor.run_factor import score_expressions as _score
    deret_path = (OUT / f"icseries_{label}.parquet") if args.simpan_deret else None
    _score(gabungan, series_path=deret_path, budget_s=args.budget, cache=cache,
           lab=lab, workers=args.workers, on_progress=_progres,
           quantile=args.quantile, cost_bps=args.cost_bps)
    if deret_path is not None and deret_path.exists():
        print(f"[skor] deret IC harian → {deret_path.name}", flush=True)
    print(f"[holdout] korpus: {len(cache) - n_awal} ekspresi baru diskor, "
          f"{len(cache)} total di cache", flush=True)

    per_tag: list[dict] = []
    per_ekspresi: list[dict] = []
    terlihat: set[tuple[str, str]] = set()
    t0 = time.time()

    for i, path in enumerate(paths, 1):
        tag = path.stem[len("frontend_"):]
        doc = json.loads(path.read_text())

        # IC seleksi (2021) yang tersimpan, dipotret SEBELUM salinan diskor —
        # inilah pembanding yang membuat holdout punya arti.
        seleksi = {}
        for r in doc["runs"]:
            for f in (r.get("factors") or []):
                e = f.get("expression", "")
                if e:
                    seleksi[e] = {k: f.get(k) for k in
                                  ("ic", "icir", "tstat", "n_days", "n_unique")}

        runs = copy.deepcopy(doc["runs"])
        ts = time.time()
        # series_path=None: deret IC harian holdout tidak ditulis. Analisis
        # klaster sinyal memakai jendela seleksi, dan menulis deret 4 tahun
        # untuk 151 ekspresi hanya membebani disk tanpa dipakai.
        score_expressions(runs, series_path=None, budget_s=args.budget,
                          cache=cache, lab=lab, quantile=args.quantile,
                          cost_bps=args.cost_bps)
        dt = time.time() - ts

        allf = [f for r in runs for f in (r.get("factors") or [])]
        hidup = [f for f in allf
                 if f.get("ic") is not None and (f.get("n_unique") or 0) > 2]
        n_sig = sum(1 for f in hidup
                    if f.get("tstat") is not None and abs(f["tstat"]) >= 1.96)

        # Berbalik tanda = ekspresi yang IC-nya positif di seleksi lalu negatif
        # di holdout (atau sebaliknya). Ini ukuran ketahanan yang lebih tegas
        # daripada rerata |IC|, yang bisa tetap tinggi meski arahnya kacau.
        n_balik = 0
        n_pasangan = 0
        for f in hidup:
            e = f["expression"]
            lama = (seleksi.get(e) or {}).get("ic")
            if lama is None:
                continue
            n_pasangan += 1
            if float(lama) * float(f["ic"]) < 0:
                n_balik += 1

        for f in allf:
            e = f.get("expression", "")
            if not e or (tag, e) in terlihat:
                continue
            terlihat.add((tag, e))
            baris = {"tag": tag, "expression": e,
                     "ic_seleksi": (seleksi.get(e) or {}).get("ic"),
                     "tstat_seleksi": (seleksi.get(e) or {}).get("tstat"),
                     "lolos_gate": f.get("passed_gate")}
            baris.update({k: f.get(k) for k in
                          ("ic", "icir", "tstat", "n_days", "coverage",
                           "n_unique", "eval_error", "sem_ok")})
            baris.update({k: v for k, v in f.items() if k.startswith("bt_")})
            per_ekspresi.append(baris)

        per_tag.append({
            "tag": tag,
            "ekspresi": len(allf),
            "hidup": len(hidup),
            "signifikan": n_sig,
            "berpasangan": n_pasangan,
            "berbalik_tanda": n_balik,
            "mean_abs_ic": (sum(abs(f["ic"]) for f in hidup) / len(hidup)
                            if hidup else None),
        })
        print(f"[{i:2d}/{len(paths)}] {tag:28s} ekspr={len(allf):3d} "
              f"hidup={len(hidup):3d} sig={n_sig:3d} balik={n_balik:2d}  "
              f"({dt:5.1f}s)", flush=True)
        cache_path.write_text(json.dumps(cache, indent=1, default=str))

    rep = OUT / f"holdout_{label}.json"
    rep.write_text(json.dumps({
        "window": [awal, akhir],
        "pasar": pasar,
        "panel": str(pv_file),
        "quantile": args.quantile,
        "cost_bps": args.cost_bps,
        "trading_days": TRADING_DAYS,
        "budget_s": args.budget,
        "n_ekspresi_unik": len(cache),
        "per_tag": per_tag,
        "per_ekspresi": per_ekspresi,
    }, indent=2, default=str))
    print(f"\n[holdout] selesai dalam {time.time() - t0:.0f}s · "
          f"{len(cache)} ekspresi unik dievaluasi", flush=True)
    print(f"laporan → {rep}", flush=True)


if __name__ == "__main__":
    main()
