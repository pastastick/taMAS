"""Skor ulang SELURUH korpus front-end di CPU — dan bandingkan dengan angka lama.

Kenapa alat ini ada. Deret IC harian (`results/icseries_*.parquet`) adalah satu-
satunya masukan untuk metrik **klaster sinyal** (A3) — sumbu yang menopang
sebagian besar keputusan arsitektur di RENCANA_PERBAIKAN, karena ia bervarians
rendah pada n=6 sementara |IC| tidak. Berkas itu `*.parquet` dan karenanya
gitignored, jadi setiap kali repo dipindah (mis. dari runpod ke mesin lokal)
seluruh sumbu A3 berhenti bisa direproduksi meskipun `frontend_*.json`
ter-commit lengkap. Skrip ini memulihkannya tanpa GPU.

Dua hal dikerjakan sekaligus, dan yang kedua justru lebih penting:

1. **Regenerasi** `icseries_<tag>.parquet` untuk semua tag.
2. **Verifikasi**: IC yang baru dihitung dibandingkan dengan IC yang sudah
   tersimpan di `frontend_*.json` (hasil skoring di mesin GPU). Kalau replika
   CPU di mesin ini tidak mereproduksi angka yang dilaporkan dokumen, itu
   HARUS ketahuan di sini, bukan di sidang.

Efisiensi: satu proses, satu kali muat data pasar, dan cache ekspresi GLOBAL
lintas-tag (805 ekspresi total → 631 unik). Menjalankan
`frontend_probe.py --score-only` 31 kali mengulang keduanya 31 kali.

Pemakaian:
    PYTHONPATH=backend python backend/eval/rescore_all.py
    PYTHONPATH=backend python backend/eval/rescore_all.py --tags b14_summary,g4_text
    PYTHONPATH=backend python backend/eval/rescore_all.py --dry-run   # verifikasi saja
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import bootstrap, ensure_out, OUT_FACTOR, OUT_PROBE, CACHE
bootstrap()

# Korpus JSON front-end ada di results/factor (baru) DAN results/probe (artefak
# lama yang ikut dipindah dari lab/out) — keduanya dipindai.
OUT = ensure_out(OUT_FACTOR)
CORPUS_DIRS = (OUT_FACTOR, OUT_PROBE)

# Ekspresi yang mahal (REGBETA/REGRESI rolling) diberi anggaran waktu supaya
# satu ekspresi tak menyandera seluruh korpus — sama seperti frontend_probe.
BUDGET_S = 90

# Cache lintas-proses. Skoring korpus penuh memakan jam-jaman dan sebagian
# ekspresi LLM meledakkan memori sementara (rolling bersarang atas ±1 jt baris),
# jadi proses ini BISA dibunuh OOM di mesin kecil. Tanpa cache di disk, kematian
# di tag ke-20 berarti mengulang 19 tag dari nol. Dengan cache, jalankan lagi
# perintah yang sama dan ia melanjutkan.
CACHE_JSON = ensure_out(CACHE) / "rescore_cache.json"
CACHE_PARQUET = CACHE / "rescore_series.parquet"


def load_cache() -> tuple[dict, dict]:
    cache: dict = {}
    series: dict = {}
    if CACHE_JSON.exists():
        try:
            cache = json.loads(CACHE_JSON.read_text())
        except Exception:  # noqa: BLE001
            cache = {}
    if CACHE_PARQUET.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(CACHE_PARQUET)
            series = {c: df[c].dropna() for c in df.columns}
        except Exception:  # noqa: BLE001
            series = {}
    return cache, series


def save_cache(cache: dict, series: dict) -> None:
    import pandas as pd

    tmp = CACHE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, default=str))
    tmp.replace(CACHE_JSON)
    if series:
        tmp2 = CACHE_PARQUET.with_suffix(".tmp")
        pd.DataFrame(series).to_parquet(tmp2)
        tmp2.replace(CACHE_PARQUET)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="", help="daftar tag dipisah koma (kosong = semua)")
    ap.add_argument("--budget", type=int, default=BUDGET_S,
                    help="anggaran detik per ekspresi")
    ap.add_argument("--dry-run", action="store_true",
                    help="hitung & bandingkan, tetapi jangan tulis JSON/parquet")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="ambang |ΔIC| yang dianggap tak sama")
    ap.add_argument("--fresh", action="store_true",
                    help="abaikan cache di disk dan hitung ulang semuanya")
    ap.add_argument("--checkpoint-detik", type=int, default=180,
                    help="setiap berapa detik cache disk disimpan di TENGAH "
                         "skoring paralel. Sebelum ini cache hanya ditulis "
                         "SETELAH seluruh pra-lewat selesai, sehingga proses "
                         "yang dibunuh di jam ke-3 (mis. pod GPU habis sewa) "
                         "kehilangan SELURUH progresnya — persis yang terjadi "
                         "pada `text` dan `kv_gumbel` 2026-08-27. 0 = matikan.")
    ap.add_argument("--workers", type=int, default=1,
                    help="proses pekerja untuk skoring ekspresi (1 = serial). "
                         "Pekerja berbagi satu salinan data pasar lewat fork; "
                         "aman karena jalur skoring hanya membaca. Naikkan ke "
                         "min(4, ncpu) untuk korpus besar.")
    args = ap.parse_args()
    import os as _os
    if args.workers <= 0:
        args.workers = min(4, _os.cpu_count() or 1)

    from eval.ic import Lab
    from factor.run_factor import score_expressions

    def _find(tag: str) -> Path:
        """Cari frontend_<tag>.json di results/factor lalu results/probe."""
        for d in CORPUS_DIRS:
            p = d / f"frontend_{tag}.json"
            if p.exists():
                return p
        return OUT / f"frontend_{tag}.json"   # biar pesan errornya menunjuk ke tempat baru

    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        paths = [_find(t) for t in tags]
    else:
        # Urutan bukan alfabetis: tag yang menopang angka HEADLINE dikerjakan
        # lebih dulu, supaya proses yang terpotong di tengah tetap memulihkan
        # sumbu yang paling banyak dikutip dokumen.
        prio = ["b14_summary", "b14_text", "a8_kv_innovate_guided", "a8_kv_full",
                "a8_kv_innovate", "a8_kv_nodesign", "a8_kv_direct",
                "a8_kv_innovate_fid", "a8_kv_full_guided",
                "g4_kv_ls10", "g4_text", "g4_kv_and_text_ls10",
                "g2_kv_ls5", "g2_kv_ls10", "g3_kv_ls10_gumbelT0.7",
                "g3_kv_ls10_raw", "b5_pre", "b5_post", "a10"]
        rank = {t: i for i, t in enumerate(prio)}
        seen: set[str] = set()
        paths = []
        for d in CORPUS_DIRS:
            for p in d.glob("frontend_*.json"):
                if p.stem not in seen:      # results/factor menang atas results/probe
                    seen.add(p.stem)
                    paths.append(p)
        paths.sort(key=lambda p: (rank.get(p.stem[len("frontend_"):], 999), p.stem))

    lab = Lab(mode="fast")           # jendela seleksi 2021, sama dengan dokumen
    cache, series_cache = ({}, {}) if args.fresh else load_cache()

    print(f"[rescore] {len(paths)} tag · jendela {lab.oos_start.date()}"
          f"..{lab.oos_end.date()} · anggaran {args.budget}s/ekspresi", flush=True)
    if cache:
        print(f"[rescore] cache dipulihkan: {len(cache)} ekspresi, "
              f"{len(series_cache)} deret IC", flush=True)
    print(flush=True)

    diffs: list[tuple] = []          # (tag, expr, ic_lama, ic_baru)
    missing_before: list[tuple] = []  # faktor yang dulu tak punya IC tersimpan
    per_tag: list[dict] = []
    t0 = time.time()

    # ── Pra-lewat paralel atas SELURUH korpus ──────────────────────────────
    # Deret IC harian ikut terkumpul di `series_cache` (dioper by-ref), jadi
    # loop per-tag di bawah tinggal menulis parquet-nya tanpa evaluasi ulang.
    # Berhenti anggun. Skoring korpus berjam-jam dan biasanya dijalankan
    # menumpang umur pod GPU, jadi ia HAMPIR SELALU dibunuh dari luar
    # (SIGTERM saat pod dimatikan, SIGINT saat Ctrl-C). Tanpa penanganan ini
    # sinyal itu membunuh proses di tengah `imap_unordered` dan seluruh
    # ekspresi yang sudah diskor sejak checkpoint terakhir hilang. Dengan ini,
    # sinyal cuma menyalakan bendera: pekerjaan yang sedang berjalan
    # diselesaikan, cache disimpan, lalu tag yang ekspresinya SUDAH lengkap
    # tetap ditulis ke JSON-nya.
    dihentikan = {"ya": False}

    def _tangkap(signum, _frame):
        if not dihentikan["ya"]:
            dihentikan["ya"] = True
            print(f"\n[rescore] sinyal {signum} diterima — menghentikan skoring "
                  f"dengan rapi, menyimpan cache, lalu menulis tag yang sudah "
                  f"lengkap. Jalankan perintah yang sama untuk melanjutkan.",
                  flush=True)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _tangkap)
        except (ValueError, OSError):   # bukan thread utama / platform aneh
            pass

    class _Dihentikan(Exception):
        """Dipakai untuk keluar dari loop konsumen Pool, bukan error sungguhan."""

    if args.workers > 1:
        gabungan = [r for p in paths if p.exists()
                    for r in json.loads(p.read_text())["runs"]]
        n0 = len(cache)
        tk = [time.time()]
        tc = [time.time()]

        def _prog(i, total, e, entry):
            if time.time() - tk[0] >= 20 or i == total:
                tk[0] = time.time()
                sisa = ""
                if i > 1:
                    laju = (time.time() - t0) / i
                    sisa = f"  ~{(total - i) * laju / 60:.0f} mnt lagi"
                print(f"[rescore/paralel] {i:3d}/{total}  {e[:56]}{sisa}", flush=True)
            # Checkpoint berkala: inilah yang membuat proses yang terbunuh
            # kehilangan paling banyak `--checkpoint-detik` detik kerja,
            # bukan berjam-jam.
            if (args.checkpoint_detik and not args.dry_run
                    and time.time() - tc[0] >= args.checkpoint_detik):
                tc[0] = time.time()
                save_cache(cache, series_cache)
                print(f"[rescore/checkpoint] {len(cache)} ekspresi tersimpan "
                      f"→ {CACHE_JSON.name}", flush=True)
            if dihentikan["ya"]:
                raise _Dihentikan

        try:
            score_expressions(gabungan, series_path=None, budget_s=args.budget,
                              cache=cache, lab=lab, series_cache=series_cache,
                              workers=args.workers, on_progress=_prog)
        except _Dihentikan:
            pass
        if not args.dry_run:
            save_cache(cache, series_cache)
        print(f"[rescore/paralel] {len(cache) - n0} ekspresi baru diskor "
              f"({time.time() - t0:.0f}s); lanjut menulis per-tag\n", flush=True)

    for i, path in enumerate(paths, 1):
        tag = path.stem[len("frontend_"):]
        if not path.exists():
            print(f"[{i:2d}/{len(paths)}] {tag}: TIDAK ADA — dilewati", flush=True)
            continue
        doc = json.loads(path.read_text())
        runs = doc["runs"]

        # Kalau pra-lewat dihentikan, tag yang ekspresinya belum lengkap di
        # cache DILEWATI — bukan diskor serial di sini. Menskornya serial
        # justru mengabaikan alasan kita berhenti (waktu habis) dan bisa
        # menggantung berjam-jam pada satu tag.
        if dihentikan["ya"]:
            belum = [f.get("expression") for r in runs
                     for f in (r.get("factors") or [])
                     if f.get("expression") and f["expression"] not in cache]
            if belum:
                print(f"[{i:2d}/{len(paths)}] {tag:28s} DILEWATI — "
                      f"{len(belum)} ekspresi belum diskor", flush=True)
                continue

        # potret angka lama SEBELUM ditimpa
        before = {}
        for r in runs:
            for f in r.get("factors", []) or []:
                e = f.get("expression", "")
                if e:
                    before[e] = f.get("ic", "__absent__")

        n_new = sum(1 for e in before if e not in cache)
        ts = time.time()
        series_path = None if args.dry_run else OUT / f"icseries_{tag}.parquet"
        score_expressions(runs, series_path=series_path, budget_s=args.budget,
                          cache=cache, lab=lab, series_cache=series_cache)
        dt = time.time() - ts

        n_diff = 0
        for r in runs:
            for f in r.get("factors", []) or []:
                e = f.get("expression", "")
                if not e:
                    continue
                old, new = before.get(e), f.get("ic")
                if old == "__absent__":
                    missing_before.append((tag, e, new))
                elif old is None and new is None:
                    pass
                elif old is None or new is None:
                    diffs.append((tag, e, old, new)); n_diff += 1
                elif abs(float(old) - float(new)) > args.tol:
                    diffs.append((tag, e, old, new)); n_diff += 1

        if not args.dry_run:
            path.write_text(json.dumps(doc, indent=2, default=str))
            save_cache(cache, series_cache)

        allf = [f for r in runs for f in (r.get("factors") or [])]
        ok = [f for f in allf if f.get("ic") is not None]
        alive = [f for f in ok if (f.get("n_unique") or 0) > 2]
        n_series = sum(1 for e in before if e in series_cache)
        per_tag.append({
            "tag": tag, "ekspresi": len(allf), "ber_ic": len(ok), "hidup": len(alive),
            "deret": n_series, "beda": n_diff,
            "mean_abs_ic": (sum(abs(f["ic"]) for f in ok) / len(ok)) if ok else None,
        })
        print(f"[{i:2d}/{len(paths)}] {tag:28s} ekspr={len(allf):3d} "
              f"baru={n_new:3d} ber-IC={len(ok):3d} hidup={len(alive):3d} "
              f"deret={n_series:3d} beda={n_diff:2d}  ({dt:5.1f}s)", flush=True)

    print(f"\n[rescore] selesai dalam {time.time() - t0:.0f}s · "
          f"{len(cache)} ekspresi unik dievaluasi · "
          f"{len(series_cache)} punya deret IC", flush=True)

    # ── laporan verifikasi ────────────────────────────────────────────────
    print("\n=== VERIFIKASI terhadap angka yang tersimpan (hasil mesin GPU) ===")
    if missing_before:
        print(f"  {len(missing_before)} faktor belum pernah punya field `ic` "
              f"(baru diskor sekarang)")
    if not diffs:
        print("  ✅ TIDAK ADA selisih di atas toleransi — replika CPU di mesin ini "
              "mereproduksi seluruh IC yang dilaporkan dokumen.")
    else:
        print(f"  ⚠️  {len(diffs)} selisih di atas {args.tol:g}:")
        for tag, e, old, new in diffs[:40]:
            o = "None" if old is None else f"{float(old):+.6f}"
            n = "None" if new is None else f"{float(new):+.6f}"
            print(f"    [{tag}] {o} → {n}   {e[:90]}")
        if len(diffs) > 40:
            print(f"    … dan {len(diffs) - 40} lainnya")

    rep = OUT / "rescore_report.json"
    if not args.dry_run:
        rep.write_text(json.dumps({
            "window": [str(lab.oos_start.date()), str(lab.oos_end.date())],
            "budget_s": args.budget,
            "n_unique_expr": len(cache),
            "n_with_series": len(series_cache),
            "per_tag": per_tag,
            "diffs": [{"tag": t, "expr": e, "ic_lama": o, "ic_baru": n}
                      for t, e, o, n in diffs],
            "missing_before": [{"tag": t, "expr": e, "ic_baru": n}
                               for t, e, n in missing_before],
        }, indent=2, default=str))
        print(f"\nlaporan → {rep}")


if __name__ == "__main__":
    main()
