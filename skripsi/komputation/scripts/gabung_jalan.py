#!/usr/bin/env python3
"""Gabungkan beberapa `frontend_<tag>.json` sebagian menjadi satu sel utuh.

KENAPA SKRIP INI ADA. `run_factor.py` menulis `frontend_<tag>.json` sekali di
akhir dan **tidak punya resume/append**: menjalankannya lagi dengan tag yang
sama akan MENIMPA berkasnya, bukan menambah jalan. Jadi sel yang terlanjur
berhenti di tengah (mis. tiga sel `kv_and_text` yang macet di 8 dari 20 jalan
karena orkestrator konsolidasi 2026-08-27 di-hardcode `target 8`) hanya punya
dua jalan keluar:

  (a) jalankan ulang SELURUH sel dengan `--seeds 0,1,2,3,4` — tapi itu
      membangkitkan ulang seed 0-1 yang sudah ada. Dengan `--temperature 0.8`
      dan batching vLLM, hasilnya belum tentu reproduksi bit-per-bit, sehingga
      angka yang sudah terdokumentasi di `docs/HASIL_TAHAP5.md` bisa bergeser
      tanpa alasan ilmiah apa pun. Delapan jalan GPU juga terbuang.

  (b) jalankan HANYA seed yang kurang ke tag sementara, lalu gabungkan. Itu
      yang dikerjakan skrip ini.

KAPAN PENGGABUNGAN ITU SAH. Hanya kalau seluruh pecahan dibangkitkan dengan
prosedur yang sama. Karena itu skrip ini MENOLAK menggabung kalau ada satu saja
argumen yang menentukan perilaku berbeda antar-pecahan (lihat `ARGS_WAJIB_SAMA`)
— pelajaran dari `results/arsip_faktor_6jalan_2026-08-10/README.md`, yaitu run
pra-`prefill` yang tidak boleh disatukan dengan run pasca-`prefill` walau
tag-nya sama. Cek versi kode ada di luar jangkauan skrip ini; pastikan sendiri
pecahannya dijalankan dari commit yang sama.

Contoh: menaikkan `kv_and_text_gumbel` dari 8 jalan (seed 0,1) ke 20 jalan.

    # 1. jalankan HANYA seed yang kurang ke tag sementara
    PYTHONPATH=backend python backend/factor/run_factor.py \
        --model Qwen/Qwen3-8B --comm-mode kv_and_text --latent-mode gumbel \
        --latent-steps 10 --latent-temp 0.7 \
        --seeds 2,3,4 --directions d0,d1,opp_mom,opp_rev \
        --chain proposal,innovate,construct --max-repair 3 \
        --tag kv_and_text_gumbel_s234 --skip-score

    # 2. gabungkan (mengintip dulu, tak menulis apa pun)
    python scripts/gabung_jalan.py --keluaran kv_and_text_gumbel \
        --dari kv_and_text_gumbel kv_and_text_gumbel_s234 --dry-run

    # 3. tulis (salinan cadangan berkas lama dibuat otomatis)
    python scripts/gabung_jalan.py --keluaran kv_and_text_gumbel \
        --dari kv_and_text_gumbel kv_and_text_gumbel_s234

    # 4. skoring IC WAJIB diulang: icseries_<tag>.parquet lama hanya memuat
    #    ekspresi dari 8 jalan dan TIDAK ikut tergabung.
    PYTHONPATH=backend python backend/factor/run_factor.py \
        --score-only --tag kv_and_text_gumbel --budget 900
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import OUT_FACTOR  # noqa: E402

# Argumen yang HARUS identik di semua pecahan. Kalau salah satu berbeda,
# pecahannya dibangkitkan dengan perlakuan berbeda dan menyatukannya dalam satu
# tabel tidak sah — persis alasan run 6-jalan diarsipkan alih-alih ditimpa.
ARGS_WAJIB_SAMA = (
    "model", "comm_mode", "latent_steps", "latent_mode", "latent_temp",
    "latent_alpha", "early_stop_cos", "no_realign", "directions",
    "temperature", "max_new_tokens", "max_repair", "chain", "free_form",
    "prompts",
)

# Argumen yang boleh berbeda dan memang diharapkan berbeda.
ARGS_BOLEH_BEDA = ("seeds", "tag", "score_only", "skip_score", "budget", "holdout")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _muat(tag: str) -> tuple[Path, dict]:
    p = OUT_FACTOR / f"frontend_{tag}.json"
    if not p.exists():
        sys.exit(f"BERHENTI: tidak ada {p}")
    doc = json.loads(p.read_text())
    if "runs" not in doc or "args" not in doc:
        sys.exit(f"BERHENTI: {p} bukan keluaran run_factor.py "
                 f"(kunci: {sorted(doc)})")
    return p, doc


def _periksa_kecocokan(pecahan: list[tuple[str, Path, dict]]) -> list[str]:
    """Kembalikan daftar ketidakcocokan; kosong berarti aman digabung."""
    dasar_tag, _, dasar = pecahan[0]
    masalah = []
    for tag, _, doc in pecahan[1:]:
        for k in ARGS_WAJIB_SAMA:
            a, b = dasar["args"].get(k), doc["args"].get(k)
            if a != b:
                masalah.append(f"`{k}`: {dasar_tag}={a!r} vs {tag}={b!r}")
    return masalah


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dari", nargs="+", required=True, metavar="TAG",
                    help="tag pecahan yang digabung, urut mana saja. Berkasnya "
                         "dicari di results/factor/frontend_<TAG>.json")
    ap.add_argument("--keluaran", required=True, metavar="TAG",
                    help="tag sel hasil gabungan. Boleh sama dengan salah satu "
                         "--dari (berkas lamanya dicadangkan lebih dulu)")
    ap.add_argument("--dry-run", action="store_true",
                    help="periksa dan laporkan saja, jangan menulis apa pun")
    ap.add_argument("--paksa", action="store_true",
                    help="tetap gabung walau argumen antar-pecahan berbeda. "
                         "JANGAN dipakai untuk angka yang masuk naskah — beda "
                         "argumen berarti beda prosedur pembangkitan")
    ap.add_argument("--izinkan-duplikat", action="store_true",
                    help="kalau pasangan (arah, seed) muncul di lebih dari satu "
                         "pecahan, ambil yang dari pecahan TERAKHIR alih-alih "
                         "berhenti. Default berhenti, karena duplikat hampir "
                         "selalu berarti salah satu pecahan salah --seeds")
    args = ap.parse_args()

    if len(args.dari) < 2:
        sys.exit("BERHENTI: --dari butuh minimal dua tag")

    pecahan = [(t, *_muat(t)) for t in args.dari]

    # ── (1) kecocokan prosedur ──────────────────────────────────────────────
    masalah = _periksa_kecocokan(pecahan)
    if masalah:
        print("ARGUMEN TIDAK COCOK antar-pecahan:")
        for m in masalah:
            print(f"  - {m}")
        if not args.paksa:
            sys.exit("BERHENTI: pecahan dibangkitkan dengan perlakuan berbeda. "
                     "Pakai --paksa hanya kalau kamu yakin perbedaannya tak "
                     "mempengaruhi hasil.")
        print("  → dilanjutkan karena --paksa\n")

    # ── (2) kumpulkan jalan, deteksi duplikat ───────────────────────────────
    per_kunci: dict[tuple[str, int], dict] = {}
    asal: dict[tuple[str, int], str] = {}
    duplikat: list[str] = []
    for tag, _, doc in pecahan:
        for r in doc["runs"]:
            k = (r.get("direction"), r.get("seed"))
            if k in per_kunci:
                duplikat.append(f"(arah={k[0]}, seed={k[1]}) ada di "
                                f"{asal[k]} dan {tag}")
            per_kunci[k] = r
            asal[k] = tag

    if duplikat and not args.izinkan_duplikat:
        print("PASANGAN (arah, seed) DUPLIKAT:")
        for d in duplikat:
            print(f"  - {d}")
        sys.exit("BERHENTI: duplikat biasanya berarti salah satu pecahan "
                 "dijalankan dengan --seeds yang keliru. Periksa dulu; kalau "
                 "memang disengaja, pakai --izinkan-duplikat (yang terakhir menang).")

    # ── (3) urutkan seperti run_factor.py: arah di luar, seed di dalam ──────
    # Bukan kosmetik: laporan per-hop dan transkrip membaca `runs` berurutan,
    # jadi urutannya harus sama seperti kalau sel dijalankan sekaligus.
    urut_arah = [d.strip() for d in
                 str(pecahan[0][2]["args"].get("directions", "")).split(",")
                 if d.strip()]
    def _kunci_urut(k: tuple[str, int]) -> tuple[int, int]:
        arah, seed = k
        i = urut_arah.index(arah) if arah in urut_arah else len(urut_arah)
        return (i, seed if seed is not None else -1)

    kunci_urut = sorted(per_kunci, key=_kunci_urut)
    runs = [per_kunci[k] for k in kunci_urut]
    seeds = sorted({k[1] for k in per_kunci if k[1] is not None})
    arah = [d for d in urut_arah if any(k[0] == d for k in per_kunci)]

    # ── (4) laporan ─────────────────────────────────────────────────────────
    print(f"pecahan  : {len(pecahan)}")
    for tag, p, doc in pecahan:
        s = sorted({r.get('seed') for r in doc['runs']})
        print(f"  - {tag:34} {len(doc['runs']):3d} jalan  seed={s}")
    lolos = sum(1 for r in runs if r.get("passing"))
    eks = sum(len(r.get("factors") or []) for r in runs)
    print(f"gabungan : {len(runs)} jalan = {len(arah)} arah x {len(seeds)} seed "
          f"{seeds}")
    print(f"           lolos gate {lolos}/{len(runs)} ({100*lolos/len(runs):.0f}%), "
          f"{eks} ekspresi")
    hilang = [(d, s) for d in arah for s in seeds if (d, s) not in per_kunci]
    if hilang:
        print(f"⚠️  {len(hilang)} sel kosong (arah x seed tak lengkap): {hilang}")

    keluar = OUT_FACTOR / f"frontend_{args.keluaran}.json"
    parquet = OUT_FACTOR / f"icseries_{args.keluaran}.parquet"

    if args.dry_run:
        print(f"\n[dry-run] akan menulis → {keluar}")
        return

    # ── (5) tulis, dengan cadangan berkas lama ──────────────────────────────
    # Cadangan TIDAK boleh mendarat di `results/factor/`. Setiap pembaca korpus
    # menandai sumber lewat nama direktori induk — `kumpulkan_pendukung.py`
    # korpus_faktor() dan `agregasi_agent_trace.py` korpus() memakai
    # `sumber = "matriks" if p.parent.name == "factor" else p.parent.name`,
    # sementara `faktor_perhop.py`, `eval/skor_holdout.py`, dan
    # `eval/rescore_all.py` men-glob `frontend_*.json` di direktori itu langsung.
    # Cadangan bernama `frontend_<tag>.sebelum_gabung_*.json` cocok dengan pola
    # itu, jadi ia terhitung sebagai sel matriks tambahan yang isinya jalan yang
    # SAMA dengan sel gabungannya — korpus tercacah dua kali tanpa peringatan.
    # Terjadi 2026-08-28 pada tiga sel kv_and_text (60 jalan ganda); pecahannya
    # ada di results/arsip_pecahan_gabung_2026-08-28/. Sama alasannya dengan
    # results/arsip_faktor_6jalan_2026-08-10/: yang tak boleh ikut matriks
    # dipisah DIREKTORINYA, bukan dihapus.
    if keluar.exists():
        arsip = keluar.parent.parent / f"arsip_pecahan_gabung_{time.strftime('%Y-%m-%d')}"
        arsip.mkdir(parents=True, exist_ok=True)
        cadangan = arsip / f"{keluar.stem}.sebelum_gabung_{time.strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(keluar, cadangan)
        print(f"\ncadangan lama → {cadangan.parent.name}/{cadangan.name}")

    doc_baru = dict(pecahan[0][2])          # salin struktur pecahan pertama
    args_baru = dict(pecahan[0][2]["args"])
    args_baru["seeds"] = ",".join(str(s) for s in seeds)
    args_baru["tag"] = args.keluaran
    args_baru["digabung_dari"] = " + ".join(args.dari)
    doc_baru["args"] = args_baru
    doc_baru["runs"] = runs
    # Jejak asal-usul: tanpa ini tak ada cara memeriksa dari mana tiap jalan
    # datang setelah tag sementaranya dihapus.
    doc_baru["gabungan"] = {
        "dibuat": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "alat": "scripts/gabung_jalan.py",
        "dipaksa": bool(args.paksa),
        "sumber": [
            {"tag": tag, "berkas": p.name, "sha256": _sha256(p),
             "n_jalan": len(doc["runs"]),
             "seeds": sorted({r.get("seed") for r in doc["runs"]}),
             "args_seeds": doc["args"].get("seeds")}
            for tag, p, doc in pecahan
        ],
    }
    keluar.write_text(json.dumps(doc_baru, indent=2, default=str))
    print(f"tersimpan → {keluar}  ({len(runs)} jalan)")

    # ── (6) ingatkan skoring ────────────────────────────────────────────────
    print("\nLANGKAH WAJIB BERIKUTNYA — skoring IC belum mencakup jalan baru.")
    if parquet.exists():
        print(f"⚠️  {parquet.name} masih dari korpus LAMA; jangan dipakai "
              f"sebelum diskor ulang.")
    print(f"    PYTHONPATH=backend python backend/factor/run_factor.py "
          f"--score-only --tag {args.keluaran} --budget 900")
    print("Lalu regenerasi analisis:")
    print("    python scripts/kekuatan_uji_faktor.py --comm-mode "
          f"{args_baru.get('comm_mode')}")
    print("    python scripts/kumpulkan_pendukung.py && "
          "python scripts/faktor_perhop.py && python scripts/visual_bab4.py --all")


if __name__ == "__main__":
    main()
