#!/usr/bin/env python3
"""Bangun panel harga-volume harian Bursa Efek Indonesia (IDX) — pengganti data A-share.

KENAPA BERKAS INI ADA
---------------------
Seluruh penilaian ekspresi DSL di penelitian ini (RankIC, ICIR, t-stat, dan
metrik backtest long--short) berjalan di CPU di atas SATU berkas panel:
`backend/hf_data/daily_pv.h5` — MultiIndex (datetime, instrument) dengan kolom
`$open $close $high $low $volume`. Tidak ada ketergantungan lain ke pasar
tertentu: agen LLM hanya memancarkan ekspresi atas simbol-simbol itu, dan
templat prompt tidak pernah menyebut bursa mana pun (`market_context` tak
pernah diisi pemanggil mana pun). Karena itu MENGGANTI PASAR = mengganti berkas
panel ini saja. Tidak ada tahap GPU yang perlu diulang.

Skrip ini membangun padanan IDX dari berkas panel tersebut.

SUMBER DATA
-----------
Harga/volume  : Yahoo Finance (`yfinance`), ticker `<KODE>.JK`, `auto_adjust=True`
                sehingga harga sudah disesuaikan aksi korporasi (split/dividen)
                — setara kolom harga ter-adjust di panel A-share.
Daftar emiten : Wikipedia (daftar tercatat BEI + konstituen LQ45/IDX30/IDX80/
                KOMPAS100). Daftar konstituen dipakai HANYA sebagai pembanding;
                universe penelitian dibentuk oleh saringan likuiditas mekanis
                di bawah, bukan oleh keanggotaan indeks.

UNIVERSE — SARINGAN LIKUIDITAS TAHUNAN
--------------------------------------
Memakai daftar konstituen indeks hari ini untuk periode 2021--2025 menimbulkan
survivorship bias yang tak perlu. Sebagai gantinya universe dibentuk ulang tiap
tahun kalender Y memakai HANYA informasi tahun Y-1:

    nilai_transaksi_harian = $close x $volume
    syarat  : >= `min_hari` hari perdagangan di tahun Y-1
    peringkat: median nilai_transaksi_harian tahun Y-1, ambil N teratas

Itu meniru cara BEI menyusun LQ45/KOMPAS100 (likuiditas + kapitalisasi) tetapi
sepenuhnya mekanis dan bisa direproduksi dari berkas ini. Sisa bias yang TETAP
ADA harus dilaporkan di skripsi: kumpulan kandidat berasal dari daftar emiten
yang tercatat SAAT INI, jadi emiten yang delisting sebelum daftar itu diambil
tidak pernah masuk kandidat.

KELUARAN (semua di `backend/hf_data_id/`)
-----------------------------------------
  emiten_idx.csv          metadata emiten (kode, nama, tanggal catat, sektor)
  indeks_membership.json  konstituen LQ45/IDX30/IDX80/KOMPAS100 saat diambil
  mentah/<batch>.parquet  unduhan per-batch (untuk melanjutkan bila putus)
  daily_pv_idx.h5         PANEL PENUH semua kandidat, key="data"
  daily_pv_idx_top<N>.h5  panel yang sudah disaring universe top-N tahunan
  universe_top<N>.json    keanggotaan universe per tahun + statistik likuiditas
  ringkasan.json          jumlah baris/emiten/hari + rentang tanggal

PEMAKAIAN
---------
    .venv/bin/python scripts/ambil_data_idx.py                 # penuh
    .venv/bin/python scripts/ambil_data_idx.py --lewati-unduh  # rakit ulang saja
    .venv/bin/python scripts/ambil_data_idx.py --topn 45,100,200
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

AKAR = Path(__file__).resolve().parents[1]
KELUAR = AKAR / "backend" / "hf_data_id"
MENTAH = KELUAR / "mentah"

MULAI = "2019-01-01"   # >= 400 hari kalender warmup sebelum jendela seleksi 2021
SELESAI = "2026-01-15"  # menutup holdout 2022--2025 (batas 2025-12-26)

WIKI = {
    "daftar": "https://id.wikipedia.org/wiki/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia",
    "lq45": "https://en.wikipedia.org/wiki/LQ45",
    "kompas100": "https://id.wikipedia.org/wiki/Indeks_Kompas100",
    "idx30": "https://id.wikipedia.org/wiki/Indeks_IDX30",
    "idx80": "https://id.wikipedia.org/wiki/IDX80",
}
HEAD = {"User-Agent": "Mozilla/5.0 (skripsi-riset-akademik; kontak via repo)"}


def _kode(x: object) -> str | None:
    """Ambil kode 4-huruf dari sel seperti 'BEI: AALI' / 'IDX: AADI' / 'AALI'."""
    s = str(x).upper()
    s = s.replace("BEI:", " ").replace("IDX:", " ")
    m = re.search(r"\b([A-Z]{4})\b", s)
    return m.group(1) if m else None


def _tabel(url: str) -> list[pd.DataFrame]:
    import requests

    r = requests.get(url, headers=HEAD, timeout=45)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


# ── 1. daftar emiten + keanggotaan indeks ────────────────────────────────────
def ambil_daftar() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    t = _tabel(WIKI["daftar"])[0]
    t = t.rename(columns={"Kode": "kode_mentah", "Nama perusahaan": "nama",
                          "Tanggal pencatatan": "tgl_catat",
                          "Papan pencatatan": "papan", "Sektor": "sektor"})
    t["kode"] = t["kode_mentah"].map(_kode)
    t = t.dropna(subset=["kode"]).drop_duplicates("kode")
    emiten = t[["kode", "nama", "tgl_catat", "papan", "sektor"]].reset_index(drop=True)

    keanggotaan: dict[str, list[str]] = {}
    for nama, url in WIKI.items():
        if nama == "daftar":
            continue
        try:
            kandidat: list[str] = []
            for tb in _tabel(url):
                kol = [c for c in tb.columns if re.search(r"ticker|kode|symbol|emiten",
                                                          str(c), re.I)]
                if not kol:
                    continue
                kandidat = [k for k in tb[kol[0]].map(_kode).tolist() if k]
                if len(kandidat) >= 20:
                    break
            if kandidat:
                keanggotaan[nama] = sorted(set(kandidat))
        except Exception as e:  # noqa: BLE001
            print(f"  ! gagal ambil {nama}: {e}")
    return emiten, keanggotaan


# ── 2. unduh OHLCV ───────────────────────────────────────────────────────────
def unduh(kode: list[str], ukuran_batch: int = 40, jeda: float = 1.5) -> None:
    import yfinance as yf

    MENTAH.mkdir(parents=True, exist_ok=True)
    batch = [kode[i:i + ukuran_batch] for i in range(0, len(kode), ukuran_batch)]
    for i, b in enumerate(batch, 1):
        f = MENTAH / f"batch_{i:03d}.parquet"
        if f.exists():
            print(f"  [{i}/{len(batch)}] lewati (sudah ada)")
            continue
        tick = [f"{k}.JK" for k in b]
        for percobaan in range(3):
            try:
                df = yf.download(tick, start=MULAI, end=SELESAI, auto_adjust=True,
                                 progress=False, threads=False, group_by="column",
                                 timeout=60)
                if df is None or df.empty:
                    raise RuntimeError("kosong")
                # kolom MultiIndex (Price, Ticker) → panel panjang
                panjang = (df.stack(level=1, future_stack=True)
                             .rename_axis(index=["datetime", "instrument"]))
                panjang = panjang.reset_index()
                panjang["instrument"] = panjang["instrument"].str.replace(
                    ".JK", "", regex=False)
                panjang.to_parquet(f)
                hidup = panjang.dropna(subset=["Close"])["instrument"].nunique()
                print(f"  [{i}/{len(batch)}] {len(b)} ticker → {len(panjang):,} baris, "
                      f"{hidup} punya data")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(batch)}] percobaan {percobaan + 1} gagal: {e}")
                time.sleep(5 * (percobaan + 1))
        time.sleep(jeda)


# ── 3. rakit panel ───────────────────────────────────────────────────────────
def rakit() -> pd.DataFrame:
    berkas = sorted(MENTAH.glob("batch_*.parquet"))
    if not berkas:
        sys.exit("tidak ada berkas mentah — jalankan tanpa --lewati-unduh dulu")
    df = pd.concat([pd.read_parquet(f) for f in berkas], ignore_index=True)
    df = df.rename(columns={"Open": "$open", "Close": "$close", "High": "$high",
                            "Low": "$low", "Volume": "$volume"})
    kol = ["$open", "$close", "$high", "$low", "$volume"]
    df = df[["datetime", "instrument"] + kol]
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["$close"])
    # hari tanpa transaksi (volume 0 & harga datar) tetap dibiarkan: itu keadaan
    # pasar sungguhan di IDX dan penyaringannya adalah keputusan universe, bukan data.
    df = df[df["$close"] > 0]
    df = df.drop_duplicates(["datetime", "instrument"])
    df = df.set_index(["datetime", "instrument"]).sort_index()
    for c in kol:
        df[c] = df[c].astype("float32")
    return df


# ── 4. universe likuiditas tahunan ───────────────────────────────────────────
def universe(df: pd.DataFrame, n: int, min_hari: int = 150) -> dict:
    nilai = (df["$close"] * df["$volume"]).rename("nilai")
    tahun = nilai.index.get_level_values("datetime").year
    hasil: dict[str, list[str]] = {}
    stat: dict[str, dict] = {}
    tahun_ada = sorted(set(tahun))
    for y in tahun_ada:
        sebelum = y - 1
        if sebelum not in tahun_ada:
            continue
        s = nilai[tahun == sebelum]
        g = s.groupby(level="instrument")
        ringkas = pd.DataFrame({"median": g.median(), "hari": g.size()})
        ringkas = ringkas[ringkas["hari"] >= min_hari]
        pilih = ringkas.sort_values("median", ascending=False).head(n)
        hasil[str(y)] = sorted(pilih.index.tolist())
        stat[str(y)] = {
            "dasar_tahun": sebelum,
            "kandidat_lolos_min_hari": int(len(ringkas)),
            "median_nilai_transaksi_terkecil_terpilih": float(pilih["median"].min()),
            "median_nilai_transaksi_terbesar": float(pilih["median"].max()),
        }
    return {"n": n, "min_hari": min_hari, "per_tahun": hasil, "statistik": stat}


def saring(df: pd.DataFrame, u: dict) -> pd.DataFrame:
    dt = df.index.get_level_values("datetime")
    inst = df.index.get_level_values("instrument")
    tahun = pd.Series(dt.year, index=range(len(df)))
    masuk = pd.Series(False, index=range(len(df)))
    for y, anggota in u["per_tahun"].items():
        m = (tahun == int(y)).to_numpy() & inst.isin(anggota)
        masuk |= pd.Series(m, index=range(len(df)))
    return df[masuk.to_numpy()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewati-unduh", action="store_true")
    ap.add_argument("--topn", default="45,100,200")
    ap.add_argument("--min-hari", type=int, default=150)
    ap.add_argument("--batas-kandidat", type=int, default=0,
                    help="0 = semua emiten tercatat; >0 untuk uji cepat")
    a = ap.parse_args()

    KELUAR.mkdir(parents=True, exist_ok=True)

    f_emiten = KELUAR / "emiten_idx.csv"
    f_indeks = KELUAR / "indeks_membership.json"
    if f_emiten.exists() and f_indeks.exists():
        emiten = pd.read_csv(f_emiten)
        keanggotaan = json.loads(f_indeks.read_text())
        print(f"daftar emiten dari cache: {len(emiten)} emiten")
    else:
        print("mengambil daftar emiten + konstituen indeks dari Wikipedia …")
        emiten, keanggotaan = ambil_daftar()
        emiten.to_csv(f_emiten, index=False)
        f_indeks.write_text(json.dumps(keanggotaan, indent=1))
        print(f"  {len(emiten)} emiten; indeks: "
              + ", ".join(f"{k}={len(v)}" for k, v in keanggotaan.items()))

    kode = emiten["kode"].tolist()
    if a.batas_kandidat:
        kode = kode[:a.batas_kandidat]

    if not a.lewati_unduh:
        print(f"mengunduh {len(kode)} ticker .JK  {MULAI} → {SELESAI} …")
        unduh(kode)

    print("merakit panel …")
    df = rakit()
    dt = df.index.get_level_values("datetime")
    print(f"  {len(df):,} baris, {df.index.get_level_values('instrument').nunique()} emiten, "
          f"{dt.min().date()} → {dt.max().date()}")

    f_penuh = KELUAR / "daily_pv_idx.h5"
    df.to_hdf(f_penuh, key="data", mode="w", complevel=5, complib="blosc")
    print(f"  → {f_penuh.name}")

    ringkas = {"panel_penuh": {"baris": int(len(df)),
                               "emiten": int(df.index.get_level_values("instrument").nunique()),
                               "mulai": str(dt.min().date()), "selesai": str(dt.max().date())},
               "universe": {}}

    for n in [int(x) for x in a.topn.split(",") if x.strip()]:
        u = universe(df, n, a.min_hari)
        (KELUAR / f"universe_top{n}.json").write_text(json.dumps(u, indent=1))
        d = saring(df, u)
        f = KELUAR / f"daily_pv_idx_top{n}.h5"
        d.to_hdf(f, key="data", mode="w", complevel=5, complib="blosc")
        per_hari = d.groupby(level="datetime").size()
        ringkas["universe"][f"top{n}"] = {
            "baris": int(len(d)),
            "emiten_unik": int(d.index.get_level_values("instrument").nunique()),
            "rerata_emiten_per_hari": float(per_hari.mean()),
            "hari_perdagangan": int(per_hari.size),
            "hari_per_tahun": {str(y): int(v) for y, v in
                               per_hari.groupby(per_hari.index.year).size().items()},
        }
        print(f"  → {f.name}: {len(d):,} baris, "
              f"{per_hari.mean():.1f} emiten/hari, {per_hari.size} hari")

    (KELUAR / "ringkasan.json").write_text(json.dumps(ringkas, indent=1))
    print("selesai.")


if __name__ == "__main__":
    main()
