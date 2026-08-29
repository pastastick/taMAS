#!/usr/bin/env python3
"""Bangun panel IDX berbasis KEANGGOTAAN INDEKS RESMI (bukan saringan likuiditas).

Dipakai ketika universe penelitian harus berupa indeks yang sudah dikenal
(LQ45 / KOMPAS100 / IDX30) alih-alih universe bentukan sendiri. Anggota yang
datanya tidak lengkap pada jendela penelitian dibuang, karena saham yang hanya
ada separuh periode menghasilkan IC harian yang dihitung atas jumlah saham yang
berubah-ubah — dan itu mencampur perubahan universe ke dalam angka yang
seharusnya mengukur ekspresi.

    .venv/bin/python scripts/panel_indeks_idx.py --indeks lq45 --min-cakupan 0.98
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

AKAR = Path(__file__).resolve().parents[1]
DIR = AKAR / "backend" / "hf_data_id"

# Jendela penelitian penuh: seleksi (2021) + holdout (2022-2025).
AWAL, AKHIR = "2021-01-01", "2025-12-26"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indeks", default="lq45,kompas100")
    ap.add_argument("--min-cakupan", type=float, default=0.98,
                    help="fraksi hari bursa yang wajib dipunyai tiap emiten")
    a = ap.parse_args()

    df = pd.read_hdf(DIR / "daily_pv_idx.h5", key="data")
    keanggotaan = json.loads((DIR / "indeks_membership.json").read_text())

    jendela = df.loc[AWAL:AKHIR]
    hari = jendela.index.get_level_values("datetime").nunique()
    cacah = jendela.groupby(level="instrument").size()

    ringkas = {}
    for nama in [s.strip() for s in a.indeks.split(",") if s.strip()]:
        anggota = keanggotaan.get(nama)
        if not anggota:
            print(f"! indeks '{nama}' tak ada di indeks_membership.json")
            continue
        lolos = sorted(k for k in anggota
                       if cacah.get(k, 0) >= a.min_cakupan * hari)
        dibuang = sorted(set(anggota) - set(lolos))

        # Panel disimpan dengan SEJARAH PENUH (termasuk 2019-2020) supaya
        # warmup 400 hari untuk jendela seleksi 2021 tetap tersedia; jendela
        # penilaiannya sendiri dipotong `eval/ic.py`, bukan di sini.
        d = df[df.index.get_level_values("instrument").isin(lolos)]
        f = DIR / f"daily_pv_idx_{nama}.h5"
        d.to_hdf(f, key="data", mode="w", complevel=5, complib="blosc")

        per_hari = d.loc[AWAL:AKHIR].groupby(level="datetime").size()
        ringkas[nama] = {
            "anggota_indeks": len(anggota),
            "dipakai": len(lolos),
            "dibuang_cakupan": dibuang,
            "emiten": lolos,
            "baris_total": int(len(d)),
            "rerata_emiten_per_hari": float(per_hari.mean()),
            "min_emiten_per_hari": int(per_hari.min()),
            "hari_bursa_jendela": int(hari),
            "desil_per_sisi": int(per_hari.mean() * 0.1),
            "kuintil_per_sisi": int(per_hari.mean() * 0.2),
        }
        print(f"{nama:10s} {len(lolos):3d}/{len(anggota)} emiten dipakai · "
              f"{per_hari.mean():.1f}/hari (min {per_hari.min()}) · "
              f"desil={int(per_hari.mean()*0.1)}/sisi · "
              f"kuintil={int(per_hari.mean()*0.2)}/sisi → {f.name}")
        if dibuang:
            print(f"           dibuang (<{a.min_cakupan:.0%} hari): {', '.join(dibuang)}")

    (DIR / "panel_indeks.json").write_text(json.dumps(ringkas, indent=1))


if __name__ == "__main__":
    main()
