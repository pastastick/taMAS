#!/usr/bin/env python3
"""Analisis lengkap lengan faktor pada pasar Indonesia (LQ45) — satu berkas JSON.

Menggabungkan SEMUA angka yang dipakai bab hasil ke satu tempat, supaya tabel
LaTeX dibangkitkan dari satu sumber dan tidak bisa saling bertentangan.

MASUKAN
  results/factor/frontend_<sel>.json                       (generasi: GPU, Agustus 2026)
  results/factor/holdout_daily_pv_idx_lq45_2021-*_q0.2.json (jendela seleksi)
  results/factor/holdout_daily_pv_idx_lq45_2022-*_q0.2.json (jendela holdout)
  results/factor/icseries_daily_pv_idx_lq45_*.parquet       (deret IC harian, opsional)
  results/factor/lantai_acak_*.json                         (lantai acak, opsional)
  results/factor/gate_lintas_pasar.json                     (uji gate lintas pasar, opsional)

KELUARAN
  results/idx/analisis_idx.json

CATATAN DENOMINATOR. Semua laju keandalan memakai denominator JALAN (arah x
seed = 20 per sel), bukan "panggilan LLM" — jumlah panggilan berbeda antar sel
karena gate memicu repair, sehingga persentase atas basis itu tidak sebanding.
Jumlah panggilan dilaporkan terpisah sebagai ukuran BIAYA.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from paths import bootstrap, OUT_FACTOR, RESULTS  # noqa: E402
bootstrap()

KELUAR = RESULTS / "idx"
SELEKSI = "holdout_daily_pv_idx_lq45_2021-01-01_2021-12-31_q0.2.json"
HOLDOUT = "holdout_daily_pv_idx_lq45_2022-01-01_2025-12-26_q0.2.json"

METODE = ["raw", "soft", "gumbel", "sample", "moi"]
KELUARGA_R = ["soft", "gumbel", "sample", "moi"]


def medium_metode(tag: str) -> tuple[str, str]:
    for m in ("kv_and_text", "kv", "text"):
        if tag == m:
            return m, "-"
        if tag.startswith(m + "_"):
            return m, tag[len(m) + 1:]
    return tag, "-"


# ── 1. keandalan & biaya, dari frontend_*.json ───────────────────────────────
def per_sel_generasi() -> dict:
    out = {}
    for p in sorted(OUT_FACTOR.glob("frontend_*.json")):
        tag = p.stem[len("frontend_"):]
        doc = json.loads(p.read_text())
        runs = doc["runs"]
        n_jalan = len(runs)

        # satu jalan "lolos gate" bila menghasilkan >=1 ekspresi yang lolos.
        lolos_jalan = sum(1 for r in runs if (r.get("passing") or []))
        ada_ekspresi = sum(1 for r in runs if (r.get("factors") or []))

        n_ekspresi = sum(len(r.get("factors") or []) for r in runs)
        n_lolos = sum(len(r.get("passing") or []) for r in runs)

        # biaya: percobaan construct per jalan = 1 + repair_attempts
        percobaan = [1 + (r.get("repair_attempts") or 0) for r in runs]

        # token & waktu dari agent_trace
        tok_keluar, detik, kv_len_akhir = [], [], []
        for r in runs:
            tr = r.get("agent_trace") or []
            tok_keluar.append(sum(a.get("n_out_tok", 0) for a in tr))
            detik.append(r.get("duration_s", 0.0))
            if tr:
                kv_len_akhir.append(tr[-1].get("kv_len", 0))

        # alasan penolakan gate
        alasan = defaultdict(int)
        for r in runs:
            for g in (r.get("gate_log") or []):
                if not g.get("ok"):
                    alasan[(g.get("reason") or "?").split(":")[0].strip()] += 1

        medium, metode = medium_metode(tag)
        out[tag] = {
            "medium": medium, "metode": metode,
            "jalan": n_jalan,
            "jalan_menghasilkan_ekspresi": ada_ekspresi,
            "jalan_lolos_gate": lolos_jalan,
            "laju_lolos_gate": lolos_jalan / n_jalan if n_jalan else None,
            "ekspresi_total": n_ekspresi,
            "ekspresi_lolos_gate": n_lolos,
            "laju_lolos_ekspresi": n_lolos / n_ekspresi if n_ekspresi else None,
            "percobaan_construct_per_jalan": float(np.mean(percobaan)),
            "token_keluar_per_jalan": float(np.mean(tok_keluar)),
            "detik_per_jalan": float(np.mean(detik)),
            "kv_len_akhir_median": float(np.median(kv_len_akhir)) if kv_len_akhir else None,
            "alasan_tolak_gate": dict(sorted(alasan.items(), key=lambda x: -x[1])),
        }
    return out


# ── 2. mutu sinyal per sel, dari laporan IDX ─────────────────────────────────
def per_sel_ic(nama_berkas: str) -> dict:
    p = OUT_FACTOR / nama_berkas
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    per_tag = defaultdict(list)
    for r in doc["per_ekspresi"]:
        per_tag[r["tag"]].append(r)

    out = {"_meta": {k: doc.get(k) for k in
                     ("window", "pasar", "quantile", "cost_bps", "trading_days")}}
    for tag, rows in per_tag.items():
        hidup = [r for r in rows
                 if r.get("ic") is not None and (r.get("n_unique") or 0) > 2]
        sig = [r for r in hidup
               if r.get("tstat") is not None and abs(r["tstat"]) >= 1.96]
        bt = [r for r in rows if r.get("bt_sharpe") is not None]

        def med(key, src):
            v = [x[key] for x in src if x.get(key) is not None]
            return float(np.median(v)) if v else None

        out[tag] = {
            "ekspresi": len(rows),
            "hidup": len(hidup),
            "signifikan": len(sig),
            "laju_signifikan": len(sig) / len(hidup) if hidup else None,
            "mean_abs_ic": float(np.mean([abs(r["ic"]) for r in hidup])) if hidup else None,
            "median_abs_ic": float(np.median([abs(r["ic"]) for r in hidup])) if hidup else None,
            "max_abs_ic": float(np.max([abs(r["ic"]) for r in hidup])) if hidup else None,
            "median_abs_tstat": float(np.median([abs(r["tstat"]) for r in hidup
                                                 if r.get("tstat") is not None])) if hidup else None,
            "n_backtest": len(bt),
            "median_sharpe": med("bt_sharpe", bt),
            "median_ann_return": med("bt_ann_return", bt),
            "median_ann_vol": med("bt_ann_vol", bt),
            "median_max_drawdown": med("bt_max_drawdown", bt),
            "median_turnover": med("bt_turnover", bt),
            "median_hit_rate": med("bt_hit_rate", bt),
        }
    return out


# ── 3. uji formal ────────────────────────────────────────────────────────────
def uji_formal(gen: dict, ic: dict) -> dict:
    from scipy.stats import fisher_exact, mannwhitneyu

    def cohen_h(p1, p2):
        return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))

    hasil = {}
    for medium in ("kv", "kv_and_text"):
        selR = [f"{medium}_{m}" for m in KELUARGA_R if f"{medium}_{m}" in gen]
        sel_raw = f"{medium}_raw"
        if sel_raw not in gen or not selR:
            continue

        # (a) laju lolos gate PER JALAN — klaim keandalan utama
        aR = sum(gen[s]["jalan_lolos_gate"] for s in selR)
        nR = sum(gen[s]["jalan"] for s in selR)
        ar = gen[sel_raw]["jalan_lolos_gate"]
        nr = gen[sel_raw]["jalan"]
        odds, p = fisher_exact([[aR, nR - aR], [ar, nr - ar]])
        blok = {"lolos_gate_per_jalan": {
            "R": [aR, nR], "raw": [ar, nr],
            "p_R": aR / nR, "p_raw": ar / nr,
            "selisih_pp": 100 * (aR / nR - ar / nr),
            "fisher_p": float(p), "odds_ratio": float(odds),
            "cohen_h": cohen_h(aR / nR, ar / nr)}}

        # (b) laju ekspresi HIDUP di pasar IDX — klaim validitas hilir
        if all(s in ic for s in selR + [sel_raw]):
            hR = sum(ic[s]["hidup"] for s in selR)
            eR = sum(ic[s]["ekspresi"] for s in selR)
            hr, er = ic[sel_raw]["hidup"], ic[sel_raw]["ekspresi"]
            odds, p = fisher_exact([[hR, eR - hR], [hr, er - hr]])
            blok["ekspresi_hidup"] = {
                "R": [hR, eR], "raw": [hr, er],
                "p_R": hR / eR, "p_raw": hr / er,
                "selisih_pp": 100 * (hR / eR - hr / er),
                "fisher_p": float(p), "odds_ratio": float(odds),
                "cohen_h": cohen_h(hR / eR, hr / er)}
        hasil[medium] = blok
    return hasil


# ── 4. mutu sinyal: R vs raw pada |IC| (bukan sekadar validitas) ────────────
def uji_mutu_sinyal(nama_berkas: str) -> dict:
    from scipy.stats import mannwhitneyu
    p = OUT_FACTOR / nama_berkas
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    ambil = lambda tags: [abs(r["ic"]) for r in doc["per_ekspresi"]
                          if r["tag"] in tags and r.get("ic") is not None
                          and (r.get("n_unique") or 0) > 2]
    out = {}
    for medium in ("kv", "kv_and_text"):
        a = ambil({f"{medium}_{m}" for m in KELUARGA_R})
        b = ambil({f"{medium}_raw"})
        if len(a) < 5 or len(b) < 5:
            continue
        u, pv = mannwhitneyu(a, b, alternative="two-sided")
        # ukuran efek rank-biserial
        rb = 1 - 2 * u / (len(a) * len(b))
        out[medium] = {"n_R": len(a), "n_raw": len(b),
                       "median_abs_ic_R": float(np.median(a)),
                       "median_abs_ic_raw": float(np.median(b)),
                       "mannwhitney_p": float(pv),
                       "rank_biserial": float(-rb)}
    return out


# ── 5. stabilitas seleksi → holdout ──────────────────────────────────────────
def stabilitas() -> dict:
    p = OUT_FACTOR / HOLDOUT
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    tot = {"berpasangan": 0, "berbalik": 0, "sig_seleksi": 0, "sig_keduanya": 0}
    per_tag = {}
    for t in doc["per_tag"]:
        per_tag[t["tag"]] = {k: t[k] for k in
                             ("berpasangan", "berbalik_tanda", "signifikan", "hidup")}
        tot["berpasangan"] += t["berpasangan"]
        tot["berbalik"] += t["berbalik_tanda"]
    for r in doc["per_ekspresi"]:
        ts, th = r.get("tstat_seleksi"), r.get("tstat")
        if ts is not None and abs(ts) >= 1.96:
            tot["sig_seleksi"] += 1
            if th is not None and abs(th) >= 1.96:
                tot["sig_keduanya"] += 1
    tot["laju_berbalik"] = tot["berbalik"] / tot["berpasangan"] if tot["berpasangan"] else None
    tot["laju_bertahan_signifikan"] = (tot["sig_keduanya"] / tot["sig_seleksi"]
                                       if tot["sig_seleksi"] else None)
    return {"total": tot, "per_tag": per_tag}


# ── 6. peluruhan alpha per tahun, dari deret IC harian ───────────────────────
def peluruhan_alpha() -> dict:
    import pandas as pd
    berkas = sorted(OUT_FACTOR.glob("icseries_daily_pv_idx_lq45_*.parquet"))
    if not berkas:
        return {}
    df = pd.read_parquet(berkas[-1])
    df.index = pd.to_datetime(df.index)
    out = {"berkas": berkas[-1].name, "n_ekspresi": int(df.shape[1]), "per_tahun": {}}
    for tahun, blok in df.groupby(df.index.year):
        m = blok.mean()
        out["per_tahun"][str(tahun)] = {
            "hari": int(len(blok)),
            "mean_abs_ic": float(m.abs().mean()),
            "median_abs_ic": float(m.abs().median()),
            "frac_positif": float((m > 0).mean()),
        }
    return out


# ── 7. klaster sinyal: berapa banyak ekspresi yang sebenarnya BERBEDA ────────
def klaster_sinyal(ambang: float = 0.7) -> dict:
    """Ekspresi dengan deret IC harian berkorelasi > `ambang` membawa sinyal yang
    sama meski teksnya berbeda. Jumlah klaster = ukuran keragaman NYATA korpus,
    yang tak terlihat dari jumlah ekspresi maupun keragaman sintaksis."""
    import pandas as pd
    berkas = sorted(OUT_FACTOR.glob("icseries_daily_pv_idx_lq45_*.parquet"))
    if not berkas:
        return {}
    df = pd.read_parquet(berkas[-1]).dropna(axis=1, how="all")
    if df.shape[1] < 2:
        return {}
    C = df.corr(method="pearson").abs().fillna(0.0).to_numpy()
    n = C.shape[0]
    # union-find sederhana pada graf ambang
    induk = list(range(n))

    def cari(x):
        while induk[x] != x:
            induk[x] = induk[induk[x]]
            x = induk[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > ambang:
                a, b = cari(i), cari(j)
                if a != b:
                    induk[a] = b
    ukuran = defaultdict(int)
    for i in range(n):
        ukuran[cari(i)] += 1
    besar = sorted(ukuran.values(), reverse=True)
    return {"n_ekspresi": n, "ambang_korelasi": ambang,
            "n_klaster": len(ukuran),
            "rasio_klaster_per_ekspresi": len(ukuran) / n,
            "ukuran_klaster_terbesar": besar[:10]}


def main() -> None:
    KELUAR.mkdir(parents=True, exist_ok=True)
    gen = per_sel_generasi()
    ic_sel = per_sel_ic(SELEKSI)
    ic_hold = per_sel_ic(HOLDOUT)

    doc = {
        "pasar": "IDX LQ45 (37 emiten)",
        "generasi": gen,
        "seleksi_2021": ic_sel,
        "holdout_2022_2025": ic_hold,
        "uji_formal": uji_formal(gen, {k: v for k, v in ic_sel.items() if k != "_meta"}),
        "uji_mutu_sinyal_seleksi": uji_mutu_sinyal(SELEKSI),
        "uji_mutu_sinyal_holdout": uji_mutu_sinyal(HOLDOUT),
        "stabilitas": stabilitas(),
        "peluruhan_alpha": peluruhan_alpha(),
        "klaster_sinyal": klaster_sinyal(),
    }
    for nama, berkas in [("lantai_acak", "lantai_acak_daily_pv_idx_lq45_2021-01-01_2021-12-31.json"),
                         ("gate_lintas_pasar", "gate_lintas_pasar.json")]:
        p = OUT_FACTOR / berkas
        if p.exists():
            d = json.loads(p.read_text())
            doc[nama] = {k: v for k, v in d.items() if k != "per_ekspresi"}

    f = KELUAR / "analisis_idx.json"
    f.write_text(json.dumps(doc, indent=1))
    print(f"→ {f}")
    for medium, blok in doc["uji_formal"].items():
        g = blok["lolos_gate_per_jalan"]
        print(f"[{medium}] lolos gate/jalan: R={g['p_R']:.1%} raw={g['p_raw']:.1%} "
              f"selisih={g['selisih_pp']:+.1f}pp  fisher p={g['fisher_p']:.2e}")


if __name__ == "__main__":
    main()
