#!/usr/bin/env python3
"""Analisis lengan benchmark: tabel utama, uji berpasangan, koreksi multiplisitas.

Menghasilkan seluruh angka Bab 4 lengan bench dari berkas mentah per-soal,
bukan dari ringkasan yang sudah jadi — supaya tiap angka bisa ditelusuri.

  analisis/bench_tabel.csv      satu baris = satu sel (akurasi, format, token, waktu)
  analisis/bench_uji.csv        seluruh uji berpasangan + p terkoreksi Holm & BH
  analisis/bench_ringkas.json   Cochran Q, kontras keluarga, disosiasi, efisiensi
"""
from __future__ import annotations

import itertools
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/tmp/results/bench")
PENDUKUNG = Path("/tmp/results/pendukung")
OUT = ROOT / "analisis"
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260811)
N_BOOT = 10000
KV_MODES = ["raw", "soft", "gumbel", "sample", "moi"]
RELAKSASI = ["soft", "gumbel", "sample", "moi"]
LABEL_TUGAS = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}


# ── uji ─────────────────────────────────────────────────────────────────────
def mcnemar_eksak(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """p dua sisi uji McNemar eksak (binomial) pada pasangan diskordan."""
    n01 = int(np.sum((a == 1) & (b == 0)))
    n10 = int(np.sum((a == 0) & (b == 1)))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return float(min(1.0, p)), n01, n10


def ci_boot_selisih(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """CI 95% bootstrap berpasangan untuk mean(a) - mean(b)."""
    d = a.astype(float) - b.astype(float)
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def ci_boot_rerata(x: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    boot = x[idx].astype(float).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cochran_q(mat: np.ndarray) -> tuple[float, int, float]:
    """Cochran's Q untuk k sampel biner berpasangan. mat: [n_soal, k]."""
    from scipy.stats import chi2
    n, k = mat.shape
    Gj = mat.sum(axis=0)          # total per perlakuan
    Li = mat.sum(axis=1)          # total per blok
    pembilang = (k - 1) * (k * np.sum(Gj ** 2) - np.sum(Gj) ** 2)
    penyebut = k * np.sum(Li) - np.sum(Li ** 2)
    if penyebut == 0:
        return 0.0, k - 1, 1.0
    Q = pembilang / penyebut
    return float(Q), k - 1, float(chi2.sf(Q, k - 1))


def koreksi(p: list[float]) -> tuple[list[float], list[float]]:
    """(Holm, Benjamini-Hochberg) — keduanya dikembalikan sebagai p terkoreksi."""
    m = len(p)
    urut = sorted(range(m), key=lambda i: p[i])
    holm = [0.0] * m
    jalan = 0.0
    for r, i in enumerate(urut):
        jalan = max(jalan, (m - r) * p[i])
        holm[i] = min(1.0, jalan)
    bh = [0.0] * m
    jalan = 1.0
    for r in range(m - 1, -1, -1):
        i = urut[r]
        jalan = min(jalan, m / (r + 1) * p[i])
        bh[i] = min(1.0, jalan)
    return holm, bh


# ── pemuatan ────────────────────────────────────────────────────────────────
def muat() -> dict:
    """{tugas: {sel: {meta, benar[], format[], n}}} — hanya sel utama (n=100)."""
    data: dict[str, dict] = {}
    for p in sorted(BENCH.glob("bench_*.json")):
        if p.name.endswith("_lampiran.json"):
            continue
        d = json.loads(p.read_text())
        m, s = d["_meta"], d["summary"]
        tugas = m["task"]
        medium = "baseline" if m.get("baseline") else m["comm_mode"]
        sel = f"{m['latent_mode']}/{medium}"
        res = sorted(d["results"], key=lambda r: r["index"])
        data.setdefault(tugas, {})[sel] = {
            "meta": m, "n": s["n"],
            "sidik_soal": tuple(r["index"] for r in res),
            "benar": np.array([1 if r.get("correct") else 0 for r in res]),
            "format": np.array([1 if r.get("format_ok") else 0 for r in res]),
            "berkas": p.name,
        }
    return data


def muat_token() -> dict:
    t = json.loads((PENDUKUNG / "token_bench.json").read_text())
    out = {}
    for r in t:
        if r["sel"].endswith("_lampiran"):
            continue
        out[(r["task"], f"{r['metode']}/{r['medium']}")] = r
    return out


# ── utama ───────────────────────────────────────────────────────────────────
def main() -> None:
    import pandas as pd

    data = muat()
    tok = muat_token()
    catatan_verifikasi = []

    # verifikasi: soal identik di dalam tiap tugas
    for tugas, sel in data.items():
        sidik = {s: v["sidik_soal"] for s, v in sel.items()}
        unik = set(sidik.values())
        catatan_verifikasi.append({
            "tugas": tugas, "n_sel": len(sel),
            "soal_identik": len(unik) == 1,
            "n_soal": len(next(iter(unik))),
            "n_variasi_sidik": len(unik),
        })
        assert len(unik) == 1, f"{tugas}: soal tidak identik antar-sel"

    # ── B1 tabel utama ──────────────────────────────────────────────────
    baris = []
    for tugas, sel in data.items():
        for s, v in sorted(sel.items()):
            t = tok.get((tugas, s), {})
            lo, hi = ci_boot_rerata(v["benar"])
            baris.append({
                "tugas": tugas, "tugas_label": LABEL_TUGAS[tugas], "sel": s,
                "metode": s.split("/")[0], "medium": s.split("/")[1],
                "n": v["n"],
                "akurasi": float(v["benar"].mean()),
                "akurasi_ci_lo": lo, "akurasi_ci_hi": hi,
                "format_rate": float(v["format"].mean()),
                "token_keluaran": t.get("token_keluaran"),
                "token_per_soal": t.get("token_per_soal"),
                "token_masukan": t.get("token_masukan"),
                "n_panggilan": t.get("n_panggilan"),
                "waktu_total_s": v["meta"].get("total_time_s"),
                # `detik_per_soal` diturunkan dari metadata bila agregat token
                # tak tersedia. Satu sel (gsm8k raw/baseline) berjalan sebelum
                # pengumpul transkrip dipasang, sehingga hitungan tokennya
                # hilang bersama pod; waktunya tetap tercatat di `_meta`.
                "detik_per_soal": t.get("detik_per_soal",
                                        (v["meta"].get("total_time_s") or 0) / v["n"]
                                        or None),
                "laju_korupsi_token": t.get("laju_korupsi_token"),
                "n_jawaban_ber_cjk": t.get("n_jawaban_ber_cjk"),
                "berkas": v["berkas"],
            })
    df = pd.DataFrame(baris).sort_values(["tugas", "sel"])
    df.to_csv(OUT / "bench_tabel.csv", index=False)

    # ── B2/B3 uji berpasangan + koreksi ─────────────────────────────────
    uji = []
    for tugas, sel in data.items():
        for a, b in itertools.combinations(sorted(sel), 2):
            va, vb = sel[a]["benar"], sel[b]["benar"]
            p, n01, n10 = mcnemar_eksak(va, vb)
            lo, hi = ci_boot_selisih(va, vb)
            fa, fb = sel[a]["format"], sel[b]["format"]
            pf, _, _ = mcnemar_eksak(fa, fb)
            uji.append({
                "tugas": tugas, "tugas_label": LABEL_TUGAS[tugas], "a": a, "b": b,
                "akurasi_a": float(va.mean()), "akurasi_b": float(vb.mean()),
                "delta": float(va.mean() - vb.mean()),
                "ci_lo": lo, "ci_hi": hi,
                "n_a_saja": n01, "n_b_saja": n10,
                "p_mcnemar": p,
                "delta_format": float(fa.mean() - fb.mean()), "p_format": pf,
            })
    holm, bh = koreksi([u["p_mcnemar"] for u in uji])
    for u, h, b_ in zip(uji, holm, bh):
        u["p_holm"], u["p_bh"] = h, b_
    du = pd.DataFrame(uji).sort_values(["tugas", "p_mcnemar"])
    du.to_csv(OUT / "bench_uji.csv", index=False)

    ring: dict = {
        "verifikasi": catatan_verifikasi,
        "n_uji_total": len(uji),
        "n_signifikan_mentah": int((du.p_mcnemar < 0.05).sum()),
        "n_signifikan_holm": int((du.p_holm < 0.05).sum()),
        "n_signifikan_bh": int((du.p_bh < 0.05).sum()),
    }

    # ── B4 Cochran Q lintas 5 mode langkah laten (medium kv) ────────────
    ring["cochran_q"] = []
    for tugas, sel in data.items():
        kol = [f"{m}/kv" for m in KV_MODES if f"{m}/kv" in sel]
        mat = np.column_stack([sel[c]["benar"] for c in kol])
        Q, dfree, p = cochran_q(mat)
        ring["cochran_q"].append({"tugas": tugas, "sel": kol, "Q": Q,
                                  "df": dfree, "p": p})

    # ── B4b kontras keluarga relaksasi vs raw ───────────────────────────
    ring["kontras_keluarga"] = []
    for tugas, sel in data.items():
        raw = sel["raw/kv"]["benar"]
        fam = np.column_stack([sel[f"{m}/kv"]["benar"] for m in RELAKSASI])
        rerata_fam = fam.mean(axis=1)
        d = rerata_fam - raw
        lo, hi = ci_boot_rerata(d)
        # uji berpasangan: raw vs tiap anggota (p sudah ada di tabel uji)
        ring["kontras_keluarga"].append({
            "tugas": tugas,
            "akurasi_raw": float(raw.mean()),
            "akurasi_keluarga_rerata": float(rerata_fam.mean()),
            "delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "per_anggota": {m: float(sel[f"{m}/kv"]["benar"].mean()) for m in RELAKSASI},
        })

    # ── B8 disosiasi antar-tugas ────────────────────────────────────────
    # Δ_t = akurasi keluarga rerata − akurasi raw pada tugas t.
    # Tugas memakai soal berbeda ⇒ perbandingan Δ antar-tugas TAK berpasangan;
    # CI selisihnya dihitung bootstrap independen per tugas.
    delta_boot = {}
    for tugas, sel in data.items():
        raw = sel["raw/kv"]["benar"].astype(float)
        fam = np.column_stack([sel[f"{m}/kv"]["benar"] for m in RELAKSASI]).mean(axis=1)
        d = fam - raw
        idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
        delta_boot[tugas] = d[idx].mean(axis=1)
    ring["disosiasi"] = []
    for t1, t2 in itertools.combinations(sorted(delta_boot), 2):
        sel_b = delta_boot[t1] - delta_boot[t2]
        ring["disosiasi"].append({
            "tugas_a": t1, "tugas_b": t2,
            "delta_a": float(delta_boot[t1].mean()),
            "delta_b": float(delta_boot[t2].mean()),
            "selisih_delta": float(sel_b.mean()),
            "ci_lo": float(np.percentile(sel_b, 2.5)),
            "ci_hi": float(np.percentile(sel_b, 97.5)),
            "p_dua_sisi": float(2 * min((sel_b <= 0).mean(), (sel_b >= 0).mean())),
        })

    # ── B6 efisiensi: kv vs text vs baseline ────────────────────────────
    ring["efisiensi"] = []
    for tugas, sel in data.items():
        t_text = tok.get((tugas, "raw/text"), {})
        t_base = tok.get((tugas, "raw/baseline"), {})
        for m in KV_MODES:
            tk = tok.get((tugas, f"{m}/kv"), {})
            if not tk or not t_text:
                continue
            ring["efisiensi"].append({
                "tugas": tugas, "sel": f"{m}/kv",
                "token_per_soal": tk.get("token_per_soal"),
                "token_per_soal_text": t_text.get("token_per_soal"),
                "token_per_soal_baseline": t_base.get("token_per_soal"),
                "penghematan_token_vs_text": (
                    1 - tk["token_per_soal"] / t_text["token_per_soal"]),
                "percepatan_vs_text": t_text["detik_per_soal"] / tk["detik_per_soal"],
                "penghematan_token_vs_baseline": (
                    1 - tk["token_per_soal"] / t_base["token_per_soal"]) if t_base else None,
            })

    (OUT / "bench_ringkas.json").write_text(json.dumps(ring, indent=2))

    # ── ringkasan layar ─────────────────────────────────────────────────
    print("VERIFIKASI")
    for v in catatan_verifikasi:
        print(f"  {v['tugas']:15s} {v['n_sel']} sel · {v['n_soal']} soal · "
              f"soal identik: {v['soal_identik']}")
    print(f"\nUJI: {ring['n_uji_total']} total · signifikan mentah "
          f"{ring['n_signifikan_mentah']} · Holm {ring['n_signifikan_holm']} · "
          f"BH {ring['n_signifikan_bh']}")
    print("\nCOCHRAN Q (5 mode langkah laten, medium kv)")
    for c in ring["cochran_q"]:
        print(f"  {c['tugas']:15s} Q={c['Q']:.2f} df={c['df']} p={c['p']:.5f}")
    print("\nKONTRAS keluarga relaksasi vs raw")
    for c in ring["kontras_keluarga"]:
        print(f"  {c['tugas']:15s} raw={c['akurasi_raw']:.2f} "
              f"keluarga={c['akurasi_keluarga_rerata']:.3f} "
              f"Δ={c['delta']:+.3f} CI[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]")
    print("\nDISOSIASI (selisih Δ antar-tugas)")
    for d in ring["disosiasi"]:
        print(f"  {d['tugas_a']:14s} vs {d['tugas_b']:14s} "
              f"ΔΔ={d['selisih_delta']:+.3f} "
              f"CI[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] p={d['p_dua_sisi']:.4f}")
    print("\nUJI SIGNIFIKAN setelah Holm")
    for _, r in du[du.p_holm < 0.05].iterrows():
        print(f"  {r.tugas_label:11s} {r.a:16s} vs {r.b:16s} Δ={r.delta:+.3f} "
              f"CI[{r.ci_lo:+.2f},{r.ci_hi:+.2f}] p={r.p_mcnemar:.2e} "
              f"Holm={r.p_holm:.2e}")
    print(f"\n[tulis] {OUT/'bench_tabel.csv'}, {OUT/'bench_uji.csv'}, "
          f"{OUT/'bench_ringkas.json'}")


if __name__ == "__main__":
    main()
