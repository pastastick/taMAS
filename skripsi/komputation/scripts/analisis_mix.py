#!/usr/bin/env python3
"""Sumbu C (`mix`) — separuh KINERJA dari kurva dose-response geometri↔kinerja.

Separuh GEOMETRI-nya sudah ada sejak 2026-08-27 (`b7_probe::geometry_mix`, figur
v08). Separuh kinerjanya baru bisa dihitung setelah 6 sel GPU `mix` selesai
2026-08-28. Skrip ini menyatukan keduanya.

KENAPA TERPISAH DARI `analisis_geometri_kinerja.py`. Skrip itu mengkorelasikan
LIMA MODE (raw/soft/gumbel/sample/moi) terhadap geometrinya — lima titik yang
formulasinya berbeda-beda, jadi apa pun yang terlihat di sana selalu bisa
dijelaskan oleh "formulasinya memang beda", bukan oleh geometri. Sumbu `mix`
memegang formulasinya TETAP (satu rumus, `z(a) = normalisasi((1-a) z_raw + a
z_soft)`) dan hanya menggeser posisinya di ruang laten. Itu yang membuat
BENTUK hubungan bisa dibaca, bukan cuma arahnya.

TIGA HAL YANG HARUS DIBACA HATI-HATI.

1. Hipotesisnya SENGAJA TAK BERARAH. Monoton, ber-ambang, dan tak berpola
   ketiganya temuan yang sah. Yang ketiga berarti klaim mekanistik Bab IV
   ("makin dekat embedding makin baik") harus dilemahkan, bukan dipaksakan.

2. Sumbu-x bukan alpha, melainkan cos TERUKUR. Geometrinya jenuh: alpha
   0->0,25 menaikkan cos 0,140 sedangkan 0,75->1 cuma 0,0025. Memplot terhadap
   alpha membuat separuh kanan sumbu terlihat punya jarak yang sebetulnya tak
   ada.

3. alpha=0,75 adalah titik falsifikasi terkuat. Geometrinya sudah praktis
   sama dengan `soft` (0,9244 vs 0,9269). Kalau geometri memang penjelasnya,
   kinerjanya harus setara `soft`; kalau tidak, ada faktor lain yang bekerja.

Titik ujung TIDAK dijalankan sebagai sel sendiri: alpha=0 identik bit-per-bit
dengan `raw` dan alpha=1 dengan `soft`, jadi sel `kv_raw`/`kv_soft` DIPAKAI
sebagai titik ujung. Itu sah persis karena rumusnya menyatu di kedua ujung.

    python scripts/analisis_mix.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import RESULTS  # noqa: E402
from eval.stats import mcnemar  # noqa: E402

OUT = RESULTS / "pendukung"

# alpha -> (tag sel faktor, tag sel bench HumanEval+)
# Ujung memakai sel raw/soft yang sudah ada: z(0)==z_raw, z(1)==z_soft.
TITIK = [
    (0.00, "kv_raw",      "bench_humanevalplus_raw_kv_s0",       "= raw"),
    (0.25, "kv_mix_a025", "bench_humanevalplus_mix_kv_s0_a025",  ""),
    (0.50, "kv_mix_a05",  "bench_humanevalplus_mix_kv_s0_a05",   ""),
    (0.75, "kv_mix_a075", "bench_humanevalplus_mix_kv_s0_a075",  ""),
    (1.00, "kv_soft",     "bench_humanevalplus_soft_kv_s0",      "= soft"),
]


def _muat(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _ci_binom(k: int, n: int) -> list[float]:
    """Clopper-Pearson 95% — eksak, bukan aproksimasi normal. Pada n=20 dengan
    proporsi di dekat 0 atau 1 (yang justru sering terjadi di sini) aproksimasi
    normal memberi batas di luar [0,1]."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def _spearman(x, y):
    from scipy.stats import spearmanr, pearsonr
    return float(spearmanr(x, y).statistic), float(pearsonr(x, y).statistic)


def sel_faktor(tag: str) -> dict:
    """Agregat satu sel lengan faktor. Unit = JALAN (arah x seed), sesuai
    DESAIN_EKSPERIMEN §4(d); 'lolos gate' = len(run['passing']) > 0, definisi
    yang sama dipakai `kekuatan_uji_faktor.py` supaya angkanya sebanding."""
    doc = _muat(RESULTS / "factor" / f"frontend_{tag}.json")
    if not doc:
        return {}
    runs = doc["runs"]
    lolos = [1.0 if (r.get("passing") or []) else 0.0 for r in runs]
    kunci = [(r.get("direction"), r.get("seed")) for r in runs]

    ekspr, ics, tstats = [], [], []
    for r in runs:
        for f in (r.get("factors") or []):
            if not f.get("expression"):
                continue
            ekspr.append(f)
            if f.get("ic") is not None:
                ics.append(abs(float(f["ic"])))
            if f.get("tstat") is not None:
                tstats.append(abs(float(f["tstat"])))

    n = len(runs)
    k = int(sum(lolos))
    return {
        "tag": tag,
        "n_jalan": n,
        "jalan_lolos_gate": k,
        "laju_lolos_gate": round(k / n, 4) if n else None,
        "ci95_lolos_gate": _ci_binom(k, n) if n else None,
        "n_ekspresi": len(ekspr),
        "n_ekspresi_lolos": sum(len(r.get("passing") or []) for r in runs),
        "ekspresi_per_jalan": round(len(ekspr) / n, 2) if n else None,
        "percobaan_construct_per_jalan": round(
            sum(1 + int(r.get("repair_attempts") or 0) for r in runs) / n, 2) if n else None,
        "jalan_perlu_repair": sum(1 for r in runs if (r.get("repair_attempts") or 0) > 0),
        "detik_per_jalan": round(statistics.fmean(r["duration_s"] for r in runs), 1) if n else None,
        "token_keluar_per_jalan": round(statistics.fmean(
            sum(h.get("n_out_tok", 0) for h in (r.get("agent_trace") or [])) for r in runs), 1) if n else None,
        "ic_n_berdert": len(ics),
        "ic_mean_abs": round(statistics.fmean(ics), 5) if ics else None,
        "ic_median_abs": round(statistics.median(ics), 5) if ics else None,
        "ic_n_tstat_ge2": sum(1 for t in tstats if t >= 2.0),
        "ic_laju_tstat_ge2": round(sum(1 for t in tstats if t >= 2.0) / len(tstats), 4) if tstats else None,
        "_lolos": lolos,
        "_kunci": kunci,
    }


def sel_bench(nama: str) -> dict:
    """Agregat + vektor benar/salah PER SOAL. Vektor itu yang membuat lengan
    bench bisa diuji berpasangan: kelima sel menjawab 100 soal HumanEval+ yang
    sama pada seed sampel yang sama, jadi soal ke-i sebanding lintas sel."""
    d = _muat(RESULTS / "bench" / f"{nama}.json")
    if not d:
        return {}
    s = d.get("summary", d)
    per_soal = {r["index"]: (1.0 if r.get("correct") else 0.0)
                for r in (d.get("results") or []) if "index" in r}
    return {
        "bench": nama,
        "n": s.get("n"),
        "akurasi": s.get("accuracy"),
        "format": s.get("format_rate"),
        "detik": round(d["_meta"]["total_time_s"], 1) if d.get("_meta", {}).get("total_time_s") else None,
        "_per_soal": per_soal,
        **_korupsi(nama),
    }


def _korupsi(nama: str) -> dict:
    """Korupsi CJK dari `token_bench.json` — aksara Han di jawaban berbahasa
    Inggris. Ini fidelitas simbolik yang bisa dilihat mata: vektor laten yang
    tidak mendarat di manifold embedding nyata sesekali men-decode token dari
    bahasa lain. Nol pada medium tanpa langkah laten."""
    p = OUT / "token_bench.json"
    if not p.exists():
        return {}
    kunci = nama[len("bench_"):] if nama.startswith("bench_") else nama
    for e in json.loads(p.read_text()):
        if e.get("sel") == kunci:
            return {
                "n_jawaban_ber_cjk": e.get("n_jawaban_ber_cjk"),
                "laju_korupsi_token": e.get("laju_korupsi_token"),
                "contoh_korupsi": e.get("contoh_korupsi"),
                "token_per_soal": e.get("token_per_soal"),
            }
    return {}


def berpasangan_bench(a: dict, b: dict) -> dict:
    """McNemar eksak pada lengan bench, dipasangkan lewat indeks soal."""
    pa, pb = a.get("_per_soal") or {}, b.get("_per_soal") or {}
    umum = sorted(set(pa) & set(pb))
    if not umum:
        return {"n_pasang": 0}
    va = [pa[i] for i in umum]
    vb = [pb[i] for i in umum]
    p, b01, b10 = mcnemar(va, vb)
    return {
        "n_pasang": len(umum),
        "mean_a": round(statistics.fmean(va), 4),
        "mean_b": round(statistics.fmean(vb), 4),
        "delta": round(statistics.fmean(va) - statistics.fmean(vb), 4),
        "b01": b01, "b10": b10,
        "p_mcnemar_exact": p,
        "signifikan_alpha05": bool(p < 0.05),
    }


def berpasangan(a: dict, b: dict) -> dict:
    """McNemar berpasangan lewat (arah, seed) yang sama di kedua sel. Sel mix
    memakai arah & seed yang identik dengan raw/soft menurut rancangan, jadi
    pasangannya benar-benar 1:1 dan bukan sekadar dua sampel berukuran sama."""
    pa = dict(zip(a["_kunci"], a["_lolos"]))
    pb = dict(zip(b["_kunci"], b["_lolos"]))
    umum = [k for k in pa if k in pb]
    if not umum:
        return {"n_pasang": 0}
    va = [pa[k] for k in umum]
    vb = [pb[k] for k in umum]
    p, b01, b10 = mcnemar(va, vb)
    return {
        "n_pasang": len(umum),
        "mean_a": round(statistics.fmean(va), 4),
        "mean_b": round(statistics.fmean(vb), 4),
        "delta": round(statistics.fmean(va) - statistics.fmean(vb), 4),
        "b01": b01, "b10": b10,
        "p_mcnemar_exact": p,
        "signifikan_alpha05": bool(p < 0.05),
    }


def main() -> None:
    geo = (_muat(RESULTS / "probe" / "b7_probe_Qwen_Qwen3-8B.json") or {}).get("geometry_mix", {})
    if not geo:
        print("geometry_mix tak ada di b7_probe — sumbu-x tak bisa dibaca", file=sys.stderr)
        raise SystemExit(1)

    baris = []
    for alpha, tag_f, tag_b, catatan in TITIK:
        kunci_geo = next((k for k in geo if abs(float(k) - alpha) < 1e-9), None)
        g = geo.get(kunci_geo, {})
        f = sel_faktor(tag_f)
        baris.append({
            "alpha": alpha,
            "catatan": catatan,
            "cos_embed_mean": g.get("max_cos_embed_mean"),
            "cos_embed_min": g.get("max_cos_embed_min"),
            "cos_embed_max": g.get("max_cos_embed_max"),
            "faktor": f,
            "bench_humanevalplus": sel_bench(tag_b),
        })

    # ── uji berpasangan terhadap kedua ujung ────────────────────────────────
    raw, soft = baris[0]["faktor"], baris[-1]["faktor"]
    braw, bsoft = baris[0]["bench_humanevalplus"], baris[-1]["bench_humanevalplus"]
    for b in baris:
        f, be = b["faktor"], b["bench_humanevalplus"]
        b["vs_raw"] = berpasangan(f, raw) if f and f["tag"] != raw["tag"] else None
        b["vs_soft"] = berpasangan(f, soft) if f and f["tag"] != soft["tag"] else None
        b["bench_vs_raw"] = (berpasangan_bench(be, braw)
                             if be and be["bench"] != braw["bench"] else None)
        b["bench_vs_soft"] = (berpasangan_bench(be, bsoft)
                              if be and be["bench"] != bsoft["bench"] else None)

    # ── bentuk kurva ────────────────────────────────────────────────────────
    alphas = [b["alpha"] for b in baris]
    cosx = [b["cos_embed_mean"] for b in baris]
    metrik = {
        "laju_lolos_gate": [b["faktor"].get("laju_lolos_gate") for b in baris],
        "akurasi_humanevalplus": [b["bench_humanevalplus"].get("akurasi") for b in baris],
        "ic_mean_abs": [b["faktor"].get("ic_mean_abs") for b in baris],
        "ekspresi_per_jalan": [b["faktor"].get("ekspresi_per_jalan") for b in baris],
        "korupsi_cjk_humanevalplus": [b["bench_humanevalplus"].get("n_jawaban_ber_cjk") for b in baris],
    }
    bentuk = {}
    for nama, ys in metrik.items():
        if any(y is None for y in ys):
            continue
        rho_a, r_a = _spearman(alphas, ys)
        rho_c, r_c = _spearman(cosx, ys)
        selisih = [round(ys[i + 1] - ys[i], 4) for i in range(len(ys) - 1)]
        naik = all(d >= 0 for d in selisih)
        turun = all(d <= 0 for d in selisih)
        imax = max(range(len(selisih)), key=lambda i: abs(selisih[i]))
        bentuk[nama] = {
            "nilai": ys,
            "spearman_vs_alpha": round(rho_a, 4), "pearson_vs_alpha": round(r_a, 4),
            "spearman_vs_cos": round(rho_c, 4), "pearson_vs_cos": round(r_c, 4),
            "selisih_antar_titik": selisih,
            "monoton": bool(naik or turun),
            "lompatan_terbesar": {
                "dari_alpha": alphas[imax], "ke_alpha": alphas[imax + 1],
                "dari_cos": cosx[imax], "ke_cos": cosx[imax + 1],
                "delta": selisih[imax],
                "bagian_dari_rentang_total": round(
                    abs(selisih[imax]) / (max(ys) - min(ys)), 4) if max(ys) > min(ys) else None,
            },
        }

    hasil = {
        "_meta": {
            "catatan": "Sumbu C (mix). Titik ujung alpha=0/1 memakai sel kv_raw/kv_soft "
                       "(z(0) identik z_raw, z(1) identik z_soft) — bukan sel terpisah. "
                       "Hipotesis TAK BERARAH: monoton / ber-ambang / tak berpola "
                       "ketiganya temuan yang sah.",
            "sumbu_x": "cos ke embedding terdekat (TERUKUR, b7_probe::geometry_mix) — "
                       "bukan alpha; geometrinya jenuh sehingga alpha bukan skala linier",
            "unit_analisis_faktor": "satu jalan (arah x seed); lolos gate = len(passing)>0",
        },
        "titik": baris,
        "bentuk_kurva": bentuk,
    }
    for b in baris:  # buang lampiran kerja sebelum ditulis
        b["faktor"].pop("_lolos", None)
        b["faktor"].pop("_kunci", None)
        b["bench_humanevalplus"].pop("_per_soal", None)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sumbu_mix.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False))

    # ── ringkasan terbaca ───────────────────────────────────────────────────
    L = ["# Sumbu C (`mix`) — geometri vs kinerja, formulasi DIPEGANG TETAP", "",
         "Regenerasi: `python scripts/analisis_mix.py`", "",
         "Titik ujung memakai sel `kv_raw`/`kv_soft`: z(alpha=0) identik dengan",
         "z_raw dan z(alpha=1) dengan z_soft, jadi menjalankannya lagi sebagai sel",
         "sendiri hanya akan menghasilkan angka yang sama.", "",
         "| alpha | cos terukur | lolos gate (faktor) | ekspr/jalan | mean abs IC | HumanEval+ | jwb ber-CJK | token/jalan |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b in baris:
        f, be = b["faktor"], b["bench_humanevalplus"]
        lg = f.get("laju_lolos_gate")
        L.append(
            f"| {b['alpha']:.2f} {b['catatan']} | {b['cos_embed_mean']:.4f} | "
            f"{lg:.0%} ({f['jalan_lolos_gate']}/{f['n_jalan']}) | "
            f"{f.get('ekspresi_per_jalan')} | {f.get('ic_mean_abs')} | "
            f"{be.get('akurasi')} | {be.get('n_jawaban_ber_cjk')} | "
            f"{f.get('token_keluar_per_jalan')} |")

    L += ["", "## Bentuk kurva", ""]
    for nama, d in bentuk.items():
        L.append(f"### {nama}")
        L.append(f"- nilai: {d['nilai']}")
        L.append(f"- selisih antar titik: {d['selisih_antar_titik']}")
        L.append(f"- **monoton: {'YA' if d['monoton'] else 'TIDAK'}**")
        L.append(f"- Spearman vs alpha {d['spearman_vs_alpha']}, vs cos terukur {d['spearman_vs_cos']}")
        j = d["lompatan_terbesar"]
        L.append(f"- lompatan terbesar {j['dari_alpha']}→{j['ke_alpha']} "
                 f"(cos {j['dari_cos']}→{j['ke_cos']}): {j['delta']:+} "
                 f"= {j['bagian_dari_rentang_total']:.0%} dari seluruh rentang")
        L.append("")

    L += ["## Uji berpasangan lolos gate (McNemar eksak, dipasangkan lewat arah x seed)", "",
          "| titik | vs raw (alpha=0) | vs soft (alpha=1) |", "|---|---|---|"]
    for b in baris:
        def fmt(u):
            if not u or not u.get("n_pasang"):
                return "—"
            return (f"delta={u['delta']:+.2f}, b01={u['b01']}/b10={u['b10']}, "
                    f"p={u['p_mcnemar_exact']:.4g}{' *' if u['signifikan_alpha05'] else ''}")
        L.append(f"| alpha={b['alpha']:.2f} | {fmt(b['vs_raw'])} | {fmt(b['vs_soft'])} |")
    L += ["", "`*` = signifikan pada alpha 0,05. Titik falsifikasi: alpha=0,75 secara",
          "geometri sudah praktis sama dengan `soft` (cos 0,9244 vs 0,9269); kalau",
          "geometri memang penjelasnya, kolom kanannya harus TIDAK signifikan.", ""]

    L += ["## Uji berpasangan HumanEval+ (McNemar eksak, dipasangkan lewat indeks soal)", "",
          "n=100 soal yang sama di kelima sel, seed sampel sama.", "",
          "| titik | vs raw (alpha=0) | vs soft (alpha=1) |", "|---|---|---|"]
    for b in baris:
        def fmtb(u):
            if not u or not u.get("n_pasang"):
                return "—"
            return (f"delta={u['delta']:+.2f}, b01={u['b01']}/b10={u['b10']}, "
                    f"p={u['p_mcnemar_exact']:.4g}{' *' if u['signifikan_alpha05'] else ''}")
        L.append(f"| alpha={b['alpha']:.2f} | {fmtb(b['bench_vs_raw'])} | {fmtb(b['bench_vs_soft'])} |")
    L.append("")

    (OUT / "sumbu_mix.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nditulis → {OUT}/sumbu_mix.json + sumbu_mix.md")


if __name__ == "__main__":
    main()
