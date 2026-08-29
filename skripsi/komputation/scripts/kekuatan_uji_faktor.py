#!/usr/bin/env python3
"""Verifikasi tabel daya TEORI.md §4.6 + jalankan SATU uji formal yang berdaya.

DUA bagian, tak bergantung satu sama lain:

(A) VERIFIKASI TABEL DAYA — murni matematis, tak butuh data run apa pun.
    TEORI.md §4.6 mengklaim daya McNemar eksak dua sisi pada alpha=0,05 untuk
    n in {6,20,40,100} dan selisih proporsi d in {0.15,0.25,0.35,0.50}, pada
    "skenario paling menguntungkan" (seluruh ketaksesuaian searah — b10=0).
    Dihitung ulang di sini lewat simulasi eksak (bukan aproksimasi normal)
    supaya klaim §4.6 bisa dicek, bukan dipercaya begitu saja.

(B) SATU UJI FORMAL LENGAN FAKTOR — kebijakan §4.6: "pada lengan faktor, SATU
    kontras diuji secara formal — keluarga R terhadap raw — sementara seluruh
    sisanya dilaporkan deskriptif". Skrip ini menjalankannya, dengan DUA cara
    yang saling melengkapi:
      B1. per-anggota vs raw, BERPASANGAN lewat (direction, seed) yang sama
          persis di kedua mode (`backend/eval/stats.py::mcnemar`, satu
          implementasi dipakai kedua lengan) — diagnostik, bukan uji utama,
          karena tiap anggota diuji terpisah (4 uji, bukan 1).
      B2. keluarga R DIKUMPULKAN jadi satu grup (4 mode x n jalan) melawan
          raw (n jalan) — uji proporsi tak-berpasangan (Fisher eksak) karena
          ukuran grupnya berbeda (4n vs n) sehingga TIDAK bisa dipasangkan
          1:1. Inilah "satu kontras formal" yang dimaksud kebijakan §4.6.

Unit analisis: SATU TRAJECTORY (satu arah x seed), bukan satu ekspresi —
sesuai DESAIN_EKSPERIMEN.md §4(d). "lolos gate" = `len(run['passing']) > 0`.
Field ini terisi begitu fase GPU selesai, TIDAK menunggu skoring CPU (ic/bt_*)
— jadi bagian (B) bisa dijalankan segera setelah sel GPU sebuah mode selesai.

    python scripts/kekuatan_uji_faktor.py              # (A) + (B) untuk comm-mode kv
    python scripts/kekuatan_uji_faktor.py --comm-mode kv_and_text
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from paths import RESULTS  # noqa: E402
from eval.stats import mcnemar, paired_report  # noqa: E402

R_FAMILY = ["soft", "sample", "gumbel", "moi"]
ALPHA = 0.05


# ── (A) verifikasi tabel daya ────────────────────────────────────────────────
def _mcnemar_power_exact(n: int, d: float, alpha: float = ALPHA) -> float:
    """Daya McNemar eksak, skenario TEORI §4.6: seluruh ketaksesuaian searah
    (b10=0, b01 ~ Binomial(n, d)) — 'skenario paling menguntungkan' persis
    seperti disebutkan dokumen. p dihitung persis seperti `stats.mcnemar`
    (binomial dua sisi terhadap b01 vs n-b01), lalu dirata-ratakan atas
    distribusi b01 untuk mendapat daya."""
    power = 0.0
    for b01 in range(n + 1):
        prob_b01 = comb(n, b01) * (d ** b01) * ((1 - d) ** (n - b01))
        if prob_b01 <= 0:
            continue
        # p-value McNemar utk (b01, b10=0), n_diskordan = b01
        if b01 == 0:
            p = 1.0
        else:
            lo = 0  # min(b01, 0)
            p = min(1.0, sum(comb(b01, i) for i in range(lo + 1)) / (2 ** b01) * 2)
        if p < alpha:
            power += prob_b01
    return power


def verifikasi_tabel_daya() -> dict:
    ns = [6, 20, 40, 100]
    ds = [0.15, 0.25, 0.35, 0.50]
    klaim = {  # disalin persis dari TEORI.md §4.6
        6:   {0.15: 0.000, 0.25: 0.000, 0.35: 0.002, 0.50: 0.016},
        20:  {0.15: 0.067, 0.25: 0.383, 0.35: 0.755, 0.50: 0.979},
        40:  {0.15: 0.567, 0.25: 0.957, 0.35: 0.999, 0.50: 1.000},
        100: {0.15: 0.998, 0.25: 1.000, 0.35: 1.000, 0.50: 1.000},
    }
    out = {}
    print("=== (A) Verifikasi tabel daya TEORI.md §4.6 ===")
    print(f"{'n':>4} {'d':>5} {'klaim':>7} {'dihitung':>9} {'selisih':>8}")
    semua_cocok = True
    for n in ns:
        for d in ds:
            hitung = round(_mcnemar_power_exact(n, d), 3)
            k = klaim[n][d]
            sel = round(hitung - k, 3)
            cocok = abs(sel) <= 0.005
            semua_cocok &= cocok
            tanda = "" if cocok else "  <-- BEDA"
            print(f"{n:>4} {d:>5.2f} {k:>7.3f} {hitung:>9.3f} {sel:>+8.3f}{tanda}")
            out.setdefault(str(n), {})[str(d)] = {"klaim": k, "dihitung": hitung}
    print(f"\n[{'SEMUA COCOK' if semua_cocok else 'ADA SELISIH'} — toleransi ±0.005]")
    return {"cocok": semua_cocok, "tabel": out}


# ── (B) uji formal lengan faktor ─────────────────────────────────────────────
def _muat(comm_mode: str, mode: str) -> dict | None:
    tag = "text" if mode == "raw" and comm_mode == "text" else f"{comm_mode}_{mode}"
    fp = RESULTS / "factor" / f"frontend_{tag}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def _gate_pass_by_traj(d: dict) -> dict[tuple, int]:
    out = {}
    for r in d.get("runs", []):
        key = (r.get("direction"), r.get("seed"))
        out[key] = 1 if (r.get("passing") or []) else 0
    return out


def _fisher_exact_2x2(a_pass: int, a_n: int, b_pass: int, b_n: int) -> float:
    """Fisher eksak dua-sisi via penjumlahan hipergeometrik langsung (kecil,
    n<=200 di sini — tak perlu scipy)."""
    a_fail, b_fail = a_n - a_pass, b_n - b_pass
    total_pass, total_n = a_pass + b_pass, a_n + b_n

    def _p(k):  # P(a_pass = k | margin tetap) — hipergeometrik
        if k < 0 or k > a_n or (total_pass - k) < 0 or (total_pass - k) > b_n:
            return 0.0
        return (comb(a_n, k) * comb(b_n, total_pass - k)) / comb(total_n, total_pass)

    p_obs = _p(a_pass)
    return sum(_p(k) for k in range(0, a_n + 1) if _p(k) <= p_obs + 1e-12)


def _cohens_h(p1: float, p2: float) -> float:
    from math import asin, sqrt
    return 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))


def _boot_ci_unpaired(a: list[int], b: list[int], iters: int = 20000) -> tuple[float, float]:
    rng = random.Random(0)
    na, nb = len(a), len(b)
    diffs = sorted(
        (sum(rng.choice(a) for _ in range(na)) / na
         - sum(rng.choice(b) for _ in range(nb)) / nb)
        for _ in range(iters))
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def uji_formal(comm_mode: str) -> dict:
    print(f"\n=== (B) Uji formal lengan faktor — comm_mode={comm_mode} ===")
    raw_d = _muat(comm_mode, "raw")
    if raw_d is None:
        print(f"[lewati] frontend_{comm_mode}_raw.json belum ada — sel GPU belum selesai.")
        return {"status": "raw belum ada"}
    raw_gate = _gate_pass_by_traj(raw_d)
    print(f"raw: {len(raw_gate)} trajectory, {sum(raw_gate.values())} lolos gate "
          f"({100*sum(raw_gate.values())/max(1,len(raw_gate)):.0f}%)")

    hasil = {"comm_mode": comm_mode, "raw": {
        "n": len(raw_gate), "lolos": sum(raw_gate.values())}}

    # B1: tiap anggota R vs raw, berpasangan per (direction, seed)
    diag = {}
    r_pooled_pass, r_pooled_n = 0, 0
    anggota_siap = []
    for m in R_FAMILY:
        d = _muat(comm_mode, m)
        if d is None:
            print(f"  [lewati] {m}: belum ada")
            continue
        gate = _gate_pass_by_traj(d)
        keys = sorted(set(gate) & set(raw_gate))
        if not keys:
            continue
        a = [gate[k] for k in keys]     # anggota R
        b = [raw_gate[k] for k in keys]  # raw, dipasangkan
        rep = paired_report(a, b, binary=True)
        n_pass = sum(a)
        print(f"  {m:8s} vs raw (n={len(keys)}, pasangan by direction+seed): "
              f"{100*n_pass/len(keys):.0f}% vs {100*sum(b)/len(keys):.0f}%  "
              f"p={rep['p']:.4f}  delta={rep['delta']:+.3f}  ci95={rep['ci95']}")
        diag[m] = rep
        r_pooled_pass += n_pass
        r_pooled_n += len(keys)
        anggota_siap.append(m)
    hasil["diagnostik_per_anggota_vs_raw"] = diag

    if not anggota_siap:
        print("  [B2 dilewati] tak ada anggota R yang siap.")
        hasil["uji_formal_keluarga_R_vs_raw"] = {"status": "belum ada data R"}
        return hasil

    # B2: keluarga R dikumpulkan (unpaired) vs raw — SATU kontras formal §4.6
    raw_pass, raw_n = sum(raw_gate.values()), len(raw_gate)
    p_fisher = _fisher_exact_2x2(r_pooled_pass, r_pooled_n, raw_pass, raw_n)
    p_r, p_raw = r_pooled_pass / r_pooled_n, raw_pass / raw_n
    h = _cohens_h(p_r, p_raw)
    ci = _boot_ci_unpaired([1] * r_pooled_pass + [0] * (r_pooled_n - r_pooled_pass),
                           [1] * raw_pass + [0] * (raw_n - raw_pass))
    lengkap = len(anggota_siap) == len(R_FAMILY)
    print(f"\n  >>> UJI FORMAL (kebijakan TEORI §4.6): keluarga R "
          f"({'LENGKAP' if lengkap else 'SEBAGIAN: ' + ','.join(anggota_siap)}, "
          f"n={r_pooled_n}) vs raw (n={raw_n})")
    print(f"      lolos gate: R={100*p_r:.1f}%  raw={100*p_raw:.1f}%  "
          f"delta={100*(p_r-p_raw):+.1f}pp")
    print(f"      Fisher eksak dua-sisi p={p_fisher:.6f}  Cohen's h={h:+.3f}  "
          f"CI95(diff)={[round(x,3) for x in ci]}")
    hasil["uji_formal_keluarga_R_vs_raw"] = {
        "lengkap": lengkap, "anggota": anggota_siap,
        "r_pass": r_pooled_pass, "r_n": r_pooled_n, "p_r": p_r,
        "raw_pass": raw_pass, "raw_n": raw_n, "p_raw": p_raw,
        "delta_pp": 100 * (p_r - p_raw), "p_fisher_exact": p_fisher,
        "cohens_h": h, "ci95_diff": list(ci),
        "signifikan_alpha05": p_fisher < ALPHA,
    }
    return hasil


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comm-mode", default="kv", choices=["kv", "kv_and_text"])
    ap.add_argument("--skip-power-table", action="store_true")
    args = ap.parse_args()

    out = {}
    if not args.skip_power_table:
        out["verifikasi_tabel_daya"] = verifikasi_tabel_daya()
    out["uji_formal"] = uji_formal(args.comm_mode)

    OUT = RESULTS / "pendukung" / f"uji_formal_faktor_{args.comm_mode}.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[tersimpan] {OUT}")


if __name__ == "__main__":
    main()
