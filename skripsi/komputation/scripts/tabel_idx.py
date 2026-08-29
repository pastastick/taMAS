#!/usr/bin/env python3
"""Bangkitkan tabel LaTeX bab hasil dari `results/idx/analisis_idx.json`.

Semua tabel bab hasil berasal dari SATU berkas JSON, sehingga angka di
naskah tidak bisa menyimpang dari angka di hasil. Keluaran ditulis ke
`results/idx/tabel/*.tex`; salin ke `skripsi/assets/tables/` saat menyusun
naskah.

Konvensi angka: desimal memakai KOMA (kaidah bahasa Indonesia), ribuan memakai
titik tipis LaTeX (\,).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from paths import bootstrap, RESULTS  # noqa: E402
bootstrap()

IDX = RESULTS / "idx"
TAB = IDX / "tabel"
BENCH = RESULTS / "bench" / "analisis.json"

URUT = ["raw", "soft", "gumbel", "sample", "moi"]
LBL_MEDIUM = {"text": "teks", "kv": "kv", "kv_and_text": "kv+teks"}
LBL_METODE = {"raw": r"\code{raw}", "soft": r"\code{soft}", "gumbel": r"\code{gumbel}",
              "sample": r"\code{sample}", "moi": r"\code{moi}", "-": "---",
              "mix_a025": r"\code{mix} $\alpha$=0,25",
              "mix_a05": r"\code{mix} $\alpha$=0,50",
              "mix_a075": r"\code{mix} $\alpha$=0,75"}


def k(x, n=2, kosong="---"):
    if x is None:
        return kosong
    return f"{x:.{n}f}".replace(".", ",")


def pers(x, n=1, kosong="---"):
    if x is None:
        return kosong
    return f"{100*x:.{n}f}".replace(".", ",") + r"\%"


def ilm(p):
    """Nilai-p dalam notasi ilmiah LaTeX."""
    if p is None:
        return "---"
    if p >= 0.001:
        return f"{p:.3f}".replace(".", ",")
    m, e = f"{p:.2e}".split("e")
    return f"${m.replace('.', ',')}\\times10^{{{int(e)}}}$"


def urutkan(tags):
    prio = {"text": 0, "kv": 1, "kv_and_text": 2}
    def key(t):
        for m in ("kv_and_text", "kv", "text"):
            if t == m:
                return (prio[m], -1)
            if t.startswith(m + "_"):
                sisa = t[len(m) + 1:]
                return (prio[m], URUT.index(sisa) if sisa in URUT else 9 + len(sisa))
        return (9, 9)
    return sorted(tags, key=key)


def belah(tag):
    for m in ("kv_and_text", "kv", "text"):
        if tag == m:
            return m, "-"
        if tag.startswith(m + "_"):
            return m, tag[len(m) + 1:]
    return tag, "-"


# ── T1: keandalan & biaya generasi ──────────────────────────────────────────
def t_keandalan(d):
    g = d["generasi"]
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Keandalan dan biaya lengan faktor per JALAN. Satu jalan = "
         r"satu pasangan (arah, \textit{seed}); denominatornya sama di seluruh "
         r"sel menurut rancangan, sehingga laju antar-sel sebanding. Kolom "
         r"terakhir mencatat berapa kali agen \textit{construct} harus dipanggil "
         r"per jalan; nilai di atas 1 berarti \textit{gate} memicu perbaikan, "
         r"sehingga kolom itu sekaligus ukuran biaya dan ukuran keandalan.}",
         r"\label{tab:keandalan}", r"\small",
         r"\begin{tabular}{llrrrrr}", r"\hline",
         r"Medium & Formulasi & Jalan & Lolos \textit{gate} & Ekspresi & "
         r"Token/jalan & Detik/jalan \\", r"\hline"]
    med_lalu = None
    for t in urutkan(g):
        m, mt = belah(t)
        v = g[t]
        if med_lalu is not None and m != med_lalu:
            b.append(r"\hline")
        med_lalu = m
        b.append(f"{LBL_MEDIUM.get(m, m)} & {LBL_METODE.get(mt, mt)} & "
                 f"{v['jalan']} & {pers(v['laju_lolos_gate'])} & "
                 f"{v['ekspresi_total']} & {v['token_keluar_per_jalan']:.0f} & "
                 f"{v['detik_per_jalan']:.0f} \\\\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── T2: mutu sinyal per sel, dua jendela ────────────────────────────────────
def t_mutu(d):
    s, h = d["seleksi_2021"], d["holdout_2022_2025"]
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Mutu sinyal ekspresi pada universe LQ45. ``Hidup'' berarti "
         r"IC terdefinisi dan nilainya tidak konstan; ``sig.'' berarti "
         r"$\lvert t\rvert\ge1{,}96$. Ambang deteksi berbeda antar-jendela "
         r"(Tabel~\ref{tab:ambang}), sehingga kolom sig. TIDAK sebanding "
         r"langsung antar-jendela.}",
         r"\label{tab:mutu}", r"\small",
         r"\begin{tabular}{llrrrrrr}", r"\hline",
         r" & & \multicolumn{3}{c}{Seleksi 2021 ($T$=247)} & "
         r"\multicolumn{3}{c}{Holdout 2022--2025 ($T$=958)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Medium & Formulasi & Hidup & Sig. & $\overline{|IC|}$ & "
         r"Hidup & Sig. & $\overline{|IC|}$ \\", r"\hline"]
    med_lalu = None
    for t in urutkan([x for x in s if x != "_meta"]):
        m, mt = belah(t)
        a, c = s[t], h.get(t, {})
        if med_lalu is not None and m != med_lalu:
            b.append(r"\hline")
        med_lalu = m
        b.append(f"{LBL_MEDIUM.get(m, m)} & {LBL_METODE.get(mt, mt)} & "
                 f"{a['hidup']} & {a['signifikan']} & {k(a['mean_abs_ic'], 4)} & "
                 f"{c.get('hidup','---')} & {c.get('signifikan','---')} & "
                 f"{k(c.get('mean_abs_ic'), 4)} \\\\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── T3: metrik backtest ─────────────────────────────────────────────────────
def t_backtest(d):
    s = d["seleksi_2021"]
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Metrik portofolio \textit{long--short} kuintil (7 emiten per "
         r"sisi), jendela seleksi 2021, median lintas-ekspresi per sel. Angka "
         r"bersifat DESKRIPTIF: pada universe 37 emiten galat baku imbal hasil "
         r"tahunan mencapai 19,9 poin persen (Persamaan~\eqref{eq:derauportofolio}), "
         r"sehingga tabel ini tidak dipakai menyimpulkan formulasi mana yang "
         r"unggul.}",
         r"\label{tab:backtest}", r"\small",
         r"\begin{tabular}{llrrrrr}", r"\hline",
         r"Medium & Formulasi & Sharpe & Imbal tahunan & \textit{Max DD} & "
         r"Perputaran & \textit{Hit rate} \\", r"\hline"]
    med_lalu = None
    for t in urutkan([x for x in s if x != "_meta"]):
        m, mt = belah(t)
        v = s[t]
        if med_lalu is not None and m != med_lalu:
            b.append(r"\hline")
        med_lalu = m
        b.append(f"{LBL_MEDIUM.get(m, m)} & {LBL_METODE.get(mt, mt)} & "
                 f"{k(v['median_sharpe'])} & {k(v['median_ann_return'], 3)} & "
                 f"{k(v['median_max_drawdown'], 3)} & {k(v['median_turnover'], 3)} & "
                 f"{k(v['median_hit_rate'], 3)} \\\\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── T4: uji formal ──────────────────────────────────────────────────────────
def t_uji(d):
    u = d["uji_formal"]
    ms = d.get("uji_mutu_sinyal_seleksi", {})
    mh = d.get("uji_mutu_sinyal_holdout", {})
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Uji formal keluarga $\mathcal{R}$ melawan \code{raw}. Dua "
         r"baris pertama tiap medium menguji VALIDITAS keluaran (uji eksak "
         r"Fisher, ukuran efek $h$ Cohen); dua baris berikutnya menguji MUTU "
         r"SINYAL (uji Mann--Whitney atas sebaran $\lvert IC\rvert$, ukuran "
         r"efek \textit{rank-biserial}). Perbedaan arah kesimpulan antara "
         r"kedua kelompok baris inilah temuan utama penelitian.}",
         r"\label{tab:uji}", r"\small",
         r"\begin{tabular}{lllrrl}", r"\hline",
         r"Medium & Ukuran & $\mathcal{R}$ vs \code{raw} & Selisih & "
         r"Ukuran efek & Nilai-$p$ \\", r"\hline"]
    for medium, blok in u.items():
        lm = LBL_MEDIUM.get(medium, medium)
        g = blok["lolos_gate_per_jalan"]
        b.append(f"{lm} & lolos \\textit{{gate}}/jalan & "
                 f"{pers(g['p_R'])} vs {pers(g['p_raw'])} & "
                 f"{k(g['selisih_pp'],1)} pp & $h$={k(g['cohen_h'],3)} & "
                 f"{ilm(g['fisher_p'])} \\\\")
        if "ekspresi_hidup" in blok:
            e = blok["ekspresi_hidup"]
            b.append(f" & ekspresi hidup & {pers(e['p_R'])} vs {pers(e['p_raw'])} & "
                     f"{k(e['selisih_pp'],1)} pp & $h$={k(e['cohen_h'],3)} & "
                     f"{ilm(e['fisher_p'])} \\\\")
        if medium in ms:
            v = ms[medium]
            b.append(f" & median $|IC|$ 2021 & {k(v['median_abs_ic_R'],4)} vs "
                     f"{k(v['median_abs_ic_raw'],4)} & --- & "
                     f"$r_{{rb}}$={k(v['rank_biserial'],3)} & {ilm(v['mannwhitney_p'])} \\\\")
        if medium in mh:
            v = mh[medium]
            b.append(f" & median $|IC|$ holdout & {k(v['median_abs_ic_R'],4)} vs "
                     f"{k(v['median_abs_ic_raw'],4)} & --- & "
                     f"$r_{{rb}}$={k(v['rank_biserial'],3)} & {ilm(v['mannwhitney_p'])} \\\\")
        b.append(r"\hline")
    b += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── T5: lengan benchmark ────────────────────────────────────────────────────
def t_bench():
    if not BENCH.exists():
        return None
    d = json.loads(BENCH.read_text())
    besar = [g for g in d["groups"] if g["n_items"] >= 50]
    tugas = [g["task"] for g in besar]
    lbl = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}
    sel = {}
    for g in besar:
        for c in g["cells"]:
            sel.setdefault(c["cell"], {})[g["task"]] = c
    urut_sel = ["raw/baseline/m10", "raw/text/m10", "raw/kv/m10", "soft/kv/m10",
                "gumbel/kv/m10", "sample/kv/m10", "moi/kv/m10"]
    nama = {"raw/baseline/m10": r"\code{raw} tanpa agen (garis dasar)",
            "raw/text/m10": r"\code{raw} / teks",
            "raw/kv/m10": r"\code{raw} / kv", "soft/kv/m10": r"\code{soft} / kv",
            "gumbel/kv/m10": r"\code{gumbel} / kv", "sample/kv/m10": r"\code{sample} / kv",
            "moi/kv/m10": r"\code{moi} / kv"}
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Lengan penalaran umum: akurasi pada 100 butir per tolok ukur "
         r"($\textit{seed}$ sampel identik antar-sel, diverifikasi lewat "
         r"\textit{fingerprint} data). Kolom waktu adalah total detik untuk 100 "
         r"butir. Perhatikan pola kolomnya: selisih \code{raw}/kv terhadap "
         r"keluarga $\mathcal{R}$ kecil pada ARC-C, sedang pada GSM8K, dan "
         r"besar pada HumanEval+.}",
         r"\label{tab:bench}", r"\small",
         r"\begin{tabular}{l" + "r" * (len(tugas) + 1) + "}", r"\hline",
         "Sel & " + " & ".join(lbl.get(t, t) for t in tugas) + r" & Detik/100 butir \\",
         r"\hline"]
    for s in urut_sel:
        if s not in sel:
            continue
        baris = [nama.get(s, s)]
        for t in tugas:
            c = sel[s].get(t)
            baris.append(k(c["accuracy"], 2) if c else "---")
        waktu = [sel[s][t]["time_s"] for t in tugas if t in sel[s]]
        baris.append(f"{sum(waktu)/len(waktu):.0f}" if waktu else "---")
        b.append(" & ".join(baris) + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── T5b: gradien efek pada lengan penalaran umum ────────────────────────────
def t_bench_uji():
    """Uji McNemar eksak `raw`/kv melawan tiap anggota R, ketiga tolok ukur.

    Tabel ini yang menopang klaim bahwa kelemahan kanal laten SPESIFIK pada
    muatan simbolik: nilai-p turun berorde-orde dari ARC-C ke HumanEval+.
    """
    if not BENCH.exists():
        return None
    from scipy.stats import binomtest
    d = json.loads(BENCH.read_text())
    R = ["soft/kv/m10", "gumbel/kv/m10", "sample/kv/m10", "moi/kv/m10"]
    nama = {"soft/kv/m10": r"\code{soft}", "gumbel/kv/m10": r"\code{gumbel}",
            "sample/kv/m10": r"\code{sample}", "moi/kv/m10": r"\code{moi}"}
    lbl = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}
    urut_tugas = ["arc_challenge", "gsm8k", "humanevalplus"]
    hasil = {}
    for g in d["groups"]:
        if g["n_items"] < 50:
            continue
        for p in g.get("pairs", []):
            a, b, c = p["a"], p["b"], p["correct"]
            if "raw/kv/m10" not in (a, b):
                continue
            lawan = b if a == "raw/kv/m10" else a
            if lawan not in R:
                continue
            n = c["a_only"] + c["b_only"]
            pv = binomtest(c["a_only"], n, 0.5).pvalue if n else 1.0
            hasil.setdefault(g["task"], {})[lawan] = pv
    if not hasil:
        return None
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Gradien efek pada lengan penalaran umum: nilai-$p$ uji "
         r"McNemar eksak \code{raw}/kv melawan tiap anggota keluarga "
         r"$\mathcal{R}$ pada medium kv, 100 butir per tolok ukur. Baca "
         r"tabel ini per BARIS: efeknya tak terdeteksi pada penalaran "
         r"berbasis pengetahuan, marginal pada penalaran aritmetika "
         r"bertingkat, dan sangat kuat pada sintesis program --- satu-satunya "
         r"tolok ukur yang menuntut kesetiaan simbol.}",
         r"\label{tab:benchuji}", r"\small",
         r"\begin{tabular}{l" + "r" * len(R) + "}", r"\hline",
         "Tolok ukur & " + " & ".join(nama[x] for x in R) + r" \\", r"\hline"]
    for t in urut_tugas:
        if t not in hasil:
            continue
        b.append(lbl.get(t, t) + " & "
                 + " & ".join(ilm(hasil[t].get(x)) for x in R) + r" \\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── T6: stabilitas & keragaman sinyal ───────────────────────────────────────
def t_stabilitas(d):
    st = d.get("stabilitas", {}).get("total", {})
    kl = d.get("klaster_sinyal", {})
    pl = d.get("peluruhan_alpha", {})
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Ketahanan dan keragaman sinyal korpus faktor.}",
         r"\label{tab:stabilitas}", r"\small",
         r"\begin{tabular}{lr}", r"\hline", r"Ukuran & Nilai \\", r"\hline"]
    if st:
        b += [f"Ekspresi berpasangan seleksi--holdout & {st['berpasangan']} \\\\",
              f"Berbalik tanda IC & {st['berbalik']} "
              f"({pers(st['laju_berbalik'])}) \\\\",
              f"Signifikan di seleksi & {st['sig_seleksi']} \\\\",
              f"Tetap signifikan di holdout & {st['sig_keduanya']} "
              f"({pers(st['laju_bertahan_signifikan'])}) \\\\"]
    if kl:
        b += [r"\hline",
              f"Ekspresi dengan deret IC harian & {kl['n_ekspresi']} \\\\",
              f"Klaster sinyal ($|\\rho|>{k(kl['ambang_korelasi'],1)}$) & "
              f"{kl['n_klaster']} \\\\",
              f"Rasio klaster per ekspresi & {k(kl['rasio_klaster_per_ekspresi'],3)} \\\\"]
    if pl.get("per_tahun"):
        b.append(r"\hline")
        for th, v in sorted(pl["per_tahun"].items()):
            b.append(f"Rerata $|IC|$ tahun {th} ({v['hari']} hari) & "
                     f"{k(v['mean_abs_ic'],4)} \\\\")
    b += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── T7: lantai acak ─────────────────────────────────────────────────────────
def t_lantai(d):
    la = d.get("lantai_acak")
    if not la:
        return None
    r = la["ringkas"]
    g = d["generasi"]
    R = [f"kv_{m}" for m in ("soft", "gumbel", "sample", "moi")]
    lolos_R = sum(g[s]["ekspresi_lolos_gate"] for s in R if s in g)
    tot_R = sum(g[s]["ekspresi_total"] for s in R if s in g)
    s21 = d["seleksi_2021"]
    hidup_R = sum(s21[s]["hidup"] for s in R if s in s21)
    ic_R = [s21[s]["mean_abs_ic"] for s in R if s in s21 and s21[s]["mean_abs_ic"]]
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Lantai acak: ekspresi yang ditarik acak dari tata bahasa DSL "
         r"yang sama, dilewatkan \textit{gate} yang sama dan dinilai pada jendela "
         r"yang sama. Pembanding ini menjawab apakah angka lolos \textit{gate} "
         r"mengukur mutu agen atau sekadar kemudahan \textit{gate}.}",
         r"\label{tab:lantai}", r"\small",
         r"\begin{tabular}{lrr}", r"\hline",
         r"Ukuran & Penarikan acak & Keluarga $\mathcal{R}$ (kv) \\", r"\hline",
         f"Ekspresi dinilai & {r['dibangkitkan']} & {tot_R} \\\\",
         f"Lolos \\textit{{gate}} & {r['lolos_gate']} "
         f"({pers(r['laju_lolos_gate'])}) & {lolos_R} "
         f"({pers(lolos_R/tot_R if tot_R else None)}) \\\\",
         f"Hidup & {r['hidup']} & {hidup_R} \\\\",
         f"Signifikan & {r['signifikan']} & --- \\\\",
         f"Rerata $|IC|$ & {k(r['mean_abs_ic'],4)} & "
         f"{k(sum(ic_R)/len(ic_R) if ic_R else None,4)} \\\\",
         r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


# ── T8: gate lintas pasar ───────────────────────────────────────────────────
def t_gate_pasar(d):
    gl = d.get("gate_lintas_pasar")
    if not gl:
        return None
    t = gl["total"]
    b = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Uji keberpindahan keputusan \textit{gate} antar-pasar. "
         r"\textit{Gate} eksekusi saat generasi memakai sampel pasar A-share; "
         r"tabel ini menjalankan \textit{gate} yang sama pada panel LQ45 dan "
         r"membandingkan keputusannya. Kesesuaian tinggi berarti angka lolos "
         r"\textit{gate} pada Tabel~\ref{tab:keandalan} tidak bergantung pasar.}",
         r"\label{tab:gatepasar}", r"\small",
         r"\begin{tabular}{lr}", r"\hline", r"Ukuran & Nilai \\", r"\hline",
         f"Ekspresi diperiksa & {t['ekspresi']} \\\\",
         f"Keputusan sama & {t['sama']} \\\\",
         f"Keputusan berbeda & {t['beda']} \\\\",
         f"Kesesuaian & {pers(t['kesesuaian'], 2)} \\\\",
         r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(b)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    d = json.loads((IDX / "analisis_idx.json").read_text())
    berkas = {
        "idx_keandalan.tex": t_keandalan(d),
        "idx_mutu.tex": t_mutu(d),
        "idx_backtest.tex": t_backtest(d),
        "idx_uji.tex": t_uji(d),
        "idx_bench.tex": t_bench(),
        "idx_bench_uji.tex": t_bench_uji(),
        "idx_stabilitas.tex": t_stabilitas(d),
        "idx_lantai.tex": t_lantai(d),
        "idx_gatepasar.tex": t_gate_pasar(d),
    }
    ditulis = []
    for nama, isi in berkas.items():
        if isi:
            (TAB / nama).write_text(isi + "\n")
            ditulis.append(nama)
    print("[tulis]", *ditulis)
    print("→", TAB)


if __name__ == "__main__":
    main()
