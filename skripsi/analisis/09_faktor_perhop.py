#!/usr/bin/env python3
"""Analisis per-hop dan biaya lengan faktor — dari `agent_trace`, tanpa GPU.

Tiga hal yang belum pernah dilaporkan skripsi, semuanya dari data yang sudah
tersimpan di `results/factor/frontend_*.json`:

1. **Laju parsing dengan denominator seragam.** Angka yang beredar sekarang
   ("raw gagal 14 dari 18 panggilan, lainnya 29%--40%") dihitung dari cacah
   berkas keluaran LLM, dan cacah itu BERBEDA-BEDA antar sel karena agen
   `construct` dipanggil ulang setiap kali gate memicu repair: `kv_raw` butuh
   15 percobaan untuk 6 jalan, `kv_soft` cukup 6. Membandingkan persentase atas
   denominator yang berbeda tidak sah. Di sini laju dihitung per JALAN
   (arah x seed) — denominatornya sama di semua sel menurut rancangan — dan
   cacah percobaan dilaporkan terpisah sebagai ukuran biaya tersendiri.

2. **Di hop mana kerusakan muncul.** Rantai faktor adalah proposal -> innovate
   -> construct. `agent_trace` merekam per-agen: panjang KV, token masuk/keluar,
   waktu laten dan waktu generasi, serta rasio pengulangan kata pada teks yang
   dikeluarkan. Lengan bench tidak menyimpan ini (`agents` selalu null), jadi
   pelacakan per-hop hanya mungkin di lengan faktor.

3. **Biaya lengan faktor per formulasi.** Bab IV melaporkan biaya untuk lengan
   bench saja. Padahal RM3 menanyakan pengaruh formulasi terhadap efisiensi,
   dan lengan faktor punya angkanya.

4. **Kesetiaan terhadap arah yang ditugaskan.** Tiap jalan diberi satu kalimat
   arah eksplorasi (`d0`, `d1`, ...) yang isinya DIKETAHUI. Kalimat itu karena
   itu berfungsi seperti muatan pada uji kapasitas kanal: seberapa banyak
   darinya yang masih terbaca pada keluaran agen terakhir mengukur fidelitas
   simbolik pada tugas nyata, bukan pada tugas buatan. Ukuran ini tersedia
   untuk SEMUA sel, termasuk KV murni, karena agen `construct` selalu
   mengeluarkan teks.

Keluaran -> analisis/faktor_perhop.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTOR = ROOT / "quantalatent" / "results" / "factor"
OUT = ROOT / "analisis" / "faktor_perhop.json"

sys.path.insert(0, str(ROOT / "quantalatent" / "backend"))
from eval.fidelity import corruption_hits, jaccard, mech_overlap  # noqa: E402

# Kalimat arah, disalin dari `backend/factor/run_factor.py::DIRECTIONS` supaya
# skrip analisis tak perlu mengimpor modul yang menarik torch.
ARAH = {
    "d0": "short-term reversal after abnormally high-volume days in small-cap stocks",
    "d1": "mean-reversion in low-volatility stocks during regime transitions",
    "opp_mom": ("long-horizon price momentum continuation: stocks that trended "
                "up over 30-60 days keep outperforming, signal carried by "
                "sustained directional drift in close prices"),
    "opp_rev": ("very short-horizon contrarian reversal: stocks with the "
                "largest 1-3 day intraday range expansion snap back and "
                "underperform, signal carried by high-low range spikes"),
}

URUTAN_HOP = ["proposal", "innovate", "construct", "repair"]


def rerata(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


def sel_dari_tag(tag: str) -> tuple[str, str]:
    """`kv_and_text_gumbel` -> ('kv_and_text', 'gumbel'); `text` -> ('text', '-')."""
    for medium in ("kv_and_text", "kv"):
        if tag.startswith(medium + "_"):
            return medium, tag[len(medium) + 1:]
    return tag, "-"


def main() -> None:
    per_sel: list[dict] = []

    for path in sorted(FACTOR.glob("frontend_*.json")):
        tag = path.stem[len("frontend_"):]
        medium, metode = sel_dari_tag(tag)
        doc = json.loads(path.read_text())
        runs = doc["runs"]

        # ── 1. laju per JALAN (denominator seragam = jumlah run) ────────────
        n_jalan = len(runs)
        jalan_berekspresi = sum(1 for r in runs if (r.get("factors") or []))
        jalan_lolos_gate = sum(1 for r in runs if (r.get("passing") or []))
        jalan_repair = sum(1 for r in runs if r.get("repaired"))

        # ── 2. per hop ──────────────────────────────────────────────────────
        hop: dict[str, list[dict]] = defaultdict(list)
        for r in runs:
            for t in (r.get("agent_trace") or []):
                hop[t["agent"]].append(t)

        per_hop = []
        for nama in URUTAN_HOP:
            tr = hop.get(nama)
            if not tr:
                continue
            emit = [t for t in tr if (t.get("n_out_tok") or 0) > 0]
            per_hop.append({
                "hop": nama,
                "n_panggilan": len(tr),
                "panggilan_per_jalan": round(len(tr) / n_jalan, 2),
                "kv_len": rerata([t.get("kv_len") for t in tr]),
                "token_masuk": rerata([t.get("n_in_tok") for t in tr]),
                "token_keluar": rerata([t.get("n_out_tok") for t in tr]),
                "latent_s": rerata([t.get("latent_s") for t in tr]),
                "gen_s": rerata([t.get("gen_s") for t in tr]),
                "detik": rerata([t.get("s") for t in tr]),
                # rep_ratio hanya bermakna untuk hop yang benar-benar
                # mengeluarkan teks; pada hop laten murni ia selalu 0 karena
                # tak ada teks untuk dihitung, bukan karena teksnya bersih.
                "rep_ratio_saat_emit": rerata([t.get("rep_ratio") for t in emit]),
                "n_emit_teks": len(emit),
                "parse_ok": sum(1 for t in tr if t.get("parsed_ok")),
            })

        # ── 3. kesetiaan terhadap arah + korupsi, per hop ───────────────────
        # Diukur pada teks yang BENAR-BENAR dikeluarkan agen. Hop laten murni
        # tak punya teks, jadi ia tak muncul di sini — itu bukan nilai nol,
        # melainkan ketiadaan pengamatan, dan keduanya tak boleh tertukar.
        fid_arah: dict[str, list[float]] = defaultdict(list)
        mech_arah: dict[str, list[float]] = defaultdict(list)
        korupsi: dict[str, list[float]] = defaultdict(list)
        rantai: list[dict] = []

        for r in runs:
            arah = ARAH.get(r.get("direction", ""), "")
            teks_hop: dict[str, str] = {}
            for t in (r.get("agent_trace") or []):
                txt = t.get("text") or ""
                if not txt:
                    continue
                nama = t["agent"]
                teks_hop.setdefault(nama, txt)
                if arah:
                    j = jaccard(arah, txt)
                    m = mech_overlap(arah, txt)
                    if j is not None:
                        fid_arah[nama].append(j)
                    if m is not None:
                        mech_arah[nama].append(m)
                # dinormalkan per 1000 karakter: keluaran yang lebih panjang
                # otomatis memuat lebih banyak kecocokan pola, jadi cacah
                # mentahnya tak sebanding antar hop maupun antar sel.
                korupsi[nama].append(corruption_hits(txt) / max(len(txt), 1) * 1000)

            # rantai hop-ke-hop hanya bisa dihitung bila hop hulu memang
            # mengeluarkan teks (medium `text` dan `kv_and_text`).
            if "proposal" in teks_hop and "innovate" in teks_hop:
                rantai.append({
                    "pi": jaccard(teks_hop["proposal"], teks_hop["innovate"]),
                    "ic": jaccard(teks_hop.get("innovate", ""),
                                  teks_hop.get("construct", "")),
                    "mech_pi": mech_overlap(teks_hop["proposal"], teks_hop["innovate"]),
                    "mech_ic": mech_overlap(teks_hop.get("innovate", ""),
                                            teks_hop.get("construct", "")),
                })

        for h in per_hop:
            nama = h["hop"]
            h["fidelitas_arah"] = rerata(fid_arah.get(nama, []))
            h["mech_arah"] = rerata(mech_arah.get(nama, []))
            h["korupsi_per_1000_char"] = rerata(korupsi.get(nama, []))

        # ── 4. biaya per jalan ──────────────────────────────────────────────
        semua = [t for tr in hop.values() for t in tr]
        tok_keluar = sum(t.get("n_out_tok") or 0 for t in semua)
        tok_masuk = sum(t.get("n_in_tok") or 0 for t in semua)
        detik = sum(r.get("duration_s") or 0 for r in runs)

        per_sel.append({
            "tag": tag, "medium": medium, "metode": metode,
            "n_jalan": n_jalan,
            "jalan_berekspresi": jalan_berekspresi,
            "laju_jalan_berekspresi": round(jalan_berekspresi / n_jalan, 3),
            "jalan_lolos_gate": jalan_lolos_gate,
            "laju_jalan_lolos_gate": round(jalan_lolos_gate / n_jalan, 3),
            "jalan_perlu_repair": jalan_repair,
            "percobaan_construct": len(hop.get("construct", [])) + len(hop.get("repair", [])),
            "percobaan_construct_per_jalan":
                round((len(hop.get("construct", [])) + len(hop.get("repair", []))) / n_jalan, 2),
            "token_keluar_per_jalan": round(tok_keluar / n_jalan, 1),
            "token_masuk_per_jalan": round(tok_masuk / n_jalan, 1),
            "detik_per_jalan": round(detik / n_jalan, 1),
            "rantai_hop": {
                "n": len(rantai),
                "jaccard_proposal_innovate": rerata([x["pi"] for x in rantai]),
                "jaccard_innovate_construct": rerata([x["ic"] for x in rantai]),
                "mech_proposal_innovate": rerata([x["mech_pi"] for x in rantai]),
                "mech_innovate_construct": rerata([x["mech_ic"] for x in rantai]),
            },
            "per_hop": per_hop,
        })

    # ── ringkasan efisiensi lengan faktor: KV terhadap teks ────────────────
    teks = next((s for s in per_sel if s["medium"] == "text"), None)
    efisiensi = []
    if teks:
        for s in per_sel:
            if s["medium"] == "text":
                continue
            hemat = (1 - s["token_keluar_per_jalan"] / teks["token_keluar_per_jalan"]
                     if teks["token_keluar_per_jalan"] else None)
            cepat = (teks["detik_per_jalan"] / s["detik_per_jalan"]
                     if s["detik_per_jalan"] else None)
            efisiensi.append({
                "tag": s["tag"], "medium": s["medium"], "metode": s["metode"],
                "token_keluar_per_jalan": s["token_keluar_per_jalan"],
                "token_keluar_per_jalan_text": teks["token_keluar_per_jalan"],
                "penghematan_token_vs_text": round(hemat, 4) if hemat is not None else None,
                "percepatan_vs_text": round(cepat, 2) if cepat is not None else None,
            })

    OUT.write_text(json.dumps({
        "catatan": ("laju dihitung per JALAN (arah x seed) supaya denominatornya "
                    "seragam antar sel; cacah percobaan construct dilaporkan "
                    "terpisah karena repair membuat jumlah panggilan berbeda"),
        "per_sel": per_sel,
        "efisiensi_vs_text": efisiensi,
    }, indent=2))
    print(f"[tulis] {OUT}")

    # ── ringkasan terbaca di terminal ──────────────────────────────────────
    print(f"\n{'sel':22s} {'jalan':>5s} {'ber-ekspr':>9s} {'lolos gate':>10s} "
          f"{'coba/jalan':>10s} {'tok/jalan':>9s} {'detik/jalan':>11s}")
    for s in per_sel:
        print(f"{s['tag']:22s} {s['n_jalan']:5d} "
              f"{s['laju_jalan_berekspresi']:9.2f} {s['laju_jalan_lolos_gate']:10.2f} "
              f"{s['percobaan_construct_per_jalan']:10.2f} "
              f"{s['token_keluar_per_jalan']:9.0f} {s['detik_per_jalan']:11.0f}")


if __name__ == "__main__":
    main()
