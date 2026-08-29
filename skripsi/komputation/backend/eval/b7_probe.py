"""B7 — persamaan realignment mana yang benar-benar berlaku, dan apa akibatnya.

Dua pertanyaan, keduanya diukur pada JALUR PRODUKSI (`_CoreEngine.latent_pass`),
bukan pada replika:

  (1) INERTNESS. Apakah `use_realign` masih berpengaruh setelah mode langkah
      laten produksi bukan lagi "raw"? Diuji secara deterministik: mode `soft`
      (tanpa noise), prompt & seed sama, `use_realign` True vs False. Bila
      hidden state akhirnya IDENTIK bit-per-bit, maka matriks ridge M memang
      tidak pernah diterapkan — dan ablasi G6 hanya berlaku untuk mode "raw".

  (2) GEOMETRI. Untuk vektor laten yang BENAR-BENAR diumpankan sebagai virtual
      token, seberapa dekat ia ke embedding token nyata? `max_v cos(z, W_in[v])`
      mengukur apakah vektor itu berada di dalam (≈1) atau di luar (≈0) manifold
      embedding. Inilah beda antara "proyeksi" dan "ekstrapolasi".

Angka ridge M sendiri (cos(h, hM), simpangan dari identitas) datang dari
`eval/realign_probe.py` — skrip itu tidak perlu GPU karena hanya membaca
safetensors.

    PYTHONPATH=backend python backend/eval/b7_probe.py --model Qwen/Qwen3-8B --steps 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import QL_ROOT as QL, bootstrap, ensure_out, OUT_PROBE as _OUT
bootstrap()

OUT = ensure_out(_OUT)


PROMPT = ("Propose a market hypothesis about price momentum in Chinese A-share "
          "stocks, then name the data columns it needs.")

# Mode deterministik saja untuk uji (1): gumbel/sample menyuntik noise, jadi dua
# jalankan tak akan identik apa pun jawabannya soal M — uji itu jadi tak bisa
# menyimpulkan apa-apa.
DETERMINISTIC = ("raw", "soft")
# `moi` ditambahkan 2026-08-27. Ia tertinggal pada jalankan pertama, sehingga
# tabel geometri skripsi memuat empat formulasi sementara teksnya membahas
# lima — dan yang hilang justru formulasi yang klaim keanggotaannya di keluarga
# konveks paling perlu bukti, karena satu-satunya yang bobotnya campuran dua
# suku (Proposisi 1). `moi` memakai token hasil sampling, jadi ia TIDAK ikut
# uji inertness yang menuntut dua jalankan identik bit-per-bit.
ALL_MODES = ("raw", "soft", "gumbel", "sample", "moi")


def _build(model: str, device: str, steps: int, realign: bool, mode: str,
           temp: float, alpha: float | None = None):
    from llm.client import LocalLLMBackend
    return LocalLLMBackend(
        model_name=model, device=device, latent_steps=steps,
        use_realign=realign, enable_thinking=False, log_tensors=False,
        store_kv=False, output_log_dir=str(OUT / "llm_outputs" / "b7"),
        max_new_tokens=64, knn_enabled=False,
        latent_step_mode=mode, latent_step_temp=temp,
        latent_step_alpha=alpha,
        latent_early_stop_cos=1.0,     # early-stop DIMATIKAN: uji ini soal
    )                                  # persamaan, bukan soal panjang rollout


def _geometri(be, prompt: str, role: str, torch, F) -> dict:
    """max_v cos(z_k, W_in[v]) per langkah laten, diringkas rerata/min/maks."""
    torch.manual_seed(0)
    r = be.build_messages_and_run(user_prompt=prompt, mode="kv_only", role=role)
    z = r.latent_vecs.detach().float()                      # [steps, d]
    W_in = be._engine.model.get_input_embeddings().weight   # noqa: SLF001
    W_n = F.normalize(W_in.detach().float(), dim=1)
    zc = F.normalize(z.to(W_n.device), dim=1)
    # dihitung berkeping agar matriks [steps, V] tak meledak di VRAM
    mx = torch.cat([(zc[i:i + 4] @ W_n.T).max(dim=1).values
                    for i in range(0, zc.shape[0], 4)]).cpu()
    return {
        "max_cos_embed_mean": round(float(mx.mean()), 4),
        "max_cos_embed_min": round(float(mx.min()), 4),
        "max_cos_embed_max": round(float(mx.max()), 4),
        "n_steps": int(z.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--alphas", default="0.25,0.5,0.75",
                    help="nilai alpha mode `mix` yang ikut diukur geometrinya; "
                         "kosongkan untuk melewati. Titik ujung (0 dan 1) tak "
                         "perlu disebut: keduanya identik dengan `raw` dan "
                         "`soft` yang sudah diukur di atas.")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    OUT.mkdir(parents=True, exist_ok=True)
    # β dicatat meski tak dioper eksplisit: `_build` memakai default engine
    # (1.0, setelan universal paper MoI) yang sama dengan sel benchmark, dan
    # tabel geometri skripsi harus bisa menyebut nilai itu tanpa menebak.
    beta = float(os.environ.get("LATENT_STEP_BETA", "1.0"))
    res: dict = {"_meta": {"model": a.model, "steps": a.steps, "temp": a.temp,
                           "moi_beta": beta, "modes": list(ALL_MODES)}}

    # ── (1) inertness use_realign per mode deterministik ────────────────────
    inert = {}
    for mode in DETERMINISTIC:
        hiddens, vecs = [], []
        for realign in (True, False):
            be = _build(a.model, a.device, a.steps, realign, mode, a.temp)
            torch.manual_seed(0)
            r = be.build_messages_and_run(user_prompt=PROMPT, mode="kv_only",
                                          role=f"b7_{mode}_{realign}")
            hiddens.append(r.hidden_last.detach().float().cpu())
            vecs.append(r.latent_vecs.detach().float().cpu())
            del be, r
            torch.cuda.empty_cache()
        identical = bool(torch.equal(hiddens[0], hiddens[1]))
        d = float((hiddens[0] - hiddens[1]).abs().max())
        cos = float(F.cosine_similarity(hiddens[0], hiddens[1], dim=-1).mean())
        inert[mode] = {"hidden_identical": identical,
                       "max_abs_diff": d, "cos": round(cos, 6),
                       "latent_vecs_identical": bool(torch.equal(vecs[0], vecs[1]))}
        verdict = ("M TIDAK dipakai → use_realign inert"
                   if identical else "M dipakai → use_realign bermakna")
        print(f"[b7/inert] {mode:5s} use_realign True vs False: "
              f"identik={identical} maxdiff={d:.3e} cos={cos:.6f}  → {verdict}",
              flush=True)
    res["inertness"] = inert

    # ── (2) geometri vektor laten produksi per mode ─────────────────────────
    geo = {}
    for mode in ALL_MODES:
        be = _build(a.model, a.device, a.steps, True, mode, a.temp)
        geo[mode] = _geometri(be, PROMPT, f"b7_geo_{mode}", torch, F)
        print(f"[b7/geo]   {mode:6s} cos ke embedding terdekat: "
              f"rata2={geo[mode]['max_cos_embed_mean']:+.4f} "
              f"[{geo[mode]['max_cos_embed_min']:+.4f}, "
              f"{geo[mode]['max_cos_embed_max']:+.4f}]", flush=True)
        del be
        torch.cuda.empty_cache()
    res["geometry"] = geo

    # ── (3) kurva geometri sepanjang sumbu interpolasi `mix` ────────────────
    # Separuh geometris dari kurva dose-response: apakah jarak ke convex hull
    # embedding benar-benar bergerak KONTINU antara `raw` dan `soft`, atau
    # melompat. Separuh kinerjanya datang dari sel benchmark/faktor bermode
    # `mix`; keduanya baru bisa disandingkan kalau sumbu-x-nya diukur, bukan
    # diasumsikan linier.
    #
    # Murah: tak ada generasi teks sama sekali, hanya rollout laten.
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    if alphas:
        kurva = {}
        for al in alphas:
            be = _build(a.model, a.device, a.steps, True, "mix", a.temp, alpha=al)
            kurva[f"{al}"] = _geometri(be, PROMPT, f"b7_geo_mix{al}", torch, F)
            print(f"[b7/mix]   alpha={al:<5} cos ke embedding terdekat: "
                  f"rata2={kurva[f'{al}']['max_cos_embed_mean']:+.4f}", flush=True)
            del be
            torch.cuda.empty_cache()
        # Titik ujung disalin dari hasil (2), bukan dihitung ulang: mode `mix`
        # pada alpha 0 dan 1 memang mereduksi persis ke `raw` dan `soft`
        # (diverifikasi numerik), jadi menjalankannya lagi hanya menambah derau
        # sampling pada dua titik yang seharusnya berimpit.
        kurva.setdefault("0.0", geo.get("raw"))
        kurva.setdefault("1.0", geo.get("soft"))
        res["geometry_mix"] = dict(sorted(kurva.items(), key=lambda kv: float(kv[0])))

    suffix = f"_{a.tag}" if a.tag else ""
    path = OUT / f"b7_probe_{a.model.replace('/', '_')}{suffix}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"tersimpan → {path}")


if __name__ == "__main__":
    main()
