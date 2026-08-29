"""A9 — kapasitas kanal laten: berapa banyak muatan SIMBOLIK yang benar-benar lewat?

Seluruh skripsi ini membandingkan dua medium komunikasi antar-agen (KV-cache
laten vs teks). Sampai sekarang perbandingan itu dilakukan lewat MUTU FAKTOR —
metrik yang dicemari puluhan hal lain (prompt, gate, backtest, keberuntungan
sampling). Skrip ini menggantinya dengan pengukuran langsung:

    titipkan muatan yang DIKETAHUI ke agen hulu → minta agen hilir menyebutkannya
    kembali → akurasi rekonstruksi = kapasitas kanal untuk muatan simbolik.

Hasilnya tidak bergantung pada mutu faktor sama sekali, jadi ia bisa
dipertahankan di sidang sebagai bukti mekanisme, bukan sebagai tafsir hasil.

── EMPAT LENGAN ─────────────────────────────────────────────────────────────
Yang BERBEDA antar-lengan hanya APA YANG DIOPER. Komputasi agen hulu dibuat
SAMA di semua lengan (prompt sama, m langkah laten sama), supaya selisih yang
terukur adalah selisih KANAL — bukan selisih "seberapa lama hulu berpikir".

  text            hulu men-decode jawabannya jadi teks; hilir membaca teks itu,
                  konteksnya bersih (tanpa KV). Medium teks.
  kv_full         hilir mewarisi SELURUH KV hulu = [token prompt hulu +
                  m vektor laten]. Inilah comm_mode "kv" produksi (NO-CROP).
  kv_prompt_only  hilir mewarisi HANYA token prompt hulu (blok laten dibuang).
                  KONTROL POSITIF: memisahkan "informasi lewat lewat token
                  prompt yang ikut terwarisi" dari "informasi lewat lewat
                  vektor laten".
  kv_latent_only  hilir hanya mewarisi m VEKTOR LATEN-nya (token prompt hulu
                  dipotong, key di-re-rotasi lewat kv_truncate/B8).
                  Ini kanal laten MURNI — satu-satunya lengan yang benar-benar
                  menguji klaim ekspresivitas "latent thoughts".
  none            hilir tidak menerima apa pun. LANTAI TEBAKAN.

Pasangan (kv_prompt_only, kv_latent_only) itulah inti A9. `kv_full` saja tidak
bisa menjawab apa-apa: kalau ia sempurna, itu bisa karena vektor latennya
ekspresif ATAU semata karena payload-nya masih tertulis verbatim di token
prompt yang ikut diwariskan. Memisahkan keduanya mengubah pertanyaan "apakah
laten lossy" dari tafsiran menjadi angka.

Catatan kejujuran: lengan `text` di sini BUKAN replika persis comm_mode="text"
produksi (di sana latent_steps=0, hulu tak berpikir laten sama sekali). Yang
ditiru adalah medium handoff-nya. Perbedaan itu disengaja agar perbandingannya
berpasangan; jangan kutip angka `text` di sini sebagai angka produksi.

Lengan `none` wajib ada. Muatan `dsl` diambil dari pustaka fungsi yang juga
disebut di prompt hilir, jadi sebagian bisa ditebak tanpa kanal apa pun; tanpa
lantai ini, akurasi 40% bisa saja berarti "nol informasi lewat".

── DUA JENIS MUATAN ─────────────────────────────────────────────────────────
  dsl     k nama fungsi acak dari pustaka DSL (in-domain — inilah muatan yang
          benar-benar dioper antar-agen di produksi: palette fungsi).
  token   k "kata" acak tanpa makna (mis. `qv7`, `zx2`) — tak bisa ditebak dari
          prior apa pun, jadi ia mengukur kapasitas kanal secara bersih.

── SKOR ─────────────────────────────────────────────────────────────────────
  recall      |ditebak ∩ benar| / k   (urutan diabaikan)
  exact       fraksi run dengan himpunan yang persis sama
  hallucinate |ditebak \\ benar| / |ditebak|

    PYTHONPATH=backend python backend/eval/channel_capacity.py --model Qwen/Qwen3-8B --k 5 --trials 12
    PYTHONPATH=backend python backend/eval/channel_capacity.py --payload token --latent-steps 40
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import QL_ROOT as QL, bootstrap, ensure_out, OUT_PROBE as _OUT
bootstrap()

OUT = ensure_out(_OUT)

# `.env` proyek dimuat DI SINI, sebelum modul apa pun dari `backend/` diimpor.
# Alasannya bukan kerapian: `llm/client.py` dan `llm/models.py` membaca
# HF_LOCAL_ONLY dari os.environ pada saat MODUL diimpor (bukan saat dipanggil),
# jadi memuat .env belakangan tidak berpengaruh. Tanpa ini, lab script berjalan
# dengan default local_files_only=True dan gagal saat model belum ada di cache —
# beda dari `launcher.py` yang memang sudah memanggil load_dotenv().
# Nilai yang SUDAH ada di shell tidak ditimpa (override=False), supaya
# `HF_LOCAL_ONLY=1 python lab/...` tetap bisa memaksa mode offline.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(QL / ".env", override=False)
except ImportError:  # dotenv belum terpasang → jatuh ke env shell apa adanya
    pass


ARMS = ("text", "kv_full", "kv_prompt_only", "kv_latent_only", "none")

# Prompt SENGAJA minimal dan tidak memakai prompts.yaml produksi. Yang diukur di
# sini adalah kanalnya, bukan rantai agen; memakai prompt mining akan
# memasukkan hipotesis, palette, dan aturan DSL sebagai variabel pengganggu.
SYS_UP = "You are a relay agent. You memorize the payload you are given."
SYS_DOWN = "You are a relay agent. You report the payload you received."

# Tugas hulu adalah MERELAI muatan — sama di semua lengan. Versi pertama
# ("hafalkan, nanti ditanya") membuat model hulu hanya menjawab "I have
# memorized the payload." sehingga lengan `text` mengukur PROMPT, bukan kanal:
# tak ada muatan yang pernah masuk ke medium teks untuk dinilai.
USER_UP = (
    "Your job is to relay a payload to the next agent.\n"
    "PAYLOAD: {payload}\n"
    "Restate all {k} items, comma-separated, so the next agent receives them."
)

# Prompt hilir TIDAK menyebut apa pun tentang muatan, selain formatnya —
# sengaja, supaya satu-satunya sumber jawabannya adalah kanal yang diuji.
USER_DOWN_KV = (
    "Report the payload you were given earlier. Output ONLY the {k} items, "
    "comma-separated, nothing else."
)
USER_DOWN_TEXT = (
    "An earlier agent was given a payload and reported this:\n"
    "---\n{prior}\n---\n"
    "Report the payload. Output ONLY the {k} items, comma-separated, nothing else."
)
USER_DOWN_NONE = (
    "Report the payload you were given earlier. Output ONLY the {k} items, "
    "comma-separated, nothing else."
)


def dsl_names() -> list[str]:
    """Nama fungsi DSL yang benar-benar terdaftar di regulator produksi."""
    import dsl.factor_ast  # noqa: F401  (putus circular import)
    from gate.factor_regulator import _build_arity_map
    return sorted(_build_arity_map())


def make_payload(kind: str, k: int, rng: random.Random,
                 pool: list[str]) -> list[str]:
    if kind == "dsl":
        return rng.sample(pool, k)
    # `token`: konsonan+vokal+digit → pasti ter-tokenisasi jadi >1 token dan
    # tidak punya prior semantik apa pun di model.
    out = set()
    while len(out) < k:
        out.add(rng.choice("bcdfghjklmnpqrstvwxz")
                + rng.choice("aeiou")
                + rng.choice("bcdfghjklmnpqrstvwxz")
                + str(rng.randint(10, 99)))
    return sorted(out)


def parse_items(text: str, kind: str, pool: list[str]) -> list[str]:
    """Ambil item dari jawaban hilir. Toleran terhadap format (koma, baris,
    bullet) — yang diuji kanal, bukan kepatuhan format."""
    t = (text or "").strip()
    if kind == "dsl":
        found = re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", t.upper())
        seen, out = set(), []
        for f in found:
            if f in pool and f not in seen:
                seen.add(f)
                out.append(f)
        return out
    found = re.findall(r"\b[a-z]{3}\d{2}\b", t.lower())
    seen, out = set(), []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def score(pred: list[str], truth: list[str]) -> dict:
    ps, ts = set(pred), set(truth)
    hit = len(ps & ts)
    return {
        "recall": hit / len(ts) if ts else 0.0,
        "exact": float(ps == ts),
        "hallucinate": (len(ps - ts) / len(ps)) if ps else 0.0,
        "n_pred": len(ps),
    }


def run_trial(backend, arm: str, payload: list[str], k: int, kind: str,
              pool: list[str], max_new: int, model) -> dict:
    """Satu titipan → satu rekonstruksi. Return skor + jejak."""
    from mas import kv_ops
    from llm._shared import kv_truncate

    up_text = ""
    past = None
    t0 = time.time()

    if arm != "none":
        up_mode = "kv_and_text" if arm == "text" else "kv_only"
        r_up = backend.build_messages_and_run(
            user_prompt=USER_UP.format(payload=", ".join(payload), k=k),
            system_prompt=SYS_UP, mode=up_mode, role=f"a9_up_{arm}",
            max_new_tokens=max_new,
            # NO-CROP seperti produksi: jawaban hulu tetap di KV.
            crop_after_generate=False,
        )
        up_text = (r_up.text or "").strip()
        m = r_up.n_latent_steps or 0
        L = kv_ops.kv_seq_len(r_up.kv_cache)
        if arm == "kv_full":
            past = r_up.kv_cache
        elif arm == "kv_latent_only":
            # Sisakan HANYA blok vektor laten (m token terakhir). kv_truncate
            # dengan `model` me-re-rotasi RoPE ke posisi kontigu (B8) — tanpa
            # itu blok laten akan tampak bergeser dan lengan ini mengukur bug,
            # bukan kanal.
            if m <= 0:
                return {"arm": arm, "error": "tidak ada vektor laten (latent_steps=0)"}
            past = kv_truncate(kv_ops.kv_deepcopy(r_up.kv_cache), m, model=model)
        elif arm == "kv_prompt_only":
            # Buang m token TERAKHIR (blok laten), sisakan token prompt hulu.
            # Di sini TIDAK perlu re-rotasi RoPE: token yang disisakan menempati
            # posisi [0, L−m) yang memang sudah kontigu dan sudah benar —
            # beda dengan lengan latent_only yang memotong dari depan.
            if m <= 0 or L - m <= 0:
                return {"arm": arm, "error": f"KV tak cukup (L={L}, m={m})"}
            past = kv_ops.kv_deepcopy(r_up.kv_cache)
            past.crop(L - m)

    if arm == "text":
        user_down = USER_DOWN_TEXT.format(prior=up_text or "(empty)", k=k)
    elif arm == "none":
        user_down = USER_DOWN_NONE.format(k=k)
    else:
        user_down = USER_DOWN_KV.format(k=k)

    r_down = backend.build_messages_and_run(
        user_prompt=user_down, system_prompt=SYS_DOWN,
        past_key_values=past, mode="text_only" if past is None else "kv_and_text",
        role=f"a9_down_{arm}", max_new_tokens=max_new,
        latent_steps=0,          # hilir hanya MEMBACA; tak perlu berpikir laten
        crop_after_generate=True,
    )
    pred = parse_items(r_down.text or "", kind, pool)
    s = score(pred, payload)
    s.update({"arm": arm, "dur_s": round(time.time() - t0, 2),
              "up_text_len": len(up_text),
              "up_text": up_text[:300],   # lengan `text` hanya sah bila hulu
                                          # BENAR-BENAR menuliskan muatannya
              "kv_len_in": kv_ops.kv_seq_len(past) if past is not None else 0,
              "pred": pred, "truth": payload,
              "down_text": (r_down.text or "")[:300]})
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=5, help="jumlah item per muatan")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--payload", default="dsl,token",
                    help="dsl | token | keduanya (dipisah koma)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--latent-steps", type=int, default=10,
                    help="m — panjang blok laten hulu (default = produksi)")
    ap.add_argument("--latent-mode", default="gumbel")
    ap.add_argument("--latent-temp", type=float, default=0.7)
    # Hanya dipakai mode "moi" (Mixture of Inputs, arXiv:2505.14827) — β=1
    # adalah setelan universal paper; sweep {0.25..8} bila perlu per-task.
    ap.add_argument("--latent-beta", type=float, default=1.0)
    # Hanya berpengaruh pada --latent-mode raw (mode lain tak pernah memakai M).
    # Repo resmi LatentMAS memperlakukan realignment sebagai HYPERPARAMETER:
    # tanpa flag `--latent_space_realign`, `_build_latent_realign_matrix`
    # mengembalikan matriks IDENTITAS dan hanya magnitudo yang dinormalkan.
    # Jadi `raw`+`--no-realign` = konfigurasi DEFAULT paper, sedangkan
    # `raw` (realign aktif) = jalur $W_a$ Teorema A.1. Keduanya perlu diukur;
    # menyamakannya akan salah mengatributkan hasilnya ke mekanisme yang keliru.
    ap.add_argument("--no-realign", action="store_true",
                    help="mode raw tanpa matriks ridge M (M = I, hanya renormalisasi)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    from llm.client import LocalLLMBackend

    OUT.mkdir(parents=True, exist_ok=True)
    kinds = [k.strip() for k in a.payload.split(",") if k.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    pool = dsl_names()

    backend = LocalLLMBackend(
        model_name=a.model, device=a.device, latent_steps=a.latent_steps,
        use_realign=not a.no_realign, enable_thinking=False, log_tensors=False,
        store_kv=False, output_log_dir=str(OUT / "llm_outputs" / "a9"),
        max_new_tokens=a.max_new_tokens, temperature=0.6, top_p=0.95,
        knn_enabled=False, latent_step_mode=a.latent_mode,
        latent_step_temp=a.latent_temp, latent_step_beta=a.latent_beta,
        # Early-stop dimatikan: A9 mengukur kapasitas blok laten sepanjang m,
        # jadi m harus benar-benar m di semua lengan.
        latent_early_stop_cos=1.0,
    )
    model = backend._engine.model   # noqa: SLF001 — dibutuhkan kv_truncate (B8)

    print(f"[a9] {a.model} k={a.k} trials={a.trials} m={a.latent_steps} "
          f"mode={a.latent_mode} realign={not a.no_realign} "
          f"muatan={kinds} lengan={arms}")

    records, rows = [], []
    for kind in kinds:
        # Muatan dibangkitkan SEKALI per trial lalu dipakai oleh SEMUA lengan,
        # supaya perbandingan antar-lengan berpasangan (paired) — bukan
        # membandingkan muatan yang kebetulan berbeda kesulitannya.
        rng = random.Random(a.seed)
        payloads = [make_payload(kind, a.k, rng, pool) for _ in range(a.trials)]
        for arm in arms:
            res = []
            for i, payload in enumerate(payloads):
                torch.manual_seed(a.seed + i)
                r = run_trial(backend, arm, payload, a.k, kind, pool,
                              a.max_new_tokens, model)
                r.update({"payload_kind": kind, "trial": i})
                records.append(r)
                res.append(r)
                torch.cuda.empty_cache()
            ok = [r for r in res if "error" not in r]
            if not ok:
                print(f"  {kind:6s} {arm:15s} GAGAL: {res[0].get('error')}")
                continue
            # Recall per POSISI muatan. Kanal yang lossy tidak kehilangan item
            # secara acak — ia mempertahankan awal urutan lalu meluruh. Kurva
            # ini yang membedakan "kanal sempit" dari "kanal rusak", dan ia
            # hilang total kalau hanya recall rata-rata yang dilaporkan.
            by_pos = [round(sum(1 for r in ok if r["truth"][i] in r["pred"])
                            / len(ok), 3) for i in range(a.k)]
            row = {
                "payload": kind, "arm": arm, "n": len(ok),
                "recall": round(st.mean(r["recall"] for r in ok), 3),
                "exact": round(st.mean(r["exact"] for r in ok), 3),
                "hallucinate": round(st.mean(r["hallucinate"] for r in ok), 3),
                "n_pred_mean": round(st.mean(r["n_pred"] for r in ok), 2),
                "recall_by_position": by_pos,
                "dur_s": round(st.mean(r["dur_s"] for r in ok), 2),
            }
            rows.append(row)
            print(f"  {kind:6s} {arm:15s} recall={row['recall']:.3f} "
                  f"exact={row['exact']:.3f} halus={row['hallucinate']:.3f} "
                  f"n_pred={row['n_pred_mean']:.1f} "
                  f"pos=[{' '.join(f'{p:.2f}' for p in by_pos)}] "
                  f"{row['dur_s']:5.2f}s", flush=True)

    # `use_realign` DICATAT walau berkas lama tak memuatnya: run lama semuanya
    # memakai True (nilai hardcoded saat itu), jadi ketiadaan kunci ini pada
    # berkas lama berarti True — bukan tidak diketahui.
    doc = {"_meta": {"model": a.model, "k": a.k, "trials": a.trials,
                     "latent_steps": a.latent_steps, "latent_mode": a.latent_mode,
                     "use_realign": not a.no_realign,
                     "latent_beta": a.latent_beta,
                     "seed": a.seed, "dsl_pool_size": len(pool)},
           "_summary": rows, "records": records}
    # Nama berkas WAJIB memuat mode+m. Tanpa ini dua run yang berbeda hanya pada
    # --latent-mode/--latent-steps menulis ke berkas yang SAMA dan yang kedua
    # menimpa yang pertama tanpa peringatan — persis pasangan yang dibandingkan
    # di Tahap 0. Tag eksplisit tetap menang bila diberikan.
    # Mode "moi" WAJIB menyertakan beta di nama berkas: beta adalah sumbu bebas
    # (bukan biner seperti no-realign), jadi tanpa ini setiap nilai beta yang
    # berbeda saling menimpa satu sama lain diam-diam.
    _nr = "_norealign" if a.no_realign else ""
    _beta = f"_b{a.latent_beta:g}" if a.latent_mode == "moi" else ""
    suffix = f"_{a.tag}" if a.tag else f"_{a.latent_mode}{_nr}{_beta}_m{a.latent_steps}"
    path = OUT / f"channel_capacity_{a.model.replace('/', '_')}{suffix}.json"
    path.write_text(json.dumps(doc, indent=2))
    print(f"tersimpan → {path}")


if __name__ == "__main__":
    main()
