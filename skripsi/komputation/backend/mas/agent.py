"""
mas/agent.py
====================
`LatentAgent` — unit agent modular yang bisa dijalankan SENDIRI-SENDIRI.

Filosofi desain
---------------
Tiap agent = (spec prompt + mode KV + parser). Tidak ada agent yang "tahu"
tentang pipeline. Akibatnya kamu bisa:

    backend = LocalLLMBackend(...)
    judger  = load_agent("judger", backend)
    res     = judger.run(past_kv=some_kv, hypothesis="...", function_lib="...")
    print(res.text)              # output teks
    kv_describe(res.kv_cache)    # cek isi KV

…tanpa menjalankan seluruh pipeline. Ini yang bikin debugging prompt &
inspeksi KV jauh lebih cepat.

Mode KV (diteruskan ke LocalLLMBackend.run)
-------------------------------------------
  kv_only      : latent reasoning saja, tidak generate teks → cuma membentuk KV.
                 Untuk proposal/construct/consistency (front-end sequential).
  kv_and_text  : latent reasoning lalu generate teks dari KV.
                 Untuk judger/repair/feedback/mutation/crossover.
  text_only    : generate teks tanpa menyimpan KV (jarang dipakai di sini).

Spec dimuat dari `prompts.yaml` (lihat `load_agent` / `load_all_agents`).
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from jinja2 import Environment, StrictUndefined, Undefined

from llm.client import LocalLLMBackend, KVCache
from mas.kv_ops import kv_describe, kv_seq_len
from mas.parsers import PARSERS

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "factor.yaml"


class _VisibleUndefined(Undefined):
    """Untuk run standalone: variabel hilang dirender sebagai penanda terlihat
    alih-alih crash — supaya gampang lihat var mana yang belum diisi."""
    def __str__(self) -> str:  # noqa: D401
        return f"[[MISSING:{self._undefined_name}]]"


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    role: str
    mode: str = "kv_only"              # kv_only | kv_and_text | text_only
    system: str = ""                   # jinja template
    user: str = ""                     # jinja template
    latent_steps: Optional[int] = None
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None
    parser: Optional[Callable[[str], Any]] = None
    json_mode: bool = False
    # B11 — guided (constrained) decoding. Infrastrukturnya sudah ada dan sudah
    # tersambung di `llm/guided_decoding.py` + `LocalLLMBackend.run(json_schema=)`,
    # tetapi TAK ADA pemanggil yang pernah mengopernya, sehingga jalur itu mati.
    # Diisi dari `json_schema:` di prompts.yaml — boleh nama terdaftar di
    # JSON_SCHEMAS ("latent_construct") atau schema JSON penuh.
    # Efek: struktur output dipaksa valid token-per-token, bukan dijinjit prompt.
    # Biaya: latensi +10–20%. Batas: grammar menjamin BENTUK, bukan ISI.
    json_schema: Optional[Any] = None
    # Prefill: teks yang mengisi awal giliran asisten, dikirim sebagai input
    # dan dipasang kembali ke hasil dekode. Menutup kasus agen yang mewarisi
    # KV berisi objek JSON utuh dari agen hulu lalu melanjutkan seolah masih
    # di dalam objek itu — keluarannya mulai dari NILAI, tanpa pembuka, dan
    # gagal diurai. Isinya spesifik kontrak keluaran agen, karena itu
    # ditulis di prompts.yaml, bukan di lapisan engine.
    prefill: str = ""
    # NO-CROP default (prod): pertahankan jawaban yang di-generate di KV agar
    # agent berikut membaca output ASLI, bukan cuma vektor laten yang lossy.
    # Set False (di YAML) untuk perilaku lama (crop, anti-contamination).
    keep_answer_in_kv: bool = True


@dataclass
class AgentResult:
    role: str
    mode: str
    text: Optional[str]
    kv_cache: Optional[KVCache]
    duration_s: float
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    kv_seq_len: int = 0
    latent_steps: int = 0
    parsed: Any = None
    hidden_last: Any = None
    latent_vecs: Any = None
    input_ids: Any = None
    output_ids: Any = None
    latent_s: float = 0.0   # durasi "berpikir" (latent_pass)
    gen_s: float = 0.0      # durasi generate teks
    # B6: `latent_steps` di atas adalah anggaran yang DIMINTA; dua field ini
    # mencatat berapa langkah yang benar-benar berjalan dan kenapa ia berhenti
    # ("budget" = anggaran habis, "early_stop" = titik tetap, "off" = tak ada
    # rollout laten). Tanpa ini penghematan B6 hanya terlihat sebagai durasi.
    n_latent_steps: int = 0
    latent_stop: str = "off"

    @property
    def ok(self) -> bool:
        if self.mode == "kv_only":
            return self.kv_cache is not None
        return bool(self.text and self.text.strip())

    def describe(self) -> Dict[str, Any]:
        return {
            "role": self.role, "mode": self.mode, "ok": self.ok,
            "duration_s": round(self.duration_s, 3),
            "latent_steps": self.latent_steps,
            "n_latent_steps": self.n_latent_steps,
            "latent_stop": self.latent_stop,
            "text_len": len(self.text) if self.text else 0,
            "n_out_tok": self.n_output_tokens,
            "kv": kv_describe(self.kv_cache),
            "parsed": self.parsed,
        }


# ─────────────────────────────────────────────────────────────────────────────

class LatentAgent:
    """Satu agent latent yang berdiri sendiri."""

    def __init__(
        self,
        spec: AgentSpec,
        backend: LocalLLMBackend,
        *,
        strict_vars: bool = True,
        runlog: Any = None,
    ) -> None:
        self.spec = spec
        self.backend = backend
        self.runlog = runlog
        self._env = Environment(
            undefined=StrictUndefined if strict_vars else _VisibleUndefined
        )

    # ── prompt rendering ─────────────────────────────────────────────────────

    def render(self, **vars: Any) -> tuple[str, str]:
        system = self._env.from_string(self.spec.system).render(**vars).strip()
        user = self._env.from_string(self.spec.user).render(**vars).strip()
        return system, user

    # ── eksekusi ─────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        past_kv: Optional[KVCache] = None,
        runlog: Any = None,
        role: Optional[str] = None,
        mode_override: Optional[str] = None,
        **vars: Any,
    ) -> AgentResult:
        """Render prompt → panggil backend → parse → AgentResult.

        `past_kv` adalah KV dari agent sebelumnya (sudah di-clone oleh
        orkestrator bila perlu — agent TIDAK meng-clone sendiri).

        `role` (opsional) MENIMPA label snapshot/log untuk call ini saja (spec
        tetap utuh) — dipakai mis. saat probe introspect dipakai ulang untuk
        membaca KV mutation/crossover agar file llm_outputs-nya jelas asalnya.

        `mode_override` (opsional) MENIMPA `spec.mode` untuk call ini saja —
        dipakai orkestrator (FrontEndPipeline) untuk menerapkan comm_mode global
        (text / kv_and_text / kv) tanpa mengubah spec agen di prompts.yaml.
        """
        rl = runlog or self.runlog
        eff_role = role or self.spec.role
        eff_mode = mode_override or self.spec.mode
        system, user = self.render(**vars)

        step_cm = rl.step(eff_role) if rl is not None else nullcontext()
        t0 = time.time()
        with step_cm:
            res = self.backend.build_messages_and_run(
                user_prompt=user,
                system_prompt=system,
                past_key_values=past_kv,
                mode=eff_mode,
                role=eff_role,
                latent_steps=self.spec.latent_steps,
                temperature=self.spec.temperature,
                max_new_tokens=self.spec.max_new_tokens,
                json_mode=self.spec.json_mode,
                json_schema=self.spec.json_schema,
                crop_after_generate=not self.spec.keep_answer_in_kv,
                prefill=self.spec.prefill,
            )
        dur = time.time() - t0

        # latent steps efektif: override spec, else default engine
        eff_latent = self.spec.latent_steps
        if eff_latent is None:
            eff_latent = getattr(getattr(self.backend, "_engine", None), "latent_steps", 0)

        parsed = None
        if self.spec.parser is not None and res.text:
            try:
                parsed = self.spec.parser(res.text)
            except Exception as e:  # noqa: BLE001
                if rl is not None:
                    rl.warn(f"parser failed for {self.spec.role}", err=repr(e))

        out = AgentResult(
            role=self.spec.role,
            mode=eff_mode,
            text=res.text,
            kv_cache=res.kv_cache,
            duration_s=dur,
            n_input_tokens=int(res.input_ids.shape[-1]) if res.input_ids is not None else 0,
            n_output_tokens=int(res.output_ids.shape[-1]) if res.output_ids is not None else 0,
            kv_seq_len=kv_seq_len(res.kv_cache),
            latent_steps=eff_latent or 0,
            parsed=parsed,
            hidden_last=res.hidden_last,
            latent_vecs=res.latent_vecs,
            input_ids=res.input_ids,
            output_ids=res.output_ids,
            latent_s=getattr(res, "latent_s", 0.0),
            gen_s=getattr(res, "gen_s", 0.0),
            n_latent_steps=getattr(res, "n_latent_steps", 0),
            latent_stop=getattr(res, "latent_stop", "off"),
        )
        if rl is not None:
            rl.event("agent_done", **out.describe())
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Loader dari prompts.yaml
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_json_schema(value: Any) -> Optional[Any]:
    """`json_schema:` di YAML boleh berupa nama terdaftar atau schema penuh.

    Dikontrol lewat env `LATENTMAS_GUIDED` (di-set dari `settings.guided_decoding`
    oleh `pipeline/loop.py` SEBELUM agen dimuat — lihat komentar di sana).

    PROMOSI 2026-08-07 (B16, lab/HASIL_A8.md §4b): diukur sebagai lengan
    (`innovate_guided`) dan LULUS — pada rantai `innovate` guided decoding di
    construct menaikkan keandalan 4/6 -> 6/6 tanpa mengorbankan cakupan pustaka
    maupun mutu sinyal, sehingga default produksi kini AKTIF
    (`settings.guided_decoding = True`). Ini TERIKAT PADA rantai `innovate`: pada
    rantai lama (`design`) hasilnya justru negatif (lengan `full_guided`, §3.4).
    Skrip lab yang belum memakai settings.py (mis. pemanggilan `_load_specs`
    langsung) tetap default ke env var mentah (mati bila tak di-set) — lihat
    pemanggil masing-masing.

    Nama yang tak dikenal DIABAIKAN dengan peringatan, bukan crash: salah ketik
    di prompts.yaml tak boleh menjatuhkan seluruh run — ia cuma mematikan guided
    decoding untuk agen itu, dan itu perilaku lama yang sudah teruji.
    """
    if value is None:
        return None
    if os.environ.get("LATENTMAS_GUIDED", "0") != "1":
        return None
    if isinstance(value, dict):
        return value
    from llm.guided_decoding import JSON_SCHEMAS
    schema = JSON_SCHEMAS.get(str(value))
    if schema is None:
        print(f"[agent] json_schema {value!r} tak dikenal; guided decoding dilewati. "
              f"Tersedia: {sorted(JSON_SCHEMAS)}")
    return schema


def _load_specs(path: Path = _PROMPTS_PATH) -> Dict[str, AgentSpec]:
    import yaml
    raw = yaml.safe_load(path.read_text())
    specs: Dict[str, AgentSpec] = {}
    for name, cfg in (raw.get("agents") or {}).items():
        parser_name = cfg.get("parser", "none")
        specs[name] = AgentSpec(
            role=cfg.get("role", name),
            mode=cfg.get("mode", "kv_only"),
            system=cfg.get("system", ""),
            user=cfg.get("user", ""),
            latent_steps=cfg.get("latent_steps"),
            temperature=cfg.get("temperature"),
            max_new_tokens=cfg.get("max_new_tokens"),
            parser=PARSERS.get(parser_name),
            json_mode=cfg.get("json_mode", False),
            json_schema=_resolve_json_schema(cfg.get("json_schema")),
            prefill=cfg.get("prefill", ""),
            keep_answer_in_kv=cfg.get("keep_answer_in_kv", True),
        )
    return specs


def load_agent(
    name: str,
    backend: LocalLLMBackend,
    *,
    strict_vars: bool = True,
    runlog: Any = None,
    path: Path = _PROMPTS_PATH,
) -> LatentAgent:
    """Muat satu agent by name dari prompts.yaml."""
    specs = _load_specs(path)
    if name not in specs:
        raise KeyError(f"agent '{name}' tidak ada di {path}. "
                       f"Tersedia: {sorted(specs)}")
    return LatentAgent(specs[name], backend, strict_vars=strict_vars, runlog=runlog)


def load_all_agents(
    backend: LocalLLMBackend,
    *,
    strict_vars: bool = True,
    runlog: Any = None,
    path: Path = _PROMPTS_PATH,
) -> Dict[str, LatentAgent]:
    """Muat semua agent sekaligus (untuk pipeline)."""
    return {
        name: LatentAgent(spec, backend, strict_vars=strict_vars, runlog=runlog)
        for name, spec in _load_specs(path).items()
    }
