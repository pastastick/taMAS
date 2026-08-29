"""
mas/pipeline.py
====================
Orkestrator yang menyambung agent + DISTRIBUSI KV yang benar.

Inti modul ini bukan "memanggil agent berurutan", tapi **mengelola KV dengan
disiplin isolasi**. Aturan emas:

    Sebuah KV yang akan dibaca oleh > 1 konsumen HARUS di-`kv_deepcopy` dulu
    untuk tiap konsumen.

Kenapa wajib: `LocalLLMBackend.run()` memutasi `past_key_values` IN-PLACE
(latent_pass meng-extend objek DynamicCache yang dioper). Jadi kalau kv_consist
dioper apa adanya ke judger, objek kv_consist ikut ter-extend oleh prompt judger
— lalu saat feedback memakai kv_consist yang sama, ia melihat token judger.
Itu kontaminasi senyap. `kv_deepcopy` memutus rantai itu.

Aliran KV (front-end NO-CROP, dipromosikan dari prod — DESIGN drift fix):

  FRONT-END (sequential, NO-CROP: jawaban tiap agent tetap di KV → agent berikut
  membaca output ASLI, bukan cuma vektor laten lossy. keep_answer_in_kv=true):
    seed → proposal(kv_and_text) → design(kv_and_text) → construct(kv_and_text)
       direction(teks)         baca HYPOTHESIS dr KV   baca palette dr KV →
                                                         JSON {hypothesis,factors}
    construct = TERMINAL EMITTER (menggantikan consistency+judger lama). Ekspresi
    diekstrak parser construct_json (JSON, fallback hypothesis_exprs).
  KONSUMEN kv_construct (kv_final):
    repair(kv_and_text) ×N     ← deepcopy(kv_construct)  per attempt (baseline sama)
    feedback(kv_and_text)      ← deepcopy(kv_final)      (loop._run_latent_feedback)

  EVOLUTION (GUIDANCE kv_only → re-entry front-end, bounded per-generasi):
    mutation   (kv_only) ← past_kv=None + TEKS 1 parent → guidance_kv (arah refine)
    crossover  (kv_only) ← past_kv=None + TEKS k parent → guidance_kv (arah fusi)
  guidance_kv lalu MENYEMAI front-end: run(seed_kv=guidance_kv) →
    proposal → design → construct → gate/repair (lihat run_evolution).
  Materi parent masuk sebagai TEKS & KV TIDAK diwariskan antar-generasi (trajectory
  .kv_cache tetap None) → KV per ronde terbatas (~guidance + front-end ≈ 3k) → tak
  ada over-KV / collapse lintas-generasi (akar regresi 2026-06-02).
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from llm.client import LocalLLMBackend, KVCache
from mas import kv_ops
from mas.agent import LatentAgent, AgentResult, load_all_agents
from mas.parsers import (
    HypothesisExprs, ConstructResult, PASS_SENTINEL,
    parse_repair_multi, parse_hypothesis_exprs, parse_construct_json,
)
from mas.operator_families import families_of, diversity_hint

# Type alias untuk gate: (expression) -> (ok, error_message)
QualityGate = Callable[[str], "tuple[bool, str]"]
# Type alias untuk backtest: (expression, hypothesis) -> dict hasil
Backtester = Callable[[str, str], dict]

# Medium komunikasi antar-agen (lihat FrontEndPipeline.__init__).
_COMM_MODES = ("text", "kv_and_text", "kv", "summary")


# ── B14: ringkasan terstruktur untuk handoff "konteks segar" ────────────────
# Mode `text` sudah memberi tiap agen konteks bersih, tetapi ia mengoper SELURUH
# teks agen hulu. Mode `summary` mengoper hanya bagian yang KONTRAKTUAL dari
# keluaran itu. Ekstraksinya DETERMINISTIK (regex atas kontrak yang sudah
# ditegakkan prompt) — tanpa panggilan LLM tambahan, jadi ia tak menambah hop,
# latensi, atau sumber kegagalan baru.
_HYP_LINE = re.compile(r"^\s*HYPOTHESIS\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def summarize_for_handoff(agent: str, text: str, max_chars: int = 1800) -> str:
    """Ringkas keluaran satu agen jadi materi handoff terstruktur (B14).

    Kontrak yang dimanfaatkan (sudah ditegakkan prompts.yaml):
      proposal  — baris terakhir berformat "HYPOTHESIS: ..."
      innovate  — hal TERAKHIR yang ditulis adalah blok JSON {hypothesis,
                  hypothesis_variants, recipes}

    FAIL-OPEN: kalau kontraknya tidak terpenuhi, kembalikan teks asli yang
    dipotong. Ini disengaja — B14 tidak boleh mengubah keandalan menjadi
    lebih buruk hanya karena satu keluaran tak sesuai format; kalau ia
    memangkas jadi kosong, lengan ini akan terlihat buruk karena parsing,
    bukan karena mediumnya.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if agent == "proposal":
        hits = _HYP_LINE.findall(t)
        if hits:
            return f"HYPOTHESIS: {hits[-1].strip()}"
    elif agent in ("innovate", "design"):
        # Ambil blok JSON TERLUAR yang paling akhir ditulis.
        # Catatan (ditemukan lab/test_b14_summary.py): mencari dari `{` TERAKHIR
        # itu salah — `{` terakhir adalah objek resep yang BERSARANG di dalam
        # blok utama, sehingga yang terekstrak cuma satu resep dan seluruh
        # hypothesis/variants-nya hilang. Karena itu semua kandidat dievaluasi,
        # lalu dipilih yang berakhir paling belakang (dan, bila seri, yang mulai
        # paling awal = paling luar).
        best: "tuple[int, int, str] | None" = None
        for start, ch in enumerate(t):
            if ch != "{":
                continue
            depth = 0
            for i in range(start, len(t)):
                if t[i] == "{":
                    depth += 1
                elif t[i] == "}":
                    depth -= 1
                    if depth == 0:
                        blob = t[start:i + 1]
                        try:
                            json.loads(blob)
                        except ValueError:
                            break
                        cand = (i, -start, blob)
                        if best is None or cand[:2] > best[:2]:
                            best = cand
                        break
        if best is not None:
            return best[2][:max_chars]
    return t[:max_chars]


def default_quality_gate(expression: str) -> "tuple[bool, str]":
    """Gate AST/arity deterministik. Lazy-import parser asli; fallback ke
    pengecekan dasar bila modul belum ada di branch ini."""
    if not expression or not expression.strip():
        return False, "empty expression"
    try:
        from dsl.expr_parser import parse_expression  # type: ignore
        parse_expression(expression)  # raises on invalid
        return True, ""
    except ImportError:
        # fallback ringan: cek kurung balance + variabel dikenal
        if expression.count("(") != expression.count(")"):
            return False, "unbalanced parentheses"
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrontEndOutput:
    hypothesis: str
    expressions: List[str]             # SEMUA ekspresi lolos regulator (≥1) → LightGBM combined
    kv_consist: Optional[KVCache]      # baseline (pristine) — KV construct, dipakai repair
    kv_judger: Optional[KVCache]       # = KV construct (terminal emitter; nama dipertahankan utk back-compat loop)
    judger_text: str                   # = teks construct (JSON faktor) — back-compat field name
    repaired: bool = False
    repair_attempts: int = 0
    gate_error: str = ""
    # KV yang BENAR-BENAR menghasilkan ekspresi final yang diterima:
    #   kv_repair bila repair berjalan & sukses, selain itu kv_construct.
    # Inilah seed feedback (trajectory-coherent) — lihat loop._run_latent_feedback.
    kv_final: Optional[KVCache] = None
    # Faktor mentah dari construct: [{name, expression, explanation}] (intent per faktor,
    # dipakai repair sadar-intent & dicatat di StrategyTrajectory.factors).
    factors: List[dict] = field(default_factory=list)
    # Jejak gate per-ekspresi: [{name, expression, ok, reason, repaired_by}] → audit
    # keputusan gate lengkap (console.log + trajectory.extra_info).
    gate_log: List[dict] = field(default_factory=list)

    @property
    def expression(self) -> str:
        """Ekspresi pertama (untuk logging/penamaan; back-compat single-expr)."""
        return self.expressions[0] if self.expressions else ""


class FrontEndPipeline:
    """proposal → design → construct → [regulator-gate → repair]  (NO-CROP).

    Front-end dipromosikan dari prod: proposal merumuskan hipotesis, design memilih
    palette VARIABLE+FUNCTION (membaca hipotesis ASLI dari KV — no-crop memperbaiki
    drift hipotesis), construct = TERMINAL EMITTER yang menulis JSON
    {hypothesis, factors:[{name,expression,explanation}]}. Ekspresi diparse
    (construct_json) lalu di-gate. Gate = FactorRegulator PENUH (parsable +
    complexity SL/PC/ER + redundansi alpha-zoo) bila tersedia, fallback
    `default_quality_gate`. Repair hanya bila SEMUA ekspresi gagal gate.
    """

    def __init__(
        self,
        backend: LocalLLMBackend,
        *,
        runlog: Any = None,
        quality_gate: Optional[QualityGate] = None,
        max_repair_attempts: int = 3,
        agents: Optional[dict] = None,
        use_regulator: bool = True,
        comm_mode: str = "kv",
        chain: Optional[tuple] = None,
        free_form: Optional[bool] = None,
    ) -> None:
        self.backend = backend
        self.runlog = runlog
        self.max_repair_attempts = max_repair_attempts
        self.agents: dict = agents or load_all_agents(backend, runlog=runlog)
        # ── chain: SUSUNAN agen front-end (sumbu A8, ablasi arsitektur) ──────
        # Agen terakhir WAJIB emitter (menulis JSON faktor); agen sebelumnya
        # menyempitkan arah. Varian yang dipakai eksperimen:
        #   ("proposal","design","construct")    rantai LAMA (sebelum B16)
        #   ("proposal","construct")             tanpa design
        #   ("construct",)                       direction langsung ke builder
        #   ("proposal","innovate","construct")  RANTAI PRODUKSI (sejak B16)
        #
        # DEFAULT diganti 2026-08-07 (B16, lab/HASIL_A8.md): A8 mengukur `design`
        # TIDAK berpengaruh signifikan terhadap IC (Welch t=0,79), sementara
        # `innovate` unggul pada |IC|/run (+60%) dan cakupan pencarian (23 vs 13
        # fungsi DSL disentuh). Keputusan user, ditegaskan eksplisit setelah
        # meninjau hasil A8 — lihat lab/HASIL_A8.md §5 untuk angka ronde final.
        # Untuk mereplikasi rantai lama: `chain=("proposal","design","construct")`.
        self.chain: tuple = tuple(chain or ("proposal", "innovate", "construct"))
        unknown = [a for a in self.chain if a not in self.agents]
        if unknown:
            raise KeyError(f"agen {unknown} tak ada di prompts.yaml; "
                           f"tersedia: {sorted(self.agents)}")
        if len(self.chain) < 1:
            raise ValueError("chain harus punya minimal satu agen (emitter)")
        # free_form: melepas klem "FIDELITY FIRST" di emitter. Menyala sendiri
        # bila rantai memakai `innovate`, karena agen itu SENGAJA membelokkan
        # hipotesis — menuntut kesetiaan ke hipotesis asal sekaligus menyuruhnya
        # menyimpang adalah dua perintah yang saling meniadakan, dan hasilnya
        # akan tertarik balik ke idiom yang sama. Bisa dipaksa lewat argumen
        # untuk memisahkan efek "ganti agen" dari efek "lepas klem" (lengan
        # `innovate_fid` di lab/gpu_suite.py).
        self.free_form = (free_form if free_form is not None
                          else ("innovate" in self.chain))
        # ── comm_mode: medium komunikasi antar-agen (variabel eksperimen utama) ──
        #   "kv"          : hanya construct & feedback yang generate TEKS; proposal/
        #                   design/guidance kv_only (laten). Handoff via KV-cache.
        #                   (= perilaku default; mode pure-latent paling hemat token).
        #   "kv_and_text" : SEMUA agen generate teks, tetapi handoff tetap via KV-cache.
        #   "text"        : SEMUA agen generate teks, handoff via TEKS (no KV).
        #                   = baseline TEKS terkontrol (prompt sama, beda medium saja).
        #   "summary"     : B14 — seperti `text` (konteks bersih tiap agen), tetapi
        #                   yang dioper adalah RINGKASAN TERSTRUKTUR keluaran hulu
        #                   (baris HYPOTHESIS proposal; blok JSON innovate), bukan
        #                   seluruh teksnya. Ekstraksi deterministik, tanpa
        #                   panggilan LLM tambahan — lihat summarize_for_handoff().
        if comm_mode not in _COMM_MODES:
            raise ValueError(f"comm_mode harus salah satu dari {_COMM_MODES}, "
                             f"dapat {comm_mode!r}")
        self.comm_mode = comm_mode
        # Riwayat family operator faktor yang BARU diterima → diversity hint
        # (soft, hybrid). Lintas-run dalam satu instance pipeline (mis. original →
        # mutation → crossover) → tekanan anti-monokultur antar-faktor.
        self._recent_families: deque = deque(maxlen=12)
        self._regulator = None
        if quality_gate is not None:
            self.gate = quality_gate
        elif use_regulator:
            self.gate, self._regulator = self._build_regulator_gate(runlog)
        else:
            self.gate = default_quality_gate

    def _a(self, name: str) -> LatentAgent:
        return self.agents[name]

    @property
    def _is_text(self) -> bool:
        """True bila handoff antar-agen lewat TEKS (no KV).

        Mencakup `summary` (B14): mediumnya tetap teks dan konteksnya tetap
        bersih — yang berbeda hanya APA yang dioper (ringkasan terstruktur,
        bukan seluruh keluaran). Menyatukannya di sini menjaga agar tak ada
        cabang alur baru yang harus dijaga tetap sinkron.
        """
        return self.comm_mode in ("text", "summary")

    @property
    def _is_summary(self) -> bool:
        """True bila materi handoff diringkas dulu (B14)."""
        return self.comm_mode == "summary"

    def _agent_mode(self, native_mode: str) -> str:
        """Mode efektif satu agen di bawah comm_mode global.

          - "text"/"summary" : semua agen → text_only.
          - "kv_and_text"    : agen yang aslinya kv_only (proposal/design/guidance)
                               dinaikkan → kv_and_text (ikut generate teks); lain tetap.
          - "kv"             : pakai mode asli dari spec (default; tak ada override).
        """
        if self._is_text:
            return "text_only"
        if self.comm_mode == "kv_and_text":
            return "kv_and_text" if native_mode == "kv_only" else native_mode
        return native_mode
    # [terjawab — skripsi Bab 4 §Kendali Mutu Faktor]: rangkaian gate berurutan
    #   (parsable → arity → variabel → degenerate → redundansi S(f) → kompleksitas
    #   SL/PC/ER) + kesetiaan/divergensi terhadap C(f) paper.
    @staticmethod
    def _build_regulator_gate(runlog: Any = None) -> "tuple[QualityGate, Any]":
        """Bangun gate berbasis FactorRegulator PENUH (dari FACTOR_COSTEER_SETTINGS).
        Fallback ke default_quality_gate bila modul/zoo tak tersedia."""
        try:
            # Pre-import factor_ast lebih dulu: memutus circular import
            # (factor_regulator → coder/__init__ → evaluators → factor_regulator)
            # yang terjadi bila factor_regulator di-import COLD. Di jalur latent,
            # FrontEndPipeline dibangun SEBELUM coder di-instansiasi, jadi tanpa ini
            # gate diam-diam fallback ke sintaks (regulator tak aktif).
            import dsl.factor_ast  # noqa: F401
            from gate.factor_regulator import (
                FactorRegulator, validate_function_arity,
                validate_known_variables, validate_no_degenerate_args,
                validate_semantics,
            )
            # `dsl/config.py` (FACTOR_COSTEER_SETTINGS, warisan CoSTEER RD-Agent)
            # ikut terhapus di rombakan 9d4e0bf, sementara baris ini tidak.
            # Akibatnya ImportError-nya ditelan `except Exception` di bawah dan
            # SELURUH rantai gate (arity, variabel, degenerate, semantik,
            # eksekusi) diam-diam mati — gate jatuh ke `default_quality_gate`
            # yang hanya memeriksa sintaks, sehingga `IF(...)`, `TS_RESIDUAL(...)`
            # (fungsi yang tak ada) dan `RANK($x, 60)` (arity salah) semuanya
            # LOLOS lalu baru meledak saat evaluasi.
            #
            # Objek settings ini hanya menyediakan OVERRIDE opsional: keempat
            # nilainya dibaca lewat getattr berdefault, dan defaultnya sama
            # dengan setelan produksi yang ditulis di metodologi (SL<=300,
            # ER<=6, duplikasi=8). Jadi ketiadaannya tidak boleh mematikan gate.
            try:
                from dsl.config import FACTOR_COSTEER_SETTINGS as S
            except ImportError:
                S = None
            reg = FactorRegulator(
                factor_zoo_path=getattr(S, "factor_zoo_path", None),
                duplication_threshold=getattr(S, "duplication_threshold", 8),
                symbol_length_threshold=getattr(S, "symbol_length_threshold", 300),
                base_features_threshold=getattr(S, "base_features_threshold", 6),
            )
            # B12 — execution gate. Semua gate di atas STRUKTURAL; ekspresi yang
            # kolomnya kosong/konstan saat dijalankan tetap lolos (11 dari 198 di
            # G5). Ini gate termurah yang selama ini hilang: ±1 detik CPU, tanpa
            # GPU. get_execution_gate() mengembalikan None bila dimatikan lewat
            # env, dan fail-open bila datanya tak ada.
            from gate.execution_gate import get_execution_gate
            exec_gate = get_execution_gate()
        except Exception as e:  # noqa: BLE001
            if runlog:
                runlog.warn("FactorRegulator unavailable; fallback to syntax gate",
                            err=repr(e))
            return default_quality_gate, None

        def gate(expr: str) -> "tuple[bool, str]":
            if not expr or not expr.strip():
                return False, "empty expression"
            try:
                if not reg.is_parsable(expr):
                    return False, "unparsable expression"
                # arity DETERMINISTIK: tangkap RANK($x, 7) dkk SEBELUM eksekusi
                # (kelas bug yang dulu lolos gate lalu crash di factor.py). Pesan
                # error ramah-LLM (termasuk hint cross-sectional→TS_) → repair.
                ar_ok, ar_errs = validate_function_arity(expr)
                if not ar_ok:
                    return False, "arity: " + " ".join(ar_errs[:2])
                # #1 variabel halusinasi ($return_1d) — parser/arity tak menangkap.
                kv_ok, kv_errs = validate_known_variables(expr)
                if not kv_ok:
                    return False, "variable: " + " ".join(kv_errs[:2])
                # #2 degenerate-args (REGRESI(x,x)) — arity valid tapi faktor mati.
                dg_ok, dg_errs = validate_no_degenerate_args(expr)
                if not dg_ok:
                    return False, "degenerate: " + dg_errs[0]
                # #3 semantik-numerik (window degenerate, kondisi non-boolean,
                # ambang pada persentil / volume absolut). Gate lain semuanya
                # STRUKTURAL → ekspresi konstan/NaN-total masih lolos. Lihat
                # lab/audit_batch.py untuk kuantifikasi di run 2026-07-05.
                sm_ok, sm_errs = validate_semantics(expr)
                if not sm_ok:
                    return False, "semantics: " + " ".join(sm_errs[:2])
                ok, ev = reg.evaluate(expr)
                if not ok or ev is None:
                    return False, "regulator evaluate failed"
                if not reg.is_expression_acceptable(ev):
                    return False, (f"regulator reject: sl={ev.get('symbol_length')}, "
                                   f"base_feat={ev.get('num_base_features')}, "
                                   f"dup={ev.get('duplicated_subtree_size')}")
                # #4 EKSEKUSI (B12) — dijalankan TERAKHIR karena paling mahal
                # (±1 detik CPU) dan hanya berguna pada ekspresi yang sudah sah
                # secara struktural. Ini satu-satunya gate yang benar-benar
                # MENJALANKAN ekspresi, jadi satu-satunya yang bisa menangkap
                # kolom kosong/konstan (AUDIT §S1).
                if exec_gate is not None:
                    ex_ok, ex_err = exec_gate.check(expr)
                    if not ex_ok:
                        return False, "execution: " + ex_err
                return True, ""
            except Exception as e:  # noqa: BLE001
                return False, f"{type(e).__name__}: {e}"

        return gate, reg

    def _register_factors(self, exprs: List[str]) -> None:
        """Daftarkan ekspresi lolos ke alpha-zoo regulator (dedup intra-run)."""
        reg = self._regulator
        if reg is None or not exprs:
            return
        try:
            names = [f"latent_{i}" for i in range(len(exprs))]
            reg.add_factor(names, exprs)
        except Exception as e:  # noqa: BLE001
            if self.runlog:
                self.runlog.warn("regulator add_factor failed", err=repr(e))

    def run(
        self,
        *,
        direction: str,
        seed_kv: Optional[KVCache] = None,
        handoff: Optional[str] = None,
        market_context: str = "",
        prior_feedback: str = "",
        negative_hint: str = "",
    ) -> FrontEndOutput:
        rl = self.runlog

        # handoff proposal — dari mana arah riset dibaca:
        #   'text' : direction disuntik sebagai TEKS (ronde original, belum ada
        #            latent memory apa pun untuk dibaca).
        #   'kv'   : arah dari Director (mutation/crossover) SUDAH ada di seed_kv →
        #            proposal membacanya dari latent memory (else-branch prompt),
        #            JANGAN re-inject direction generik (akan menarik balik ke arah
        #            umum, bukan refinement spesifik Director).
        # Auto bila None: ada seed_kv (guidance) → 'kv'; else → 'text'. Aman karena
        # di latent mode _pipeline_kv di-reset tiap iterasi (loop.py:490), jadi
        # seed_kv hanya non-None saat evolution men-seed guidance_kv. Bila guidance
        # gagal hasilkan KV (seed_kv None) → fallback ke 'text' (pakai direction).
        if handoff is None:
            handoff = "kv" if seed_kv is not None else "text"

        # Diversity hint (soft) dari family faktor yang sudah diterima sebelumnya —
        # mengarahkan ke family rich yang jarang dipakai TANPA mengorbankan kesetiaan
        # ke hipotesis. Disuntik ke construct (yang menulis ekspresi); proposal SENGAJA
        # tidak (jaga hipotesis tetap di altitude mekanisme, bukan operator).
        dhint = diversity_hint(list(self._recent_families))

        # ── front-end: proposal → design → construct ─────────────────────────
        # comm_mode menentukan medium handoff + mode tiap agen:
        #   text        : no KV; tiap agen baca TEKS output agen sebelumnya.
        #   kv/kv_and_text : handoff via KV (proposal: handoff='text' original /
        #                    'kv' mutation-crossover). Beda keduanya hanya apakah
        #                    proposal/design ikut men-decode teks (kv_and_text) atau
        #                    laten murni (kv).
        r_con, kv_construct, cr = self._run_chain(
            direction=direction, seed_kv=seed_kv, handoff=handoff,
            market_context=market_context, prior_feedback=prior_feedback,
            negative_hint=negative_hint, diversity_hint=dhint,
        )

        hypothesis, factors = ("", []) if cr is None else (cr.hypothesis, list(cr.factors))

        # ── regulator-gate semua ekspresi; repair hanya bila SEMUA gagal ──────
        passing, kv_final, repaired, attempts, gate_err, gate_log = \
            self._gate_and_repair_factors(factors, kv_construct)

        # Catat family faktor yang diterima → diversity hint run berikutnya.
        for e in passing:
            self._recent_families.append(families_of(e))
        return FrontEndOutput(
            hypothesis=hypothesis, expressions=passing,
            kv_consist=kv_construct, kv_judger=kv_construct,
            judger_text=r_con.text or "", kv_final=kv_final,
            repaired=repaired, repair_attempts=attempts, gate_error=gate_err,
            factors=factors, gate_log=gate_log,
        )

    # Mode NATIF tiap agen front-end sebelum comm_mode diterapkan. Agen perantara
    # bernalar laten (kv_only); emitter wajib menulis teks.
    _NATIVE_MODE = {"proposal": "kv_only", "design": "kv_only",
                    "innovate": "kv_only", "construct": "kv_and_text"}

    def _run_chain(
        self,
        *,
        direction: str,
        seed_kv: Optional[KVCache],
        handoff: str,
        market_context: str,
        prior_feedback: str,
        negative_hint: str,
        diversity_hint: str,
    ) -> "tuple[AgentResult, Optional[KVCache], Optional[ConstructResult]]":
        """Jalankan `self.chain` berurutan dan kembalikan (hasil emitter, KV, parsed).

        Satu implementasi untuk SEMUA susunan rantai dan SEMUA comm_mode. Dulu
        ada dua salinan alur (cabang `text` dan cabang KV) yang harus dijaga
        tetap sinkron; menambah satu varian rantai berarti menambah dua cabang
        lagi. Di sini medium hanya menentukan (a) apakah `past_kv` diteruskan dan
        (b) `handoff` mana yang dirender — bukan alurnya.

        Aliran materi antar-agen:
          text : tiap agen membaca TEKS agen sebelumnya lewat variabel prompt.
          kv   : tiap agen membaca KV agen sebelumnya; variabel teks kosong.
        """
        rl = self.runlog
        is_text = self._is_text
        # Teks yang terkumpul sepanjang rantai — dipakai HANYA di mode text.
        hyp_text = ""            # keluaran agen hipotesis (proposal)
        prior_parts: List[str] = []   # semua keluaran hulu, untuk emitter
        prev_kv = seed_kv
        # Apakah pustaka fungsi PENUH sudah masuk KV oleh agen hulu (B4). Hanya
        # design/innovate yang membawanya; proposal tidak. Kalau design dipangkas
        # (A8), emitter WAJIB memuat pustakanya sendiri — karena itu ini dihitung
        # dari rantai yang benar-benar berjalan, bukan diasumsikan.
        lib_in_kv = False
        res: Optional[AgentResult] = None
        # KV persis SEBELUM emitter dijalankan — titik berangkat yang bersih bila
        # output emitter tak bisa diparse.
        kv_before_emitter: Optional[KVCache] = None

        for i, name in enumerate(self.chain):
            is_last = i == len(self.chain) - 1
            if is_last:
                kv_before_emitter = prev_kv
            mode = self._agent_mode(self._NATIVE_MODE.get(name, "kv_and_text"))
            # Agen pertama membaca arah; sisanya membaca agen sebelumnya.
            eff_handoff = handoff if i == 0 else ("text" if is_text else "kv")
            kw: dict = {"handoff": eff_handoff}
            if name == "proposal":
                kw.update(direction=direction, market_context=market_context,
                          prior_feedback=prior_feedback, negative_hint=negative_hint)
            elif name in ("design", "innovate"):
                kw.update(hypothesis_text=hyp_text if is_text else "")
            elif name == "construct":
                # Emitter pertama dalam rantai (mis. varian `("construct",)`)
                # tak punya hulu: arah riset masuk langsung sebagai teks.
                kw.update(
                    prior_factors=("\n\n".join(p for p in prior_parts if p).strip()
                                   if i > 0 else direction),
                    diversity_hint=diversity_hint,
                    lib_in_kv=lib_in_kv,
                    free_form=self.free_form,
                    # Emitter yang berjalan SENDIRIAN tak boleh diberi tahu bahwa
                    # ada hipotesis & palette dari agen hulu — kalimat itu akan
                    # menyuruhnya membaca sesuatu yang tak pernah ada, dan lengan
                    # `direct` akan kalah karena promptnya berbohong, bukan karena
                    # rantai pendek memang lebih buruk.
                    from_direction=(i == 0),
                )
            res = self._a(name).run(
                past_kv=None if is_text else prev_kv, mode_override=mode, **kw)
            prev_kv = res.kv_cache
            if name in ("design", "innovate"):
                lib_in_kv = not is_text
            text = (res.text or "").strip()
            if text:
                # B14: di mode `summary` yang dioper ke hilir adalah ringkasan
                # TERSTRUKTUR keluaran ini, bukan seluruh teksnya. `text` asli
                # tetap utuh di AgentResult (untuk log & sumbu A7), jadi yang
                # menyempit hanya materi handoff.
                handoff_text = (summarize_for_handoff(name, text)
                                if self._is_summary else text)
                if handoff_text:
                    prior_parts.append(handoff_text)
                if name == "proposal":
                    hyp_text = handoff_text or text
            if is_last:
                break

        assert res is not None  # chain dijamin non-kosong di __init__
        kv_last = res.kv_cache
        cr: Optional[ConstructResult] = res.parsed
        if cr is None:
            # Retry sekali dari KV agen SEBELUM emitter. `prev_kv` sudah menunjuk
            # KV emitter (yang berisi output rusaknya), jadi dipakai ulang akan
            # meminta model melanjutkan kekacauannya sendiri.
            if rl: rl.warn("construct output unparseable; retry once",
                           head=(res.text or "")[:120])
            emitter = self.chain[-1]
            retry_kv = (kv_ops.kv_deepcopy(kv_before_emitter)
                        if (not is_text and kv_before_emitter is not None) else None)
            res = self._a(emitter).run(
                past_kv=retry_kv,
                handoff="text" if retry_kv is None else "kv",
                prior_factors=("\n\n".join(p for p in prior_parts if p).strip()
                               if len(self.chain) > 1 else direction),
                diversity_hint=diversity_hint, lib_in_kv=lib_in_kv,
                free_form=self.free_form, from_direction=(len(self.chain) == 1),
                mode_override=self._agent_mode(self._NATIVE_MODE.get(emitter, "kv_and_text")),
            )
            kv_last = res.kv_cache
            cr = res.parsed
        return res, kv_last, cr

    def run_evolution(
        self,
        *,
        kind: str,                       # "mutation" | "crossover"
        parent_text: str,
        n_parents: int = 1,
        direction: str = "",
        negative_hint: str = "",
    ) -> FrontEndOutput:
        """Evolution GUIDANCE → re-entry front-end (bounded per-generasi).

          mutation  : arahkan refine SATU target (eksploitasi).
          crossover : arahkan fusi k parent (eksplorasi).

        Agent guidance (`mutation`/`crossover`, mode **kv_only**) di-seed dari
        **None** dan membaca materi parent sebagai **TEKS** (`parent_text`); ia
        bernalar laten untuk menetapkan ARAH (diagnosis + direction) TANPA menulis
        ekspresi. KV-nya (`guidance_kv`) lalu MENYEMAI front-end via
        `run(seed_kv=guidance_kv)` → proposal→design→construct→gate/repair
        yang menyusun ekspresinya. Downstream (backtest/feedback) tak
        berubah; output tetap `FrontEndOutput`.

        Karena guidance di-seed None + parent=teks, dan `trajectory.kv_cache` tetap
        None (transfer antar-generasi via teks `_format_parents_text`), KV per ronde
        terbatas (~guidance + front-end ≈ 3k) — tak ada warisan/akumulasi KV parent,
        jadi tak ada over-KV maupun collapse lintas-generasi (akar regresi 2026-06-02).
        """
        rl = self.runlog
        if kind == "mutation":
            agent_name = "mutation"
            kw = dict(target_text=parent_text, direction=direction)
        elif kind == "crossover":
            agent_name = "crossover"
            kw = dict(parents_text=parent_text, n_parents=n_parents, direction=direction)
        else:
            raise ValueError(f"unknown evolution kind: {kind!r}")

        # ── 1. GUIDANCE (kv_only, seed=None): parent sebagai TEKS → arah laten ──
        # past_kv=None → tak warisi KV parent (RESET tiap generasi). Pada comm_mode
        # kv: output tak di-decode (kv_only), dipakai HANYA KV-nya sebagai seed
        # proposal. Pada comm_mode text: guidance di-decode jadi TEKS arah (tak ada
        # KV) → disuntik ke front-end sebagai tambahan `direction`.
        guide_mode = self._agent_mode("kv_only")
        r_guide = self._a(agent_name).run(past_kv=None, mode_override=guide_mode, **kw)
        guidance_kv = r_guide.kv_cache
        if guidance_kv is None and rl and not self._is_text:
            rl.warn(f"{kind} guidance produced no KV; front-end runs unseeded")

        # ── 1b. decode guidance KV → readout ke llm_outputs (observability) ──
        # mutation/crossover berjalan kv_only (tak decode teks) → TANPA langkah ini
        # keputusan arah mereka tak pernah muncul di llm_outputs seperti agent lain.
        # Probe introspect (kv_and_text) men-decode KV-nya jadi readout yang OTOMATIS
        # ter-snapshot (mode != kv_only). DEEPCOPY: introspect meng-extend KV in-place →
        # jaga guidance_kv pristine untuk konsumen sebenarnya (proposal). Role di-label
        # `{kind}_guidance` agar file jelas asalnya. Default ON; opt-out
        # LATENTMAS_EVO_PROBE=0 (mis. saat ingin run secepat mungkin).
        if guidance_kv is not None and os.environ.get("LATENTMAS_EVO_PROBE", "1") != "0":
            try:
                probe = self._a("introspect").run(
                    past_kv=kv_ops.kv_deepcopy(guidance_kv),
                    role=f"{kind}_guidance",
                    mode_override=self._agent_mode("kv_and_text"),
                )
                if rl: rl.info(f"{kind} guidance (probe)", head=(probe.text or "")[:400])
            except Exception as e:  # noqa: BLE001
                if rl: rl.warn("evo guidance probe failed", err=repr(e))

        # ── 2. RE-ENTER front-end ────────────────────────────────────────────
        # comm_mode kv/kv_and_text: di-seed guidance_kv (1 konsumen → no deepcopy).
        # comm_mode text: tak ada KV → arah guidance (teks) digabung ke `direction`.
        if self._is_text:
            guide_text = (r_guide.text or "").strip()
            eff_dir = f"{direction}\n\n{guide_text}".strip() if guide_text else direction
            return self.run(direction=eff_dir, seed_kv=None, negative_hint=negative_hint)
        return self.run(direction=direction, seed_kv=guidance_kv, negative_hint=negative_hint)

    @staticmethod
    def _auto_fix_arity(candidates: List[str], rl: Any = None) -> List[str]:
        """Pra-perbaikan arity DETERMINISTIK (tanpa LLM) sebelum gate: menulis
        ulang kasus cross-sectional→time-series yang TAK AMBIGU — RANK(A,n)→
        TS_RANK(A,n), ZSCORE(A,n)→TS_ZSCORE(A,n), dst. (argumen ke-2 numerik).
        Kandidat yang tak bisa diperbaiki aman dibiarkan apa adanya (gate/LLM-
        repair yang menangani). Inilah penambal kelas bug arity yang dulu lolos
        gate lalu crash saat eksekusi (mis. RANK($return, 7))."""
        try:
            # Pre-import factor_ast memutus circular import bila dipanggil COLD
            # (sama seperti _build_regulator_gate). Di alur produksi gate sudah
            # dibangun lebih dulu → no-op; ini insurance bila urutan berubah.
            import dsl.factor_ast  # noqa: F401
            from gate.factor_regulator import auto_repair_function_arity
        except Exception:
            return candidates
        out: List[str] = []
        for e in candidates:
            try:
                rep, applied = auto_repair_function_arity(e)
            except Exception:
                rep, applied = None, []
            if rep and rep != e:
                if rl: rl.info("deterministic arity fix", before=e, after=rep, applied=applied)
                out.append(rep)
            else:
                out.append(e)
        return out
    # [terjawab — investigasi]: dua lapis. (1) auto_fix_arity = deterministik tanpa LLM
    #   (hanya cross-sectional→TS_ tak ambigu). (2) agen repair = berbasis LLM, MEMBACA
    #   konteks via past_kv=kv_deepcopy(kv_construct) (clone KV construct) PLUS teks
    #   (former_expression + error_log). Pada comm_mode=text, kv_construct=None →
    #   repair murni teks. Jadi repair memperoleh KEDUANYA (KV + teks) di mode KV.
    def _gate_and_repair_factors(
        self,
        factors: List[dict],
        kv_baseline: Optional[KVCache],
    ) -> "tuple[List[str], Optional[KVCache], bool, int, str, List[dict]]":
        """Gate tiap ekspresi faktor + LOG keputusan per-faktor. Aturan:
          - ≥1 lolos → pakai yang lolos, TANPA repair. kv_final = kv_construct.
          - SEMUA gagal → repair legacy (≤max attempts), gate ulang. kv_final = kv_repair.
          - repair habis → fail-closed (drop yang gagal-gate, jangan ke backtest).

        Returns: (passing, kv_final, repaired, attempts, gate_error, gate_log).
        gate_log: [{name, expression, ok, reason, repaired_by}] → audit lengkap.
        """
        rl = self.runlog
        gate_log: List[dict] = []
        # Pra-perbaikan arity DETERMINISTIK (no-LLM) → RANK(A,n)→TS_RANK(A,n) dkk.
        exprs = self._auto_fix_arity([f.get("expression", "") for f in factors], rl)
        names = [f.get("name") or f"f{i+1}" for i, f in enumerate(factors)]

        passing: List[str] = []
        first_err = ""
        for nm, e in zip(names, exprs):
            ok, reason = self.gate(e) if e else (False, "empty expression")
            gate_log.append({"name": nm, "expression": e, "ok": bool(ok),
                             "reason": reason, "repaired_by": None})
            if ok:
                passing.append(e)
                if rl: rl.info("GATE PASS", name=nm, expr=e)
            else:
                if not first_err:
                    first_err = reason
                if rl: rl.warn("GATE REJECT", name=nm, expr=e, reason=reason)

        if passing:
            self._register_factors(passing)
            return passing, kv_baseline, False, 0, "", gate_log

        # Tanpa kandidat: repair dengan former kosong → sampah (run 20260608_064005).
        usable = [e for e in exprs if e]
        if not usable:
            if rl: rl.error("no expression from construct; skipping repair")
            return [], kv_baseline, False, 0, "no expression from construct", gate_log

        gate_err = first_err or self.gate(usable[0])[1]
        former, err = usable, gate_err

        # Signature normalisasi sekumpulan ekspresi → deteksi "output tak berubah".
        def _sig(exprs: List[str]) -> tuple:
            return tuple(sorted(e.replace(" ", "").lower() for e in exprs if e))

        # seen berisi signature input awal; bila repair mengembalikan set yang sama
        # (mis. model echo ekspresi rusak apa adanya) → memanggil lagi sia-sia.
        seen = {_sig(former)}
        attempts_made = 0

        # Loop adaptif: SATU call repair per attempt → gate deterministik.
        #   - ada yang lolos gate  → return (lanjut backtest), STOP.
        #   - gagal tapi ada ekspresi BARU → umpan-balik error terbaru, call lagi.
        #   - gagal & output tak berubah (PASS tanpa ekspresi / set berulang) →
        #     EARLY-EXIT: call ulang dgn input identik hanya buang kuota & latency.
        for attempt in range(self.max_repair_attempts):
            attempts_made = attempt + 1
            mode = ["minimal", "different", "bold"][min(attempt, 2)]
            r_rep = self._a("repair").run(
                past_kv=kv_ops.kv_deepcopy(kv_baseline),  # None di mode text
                former_expression="; ".join(former) if former else "",
                error_log=err, value_feedback="", attempt_mode=mode,
                mode_override=self._agent_mode("kv_and_text"),
            )
            is_pass, rep_exprs = parse_repair_multi(r_rep.text or "")
            if rl: rl.info("REPAIR attempt", n=attempt + 1, mode=mode,
                           pass_claim=is_pass, n_expr=len(rep_exprs))

            # Kandidat yg digate = output FIXED model. Bila model menjawab PASS
            # (tanpa ekspresi baru), uji ULANG `former` — gate deterministik adalah
            # sumber kebenaran, bukan klaim model.
            candidates = rep_exprs if rep_exprs else former
            passing = [e for e in candidates if self.gate(e)[0]]
            if passing:
                self._register_factors(passing)
                for e in passing:
                    gate_log.append({"name": "repaired", "expression": e,
                                     "ok": True, "reason": "", "repaired_by": "agent"})
                if rl: rl.info("REPAIR success", n=attempt + 1,
                               confirmed_by="gate", pass_claim=is_pass,
                               n_passing=len(passing))
                return passing, r_rep.kv_cache, True, attempt + 1, gate_err, gate_log

            new_sig = _sig(rep_exprs)
            if not rep_exprs or new_sig in seen:
                if rl: rl.warn("REPAIR no-change; early-exit", n=attempt + 1,
                               reason="pass-no-expr" if not rep_exprs else "repeat-expr")
                break
            seen.add(new_sig)
            former = rep_exprs
            err = self.gate(rep_exprs[0])[1]

        # FAIL-CLOSED: repair mentok → drop ekspresi gagal-gate (jangan ke backtest:
        # ekspresi rusak → factor.py crash + waktu terbuang).
        survivors = [e for e in usable if self.gate(e)[0]]
        if rl: rl.error("repair exhausted; fail-closed drop",
                        n_dropped=len(usable) - len(survivors), n_kept=len(survivors))
        if survivors:
            self._register_factors(survivors)
        return survivors, kv_baseline, False, attempts_made, gate_err, gate_log

    def _gate_and_repair(
        self,
        expression: str,
        kv_baseline: Optional[KVCache],
    ) -> "tuple[str, bool, int, str, Optional[KVCache]]":
        """Gate ekspresi; jika gagal jalankan repair (≤ max attempts), tiap attempt
        berangkat dari CLONE pristine kv_baseline.

        Returns: (final_expression, repaired, attempts, gate_error, kv_repair).
        `kv_repair` = KV dari attempt repair yang diterima (untuk dijadikan kv_final),
        atau None bila gate langsung lolos / repair gagal (caller pakai kv_judger).
        """
        rl = self.runlog
        if expression:
            ok, err = self.gate(expression)
        else:
            ok, err = False, "no expression"
        if ok:
            return expression, False, 0, "", None

        gate_error = err
        tried = {expression.replace(" ", "").lower()} if expression else set()
        modes = ["minimal", "different", "bold"]
        former = expression

        for attempt in range(self.max_repair_attempts):
            mode = modes[min(attempt, len(modes) - 1)]
            r_rep = self._a("repair").run(
                past_kv=kv_ops.kv_deepcopy(kv_baseline),  # None di mode text
                former_expression=former, error_log=err,
                value_feedback="", attempt_mode=mode,
                mode_override=self._agent_mode("kv_and_text"),
            )
            parsed = r_rep.parsed
            if parsed is None:
                if rl: rl.warn(f"repair attempt {attempt+1} unparseable")
                continue
            if parsed == PASS_SENTINEL:
                if rl: rl.info("repair returned PASS; keeping expression")
                return former, True, attempt + 1, gate_error, r_rep.kv_cache
            norm = parsed.replace(" ", "").lower()
            if norm in tried:
                if rl: rl.warn(f"repair attempt {attempt+1} repeated a tried expr")
                former = parsed
                continue
            ok2, err2 = self.gate(parsed)
            if ok2:
                return parsed, True, attempt + 1, gate_error, r_rep.kv_cache
            tried.add(norm)
            former, err = parsed, err2

        if rl: rl.error("repair exhausted; keeping original expression", expr=expression)
        return expression, False, self.max_repair_attempts, gate_error, None
