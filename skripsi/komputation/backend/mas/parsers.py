"""
mas/parsers.py
====================
Parser output teks agent. Sengaja permisif — model 4B sering menambah
penjelasan/markdown walau diminta satu baris. Tiap parser mengembalikan
struktur kecil yang gampang dicek, atau None bila benar-benar gagal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Judger / Mutation / Crossover: HYPOTHESIS + EXPRESSION ───────────────────

@dataclass
class HypothesisExpr:
    hypothesis: str
    expression: str


@dataclass
class HypothesisExprs:
    """1 hipotesis → N ekspresi (judger boleh keluarkan banyak; semua yang lolos
    regulator masuk ke model LightGBM gabungan)."""
    hypothesis: str
    expressions: list


def parse_hypothesis_exprs(raw: str) -> Optional[HypothesisExprs]:
    """Versi multi dari parse_hypothesis_expr: kumpulkan SEMUA baris EXPRESSION.

    Toleran sama seperti versi tunggal (label terpotong, backtick, fence, <think>).
    Mengembalikan list ekspresi unik (urutan dipertahankan). Untuk output judger
    single-expr lama, list berisi 1 — backward-compatible.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "")
    text = re.sub(r"</?think>", "", text)
    # Model 4B sering membungkus label dengan markdown bold/italic
    # (`**HYPOTHESIS**:`) — tapi '*' juga operator aritmetik DSL yang valid.
    # Buang HANYA '*' yang menempel ke karakter kata (markdown), bukan '*' aritmetik
    # yang selalu diapiti spasi atau tanda kurung (mis. `(A) * (B)`).
    text = re.sub(r"(?<=\w)\*+|\*+(?=\w)", "", text)
    # Model 4B memakai 'AND'/'OR' kata benda (bahasa Inggris alami) alih-alih
    # operator DSL '&&'/'||'. Substitusi aman: tidak ada operator/variabel DSL yang
    # mengandung string '\bAND\b' atau '\bOR\b'.
    text = re.sub(r"\bAND\b", "&&", text)
    text = re.sub(r"\bOR\b", "||", text)
    # Model kadang menulis '→' sebagai panah kausalitas setelah ekspresi ('expr → return < 0').
    # Potong semua yang muncul setelah '→' — ekspresi DSL valid tak pernah mengandung '→'.
    text = re.sub(r"→.*", "", text)

    hyp_m = re.search(
        r"hypo\w*\s*:\s*(.+?)(?=\n\s*expr\w*\s*\d*\s*:|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    hypothesis = hyp_m.group(1).strip() if hyp_m else ""

    exprs: list = []
    # tiap baris 'EXPR...:' / 'EXPRESSION 2:' → satu/lebih ekspresi. Model 4B kadang
    # menaruh beberapa ekspresi dalam satu baris dipisah ';' → pecah (';' bukan
    # operator DSL valid, jadi aman & wajib di-split agar tak jadi 1 kandidat rusak).
    for m in re.finditer(r"expr\w*\s*\d*\s*:\s*(.+)", text, flags=re.IGNORECASE):
        raw = _extract_code_span(m.group(1).strip())
        for piece in raw.split(";"):
            line = _balance_parens(_strip_wrappers(piece.strip()))
            if line:
                exprs.append(line)

    uniq = _dedup_exprs(exprs)
    if not uniq:
        return None
    return HypothesisExprs(hypothesis=hypothesis, expressions=uniq)


def parse_repair_multi(raw: str) -> "tuple[bool, list]":
    """Repair versi multi → (is_pass, expressions).

    is_pass=True bila model menjawab 'PASS' (ekspresi dianggap valid apa adanya).
    Selain itu kumpulkan semua 'FIXED: <expr>' / 'EXPR..: <expr>' → list.
    """
    if not raw or not raw.strip():
        return False, []
    text = re.sub(r"</?think>", "", raw).strip()
    text = re.sub(r"(?<=\w)\*+|\*+(?=\w)", "", text)
    text = re.sub(r"\bAND\b", "&&", text)   # repair agent juga pakai AND/OR literal
    text = re.sub(r"\bOR\b", "||", text)
    text = re.sub(r"→.*", "", text)          # potong panah kausalitas (→ $return < 0)
    lines = text.splitlines()
    first = lines[0].strip() if lines else ""
    if re.fullmatch(r"pass[.!]?", first, flags=re.IGNORECASE):
        return True, []
    exprs: list = []
    kw = re.compile(r"^\s*(?:fixed|expr\w*|result)\s*\d*\s*:\s*(.+?)\s*$",
                    flags=re.IGNORECASE)
    for line in lines:
        m = kw.match(line)
        if m:
            raw = _extract_code_span(m.group(1).strip())
            for piece in raw.split(";"):   # pecah multi-ekspresi ';'-joined
                e = _balance_parens(_strip_wrappers(piece.strip()))
                if e:
                    exprs.append(e)
    if not exprs:  # fallback: span ber-backtick
        for m in re.finditer(r"`([^`]+)`", text):
            e = _balance_parens(_strip_wrappers(m.group(1).strip()))
            if e:
                exprs.append(e)
    return False, _dedup_exprs(exprs)


def _dedup_exprs(exprs: list) -> list:
    seen, out = set(), []
    for e in exprs:
        k = e.replace(" ", "").lower()
        if k and k not in seen:
            seen.add(k)
            out.append(e)
    return out


def parse_hypothesis_expr(raw: str) -> Optional[HypothesisExpr]:
    """Ambil 'HYPOTHESIS: ...' dan 'EXPRESSION: ...' dari output judger.

    Toleran terhadap:
      - label case-insensitive + terpotong (HYPOTHESIS/Hypothesis/hypo/HYPOTH/
        HYPOTHS, EXPRESSION/EXPR) — model 4B sering memenggal label,
      - hypothesis multi-baris sampai ketemu baris EXPRESSION,
      - expression dibungkus backtick/quote,
      - markdown fence,
      - tag <think>/</think> yatim yang lolos dari strip.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # buang markdown fence global + tag think yatim
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "")
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"(?<=\w)\*+|\*+(?=\w)", "", text)  # markdown bold/italic, bukan aritmetik
    text = re.sub(r"\bAND\b", "&&", text)  # lihat parse_hypothesis_exprs
    text = re.sub(r"\bOR\b", "||", text)

    # Label match longgar: 'hypo' diikuti word-char apa pun (hypo, hypoth,
    # hypoths, hypothesis) lalu ':'. Idem 'expr' (expr, expression).
    hyp_m = re.search(
        r"hypo\w*\s*:\s*(.+?)(?=\n\s*expr\w*\s*:|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    expr_m = re.search(
        r"expr\w*\s*:\s*(.+?)\s*\Z",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if not expr_m:
        return None
    expression = expr_m.group(1).strip()
    # expression: ambil baris non-kosong pertama (model sering menambah catatan)
    for line in expression.splitlines():
        if line.strip():
            expression = line.strip()
            break
    expression = _extract_code_span(expression)
    expression = _strip_wrappers(expression)
    expression = _balance_parens(expression)

    hypothesis = hyp_m.group(1).strip() if hyp_m else ""
    if not expression:
        return None
    return HypothesisExpr(hypothesis=hypothesis, expression=expression)


# ── Repair-or-pass: PASS | FIXED: <expr> ─────────────────────────────────────

PASS_SENTINEL = "__PASS__"


def parse_repair(raw: str) -> Optional[str]:
    """Kembalikan PASS_SENTINEL, ekspresi (string), atau None.

    Kontrak: satu baris — 'PASS' atau 'FIXED: <expression>'.
    """
    if not raw or not raw.strip():
        return None
    text = re.sub(r"</?think>", "", raw).strip()
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if re.fullmatch(r"pass[.!]?", first, flags=re.IGNORECASE):
        return PASS_SENTINEL
    kw = re.compile(r"^\s*(?:fixed|expr\w*|result)\s*:\s*(.+?)\s*$",
                    flags=re.IGNORECASE)
    for line in text.splitlines():
        if not line.strip():
            continue
        m = kw.match(line)
        if m:
            expr = _extract_code_span(m.group(1).strip())
            expr = _balance_parens(_strip_wrappers(expr))
            if expr:
                return expr
    # fallback JSON {"expr": "..."}
    m = re.search(r'"(?:expr|fixed|expression)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        return m.group(1).strip()
    return None


# ── Mutation reflection: FAILURE_STEP + REASON ───────────────────────────────

@dataclass
class MutationDiagnosis:
    failure_step: str          # propose | construct | consistency | expression | unknown
    reason: str


_VALID_STEPS = {"propose", "construct", "consistency", "expression", "unknown"}


def parse_mutation_diagnosis(raw: str) -> MutationDiagnosis:
    """Ambil 'FAILURE_STEP: ...' + 'REASON: ...'. Tidak pernah None —
    fallback ke ('construct', <teks mentah>) bila gagal parse, karena
    construct adalah titik revisi paling umum."""
    text = (raw or "").strip()
    step_m = re.search(r"failure_step\s*:\s*(\w+)", text, flags=re.IGNORECASE)
    reason_m = re.search(r"reason\s*:\s*(.+?)\s*\Z", text,
                         flags=re.IGNORECASE | re.DOTALL)
    step = (step_m.group(1).lower() if step_m else "construct")
    if step not in _VALID_STEPS:
        step = "construct"
    reason = reason_m.group(1).strip() if reason_m else text
    return MutationDiagnosis(failure_step=step, reason=reason)


# ── Introspect: bebas (teks apa adanya) ──────────────────────────────────────

def parse_passthrough(raw: str) -> str:
    return (raw or "").strip()


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_code_span(expr: str) -> str:
    """Bila ada span ber-backtick (`...`), ambil isinya — model 4B sering
    membungkus ekspresi dalam backtick lalu menambah catatan setelahnya."""
    m = re.search(r"`([^`]+)`", expr)
    return m.group(1).strip() if m else expr


def _strip_wrappers(expr: str) -> str:
    expr = expr.strip()
    for q in ("`", '"', "'"):
        if len(expr) >= 2 and expr.startswith(q) and expr.endswith(q):
            expr = expr[1:-1].strip()
    return expr


def _balance_parens(expr: str) -> str:
    """Potong di titik kelebihan ')' (model 4B sering menambah junk di akhir)."""
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return expr[:i].rstrip()
    return expr


# ── Construct (JSON): hypothesis + factors[{name, expression, explanation}] ───

@dataclass
class ConstructResult:
    """Output agent `construct` (front-end no-crop). Superset dari HypothesisExprs:
    selain hypothesis + daftar ekspresi, menyimpan NAMA + EXPLANATION (intent) tiap
    faktor. `explanation` dipakai repair sadar-intent & dicatat di StrategyTrajectory.
    """
    hypothesis: str
    factors: list          # list[dict]: {name, expression, explanation}

    @property
    def expressions(self) -> list:
        """Daftar ekspresi (urutan dipertahankan) — kompatibel HypothesisExprs."""
        return [f["expression"] for f in self.factors if f.get("expression")]


def _extract_json_block(text: str) -> Optional[dict]:
    """Ambil objek JSON pertama..terakhir dari teks (toleran fence ```json)."""
    if not text:
        return None
    import json
    t = text.strip()
    fence = re.search(r"```(?:json|text)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1 or e < s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None


def parse_construct_json(raw: str) -> Optional[ConstructResult]:
    """Parse output construct. Utama: JSON {hypothesis, factors:[{name,expression,
    explanation}]}. Fallback: parser DSL hypothesis_exprs (nama f1..fn, explanation
    kosong) bila JSON rusak — supaya pipeline tak dead-end."""
    obj = _extract_json_block(raw)
    if obj and isinstance(obj.get("factors"), list):
        facs = []
        for i, f in enumerate(obj["factors"]):
            if not isinstance(f, dict):
                continue
            expr = _balance_parens(str(f.get("expression") or "").strip())
            if not expr:
                continue
            facs.append({
                "name": str(f.get("name") or f"f{i+1}").strip(),
                "expression": expr,
                "explanation": str(f.get("explanation") or "").strip(),
            })
        if facs:
            return ConstructResult(
                hypothesis=str(obj.get("hypothesis") or "").strip(), factors=facs)

    he = parse_hypothesis_exprs(raw or "")
    if he is None:
        return None
    facs = [{"name": f"f{i+1}", "expression": e, "explanation": ""}
            for i, e in enumerate(he.expressions)]
    return ConstructResult(hypothesis=he.hypothesis, factors=facs)

# [terjawab — investigasi]: BUKAN duplikat fungsi. parsers.py mem-parse OUTPUT LLM
#   (hypothesis/expression/JSON construct). factors/coder/expr_parser.py + factor_ast.py
#   mem-parse EKSPRESI DSL → AST (untuk eksekusi/regulator). Domain berbeda; hanya
#   sebagian kecil helper (balance-parens/strip) yang berpotensi tumpang-tindih.
# registry untuk lookup by name dari YAML
PARSERS = {
    "hypothesis_expr": parse_hypothesis_expr,
    "hypothesis_exprs": parse_hypothesis_exprs,
    "construct_json": parse_construct_json,
    "repair": parse_repair,
    "mutation_diagnosis": parse_mutation_diagnosis,
    "passthrough": parse_passthrough,
    "none": None,
}
