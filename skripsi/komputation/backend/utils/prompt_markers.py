"""Prompt variable markers for human-readable debug output.

Goal: at prompt-render time, wrap each variable value with a sentinel so the
debug writer can re-discover which slice of the rendered prompt came from
which YAML variable. Markers are stripped before the prompt reaches the LLM
engine — the model never sees them.

Marker format:
    [[VAR:<name>]]<value>[[/VAR:<name>]]

Why this format: the sequence "[[VAR:" is vanishingly unlikely to appear in
natural text or prompts, and the closing tag carries the name so nested
or repeated variables remain unambiguous.

Typical use in a prompt loader:

    from utils.prompt_markers import wrap
    rendered = template.render(
        scenario=wrap("scenario", scen_desc),
        targets=wrap("targets", self.targets),
    )

Or via :func:`render_marked` which wraps every string kwarg automatically.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

# ── Marker syntax ──────────────────────────────────────────────────────────
# Capture group 1 = variable name; group 2 = value (non-greedy, DOTALL).
# Backreference \1 ensures opening name matches closing name.
_MARKER_RE = re.compile(
    r"\[\[VAR:([A-Za-z_][A-Za-z0-9_]*)\]\](.*?)\[\[/VAR:\1\]\]",
    re.DOTALL,
)
_OPEN  = "[[VAR:{name}]]"
_CLOSE = "[[/VAR:{name}]]"


def wrap(name: str, value: Any) -> Any:
    """Wrap a string value with VAR markers; pass non-strings through.

    Non-str values (None, int, list, Trace objects, ...) are returned
    unchanged so Jinja's falsy semantics and method/attribute access on
    objects remain identical to a normal render. Wrapping None as the
    string "None" would silently flip ``{% if RAG %}`` from falsy to truthy.
    """
    if not isinstance(value, str):
        return value
    return f"{_OPEN.format(name=name)}{value}{_CLOSE.format(name=name)}"


def render_marked(template, **kwargs) -> str:
    """Render a Jinja2 template, auto-wrapping string kwargs with markers.

    Non-string kwargs (e.g. ``trace`` objects) are passed through unchanged
    since Jinja templates may call methods/attributes on them.
    """
    marked = {
        k: (wrap(k, v) if isinstance(v, str) else v)
        for k, v in kwargs.items()
    }
    return template.render(**marked)


def strip_text(text: str) -> str:
    """Remove all VAR markers, leaving the underlying values in place."""
    if not text or "[[VAR:" not in text:
        return text
    return _MARKER_RE.sub(r"\2", text)


def strip_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return a copy of ``messages`` with markers stripped from each content.

    The original list is not mutated — callers can keep a marked reference
    for debug writing while sending the stripped copy to the engine.
    """
    out: List[Dict[str, str]] = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str) and "[[VAR:" in c:
            out.append({**m, "content": strip_text(c)})
        else:
            out.append(m)
    return out


def iter_marked(text: str) -> Iterable[Tuple[int, int, str, str]]:
    """Yield ``(start, end, name, value)`` for every marker pair in ``text``."""
    if not text or "[[VAR:" not in text:
        return
    for m in _MARKER_RE.finditer(text):
        yield m.start(), m.end(), m.group(1), m.group(2)


def extract_vars(text: str) -> List[Tuple[str, str]]:
    """Collect ``(name, value)`` for every marker pair, in document order.

    Duplicates are kept (the same variable may legitimately be wrapped at
    multiple spots — e.g. ``{{scenario}}`` referenced in both system and
    user templates).
    """
    return [(name, value) for _, _, name, value in iter_marked(text)]


def render_for_debug(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace markers with inline human-readable labels.

    Returns ``(annotated_text, variables)`` where ``annotated_text`` shows
    each variable's name on its own line above/below the value, and
    ``variables`` is the ordered list of ``(name, value)`` pairs found.

    Output shape for each marker:

        <<<scenario>>>
        ...value...
        <<</scenario>>>
    """
    if not text:
        return text, []
    found: List[Tuple[str, str]] = []

    def _sub(match: re.Match) -> str:
        name, value = match.group(1), match.group(2)
        found.append((name, value))
        return f"<<<{name}>>>\n{value}\n<<</{name}>>>"

    annotated = _MARKER_RE.sub(_sub, text)
    return annotated, found
