#!/usr/bin/env python3
"""Shared bb→Python EDN compatibility (E-superpod-hardening H12, 2026-08-06).

The IATC graphs are PRODUCED and gated under bb (Clojure's lenient reader) and
CONSUMED by the Python stages through edn_format (strict). Tokens bb accepts and
edn_format rejects therefore make a graph gate-PASS but silently unreadable
downstream — the failure is a scattered `load error` line, not a gate failure.

This module is the single place that reconciles the two readers, because the
same divergence was previously patched ad hoc in three separate loaders
(iatc_to_clean._edn_safe handled `'`; r2d_concept_coverage.load_edn handled `'`
with a different regex; clean_box_typing inherited whichever it imported), so a
new divergence had to be found and fixed three times. It was found once and
fixed once, and the other two paths kept dropping graphs.

Known divergences, both repaired ONLY outside double-quoted strings so :text
prose keeps its content verbatim, and both deterministic + global so node ids
and the edge refs pointing at them stay aligned:

  `'` in symbols/keywords  (:phi' — CT primes)        -> `prime`
  non-ASCII in symbols/keywords  (:hom→cone, :μ-nat)  -> `u<hex codepoint>`
"""
from __future__ import annotations


_EDN_SIMPLE_ESCAPES = set('"\\/bfnrt')


def repair_string_escapes(text: str) -> str:
    """Escape backslashes INSIDE EDN strings that are not legal EDN escapes.

    The models write mathematical prose, so `:text` fields are full of LaTeX —
    `\\Phi`, `\\xi`, `\\circ`, `\\lambda`. EDN permits only `\\" \\\\ \\/ \\b \\f
    \\n \\r \\t` and `\\uXXXX`; every other backslash is a hard parse error, so
    the graph is rejected by the bb gate with `Unsupported escape character`.
    On the Zone e2e run this was the cause of **all 42 S4 failures (15% of the
    expository layer)** and a contributor to S3's 48% retry rate — a
    serialization-contract problem, not a model-quality one
    (E-superpod-hardening H18, 2026-08-06).

    Doubling the backslash preserves the author's intent exactly: EDN then reads
    `"\\\\Phi"` back as the literal text `\\Phi`. Complementary to `edn_safe`,
    which repairs tokens OUTSIDE strings and leaves string content untouched.
    """
    out, in_str, i, n = [], False, 0, len(text)
    while i < n:
        ch = text[i]
        if not in_str:
            if ch == '"':
                in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == '\\':
            nxt = text[i + 1] if i + 1 < n else ''
            if nxt == 'u':
                hexes = text[i + 2:i + 6]
                if len(hexes) == 4 and all(c in '0123456789abcdefABCDEF' for c in hexes):
                    out.append(text[i:i + 6])   # legal \uXXXX
                    i += 6
                    continue
                out.append('\\\\')              # \upsilon etc — not a codepoint
                i += 1
                continue
            if nxt in _EDN_SIMPLE_ESCAPES:
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            out.append('\\\\')                  # \Phi, \xi, ... — escape it
            i += 1
            continue
        if ch == '"':
            in_str = False
        out.append(ch)
        i += 1
    return "".join(out)


def edn_safe(text: str) -> str:
    """Rewrite bb-legal-but-edn_format-illegal tokens outside string literals."""
    out, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif ch == "'":
            out.append("prime")
        elif ord(ch) > 127:
            out.append("u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)
