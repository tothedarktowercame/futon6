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
