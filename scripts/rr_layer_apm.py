#!/usr/bin/env python3
"""render_run · ⑥ APM structure match — matches an APM prelim proof's scopes
against the eprint scope pool. Conditional (APM proofs only) + corpus-scale, so no
per-paper overlay for a plain eprint (honest stub)."""
from __future__ import annotations
from rr_compositor import Annotation, Layer


def layer(pid: str) -> Layer:
    body = ('<div class="verdict">⑥ matches an <b>APM prelim proof</b> against the eprint scope pool '
            '(type-only → tightest). It applies to APM proofs, not a plain eprint like this one, and reads '
            'against the corpus — so no per-paper overlay here. Gate: mean ≥.20 / median ≥.10 (currently '
            '<code>gate_pass=true</code>).</div>')
    return Layer("⑥", "APM structure match", "#9ca3af", "none", False, [],
                 [Annotation(1, "⑥", "APM match (⑥) — APM-only / corpus-scale", body, "#9ca3af")])
