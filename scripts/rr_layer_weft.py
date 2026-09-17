#!/usr/bin/env python3
"""render_run · ① Weft — CPU per-paper anatomy, rendered INLINE and composited so
nested marks (symbol ⊂ binder ⊂ math scope) all keep their detail."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import futon6_config as config

import glob, json, re
from collections import Counter
from pathlib import Path
from rr_compositor import Annotation, Layer, Span, golden_class

ROOT = config.ROOT
GOLD = config.marks()


def load_text(pid: str):
    g = json.load(open(glob.glob(f"{GOLD}/*{pid}*dp-emacs.json")[0]))
    return g["text"], g["marks"]


def layer(pid: str, marks=None) -> Layer:
    if marks is None:
        _, marks = load_text(pid)
    spans = []
    for m in marks:
        if not (isinstance(m.get("start"), int) and m.get("end", 0) > m.get("start", 0)):
            continue
        cls = golden_class(m.get("kind", ""))   # golden vocabulary; None = skip kind
        if cls:
            spans.append(Span(m["start"], m["end"], cls,
                              str(m.get("tip") or m.get("kind") or "")))
    kc = Counter(m.get("kind") for m in marks)
    lc = Counter(m.get("layer") for m in marks)
    top = ", ".join(f"{k} {c}" for k, c in kc.most_common(6))
    body = (f'<div class="fact">{len(marks)} marks · layers {dict(lc)} · wf=0 (validated floor)</div>'
            f'<div class="fact">top: {top}</div>'
            '<div class="verdict"><b>The solid floor.</b> Deterministic, checked, dense — every symbol '
            'typed, binders &amp; quantifiers scoped. This is what already works; the GPU stages must '
            '<i>add to</i> it, not compete with it.</div>')
    return Layer("①", "Weft / CPU anatomy", "#2456a6", "inline", False, spans,
                 [Annotation(1, "①", "CPU anatomy (Weft)", body, "#2456a6")])
