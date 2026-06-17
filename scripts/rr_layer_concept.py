#!/usr/bin/env python3
"""render_run · ② Concept substrate — the noun layer. Merged into ① by dp_enrich
and the weakest measured stage, so it has no independent overlay (honest stub)."""
from __future__ import annotations
from rr_compositor import Annotation, Layer


def layer(pid: str) -> Layer:
    body = ('<div class="fact">prose-concept precision 0.108 / recall 0.22 (weak)</div>'
            '<div class="verdict">The noun layer is <b>merged into ① by dp_enrich</b> and is the weakest '
            'measured stage. Not drawn as a separate overlay to avoid overclaiming — its marks ride inside '
            'the Weft composite. <i>The honest follow-on is detector quality, not presentation.</i></div>')
    return Layer("②", "Concept substrate", "#6b7280", "none", False, [],
                 [Annotation(1, "②", "Concept substrate (merged into ①)", body, "#6b7280")])
