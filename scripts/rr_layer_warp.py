#!/usr/bin/env python3
"""render_run · ③ Warp — cross-corpus second layer. Corpus-scale (lexicon /
dependency PageRank / embeddings): no per-paper overlay. Shown for structure."""
from __future__ import annotations
from rr_compositor import Annotation, Layer


def layer(pid: str) -> Layer:
    body = ('<div class="verdict">③ is a <b>cross-corpus</b> pass (per-class lexicon, definition-dependency '
            'PageRank, embeddings) — it only means something at MSC scale, so it has no per-paper overlay. '
            'Present here for structural fidelity with the runner.</div>')
    return Layer("③", "Warp / cross-corpus", "#9ca3af", "none", False, [],
                 [Annotation(1, "③", "Warp (③) — corpus-scale, n/a per-paper", body, "#9ca3af")])
