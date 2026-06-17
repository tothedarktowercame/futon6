#!/usr/bin/env python3
"""render_run — compose a paper's FULL pipeline run as an additive, two-up,
scroll-linked demo. Mirrors the runner: one stage module per phase, composited by
rr_compositor with a checked additivity invariant.

    render_run.py [PAPER_ID]        (default 0801.0199)

Writes data/showcases/ct-anatomy/dp-demo/render-run-<paper>.html
"""
from __future__ import annotations
import sys
from pathlib import Path

import rr_layer_weft, rr_layer_concept, rr_layer_warp
import rr_layer_iatc, rr_layer_expository, rr_layer_apm
from rr_compositor import line_index, render_two_up

ROOT = Path("/home/joe/code/futon6")
OUT = ROOT / "data/showcases/ct-anatomy/dp-demo"


def main() -> int:
    pid = sys.argv[1] if len(sys.argv) > 1 else "0801.0199"
    text, marks = rr_layer_weft.load_text(pid)
    starts, _ = line_index(text)          # line -> char start, for rail anchors

    layers = [
        rr_layer_weft.layer(pid, marks),        # ①
        rr_layer_concept.layer(pid),            # ②
        rr_layer_warp.layer(pid),               # ③
        rr_layer_iatc.layer(pid, starts),       # ④
        rr_layer_expository.layer(pid, starts), # ⑤
        rr_layer_apm.layer(pid),                # ⑥
    ]
    html = render_two_up(pid, text, layers)     # raises if additivity violated
    out = OUT / f"render-run-{pid}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)//1024} KB) — assert_additive PASSED")
    for L in layers:
        print(f"  {L.sigil} {L.name:26} mode={L.mode:6} {len(L.spans):5} spans  {len(L.annotations)} card(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
