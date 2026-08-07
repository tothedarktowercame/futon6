#!/usr/bin/env python3
"""render_run — compose a paper's FULL pipeline run as an additive, two-up,
scroll-linked demo. Mirrors the runner: one stage module per phase, composited by
rr_compositor with a checked additivity invariant.

    render_run.py [PAPER_ID]        (default 0801.0199)
    render_run.py --all             (render every paper with golden marks; skip+report the rest)
    render_run.py --all --graph-dir data/iatc-argument-graphs/run --out-dir ...

Writes <out-dir>/render-run-<paper>.html

Layer six (APM structure match) was removed 2026-08-07 alongside the S9
deprecation. It was a static stub that displayed "gate_pass=true" for a gate
which, per the DAG contract, "had failed silently on every run since it was
written" — so the demo asserted a passing gate that had never passed. A renderer
is a claim about the run; a hardcoded verdict in one is worse than no layer.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import rr_layer_weft, rr_layer_concept, rr_layer_warp
import rr_layer_iatc, rr_layer_expository
from rr_compositor import line_index, render_two_up
from paper_ids import proof_pid_from_graph_name
from run_artifacts import proof_graphs

# Derived, not hardcoded: this ran only from Joe's checkout before.
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/showcases/ct-anatomy/dp-demo"
GRAPH_DIR = ROOT / "data/iatc-argument-graphs/run"


def render_one(pid: str) -> Path:
    """Render one paper's full-pipeline demo. Raises if its artifacts are missing
    or the additivity invariant is violated."""
    text, marks = rr_layer_weft.load_text(pid)
    starts, _ = line_index(text)          # line -> char start, for rail anchors

    layers = [
        rr_layer_weft.layer(pid, marks),        # ①
        rr_layer_concept.layer(pid),            # ②
        rr_layer_warp.layer(pid),               # ③
        rr_layer_iatc.layer(pid, starts),       # ④
        rr_layer_expository.layer(pid, starts), # ⑤
    ]
    html = render_two_up(pid, text, layers)     # raises if additivity violated
    out = OUT / f"render-run-{pid}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)//1024} KB) — assert_additive PASSED")
    for L in layers:
        print(f"  {L.sigil} {L.name:26} mode={L.mode:6} {len(L.spans):5} spans  {len(L.annotations)} card(s)")
    return out


def discover_pids(graph_dir: Path) -> list[str]:
    """Papers with a final IATC graph — the canonical pipeline set.

    Two corrections over the previous glob. It matched `*.edn`, so the
    `.rung2.edn` sidecars were counted as papers (the same selection defect that
    put 98 spurious rows in S5's verdicts); `proof_graphs` is the shared rule.
    And the run directory names graphs per *proof* (`0705.0102__p0`), so the
    stem is not a paper id — `proof_pid_from_graph_name` collapses them, which
    is why this returns a set.
    """
    return sorted({proof_pid_from_graph_name(Path(p).name) for p in proof_graphs(str(graph_dir))})


def render_all(graph_dir: Path) -> int:
    pids = discover_pids(graph_dir)
    rendered, skipped = [], []
    for pid in pids:
        try:
            render_one(pid)
            rendered.append(pid)
        except Exception as e:  # missing golden marks / layer artifacts → skip, don't abort
            reason = f"{type(e).__name__}: {e}"
            skipped.append((pid, reason))
            print(f"skip {pid} — {reason}")
    print(f"\nrender_run --all: {len(rendered)}/{len(pids)} rendered, {len(skipped)} skipped")
    for pid, reason in skipped:
        print(f"  skipped {pid}: {reason}")
    return 0


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", nargs="?", default="0801.0199")
    ap.add_argument("--all", action="store_true", help="render every paper with a final graph")
    ap.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()

    OUT = a.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    if a.all:
        return render_all(a.graph_dir)
    render_one(a.paper)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
