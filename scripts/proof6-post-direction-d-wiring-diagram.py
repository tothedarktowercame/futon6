#!/usr/bin/env python3
"""Generate post-Direction-D wiring diagram for Problem 6.

This diagram captures the post-D strategy split:
- Low-gap perturbative shadow branch
- High-gap drift-to-score branch
- Trivial zero-score branch
and the glue obligation to close GPL-H.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from futon6.thread_performatives import (
    ThreadNode,
    ThreadEdge,
    ThreadWiringDiagram,
    diagram_to_dict,
)


def run_summary_text(run_path: Path) -> str:
    if not run_path.exists():
        return (
            "Post-D split run not loaded yet. Expected output at "
            f"{run_path}."
        )

    try:
        payload = json.loads(run_path.read_text())
        s = payload["summary"]
        fr = s["fractions"]
        return (
            "Empirical split diagnostics (n<=48): "
            f"low={fr['low']:.3f}, high={fr['high']:.3f}, "
            f"split={fr['split_covered']:.3f}, total={fr['covered']:.3f}; "
            f"split misses={s['counts']['split_miss_rows']} "
            f"(max min_score={s['miss_stats']['split_miss_max_min_score']})."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Could not parse split summary at {run_path}: {exc}"


def build_post_d_diagram(run_path: Path) -> ThreadWiringDiagram:
    d = ThreadWiringDiagram(thread_id="first-proof-p6-post-d")

    d.nodes = [
        ThreadNode(
            id="p6d-q",
            node_type="question",
            post_id=6300,
            body_text=(
                "Post-D target: close GPL-H via regime split. Prove that at each "
                "Case-2b step, at least one branch certifies a vertex with "
                "||Y_t(v)|| < 1."
            ),
            score=0,
            creation_date="2026-02-12",
            tags={"verification_status": "open"},
        ),
        ThreadNode(
            id="p6d-a1",
            node_type="answer",
            post_id=6301,
            body_text=(
                "Direction D finding: uniform near-rank-1 universality fails up to "
                "n<=48 (max rank gap > 2 in dense late steps)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "proved_negative"},
        ),
        ThreadNode(
            id="p6d-a2",
            node_type="answer",
            post_id=6302,
            body_text=(
                "Regime split template: low-gap branch + high-gap branch + trivial "
                "zero-score branch; glue by disjunction at each step."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "reformulation"},
        ),
        ThreadNode(
            id="p6d-l",
            node_type="answer",
            post_id=6303,
            body_text=(
                "Low-gap branch (open): if p90 gap is small, prove perturbative "
                "shadow transfer from Y_t(v) to rank-1 proxy sigma_v q_v q_v^T, "
                "then certify min_v ||Y_t(v)|| < 1."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "open"},
        ),
        ThreadNode(
            id="p6d-h",
            node_type="answer",
            post_id=6304,
            body_text=(
                "High-gap branch (open): if mean gap is large, convert drift control "
                "into score control using ||Y|| = tr(Y)/gap, yielding a good vertex."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "open"},
        ),
        ThreadNode(
            id="p6d-t",
            node_type="answer",
            post_id=6305,
            body_text=(
                "Trivial branch (proved): if min_v score_t(v)=0 then GPL-H step is "
                "immediate for any theta>0."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "proved"},
        ),
        ThreadNode(
            id="p6d-e",
            node_type="comment",
            post_id=6306,
            body_text=run_summary_text(run_path),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6302,
            tags={"verification_status": "empirical"},
        ),
        ThreadNode(
            id="p6d-g",
            node_type="answer",
            post_id=6307,
            body_text=(
                "Glue obligation (open): prove exhaustive either-or theorem so every "
                "step falls into low-gap, high-gap, or trivial branch with a "
                "certified good vertex."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "open"},
        ),
        ThreadNode(
            id="p6d-s",
            node_type="answer",
            post_id=6308,
            body_text=(
                "Existing sufficiency (proved): if each step has score_t(v)<=theta<1, "
                "barrier closes and yields |S| = Omega(epsilon n)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "proved"},
        ),
    ]

    d.edges = [
        ThreadEdge(
            source="p6d-a1",
            target="p6d-q",
            edge_type="challenge",
            evidence="Uniform near-rank-1 not valid globally; need split",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-a2",
            target="p6d-q",
            edge_type="reform",
            evidence="Replace universal Direction D with branch-wise theorem plan",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-l",
            target="p6d-a2",
            edge_type="reference",
            evidence="Low-gap theorem branch",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-h",
            target="p6d-a2",
            edge_type="reference",
            evidence="High-gap theorem branch",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-t",
            target="p6d-a2",
            edge_type="reference",
            evidence="Zero-score trivial branch",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-e",
            target="p6d-a2",
            edge_type="exemplify",
            evidence="Empirical branch coverage from post-D run",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-g",
            target="p6d-q",
            edge_type="reform",
            evidence="Main missing theorem is an exhaustive branch disjunction",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-s",
            target="p6d-g",
            edge_type="reference",
            evidence="Once branch theorem gives per-step good vertex, closure follows",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-l",
            target="p6d-g",
            edge_type="assert",
            evidence="Low-gap certification contributes to glue theorem",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-h",
            target="p6d-g",
            edge_type="assert",
            evidence="High-gap certification contributes to glue theorem",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-t",
            target="p6d-g",
            edge_type="assert",
            evidence="Trivial certification contributes to glue theorem",
            detection="structural",
        ),
    ]

    return d


def write_diagram(diagram: ThreadWiringDiagram, output_path: Path) -> None:
    out = diagram_to_dict(diagram)

    vstatus = {}
    for node in out["nodes"]:
        status = node.get("tags", {}).get("verification_status", "unknown")
        vstatus[status] = vstatus.get(status, 0) + 1
    out["stats"]["verification_status"] = vstatus

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"=== {diagram.thread_id} ===")
    print(f"{out['stats']['n_nodes']} nodes, {out['stats']['n_edges']} edges")
    print(f"Edge types: {out['stats']['edge_types']}")
    print(f"Verification: {vstatus}")
    print(f"Wrote {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate post-D wiring diagram")
    ap.add_argument(
        "--run-summary",
        default="data/first-proof/problem6-post-d-regime-split-results.json",
        help="Path to post-D run summary JSON",
    )
    ap.add_argument(
        "--out",
        default="data/first-proof/problem6-post-direction-d-wiring.json",
        help="Output wiring JSON",
    )
    args = ap.parse_args()

    run_path = Path(args.run_summary)
    out_path = Path(args.out)
    write_diagram(build_post_d_diagram(run_path), out_path)


if __name__ == "__main__":
    main()
