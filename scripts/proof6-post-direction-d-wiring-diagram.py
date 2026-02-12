#!/usr/bin/env python3
"""Generate post-Direction-D wiring diagram for Problem 6.

Current bridge focus:
- proved ratio certificate: min score <= dbar/gbar
- open AR-NT target: nontrivial => dbar<gbar
- proved AR lemmas: exact rho threshold and valid extremal sufficient threshold
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
            "Post-D ratio run not loaded yet. Expected output at "
            f"{run_path}."
        )

    try:
        payload = json.loads(run_path.read_text())
        s = payload["summary"]
        return (
            "Empirical AR diagnostics (n<=48): "
            f"nontrivial_ratio_fail={s['nontrivial_ratio_fail_rows']}, "
            f"rho_safe_fail={s['rho_safe_fail_rows']}, "
            f"rho_simple_fail={s['rho_simple_fail_rows']}, "
            f"rho_safe_margin_min={s['rho_safe_margin_min']:.3f}."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Could not parse aggregate-ratio summary at {run_path}: {exc}"


def build_post_d_diagram(run_path: Path) -> ThreadWiringDiagram:
    d = ThreadWiringDiagram(thread_id="first-proof-p6-post-d")

    d.nodes = [
        ThreadNode(
            id="p6d-q",
            node_type="question",
            post_id=6300,
            body_text="Post-D target: close GPL-H from H1-H4.",
            score=0,
            creation_date="2026-02-12",
            tags={"verification_status": "open"},
        ),
        ThreadNode(
            id="p6d-a1",
            node_type="answer",
            post_id=6301,
            body_text=(
                "Direction D finding: uniform near-rank-1 universality fails "
                "at n<=48 (dense late-step rows have rank gap > 2)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "proved_negative"},
        ),
        ThreadNode(
            id="p6d-r",
            node_type="answer",
            post_id=6302,
            body_text=(
                "Ratio certificate (proved): with active-set averages "
                "dbar=avg tr(Y_t(v)), gbar=avg tr(Y_t(v))/||Y_t(v)||, "
                "we have min_v ||Y_t(v)|| <= dbar/gbar."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "proved"},
        ),
        ThreadNode(
            id="p6d-ar",
            node_type="answer",
            post_id=6303,
            body_text=(
                "AR-NT target (open): under H1-H4, if min-score > 0 then dbar < gbar."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "open"},
        ),
        ThreadNode(
            id="p6d-c",
            node_type="answer",
            post_id=6304,
            body_text=(
                "Bridge lemmas (proved): AR1 exact reciprocal form, AR2 exact "
                "rho_exact threshold, AR3 valid extremal sufficient threshold rho_safe(M_-,m_+)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "proved"},
        ),
        ThreadNode(
            id="p6d-b",
            node_type="answer",
            post_id=6305,
            body_text=(
                "Open bridge obligation: derive rho_+ < rho_exact (or rho_+ < rho_safe) from H1-H4 on nontrivial steps."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "open"},
        ),
        ThreadNode(
            id="p6d-t",
            node_type="answer",
            post_id=6306,
            body_text=(
                "Trivial branch (proved): if min-score = 0, step is already good for any theta>0."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "proved"},
        ),
        ThreadNode(
            id="p6d-e",
            node_type="comment",
            post_id=6307,
            body_text=run_summary_text(run_path),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "empirical"},
        ),
        ThreadNode(
            id="p6d-s",
            node_type="answer",
            post_id=6308,
            body_text=(
                "Existing sufficiency (proved): per-step min-score < 1 implies barrier closure "
                "and |S| = Omega(epsilon n)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "proved"},
        ),
        ThreadNode(
            id="p6d-o",
            node_type="comment",
            post_id=6309,
            body_text=(
                "Optional shortcut AR4 (rho_+ < 1-M_-) is only sufficient, not equivalent; "
                "empirically it has one nontrivial miss at n<=48."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6300,
            tags={"verification_status": "empirical"},
        ),
    ]

    d.edges = [
        ThreadEdge(
            source="p6d-a1",
            target="p6d-q",
            edge_type="challenge",
            evidence="Uniform near-rank-1 route fails as unconditional bridge",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-r",
            target="p6d-q",
            edge_type="reform",
            evidence="Reduce min-score control to aggregate ratio dbar/gbar",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-ar",
            target="p6d-q",
            edge_type="reform",
            evidence="Current open target is AR-NT (nontrivial dbar<gbar)",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-c",
            target="p6d-ar",
            edge_type="reference",
            evidence="AR-NT reduced to exact/safe rho_+ threshold control",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-b",
            target="p6d-ar",
            edge_type="assert",
            evidence="Remaining theorem-level bridge is proving rho_+ threshold from H1-H4",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-t",
            target="p6d-ar",
            edge_type="assert",
            evidence="Trivial min-score=0 rows are already solved",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-e",
            target="p6d-ar",
            edge_type="exemplify",
            evidence="Empirics support AR-NT and the rho_safe bridge on tested nontrivial rows",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-s",
            target="p6d-ar",
            edge_type="reference",
            evidence="AR-NT + ratio certificate implies full step closure",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-r",
            target="p6d-c",
            edge_type="reference",
            evidence="AR bridge lemmas are built on ratio-certificate variables",
            detection="structural",
        ),
        ThreadEdge(
            source="p6d-o",
            target="p6d-c",
            edge_type="challenge",
            evidence="One-parameter shortcut is too strong to be the primary bridge",
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
        default="data/first-proof/problem6-aggregate-ratio-results.json",
        help="Path to aggregate-ratio run summary JSON",
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
