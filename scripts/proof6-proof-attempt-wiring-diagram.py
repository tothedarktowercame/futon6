#!/usr/bin/env python3
"""Generate wiring diagrams for Problem 6 proof attempt + GPL zoom.

Usage:
  python3 scripts/proof6-proof-attempt-wiring-diagram.py
  python3 scripts/proof6-proof-attempt-wiring-diagram.py --overall-out path --zoom-out path
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from futon6.thread_performatives import (
    ThreadNode,
    ThreadEdge,
    ThreadWiringDiagram,
    diagram_to_dict,
    diagram_stats,
)


def build_overall_diagram() -> ThreadWiringDiagram:
    d = ThreadWiringDiagram(thread_id="first-proof-p6-attempt")

    d.nodes = [
        ThreadNode(
            id="p6a-problem",
            node_type="question",
            post_id=6100,
            body_text=(
                "Problem 6 theorem target: find universal c0>0 such that for every "
                "weighted graph and epsilon in (0,1), there exists S with "
                "|S|>=c0*epsilon*n and L_{G[S]}<=epsilon*L (induced, no reweighting)."
            ),
            score=0,
            creation_date="2026-02-12",
        ),
        ThreadNode(
            id="p6a-s1",
            node_type="answer",
            post_id=6101,
            body_text=(
                "Leverage threshold lemma: any epsilon-light S must avoid heavy edges "
                "(tau_e>epsilon), so S is independent in G_H."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6100,
        ),
        ThreadNode(
            id="p6a-s2",
            node_type="answer",
            post_id=6102,
            body_text=(
                "Turan bound on heavy graph gives independent set I with |I|>=epsilon*n/3. "
                "Case split: Case 1 and Case 2a close; only Case 2b remains."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6100,
        ),
        ThreadNode(
            id="p6a-s3",
            node_type="answer",
            post_id=6103,
            body_text=(
                "Subsampling/trace route has sublinear ceiling in worst case; cannot prove "
                "universal linear-size c0*epsilon*n from current hypotheses."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6100,
        ),
        ThreadNode(
            id="p6a-s4",
            node_type="answer",
            post_id=6104,
            body_text=(
                "Core regularization + L1: extract I0 with bounded leverage degree; "
                "prove averaged drift bound for barrier-normalized grouped updates Y_t(v)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6100,
        ),
        ThreadNode(
            id="p6a-s5",
            node_type="answer",
            post_id=6105,
            body_text=(
                "Reduction layer: (i) L2+L3 imply linear-size closure; "
                "(ii) sharper: L2* alone already implies existence."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6100,
        ),
        ThreadNode(
            id="p6a-s6",
            node_type="answer",
            post_id=6106,
            body_text=(
                "MSS/KS gap map: standard interlacing/paving/concentration tools do not "
                "directly prove the grouped fixed-block step needed for L2*."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6100,
        ),
        ThreadNode(
            id="p6a-s7",
            node_type="answer",
            post_id=6107,
            body_text=(
                "Open bridge formalized as GPL-H: from H1-H4 infer min_v ||Y_t(v)||<=theta "
                "for t<=c_step*epsilon*n."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6100,
        ),
        ThreadNode(
            id="p6a-s8",
            node_type="answer",
            post_id=6108,
            body_text=(
                "Current proof status: all preprocessing/reduction pieces are in place; "
                "single remaining open implication is GPL-H conclusion from H1-H4."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6100,
        ),
        ThreadNode(
            id="p6a-c1",
            node_type="comment",
            post_id=61081,
            body_text=(
                "Empirical signal for L2*: baseline and randomized trajectories keep "
                "max_t min_v score_t(v) below 1 on tested Case-2b instances."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6108,
        ),
        ThreadNode(
            id="p6a-c2",
            node_type="comment",
            post_id=61082,
            body_text=(
                "Stronger sufficient condition sum_v Y_t(v)<=rho I with rho<1 fails in dense "
                "cases, so GPL-H needs finer anisotropic structure."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6108,
        ),
    ]

    d.edges = [
        ThreadEdge(
            source="p6a-s1",
            target="p6a-problem",
            edge_type="clarify",
            evidence="Reformulate feasibility via heavy-edge exclusion",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s2",
            target="p6a-s1",
            edge_type="reference",
            evidence="Turan converts heavy-edge exclusion into large candidate set I",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s2",
            target="p6a-problem",
            edge_type="reform",
            evidence="Reduce theorem to Case 2b after Cases 1/2a",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s3",
            target="p6a-s2",
            edge_type="challenge",
            evidence="Subsampling/trace cannot close Case 2b with universal c0",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s4",
            target="p6a-s3",
            edge_type="reform",
            evidence="Switch from trace-only to barrier/grouped-update framework",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s5",
            target="p6a-s4",
            edge_type="assert",
            evidence="Reductions show closure follows from L2* (L3 optional)",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s6",
            target="p6a-s5",
            edge_type="challenge",
            evidence="Existing MSS/KS templates do not directly provide L2*",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s7",
            target="p6a-s6",
            edge_type="reform",
            evidence="State precise open theorem as GPL-H (H1-H4 => min-score)",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s8",
            target="p6a-problem",
            edge_type="assert",
            evidence="Overall proof reduces to one open GPL-H implication",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-s8",
            target="p6a-s7",
            edge_type="reference",
            evidence="Open bridge isolated at H1-H4 => conclusion step",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-c1",
            target="p6a-s7",
            edge_type="exemplify",
            evidence="Trajectory tests support feasibility of theta<1 regime",
            detection="structural",
        ),
        ThreadEdge(
            source="p6a-c2",
            target="p6a-s7",
            edge_type="clarify",
            evidence="Rules out overly strong budget route; narrows search space",
            detection="structural",
        ),
    ]

    return d


def build_gpl_zoom_diagram() -> ThreadWiringDiagram:
    d = ThreadWiringDiagram(thread_id="first-proof-p6-gpl-zoom")

    d.nodes = [
        ThreadNode(
            id="p6z-q",
            node_type="question",
            post_id=6200,
            body_text=(
                "GPL-H zoom target: from H1-H4, prove min_{v in R_t} ||Y_t(v)||<=theta<1 "
                "for t<=c_step*epsilon*n."
            ),
            score=0,
            creation_date="2026-02-12",
        ),
        ThreadNode(
            id="p6z-a1",
            node_type="answer",
            post_id=6201,
            body_text="H1: all internal edges of I0 are light (tau_uv<=epsilon).",
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6200,
        ),
        ThreadNode(
            id="p6z-a2",
            node_type="answer",
            post_id=6202,
            body_text=(
                "H2: core regularity bound ell_v<=D0/epsilon on I0 (coarse proved constant D0=12)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6200,
        ),
        ThreadNode(
            id="p6z-a3",
            node_type="answer",
            post_id=6203,
            body_text="H3: barrier-valid prefix M_t<=epsilon I.",
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6200,
        ),
        ThreadNode(
            id="p6z-a4",
            node_type="answer",
            post_id=6204,
            body_text=(
                "H4: residual pool size r_t>=eta*epsilon*n (enforced by choosing c_step small enough)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6200,
        ),
        ThreadNode(
            id="p6z-a5",
            node_type="answer",
            post_id=6205,
            body_text=(
                "Definitions: Y_t(v)=B_t^(1/2)C_t(v)B_t^(1/2), score_t(v)=||Y_t(v)||."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6200,
        ),
        ThreadNode(
            id="p6z-c1",
            node_type="comment",
            post_id=62051,
            body_text=(
                "Proved weak averaging bound: min_v score_t(v) <= (tD/r_t) tr(B_t); "
                "not enough for universal theta<1."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6205,
        ),
        ThreadNode(
            id="p6z-c2",
            node_type="comment",
            post_id=62052,
            body_text=(
                "Strong budget condition sum_v Y_t(v)<=rho I (rho<1) fails in dense examples."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6205,
        ),
        ThreadNode(
            id="p6z-a6",
            node_type="answer",
            post_id=6206,
            body_text=(
                "If zoom target holds, pick such v_t each step; then C_t(v_t)<=theta(epsilon I-M_t), "
                "so barrier is preserved and |S|=Omega(epsilon n)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6200,
        ),
        ThreadNode(
            id="p6z-open",
            node_type="answer",
            post_id=6207,
            body_text=(
                "Open implication: H1-H4 => min_v ||Y_t(v)||<=theta. This is the only missing theorem-level bridge."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=6200,
        ),
    ]

    d.edges = [
        ThreadEdge(
            source="p6z-a5",
            target="p6z-q",
            edge_type="clarify",
            evidence="Defines grouped barrier objects and score",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-a1",
            target="p6z-q",
            edge_type="reference",
            evidence="Hypothesis H1",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-a2",
            target="p6z-q",
            edge_type="reference",
            evidence="Hypothesis H2",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-a3",
            target="p6z-q",
            edge_type="reference",
            evidence="Hypothesis H3",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-a4",
            target="p6z-q",
            edge_type="reference",
            evidence="Hypothesis H4",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-c1",
            target="p6z-q",
            edge_type="challenge",
            evidence="Trace averaging too weak to prove theta<1",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-c2",
            target="p6z-q",
            edge_type="challenge",
            evidence="Strong PSD budget route fails empirically in dense cases",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-a6",
            target="p6z-q",
            edge_type="assert",
            evidence="Zoom target is sufficient for linear-size existence",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-open",
            target="p6z-q",
            edge_type="reform",
            evidence="Isolates single missing implication as theorem-level bridge",
            detection="structural",
        ),
        ThreadEdge(
            source="p6z-open",
            target="p6z-a6",
            edge_type="reference",
            evidence="Once proved, closure follows by barrier-preserving selection",
            detection="structural",
        ),
    ]

    return d


def write_diagram(diagram: ThreadWiringDiagram, output_path: str, quiet: bool = False) -> None:
    if not quiet:
        stats = diagram_stats(diagram)
        print(f"=== {diagram.thread_id} ===")
        print(f"{stats['n_nodes']} nodes, {stats['n_edges']} edges")
        print(f"Edge types: {stats['edge_types']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(diagram_to_dict(diagram), f, indent=2, ensure_ascii=False)

    if not quiet:
        print(f"Wrote {output_path}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--overall-out",
        default="data/first-proof/problem6-proof-attempt-wiring.json",
        help="Output path for overall proof-attempt wiring JSON",
    )
    ap.add_argument(
        "--zoom-out",
        default="data/first-proof/problem6-proof-attempt-gpl-zoom-wiring.json",
        help="Output path for GPL-H zoom wiring JSON",
    )
    ap.add_argument("--quiet", "-q", action="store_true")
    args = ap.parse_args()

    write_diagram(build_overall_diagram(), args.overall_out, quiet=args.quiet)
    write_diagram(build_gpl_zoom_diagram(), args.zoom_out, quiet=args.quiet)


if __name__ == "__main__":
    main()
