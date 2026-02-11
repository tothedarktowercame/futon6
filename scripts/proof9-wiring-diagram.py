#!/usr/bin/env python3
"""Generate a wiring diagram for the Problem 9 (quadrifocal tensor) proof.

Usage:
    python3 scripts/proof9-wiring-diagram.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from futon6.thread_performatives import (
    ThreadNode, ThreadEdge, ThreadWiringDiagram,
    diagram_to_dict, diagram_stats,
)


def build_problem9_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p9")

    nodes = [
        ThreadNode(
            id="p9-problem", node_type="question", post_id=9,
            body_text=(
                "Given n>=5 generic 3x4 matrices, Q^(abgd)_{ijkl} = det[rows]. "
                "Does a polynomial map F (camera-independent, bounded degree) "
                "detect rank-1 scaling lambda = u⊗v⊗w⊗x?"
            ),
            score=0, creation_date="2026-02-11",
        ),
        ThreadNode(
            id="p9-s1", node_type="answer", post_id=901,
            body_text=(
                "Bilinear form reduction: fix (gamma,k) and (delta,l). "
                "omega(p,q) = det[p;q;c;d] is alternating bilinear on R^4 "
                "with rank 2 (Hodge dual of simple 2-form c∧d)."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=9,
        ),
        ThreadNode(
            id="p9-s1a", node_type="comment", post_id=9011,
            body_text=(
                "Null space of omega is span{c,d}; omega induces non-degenerate "
                "alternating form on V/span{c,d} ≅ R^2."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=901,
        ),
        ThreadNode(
            id="p9-s2", node_type="answer", post_id=902,
            body_text=(
                "3x3 minor formulation: rank-2 bilinear form => all 3x3 "
                "minors det[omega(p_m,q_n)]_{3x3} = 0 for any choice of "
                "6 camera-row pairs. This is degree-3, camera-independent."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=9,
        ),
        ThreadNode(
            id="p9-s3", node_type="answer", post_id=903,
            body_text=(
                "Rank-1 scaling preserves vanishing: if lambda = u⊗v⊗w⊗x, "
                "then Lambda_{mn} = u_{a_m} v_{b_n} w_g x_d has matrix rank 1. "
                "M = Lambda ∘ Omega => rank(M) = rank(Omega) = 2 < 3."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=9,
        ),
        ThreadNode(
            id="p9-s3a", node_type="comment", post_id=9031,
            body_text=(
                "Hadamard product with rank-1 matrix = row/column scaling: "
                "M = diag(u) Omega diag(v) * const, preserving rank."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=903,
        ),
        ThreadNode(
            id="p9-s4", node_type="answer", post_id=904,
            body_text=(
                "Converse: non-rank-1 lambda => some 3x3 minor nonzero. "
                "For generic cameras, projected vectors f_bar in R^2 are in "
                "general position; Hadamard product with rank>=2 Lambda "
                "generically gives rank-3 minor."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=9,
        ),
        ThreadNode(
            id="p9-s4a", node_type="comment", post_id=9041,
            body_text=(
                "Zariski-genericity: det(Lambda ∘ Omega) = 0 for ALL "
                "row/column triples is a polynomial condition on cameras "
                "that doesn't vanish identically; fails generically."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=904,
        ),
        ThreadNode(
            id="p9-s5", node_type="answer", post_id=905,
            body_text=(
                "All matricizations: test (1,2) vs (3,4), (1,3) vs (2,4), "
                "(1,4) vs (2,3) by fixing different mode pairs. "
                "Rank-1 in all three matricizations <=> lambda rank-1."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=9,
        ),
        ThreadNode(
            id="p9-s6", node_type="answer", post_id=906,
            body_text=(
                "Construction of F: coordinate functions are all 3x3 minors "
                "det[T^(a_m,b_n,g,d)_{i_m,j_n,k,l}]_{3x3} over all "
                "mode-pair fixings. Degree 3, camera-independent, O(n^8) eqns."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=9,
        ),
        ThreadNode(
            id="p9-s7", node_type="answer", post_id=907,
            body_text=(
                "Geometric interpretation: Q tensors are quadrifocal tensors "
                "from multiview geometry. Rank-1 scaling = gauge freedom. "
                "F detects consistency with the gauge group."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=9,
        ),
    ]

    edges = [
        # S1 clarifies the bilinear form structure
        ThreadEdge(
            source="p9-s1", target="p9-problem",
            edge_type="clarify",
            evidence="Fix two modes => alternating bilinear form of rank 2",
            detection="structural",
        ),
        # S1a clarifies S1
        ThreadEdge(
            source="p9-s1a", target="p9-s1",
            edge_type="clarify",
            evidence="Null space is span{c,d}, quotient is R^2",
            detection="structural",
        ),
        # S2 reformulates via 3x3 minors
        ThreadEdge(
            source="p9-s2", target="p9-s1",
            edge_type="reform",
            evidence="Rank 2 => all 3x3 minors vanish; degree-3 polynomials",
            detection="structural",
        ),
        # S3 asserts forward direction
        ThreadEdge(
            source="p9-s3", target="p9-s2",
            edge_type="assert",
            evidence="Rank-1 Lambda => Hadamard preserves rank => det still 0",
            detection="structural",
        ),
        # S3a clarifies with Hadamard product rank
        ThreadEdge(
            source="p9-s3a", target="p9-s3",
            edge_type="clarify",
            evidence="Hadamard with rank-1 = diagonal scaling, preserves rank",
            detection="structural",
        ),
        # S4 asserts converse
        ThreadEdge(
            source="p9-s4", target="p9-s3",
            edge_type="assert",
            evidence="Non-rank-1 gives rank-3 Hadamard product for generic cameras",
            detection="structural",
        ),
        # S4a clarifies genericity argument
        ThreadEdge(
            source="p9-s4a", target="p9-s4",
            edge_type="clarify",
            evidence="Zariski genericity: polynomial non-vanishing condition",
            detection="structural",
        ),
        # S5 clarifies all matricizations needed
        ThreadEdge(
            source="p9-s5", target="p9-s4",
            edge_type="clarify",
            evidence="Three mode-pair tests cover all rank-1 conditions",
            detection="structural",
        ),
        # S6 asserts the construction
        ThreadEdge(
            source="p9-s6", target="p9-problem",
            edge_type="assert",
            evidence="F = all 3x3 minors; degree 3, camera-independent",
            detection="structural",
        ),
        # S6 references key components
        ThreadEdge(
            source="p9-s6", target="p9-s2",
            edge_type="reference",
            evidence="Minor formulation from bilinear form rank",
            detection="structural",
        ),
        ThreadEdge(
            source="p9-s6", target="p9-s5",
            edge_type="reference",
            evidence="All three matricizations included in F",
            detection="structural",
        ),
        # S7 provides geometric context
        ThreadEdge(
            source="p9-s7", target="p9-problem",
            edge_type="exemplify",
            evidence="Quadrifocal tensors from multiview geometry",
            detection="structural",
        ),
        ThreadEdge(
            source="p9-s7", target="p9-s6",
            edge_type="reference",
            evidence="Gauge freedom interpretation of rank-1 scaling",
            detection="structural",
        ),
    ]

    diagram.nodes = nodes
    diagram.edges = edges
    return diagram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o",
                        default="data/first-proof/problem9-wiring.json")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    diagram = build_problem9_proof_diagram()

    if not args.quiet:
        stats = diagram_stats(diagram)
        print(f"=== Proof Wiring Diagram: {diagram.thread_id} ===")
        print(f"{stats['n_nodes']} nodes, {stats['n_edges']} edges")
        print(f"Edge types: {stats['edge_types']}")
        print()
        for edge in diagram.edges:
            arrow = {"challenge": "~~>", "reform": "=>", "clarify": "-->",
                     "assert": "==>", "exemplify": "..>", "reference": "-~>",
                     "agree": "++>"}[edge.edge_type]
            print(f"  {edge.source:12s} {arrow} {edge.target:12s}  [{edge.edge_type}]")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = diagram_to_dict(diagram)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
