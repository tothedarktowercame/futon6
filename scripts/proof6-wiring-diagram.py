#!/usr/bin/env python3
"""Generate a wiring diagram for the Problem 6 (epsilon-light subsets) proof.

Usage:
    python3 scripts/proof6-wiring-diagram.py [--output data/first-proof/problem6-wiring.json]
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


def build_problem6_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p6")

    nodes = [
        ThreadNode(
            id="p6-problem", node_type="question", post_id=6,
            body_text=(
                "Does there exist universal c>0 such that for every finite "
                "undirected graph G=(V,E,w) with nonnegative weights and every "
                "epsilon in (0,1), there exists S subseteq V with |S|>=c*epsilon*|V| "
                "and L_{G[S]}~ <= epsilon*L_G on R^V (where L_{G[S]}~ is zero-padded)?"
            ),
            score=0, creation_date="2026-02-11",
        ),
        ThreadNode(
            id="p6-s1", node_type="answer", post_id=601,
            body_text=(
                "Laplacian decomposes edge-by-edge: L = sum L_e. "
                "Condition L_S <= epsilon*L means internal edges are "
                "spectrally dominated. In effective-resistance frame: "
                "||L^{+/2} L_S L^{+/2}|| <= epsilon."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=6,
        ),
        ThreadNode(
            id="p6-s2", node_type="answer", post_id=602,
            body_text=(
                "K_n tight example: induced K_s has max eigenvalue s. "
                "Condition reduces to s <= epsilon*n. Max epsilon-light "
                "subset: floor(epsilon*n), giving c = 1."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=6,
        ),
        ThreadNode(
            id="p6-s3", node_type="answer", post_id=603,
            body_text=(
                "Probabilistic construction: include each vertex i.i.d. "
                "with probability p = epsilon. E[|S|] = epsilon*n. "
                "E[L_S] = epsilon^2 * L, so E[epsilon*L - L_S] = "
                "epsilon(1-epsilon)*L >= 0. This is expectation-only."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=6,
        ),
        ThreadNode(
            id="p6-s3a", node_type="comment", post_id=6031,
            body_text=(
                "Size concentration: Chernoff gives |S| >= epsilon*n/2 "
                "with probability >= 1 - exp(-epsilon*n/8)."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=603,
        ),
        ThreadNode(
            id="p6-s3b", node_type="comment", post_id=6032,
            body_text=(
                "Spectral expectation: E[L_S] = p^2*L = epsilon^2*L. "
                "Since epsilon^2 < epsilon for epsilon in (0,1), "
                "E[L_S] <= epsilon*L. Need concentration for a realized subset."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=603,
        ),
        ThreadNode(
            id="p6-s4", node_type="answer", post_id=604,
            body_text=(
                "Core technical step: matrix concentration for degree-2 "
                "polynomial in independent Bernoulli variables. "
                "Star domination: L_S <= sum_v Z_v L_v converts "
                "edge-dependent to vertex-independent form."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=6,
        ),
        ThreadNode(
            id="p6-s4a", node_type="comment", post_id=6041,
            body_text=(
                "Star domination with explicit counting: "
                "L_S = sum_{uv} Z_u Z_v L_uv <= (1/2) sum_v Z_v sum_{u~v} L_uv. "
                "With independent Bernoulli Z_v, RHS is a sum of independent PSD matrices."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=604,
        ),
        ThreadNode(
            id="p6-s4b", node_type="comment", post_id=6042,
            body_text=(
                "Matrix Freedman/Bernstein setup: define vertex-reveal "
                "martingale for centered sum X=sum_i (Z_i-p_i)A_i with self-adjoint "
                "differences Delta_i, increment bound ||Delta_i||<=R_*, and predictable "
                "quadratic variation W_n=sum E[Delta_i^2|F_{i-1}]."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=604,
        ),
        ThreadNode(
            id="p6-s5", node_type="answer", post_id=605,
            body_text=(
                "General-graph step is an external dependency in this draft: "
                "assume a published theorem providing universal c0>0 with "
                "|S|>=c0*epsilon*n and L_S<=epsilon*L; this writeup does not "
                "rederive that theorem."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=6,
        ),
        ThreadNode(
            id="p6-s6", node_type="answer", post_id=606,
            body_text=(
                "Conclusion in this draft: unconditionally we prove c<=1 via K_n "
                "and a correct concentration framework; conditional on the external "
                "general theorem (p6-s5), the full existential answer is yes."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=6,
        ),
    ]

    edges = [
        # S1 clarifies the problem setup
        ThreadEdge(
            source="p6-s1", target="p6-problem",
            edge_type="clarify",
            evidence="Rewrite condition in effective-resistance frame",
            detection="structural",
        ),
        # S2 exemplifies with K_n
        ThreadEdge(
            source="p6-s2", target="p6-problem",
            edge_type="exemplify",
            evidence="K_n tight example: s <= epsilon*n, c = 1",
            detection="structural",
        ),
        # S3 asserts the construction
        ThreadEdge(
            source="p6-s3", target="p6-problem",
            edge_type="assert",
            evidence="Random vertex sampling with p = epsilon",
            detection="structural",
        ),
        # S3a clarifies size
        ThreadEdge(
            source="p6-s3a", target="p6-s3",
            edge_type="clarify",
            evidence="Chernoff: |S| >= epsilon*n/2 w.h.p.",
            detection="structural",
        ),
        # S3b clarifies spectral expectation
        ThreadEdge(
            source="p6-s3b", target="p6-s3",
            edge_type="clarify",
            evidence="E[L_S] = epsilon^2*L <= epsilon*L; still expectation-level",
            detection="structural",
        ),
        # S4 challenges: need per-realization bound
        ThreadEdge(
            source="p6-s4", target="p6-s3b",
            edge_type="challenge",
            evidence="Expectation insufficient; need concentration for all directions",
            detection="structural",
        ),
        # S4a clarifies the domination trick
        ThreadEdge(
            source="p6-s4a", target="p6-s4",
            edge_type="clarify",
            evidence="1/2-star domination converts dependent edges to independent vertices",
            detection="structural",
        ),
        # S4b references the concentration tool
        ThreadEdge(
            source="p6-s4b", target="p6-s4",
            edge_type="reference",
            evidence="Matrix Freedman/Bernstein with explicit martingale parameters",
            detection="structural",
        ),
        # S4b uses effective resistance from S1
        ThreadEdge(
            source="p6-s4b", target="p6-s1",
            edge_type="reference",
            evidence="Effective-resistance normalization and leverage bounds",
            detection="structural",
        ),
        # S5 reforms for general graphs
        ThreadEdge(
            source="p6-s5", target="p6-s4",
            edge_type="reform",
            evidence="Mark universal-existence theorem as explicit external dependency",
            detection="structural",
        ),
        # S6 asserts the final answer
        ThreadEdge(
            source="p6-s6", target="p6-problem",
            edge_type="assert",
            evidence="Unconditional: c<=1; conditional on p6-s5 dependency: existential yes",
            detection="structural",
        ),
        # S6 references the construction
        ThreadEdge(
            source="p6-s6", target="p6-s3",
            edge_type="reference",
            evidence="Probabilistic construction with p = epsilon",
            detection="structural",
        ),
        # S6 references concentration
        ThreadEdge(
            source="p6-s6", target="p6-s5",
            edge_type="reference",
            evidence="Final existential claim depends on external theorem assumption",
            detection="structural",
        ),
        # S6 references tightness
        ThreadEdge(
            source="p6-s6", target="p6-s2",
            edge_type="reference",
            evidence="K_n shows c = 1 is tight",
            detection="structural",
        ),
    ]

    diagram.nodes = nodes
    diagram.edges = edges
    return diagram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o",
                        default="data/first-proof/problem6-wiring.json")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    diagram = build_problem6_proof_diagram()

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
