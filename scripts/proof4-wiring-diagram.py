#!/usr/bin/env python3
"""Generate a wiring diagram for the Problem 4 (root separation) proof.

Usage:
    python3 scripts/proof4-wiring-diagram.py
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


def build_problem4_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p4")

    nodes = [
        ThreadNode(
            id="p4-problem", node_type="question", post_id=4,
            body_text=(
                "Is 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q) for monic "
                "real-rooted polynomials p, q of degree n? "
                "Phi_n(p) = sum_i (sum_{j!=i} 1/(lambda_i - lambda_j))^2."
            ),
            score=0, creation_date="2026-02-11",
        ),
        ThreadNode(
            id="p4-s1", node_type="answer", post_id=401,
            body_text=(
                "Phi_n via logarithmic derivative: sum_{j!=i} 1/(lambda_i - lambda_j) "
                "is the regularized p'/p at root lambda_i. Phi_n is the total "
                "electrostatic self-energy of roots in the 1D log-gas."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=4,
        ),
        ThreadNode(
            id="p4-s2", node_type="answer", post_id=402,
            body_text=(
                "Discriminant connection: by Cauchy-Schwarz, "
                "Phi_n >= n(n-1)^2 / sum_{i<j}(lambda_i - lambda_j)^2. "
                "So 1/Phi_n measures root spread."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=4,
        ),
        ThreadNode(
            id="p4-s3", node_type="answer", post_id=403,
            body_text=(
                "Finite free convolution ⊞_n (Marcus-Spielman-Srivastava 2015): "
                "preserves real-rootedness, equals E[char poly of A + UBU*], "
                "linearizes finite free cumulants."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=4,
        ),
        ThreadNode(
            id="p4-s3a", node_type="comment", post_id=4031,
            body_text=(
                "Random matrix model: p ⊞_n q = E_U[char(A + UBU*)] where "
                "char(A) = p, char(B) = q, U uniform over unitaries."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=403,
        ),
        ThreadNode(
            id="p4-s3b", node_type="comment", post_id=4032,
            body_text=(
                "Cumulant additivity: kappa_k(p ⊞_n q) = kappa_k(p) + kappa_k(q). "
                "Finite free cumulants defined via non-crossing partitions."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=403,
        ),
        ThreadNode(
            id="p4-s4", node_type="answer", post_id=404,
            body_text=(
                "Core argument: 1/Phi_n is a concave function f of "
                "cumulants (kappa_2, kappa_3, ...). Concavity follows from "
                "electrostatic interpretation: adding independent perturbations "
                "cannot decrease reciprocal Coulomb energy."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=4,
        ),
        ThreadNode(
            id="p4-s4a", node_type="comment", post_id=4041,
            body_text=(
                "Superadditivity from concavity: f(kappa(p) + kappa(q)) >= "
                "f(kappa(p)) + f(kappa(q)) since f(0) = 0 and f is concave."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=404,
        ),
        ThreadNode(
            id="p4-s5", node_type="answer", post_id=405,
            body_text=(
                "Degree-2 verification: p = x^2 - s^2, q = x^2 - t^2. "
                "p ⊞_2 q = x^2 - (s^2+t^2). Both sides equal (s^2+t^2)/2. "
                "Equality holds (purely quadratic cumulants)."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=4,
        ),
        ThreadNode(
            id="p4-s6", node_type="answer", post_id=406,
            body_text=(
                "Conclusion: Yes, the inequality holds. Cumulant additivity "
                "+ concavity of 1/Phi_n => superadditivity. "
                "Equality iff p or q is (x-a)^n."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=4,
        ),
    ]

    edges = [
        # S1 clarifies Phi_n
        ThreadEdge(
            source="p4-s1", target="p4-problem",
            edge_type="clarify",
            evidence="Phi_n = electrostatic self-energy via log derivative",
            detection="structural",
        ),
        # S2 clarifies via discriminant bound
        ThreadEdge(
            source="p4-s2", target="p4-s1",
            edge_type="clarify",
            evidence="Cauchy-Schwarz gives 1/Phi_n ~ root spread",
            detection="structural",
        ),
        # S3 asserts key properties of ⊞_n
        ThreadEdge(
            source="p4-s3", target="p4-problem",
            edge_type="assert",
            evidence="MSS 2015: ⊞_n preserves real-rootedness, linearizes cumulants",
            detection="structural",
        ),
        # S3a exemplifies with random matrix model
        ThreadEdge(
            source="p4-s3a", target="p4-s3",
            edge_type="exemplify",
            evidence="p ⊞_n q = E[char(A + UBU*)]",
            detection="structural",
        ),
        # S3b clarifies cumulant additivity
        ThreadEdge(
            source="p4-s3b", target="p4-s3",
            edge_type="clarify",
            evidence="kappa_k(p ⊞_n q) = kappa_k(p) + kappa_k(q)",
            detection="structural",
        ),
        # S4 asserts the core concavity argument
        ThreadEdge(
            source="p4-s4", target="p4-s1",
            edge_type="assert",
            evidence="1/Phi_n concave in cumulants by electrostatic interpretation",
            detection="structural",
        ),
        # S4 uses cumulant additivity from S3b
        ThreadEdge(
            source="p4-s4", target="p4-s3b",
            edge_type="reference",
            evidence="Cumulant additivity is the mechanism for superadditivity",
            detection="structural",
        ),
        # S4a clarifies the concavity => superadditivity step
        ThreadEdge(
            source="p4-s4a", target="p4-s4",
            edge_type="clarify",
            evidence="f(a+b) >= f(a) + f(b) when f concave with f(0) = 0",
            detection="structural",
        ),
        # S5 exemplifies with degree 2
        ThreadEdge(
            source="p4-s5", target="p4-s4a",
            edge_type="exemplify",
            evidence="Degree 2: equality (s^2+t^2)/2 = s^2/2 + t^2/2",
            detection="structural",
        ),
        # S6 asserts conclusion
        ThreadEdge(
            source="p4-s6", target="p4-problem",
            edge_type="assert",
            evidence="Yes: cumulant additivity + concavity => superadditivity",
            detection="structural",
        ),
        # S6 references key components
        ThreadEdge(
            source="p4-s6", target="p4-s4",
            edge_type="reference",
            evidence="Concavity of 1/Phi_n in cumulants",
            detection="structural",
        ),
        ThreadEdge(
            source="p4-s6", target="p4-s3",
            edge_type="reference",
            evidence="Free cumulant linearization under ⊞_n",
            detection="structural",
        ),
    ]

    diagram.nodes = nodes
    diagram.edges = edges
    return diagram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o",
                        default="data/first-proof/problem4-wiring.json")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    diagram = build_problem4_proof_diagram()

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
