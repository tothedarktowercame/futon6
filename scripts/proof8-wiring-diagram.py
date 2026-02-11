#!/usr/bin/env python3
"""Generate a wiring diagram for the Problem 8 (Lagrangian smoothing) proof.

Usage:
    python3 scripts/proof8-wiring-diagram.py
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


def build_problem8_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p8")

    nodes = [
        ThreadNode(
            id="p8-problem", node_type="question", post_id=8,
            body_text=(
                "Polyhedral Lagrangian surface K in R^4, 4 faces per vertex. "
                "Does K necessarily have a Lagrangian smoothing "
                "(Hamiltonian isotopy to smooth Lagrangian)?"
            ),
            score=0, creation_date="2026-02-11",
        ),
        ThreadNode(
            id="p8-s1", node_type="answer", post_id=801,
            body_text=(
                "Lagrangian Grassmannian Lambda(2) = U(2)/O(2), pi_1 = Z "
                "(Maslov class). Each face in a Lagrangian plane, edges are "
                "creases, vertices are multi-plane singularities."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=8,
        ),
        ThreadNode(
            id="p8-s2", node_type="answer", post_id=802,
            body_text=(
                "4-valent vertex structure: 4 faces pair into two 'sheets' "
                "(opposite faces). Two sheets cross transversally at vertex, "
                "forming a Lagrangian crossing (transverse double point)."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=8,
        ),
        ThreadNode(
            id="p8-s3", node_type="answer", post_id=803,
            body_text=(
                "Maslov index of vertex loop L_1->L_2->L_3->L_4->L_1 is 0. "
                "Transverse Lagrangian crossing has vanishing Maslov index. "
                "The paired-sheet decomposition inherits this."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=8,
        ),
        ThreadNode(
            id="p8-s4", node_type="answer", post_id=804,
            body_text=(
                "Lagrangian surgery (Polterovich 1991, Lalonde-Sikorav 1991): "
                "transverse double point of two Lagrangian disks can be "
                "resolved by a smooth Lagrangian neck. Via generating function."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=8,
        ),
        ThreadNode(
            id="p8-s4a", node_type="comment", post_id=8041,
            body_text=(
                "Surgery in Darboux coords: L_1 = {y=0}, L_2 transverse. "
                "Replace crossing with smooth y = grad S(x) neck; "
                "omega|_{graph} = 0 automatically."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=804,
        ),
        ThreadNode(
            id="p8-s5", node_type="answer", post_id=805,
            body_text=(
                "Application: resolve vertices (Lagrangian surgery on paired "
                "sheets), smooth edges (generating function interpolation), "
                "compose Hamiltonian isotopies for global K_t."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=8,
        ),
        ThreadNode(
            id="p8-s5a", node_type="comment", post_id=8051,
            body_text=(
                "4-face condition is essential: odd valence prevents "
                "sheet pairing; vertex Maslov index may be nonzero, "
                "obstructing Lagrangian surgery."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=805,
        ),
        ThreadNode(
            id="p8-s6", node_type="answer", post_id=806,
            body_text=(
                "Conclusion: Yes, smoothing exists. 4-face => paired sheets "
                "=> transverse crossing => Maslov 0 => surgery unobstructed "
                "=> Hamiltonian isotopy to smooth Lagrangian."
            ),
            score=0, creation_date="2026-02-11", parent_post_id=8,
        ),
    ]

    edges = [
        ThreadEdge(
            source="p8-s1", target="p8-problem",
            edge_type="clarify",
            evidence="Lagrangian Grassmannian and Maslov class setup",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s2", target="p8-s1",
            edge_type="reform",
            evidence="4 faces pair into two crossing sheets at each vertex",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s3", target="p8-s2",
            edge_type="assert",
            evidence="Maslov index of transverse crossing is 0",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s4", target="p8-problem",
            edge_type="assert",
            evidence="Lagrangian surgery resolves transverse double points",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s4a", target="p8-s4",
            edge_type="exemplify",
            evidence="Darboux coordinates construction of surgery neck",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s5", target="p8-s4",
            edge_type="assert",
            evidence="Compose vertex surgery + edge smoothing for global K_t",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s5", target="p8-s3",
            edge_type="reference",
            evidence="Maslov 0 ensures surgery is unobstructed",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s5a", target="p8-s5",
            edge_type="challenge",
            evidence="Odd valence breaks pairing; obstruction possible",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s6", target="p8-problem",
            edge_type="assert",
            evidence="Yes: 4-face => paired sheets => surgery => smoothing",
            detection="structural",
        ),
        ThreadEdge(
            source="p8-s6", target="p8-s5",
            edge_type="reference",
            evidence="Full smoothing pipeline from vertex + edge resolution",
            detection="structural",
        ),
    ]

    diagram.nodes = nodes
    diagram.edges = edges
    return diagram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o",
                        default="data/first-proof/problem8-wiring.json")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    diagram = build_problem8_proof_diagram()

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
