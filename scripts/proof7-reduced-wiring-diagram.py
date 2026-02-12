#!/usr/bin/env python3
"""Generate a wiring diagram for the reduced Problem 7 conditional theorem.

Usage:
    python3 scripts/proof7-reduced-wiring-diagram.py \
        [--output data/first-proof/problem7-reduced-wiring.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from futon6.thread_performatives import (
    ThreadEdge,
    ThreadNode,
    ThreadWiringDiagram,
    diagram_stats,
    diagram_to_dict,
)


def build_problem7_reduced_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p7-reduced")

    diagram.nodes = [
        ThreadNode(
            id="p7r-problem",
            node_type="question",
            post_id=7000,
            body_text=(
                "Reduced Problem 7: for a uniform lattice Gamma in a real semisimple "
                "Lie group with an order-2 element, prove the conditional theorem "
                "(E2 + S => existence of closed M with pi_1(M)=Gamma and "
                "H_*(M_tilde;Q)=0 for *>0)."
            ),
            score=0,
            creation_date="2026-02-12",
        ),
        ThreadNode(
            id="p7r-s1",
            node_type="answer",
            post_id=7001,
            body_text=(
                "Decompose into two proof obligations: (E2) place Gamma in FH(Q) "
                "via an equivariant finiteness criterion, and (S) perform a "
                "manifold-upgrade step with pi_1 control and vanishing rational "
                "surgery obstruction."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7000,
        ),
        ThreadNode(
            id="p7r-s2",
            node_type="answer",
            post_id=7002,
            body_text=(
                "Obligation E2 (finite-complex realization): construct/identify a "
                "finite CW complex Y with pi_1(Y)=Gamma and Q-acyclic universal "
                "cover, i.e. Gamma in FH(Q)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7000,
        ),
        ThreadNode(
            id="p7r-s2a",
            node_type="comment",
            post_id=70021,
            body_text=(
                "Fowler template: if an orbifold-group extension from a finite group "
                "action has fixed-set Euler characteristics zero on nontrivial "
                "subgroups/components, then the orbifold group is in FH(Q)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7002,
        ),
        ThreadNode(
            id="p7r-s2b",
            node_type="comment",
            post_id=70022,
            body_text=(
                "Instantiation subtask: choose a concrete cocompact lattice family "
                "with 2-torsion and verify the fixed-set Euler-vanishing hypotheses "
                "for the corresponding finite-group action."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7002,
        ),
        ThreadNode(
            id="p7r-s3",
            node_type="answer",
            post_id=7003,
            body_text=(
                "Obligation S (manifold upgrade): from the finite-complex model, "
                "produce a closed manifold M with the SAME fundamental group Gamma "
                "and Q-acyclic universal cover."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7000,
        ),
        ThreadNode(
            id="p7r-s3a",
            node_type="comment",
            post_id=70031,
            body_text=(
                "Surgery setup interface: identify required dimension/finiteness "
                "conditions, normal map data, and explicit pi_1-isomorphism control "
                "for the target lattice family."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7003,
        ),
        ThreadNode(
            id="p7r-s3b",
            node_type="comment",
            post_id=70032,
            body_text=(
                "Obstruction interface: compute/cite the relevant class in "
                "L_d(Z[Gamma]) (or equivalent surgery obstruction group) and prove "
                "its rational vanishing for the chosen family."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7003,
        ),
        ThreadNode(
            id="p7r-s4",
            node_type="answer",
            post_id=7004,
            body_text=(
                "Side constraint: Smith theory over Z/2 is not a Q-coefficient "
                "obstruction; this rules out a common false negative but does not "
                "construct M."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7000,
        ),
        ThreadNode(
            id="p7r-s5",
            node_type="answer",
            post_id=7005,
            body_text=(
                "Reduced conditional theorem: if obligations E2 and S are both "
                "discharged for a concrete 2-torsion lattice Gamma, then there exists "
                "a closed manifold M with pi_1(M)=Gamma and H_*(M_tilde;Q)=0 for *>0."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7000,
        ),
        ThreadNode(
            id="p7r-s6",
            node_type="answer",
            post_id=7006,
            body_text=(
                "Mining target for papers: extract reusable proof modules matching "
                "nodes p7r-s2a/p7r-s2b (FH(Q) placement) and p7r-s3a/p7r-s3b "
                "(manifold-upgrade plus obstruction vanishing)."
            ),
            score=0,
            creation_date="2026-02-12",
            parent_post_id=7000,
        ),
    ]

    diagram.edges = [
        ThreadEdge(
            source="p7r-s1",
            target="p7r-problem",
            edge_type="clarify",
            evidence="Split the reduced theorem into E2 and S obligations",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s2",
            target="p7r-s1",
            edge_type="assert",
            evidence="E2 is the finite-CW (FH(Q)) obligation",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s2a",
            target="p7r-s2",
            edge_type="reference",
            evidence="Fowler fixed-set Euler criterion as a strategy template",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s2b",
            target="p7r-s2",
            edge_type="exemplify",
            evidence="Concrete lattice-family verification task for E2",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s3",
            target="p7r-s1",
            edge_type="assert",
            evidence="S is the manifold-upgrade with pi_1-preservation obligation",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s3a",
            target="p7r-s3",
            edge_type="clarify",
            evidence="Specify surgery setup and pi_1-control interface",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s3b",
            target="p7r-s3",
            edge_type="reference",
            evidence="Specify obstruction computation and vanishing interface",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s4",
            target="p7r-problem",
            edge_type="challenge",
            evidence="Eliminates the false Smith-theory obstruction over Q",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s5",
            target="p7r-problem",
            edge_type="assert",
            evidence="Conditional existence follows after E2 and S are discharged",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s5",
            target="p7r-s2",
            edge_type="reference",
            evidence="Depends on FH(Q) realization of Gamma",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s5",
            target="p7r-s3",
            edge_type="reference",
            evidence="Depends on manifold-upgrade and obstruction closure",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s5",
            target="p7r-s4",
            edge_type="reference",
            evidence="Uses Smith discussion only as anti-obstruction context",
            detection="structural",
        ),
        ThreadEdge(
            source="p7r-s6",
            target="p7r-s5",
            edge_type="reform",
            evidence="Convert theorem obligations into paper-mining module targets",
            detection="structural",
        ),
    ]

    return diagram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        default="data/first-proof/problem7-reduced-wiring.json",
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    diagram = build_problem7_reduced_diagram()

    if not args.quiet:
        stats = diagram_stats(diagram)
        print(f"=== Proof Wiring Diagram: {diagram.thread_id} ===")
        print(f"{stats['n_nodes']} nodes, {stats['n_edges']} edges")
        print(f"Edge types: {stats['edge_types']}")
        print()
        for edge in diagram.edges:
            arrow = {
                "challenge": "~~>",
                "reform": "=>",
                "clarify": "-->",
                "assert": "==>",
                "exemplify": "..>",
                "reference": "-~>",
                "agree": "++>",
            }[edge.edge_type]
            print(f"  {edge.source:12s} {arrow} {edge.target:12s}  [{edge.edge_type}]")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(diagram_to_dict(diagram), f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
