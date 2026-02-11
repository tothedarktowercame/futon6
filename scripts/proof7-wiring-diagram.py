#!/usr/bin/env python3
"""Generate a wiring diagram for the Problem 7 (lattices/acyclic manifolds) proof."""

import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from futon6.thread_performatives import (
    ThreadNode, ThreadEdge, ThreadWiringDiagram, diagram_to_dict, diagram_stats,
)

def build_problem7_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p7")
    diagram.nodes = [
        ThreadNode(id="p7-problem", node_type="question", post_id=7,
            body_text="Gamma uniform lattice in semi-simple group with 2-torsion. Can Gamma = pi_1(M) for closed M with rationally acyclic universal cover?",
            score=0, creation_date="2026-02-11"),
        ThreadNode(id="p7-s1", node_type="answer", post_id=701,
            body_text="Torsion-free case: Gamma acts freely on X = G/K (contractible symmetric space), M = X/Gamma is aspherical. With torsion: X/Gamma is orbifold, not manifold.",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s2", node_type="answer", post_id=702,
            body_text="Gamma is rational PD group of dim d = dim(X). Orbifold X/Gamma satisfies rational Poincare duality. So H*(Gamma;Q) has PD structure.",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s3", node_type="answer", post_id=703,
            body_text="Surgery theory: existence of closed d-manifold M with pi_1=Gamma and rationally acyclic cover reduces to surgery obstruction in L_d(Z[Gamma]).",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s3a", node_type="comment", post_id=7031,
            body_text="Farrell-Jones conjecture (known for lattices in semi-simple groups, Bartels-Luck 2012) computes L_d(Z[Gamma]) from equivariant homology of X.",
            score=0, creation_date="2026-02-11", parent_post_id=703),
        ThreadNode(id="p7-s4", node_type="answer", post_id=704,
            body_text="Rational surgery obstruction vanishes for suitable choices: odd-dimensional hyperbolic lattices in SO(2k+1,1), k>=3. Parity kills the signature obstruction.",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s5", node_type="answer", post_id=705,
            body_text="Smith theory does NOT obstruct over Q: Z/2 acting freely on rationally acyclic space has no fixed-point constraint (2 invertible in Q). Only mod-2 Smith theory applies.",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s6", node_type="answer", post_id=706,
            body_text="Conclusion: Yes. Rational PD + Farrell-Jones + vanishing surgery obstruction + no Smith obstruction over Q => construction succeeds.",
            score=0, creation_date="2026-02-11", parent_post_id=7),
    ]
    diagram.edges = [
        ThreadEdge(source="p7-s1", target="p7-problem", edge_type="clarify",
            evidence="Torsion-free case works; torsion creates orbifold issue", detection="structural"),
        ThreadEdge(source="p7-s2", target="p7-s1", edge_type="reform",
            evidence="Reframe: Gamma is rational PD group, seek manifold realization", detection="structural"),
        ThreadEdge(source="p7-s3", target="p7-s2", edge_type="assert",
            evidence="Surgery theory reduces to L_d(Z[Gamma]) obstruction", detection="structural"),
        ThreadEdge(source="p7-s3a", target="p7-s3", edge_type="clarify",
            evidence="Farrell-Jones (proven for these lattices) computes L-groups", detection="structural"),
        ThreadEdge(source="p7-s4", target="p7-s3a", edge_type="assert",
            evidence="Rational obstruction vanishes for odd-dim hyperbolic lattices", detection="structural"),
        ThreadEdge(source="p7-s5", target="p7-problem", edge_type="challenge",
            evidence="Smith theory potential obstruction — but only over Z/2, not Q", detection="structural"),
        ThreadEdge(source="p7-s6", target="p7-problem", edge_type="assert",
            evidence="Yes: construction via surgery with Farrell-Jones", detection="structural"),
        ThreadEdge(source="p7-s6", target="p7-s4", edge_type="reference",
            evidence="Uses vanishing surgery obstruction for specific lattices", detection="structural"),
        ThreadEdge(source="p7-s6", target="p7-s5", edge_type="reference",
            evidence="No Q-coefficient Smith obstruction", detection="structural"),
    ]
    return diagram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/first-proof/problem7-wiring.json")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    diagram = build_problem7_proof_diagram()
    if not args.quiet:
        stats = diagram_stats(diagram)
        print(f"=== Proof Wiring Diagram: {diagram.thread_id} ===")
        print(f"{stats['n_nodes']} nodes, {stats['n_edges']} edges")
        print(f"Edge types: {stats['edge_types']}")
        print()
        for edge in diagram.edges:
            arrow = {"challenge": "~~>", "reform": "=>", "clarify": "-->",
                     "assert": "==>", "exemplify": "..>", "reference": "-~>", "agree": "++>"}[edge.edge_type]
            print(f"  {edge.source:12s} {arrow} {edge.target:12s}  [{edge.edge_type}]")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(diagram_to_dict(diagram), f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main()
