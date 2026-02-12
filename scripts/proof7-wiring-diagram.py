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
            body_text="Rational PD structure comes from orbifold/Bredon cohomology for proper cocompact Gamma-action on X=G/K, with formal dimension d=dim(X).",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s3", node_type="answer", post_id=703,
            body_text="Fowler's equivariant finiteness theorem gives a finite-CW result: under fixed-set Euler-vanishing hypotheses, the orbifold group Gamma lies in FH(Q).",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s3a", node_type="comment", post_id=7031,
            body_text="Fowler's arithmetic-lattice constructions supply concrete lattice extensions in FH(Q), but these are finite-complex outputs, not yet closed manifolds.",
            score=0, creation_date="2026-02-11", parent_post_id=703),
        ThreadNode(id="p7-s4", node_type="answer", post_id=704,
            body_text="Open step: upgrade FH(Q) to a closed manifold with the SAME pi_1=Gamma and verify the needed rational surgery obstruction vanishing for the selected 2-torsion lattice.",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s5", node_type="answer", post_id=705,
            body_text="Smith theory does NOT obstruct over Q: Z/2 acting freely on rationally acyclic space has no fixed-point constraint (2 invertible in Q). Only mod-2 Smith theory applies.",
            score=0, creation_date="2026-02-11", parent_post_id=7),
        ThreadNode(id="p7-s6", node_type="answer", post_id=706,
            body_text="Conditional conclusion: if a 2-torsion lattice Gamma is placed in FH(Q) and the manifold-upgrade surgery obstruction vanishes, then the desired closed manifold exists.",
            score=0, creation_date="2026-02-11", parent_post_id=7),
    ]
    diagram.edges = [
        ThreadEdge(source="p7-s1", target="p7-problem", edge_type="clarify",
            evidence="Torsion-free case works; torsion creates orbifold issue", detection="structural"),
        ThreadEdge(source="p7-s2", target="p7-s1", edge_type="reform",
            evidence="Reframe: Gamma is rational PD group, seek manifold realization", detection="structural"),
        ThreadEdge(source="p7-s3", target="p7-s2", edge_type="assert",
            evidence="Equivariant finiteness gives finite-CW rationally acyclic models (FH(Q)) under fixed-set Euler criteria", detection="structural"),
        ThreadEdge(source="p7-s3a", target="p7-s3", edge_type="clarify",
            evidence="Arithmetic-lattice examples show the theorem is not purely abstract", detection="structural"),
        ThreadEdge(source="p7-s4", target="p7-s3a", edge_type="assert",
            evidence="Remaining gap is manifold-upgrade with pi_1 control plus explicit obstruction computation", detection="structural"),
        ThreadEdge(source="p7-s5", target="p7-problem", edge_type="challenge",
            evidence="Smith theory potential obstruction — but only over Z/2, not Q", detection="structural"),
        ThreadEdge(source="p7-s6", target="p7-problem", edge_type="assert",
            evidence="Conditional yes after FH(Q) realization plus manifold-upgrade obstruction vanishing", detection="structural"),
        ThreadEdge(source="p7-s6", target="p7-s4", edge_type="reference",
            evidence="Depends on closing the manifold-upgrade and obstruction gap", detection="structural"),
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
