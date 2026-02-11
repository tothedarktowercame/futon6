#!/usr/bin/env python3
"""Generate a wiring diagram for Problem 8 (Lagrangian smoothing) v2.

v2 incorporates the symplectic direct sum decomposition argument.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from futon6.thread_performatives import (
    ThreadNode, ThreadEdge, ThreadWiringDiagram, diagram_to_dict, diagram_stats,
)

def build_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p8")
    diagram.nodes = [
        ThreadNode(id="p8-problem", node_type="question", post_id=8,
            body_text="Polyhedral Lagrangian K in R^4, 4 faces per vertex. Does K have a Lagrangian smoothing?",
            score=0, creation_date="2026-02-11"),
        ThreadNode(id="p8-s1", node_type="answer", post_id=801,
            body_text="Lambda(2) = U(2)/O(2), pi_1 = Z (Maslov class). Each face Lagrangian, edges are creases, vertices are multi-plane singularities.",
            score=0, creation_date="2026-02-11", parent_post_id=8),
        ThreadNode(id="p8-s2", node_type="answer", post_id=802,
            body_text="Edge-sharing: L_i = span(e_{i-1,i}, e_{i,i+1}). Lagrangian condition: omega(e_{i-1,i}, e_{i,i+1}) = 0 for each face. This kills 4 of 6 omega entries in edge basis.",
            score=0, creation_date="2026-02-11", parent_post_id=8),
        ThreadNode(id="p8-s3", node_type="answer", post_id=803,
            body_text="SYMPLECTIC DIRECT SUM: remaining omega entries are a=omega(e1,e3), b=omega(e2,e4). In basis (e1,e3,e2,e4), omega is block diagonal. R^4 = V1 ⊕ V2 with V1=span(opposite edges).",
            score=0, creation_date="2026-02-11", parent_post_id=8),
        ThreadNode(id="p8-s4", node_type="answer", post_id=804,
            body_text="Each L_i = (line in V1) ⊕ (line in V2). Maslov loop decomposes: mu = mu1 + mu2. Each mu_j = winding number in RP^1. Both trace back-and-forth (not winding). mu = 0 exactly.",
            score=0, creation_date="2026-02-11", parent_post_id=8),
        ThreadNode(id="p8-s5", node_type="answer", post_id=805,
            body_text="Lagrangian surgery (Polterovich 1991): transverse crossing with Maslov 0 resolved by smooth neck. Darboux coords adapted to V1 ⊕ V2. Neck via generating function y = grad S(x).",
            score=0, creation_date="2026-02-11", parent_post_id=8),
        ThreadNode(id="p8-s6", node_type="answer", post_id=806,
            body_text="Global smoothing: resolve vertices (surgery), smooth edges (generating function interpolation), compose Hamiltonian isotopies for K_t.",
            score=0, creation_date="2026-02-11", parent_post_id=8),
        ThreadNode(id="p8-s7", node_type="answer", post_id=807,
            body_text="3-face impossible: 3 Lagrangian faces need omega=0 on all edge pairs (all adjacent = all pairs for 3). 3 isotropic vectors can't span 3D in R^4 (max isotropic dim = 2).",
            score=0, creation_date="2026-02-11", parent_post_id=8),
        ThreadNode(id="p8-c1", node_type="comment", post_id=8011,
            body_text="Numerical: 998/998 valid 4-valent configs give mu=0. Without edge-sharing, only 55% give mu=0. The constraint is essential, not generic.",
            score=0, creation_date="2026-02-11", parent_post_id=804),
    ]
    diagram.edges = [
        ThreadEdge(source="p8-s1", target="p8-problem", edge_type="clarify",
            evidence="Lagrangian Grassmannian and Maslov class setup", detection="structural"),
        ThreadEdge(source="p8-s2", target="p8-s1", edge_type="reform",
            evidence="Edge-sharing kills omega entries in edge basis", detection="structural"),
        ThreadEdge(source="p8-s3", target="p8-s2", edge_type="assert",
            evidence="Block diagonal omega → symplectic direct sum V1 ⊕ V2", detection="structural"),
        ThreadEdge(source="p8-s4", target="p8-s3", edge_type="assert",
            evidence="Decomposed Maslov loop has winding 0 in each factor", detection="structural"),
        ThreadEdge(source="p8-s5", target="p8-s4", edge_type="reference",
            evidence="Maslov 0 is the hypothesis for Polterovich surgery", detection="structural"),
        ThreadEdge(source="p8-s5", target="p8-problem", edge_type="assert",
            evidence="Lagrangian surgery resolves transverse double points", detection="structural"),
        ThreadEdge(source="p8-s6", target="p8-s5", edge_type="assert",
            evidence="Compose vertex surgery + edge smoothing for global K_t", detection="structural"),
        ThreadEdge(source="p8-s6", target="p8-problem", edge_type="assert",
            evidence="Yes: 4-face → V1⊕V2 → Maslov 0 → surgery → smoothing", detection="structural"),
        ThreadEdge(source="p8-s7", target="p8-s3", edge_type="challenge",
            evidence="3-face: isotropic dimension bound prevents non-degenerate vertex", detection="structural"),
        ThreadEdge(source="p8-c1", target="p8-s4", edge_type="exemplify",
            evidence="998/998 numerical verification; comparison with non-edge-sharing", detection="classical"),
    ]
    return diagram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/first-proof/problem8-wiring.json")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    diagram = build_proof_diagram()
    if not args.quiet:
        stats = diagram_stats(diagram)
        print(f"=== Proof Wiring Diagram: {diagram.thread_id} ===")
        print(f"{stats['n_nodes']} nodes, {stats['n_edges']} edges")
        print(f"Edge types: {stats['edge_types']}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(diagram_to_dict(diagram), f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main()
