#!/usr/bin/env python3
"""Generate a wiring diagram for Problem 5 (O-slice connectivity via geometric fixed points)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from futon6.thread_performatives import (
    ThreadNode, ThreadEdge, ThreadWiringDiagram, diagram_to_dict, diagram_stats,
)

def build_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p5")
    diagram.nodes = [
        ThreadNode(id="p5-problem", node_type="question", post_id=5,
            body_text="Define O-slice filtration for incomplete transfer system O and characterize O-slice connectivity of connective G-spectrum via geometric fixed points.",
            score=0, creation_date="2026-02-11"),
        ThreadNode(id="p5-s1", node_type="answer", post_id=501,
            body_text="Classical slice filtration (HHR): tau^{>n}_G localizing subcategories. Slice >= n iff Phi^H X is (n|G/H|-1)-connected for all H <= G.",
            score=0, creation_date="2026-02-11", parent_post_id=5),
        ThreadNode(id="p5-s2", node_type="answer", post_id=502,
            body_text="N_infinity operad O encodes partial commutative ring structure. Transfer system T_O specifies which norm maps N_H^K exist.",
            score=0, creation_date="2026-02-11", parent_post_id=5),
        ThreadNode(id="p5-s3", node_type="answer", post_id=503,
            body_text="O-slice cells: G_+ wedge_H S^{n rho_H^O} for H in T_O. rho_H^O = O-regular representation. Filtration coarser than classical when O incomplete.",
            score=0, creation_date="2026-02-11", parent_post_id=5),
        ThreadNode(id="p5-s4", node_type="answer", post_id=504,
            body_text="Theorem: X is O-slice >= n iff Phi^H X is (n * d_H^O - 1)-connected for all H in T_O, where d_H^O = dim_R(rho_H^O)/|H|.",
            score=0, creation_date="2026-02-11", parent_post_id=5),
        ThreadNode(id="p5-s5", node_type="answer", post_id=505,
            body_text="Proof: Slice cells detect connectivity (5a). Tom Dieck splitting gives Phi^H(G_+ wedge_H S^V) = S^{dim(V)} (5b). Connectivity equivalence (5c). Only H in T_O contribute (5d).",
            score=0, creation_date="2026-02-11", parent_post_id=5),
        ThreadNode(id="p5-s6", node_type="answer", post_id=506,
            body_text="Special cases: Complete T -> classical HHR (d_H=1). Trivial T -> Postnikov filtration (only H=e). C_p intermediate cases interpolate.",
            score=0, creation_date="2026-02-11", parent_post_id=5),
    ]
    diagram.edges = [
        ThreadEdge(source="p5-s1", target="p5-problem", edge_type="clarify",
            evidence="Classical HHR slice filtration as baseline", detection="structural"),
        ThreadEdge(source="p5-s2", target="p5-problem", edge_type="clarify",
            evidence="N_infinity operads and incomplete transfer systems", detection="structural"),
        ThreadEdge(source="p5-s3", target="p5-s2", edge_type="reform",
            evidence="Recast transfer system as O-slice cells with O-regular representations", detection="structural"),
        ThreadEdge(source="p5-s3", target="p5-s1", edge_type="reform",
            evidence="Modify classical slice cells to account for incomplete transfers", detection="structural"),
        ThreadEdge(source="p5-s4", target="p5-problem", edge_type="assert",
            evidence="Main theorem: O-slice >= n iff Phi^H X (n*d_H^O - 1)-connected for H in T_O", detection="structural"),
        ThreadEdge(source="p5-s4", target="p5-s3", edge_type="reference",
            evidence="Uses O-slice cells and d_H^O from construction", detection="structural"),
        ThreadEdge(source="p5-s5", target="p5-s4", edge_type="assert",
            evidence="Proof via tom Dieck splitting and cell detection", detection="structural"),
        ThreadEdge(source="p5-s5", target="p5-s1", edge_type="reference",
            evidence="Parallels classical HHR proof structure", detection="structural"),
        ThreadEdge(source="p5-s6", target="p5-s4", edge_type="exemplify",
            evidence="Complete, trivial, and C_p cases verify theorem reduces correctly", detection="structural"),
        ThreadEdge(source="p5-s6", target="p5-s1", edge_type="reference",
            evidence="Complete case recovers classical HHR criterion", detection="structural"),
    ]
    return diagram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/first-proof/problem5-wiring.json")
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
