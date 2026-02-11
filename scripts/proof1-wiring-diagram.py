#!/usr/bin/env python3
"""Generate a wiring diagram for Problem 1 (Phi^4_3 measure equivalence)."""

import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from futon6.thread_performatives import (
    ThreadNode, ThreadEdge, ThreadWiringDiagram, diagram_to_dict, diagram_stats,
)

def build_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p1")
    diagram.nodes = [
        ThreadNode(id="p1-problem", node_type="question", post_id=1,
            body_text="Are mu (Phi^4_3 measure) and T_psi^* mu equivalent (same null sets) for smooth nonzero psi?",
            score=0, creation_date="2026-02-11"),
        ThreadNode(id="p1-s1", node_type="answer", post_id=101,
            body_text="Phi^4_3 measure: dmu = Z^{-1} exp(-V) dmu_0 where V = int(:phi^4: - C:phi^2:)dx. Constructed by Hairer 2014, GIP 2015, Barashkov-Gubinelli 2020.",
            score=0, creation_date="2026-02-11", parent_post_id=1),
        ThreadNode(id="p1-s2", node_type="answer", post_id=102,
            body_text="mu ~ mu_0 (equivalent to GFF): density exp(-V) > 0 a.s. and in L^1(mu_0). So mu and mu_0 have the same null sets.",
            score=0, creation_date="2026-02-11", parent_post_id=1),
        ThreadNode(id="p1-s3", node_type="answer", post_id=103,
            body_text="Cameron-Martin: psi smooth => psi in H^1 = CM space of GFF. So T_psi^* mu_0 ~ mu_0 with explicit RN derivative.",
            score=0, creation_date="2026-02-11", parent_post_id=1),
        ThreadNode(id="p1-s4", node_type="answer", post_id=104,
            body_text="Shifted interaction: V(phi-psi) - V(phi) involves 4*int(psi :phi^3:) + lower order. Requires renormalization (mass counterterm shift for psi^2 :phi^2: term).",
            score=0, creation_date="2026-02-11", parent_post_id=1),
        ThreadNode(id="p1-s4a", node_type="comment", post_id=1041,
            body_text="Regularity: :phi^3: in C^{-3/2-eps}, psi smooth => int(psi :phi^3:)dx well-defined. Renormalized shift has finite expectation.",
            score=0, creation_date="2026-02-11", parent_post_id=104),
        ThreadNode(id="p1-s5", node_type="answer", post_id=105,
            body_text="Integrability: exp(cubic perturbation) in L^1(mu) by log-Sobolev inequality (Barashkov-Gubinelli 2020). Quartic coercivity dominates cubic.",
            score=0, creation_date="2026-02-11", parent_post_id=1),
        ThreadNode(id="p1-s6", node_type="answer", post_id=106,
            body_text="Conclusion: Yes, mu ~ T_psi^* mu. Chain: mu ~ mu_0, T_psi^* mu_0 ~ mu_0 (CM), renormalized RN derivative is positive and integrable.",
            score=0, creation_date="2026-02-11", parent_post_id=1),
    ]
    diagram.edges = [
        ThreadEdge(source="p1-s1", target="p1-problem", edge_type="clarify",
            evidence="Phi^4_3 construction via renormalized density", detection="structural"),
        ThreadEdge(source="p1-s2", target="p1-s1", edge_type="assert",
            evidence="exp(-V) > 0 a.s. gives mu ~ mu_0", detection="structural"),
        ThreadEdge(source="p1-s3", target="p1-problem", edge_type="assert",
            evidence="Cameron-Martin: smooth psi in H^1, shift preserves measure class", detection="structural"),
        ThreadEdge(source="p1-s4", target="p1-s2", edge_type="reform",
            evidence="Reduce to: is V(phi-psi)-V(phi) well-defined and integrable?", detection="structural"),
        ThreadEdge(source="p1-s4a", target="p1-s4", edge_type="clarify",
            evidence="Regularity of Wick powers allows pairing with smooth psi", detection="structural"),
        ThreadEdge(source="p1-s5", target="p1-s4", edge_type="assert",
            evidence="Log-Sobolev + quartic coercivity => exponential integrability of cubic", detection="structural"),
        ThreadEdge(source="p1-s6", target="p1-problem", edge_type="assert",
            evidence="Yes: mu ~ mu_0 ~ T_psi^* mu_0, RN derivative positive and integrable", detection="structural"),
        ThreadEdge(source="p1-s6", target="p1-s3", edge_type="reference",
            evidence="Cameron-Martin equivalence for GFF", detection="structural"),
        ThreadEdge(source="p1-s6", target="p1-s5", edge_type="reference",
            evidence="Integrability from log-Sobolev", detection="structural"),
    ]
    return diagram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/first-proof/problem1-wiring.json")
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
