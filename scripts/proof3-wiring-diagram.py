#!/usr/bin/env python3
"""Generate a wiring diagram for Problem 3 (ASEP Markov chain)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from futon6.thread_performatives import (
    ThreadNode, ThreadEdge, ThreadWiringDiagram, diagram_to_dict, diagram_stats,
)

def build_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p3")
    diagram.nodes = [
        ThreadNode(id="p3-problem", node_type="question", post_id=3,
            body_text="Nontrivial Markov chain on S_n(lambda) with stationary dist F*_mu/P*_lambda at q=1?",
            score=0, creation_date="2026-02-11"),
        ThreadNode(id="p3-s1", node_type="answer", post_id=301,
            body_text="State space S_n(lambda) = n! rearrangements of lambda's distinct parts. q=1 specialization reduces to Hall-Littlewood-type structure.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s2", node_type="answer", post_id=302,
            body_text="Multispecies ASEP: nearest-neighbor swaps of adjacent parts with asymmetric rates depending on t and x_i/x_{i+1}.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s3", node_type="answer", post_id=303,
            body_text="Hecke algebra rates: T_i generator gives M_i = (T_i + t*I)/(1+t). Satisfies T_i^2 = (t-1)T_i + t.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s4", node_type="answer", post_id=304,
            body_text="Stationarity from exchange relations: T_i F*_mu = c F*_mu + d F*_{s_i mu}. Detailed balance follows from Hecke algebra relation.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s5", node_type="answer", post_id=305,
            body_text="Nontriviality: rates depend on (t, x_i), not F*_mu values. Chain is a genuine exclusion process, not a trivial Gibbs sampler.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s6", node_type="answer", post_id=306,
            body_text="Conclusion: Yes. Multispecies ASEP with Hecke algebra rates on S_n(lambda). Stationary distribution = F*_mu/P*_lambda at q=1.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
    ]
    diagram.edges = [
        ThreadEdge(source="p3-s1", target="p3-problem", edge_type="clarify",
            evidence="S_n(lambda) = n! states; q=1 gives Hall-Littlewood", detection="structural"),
        ThreadEdge(source="p3-s2", target="p3-problem", edge_type="reform",
            evidence="Recast as multispecies exclusion process with swaps", detection="structural"),
        ThreadEdge(source="p3-s3", target="p3-s2", edge_type="clarify",
            evidence="Hecke algebra gives transition rates", detection="structural"),
        ThreadEdge(source="p3-s4", target="p3-s3", edge_type="assert",
            evidence="Exchange relations + Hecke identity => detailed balance", detection="structural"),
        ThreadEdge(source="p3-s4", target="p3-s1", edge_type="reference",
            evidence="Uses q=1 simplification of exchange coefficients", detection="structural"),
        ThreadEdge(source="p3-s5", target="p3-s2", edge_type="assert",
            evidence="Rates are local functions of (t, x_i), not F*_mu", detection="structural"),
        ThreadEdge(source="p3-s6", target="p3-problem", edge_type="assert",
            evidence="Yes: multispecies ASEP with Hecke rates", detection="structural"),
        ThreadEdge(source="p3-s6", target="p3-s4", edge_type="reference",
            evidence="Stationarity from exchange relations", detection="structural"),
        ThreadEdge(source="p3-s6", target="p3-s5", edge_type="reference",
            evidence="Nontriviality from local rates", detection="structural"),
    ]
    return diagram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/first-proof/problem3-wiring.json")
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
