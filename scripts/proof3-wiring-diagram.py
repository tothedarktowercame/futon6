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
            body_text="Does there exist a nontrivial Markov chain on S_n(lambda) with stationary law F*_mu/P*_lambda at q=1?",
            score=0, creation_date="2026-02-11"),
        ThreadNode(id="p3-s1", node_type="answer", post_id=301,
            body_text="Construct chain: inhomogeneous multispecies t-PushTASEP on ring with site rates 1/x_i and t-geometric weaker-particle choice.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s2", node_type="answer", post_id=302,
            body_text="Finite-state CTMC on S_n(lambda): explicit generator, well-defined rates, at least one vacancy from lambda_n=0.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s3", node_type="answer", post_id=303,
            body_text="Nontriviality lemma: transition rates depend only on local species ordering, x_i, and t, not on F*_mu values.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s4", node_type="answer", post_id=304,
            body_text="AMW Theorem 1.1: stationary distribution of this chain is pi(eta)=F_eta(x;1,t)/P_lambda(x;1,t).",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s5", node_type="answer", post_id=305,
            body_text="Notation bridge: prompt's F*_mu,P*_lambda are the same q=1 ASEP/Macdonald family (up to harmless normalization conventions).",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s6", node_type="answer", post_id=306,
            body_text="Sanity lemma n=2: lambda=(a,0) gives two-state chain with rates 1/x_1 and 1/x_2, yielding explicit stationary ratio x_1:x_2.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
        ThreadNode(id="p3-s7", node_type="answer", post_id=307,
            body_text="Conclusion: yes, this explicit nontrivial chain realizes the required stationary ratio F*_mu/P*_lambda.",
            score=0, creation_date="2026-02-11", parent_post_id=3),
    ]
    diagram.edges = [
        ThreadEdge(source="p3-s1", target="p3-problem", edge_type="reform",
            evidence="Explicit candidate chain: inhomogeneous t-PushTASEP", detection="structural"),
        ThreadEdge(source="p3-s2", target="p3-s1", edge_type="clarify",
            evidence="Finite-state CTMC with explicit rates", detection="structural"),
        ThreadEdge(source="p3-s3", target="p3-s1", edge_type="assert",
            evidence="Transition rule independent of F* polynomials", detection="structural"),
        ThreadEdge(source="p3-s4", target="p3-problem", edge_type="assert",
            evidence="Theorem 1.1 gives stationary law F/P at q=1", detection="structural"),
        ThreadEdge(source="p3-s4", target="p3-s1", edge_type="reference",
            evidence="Applies to the same t-PushTASEP dynamics", detection="structural"),
        ThreadEdge(source="p3-s5", target="p3-s4", edge_type="clarify",
            evidence="Identifies paper notation with prompt notation", detection="structural"),
        ThreadEdge(source="p3-s6", target="p3-s1", edge_type="exemplify",
            evidence="n=2 chain computes stationary ratio explicitly", detection="structural"),
        ThreadEdge(source="p3-s7", target="p3-problem", edge_type="assert",
            evidence="Existence + nontriviality + target stationary ratio", detection="structural"),
        ThreadEdge(source="p3-s7", target="p3-s3", edge_type="reference",
            evidence="Uses nontriviality lemma", detection="structural"),
        ThreadEdge(source="p3-s7", target="p3-s4", edge_type="reference",
            evidence="Uses AMW stationary-distribution theorem", detection="structural"),
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
