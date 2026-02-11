#!/usr/bin/env python3
"""Generate a wiring diagram for Problem 2 (Rankin-Selberg test vector)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from futon6.thread_performatives import (
    ThreadNode, ThreadEdge, ThreadWiringDiagram, diagram_to_dict, diagram_stats,
)

def build_proof_diagram() -> ThreadWiringDiagram:
    diagram = ThreadWiringDiagram(thread_id="first-proof-p2")
    diagram.nodes = [
        ThreadNode(id="p2-problem", node_type="question", post_id=2,
            body_text="Must there exist W in W(Pi, psi^{-1}) such that for all generic pi of GL_n, the u_Q-twisted Rankin-Selberg integral is finite and nonzero for all s?",
            score=0, creation_date="2026-02-11"),
        ThreadNode(id="p2-s1", node_type="answer", post_id=201,
            body_text="Rankin-Selberg theory: I(s,W,V) converges for Re(s)>>0, extends to rational function of q_F^{-s}. Integrals generate L(s,Pi x pi) * C[q^s,q^{-s}].",
            score=0, creation_date="2026-02-11", parent_post_id=2),
        ThreadNode(id="p2-s2", node_type="answer", post_id=202,
            body_text="Finite and nonzero for all s means f(q^{-s}) has no poles/zeros on C^x, hence equals c*q^{-ks}. Requires V to cancel L-factor to a monomial.",
            score=0, creation_date="2026-02-11", parent_post_id=2),
        ThreadNode(id="p2-s3", node_type="answer", post_id=203,
            body_text="u_Q twist: R(u_Q)W nonzero for all Q since right-translation by unipotent preserves Whittaker model. Restriction to mirabolic gives nonzero Kirillov function.",
            score=0, creation_date="2026-02-11", parent_post_id=2),
        ThreadNode(id="p2-s4", node_type="answer", post_id=204,
            body_text="Nondegeneracy: nonzero phi_Q in Kirillov model => integrals over V span full fractional ideal (JPSS 1983). Can choose V to get any element of ideal.",
            score=0, creation_date="2026-02-11", parent_post_id=2),
        ThreadNode(id="p2-s5", node_type="answer", post_id=205,
            body_text="Choose W_0 = new vector of Pi. R(u_Q)W_0 nonzero for all Q. u_Q shift at scale q^{-c(pi)} matches conductor of pi, ensuring support overlap.",
            score=0, creation_date="2026-02-11", parent_post_id=2),
        ThreadNode(id="p2-s6", node_type="answer", post_id=206,
            body_text="Conclusion: Yes. New vector W_0 is universal test vector. u_Q compensates conductor; V chosen from fractional ideal to cancel L-factor to monomial.",
            score=0, creation_date="2026-02-11", parent_post_id=2),
        ThreadNode(id="p2-c1", node_type="comment", post_id=2001,
            body_text="Without u_Q twist, integral degenerates for highly ramified pi: new vector of Pi misses conductor scale. The twist is the key innovation.",
            score=0, creation_date="2026-02-11", parent_post_id=203),
    ]
    diagram.edges = [
        ThreadEdge(source="p2-s1", target="p2-problem", edge_type="clarify",
            evidence="Rankin-Selberg integral theory and fractional ideal structure", detection="structural"),
        ThreadEdge(source="p2-s2", target="p2-s1", edge_type="clarify",
            evidence="Translates 'finite and nonzero for all s' to algebraic condition on monomial", detection="structural"),
        ThreadEdge(source="p2-s3", target="p2-problem", edge_type="reform",
            evidence="Recast as Kirillov model nondegeneracy via u_Q twist", detection="structural"),
        ThreadEdge(source="p2-s4", target="p2-s3", edge_type="assert",
            evidence="JPSS nondegeneracy: nonzero Kirillov function pairs nontrivially with all pi", detection="structural"),
        ThreadEdge(source="p2-s4", target="p2-s1", edge_type="reference",
            evidence="Uses fractional ideal structure from RS theory", detection="structural"),
        ThreadEdge(source="p2-s5", target="p2-s4", edge_type="assert",
            evidence="New vector satisfies nondegeneracy; u_Q matches conductor scale", detection="structural"),
        ThreadEdge(source="p2-s5", target="p2-s2", edge_type="reference",
            evidence="Uses monomial condition to choose V", detection="structural"),
        ThreadEdge(source="p2-s6", target="p2-problem", edge_type="assert",
            evidence="Yes: new vector is universal test vector with u_Q twist", detection="structural"),
        ThreadEdge(source="p2-s6", target="p2-s5", edge_type="reference",
            evidence="Universality from new vector + conductor matching", detection="structural"),
        ThreadEdge(source="p2-s6", target="p2-s4", edge_type="reference",
            evidence="Nondegeneracy guarantees V exists for each pi", detection="structural"),
        ThreadEdge(source="p2-c1", target="p2-s3", edge_type="clarify",
            evidence="Explains why u_Q twist is necessary: conductor mismatch", detection="classical"),
    ]
    return diagram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/first-proof/problem2-wiring.json")
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
