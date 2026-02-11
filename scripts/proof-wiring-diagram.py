#!/usr/bin/env python3
"""Generate a wiring diagram for the Problem 10 (RKHS tensor CP/PCG) proof.

Uses the Stage 7 ThreadWiringDiagram infrastructure to model proof steps
as nodes and logical relationships as typed IATC edges. This is a
meta-demonstration: the pipeline models its own mathematical output.

Usage:
    python3 scripts/proof-wiring-diagram.py [--output data/first-proof/problem10-wiring.json]
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from futon6.thread_performatives import (
    ThreadNode, ThreadEdge, ThreadWiringDiagram,
    diagram_to_dict, diagram_to_hyperedges, diagram_stats,
)


def build_problem10_proof_diagram() -> ThreadWiringDiagram:
    """Build a wiring diagram for the Problem 10 proof.

    The proof has 7 sections with substeps. Each section/substep becomes a
    node; logical dependencies between them become typed edges using the
    IATC performative vocabulary.
    """
    diagram = ThreadWiringDiagram(thread_id="first-proof-p10")

    # --- Nodes: proof steps ---

    nodes = [
        ThreadNode(
            id="p10-problem",
            node_type="question",
            post_id=10,
            body_text=(
                "Given the mode-k subproblem [(Z⊗K)ᵀSS'(Z⊗K) + λ(Iᵣ⊗K)]vec(W) "
                "= (Iᵣ⊗K)vec(B), explain how PCG solves this without O(N) computation."
            ),
            score=0,
            creation_date="2026-02-11",
        ),
        ThreadNode(
            id="p10-s1",
            node_type="answer",
            post_id=101,
            body_text=(
                "Why direct methods fail: The system matrix A is nr×nr. Direct "
                "solve costs O(n³r³). But forming A explicitly requires materializing "
                "(Z⊗K) ∈ ℝ^{N×nr}, costing O(Nnr) — proportional to N. Infeasible."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=10,
        ),
        ThreadNode(
            id="p10-s2",
            node_type="answer",
            post_id=102,
            body_text=(
                "The key insight: CG only requires the action v ↦ Av, never the "
                "matrix A itself. We compute this in O(n²r + qr), independent of N."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=10,
        ),
        ThreadNode(
            id="p10-s2a",
            node_type="comment",
            post_id=1021,
            body_text=(
                "Forward map at observed entries: By the Kronecker identity "
                "(A⊗B)vec(X) = vec(BXAᵀ), we get (Z⊗K)vec(V) = vec(KVZᵀ). "
                "SS'selects only q entries. Grouping by row: O(n²r + qr)."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=102,
        ),
        ThreadNode(
            id="p10-s2b",
            node_type="comment",
            post_id=1022,
            body_text=(
                "Adjoint map from sparse result: The sparse vector w ∈ ℝ^N maps "
                "back via (Zᵀ⊗K)w = vec(KW'Z) where W' has q nonzeros. "
                "W'Z: O(qr), K(W'Z): O(n²r). Total: O(qr + n²r)."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=102,
        ),
        ThreadNode(
            id="p10-s2c",
            node_type="comment",
            post_id=1023,
            body_text=(
                "Regularization term: λ(Iᵣ⊗K)vec(V) = λ·vec(KV). Cost: O(n²r)."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=102,
        ),
        ThreadNode(
            id="p10-s2-total",
            node_type="comment",
            post_id=1024,
            body_text=(
                "Total per matvec: O(n²r + qr). No dependence on N. "
                "This is the central result that makes PCG feasible."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=102,
        ),
        ThreadNode(
            id="p10-s3",
            node_type="answer",
            post_id=103,
            body_text=(
                "Right-hand side: b = (Iᵣ⊗K)vec(B) where B = TZ. "
                "T is sparse (q nonzeros), so TZ: O(qr), KB: O(n²r). "
                "Total: O(qr + n²r)."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=10,
        ),
        ThreadNode(
            id="p10-s4",
            node_type="answer",
            post_id=104,
            body_text=(
                "Preconditioner choice: P = (H⊗K) where H = ZᵀZ + λIᵣ. "
                "Approximates A by replacing SSᵀ with I (full observation). "
                "Captures kernel structure (K) and inter-factor coupling (ZᵀZ)."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=10,
        ),
        ThreadNode(
            id="p10-s4-hadamard",
            node_type="comment",
            post_id=1041,
            body_text=(
                "Why this structure? The Khatri-Rao Hadamard property gives "
                "ZᵀZ = (A₁ᵀA₁)*(A₂ᵀA₂)*...*(Aₐᵀ Aₐ). Each Aᵢᵀ Aᵢ costs "
                "O(nᵢr²), so ZᵀZ costs O(Σᵢ nᵢr²) — vastly cheaper than O(Mr²)."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=104,
        ),
        ThreadNode(
            id="p10-s4-solve",
            node_type="comment",
            post_id=1042,
            body_text=(
                "Preconditioner solve: P⁻¹ = H⁻¹⊗K⁻¹. Precompute Cholesky of K "
                "O(n³) and H O(r³). Each application: solve KYHᵀ = Z' via "
                "triangular substitution. Cost per solve: O(n²r)."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=104,
        ),
        ThreadNode(
            id="p10-s5",
            node_type="answer",
            post_id=105,
            body_text=(
                "Convergence: CG on P⁻¹A converges in t = O(√κ·log(1/ε)) "
                "iterations. Since P ≈ A well when q/N is not too small, "
                "practical convergence: t = O(r) to O(r√(n/q)·log(1/ε))."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=10,
        ),
        ThreadNode(
            id="p10-s6",
            node_type="answer",
            post_id=106,
            body_text=(
                "Complexity summary: Total per mode-k subproblem O(n³ + t(n²r + qr)) "
                "where t ~ O(r). Compare direct solve O(n³r³ + Nnr). PCG replaces "
                "N-dependent term with q-dependent, n³r³ with n³ + n²r². Since "
                "n, r ≪ q ≪ N, this achieves the required complexity reduction."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=10,
        ),
        ThreadNode(
            id="p10-s7",
            node_type="answer",
            post_id=107,
            body_text=(
                "Algorithm: SETUP computes Cholesky factors and RHS. PCG iteration "
                "applies implicit matvec and preconditioner solve per step. "
                "MATVEC evaluates at observed entries only. "
                "PRECOND_SOLVE uses Kronecker inverse structure."
            ),
            score=0,
            creation_date="2026-02-11",
            parent_post_id=10,
        ),
    ]

    diagram.nodes = nodes

    # --- Edges: logical relationships ---
    # Using IATC vocabulary: assert, challenge, query, clarify, reform,
    # exemplify, reference, agree, retract

    edges = [
        # S1 challenges the naive approach
        ThreadEdge(
            source="p10-s1", target="p10-problem",
            edge_type="challenge",
            evidence="Direct methods cost O(Nnr), proportional to N — infeasible",
            detection="structural",
        ),
        # S2 reforms the problem: from forming A to implicit matvec
        ThreadEdge(
            source="p10-s2", target="p10-s1",
            edge_type="reform",
            evidence="CG only requires v ↦ Av, never the matrix A itself",
            detection="structural",
        ),
        # S2a clarifies S2: details the forward map
        ThreadEdge(
            source="p10-s2a", target="p10-s2",
            edge_type="clarify",
            evidence="Kronecker identity gives forward map; grouping by row avoids N",
            detection="structural",
        ),
        # S2b clarifies S2: details the adjoint map
        ThreadEdge(
            source="p10-s2b", target="p10-s2",
            edge_type="clarify",
            evidence="Sparse W' has only q nonzeros; adjoint via (Zᵀ⊗K)",
            detection="structural",
        ),
        # S2c clarifies S2: the regularization term
        ThreadEdge(
            source="p10-s2c", target="p10-s2",
            edge_type="clarify",
            evidence="Regularization is just λ·vec(KV), O(n²r)",
            detection="structural",
        ),
        # S2-total asserts the combined cost
        ThreadEdge(
            source="p10-s2-total", target="p10-s2a",
            edge_type="assert",
            evidence="Combining forward + adjoint + regularization: O(n²r + qr)",
            detection="structural",
        ),
        ThreadEdge(
            source="p10-s2-total", target="p10-s2b",
            edge_type="assert",
            evidence="Combining forward + adjoint + regularization: O(n²r + qr)",
            detection="structural",
        ),
        ThreadEdge(
            source="p10-s2-total", target="p10-s2c",
            edge_type="agree",
            evidence="Regularization absorbed into n²r term",
            detection="structural",
        ),
        # S3 clarifies: RHS computation uses same sparse structure
        ThreadEdge(
            source="p10-s3", target="p10-problem",
            edge_type="clarify",
            evidence="RHS uses sparse T and Kronecker structure, same O(qr + n²r)",
            detection="structural",
        ),
        # S4 reforms: introduces preconditioner to accelerate CG
        ThreadEdge(
            source="p10-s4", target="p10-s2",
            edge_type="reform",
            evidence="P = (H⊗K) approximates A for faster convergence",
            detection="structural",
        ),
        # S4-hadamard clarifies: why the preconditioner is cheap
        ThreadEdge(
            source="p10-s4-hadamard", target="p10-s4",
            edge_type="clarify",
            evidence="Khatri-Rao Hadamard: ZᵀZ = ∏(AᵢᵀAᵢ), O(Σnᵢr²) not O(Mr²)",
            detection="classical",
        ),
        # S4-solve exemplifies: how to apply the preconditioner
        ThreadEdge(
            source="p10-s4-solve", target="p10-s4",
            edge_type="exemplify",
            evidence="P⁻¹ = H⁻¹⊗K⁻¹ via Cholesky; each solve O(n²r)",
            detection="structural",
        ),
        # S5 asserts convergence, depends on preconditioner quality
        ThreadEdge(
            source="p10-s5", target="p10-s4",
            edge_type="assert",
            evidence="P ≈ A ⟹ κ(P⁻¹A) small ⟹ t = O(r) iterations",
            detection="structural",
        ),
        # S6 asserts the final answer, aggregating all costs
        ThreadEdge(
            source="p10-s6", target="p10-problem",
            edge_type="assert",
            evidence="O(n³ + t(n²r + qr)), no N-dependence, QED",
            detection="structural",
        ),
        # S6 references the components it aggregates
        ThreadEdge(
            source="p10-s6", target="p10-s2-total",
            edge_type="reference",
            evidence="Matvec cost O(n²r + qr) per iteration",
            detection="structural",
        ),
        ThreadEdge(
            source="p10-s6", target="p10-s4-solve",
            edge_type="reference",
            evidence="Preconditioner solve O(n²r) per iteration",
            detection="structural",
        ),
        ThreadEdge(
            source="p10-s6", target="p10-s5",
            edge_type="reference",
            evidence="t = O(r) iterations from convergence analysis",
            detection="structural",
        ),
        # S7 exemplifies: the algorithm instantiates the analysis
        ThreadEdge(
            source="p10-s7", target="p10-s6",
            edge_type="exemplify",
            evidence="Pseudocode instantiates the complexity analysis",
            detection="structural",
        ),
        # S7 also references the core components
        ThreadEdge(
            source="p10-s7", target="p10-s2",
            edge_type="reference",
            evidence="MATVEC subroutine implements implicit matvec",
            detection="structural",
        ),
        ThreadEdge(
            source="p10-s7", target="p10-s4-solve",
            edge_type="reference",
            evidence="PRECOND_SOLVE subroutine implements Kronecker inverse",
            detection="structural",
        ),
    ]

    diagram.edges = edges

    return diagram


def print_ascii_diagram(diagram: ThreadWiringDiagram):
    """Print a human-readable ASCII summary of the wiring diagram."""
    print(f"=== Proof Wiring Diagram: {diagram.thread_id} ===\n")

    print("NODES:")
    for node in diagram.nodes:
        prefix = {"question": "?", "answer": "A", "comment": "C"}[node.node_type]
        print(f"  [{prefix}] {node.id}")
        # first 80 chars of body
        text = node.body_text[:80].replace("\n", " ")
        print(f"      {text}...")
    print()

    print("EDGES:")
    for edge in diagram.edges:
        arrow = {
            "challenge": "~~>",
            "reform": "=>",
            "clarify": "-->",
            "assert": "==>",
            "exemplify": "..>",
            "reference": "-~>",
            "agree": "++>",
            "retract": "x->",
            "query": "?->",
        }.get(edge.edge_type, "-->")

        print(f"  {edge.source}  {arrow}  {edge.target}")
        print(f"      [{edge.edge_type}] {edge.evidence[:70]}")
    print()

    stats = diagram_stats(diagram)
    print(f"STATS: {stats['n_nodes']} nodes, {stats['n_edges']} edges")
    print(f"  Node types: {stats['node_types']}")
    print(f"  Edge types: {stats['edge_types']}")
    print(f"  Detection: {stats['detection_types']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate wiring diagram for Problem 10 proof")
    parser.add_argument(
        "--output", "-o",
        default="data/first-proof/problem10-wiring.json",
        help="Output JSON path (default: data/first-proof/problem10-wiring.json)")
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress ASCII output")
    args = parser.parse_args()

    diagram = build_problem10_proof_diagram()

    if not args.quiet:
        print_ascii_diagram(diagram)
        print()

    # Write JSON
    out = diagram_to_dict(diagram)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")
    print(f"  {len(out['nodes'])} nodes, {len(out['edges'])} edges, "
          f"{len(out['hyperedges'])} hyperedges")


if __name__ == "__main__":
    main()
