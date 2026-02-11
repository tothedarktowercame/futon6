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
                "Given [(Z⊗K_tau)ᵀSS'(Z⊗K_tau) + λ(Iᵣ⊗K_tau)]vec(W) "
                "= (Iᵣ⊗K_tau)vec(B), with K_tau = K + τI and λ>0, explain how "
                "PCG solves this without O(N) computation."
            ),
            score=0,
            creation_date="2026-02-11",
        ),
        ThreadNode(
            id="p10-s1",
            node_type="answer",
            post_id=101,
            body_text=(
                "Why naive direct methods fail: A is nr×nr, so dense factorization "
                "costs O(n³r³). The naive explicit route materializes (Z⊗K_tau) "
                "in ℝ^{N×nr}, costing O(Nnr), which is N-dependent and infeasible."
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
                "Key insight: with λ>0 and K_tau ≻ 0, PCG only needs v ↦ A v, "
                "never A explicitly. The implicit matvec costs O(n²r + qr), "
                "independent of N."
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
                "back via (Zᵀ⊗K_tau)w = vec(K_tau W'Z) where nnz(W') = s ≤ q. "
                "W'Z: O(sr), K_tau(W'Z): O(n²r). Total: O(qr + n²r)."
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
                "Regularization term: λ(Iᵣ⊗K_tau)vec(V) = λ·vec(K_tau V). "
                "Cost: O(n²r)."
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
                "Right-hand side: b = (Iᵣ⊗K_tau)vec(B) where B = TZ. "
                "T is sparse (q nonzeros), so TZ: O(qr), K_tau B: O(n²r). "
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
                "Preconditioner choice: after whitening by K_tau^{-1/2} and using "
                "SSᵀ ≈ cI (c=q/N), choose P = (H⊗K_tau) with "
                "H = c ZᵀZ + λIᵣ. This is a structured surrogate for A."
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
                "Preconditioner solve: P⁻¹ = H⁻¹⊗K_tau⁻¹. Precompute Cholesky of "
                "K_tau (O(n³)) and H (O(r³)). Each application solves "
                "K_tau Y Hᵀ = Z' via triangular substitution in O(n²r + nr²)."
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
                "Convergence: PCG has t = O(√κ log(1/ε)) with "
                "κ = cond(P^{-1/2} A P^{-1/2}). If "
                "(1-δ)P ≼ A ≼ (1+δ)P for δ in (0,1), then "
                "κ ≤ (1+δ)/(1-δ), giving a concrete fast-rate guarantee."
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
                "Complexity summary: total per mode-k subproblem is "
                "O(n³ + r³ + t(n²r + qr + nr²)); for n >= r this is "
                "O(n³ + t(n²r + qr)). Compare direct O(n³r³ + Nnr). "
                "PCG removes explicit dependence on N."
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
                "Algorithm: SETUP builds K_tau, H, and RHS; PCG then applies "
                "implicit MATVEC on observed entries and PRECOND_SOLVE via "
                "Kronecker factors each iteration."
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
            evidence="Naive explicit direct methods incur O(Nnr) memory/work",
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
            evidence="Sparse W' has s<=q nonzeros; adjoint via (Zᵀ⊗K_tau)",
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
            evidence="Whitened surrogate motivates P = (H⊗K_tau) for faster PCG",
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
            evidence="P⁻¹ = H⁻¹⊗K_tau⁻¹ via Cholesky; solve cost O(n²r + nr²)",
            detection="structural",
        ),
        # S5 asserts convergence, depends on preconditioner quality
        ThreadEdge(
            source="p10-s5", target="p10-s4",
            edge_type="assert",
            evidence="Spectral equivalence bounds κ(P⁻¹A), giving PCG rate O(√κ log(1/ε))",
            detection="structural",
        ),
        # S6 asserts the final answer, aggregating all costs
        ThreadEdge(
            source="p10-s6", target="p10-problem",
            edge_type="assert",
            evidence="O(n³ + r³ + t(n²r + qr + nr²)); no explicit N-dependence",
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
            evidence="Preconditioner solve O(n²r + nr²) per iteration",
            detection="structural",
        ),
        ThreadEdge(
            source="p10-s6", target="p10-s5",
            edge_type="reference",
            evidence="Iteration count from standard PCG bound and κ(P⁻¹A)",
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
