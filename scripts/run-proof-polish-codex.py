#!/usr/bin/env python3
"""Polish the Problem 10 proof via Codex CLI.

For each proof node in the wiring diagram, generates a verification prompt
and runs it through `codex exec`. Codex cross-references with math.SE data
(when available), verifies mathematical claims, and identifies gaps.

Usage:
    # Generate prompts only (dry run)
    python3 scripts/run-proof-polish-codex.py --dry-run

    # Run through Codex (needs math.SE data in se-data/math-processed/)
    python3 scripts/run-proof-polish-codex.py --limit 14

    # With custom math.SE search path
    python3 scripts/run-proof-polish-codex.py --math-se-dir se-data/math-processed/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WIRING_JSON = REPO_ROOT / "data" / "first-proof" / "problem10-wiring.json"
SOLUTION_MD = REPO_ROOT / "data" / "first-proof" / "problem10-solution.md"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem10-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem10-codex-prompts.jsonl"


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "node_id": {"type": "string"},
        "claim_verified": {
            "type": "string",
            "enum": ["verified", "plausible", "gap", "error"],
        },
        "verification_notes": {"type": "string"},
        "math_se_references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "relevance": {"type": "string"},
                    "site": {
                        "type": "string",
                        "enum": ["math.stackexchange.com", "mathoverflow.net", "other", "unknown"],
                    },
                },
                "required": ["question_id", "title", "relevance", "site"],
                "additionalProperties": False,
            },
        },
        "suggested_improvement": {"type": "string"},
        "missing_assumptions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": [
        "node_id",
        "claim_verified",
        "verification_notes",
        "math_se_references",
        "suggested_improvement",
        "missing_assumptions",
        "confidence",
    ],
    "additionalProperties": False,
}


# Verification focus for each proof step (Problem 10 profile)
P10_NODE_VERIFICATION_FOCUS = {
    "p10-problem": (
        "Verify the problem is well-posed with explicit assumptions: lambda > 0 and "
        "K_tau = K + tau I is PD (or solve restricted to Range(K)). Confirm "
        "dimensions are consistent and SPD conditions are stated correctly."
    ),
    "p10-s1": (
        "Verify the complexity claim for the naive explicit route: direct solve of "
        "an nr×nr dense system costs O(n³r³), and explicitly forming (Z⊗K_tau) "
        "costs O(Nnr) storage/operations."
    ),
    "p10-s2": (
        "Verify PCG applicability under the stated assumptions (lambda>0, K_tau PD). "
        "Confirm the claimed O(n²r + qr) implicit matvec is achievable without N-sized "
        "materialization."
    ),
    "p10-s2a": (
        "Verify the Kronecker identity (A⊗B)vec(X) = vec(BXAᵀ). Confirm that "
        "evaluating only at observed entries via SS' costs O(n²r + qr) by the "
        "row-grouping argument."
    ),
    "p10-s2b": (
        "Verify the adjoint computation: (Zᵀ⊗K_tau)w = vec(K_tau W'Z) where W' is sparse. "
        "Confirm O(qr + n²r) cost and check sparsity is s = nnz(W') <= q (not necessarily exactly q)."
    ),
    "p10-s2c": (
        "Verify the regularization term: λ(Iᵣ⊗K_tau)vec(V) = λ·vec(K_tau V). "
        "This is straightforward but confirm the Kronecker-vec identity."
    ),
    "p10-s2-total": (
        "Verify the total matvec cost O(n²r + qr) by combining steps 2a, 2b, 2c. "
        "Confirm no hidden N-dependent terms."
    ),
    "p10-s3": (
        "Verify the RHS computation: B = TZ costs O(qr) when T is sparse with q "
        "nonzeros, and K_tau B costs O(n²r). Confirm dimensions: T ∈ ℝ^{n×M}, Z ∈ ℝ^{M×r}."
    ),
    "p10-s4": (
        "Verify the preconditioner derivation via kernel whitening and SSᵀ ≈ cI: "
        "P = (H⊗K_tau), H = c ZᵀZ + λIᵣ. Check the algebraic consistency and "
        "the SPD conditions."
    ),
    "p10-s4-hadamard": (
        "Verify the Khatri-Rao Hadamard property: ZᵀZ = ∏ᵢ(AᵢᵀAᵢ) where ∏ "
        "is elementwise product. Confirm the O(Σᵢ nᵢr²) cost. This is a key "
        "identity in tensor decomposition — find math.SE references."
    ),
    "p10-s4-solve": (
        "Verify the Kronecker inverse: (H⊗K_tau)⁻¹ = H⁻¹⊗K_tau⁻¹. Confirm this "
        "requires K_tau and H invertible (PD). Verify Cholesky-based apply cost "
        "O(n²r + nr²)."
    ),
    "p10-s5": (
        "Verify the standard PCG convergence bound t = O(√κ·log(1/ε)) and the "
        "spectral-equivalence implication: if (1-δ)P ≼ A ≼ (1+δ)P, then "
        "κ(P⁻¹A) ≤ (1+δ)/(1-δ). Check assumptions needed for this."
    ),
    "p10-s6": (
        "Verify the final complexity: O(n³ + r³ + t(n²r + qr + nr²)) "
        "(or simplified O(n³ + t(n²r + qr)) when n≥r) vs direct O(n³r³ + Nnr). "
        "Confirm the reduction holds when n, r ≪ q ≪ N."
    ),
}


# Verification focus for each proof step (Problem 6 profile)
P6_NODE_VERIFICATION_FOCUS = {
    "p6-problem": (
        "Verify the statement is mathematically well-posed: define epsilon-light "
        "subset precisely in Laplacian PSD order and confirm the existential "
        "quantifiers over all graphs and epsilon in (0,1)."
    ),
    "p6-s1": (
        "Verify the Laplacian reformulation and effective-resistance framing. "
        "Check whether ||L^{+/2} L_S L^{+/2}|| <= epsilon is equivalent to "
        "epsilon L - L_S >= 0 on the appropriate subspace."
    ),
    "p6-s2": (
        "Verify the K_n tight example: eigenvalues of induced K_s Laplacian and "
        "the reduction to s <= epsilon n. Confirm that this yields c <= 1 upper bound."
    ),
    "p6-s3": (
        "Verify random vertex sampling with p=epsilon: E[|S|]=epsilon n and "
        "E[L_S]=epsilon^2 L. Check that this only proves expectation-level control."
    ),
    "p6-s3a": (
        "Verify the Chernoff concentration constants in |S| >= epsilon n / 2 and "
        "the stated failure probability."
    ),
    "p6-s3b": (
        "Verify E[L_S]=epsilon^2 L and the comparison to epsilon L. Check wording "
        "around multiplicative gap and whether it is stated correctly."
    ),
    "p6-s4": (
        "Verify the core concentration strategy: why expectation is insufficient, "
        "and how star domination plus matrix concentration yields high-probability "
        "operator inequality."
    ),
    "p6-s4a": (
        "Verify star domination algebraically (edge indicators to vertex indicators) "
        "and confirm that the resulting matrix sum has independent summands."
    ),
    "p6-s4b": (
        "Verify Matrix Freedman/Bernstein setup: martingale definition, "
        "difference bounds, and explicit predictable-variation control."
    ),
    "p6-s5": (
        "Verify this step is explicitly marked as an external dependency "
        "(not re-proved in-text) and that no stronger claim is asserted."
    ),
    "p6-s6": (
        "Verify the final logic is conditional: unconditional results in-text plus "
        "the existential YES answer only under the external theorem assumption."
    ),
}

P3_NODE_VERIFICATION_FOCUS = {
    "p3-problem": (
        "Verify the existence claim scope: nontrivial CTMC on S_n(lambda) with "
        "stationary ratio F*_mu/P*_lambda at q=1."
    ),
    "p3-s1": (
        "Verify the chain construction is explicit: ring dynamics, site rates 1/x_i, "
        "and t-geometric weaker-particle choice."
    ),
    "p3-s2": (
        "Verify finite-state CTMC validity from the explicit generator definition "
        "q(eta,eta') and diagonal convention q(eta,eta)=-sum_{eta'!=eta}q(eta,eta'). "
        "Check finite exit rates and vacancy interpretation from lambda_n=0."
    ),
    "p3-s3": (
        "Verify nontriviality means transition rates depend only on (x,t) and current "
        "configuration dynamics, not on values of F*_mu or P*_lambda."
    ),
    "p3-s4": (
        "Verify AMW Theorem 1.1 is applied with matching hypotheses/domain "
        "(ring model, q=1, parameter range) to obtain stationary ratio F/P."
    ),
    "p3-s5": (
        "Verify the notation convention/bridge is explicit and logically sufficient: "
        "either F*_eta := F_eta by definition in this writeup, or an eta-independent "
        "normalization factor cancels in F*/P*."
    ),
    "p3-s6": (
        "Verify n=2 sanity calculation: two-state CTMC rates 1/x_1 and 1/x_2 produce "
        "stationary ratio x_1:x_2."
    ),
    "p3-s7": (
        "Verify the final composition for the existence scope only: from explicit "
        "nontrivial chain construction plus AMW stationary-law theorem plus notation "
        "convention, conclude existence of a CTMC with stationary ratio F*_mu/P*_lambda."
        " Treat uniqueness/irreducibility as optional and out-of-scope unless explicitly claimed."
    ),
}


PROOF_PROFILES = {
    "first-proof-p10": {
        "role": (
            "You are a mathematical proof verifier with expertise in numerical linear "
            "algebra, tensor decomposition, and RKHS methods."
        ),
        "task": (
            "Verify one step of a proof that PCG solves RKHS-constrained tensor CP "
            "decomposition without O(N) computation."
        ),
        "search_topics": (
            "Kronecker products, CG/PCG for structured systems, tensor decomposition, "
            "RKHS kernels, sparse observation operators."
        ),
        "problem_context": [
            "The system is: [(Z⊗K_tau)^T S S^T (Z⊗K_tau) + lambda(I_r⊗K_tau)]vec(W) = (I_r⊗K_tau)vec(B)",
            "where W in R^{n x r}, K_tau = K + tau I in R^{n x n} (PD for tau>0), Z in R^{M x r},",
            "S selects q observed entries from N = nM, and B = T Z.",
            "Regime: n, r << q << N.",
        ],
        "node_focus": P10_NODE_VERIFICATION_FOCUS,
        "synthesis_node_id": "p10-synthesis",
        "synthesis_points": [
            "Is the proof complete, and does the final complexity claim follow?",
            "Are SPD assumptions, preconditioner derivation, and convergence assumptions explicit?",
            "Which steps are still weakest, and how can they be tightened?",
        ],
    },
    "first-proof-p6": {
        "role": (
            "You are a mathematical proof verifier with expertise in spectral graph "
            "theory, Laplacians, and matrix concentration inequalities."
        ),
        "task": (
            "Verify one step of a proof about epsilon-light vertex subsets in graphs "
            "using Laplacian PSD inequalities."
        ),
        "search_topics": (
            "Graph Laplacians, effective resistance, Matrix Chernoff/Freedman, "
            "independent Bernoulli matrix sums, induced subgraph spectra."
        ),
        "problem_context": [
            "Goal: for every graph G=(V,E) and epsilon in (0,1), find S subseteq V with |S| >= c*epsilon*|V|",
            "such that S is epsilon-light, i.e. epsilon*L - L_S is PSD (L graph Laplacian, L_S induced-subgraph Laplacian).",
            "Proof strategy uses random vertex sampling, star domination, and matrix concentration.",
        ],
        "node_focus": P6_NODE_VERIFICATION_FOCUS,
        "synthesis_node_id": "p6-synthesis",
        "synthesis_points": [
            "Is the conditional status of the existential conclusion stated clearly?",
            "Are concentration assumptions and martingale parameters explicit and valid?",
            "Are any remaining claims stronger than what is actually proved in-text?",
        ],
    },
    "first-proof-p3": {
        "role": (
            "You are a mathematical proof verifier with expertise in Markov chains, "
            "interacting particle systems, and ASEP/Macdonald polynomial interfaces."
        ),
        "task": (
            "Verify one step of a proof that an explicit nontrivial CTMC has "
            "stationary ratio F*_mu/P*_lambda at q=1."
        ),
        "search_topics": (
            "t-PushTASEP, finite-state CTMC irreducibility, stationary distributions, "
            "ASEP/Macdonald polynomial notation conventions."
        ),
        "problem_context": [
            "State space: S_n(lambda), permutations of a restricted partition lambda with distinct parts and lambda_n=0.",
            "Candidate dynamics: inhomogeneous multispecies t-PushTASEP on an n-site ring with rates 1/x_i and t-geometric weaker-particle choice.",
            "Target: existence of nontrivial chain with stationary distribution proportional to q=1 ASEP polynomial weights.",
        ],
        "node_focus": P3_NODE_VERIFICATION_FOCUS,
        "synthesis_node_id": "p3-synthesis",
        "synthesis_points": [
            "Is the proof complete for existence and nontriviality of the chain?",
            "Are assumptions and notation conventions explicit and sufficient?",
            "Are remaining weaknesses theorem-level or validation/citation-level for the existence claim?",
        ],
    },
}


def infer_profile(wiring: dict) -> dict:
    """Infer prompt profile from wiring thread_id."""
    thread_id = wiring.get("thread_id", "")
    if thread_id in PROOF_PROFILES:
        return PROOF_PROFILES[thread_id]
    return {
        "role": (
            "You are a mathematical proof verifier. Check correctness, assumptions, "
            "and logical completeness."
        ),
        "task": "Verify one step of the supplied proof.",
        "search_topics": "Relevant mathematical references on Math StackExchange.",
        "problem_context": ["Use the full proof text and local claim context below."],
        "node_focus": {},
        "synthesis_node_id": "proof-synthesis",
        "synthesis_points": [
            "Is the proof complete?",
            "What assumptions are missing?",
            "How should the argument be tightened?",
        ],
    }


def build_node_prompt(
    node: dict,
    edges: list[dict],
    solution_text: str,
    profile: dict,
    require_math_se_search: bool = True,
) -> str:
    """Build a verification prompt for a single proof node."""
    node_id = node["id"]
    focus_map = profile.get("node_focus", {})
    focus = focus_map.get(node_id, "Verify the mathematical claim.")

    # Find edges involving this node
    incoming = [e for e in edges if e["target"] == node_id]
    outgoing = [e for e in edges if e["source"] == node_id]

    task_line = profile["task"]
    if require_math_se_search:
        task_line += " Cross-reference with relevant math.SE discussions when possible."
    else:
        task_line += " Use only provided local context; do not perform web search."

    lines = [
        profile["role"],
        "",
        "## Task",
        "",
        task_line,
        "",
        "## Proof Step Under Review",
        "",
        f"**Node ID**: {node_id}",
        f"**Type**: {node['node_type']}",
        f"**Claim**: {node['body_text']}",
        "",
        "## Verification Focus",
        "",
        focus,
        "",
    ]

    if incoming:
        lines.append("## Incoming Edges (this step depends on)")
        for e in incoming:
            lines.append(f"  - {e['source']} → {node_id} [{e['edge_type']}]: {e['evidence']}")
        lines.append("")

    if outgoing:
        lines.append("## Outgoing Edges (this step supports)")
        for e in outgoing:
            lines.append(f"  - {node_id} → {e['target']} [{e['edge_type']}]: {e['evidence']}")
        lines.append("")

    lines.extend([
        "## Full Problem Context",
        "",
        *profile["problem_context"],
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim in this proof step.",
    ])
    if require_math_se_search:
        lines.extend([
            f"2. Search math.SE for relevant discussions ({profile['search_topics']}).",
            "3. Identify any gaps, unstated assumptions, or potential errors.",
            "4. Suggest improvements if the claim could be tightened or clarified.",
            "5. Reply as a single JSON object matching the required schema.",
        ])
    else:
        lines.extend([
            "2. Use only the provided local context to identify gaps, unstated assumptions, or potential errors.",
            "3. Suggest improvements if the claim could be tightened or clarified.",
            "4. Reply as a single JSON object matching the required schema.",
        ])

    return "\n".join(lines)


def build_synthesis_prompt(
    solution_text: str,
    wiring: dict,
    profile: dict,
    require_math_se_search: bool = True,
) -> str:
    """Build a synthesis prompt that reviews the entire proof."""
    stats = wiring.get("stats", {})
    synthesis_points = profile.get("synthesis_points", [])
    instructions = []
    for i, point in enumerate(synthesis_points, start=1):
        instructions.append(f"{i}. {point}")
    if require_math_se_search:
        instructions.extend([
            f"{len(instructions)+1}. Search math.SE for references relevant to this proof domain.",
            f"{len(instructions)+1}. Reply as a single JSON object matching the required schema. "
            f"Use node_id='{profile['synthesis_node_id']}' for the synthesis.",
        ])
    else:
        instructions.extend([
            f"{len(instructions)+1}. Use only the provided local proof text and wiring context (no web search).",
            f"{len(instructions)+1}. Reply as a single JSON object matching the required schema. "
            f"Use node_id='{profile['synthesis_node_id']}' for the synthesis.",
        ])
    return "\n".join([
        "You are a mathematical proof verifier reviewing a complete proof.",
        "",
        "## Task",
        "",
        profile["task"] + " Assess completeness, correctness, and suggest improvements.",
        "",
        "## Proof",
        "",
        solution_text,
        "",
        "## Wiring Diagram Summary",
        "",
        f"Nodes: {stats.get('n_nodes', '?')}, Edges: {stats.get('n_edges', '?')}",
        f"Edge types: {json.dumps(stats.get('edge_types', {}))}",
        "",
        "## Instructions",
        "",
        *instructions,
    ])


def run_codex_once(
    codex_bin: str,
    model: str,
    cwd: Path,
    schema_path: Path,
    prompt_text: str,
    timeout_sec: int | None = None,
) -> tuple[int, str, str]:
    """Run a single prompt through codex exec."""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as out_f:
        out_path = Path(out_f.name)

    instruction = (
        "You must answer exactly as one JSON object matching the required schema. "
        "Do not wrap JSON in markdown fences. Do not add extra commentary.\n\n"
        + prompt_text
    )
    cmd = [
        codex_bin,
        "exec",
        "--cd", str(cwd),
        "--sandbox", "workspace-write",
        "--model", model,
        "--output-schema", str(schema_path),
        "--output-last-message", str(out_path),
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=instruction,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        rc = proc.returncode
        stderr_text = proc.stderr.strip()
    except subprocess.TimeoutExpired as e:
        rc = 124
        stderr_text = f"timeout after {timeout_sec}s"
        if e.stderr:
            stderr_text = f"{stderr_text}\n{e.stderr}"
    try:
        response_text = out_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        response_text = ""
    out_path.unlink(missing_ok=True)
    return rc, response_text, stderr_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiring", type=Path, default=WIRING_JSON)
    ap.add_argument("--solution", type=Path, default=SOLUTION_MD)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--prompts-out", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--limit", type=int, default=15,
                    help="Max nodes to process (14 nodes + 1 synthesis = 15)")
    ap.add_argument("--model", default="gpt-5.3-codex")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument(
        "--skip-math-se-search",
        action="store_true",
        help="Do not request math.SE/web lookups; keep verification local-context only.",
    )
    ap.add_argument(
        "--per-call-timeout-sec",
        type=int,
        default=120,
        help="Timeout in seconds per codex invocation (prevents indefinite hangs).",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate prompts only, don't call Codex")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args()

    # Load wiring diagram
    if not args.wiring.exists():
        print(f"Wiring diagram not found: {args.wiring}", file=sys.stderr)
        print("Run: python3 scripts/proof-wiring-diagram.py", file=sys.stderr)
        return 2
    wiring = json.loads(args.wiring.read_text())
    profile = infer_profile(wiring)

    # Load solution
    solution_text = ""
    if args.solution.exists():
        solution_text = args.solution.read_text()

    nodes = wiring["nodes"]
    edges = wiring["edges"]

    # Build prompts for each node
    prompts = []
    for node in nodes:
        prompt_text = build_node_prompt(
            node=node,
            edges=edges,
            solution_text=solution_text,
            profile=profile,
            require_math_se_search=not args.skip_math_se_search,
        )
        prompts.append({
            "node_id": node["id"],
            "node_type": node["node_type"],
            "prompt": prompt_text,
        })

    # Add synthesis prompt
    prompts.append({
        "node_id": profile["synthesis_node_id"],
        "node_type": "synthesis",
        "prompt": build_synthesis_prompt(
            solution_text,
            wiring,
            profile,
            require_math_se_search=not args.skip_math_se_search,
        ),
    })

    # Write prompts
    args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.prompts_out, "w") as f:
        for rec in prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} prompts to {args.prompts_out}")

    if args.dry_run:
        print("Dry run — not calling Codex.")
        print(f"\nPrompt summary ({len(prompts)} prompts):")
        for p in prompts:
            lines = p["prompt"].count("\n") + 1
            print(f"  {p['node_id']:20s} [{p['node_type']:10s}] ~{lines} lines")
        return 0

    # Run through Codex
    args.output.parent.mkdir(parents=True, exist_ok=True)
    verified = Counter()
    processed = 0

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(RESPONSE_SCHEMA, sf, ensure_ascii=True, indent=2)
        schema_path = Path(sf.name)

    try:
        with open(args.output, "w") as fout:
            for rec in prompts[:args.limit]:
                rc, raw_response, stderr_text = run_codex_once(
                    codex_bin=args.codex_bin,
                    model=args.model,
                    cwd=args.repo_root,
                    schema_path=schema_path,
                    prompt_text=rec["prompt"],
                    timeout_sec=args.per_call_timeout_sec,
                )

                out = {"node_id": rec["node_id"]}
                try:
                    parsed = json.loads(raw_response)
                    if isinstance(parsed, dict):
                        out.update(parsed)
                        v = parsed.get("claim_verified", "")
                        if v in ("verified", "plausible", "gap", "error"):
                            verified[v] += 1
                    else:
                        out["parse_error"] = True
                        out["raw"] = raw_response
                except Exception:
                    out["parse_error"] = True
                    parts = []
                    if raw_response:
                        parts.append(raw_response)
                    if rc != 0:
                        parts.append(f"[codex_exit_code={rc}]")
                    if stderr_text:
                        parts.append(f"[stderr]\n{stderr_text}")
                    out["raw"] = "\n".join(parts).strip()

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1
                status = out.get("claim_verified", "parse_error" if out.get("parse_error") else "?")
                print(f"[{processed:02d}/{len(prompts)}] {rec['node_id']:20s} → {status}")
                sys.stdout.flush()
    finally:
        schema_path.unlink(missing_ok=True)

    print("\n---SUMMARY---")
    print(f"processed={processed}")
    print(f"verified={verified['verified']}, plausible={verified['plausible']}, "
          f"gap={verified['gap']}, error={verified['error']}")
    total_refs = 0  # would need to count from results
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
