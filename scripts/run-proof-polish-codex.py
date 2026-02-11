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
                },
                "required": ["question_id", "title", "relevance"],
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


# Verification focus for each proof step
NODE_VERIFICATION_FOCUS = {
    "p10-problem": (
        "Verify the problem is well-posed: the system matrix is SPD when K is PSD "
        "and Z has full column rank. Confirm the dimensions are consistent."
    ),
    "p10-s1": (
        "Verify the complexity claim: direct solve of an nr×nr system costs O(n³r³). "
        "Confirm that forming (Z⊗K) explicitly requires O(Nnr) storage/operations."
    ),
    "p10-s2": (
        "Verify the CG applicability: the system matrix is SPD (required for CG). "
        "Confirm the claimed O(n²r + qr) matvec cost is achievable."
    ),
    "p10-s2a": (
        "Verify the Kronecker identity (A⊗B)vec(X) = vec(BXAᵀ). Confirm that "
        "evaluating only at observed entries via SS' costs O(n²r + qr) by the "
        "row-grouping argument."
    ),
    "p10-s2b": (
        "Verify the adjoint computation: (Zᵀ⊗K)w = vec(KW'Z) where W' is sparse. "
        "Confirm the O(qr + n²r) cost. Check that sparsity of W' is exactly q."
    ),
    "p10-s2c": (
        "Verify the regularization term: λ(Iᵣ⊗K)vec(V) = λ·vec(KV). "
        "This is straightforward but confirm the Kronecker-vec identity."
    ),
    "p10-s2-total": (
        "Verify the total matvec cost O(n²r + qr) by combining steps 2a, 2b, 2c. "
        "Confirm no hidden N-dependent terms."
    ),
    "p10-s3": (
        "Verify the RHS computation: B = TZ costs O(qr) when T is sparse with q "
        "nonzeros, and KB costs O(n²r). Confirm dimensions: T ∈ ℝ^{n×M}, Z ∈ ℝ^{M×r}."
    ),
    "p10-s4": (
        "Verify the preconditioner choice P = (H⊗K) where H = ZᵀZ + λIᵣ. "
        "Confirm it approximates A when SSᵀ ≈ I. Check that P is SPD."
    ),
    "p10-s4-hadamard": (
        "Verify the Khatri-Rao Hadamard property: ZᵀZ = ∏ᵢ(AᵢᵀAᵢ) where ∏ "
        "is elementwise product. Confirm the O(Σᵢ nᵢr²) cost. This is a key "
        "identity in tensor decomposition — find math.SE references."
    ),
    "p10-s4-solve": (
        "Verify the Kronecker inverse: (H⊗K)⁻¹ = H⁻¹⊗K⁻¹. Confirm this requires "
        "K and H both invertible (K is PSD, H = ZᵀZ + λI is PD for λ>0). "
        "Verify the Cholesky solve cost O(n²r + nr²)."
    ),
    "p10-s5": (
        "Verify the CG convergence bound t = O(√κ·log(1/ε)). Assess the claim "
        "that practical convergence is t = O(r) to O(r√(n/q)·log(1/ε)). "
        "This is the weakest part of the proof — what assumptions are needed?"
    ),
    "p10-s6": (
        "Verify the final complexity: O(n³ + t(n²r + qr)) vs direct O(n³r³ + Nnr). "
        "Confirm the reduction holds when n, r ≪ q ≪ N."
    ),
}


def build_node_prompt(node: dict, edges: list[dict], solution_text: str) -> str:
    """Build a verification prompt for a single proof node."""
    node_id = node["id"]
    focus = NODE_VERIFICATION_FOCUS.get(node_id, "Verify the mathematical claim.")

    # Find edges involving this node
    incoming = [e for e in edges if e["target"] == node_id]
    outgoing = [e for e in edges if e["source"] == node_id]

    lines = [
        "You are a mathematical proof verifier with expertise in numerical linear "
        "algebra, tensor decomposition, and RKHS methods.",
        "",
        "## Task",
        "",
        "Verify one step of a proof that PCG solves RKHS-constrained tensor CP "
        "decomposition without O(N) computation. Cross-reference with math.SE "
        "discussions when possible.",
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
        "The system is: [(Z⊗K)ᵀSSᵀ(Z⊗K) + λ(Iᵣ⊗K)]vec(W) = (Iᵣ⊗K)vec(B)",
        "where W ∈ ℝ^{n×r}, K ∈ ℝ^{n×n} PSD kernel, Z ∈ ℝ^{M×r} Khatri-Rao product,",
        "S selects q observed entries from N = nM total, B = TZ is the MTTKRP.",
        "Regime: n, r ≪ q ≪ N.",
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim in this proof step.",
        "2. Search math.SE for relevant discussions (Kronecker products, CG for "
        "structured systems, tensor decomposition, RKHS, sparse observation).",
        "3. Identify any gaps, unstated assumptions, or potential errors.",
        "4. Suggest improvements if the claim could be tightened or clarified.",
        "5. Reply as a single JSON object matching the required schema.",
    ])

    return "\n".join(lines)


def build_synthesis_prompt(solution_text: str, wiring: dict) -> str:
    """Build a synthesis prompt that reviews the entire proof."""
    stats = wiring.get("stats", {})
    return "\n".join([
        "You are a mathematical proof verifier reviewing a complete proof.",
        "",
        "## Task",
        "",
        "Review this proof that PCG solves RKHS-constrained tensor CP decomposition "
        "without O(N) computation. Assess completeness, correctness, and suggest "
        "improvements.",
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
        "1. Is the proof complete? Does the final complexity claim follow from "
        "the stated steps?",
        "2. Are there unstated assumptions (e.g., about K being well-conditioned, "
        "Z having full rank, the observation pattern)?",
        "3. The convergence bound (Section 5) is the weakest link — can it be "
        "tightened with references from math.SE?",
        "4. Search math.SE for: Kronecker product CG, tensor decomposition "
        "preconditioning, RKHS regression complexity, sparse observation matvec.",
        "5. Suggest a tighter or more elegant statement of the main result.",
        "6. Reply as a single JSON object matching the required schema. "
        "Use node_id='p10-synthesis' for the synthesis.",
    ])


def run_codex_once(
    codex_bin: str,
    model: str,
    cwd: Path,
    schema_path: Path,
    prompt_text: str,
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
    proc = subprocess.run(cmd, input=instruction, text=True, capture_output=True)
    try:
        response_text = out_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        response_text = ""
    out_path.unlink(missing_ok=True)
    return proc.returncode, response_text, proc.stderr.strip()


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

    # Load solution
    solution_text = ""
    if args.solution.exists():
        solution_text = args.solution.read_text()

    nodes = wiring["nodes"]
    edges = wiring["edges"]

    # Build prompts for each node
    prompts = []
    for node in nodes:
        prompt_text = build_node_prompt(node, edges, solution_text)
        prompts.append({
            "node_id": node["id"],
            "node_type": node["node_type"],
            "prompt": prompt_text,
        })

    # Add synthesis prompt
    prompts.append({
        "node_id": "p10-synthesis",
        "node_type": "synthesis",
        "prompt": build_synthesis_prompt(solution_text, wiring),
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
