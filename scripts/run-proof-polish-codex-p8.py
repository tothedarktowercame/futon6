#!/usr/bin/env python3
"""Polish the Problem 8 proof via Codex CLI.

For each proof node in the wiring diagram, generates a verification prompt
and runs it through `codex exec`. Codex cross-references with math.SE data
(when available), verifies mathematical claims, and identifies gaps.

Usage:
    # Generate prompts only (dry run)
    python3 scripts/run-proof-polish-codex-p8.py --dry-run

    # Run through Codex (needs math.SE data in se-data/math-processed/)
    python3 scripts/run-proof-polish-codex-p8.py --limit 10

    # With custom math.SE search path
    python3 scripts/run-proof-polish-codex-p8.py --math-se-dir se-data/math-processed/
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
WIRING_JSON = REPO_ROOT / "data" / "first-proof" / "problem8-wiring.json"
SOLUTION_MD = REPO_ROOT / "data" / "first-proof" / "problem8-solution.md"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem8-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem8-codex-prompts.jsonl"
DEFAULT_MATH_SE_DIR = REPO_ROOT / "se-data" / "math-processed"


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


# Verification focus for each proof step
NODE_VERIFICATION_FOCUS = {
    "p8-problem": (
        "Verify the problem is well-posed: a polyhedral Lagrangian surface K in "
        "R^4 = C^2 with 4 faces per vertex, asking whether K admits a Lagrangian "
        "smoothing (Hamiltonian isotopy K_t of smooth Lagrangian submanifolds for "
        "t in (0,1], extending topologically to K_0 = K). Check whether the "
        "definition of 'polyhedral Lagrangian' is standard — is every face required "
        "to be a Lagrangian 2-plane? Is the topological submanifold condition "
        "sufficient or should one require a PL submanifold? Search for prior "
        "definitions of polyhedral Lagrangian surfaces in the literature."
    ),
    "p8-s1": (
        "Verify the Lagrangian Grassmannian claim: Lambda(2) = U(2)/O(2) with "
        "pi_1(Lambda(2)) = Z (the Maslov class). This is a classical result — "
        "confirm the identification U(2)/O(2) is correct for 2-planes in R^4 = C^2, "
        "not U(n)/O(n) for some other n. Verify the dimension (should be 3). "
        "Confirm pi_1 = Z (not Z/2 or something else). Cross-reference with "
        "Arnold 1967, Maslov 1972, or standard symplectic topology references "
        "(McDuff-Salamon). Search math.SE/MO for 'Lagrangian Grassmannian "
        "fundamental group Maslov'."
    ),
    "p8-s2": (
        "Verify the edge-sharing constraint and the counting argument: 4 edges "
        "e_1, ..., e_4 in cyclic order, each face L_i = span(e_{i-1,i}, e_{i,i+1}), "
        "Lagrangian condition omega(e_{i-1,i}, e_{i,i+1}) = 0 for each i (mod 4). "
        "This gives 4 equations omega(e_i, e_{i+1}) = 0 for i=1,2,3,4 mod 4. "
        "The omega matrix on 4 vectors has C(4,2) = 6 independent entries. "
        "VERIFY THE COUNTING: 4 of 6 entries are killed, leaving exactly "
        "a = omega(e_1, e_3) and b = omega(e_2, e_4). Is this correct? "
        "Check: the killed entries are omega(e_1,e_2), omega(e_2,e_3), "
        "omega(e_3,e_4), omega(e_4,e_1) — that's 4, leaving omega(e_1,e_3) "
        "and omega(e_2,e_4). Confirm this is exactly right."
    ),
    "p8-s3": (
        "THIS IS THE KEY NEW ARGUMENT — verify with extreme care. "
        "Claim: in the reordered basis (e_1, e_3, e_2, e_4), the omega matrix is "
        "block diagonal: [[0,a,0,0],[-a,0,0,0],[0,0,0,b],[0,0,-b,0]]. "
        "Verify: (1) the reordering (e_1,e_3,e_2,e_4) puts opposite edges together; "
        "(2) omega(e_1,e_2) = 0 becomes the (1,3) entry in the NEW basis — check "
        "it IS zero; (3) omega(e_3,e_4) = 0 becomes the (2,4) entry — check; "
        "(4) omega(e_1,e_4) = omega(e_1,e_{41}) = 0 becomes the (1,4) entry — check; "
        "(5) omega(e_3,e_2) = omega(e_{34},e_{23}) = 0 becomes the (2,3) entry — check. "
        "Verify that V_1 = span(e_1, e_3) and V_2 = span(e_2, e_4) are genuinely "
        "symplectic (a ≠ 0 and b ≠ 0), and that this follows from non-degeneracy "
        "of omega on R^4 plus the 4 edges forming a basis."
    ),
    "p8-s4": (
        "Verify the Maslov index decomposition: mu = mu_1 + mu_2 where each "
        "mu_j is the winding number of the component loop in Lambda(1) of V_j. "
        "(1) Is the additivity mu = mu_1 + mu_2 under symplectic direct sum "
        "standard? Reference? (2) In V_1 = span(e_1, e_3): the loop of lines is "
        "span(e_1) -> span(e_1) -> span(e_3) -> span(e_3) -> span(e_1). This "
        "traces a back-and-forth path in RP^1, not a loop that winds — verify "
        "that the winding number in pi_1(RP^1) = Z is indeed 0. (3) Similarly "
        "for V_2. (4) Is back-and-forth always winding 0? A path that goes from "
        "a point to another and back is contractible in RP^1, hence winding 0. "
        "Verify this topological reasoning."
    ),
    "p8-s5": (
        "Verify the Lagrangian surgery reference: Polterovich 1991 'The surgery of "
        "Lagrange submanifolds'. (1) Does Polterovich actually prove that a transverse "
        "Lagrangian crossing with Maslov index 0 can be resolved by surgery? Or is "
        "the result from Lalonde-Sikorav 1991? (2) Is 'Maslov index 0' the correct "
        "hypothesis, or is it 'vanishing local Maslov index at the crossing'? "
        "(3) The neck construction via generating function y = grad S(x): is this "
        "well-defined in the V_1 ⊕ V_2 adapted Darboux coordinates? (4) Does the "
        "surgery preserve the Lagrangian condition? (5) Is the result stated for "
        "surfaces (dim 2) or in general dimension? Search math.SE/MO for "
        "'Lagrangian surgery Maslov index Polterovich'."
    ),
    "p8-s6": (
        "Verify the global smoothing argument: vertex surgery + edge smoothing "
        "compose via Hamiltonian isotopy to give K_t. (1) Is composition of "
        "Hamiltonian isotopies standard? (Yes — the composition of flows of "
        "compactly supported Hamiltonians is again Hamiltonian.) Verify this. "
        "(2) Do the local surgeries at vertices interfere with each other? The "
        "proof assumes they can be done independently in disjoint neighborhoods — "
        "is this justified? (3) Edge smoothing via generating function "
        "interpolation: is this standard in symplectic topology? (4) Does the "
        "Hamiltonian isotopy extend to t=0 as a topological isotopy? This "
        "requires the surgery neck to collapse continuously — verify."
    ),
    "p8-s7": (
        "Verify the 3-face impossibility argument: 3 isotropic vectors can't span "
        "a 3-dimensional subspace in R^4. (1) For 3 faces, the 3 edge vectors "
        "satisfy omega(e_i, e_j) = 0 for ALL pairs (since all pairs are adjacent "
        "in a 3-cycle). This makes span(e_1, e_2, e_3) isotropic. (2) In (R^4, omega), "
        "the maximum isotropic dimension is n = 2 (half of 2n = 4). (3) Therefore "
        "3 independent isotropic vectors cannot exist — the span has dimension at "
        "most 2. (4) But we need 3 independent edges for a non-degenerate 3-face "
        "vertex, contradiction. Verify each step. Is the bound dim(isotropic) <= n "
        "standard and correctly applied here?"
    ),
    "p8-c1": (
        "Evaluate the numerical evidence: 998/998 valid 4-valent configurations "
        "give Maslov index exactly 0. (1) Is this evidence or proof? It's evidence "
        "supporting the algebraic proof, not a substitute. (2) The comparison: "
        "without edge-sharing, only 55% give mu = 0 — does this convincingly show "
        "the edge-sharing constraint is essential? (3) How are 'valid 4-valent "
        "configurations' generated? Random vectors in R^4 satisfying the omega "
        "constraints? (4) Is 998 a large enough sample? (5) Could there be a "
        "measure-zero counterexample that random sampling misses? Note: the "
        "algebraic proof (Section 3-4) already covers all cases, so the numerics "
        "are confirmatory, not primary."
    ),
}


def build_node_prompt(
    node: dict,
    edges: list[dict],
    solution_text: str,
    math_se_dir: Path | None = None,
) -> str:
    """Build a verification prompt for a single proof node."""
    node_id = node["id"]
    focus = NODE_VERIFICATION_FOCUS.get(node_id, "Verify the mathematical claim.")

    # Find edges involving this node
    incoming = [e for e in edges if e["target"] == node_id]
    outgoing = [e for e in edges if e["source"] == node_id]

    lines = [
        "You are a mathematical proof verifier with expertise in "
        "symplectic geometry, Lagrangian submanifolds, Maslov index, "
        "Lagrangian surgery, polyhedral geometry in R^4.",
        "",
        "## Task",
        "",
        "Verify one step of a proof that a polyhedral Lagrangian surface "
        "in R^4 with 4 faces per vertex admits a Lagrangian smoothing. "
        "The proof uses a symplectic direct sum decomposition at each vertex "
        "to show the Maslov index vanishes, enabling Lagrangian surgery "
        "(Polterovich 1991). Cross-reference with math.SE/MO discussions "
        "and primary sources when possible.",
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
            lines.append(f"  - {e['source']} -> {node_id} [{e['edge_type']}]: {e['evidence']}")
        lines.append("")

    if outgoing:
        lines.append("## Outgoing Edges (this step supports)")
        for e in outgoing:
            lines.append(f"  - {node_id} -> {e['target']} [{e['edge_type']}]: {e['evidence']}")
        lines.append("")

    lines.extend([
        "## Full Problem Context",
        "",
        "Problem 8 asks: Given a polyhedral Lagrangian surface K in R^4 with "
        "exactly 4 faces meeting at every vertex, does K necessarily have a "
        "Lagrangian smoothing (a Hamiltonian isotopy K_t of smooth Lagrangian "
        "submanifolds for t in (0,1], with K_0 = K topologically)?",
        "",
        "The answer is: Yes. The key insight is a symplectic direct sum "
        "decomposition at each 4-valent vertex. The 4 edge-sharing constraints "
        "omega(e_i, e_{i+1}) = 0 force the omega matrix to be block diagonal in "
        "the basis (e_1, e_3, e_2, e_4), giving R^4 = V_1 + V_2 with V_1 = "
        "span(e_1, e_3), V_2 = span(e_2, e_4). Each Lagrangian face decomposes "
        "as a line in V_1 direct sum a line in V_2. The Maslov index of the "
        "vertex loop decomposes as mu = mu_1 + mu_2 = 0 + 0 = 0. Zero Maslov "
        "enables Lagrangian surgery (Polterovich 1991).",
        "",
        "Key references: Polterovich 1991 (Lagrangian surgery), "
        "Lalonde-Sikorav 1991, Arnold 1967 / Maslov 1972 (Maslov index), "
        "McDuff-Salamon (symplectic topology).",
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim in this proof step.",
        "2. Search math.SE/MO for relevant discussions (Lagrangian Grassmannian, "
        "Maslov index, Lagrangian surgery, polyhedral Lagrangian, symplectic "
        "direct sum, isotropic subspaces, generating functions).",
        "3. Identify any gaps, unstated assumptions, or potential errors.",
        "4. Suggest improvements if the claim could be tightened or clarified.",
        "5. For each reference, include `site` when known "
        "(e.g., `mathoverflow.net` or `math.stackexchange.com`).",
        "6. Reply as a single JSON object matching the required schema.",
    ])
    if math_se_dir:
        lines.extend([
            "",
            "## Local Corpus Hint",
            "",
            f"Use local processed data if available under: {math_se_dir}",
        ])

    return "\n".join(lines)


def build_synthesis_prompt(
    solution_text: str,
    wiring: dict,
    math_se_dir: Path | None = None,
) -> str:
    """Build a synthesis prompt that reviews the entire proof."""
    stats = wiring.get("stats", {})
    lines = [
        "You are a mathematical proof verifier reviewing a complete proof.",
        "",
        "## Task",
        "",
        "Review this proof that a polyhedral Lagrangian surface in R^4 with "
        "4 faces per vertex admits a Lagrangian smoothing. The proof proceeds "
        "via symplectic direct sum decomposition at each vertex, vanishing "
        "Maslov index, and Lagrangian surgery (Polterovich 1991). Assess "
        "completeness, correctness, and suggest improvements.",
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
        "1. Is the proof complete? Does the Lagrangian smoothing follow from "
        "the stated steps?",
        "2. Are there unstated assumptions (e.g., genericity of edge vectors, "
        "compactness of K, orientability, smoothness of the faces away from "
        "edges/vertices, the PL vs topological submanifold distinction)?",
        "3. The symplectic direct sum decomposition (Section 3) is the key new "
        "argument — is the block diagonalization of omega in basis (e_1,e_3,e_2,e_4) "
        "correct? Does non-degeneracy of omega truly force a ≠ 0 and b ≠ 0?",
        "4. The Maslov index computation (Section 4) — is the additivity under "
        "symplectic direct sum standard? Is the back-and-forth path argument "
        "for winding number 0 rigorous?",
        "5. The Lagrangian surgery step (Section 6) — does Polterovich 1991 apply "
        "directly, or does the polyhedral (non-smooth) setting require adaptation?",
        "6. The global composition (Section 7) — do the local surgeries at "
        "different vertices compose without interference? Is the Hamiltonian "
        "isotopy argument complete?",
        "7. The 3-face impossibility (Section 5) — is the isotropic dimension "
        "bound correctly applied?",
        "8. Search math.SE/MO for: Lagrangian Grassmannian Maslov index, "
        "Lagrangian surgery polyhedral, symplectic direct sum decomposition, "
        "polyhedral Lagrangian smoothing, isotropic subspace dimension bound.",
        "9. Suggest a tighter or more elegant statement of the main result.",
        "10. For each reference, include `site` when known "
        "(e.g., `mathoverflow.net` or `math.stackexchange.com`).",
        "11. Reply as a single JSON object matching the required schema. "
        "Use node_id='p8-synthesis' for the synthesis.",
    ]
    if math_se_dir:
        lines.extend([
            "",
            "## Local Corpus Hint",
            "",
            f"Use local processed data if available under: {math_se_dir}",
        ])
    return "\n".join(lines)


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
    ap.add_argument("--limit", type=int, default=None,
                    help="Max prompts to process (default: all generated prompts)")
    ap.add_argument("--model", default="gpt-5.3-codex")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate prompts only, don't call Codex")
    ap.add_argument("--math-se-dir", type=Path, default=DEFAULT_MATH_SE_DIR,
                    help="Local processed StackExchange data directory (hinted to Codex prompts)")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args()

    # Load wiring diagram
    if not args.wiring.exists():
        print(f"Wiring diagram not found: {args.wiring}", file=sys.stderr)
        print("Run: python3 scripts/proof8-wiring-diagram.py", file=sys.stderr)
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
        prompt_text = build_node_prompt(
            node,
            edges,
            solution_text,
            math_se_dir=args.math_se_dir,
        )
        prompts.append({
            "node_id": node["id"],
            "node_type": node["node_type"],
            "prompt": prompt_text,
        })

    # Add synthesis prompt
    prompts.append({
        "node_id": "p8-synthesis",
        "node_type": "synthesis",
        "prompt": build_synthesis_prompt(solution_text, wiring, math_se_dir=args.math_se_dir),
    })
    run_limit = len(prompts) if args.limit is None else min(args.limit, len(prompts))

    # Write prompts
    args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.prompts_out, "w") as f:
        for rec in prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} prompts to {args.prompts_out}")

    if args.dry_run:
        print("Dry run -- not calling Codex.")
        print(f"\nPrompt summary ({len(prompts)} prompts, run_limit={run_limit}):")
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
            for rec in prompts[:run_limit]:
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
                print(f"[{processed:02d}/{run_limit}] {rec['node_id']:20s} -> {status}")
                sys.stdout.flush()
    finally:
        schema_path.unlink(missing_ok=True)

    print("\n---SUMMARY---")
    print(f"processed={processed}")
    print(f"verified={verified['verified']}, plausible={verified['plausible']}, "
          f"gap={verified['gap']}, error={verified['error']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
