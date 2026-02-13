#!/usr/bin/env python3
"""Polish the Problem 9 proof via Codex CLI.

For each proof node in the wiring diagram, generates a verification prompt
and runs it through `codex exec`. Codex cross-references with math.SE data
(when available), verifies mathematical claims, and identifies gaps.

Usage:
    # Generate prompts only (dry run)
    python3 scripts/run-proof-polish-codex-p9.py --dry-run

    # Run through Codex (needs math.SE data in se-data/math-processed/)
    python3 scripts/run-proof-polish-codex-p9.py --limit 12

    # With custom math.SE search path
    python3 scripts/run-proof-polish-codex-p9.py --math-se-dir se-data/math-processed/
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
WIRING_JSON = REPO_ROOT / "data" / "first-proof" / "problem9-wiring.json"
SOLUTION_MD = REPO_ROOT / "data" / "first-proof" / "problem9-solution.md"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem9-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem9-codex-prompts.jsonl"
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
    "p9-problem": (
        "Verify the problem is well-posed: check that Q^(abgd)_{ijkl} = "
        "det[rows] is correctly defined as the 4x4 determinant of stacked "
        "camera rows from n >= 5 generic 3x4 matrices. Confirm that 'rank-1 "
        "scaling' lambda = u otimes v otimes w otimes x is the correct notion "
        "for 4-tensors (Segre variety). Check that the polynomial map F is "
        "required to be camera-independent and of bounded degree. Verify the "
        "non-identical index condition lambda_{abgd} != 0 iff a,b,g,d not all "
        "identical is correctly stated."
    ),
    "p9-s1": (
        "Verify the bilinear form reduction: fixing (gamma,k) and (delta,l), "
        "the map omega(p,q) = det[p;q;c;d] is alternating bilinear on R^4 "
        "with rank 2. Confirm from Hodge theory: c wedge d is a simple 2-form "
        "in Lambda^2(R^4), its Hodge dual *(c wedge d) is also simple, and "
        "the associated bilinear form has rank exactly 2 (not 0 or 4). Verify "
        "that the genericity of cameras ensures c and d are linearly "
        "independent (so c wedge d != 0). Cross-reference with exterior "
        "algebra / Grassmannian literature."
    ),
    "p9-s1a": (
        "Verify the null space characterization: the null space of "
        "omega(p,q) = det[p;q;c;d] is exactly span{c,d}. Check: if p in "
        "span{c,d} then det[p;q;c;d] = 0 for all q (row dependence), and "
        "conversely if p not in span{c,d} then there exists q with "
        "det[p;q;c;d] != 0. Confirm the quotient V/span{c,d} is isomorphic "
        "to R^2 and omega induces a non-degenerate alternating form there."
    ),
    "p9-s2": (
        "Verify the rank-2 to 3x3 minor connection: a bilinear form of rank 2 "
        "has ALL 3x3 minors vanishing. This is the standard linear algebra "
        "fact: rank(Omega) = 2 means any 3x3 submatrix of the Gram matrix "
        "Omega_{mn} = omega(p_m, q_n) has rank <= 2, hence determinant 0. "
        "Verify this is degree-3 in the Q entries and camera-independent "
        "(the polynomial only involves T entries, not A^(i) entries directly). "
        "Search for: 'rank of bilinear form determinantal variety' on math.SE/MO."
    ),
    "p9-s3": (
        "Verify that rank-1 scaling preserves the minor vanishing. The key "
        "claim: if lambda = u otimes v otimes w otimes x, then Lambda_{mn} = "
        "u_{alpha_m} v_{beta_n} w_gamma x_delta has matrix rank 1 as a 3x3 "
        "matrix (outer product of (u_{alpha_m}) and (v_{beta_n}) times a "
        "scalar). Then M = Lambda circ Omega = diag(u) * Omega * diag(v) "
        "times a constant, so rank(M) = rank(Omega) = 2 < 3. Verify: does "
        "Hadamard product with a rank-1 matrix equal diagonal scaling? Is "
        "the formula M = diag(u) Omega diag(v) correct? Does diagonal "
        "scaling preserve matrix rank?"
    ),
    "p9-s3a": (
        "Verify the Hadamard product claim in detail: for rank-1 matrix "
        "Lambda_{mn} = a_m b_n, the Hadamard product Lambda circ Omega has "
        "entry a_m b_n Omega_{mn} = (diag(a) Omega diag(b))_{mn}. Confirm "
        "this is standard. Verify that diag(u) * Omega * diag(v) preserves "
        "rank when u, v have all nonzero entries (which follows from "
        "lambda_{abgd} != 0). Search for: 'Hadamard product rank-1 diagonal "
        "scaling' on math.SE."
    ),
    "p9-s4": (
        "This is the HARD DIRECTION (converse). Verify: if lambda is NOT "
        "rank-1 in its first two indices (Lambda has rank >= 2), then for "
        "generic cameras some 3x3 minor det(Lambda circ Omega) is nonzero. "
        "The argument uses Zariski genericity: det(Lambda circ Omega) = 0 "
        "for ALL row/column triples is a polynomial condition on camera "
        "parameters that is 'not identically zero'. Verify: (a) is the "
        "'not identically zero' claim justified? Does an explicit "
        "construction exist? (b) Does Zariski-genericity ('polynomial "
        "condition, not identically zero, hence generic') apply correctly "
        "here — are we working over an irreducible variety? (c) Is the "
        "rank bound rank(Lambda circ Omega) <= rank(Lambda) * rank(Omega) "
        "used correctly?"
    ),
    "p9-s4a": (
        "Verify the Zariski-genericity argument rigorously: 'det(Lambda circ "
        "Omega) = 0 for ALL row/column triples is a polynomial condition on "
        "cameras that does not vanish identically; hence it fails generically.' "
        "Check: (a) Is the space of camera configurations irreducible (it is "
        "an open subset of (R^{3x4})^n, hence irreducible)? (b) Is the claim "
        "'not identically zero' proven or just asserted? A rigorous proof "
        "would need an explicit camera configuration where the minor is nonzero. "
        "(c) Does 'Zariski-generic' correctly mean 'outside a proper Zariski-"
        "closed subset'? Search for: 'Zariski genericity polynomial condition' "
        "on MathOverflow."
    ),
    "p9-s5": (
        "Verify the matricization argument: testing (1,2) vs (3,4), (1,3) vs "
        "(2,4), and (1,4) vs (2,3) suffices for rank-1 detection. The claim "
        "is: a 4-tensor lambda has rank 1 iff all three matricizations have "
        "matrix rank 1. Verify: (a) rank-1 tensor trivially has rank-1 "
        "matricizations. (b) For the converse: if lambda_{(ab),(gd)} = "
        "f_{ab} g_{gd} and lambda_{(ag),(bd)} = h_{ag} k_{bd}, does this "
        "force lambda = u_a v_b w_g x_d? This requires a nontrivial argument "
        "about tensor decomposition. Search for: 'tensor rank-1 matricization "
        "characterization' or 'Segre variety flattenings' on math.SE/MO."
    ),
    "p9-s6": (
        "Verify the construction of F: coordinate functions are all 3x3 minors "
        "det[T^(alpha_m, beta_n, gamma, delta)_{i_m, j_n, k, l}]_{3x3} over "
        "all mode-pair fixings. Check: (a) Degree 3 — each minor is a 3x3 "
        "determinant of T entries, so degree 3 in T. Correct. (b) Camera-"
        "independent — the polynomials have coefficients +/-1 from determinant "
        "expansion, no camera parameters. Correct? (c) O(n^8) coordinate "
        "functions — count: 3 choices of (alpha_m, i_m) from 3n camera-row "
        "pairs, 3 choices of (beta_n, j_n), and (gamma,k), (delta,l) fixed. "
        "That gives C(3n,3)^2 * (3n)^2 ~ O(n^8). Verify this counting. "
        "(d) Do the three matricization variants multiply this by 3?"
    ),
    "p9-s7": (
        "Verify the geometric interpretation: Q tensors are quadrifocal "
        "tensors from multiview geometry. Confirm: (a) the standard definition "
        "of quadrifocal tensor matches Q^(abgd)_{ijkl} = det[rows]. (b) Rank-1 "
        "scaling corresponds to gauge freedom (independent rescaling per camera "
        "per role). (c) The rank-2 bilinear form structure is the classical "
        "constraint from projective geometry. Cross-reference with Hartley-"
        "Zisserman (Multiple View Geometry) and Heyden (1998) on quadrifocal "
        "tensors. Search for: 'quadrifocal tensor rank constraint' on math.SE/MO."
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
        "multiview geometry, algebraic geometry, tensor rank, "
        "determinantal varieties, Segre varieties, projective geometry.",
        "",
        "## Task",
        "",
        "Verify one step of a proof that a polynomial map F (camera-independent, "
        "degree 3) detects rank-1 scaling of quadrifocal tensors. The quadrifocal "
        "tensor Q^(abgd)_{ijkl} = det[rows] is constructed from n >= 5 generic "
        "3x4 camera matrices. Cross-reference with math.SE/MO discussions and "
        "standard references (Hartley-Zisserman, Heyden 1998, Landsberg's "
        "Tensors: Geometry and Applications) when possible.",
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
        "Problem 9 asks: given n >= 5 generic 3x4 camera matrices, with "
        "Q^(abgd)_{ijkl} = det[rows], does a polynomial map F (camera-independent, "
        "bounded degree) detect rank-1 scaling lambda = u otimes v otimes w otimes x?",
        "",
        "The answer is: Yes. F consists of all 3x3 minors of the bilinear form "
        "matrices induced by fixing two of the four tensor modes, taken over all "
        "three matricization pairs. Each coordinate function has degree 3 in the "
        "scaled tensor T entries, with coefficients +/-1 (camera-independent). "
        "The forward direction uses Hadamard-product rank preservation under "
        "diagonal scaling; the converse uses Zariski-genericity of cameras.",
        "",
        "Key mathematical tools: Hodge duality of 2-forms in R^4, rank of "
        "alternating bilinear forms, Hadamard product with rank-1 matrices as "
        "diagonal scaling, Segre variety characterization via flattenings, "
        "Zariski-genericity arguments over camera parameter space.",
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim in this proof step.",
        "2. Search math.SE/MO for relevant discussions (quadrifocal tensor, "
        "tensor rank-1 detection, Segre variety, determinantal varieties, "
        "Hadamard product rank, alternating bilinear forms, Zariski genericity, "
        "multiview geometry rank constraints).",
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
        "Review this proof that a degree-3, camera-independent polynomial map F "
        "detects rank-1 scaling of quadrifocal tensors Q^(abgd)_{ijkl} = det[rows] "
        "constructed from n >= 5 generic 3x4 camera matrices. Assess completeness, "
        "correctness, and suggest improvements.",
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
        "1. Is the proof complete? Does the answer follow from the stated steps?",
        "2. Are there unstated assumptions (e.g., about linear independence of "
        "camera rows, genericity conditions, field of definition R vs algebraically "
        "closed, dimension requirements n >= 5)?",
        "3. The bilinear form reduction (Section 1) is foundational — verify that "
        "det[p;q;c;d] with c,d linearly independent gives rank exactly 2, not "
        "rank 0 (which would happen if c,d were dependent).",
        "4. The converse direction (Section 4) is the hardest part — is the "
        "Zariski-genericity argument rigorous? Does it need an explicit witness "
        "(specific camera configuration where the minor is nonzero)?",
        "5. The matricization argument (Section 5) claims rank-1 in all three "
        "flattenings implies tensor rank-1. This is true for order-2 tensors "
        "but needs justification for order-4. Is the proof of this step complete?",
        "6. The O(n^8) count (Section 6) — verify the combinatorial counting. "
        "Does this include all three matricization variants?",
        "7. Search math.SE/MO for: quadrifocal tensor rank constraint, Segre "
        "variety flattenings, tensor rank-1 detection polynomial, Hadamard "
        "product diagonal scaling rank, alternating bilinear form rank Hodge.",
        "8. Suggest a tighter or more elegant statement of the main result.",
        "9. For each reference, include `site` when known "
        "(e.g., `mathoverflow.net` or `math.stackexchange.com`).",
        "10. Reply as a single JSON object matching the required schema. "
        "Use node_id='p9-synthesis' for the synthesis.",
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
        print("Run: python3 scripts/proof9-wiring-diagram.py", file=sys.stderr)
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
        "node_id": "p9-synthesis",
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
