#!/usr/bin/env python3
"""Polish the Problem 4 proof via Codex CLI.

For each proof node in the wiring diagram, generates a verification prompt
and runs it through `codex exec`. Codex cross-references with math.SE data
(when available), verifies mathematical claims, and identifies gaps.

Usage:
    # Generate prompts only (dry run)
    python3 scripts/run-proof-polish-codex-p4.py --dry-run

    # Run through Codex (needs math.SE data in se-data/math-processed/)
    python3 scripts/run-proof-polish-codex-p4.py --limit 11

    # With custom math.SE search path
    python3 scripts/run-proof-polish-codex-p4.py --math-se-dir se-data/math-processed/
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
WIRING_JSON = REPO_ROOT / "data" / "first-proof" / "problem4-wiring.json"
SOLUTION_MD = REPO_ROOT / "data" / "first-proof" / "problem4-solution.md"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem4-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem4-codex-prompts.jsonl"
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
                "required": ["question_id", "title", "relevance"],
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
    "p4-problem": (
        "Verify the problem is well-posed: check that Phi_n(p) = sum_i "
        "(sum_{j!=i} 1/(lambda_i - lambda_j))^2 is well-defined for monic "
        "real-rooted polynomials with distinct roots (Phi_n = infinity for "
        "repeated roots). Confirm the inequality "
        "1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q) is a valid statement: "
        "does ⊞_n preserve simple roots when both inputs have simple roots? "
        "Is the condition 'monic real-rooted of degree n' sufficient? "
        "Check whether the equality condition (p or q equals (x-a)^n) is "
        "consistent — note that (x-a)^n has ALL roots equal, so Phi_n = infinity "
        "and 1/Phi_n = 0. Verify this edge case is handled correctly."
    ),
    "p4-s1": (
        "Verify the electrostatic interpretation of Phi_n. Check: "
        "(a) the logarithmic derivative p'(x)/p(x) = sum_i 1/(x - lambda_i) "
        "is standard; (b) that evaluating the regularized version at root "
        "lambda_i gives sum_{j!=i} 1/(lambda_i - lambda_j); (c) by L'Hopital, "
        "p'(lambda_i) = prod_{j!=i} (lambda_i - lambda_j) is correct; "
        "(d) the interpretation as '1D log-gas Coulomb self-energy' is physically "
        "meaningful — in the log-gas model, the force on particle i is "
        "sum_{j!=i} 1/(lambda_i - lambda_j), so Phi_n is the total squared force. "
        "Search math.SE/MO for 'electrostatic energy polynomial roots', "
        "'log-gas Coulomb energy', 'Stieltjes electrostatic interpretation'."
    ),
    "p4-s2": (
        "Verify the discriminant/Cauchy-Schwarz bound: Phi_n >= n(n-1)^2 / "
        "sum_{i<j}(lambda_i - lambda_j)^2. Check the Cauchy-Schwarz application: "
        "Phi_n = sum_i (sum_{j!=i} 1/(lambda_i-lambda_j))^2. By C-S, "
        "(sum_i f_i^2)(sum_i 1^2) >= (sum_i |f_i|)^2. But the bound claimed "
        "involves sum_{i<j}(lambda_i-lambda_j)^2 — verify the exact algebraic "
        "manipulation. Also verify the consequence 1/Phi_n <= spread^2/(n(n-1)^2). "
        "Is the factor n(n-1)^2 correct, or should it be n(n-1) or similar? "
        "Search for 'polynomial root spread Cauchy-Schwarz bound'."
    ),
    "p4-s3": (
        "Verify the MSS (Marcus-Spielman-Srivastava 2015) finite free convolution "
        "properties: (a) real-rootedness preservation — this is a deep result, "
        "confirm the correct citation (MSS 2015 or 2022?); (b) the expected "
        "characteristic polynomial interpretation E_U[char(A + UBU*)] = p ⊞_n q; "
        "(c) linearization of finite free cumulants — is this exact at finite n "
        "or only asymptotic? The claim 'linearizes finite free cumulants' needs "
        "careful verification: the standard MSS result is about coefficients, "
        "and the cumulant interpretation requires additional work (cf. Arizmendi, "
        "Perales). Search MO for 'finite free convolution cumulants additive', "
        "'Marcus Spielman Srivastava real-rooted'."
    ),
    "p4-s3a": (
        "Verify the random matrix model: p ⊞_n q = E_U[char(A + UBU*)] where "
        "U is Haar-uniform on U(n). Check: (a) does A need to be Hermitian "
        "or just normal? (b) is the expectation over U(n) or O(n)? "
        "(c) the characteristic polynomial of A + UBU* is a random polynomial — "
        "does the expectation of a product of linear factors factor nicely? "
        "(d) MSS originally used a different formulation (interlacing families); "
        "verify the random matrix interpretation is equivalent. "
        "Cross-reference with arXiv:1504.00350 (MSS) and arXiv:1507.05020."
    ),
    "p4-s3b": (
        "CRITICAL NODE: Verify that finite free cumulants truly ADD under ⊞_n. "
        "This is the linchpin of the proof. Check: (a) the definition of finite "
        "free cumulants via non-crossing partitions — is this the same as "
        "Arizmendi-Perales (2018) or a different convention? (b) at finite n, "
        "the cumulant additivity kappa_k(p ⊞_n q) = kappa_k(p) + kappa_k(q) "
        "is claimed — is this exact or approximate? The classical free cumulants "
        "are additive under Voiculescu's ⊞, but the FINITE version ⊞_n may "
        "have corrections. (c) Search for: 'finite free cumulant additivity "
        "exact', 'Arizmendi Perales finite free cumulant'. If the additivity "
        "is only asymptotic (n -> infinity), the entire proof has a gap."
    ),
    "p4-s4": (
        "THIS IS THE HARDEST CLAIM — verify with extreme care. The claim: "
        "1/Phi_n is a CONCAVE function of the cumulants (kappa_2, kappa_3, ...). "
        "Check: (a) is there a rigorous proof of this concavity, or is it "
        "heuristic from the electrostatic analogy? 'Adding independent "
        "perturbations cannot decrease reciprocal Coulomb energy' is physically "
        "intuitive but not a proof. (b) What is the domain? Concavity requires "
        "a convex domain — is the set of valid cumulant vectors convex? "
        "(c) Phi_n depends on the ROOTS, not the cumulants directly. The map "
        "cumulants -> roots -> Phi_n involves solving a degree-n polynomial. "
        "Is the composition concave? (d) For concavity, we need the Hessian "
        "of 1/Phi_n w.r.t. kappa to be negative semidefinite. Has anyone "
        "computed this? (e) The proof uses f(0) = 0 for the superadditivity "
        "step — verify that kappa = 0 corresponds to (x-a)^n and 1/Phi_n = 0. "
        "Search MO for 'concavity reciprocal discriminant cumulants'."
    ),
    "p4-s4a": (
        "Verify the superadditivity-from-concavity argument: if f is concave "
        "and f(0) = 0, then f(a+b) >= f(a) + f(b). Check: (a) this is a "
        "standard fact — f concave implies f(ta + (1-t)b) >= tf(a) + (1-t)f(b). "
        "Setting a' = a+b, b' = 0, t = 1/2 doesn't immediately give the "
        "claimed inequality. The correct argument: f(a+b) = f(a+b + 0) >= "
        "... Actually, for f concave with f(0) >= 0, we need: "
        "f(a) = f((a/(a+b))(a+b) + (b/(a+b))*0) >= (a/(a+b))f(a+b). "
        "Similarly for f(b). Adding: f(a)+f(b) >= f(a+b). Wait — this gives "
        "the OPPOSITE inequality! Concavity gives SUBadditivity f(a+b) <= f(a)+f(b) "
        "when f(0) >= 0 and a,b >= 0. Check whether the proof has the direction "
        "of concavity/convexity BACKWARDS. This could be a critical error. "
        "Verify: is 1/Phi_n actually CONVEX (not concave) in the cumulants, "
        "or is f(0) < 0, or is the argument structured differently?"
    ),
    "p4-s5": (
        "Verify the degree-2 case: p(x) = x^2 - s^2, q(x) = x^2 - t^2. "
        "Check: (a) p ⊞_2 q = x^2 - (s^2 + t^2) — verify from the "
        "coefficient formula c_k = sum_{i+j=k} [(n-i)!(n-j)!/(n!(n-k)!)] a_i b_j "
        "with n=2. For k=1: c_1 = a_0*b_1*(2!*1!/(2!*1!)) + a_1*b_0*(1!*2!/(2!*1!)) "
        "= 0 (since a_1 = b_1 = 0 for symmetric roots). For k=2: compute "
        "carefully. (b) Phi_2(p) = 2/(a-b)^2 = 2/(2s)^2 = 1/(2s^2). So "
        "1/Phi_2(p) = 2s^2... wait, the solution says 1/Phi_2(p) = (a-b)^2/2 = "
        "4s^2/2 = 2s^2. But the general formula says 1/Phi_2 = s^2/2. "
        "THERE IS A POTENTIAL FACTOR-OF-2 ERROR. Check carefully. "
        "(c) Verify that equality holds in degree 2 (this is claimed)."
    ),
    "p4-s6": (
        "Verify the conclusion: (a) does cumulant additivity + concavity of "
        "1/Phi_n actually imply superadditivity? (See the concern in p4-s4a "
        "about the direction of the inequality.) (b) The equality condition "
        "'iff p or q is (x-a)^n' — verify this is correct. If p = (x-a)^n, "
        "then kappa(p) = 0 (except kappa_1 = a), so 1/Phi_n(p) = 0, and the "
        "inequality becomes 1/Phi_n(p ⊞_n q) >= 1/Phi_n(q), which should "
        "hold since p ⊞_n (x-a)^n shifts roots of q by a (translation). "
        "But does Phi_n change under translation? No — Phi_n is "
        "translation-invariant since it depends on root DIFFERENCES. So "
        "1/Phi_n(p ⊞_n q) = 1/Phi_n(q) and equality holds. Good. "
        "(c) Is the converse true? If equality holds, must one polynomial "
        "be degenerate? This needs the strict concavity of 1/Phi_n. "
        "Search for 'equality condition free convolution root separation'."
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
        "free probability, random matrix theory, finite free convolution, "
        "symmetric functions, electrostatic analogies for polynomial roots.",
        "",
        "## Task",
        "",
        "Verify one step of a proof that the root separation energy Phi_n "
        "satisfies a superadditivity inequality under finite free convolution: "
        "1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q). "
        "Cross-reference with math.SE/MO discussions and the primary sources "
        "(Marcus-Spielman-Srivastava 2015, Arizmendi-Perales 2018) when possible.",
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
        "Problem 4 asks: for monic real-rooted polynomials p, q of degree n, is "
        "1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q), where "
        "Phi_n(p) = sum_i (sum_{j!=i} 1/(lambda_i - lambda_j))^2 is the root "
        "separation energy and ⊞_n is the Marcus-Spielman-Srivastava finite "
        "free additive convolution?",
        "",
        "The answer is: Yes. The inequality holds, with equality iff p or q is "
        "(x-a)^n (all roots coincide). The proof strategy: (1) finite free "
        "cumulants add under ⊞_n, (2) 1/Phi_n is concave in the cumulants, "
        "(3) concavity + additivity => superadditivity.",
        "",
        "Key references: Marcus-Spielman-Srivastava (2015, arXiv:1504.00350), "
        "Arizmendi-Perales (2018, arXiv:1803.01353), "
        "Marcus (2021, arXiv:2103.05006).",
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim in this proof step.",
        "2. Search math.SE/MO for relevant discussions (finite free convolution, "
        "root separation energy, electrostatic polynomial root interpretation, "
        "free cumulants, log-gas Coulomb energy, MSS interlacing families, "
        "discriminant bounds).",
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
        "Review this proof that 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q) "
        "for monic real-rooted polynomials under finite free convolution. "
        "Assess completeness, correctness, and suggest improvements.",
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
        "1. Is the proof complete? Does the inequality follow from the stated steps?",
        "2. CRITICAL: The concavity claim (Section 4/5) is the hardest part. "
        "Is the concavity of 1/Phi_n in the finite free cumulants rigorously "
        "established, or only heuristically motivated by the electrostatic analogy? "
        "If heuristic, this is the main gap.",
        "3. CRITICAL: Check the direction of the concavity/superadditivity "
        "argument. If f is concave with f(0) = 0, is f(a+b) >= f(a) + f(b) "
        "or f(a+b) <= f(a) + f(b)? The standard result is that concave f with "
        "f(0) >= 0 is SUBadditive, not SUPERadditive. If the proof claims "
        "superadditivity from concavity, it may have the inequality backwards. "
        "Is 1/Phi_n actually CONVEX in the cumulants?",
        "4. The finite free cumulant additivity "
        "kappa_k(p ⊞_n q) = kappa_k(p) + kappa_k(q) — is this exact at "
        "finite n or only asymptotic? Cite the precise reference.",
        "5. The equality condition 'iff p or q is (x-a)^n' — does this require "
        "strict concavity/convexity? Is strict concavity proved?",
        "6. The degree-2 verification: check for factor-of-2 errors in the "
        "computation of Phi_2 and 1/Phi_2.",
        "7. Search math.SE/MO for: finite free convolution root separation, "
        "electrostatic energy polynomial roots superadditivity, MSS finite "
        "free cumulants, Arizmendi Perales finite free cumulant.",
        "8. Suggest a tighter or more elegant statement of the main result.",
        "9. For each reference, include `site` when known "
        "(e.g., `mathoverflow.net` or `math.stackexchange.com`).",
        "10. Reply as a single JSON object matching the required schema. "
        "Use node_id='p4-synthesis' for the synthesis.",
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
        print("Run: python3 scripts/proof4-wiring-diagram.py", file=sys.stderr)
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
        "node_id": "p4-synthesis",
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
