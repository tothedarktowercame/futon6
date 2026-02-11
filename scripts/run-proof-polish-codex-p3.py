#!/usr/bin/env python3
"""Polish the Problem 3 proof via Codex CLI.

For each proof node in the wiring diagram, generates a verification prompt
and runs it through `codex exec`. Codex cross-references with math.SE data
(when available), verifies mathematical claims, and identifies gaps.

Usage:
    # Generate prompts only (dry run)
    python3 scripts/run-proof-polish-codex-p3.py --dry-run

    # Run through Codex (needs math.SE data in se-data/math-processed/)
    python3 scripts/run-proof-polish-codex-p3.py --limit 9

    # With custom math.SE search path
    python3 scripts/run-proof-polish-codex-p3.py --math-se-dir se-data/math-processed/
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
WIRING_JSON = REPO_ROOT / "data" / "first-proof" / "problem3-wiring.json"
SOLUTION_MD = REPO_ROOT / "data" / "first-proof" / "problem3-solution.md"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem3-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem3-codex-prompts.jsonl"
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
    "p3-problem": (
        "Verify the problem is well-posed: check that S_n(lambda) is the right "
        "state space for permutations of a restricted partition with distinct parts "
        "(unique part 0, no part 1). Confirm that the target stationary distribution "
        "F*_mu(x;q=1,t)/P*_lambda(x;q=1,t) is well-defined and sums to 1. Check "
        "whether 'nontrivial' means rates don't depend on F*_mu values."
    ),
    "p3-s1": (
        "Verify the construction: inhomogeneous multispecies t-PushTASEP on a ring "
        "of n sites with content lambda and site rates 1/x_i. Confirm that: "
        "(a) when a clock rings at site j, the particle at j becomes active, "
        "(b) the active particle chooses the k-th weaker particle with probability "
        "t^(k-1)/[m]_t, (c) the cascade terminates when a vacancy is displaced. "
        "Cross-reference with Ayyer-Martin-Williams (2024), arXiv:2403.10485 "
        "Definition 2.1 or equivalent."
    ),
    "p3-s2": (
        "Verify that the t-PushTASEP is a valid continuous-time Markov chain: "
        "finite state space S_n(lambda) = n! states (distinct parts), explicit "
        "generator matrix with finite nonneg rates, irreducibility. Confirm that "
        "lambda_n = 0 guarantees at least one vacancy. Check whether irreducibility "
        "is stated or proved in AMW."
    ),
    "p3-s3": (
        "Verify the nontriviality claim: the transition rates depend only on "
        "local species ordering, x_i, and t — NOT on values of F*_mu or P*_lambda. "
        "This is the key distinction from a Metropolis chain. Search math.SE/MO for "
        "discussions of 'nontrivial' Markov chains with polynomial stationary "
        "distributions — is there a standard definition?"
    ),
    "p3-s4": (
        "This is the critical node. Verify that AMW Theorem 1.1 (arXiv:2403.10485) "
        "actually states: the stationary probability of eta in S_n(lambda) for the "
        "multispecies t-PushTASEP is pi(eta) = F_eta(x;1,t)/P_lambda(x;1,t). "
        "Check: (a) is this for the ring or the line? (b) is q=1 or general q? "
        "(c) are there conditions on x_i (positivity, distinctness)? "
        "(d) is this the single-species or multispecies version?"
    ),
    "p3-s5": (
        "Verify the notation bridge: the problem statement uses F*_mu, P*_lambda "
        "(star/interpolation notation) while AMW uses F_mu, P_lambda. Confirm these "
        "are the same family at q=1, or identify the precise normalization difference. "
        "Search for: 'interpolation Macdonald polynomial' vs 'ASEP polynomial' "
        "vs 'nonsymmetric Macdonald polynomial at q=1'. Is any normalization "
        "cancelled by taking the ratio F/P?"
    ),
    "p3-s6": (
        "Verify the n=2 sanity check: lambda=(a,0), states {(a,0),(0,a)}. "
        "The chain should have: rate (a,0)->(0,a) = 1/x_1, rate (0,a)->(a,0) = 1/x_2. "
        "Stationary distribution: pi(a,0) = x_1/(x_1+x_2), pi(0,a) = x_2/(x_1+x_2). "
        "Confirm this matches AMW Proposition 2.4 (single-species recoloring). "
        "Also verify: does F_{(a,0)}(x;1,t)/P_{(a,0)}(x;1,t) = x_1/(x_1+x_2)?"
    ),
    "p3-s7": (
        "Verify the conclusion composes correctly: existence (from s1), "
        "nontriviality (from s3), correct stationary distribution (from s4+s5). "
        "Check for gaps: (a) does the proof need irreducibility/ergodicity "
        "for uniqueness of stationary distribution? (b) are there parameter "
        "constraints (t in [0,1), x_i > 0) that should be stated? "
        "(c) does the answer address the 'nontrivial' requirement fully?"
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
        "You are a mathematical proof verifier with expertise in algebraic "
        "combinatorics, integrable probability, Macdonald polynomials, "
        "Hecke algebras, and interacting particle systems (ASEP/TASEP).",
        "",
        "## Task",
        "",
        "Verify one step of a proof that an explicit nontrivial Markov chain "
        "(the inhomogeneous multispecies t-PushTASEP) has stationary distribution "
        "given by a ratio of ASEP polynomials / Macdonald polynomials at q=1. "
        "Cross-reference with math.SE/MO discussions and the primary source "
        "(Ayyer-Martin-Williams 2024, arXiv:2403.10485) when possible.",
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
        "Problem 3 asks: does there exist a NONTRIVIAL Markov chain on S_n(lambda) "
        "(permutations of a restricted partition lambda with distinct parts) whose "
        "stationary distribution is pi(mu) = F*_mu(x;q=1,t) / P*_lambda(x;q=1,t)?",
        "",
        "The answer is: Yes, take the inhomogeneous multispecies t-PushTASEP on the "
        "ring of n sites with content lambda. Site j has exponential clock of rate "
        "1/x_j. When it rings, the particle there chooses a weaker particle with "
        "t-geometric probabilities and swaps, cascading until a vacancy is displaced.",
        "",
        "Key reference: Ayyer-Martin-Williams (2024), arXiv:2403.10485, Theorem 1.1.",
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim in this proof step.",
        "2. Search math.SE/MO for relevant discussions (ASEP polynomials, Macdonald "
        "polynomials at q=1, t-PushTASEP, multispecies exclusion processes, Hecke "
        "algebra exchange relations, interpolation polynomials).",
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
    return "\n".join([
        "You are a mathematical proof verifier reviewing a complete proof.",
        "",
        "## Task",
        "",
        "Review this proof that the inhomogeneous multispecies t-PushTASEP has "
        "stationary distribution F_mu(x;1,t)/P_lambda(x;1,t). Assess completeness, "
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
        "2. Are there unstated assumptions (e.g., about distinctness of parts, "
        "positivity of x_i, range of t, irreducibility of the chain)?",
        "3. The notation bridge (Section 5) is a potential weak point — is the "
        "identification F*_mu = F_mu at q=1 correct, or is there a normalization "
        "subtlety that matters?",
        "4. The nontriviality claim (Section 3) could be challenged — is there "
        "a precise definition of 'nontrivial' in this context? Could someone argue "
        "that knowing the AMW theorem implicitly uses F*_mu in the design?",
        "5. Search math.SE/MO for: multispecies ASEP stationary distribution, "
        "t-PushTASEP Macdonald polynomial, interpolation ASEP polynomial q=1, "
        "Hecke algebra exchange relation ASEP.",
        "6. Suggest a tighter or more elegant statement of the main result.",
        "7. For each reference, include `site` when known "
        "(e.g., `mathoverflow.net` or `math.stackexchange.com`).",
        "8. Reply as a single JSON object matching the required schema. "
        "Use node_id='p3-synthesis' for the synthesis.",
    ])
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
        print("Run: python3 scripts/proof3-wiring-diagram.py", file=sys.stderr)
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
        "node_id": "p3-synthesis",
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
