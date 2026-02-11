#!/usr/bin/env python3
"""Polish the Problem 4 proof (n<4 cases) via Codex CLI.

Focused on verifying the COMPLETE proof for n=2 (equality) and n=3
(Phi_3*disc identity + Titu's lemma). This is distinct from the earlier
run-proof-polish-codex-p4.py which targeted the now-broken concavity
argument. Here the proof chain is:

  n=2: 1/Phi_2 linear in disc → ⊞_2 preserves → equality
  n=3: centering → Phi_3*disc=18a_2^2 → ⊞_3 simplifies → Titu's lemma → QED

Usage:
    python3 scripts/run-proof-polish-codex-p4-lt4.py --dry-run
    python3 scripts/run-proof-polish-codex-p4-lt4.py --limit 10
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
STRATEGY_MD = REPO_ROOT / "data" / "first-proof" / "problem4-proof-strategy-skeleton.md"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem4-lt4-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem4-lt4-codex-prompts.jsonl"
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
                        "enum": [
                            "math.stackexchange.com",
                            "mathoverflow.net",
                            "other",
                            "unknown",
                        ],
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


# The n<4 proof chain focuses on nodes p4-s5, p4-s5a, p4-s5b, p4-s5c, p4-s5d
# plus context from p4-problem, p4-s1, p4-s3, p4-s3a, p4-s6.

NODE_VERIFICATION_FOCUS = {
    "p4-problem": (
        "Verify the problem statement is well-posed for n=2 and n=3 specifically. "
        "For n=2: Phi_2(p) = 2/(lambda_1 - lambda_2)^2 for distinct roots. "
        "For n=3: Phi_3 has 3 squared-force terms. Check that the inequality "
        "1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q) is meaningful when all "
        "roots are distinct (Phi_n finite, 1/Phi_n > 0)."
    ),
    "p4-s5": (
        "VERIFY THE n=2 PROOF. The claim: 1/Phi_2(p) = (a_1^2 - 4a_2)/2 = disc(p)/2. "
        "Check: (a) Phi_2 = 2/(a-b)^2 where a,b are roots of x^2 + a_1*x + a_2. "
        "disc = (a-b)^2 = a_1^2 - 4a_2. So 1/Phi_2 = (a_1^2 - 4a_2)/2. ✓ "
        "(b) The ⊞_2 formula: c_1 = a_1 + b_1, c_2 = a_2 + a_1*b_1/2 + b_2. "
        "Verify the weight w(2,1,1) = 1!*1!/(2!*0!) = 1/2. ✓ "
        "(c) Compute disc(p⊞_2 q) = c_1^2 - 4c_2 and verify it equals "
        "(a_1^2 - 4a_2) + (b_1^2 - 4b_2). This is the key cancellation. "
        "EXPLICITLY expand and simplify. This is a complete proof, not just "
        "a verification — the algebra should close."
    ),
    "p4-s5a": (
        "VERIFY THE CENTERING REDUCTION. Two claims: "
        "(1) Phi_n is translation-invariant: shifting all roots by c does not "
        "change Phi_n because it depends only on differences lambda_i - lambda_j. "
        "This is trivially true. ✓ "
        "(2) ⊞_n commutes with translation: if p̃(x) = p(x+α), q̃(x) = q(x+β), "
        "then p̃ ⊞_n q̃ = (p ⊞_n q)(x + α + β). The argument via the random "
        "matrix model: A → A+αI, B → B+βI, then A+αI + Q(B+βI)Q* = "
        "(A+QBQ*) + (α+β)I. So eigenvalues shift by α+β. Hence the expected "
        "characteristic polynomial shifts by α+β. "
        "CHECK: is this argument rigorous? Does the Haar expectation commute "
        "with the translation? Yes — E[f(M + cI)] = E[f(M)](· - c) for "
        "characteristic polynomials. Confirm this is standard."
    ),
    "p4-s5b": (
        "THIS IS THE KEY IDENTITY — VERIFY WITH EXTREME CARE. "
        "Claim: For centered cubics p(x) = x^3 + a_2*x + a_3 with "
        "disc = -4a_2^3 - 27a_3^2 > 0 and a_2 < 0: "
        "    Phi_3(p) * disc(p) = 18 * a_2^2 "
        "Equivalently: 1/Phi_3 = disc/(18a_2^2) = -2a_2/9 - 3a_3^2/(2a_2^2). "
        "VERIFY by: "
        "(a) For centered cubic, p'(l_i) = 3l_i^2 + a_2 (since e_1 = 0, "
        "p'(x) = 3x^2 + a_2). The force at root l_i is f_i = (3l_i)/(3l_i^2 + a_2). "
        "Phi_3 = sum_i f_i^2 = 9 * sum_i l_i^2/(3l_i^2 + a_2)^2. "
        "(b) disc = (l_1-l_2)^2(l_1-l_3)^2(l_2-l_3)^2 with l_1+l_2+l_3 = 0. "
        "(c) Compute Phi_3 * disc symbolically and verify = 18*(l_1^2+l_1*l_2+l_2^2)^2 "
        "= 18*a_2^2 (since a_2 = e_2 = -(l_1^2+l_1*l_2+l_2^2) with l_3=-l_1-l_2). "
        "This has been verified in SymPy. Cross-check by independent computation. "
        "Search MO/math.SE for: 'discriminant times Coulomb energy polynomial', "
        "'Phi times discriminant identity cubic'."
    ),
    "p4-s5c": (
        "VERIFY the ⊞_3 simplification for centered cubics. "
        "Claim: when a_1 = b_1 = 0, the MSS coefficient formula gives: "
        "c_2 = a_2 + b_2 (the weight w(3,1,1) = 2!*2!/(3!*1!) = 2/3, but "
        "the cross-term is (2/3)*a_1*b_1 = 0). "
        "c_3 = a_3 + b_3 (cross-terms: (1/3)*a_2*b_1 + (1/3)*a_1*b_2 = 0). "
        "VERIFY the MSS weights: w(n,i,j) = (n-i)!(n-j)!/(n!(n-k)!) with n=3. "
        "For c_2 (k=2): w(3,0,2)=1, w(3,1,1)=2!*2!/(3!*1!)=4/6=2/3, w(3,2,0)=1. "
        "For c_3 (k=3): w(3,0,3)=1, w(3,1,2)=2!*1!/(3!*0!)=2/6=1/3, "
        "w(3,2,1)=1!*2!/(3!*0!)=2/6=1/3, w(3,3,0)=1. "
        "All cross-terms vanish when a_1=b_1=0. ✓ "
        "IMPORTANT: verify this does NOT hold for n=4. At n=4 centered, "
        "c_4 has cross-term w(4,2,2)*a_2*b_2 = (2!*2!/(4!*0!))*a_2*b_2 = a_2*b_2/6. "
        "This is why the n=3 argument doesn't generalize."
    ),
    "p4-s5d": (
        "VERIFY THE CAUCHY-SCHWARZ STEP (completes the n=3 proof). "
        "With s = -a_2 > 0, t = -b_2 > 0, u = a_3, v = b_3: "
        "1/Phi(p) = 2s/9 - 3u^2/(2s^2), "
        "1/Phi(q) = 2t/9 - 3v^2/(2t^2), "
        "1/Phi(conv) = 2(s+t)/9 - 3(u+v)^2/(2(s+t)^2). "
        "Surplus = (3/2)[u^2/s^2 + v^2/t^2 - (u+v)^2/(s+t)^2]. "
        "VERIFY: (a) The algebra: substitute and simplify to get the surplus. "
        "(b) The inequality: by Titu's lemma (Engel form of Cauchy-Schwarz), "
        "x^2/a + y^2/b >= (x+y)^2/(a+b) for a,b > 0. Applied with "
        "x=u, y=v, a=s^2, b=t^2: u^2/s^2 + v^2/t^2 >= (u+v)^2/(s^2+t^2). "
        "Since s,t > 0: (s+t)^2 = s^2+2st+t^2 > s^2+t^2, so "
        "1/(s^2+t^2) > 1/(s+t)^2, giving (u+v)^2/(s^2+t^2) >= (u+v)^2/(s+t)^2. "
        "Combining: surplus >= 0. ✓ "
        "(c) Equality condition: surplus = 0 iff (u+v)^2 = 0 (i.e., a_3 = -b_3) "
        "AND the Titu step is tight (u*t^2 = v*s^2). The only solution with "
        "u = -v is u = v = 0 (both symmetric cubics) or s = t and u = -v. "
        "In the latter case the polynomials have the same discriminant. "
        "Search for: 'Titu lemma Engel form Cauchy Schwarz', "
        "'superadditivity from Cauchy Schwarz reciprocal'."
    ),
    "p4-s6": (
        "VERIFY THE CONCLUSION consolidates correctly. Check: "
        "(a) n=2 proved with equality (Section 5, p4-s5). ✓ "
        "(b) n=3 proved with strict inequality (p4-s5a through p4-s5d). ✓ "
        "(c) n>=4 status: numerically verified (0/5000 violations for n=4,5) "
        "but analytic proof open. The n=3 method fails because: "
        "(i) Phi_4*disc is not const*a_2^2, and "
        "(ii) ⊞_4 has cross-term (1/6)*a_2*b_2 in c_4 even when centered. "
        "(d) Plain coefficient addition fails ~29% at n=4, so the cross-term "
        "is essential — the superadditivity is specific to ⊞_n. "
        "Verify these structural claims are correctly stated."
    ),
}

# Nodes to include in the verification (the n<4 proof chain + context)
PROOF_CHAIN_NODES = [
    "p4-problem",
    "p4-s5",      # n=2 proof
    "p4-s5a",     # centering reduction
    "p4-s5b",     # Phi_3*disc identity
    "p4-s5c",     # ⊞_3 simplification
    "p4-s5d",     # Cauchy-Schwarz / Titu
    "p4-s6",      # conclusion
]


def build_node_prompt(
    node: dict,
    edges: list[dict],
    solution_text: str,
    strategy_text: str,
    math_se_dir: Path | None = None,
) -> str:
    node_id = node["id"]
    focus = NODE_VERIFICATION_FOCUS.get(node_id, "Verify the mathematical claim.")

    incoming = [e for e in edges if e["target"] == node_id]
    outgoing = [e for e in edges if e["source"] == node_id]

    lines = [
        "You are a mathematical proof verifier with expertise in "
        "polynomial algebra, symmetric functions, discriminants, "
        "Cauchy-Schwarz inequalities, and finite free convolution (MSS 2015).",
        "",
        "## Task",
        "",
        "Verify one step of a COMPLETE proof that for monic real-rooted degree-n "
        "polynomials p, q: 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q), "
        "focusing on n=2 (equality) and n=3 (strict inequality via the identity "
        "Phi_3 * disc = 18 * a_2^2 and Titu's lemma).",
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
        "## Full Proof (for reference)",
        "",
        solution_text[:8000],  # truncate if very long
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim by independent computation.",
        "2. For algebraic identities, EXPAND and SIMPLIFY explicitly.",
        "3. Search math.SE/MO for related discussions.",
        "4. Identify any gaps, unstated assumptions, or errors.",
        "5. Reply as a single JSON object matching the required schema.",
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
    strategy_text: str,
    wiring: dict,
    math_se_dir: Path | None = None,
) -> str:
    stats = wiring.get("stats", {})
    lines = [
        "You are a mathematical proof verifier reviewing a COMPLETE proof "
        "for the n=2 and n=3 cases of a superadditivity inequality.",
        "",
        "## Task",
        "",
        "Review the proof that 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q) "
        "for n=2 (equality) and n=3 (strict inequality). The proof chain is:",
        "",
        "n=2: 1/Phi_2 = disc/2 is linear → ⊞_2 preserves → surplus = 0.",
        "n=3: (1) center WLOG, (2) Phi_3*disc = 18*a_2^2 identity, "
        "(3) ⊞_3 = plain addition for centered cubics, (4) Titu's lemma.",
        "",
        "## Proof",
        "",
        solution_text,
        "",
        "## Proof Strategy Context",
        "",
        strategy_text[:4000],
        "",
        "## Instructions",
        "",
        "1. Is the n=2 proof complete and correct?",
        "2. Is the n=3 proof complete and correct? Specifically:",
        "   (a) Is the centering reduction valid?",
        "   (b) Is the identity Phi_3 * disc = 18*a_2^2 correct?",
        "   (c) Does ⊞_3 truly reduce to coefficient addition when a_1=b_1=0?",
        "   (d) Is the Titu's lemma application correct?",
        "3. Are there any hidden assumptions (e.g., a_2 < 0 for real roots)?",
        "4. Is the equality condition for n=3 correctly characterized?",
        "5. Does the proof correctly explain why n>=4 requires a different approach?",
        "6. Search math.SE/MO for the Phi_3*disc identity — has this been "
        "noted before?",
        "7. Reply as JSON with node_id='p4-lt4-synthesis'.",
    ]
    if math_se_dir:
        lines.extend([
            "",
            "## Local Corpus Hint",
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
    ap.add_argument("--strategy", type=Path, default=STRATEGY_MD)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--prompts-out", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gpt-5.3-codex")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--math-se-dir", type=Path, default=DEFAULT_MATH_SE_DIR)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args()

    # Load wiring diagram
    if not args.wiring.exists():
        print(f"Wiring diagram not found: {args.wiring}", file=sys.stderr)
        return 2
    wiring = json.loads(args.wiring.read_text())

    # Load solution and strategy
    solution_text = args.solution.read_text() if args.solution.exists() else ""
    strategy_text = args.strategy.read_text() if args.strategy.exists() else ""

    nodes_by_id = {n["id"]: n for n in wiring["nodes"]}
    edges = wiring["edges"]

    # Build prompts for the n<4 proof chain only
    prompts = []
    for node_id in PROOF_CHAIN_NODES:
        if node_id not in nodes_by_id:
            print(f"Warning: node {node_id} not in wiring, skipping", file=sys.stderr)
            continue
        node = nodes_by_id[node_id]
        prompt_text = build_node_prompt(
            node, edges, solution_text, strategy_text,
            math_se_dir=args.math_se_dir,
        )
        prompts.append({
            "node_id": node_id,
            "node_type": node["node_type"],
            "prompt": prompt_text,
        })

    # Add synthesis prompt
    prompts.append({
        "node_id": "p4-lt4-synthesis",
        "node_type": "synthesis",
        "prompt": build_synthesis_prompt(
            solution_text, strategy_text, wiring,
            math_se_dir=args.math_se_dir,
        ),
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
                status = out.get(
                    "claim_verified",
                    "parse_error" if out.get("parse_error") else "?",
                )
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
