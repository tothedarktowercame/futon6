#!/usr/bin/env python3
"""Polish the Problem 7 proof via Codex CLI.

For each proof node in the wiring diagram, generates a verification prompt
and runs it through `codex exec`. Codex cross-references with math.SE data
(when available), verifies mathematical claims, and identifies gaps.

Usage:
    # Generate prompts only (dry run)
    python3 scripts/run-proof-polish-codex-p7.py --dry-run

    # Run through Codex (needs math.SE data in se-data/math-processed/)
    python3 scripts/run-proof-polish-codex-p7.py --limit 9

    # With custom math.SE search path
    python3 scripts/run-proof-polish-codex-p7.py --math-se-dir se-data/math-processed/
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
WIRING_JSON = REPO_ROOT / "data" / "first-proof" / "problem7-wiring.json"
SOLUTION_MD = REPO_ROOT / "data" / "first-proof" / "problem7-solution.md"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem7-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem7-codex-prompts.jsonl"
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
    "p7-problem": (
        "Verify the problem is well-posed: Gamma is a uniform (cocompact) lattice "
        "in a connected real semi-simple Lie group G, containing an element of order 2. "
        "The question asks whether Gamma = pi_1(M) for a closed manifold M whose "
        "universal cover M_tilde is rationally acyclic (H_*(M_tilde; Q) = 0 in positive "
        "degrees). Confirm: (a) 'rationally acyclic universal cover' is equivalent to "
        "M being a rational model for B Gamma, i.e., H_*(M; Q) = H_*(Gamma; Q). "
        "(b) The 2-torsion assumption is essential — without it the answer is trivially "
        "yes (take X/Gamma for the symmetric space X = G/K). (c) Is the question asking "
        "'for ALL such Gamma' or 'there EXISTS such Gamma'? The answer claims existence "
        "for specific choices — is this the right reading?"
    ),
    "p7-s1": (
        "Verify the orbifold issue: when Gamma has torsion, the action on X = G/K is "
        "NOT free. (a) Check that an element g of order 2 in Gamma necessarily has a "
        "nonempty fixed-point set X^g in the symmetric space (use: isometries of "
        "non-positively curved spaces have convex fixed-point sets; g has order 2 so it "
        "is an involution of X). (b) Confirm X/Gamma is an orbifold, not a manifold. "
        "(c) But crucially: the proof does NOT need X/Gamma to be a manifold — it seeks "
        "ANY closed manifold M with pi_1 = Gamma. Verify this logical distinction is "
        "correctly stated. Search MO/math.SE for 'lattice torsion symmetric space orbifold'."
    ),
    "p7-s2": (
        "Verify that Gamma is a rational Poincare duality group. (a) A uniform lattice "
        "in a semi-simple group has the rational cohomology of the orbifold X/Gamma, "
        "which satisfies Poincare duality over Q in dimension d = dim(X). Confirm this "
        "is standard — reference: Brown 'Cohomology of Groups' or Borel-Serre. "
        "(b) Does rational PD require Gamma to be torsion-free? NO — rational PD holds "
        "for uniform lattices WITH torsion (the orbifold still has rational PD). Verify "
        "this carefully. (c) What is the formal dimension? d = dim(G/K). Confirm this "
        "equals the real rank computation for specific examples. Search for 'rational "
        "Poincare duality group lattice torsion'."
    ),
    "p7-s3": (
        "Verify the finite-complex step. (a) The claim now uses Fowler's equivariant "
        "finiteness theorem (arXiv:1204.4667): under fixed-set Euler-vanishing "
        "hypotheses, orbifold groups lie in FH(Q). Confirm this theorem and hypotheses. "
        "(b) Check that FH(Q) means existence of a finite CW complex Y with pi_1(Y)=Gamma "
        "and rationally acyclic universal cover. (c) Verify this is a finite-CW result, "
        "not yet a closed manifold realization."
    ),
    "p7-s3a": (
        "Verify the arithmetic-lattice example claim from Fowler Section 5. "
        "(a) Check that explicit lattice extensions in FH(Q) are given there. "
        "(b) Confirm what torsion orders are covered in the explicit arithmetic "
        "example(s), and whether order-2 torsion is directly produced. "
        "(c) Verify this step is evidence/support, not a complete proof of the final "
        "manifold statement."
    ),
    "p7-s4": (
        "Verify the remaining-gap statement. (a) Is it correct that FH(Q) does NOT by "
        "itself give a closed manifold with the same pi_1? (b) Is a separate manifold-"
        "upgrade/surgery step needed? (c) Check whether the writeup correctly marks the "
        "obstruction computation as unresolved rather than claimed solved."
    ),
    "p7-s5": (
        "Verify the Smith theory argument. (a) Classical Smith theory: if Z/p acts on a "
        "mod-p acyclic space, the fixed-point set is mod-p acyclic (hence nonempty). "
        "For p=2, this means: Z/2 acting on a mod-2 acyclic space must have fixed points "
        "=> the action cannot be free. (b) Over Q: the proof claims this obstruction "
        "vanishes because 2 is invertible in Q. Verify: the transfer argument gives "
        "H_*(X/G; Q) -> H_*(X; Q) is injective (or split), which doesn't force fixed "
        "points. Confirm: Z/2 CAN act freely on a rationally acyclic space (example: "
        "any odd sphere is rationally acyclic after appropriate modification). "
        "(c) The key logical point: M_tilde (universal cover) is rationally acyclic, and "
        "the Z/2 subgroup of Gamma acts FREELY on M_tilde (because Gamma acts freely on "
        "the universal cover by definition of pi_1). So Smith theory over Z/2 doesn't "
        "apply because the space is only rationally acyclic, not mod-2 acyclic. Verify "
        "this distinction is correctly handled."
    ),
    "p7-s6": (
        "Verify the conclusion composes correctly. (a) The chain of reasoning: "
        "rational PD (s2) + FH(Q) finite-complex realization (s3,s3a) + unresolved "
        "manifold-upgrade obstruction (s4) + no Smith obstruction (s5) => conditional "
        "existence statement. "
        "Check for gaps: (i) Is the surgery obstruction the ONLY obstruction, or are "
        "there further obstructions (e.g., from the normal invariant, from Pi-Pi theorem)? "
        "(ii) Does the proof actually construct M, or only show it's not obstructed? "
        "(iii) The answer is now conditional: is that appropriately scoped? "
        "(b) Verify completeness: does every node in the wiring diagram contribute to "
        "the conditional conclusion? Are there dangling nodes? (c) Is the confidence "
        "level now calibrated to the unresolved manifold-upgrade step?"
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
        "geometric group theory, surgery theory, equivariant finiteness "
        "obstructions, lattices in Lie groups, Smith theory, and "
        "Poincare duality groups.",
        "",
        "## Task",
        "",
        "Verify one step of a proof that a uniform lattice Gamma (with 2-torsion) "
        "in a semi-simple Lie group can be realized as pi_1(M) for a closed manifold "
        "M with rationally acyclic universal cover. The proof uses orbifold/Bredon "
        "duality, equivariant finiteness (FH(Q)), surgery-theoretic upgrades, and a "
        "Smith theory analysis. "
        "Cross-reference with math.SE/MO discussions and primary sources "
        "(Fowler 2012, Avramidi 2015, Wall surgery theory) "
        "when possible.",
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
        "Problem 7 asks: Suppose Gamma is a uniform lattice in a real semi-simple "
        "group, and Gamma contains some 2-torsion. Is it possible for Gamma to be "
        "the fundamental group of a compact manifold without boundary whose universal "
        "cover is acyclic over Q?",
        "",
        "Current status in the writeup: conditional, not unconditional. "
        "The proof chain now establishes rational PD context and a nearby finite-CW "
        "result via Fowler's FH(Q) theorem, then isolates the remaining manifold-upgrade "
        "surgery obstruction step for the torsion lattice case.",
        "",
        "Key references: Fowler (arXiv:1204.4667), Avramidi (arXiv:1506.06293), "
        "Wall 'Surgery on Compact Manifolds' (1999 2nd ed), Brown and Luck on "
        "group/orbifold cohomology.",
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim in this proof step.",
        "2. Search math.SE/MO for relevant discussions (lattices in Lie groups, "
        "surgery theory, FH(Q), rational Poincare duality, "
        "Smith theory, L-groups, Wall surgery obstruction, orbifolds).",
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
        "Review this proof that a uniform lattice Gamma (with 2-torsion) in a "
        "semi-simple Lie group can be realized as pi_1(M) for a closed manifold M "
        "with rationally acyclic universal cover. Assess completeness, correctness, "
        "and suggest improvements.",
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
        "2. Are there unstated assumptions (e.g., about the specific Lie group G, "
        "the arithmetic construction of Gamma, and the orientation character for "
        "Poincare duality)?",
        "3. Verify the finite-complex/manifold distinction: does the argument only "
        "establish Gamma in FH(Q), or does it actually produce a closed manifold "
        "with pi_1=Gamma?",
        "4. The manifold-upgrade surgery step is the delicate part — verify what is "
        "still missing (normal map setup, obstruction computation, pi_1 control).",
        "5. The Smith theory argument (Section 7) could be challenged — is the "
        "distinction between mod-2 acyclicity and rational acyclicity correctly "
        "drawn? Could someone construct a counterexample using integral Smith "
        "theory?",
        "6. Check whether the concrete arithmetic examples in Fowler directly give "
        "order-2 torsion, or only nearby torsion cases.",
        "7. Is the confidence level calibrated to the remaining unresolved "
        "manifold-upgrade step?",
        "8. Search math.SE/MO for: 'uniform lattice 2-torsion manifold', "
        "'FH(Q) lattices', 'rational Poincare duality group surgery', "
        "'Smith theory rational acyclic'.",
        "9. For each reference, include `site` when known "
        "(e.g., `mathoverflow.net` or `math.stackexchange.com`).",
        "10. Reply as a single JSON object matching the required schema. "
        "Use node_id='p7-synthesis' for the synthesis.",
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
        print("Run: python3 scripts/proof7-wiring-diagram.py", file=sys.stderr)
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
        "node_id": "p7-synthesis",
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
