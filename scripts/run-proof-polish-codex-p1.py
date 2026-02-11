#!/usr/bin/env python3
"""Polish the Problem 1 proof via Codex CLI.

For each proof node in the wiring diagram, generates a verification prompt
and runs it through `codex exec`. Codex cross-references with math.SE data
(when available), verifies mathematical claims, and identifies gaps.

Usage:
    # Generate prompts only (dry run)
    python3 scripts/run-proof-polish-codex-p1.py --dry-run

    # Run through Codex (needs math.SE data in se-data/math-processed/)
    python3 scripts/run-proof-polish-codex-p1.py --limit 9

    # With custom math.SE search path
    python3 scripts/run-proof-polish-codex-p1.py --math-se-dir se-data/math-processed/
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
WIRING_JSON = REPO_ROOT / "data" / "first-proof" / "problem1-wiring.json"
SOLUTION_MD = REPO_ROOT / "data" / "first-proof" / "problem1-solution.md"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem1-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem1-codex-prompts.jsonl"
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
    "p1-problem": (
        "Verify the problem is well-posed: check that the Phi^4_3 measure mu "
        "on D'(T^3) is a well-defined probability measure, that T_psi(u) = u + psi "
        "is a measurable shift for smooth psi, and that the question of equivalence "
        "(mu ~ T_psi^* mu, i.e., same null sets) is the correct notion. Confirm "
        "the distinction between equivalence and absolute continuity. Check whether "
        "'smooth nonzero psi' should be 'smooth' or 'smooth and compactly supported' "
        "(on the torus T^3 all smooth functions are acceptable)."
    ),
    "p1-s1": (
        "Verify the construction references: (a) Hairer 2014 (Inventiones) uses "
        "regularity structures to construct the dynamic Phi^4_3 — does the invariant "
        "measure coincide with the Phi^4_3 measure defined here? (b) Gubinelli-Imkeller-"
        "Perkowski 2015 (GIP) use paracontrolled distributions — confirm the static "
        "measure construction follows. (c) Barashkov-Gubinelli 2020 use the variational "
        "(Boue-Dupuis) approach — verify this gives the same measure. Check that the "
        "Wick ordering :phi^4: - C:phi^2: is stated with the correct sign and that "
        "the mass counterterm C is identified as the one that diverges logarithmically "
        "in the UV cutoff."
    ),
    "p1-s2": (
        "Check that exp(-V) > 0 a.s. and integrability E_{mu_0}[exp(-V)] < infinity "
        "are sufficient for equivalence mu ~ mu_0. The argument is: (1) exp(-V) > 0 "
        "a.s. because the exponential function is strictly positive, so mu is absolutely "
        "continuous w.r.t. mu_0, AND mu_0 is absolutely continuous w.r.t. mu (since "
        "1/exp(-V) = exp(V) is finite mu-a.s. if V < infinity mu-a.s.). Verify that "
        "V < infinity mu_0-a.s. is actually established in the construction — this "
        "requires that the renormalized :phi^4: integral is finite a.s. under the GFF. "
        "Search math.SE/MO for discussions of absolute continuity of Gibbs measures "
        "with respect to Gaussian reference measures."
    ),
    "p1-s3": (
        "Verify that the Cameron-Martin space of the GFF mu_0 with covariance "
        "(m^2 - Delta)^{-1} on T^3 is indeed H^1(T^3). More precisely, the CM space "
        "should be the range of C^{1/2} = (m^2 - Delta)^{-1/2}, which is H^1(T^3) "
        "(Sobolev space of order 1). Check: psi smooth implies psi in H^1(T^3), which "
        "is immediate since C^infinity(T^3) is dense in H^1(T^3). Verify the explicit "
        "Radon-Nikodym derivative formula: dT_psi^* mu_0 / dmu_0 (phi) = "
        "exp(l_psi(phi) - ||psi||_H^2 / 2). Search for Cameron-Martin theorem for "
        "Gaussian measures on Sobolev spaces / distribution spaces."
    ),
    "p1-s4": (
        "The expansion of V(phi - psi) in Wick powers is the technical heart. Check: "
        "(a) the binomial expansion of :( phi - psi )^4: produces terms "
        "4 psi :phi^3:, 6 psi^2 :phi^2:, 4 psi^3 :phi:, psi^4 — verify signs and "
        "combinatorial coefficients; (b) the Wick ordering is relative to mu_0, so "
        ":(phi-psi)^4:_{mu_0} != :phi^4:_{mu_0} - 4 psi :phi^3:_{mu_0} + ... — "
        "there are renormalization corrections from re-Wick-ordering; (c) the term "
        "6 psi^2 :phi^2: generates an additional UV divergence that must be absorbed "
        "into the mass counterterm. Verify that after this re-renormalization, "
        "V(phi-psi) - V(phi) is a well-defined random variable. This is the step "
        "most likely to contain a subtle error."
    ),
    "p1-s5": (
        "Exponential integrability of the cubic perturbation 4 int psi :phi^3: dx "
        "under the Phi^4_3 measure is the hardest analytical claim. Verify: "
        "(a) the log-Sobolev inequality for Phi^4_3 (Barashkov-Gubinelli 2020) — "
        "does it actually give exponential integrability of :phi^3: tested against "
        "smooth functions? (b) the 'quartic coercivity dominates cubic' heuristic — "
        "is this made rigorous via a Brascamp-Lieb or log-Sobolev argument? "
        "(c) E_mu[exp(t |int psi :phi^3: dx|)] < infinity for ALL t — this is a "
        "strong claim (Gaussian-type tail). Is this stated in the literature, or "
        "only for small t? If only for small t, the equivalence argument may need "
        "modification. Search math.SE/MO for: exponential integrability Phi^4, "
        "log-Sobolev inequality Euclidean field theory, concentration for Gibbs measures."
    ),
    "p1-s6": (
        "The Boue-Dupuis variational formula argument: Barashkov-Gubinelli 2020 "
        "construct Phi^4_3 via -log Z = inf_u E[V(phi + I(u)) + 1/2 int ||u_s||^2 ds]. "
        "Verify: (a) does shifting phi -> phi - psi in the variational problem produce "
        "an equivalent measure? (b) is 'the infimum shifts by a finite amount' a "
        "rigorous statement — what precisely changes in the variational problem? "
        "(c) does this alternative argument avoid the need for explicit exponential "
        "integrability estimates, or does it implicitly require them? Check whether "
        "this is truly an independent argument or just a repackaging of the same "
        "analytical ingredients."
    ),
    "p1-s7": (
        "Verify the chain of equivalences composes correctly: mu ~ mu_0 (from s2), "
        "T_psi^* mu_0 ~ mu_0 (from s3, Cameron-Martin), and then T_psi^* mu ~ mu "
        "requires more than transitivity — it requires that the interacting density "
        "exp(-V) behaves well under the shift. Check: (a) the argument is NOT simply "
        "'mu ~ mu_0 and T_psi^* mu_0 ~ mu_0 therefore T_psi^* mu ~ mu' — this would "
        "be wrong because T_psi^* mu != T_psi^* mu_0. (b) The correct argument must "
        "show dT_psi^* mu / dmu is well-defined, positive, and integrable. (c) Verify "
        "that steps s4 and s5 supply exactly what is needed for this RN derivative. "
        "(d) Are there parameter constraints (m^2 > 0, psi in C^infinity) that should "
        "be stated explicitly?"
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
        "You are a mathematical proof verifier with expertise in constructive "
        "QFT, stochastic PDEs, Gaussian measures, Cameron-Martin theory, "
        "renormalization.",
        "",
        "## Task",
        "",
        "Verify one step of a proof that the Phi^4_3 measure mu on the 3D torus "
        "is equivalent to its translate T_psi^* mu under any smooth shift psi. "
        "The argument combines Cameron-Martin theory for the GFF with "
        "renormalization analysis of the shifted interaction V(phi - psi). "
        "Cross-reference with math.SE/MO discussions and the primary sources "
        "(Hairer 2014, GIP 2015, Barashkov-Gubinelli 2020) when possible.",
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
        "Problem 1 asks: Let mu be the Phi^4_3 measure on D'(T^3) and psi a smooth "
        "nonzero function on T^3. Are mu and T_psi^* mu equivalent (same null sets)?",
        "",
        "The answer is: Yes. The proof proceeds by: (1) establishing mu ~ mu_0 (the "
        "Phi^4_3 measure is equivalent to the GFF because exp(-V) > 0 a.s.); "
        "(2) applying Cameron-Martin theory (psi smooth => psi in H^1 = CM space, "
        "so T_psi^* mu_0 ~ mu_0); (3) analyzing the shifted interaction V(phi - psi) "
        "via Wick power expansion and re-renormalization; (4) proving exponential "
        "integrability of the cubic perturbation using log-Sobolev / quartic coercivity; "
        "(5) composing these to get T_psi^* mu ~ mu.",
        "",
        "Key references: Hairer 2014 (regularity structures), Gubinelli-Imkeller-"
        "Perkowski 2015 (paracontrolled distributions), Barashkov-Gubinelli 2020 "
        "(variational / Boue-Dupuis approach).",
        "",
        "## Instructions",
        "",
        "1. Verify the mathematical claim in this proof step.",
        "2. Search math.SE/MO for relevant discussions (Phi^4_3 measure, Gaussian "
        "free field, Cameron-Martin theorem, Wick ordering, renormalization in "
        "constructive QFT, log-Sobolev inequalities for Gibbs measures, "
        "Boue-Dupuis variational formula, absolute continuity of shifted measures).",
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
        "Review this proof that the Phi^4_3 measure mu on D'(T^3) is equivalent "
        "to its translate T_psi^* mu under any smooth shift psi. Assess completeness, "
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
        "1. Is the proof complete? Does the equivalence mu ~ T_psi^* mu follow "
        "from the stated steps?",
        "2. Are there unstated assumptions (e.g., about the mass parameter m^2 > 0, "
        "smoothness vs. H^1 regularity of psi, the specific form of the "
        "renormalization counterterm, the dimension d=3 being critical)?",
        "3. The exponential integrability claim (Section 5) is the analytical heart — "
        "is E_mu[exp(t |int psi :phi^3: dx|)] < infinity for ALL t truly justified, "
        "or only for small t? If only small t, does the argument still close?",
        "4. The chain of equivalences (Section 7) is a potential logical weak point — "
        "verify it is not simply 'mu ~ mu_0 and T_psi^* mu_0 ~ mu_0 therefore "
        "T_psi^* mu ~ mu' (which would be incorrect). The correct argument requires "
        "the shifted RN derivative to be positive and integrable.",
        "5. The Boue-Dupuis alternative (Section 6) — is this truly independent, or "
        "does it implicitly require the same exponential integrability estimates?",
        "6. Search math.SE/MO for: Phi^4_3 measure equivalence under shifts, "
        "Cameron-Martin for non-Gaussian measures, quasi-invariance of Gibbs measures, "
        "renormalization under translation, log-Sobolev for Phi^4_3.",
        "7. Suggest improvements: tighter statements, missing references, cleaner "
        "argument structure.",
        "8. For each reference, include `site` when known "
        "(e.g., `mathoverflow.net` or `math.stackexchange.com`).",
        "9. Reply as a single JSON object matching the required schema. "
        "Use node_id='p1-synthesis' for the synthesis.",
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
        print("Run: python3 scripts/proof1-wiring-diagram.py", file=sys.stderr)
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
        "node_id": "p1-synthesis",
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
