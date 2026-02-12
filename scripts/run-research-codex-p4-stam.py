#!/usr/bin/env python3
"""Research the finite free Stam inequality via Codex.

Strategy A: Finitize Voiculescu's (1998) proof of 1/Phi*(mu boxplus nu) >=
1/Phi*(mu) + 1/Phi*(nu) using Dyson Brownian Motion and Ito calculus on
eigenvalues.

Usage:
    # Generate prompts only (dry run)
    python3 scripts/run-research-codex-p4-stam.py --dry-run

    # Run through Codex
    python3 scripts/run-research-codex-p4-stam.py --limit 6

    # With custom math.SE search path
    python3 scripts/run-research-codex-p4-stam.py --math-se-dir se-data/math-processed/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "first-proof" / "problem4-stam-codex-results.jsonl"
DEFAULT_PROMPTS = REPO_ROOT / "data" / "first-proof" / "problem4-stam-codex-prompts.jsonl"
DEFAULT_MATH_SE_DIR = REPO_ROOT / "se-data" / "math-processed"


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "step_id": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["established", "plausible", "gap", "blocked", "needs_computation"],
        },
        "findings": {"type": "string"},
        "key_formulas": {
            "type": "array",
            "items": {"type": "string"},
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation": {"type": "string"},
                    "relevance": {"type": "string"},
                    "site": {
                        "type": "string",
                        "enum": ["mathoverflow.net", "math.stackexchange.com",
                                 "arxiv.org", "textbook", "other"],
                    },
                },
                "required": ["citation", "relevance", "site"],
                "additionalProperties": False,
            },
        },
        "next_steps": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["step_id", "status", "findings", "key_formulas",
                  "references", "next_steps", "confidence"],
    "additionalProperties": False,
}


# ── Research steps ──────────────────────────────────────────────────────

PREAMBLE = """\
You are a mathematical researcher working on finite free probability and
random matrix theory. You have expertise in Dyson Brownian motion, Ito
calculus for eigenvalue processes, Voiculescu's free entropy/Fisher information
theory, and the Marcus-Spielman-Srivastava finite free convolution.

## THE PROBLEM

For monic real-rooted polynomials p, q of degree n, with roots
lambda_1 < ... < lambda_n and mu_1 < ... < mu_n respectively, define:

    Phi_n(p) = sum_{i=1}^n (sum_{j != i} 1/(lambda_i - lambda_j))^2

This is the discrete free Fisher information (sum of squared Coulomb forces
in a 1D log-gas). The finite free additive convolution is:

    p ⊞_n q = E_U[det(xI - A - UBU*)]

where A = diag(lambda_i), B = diag(mu_j), U ~ Haar on U(n).

CONJECTURE (proved for n=2,3; verified numerically for n=4..8):

    1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q)

This is the FINITE-DIMENSIONAL ANALOG of the free Stam inequality
(Voiculescu 1998): 1/Phi*(mu ⊞ nu) >= 1/Phi*(mu) + 1/Phi*(nu).

## PROOF STRATEGY UNDER INVESTIGATION

Finitize Voiculescu's proof using Dyson Brownian Motion.

## KEY REFERENCES

- Voiculescu (1998), "Analogues of entropy and Fisher information, V,"
  Invent. Math. 132 — the free Stam inequality
- Anderson, Guionnet, Zeitouni (2010), "An Introduction to Random Matrices,"
  Ch. 4 — Dyson Brownian motion
- Biane (1997), "Free Brownian motion, free stochastic calculus and
  random matrices" — bridge between free and matrix
- Marcus, Spielman, Srivastava (2015), arXiv:1504.00350 — finite free
  convolution definition and real-rootedness
- Marcus (2021), arXiv:2108.07054 — finite free probability survey

## LOCAL MO/MSE REFERENCES (from library mining)

- MO 287724 + answer 287799: finite free convolution bilinearity/induction
- MO 114267, 228718, 256066, 288059, 248315, 419941: Weingarten/HCIZ toolbox
- MO 454139 + answers 454386, 454391, 454586: compression/scaling heuristics
"""

RESEARCH_STEPS = {
    "step-1-voiculescu-proof": {
        "title": "Reconstruct Voiculescu's proof of the free Stam inequality",
        "prompt": PREAMBLE + """
## YOUR TASK (Step 1 of 6)

Reconstruct the key steps of Voiculescu's proof of the free Stam inequality:

    1/Phi*(X + Y) >= 1/Phi*(X) + 1/Phi*(Y)   (X, Y freely independent)

Specifically:
1. How is Phi*(X) defined? (Hilbert transform squared, integrated against
   the distribution of X.)
2. What is the proof structure? Does it use:
   (a) Free Brownian motion X_t = X + S_t where S_t is semicircular of
       variance t?
   (b) The fact that d/dt[1/Phi*(X_t)] has a definite sign?
   (c) A heat equation / free entropy argument?
3. What is the key calculation that gives the inequality?
4. Where does the proof use free independence specifically?

Search MO/MSE for "Voiculescu free Stam inequality proof", "free Fisher
information convolution", "free entropy power inequality proof technique".

IMPORTANT: We need the actual proof mechanism, not just the statement.
The goal is to identify which steps can be finitized.
""",
    },
    "step-2-dyson-process": {
        "title": "Dyson Brownian motion and Ito calculus for Phi_n",
        "prompt": PREAMBLE + """
## YOUR TASK (Step 2 of 6)

Set up the Ito calculus for Phi_n along Dyson Brownian motion.

The eigenvalue process of H_t = A + sqrt(t) * GUE is:

    d(lambda_i) = dB_i / sqrt(n) + (1/n) sum_{j != i} dt / (lambda_i - lambda_j)

where B_i are independent standard Brownian motions and the second term
is the eigenvalue repulsion.

1. Write Phi_n(lambda_1,...,lambda_n) = sum_i (sum_{j!=i} 1/(lambda_i - lambda_j))^2
   as a function of the eigenvalues.

2. Apply Ito's formula to compute d[1/Phi_n] along the Dyson process.
   You will need:
   - Partial derivatives of 1/Phi_n with respect to each lambda_i
   - Second partial derivatives (for the Ito correction)
   - The drift and diffusion coefficients from the Dyson SDE

3. Identify whether d[1/Phi_n] or d[E[1/Phi_n]] has a definite sign.

4. If the Ito formula gives E[1/Phi_n(H_t)] = 1/Phi_n(A) + integral of
   something, determine whether the integrand is non-negative.

Search MO/MSE for "Ito formula eigenvalue process", "Dyson Brownian motion
functional", "eigenvalue diffusion energy functional".

Note: This is the HARDEST step. The partial derivatives of 1/Phi_n are
rational functions of the gaps (lambda_i - lambda_j) and computing the
Ito correction is nontrivial. Focus on identifying the structure rather
than completing every calculation.
""",
    },
    "step-3-connection": {
        "title": "Connect Dyson process at t=0 to the ⊞_n convolution",
        "prompt": PREAMBLE + """
## YOUR TASK (Step 3 of 6)

Connect the Dyson Brownian motion to the finite free convolution ⊞_n.

The finite free convolution is:
    p ⊞_n q = E_U[det(xI - A - UBU*)]     (U ~ Haar on U(n))

The Dyson process gives:
    eigenvalues of A + sqrt(t) * GUE       (continuous-time process)

These are related but different:
- ⊞_n adds a FIXED matrix B conjugated by Haar unitary
- Dyson adds a GUE perturbation (Gaussian, not fixed spectrum)

Questions to investigate:
1. Can we write A + UBU* as a stopped Dyson process? Specifically, is
   there a time t* and conditioning such that the eigenvalue distribution
   of A + UBU* at time t* matches what we need?

2. Alternatively: the HCIZ (Harish-Chandra-Itzykson-Zuber) integral gives
   the density of eigenvalues of A + UBU*. Can we use this density to
   compute E[1/Phi_n(A + UBU*)] directly?

3. A third possibility: use the subordination approach (Biane 1998). The
   free convolution mu ⊞ nu is computed via subordination functions
   omega_1, omega_2. Is there a finite-n analog?

4. What is the relationship between E_U[1/Phi_n(A + UBU*)] and
   1/Phi_n(E_U[det(xI - A - UBU*)])? The former is the expected
   reciprocal energy; the latter is the reciprocal energy of the expected
   polynomial (which IS 1/Phi_n(p ⊞_n q)). These are NOT the same
   by Jensen's inequality. Which direction does Jensen go?

Search MO for "finite free convolution Haar unitary eigenvalue distribution",
"HCIZ integral eigenvalue density", "subordination finite dimensional".

MO references to check: 114267, 228718, 248315, 419941 (HCIZ/Weingarten).
""",
    },
    "step-4-jensen-gap": {
        "title": "Handle the Jensen gap: E[1/Phi_n] vs 1/Phi_n(E[poly])",
        "prompt": PREAMBLE + """
## YOUR TASK (Step 4 of 6)

Address the fundamental obstacle: the inequality we want is about
1/Phi_n of the EXPECTED POLYNOMIAL, not the EXPECTED 1/Phi_n.

Let p ⊞_n q = E_U[det(xI - A - UBU*)]. Then:

    1/Phi_n(p ⊞_n q) = 1/Phi_n(E_U[polynomial])

We want: 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q).

But the natural quantity from the Dyson/Ito approach would be:

    E_U[1/Phi_n(eigenvalues of A + UBU*)]

These differ by a Jensen-type gap.

Questions:
1. Is 1/Phi_n CONVEX or CONCAVE as a function of polynomial coefficients?
   If convex: E[1/Phi_n(random poly)] >= 1/Phi_n(E[random poly]), so
   the Dyson approach gives something STRONGER than we need.
   If concave: the Jensen gap goes the WRONG way.

2. Alternatively, can we bypass the Jensen issue entirely? Voiculescu's
   proof works at the MEASURE level, not the matrix level. Maybe the
   finite analog should work with the EXPECTED POLYNOMIAL directly
   (via its coefficients or cumulants) rather than with random matrices.

3. At n=3, our proof works with coefficients directly (no random matrices
   needed). The surplus is (3/2)[u^2/s^2 + v^2/t^2 - (u+v)^2/(s+t)^2].
   Can this algebraic approach be generalized to n=4 via the MSS
   coefficient formula c_k = sum w(n,i,j) a_i b_j?

4. Is 1/Phi_n a convex function of the FREE CUMULANTS (not the coefficients)?
   The finite free cumulants add under ⊞_n, so if 1/Phi_n is convex in
   cumulants with 1/Phi_n(0) = 0, then superadditivity follows immediately.
   [But note: concave + f(0)=0 gives SUBadditivity. Need CONVEXITY.]

This step is CRITICAL. The entire proof strategy depends on resolving the
direction of the Jensen gap.

Search MO for "convexity reciprocal Coulomb energy", "free cumulant convexity",
"expected polynomial vs polynomial of expectation".
""",
    },
    "step-5-finite-stam": {
        "title": "Attempt the finite-n Stam inequality proof",
        "prompt": PREAMBLE + """
## YOUR TASK (Step 5 of 6)

Based on the findings from Steps 1-4, attempt to assemble a proof of:

    1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q)

for all n. You may assume the following established facts:

PROVED:
- n=2: equality (1/Phi_2 = disc/2, linear in coefficients)
- n=3: strict inequality via Phi_3*disc = 18*a_2^2 + Titu's lemma
- ⊞_n preserves real-rootedness (MSS 2015)
- ⊞_n is defined via E_U[det(xI - A - UBU*)] (MSS 2015)
- Centering reduction: WLOG a_1 = b_1 = 0 (translation invariance)
- The free Stam inequality holds in the n→∞ limit (Voiculescu 1998)

NUMERICALLY VERIFIED:
- 0 genuine violations in 35K+ tests, n=4..8
- Adversarial infimum = 1.0 for all n (via scale separation)

KEY STRUCTURAL FACTS:
- MSS coefficient formula: c_k = sum_{i+j=k} w(n,i,j) a_i b_j
  where w(n,i,j) = (n-i)!(n-j)! / (n!(n-k)!)
- For n>=4, ⊞_n has essential cross-terms even centered
  (e.g. c_4 = a_4 + (1/6)a_2*b_2 + b_4 at n=4)
- Phi_n*disc is NOT constant for n>=4

Attempt one of:
(a) A Dyson Brownian motion / Ito calculus argument
(b) A convexity-in-cumulants argument
(c) An induction via differentiation from n=3
(d) A direct algebraic argument for n=4

If you cannot complete the proof, identify precisely where it breaks down
and what additional input would be needed.
""",
    },
    "step-6-synthesis": {
        "title": "Synthesize findings and assess proof viability",
        "prompt": PREAMBLE + """
## YOUR TASK (Step 6 of 6)

Synthesize the research from Steps 1-5. Provide:

1. **Proof viability assessment**: What is the probability of a complete
   proof for n>=4 given what we've found?

2. **Most promising route**: Which of the three strategies (Dyson/Ito,
   induction/differentiation, direct algebraic) is most likely to work?

3. **Precise blockers**: What are the specific mathematical obstacles
   remaining? For each, state:
   - The exact claim that needs to be established
   - Whether it's likely true (based on numerics and heuristics)
   - What tools/techniques might establish it

4. **Fallback positions**: If a full proof for all n is not achievable,
   what partial results are within reach? Examples:
   - Proof for n=4 only (via computer algebra)
   - Proof conditional on a specific conjecture
   - Proof for n large enough (asymptotic)

5. **Connection to known results**: Map our inequality to the existing
   literature. Is this inequality already known under a different name?
   Is it implied by any known result? Is it a special case of something
   more general?

6. **Recommended next actions**: What should the next research cycle
   focus on?

Be honest about gaps. A well-characterized open problem is more valuable
than a claimed proof with hidden gaps (we learned this lesson from the
P2/P7/P8 review cycle — see checkpoint-verification-dynamics.md).
""",
    },
}


def build_prompt(step_id: str, step_data: dict, math_se_dir: Path | None = None) -> str:
    """Build a research prompt for one step."""
    lines = [step_data["prompt"]]
    if math_se_dir:
        lines.extend([
            "",
            "## Local Corpus Hint",
            "",
            f"Use local processed data if available under: {math_se_dir}",
        ])
    lines.extend([
        "",
        "## Instructions",
        "",
        "Reply as a single JSON object matching the required schema.",
        "For key_formulas, include the most important mathematical formulas",
        "or identities you found or derived (as LaTeX strings).",
        "Be precise about what is established vs. conjectured vs. open.",
    ])
    return "\n".join(lines)


def run_codex_once(
    codex_bin: str,
    model: str,
    cwd: Path,
    schema_path: Path,
    prompt_text: str,
    timeout_sec: int,
    codex_home: Path | None,
    retries: int,
) -> tuple[int, str, str]:
    """Run a single prompt through codex exec."""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as out_f:
        out_path = Path(out_f.name)

    instruction = (
        "You must answer exactly as one JSON object matching the required schema. "
        "Do not wrap JSON in markdown fences. Do not add extra commentary. "
        "Do not perform shell/tool calls or emit progress updates; return only final JSON.\n\n"
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
    env = os.environ.copy()
    if codex_home is not None:
        env["CODEX_HOME"] = str(codex_home)
    attempts = max(1, retries + 1)
    rc = 1
    stderr_text = ""
    stdout_text = ""
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(
                cmd,
                input=instruction,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                env=env,
            )
            rc = proc.returncode
            stderr_text = proc.stderr.strip()
            stdout_text = proc.stdout.strip()
        except subprocess.TimeoutExpired as e:
            rc = 124
            stderr_text = f"codex exec timed out after {timeout_sec}s"
            stdout_text = (e.stdout or "").strip()

        if attempt >= attempts or not _is_transient_codex_error(rc, stderr_text):
            break
        sleep_s = min(8, 2 ** (attempt - 1))
        time.sleep(sleep_s)

    try:
        response_text = out_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        response_text = ""
    out_path.unlink(missing_ok=True)
    if not response_text and stdout_text:
        response_text = stdout_text
    return rc, response_text, stderr_text


def _is_transient_codex_error(rc: int, stderr_text: str) -> bool:
    if rc == 124:
        return True
    lowered = stderr_text.lower()
    transient_tokens = [
        "stream disconnected before completion",
        "error sending request for url",
        "network error",
        "reconnecting...",
        "timed out",
        "429",
        "rate limit",
        "503",
    ]
    return any(tok in lowered for tok in transient_tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--prompts-out", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max prompts to process (default: all)")
    ap.add_argument("--model", default="gpt-5.3-codex")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate prompts only, don't call Codex")
    ap.add_argument("--math-se-dir", type=Path, default=DEFAULT_MATH_SE_DIR)
    ap.add_argument("--timeout-sec", type=int, default=90,
                    help="Timeout per Codex call in seconds")
    ap.add_argument("--retries", type=int, default=0,
                    help="Retries per Codex call on transient network/timeouts")
    ap.add_argument("--codex-home", type=Path, default=None,
                    help="Optional writable CODEX_HOME used for codex exec session files")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args()

    # Build prompts
    prompts = []
    for step_id, step_data in RESEARCH_STEPS.items():
        prompt_text = build_prompt(step_id, step_data, math_se_dir=args.math_se_dir)
        prompts.append({
            "step_id": step_id,
            "title": step_data["title"],
            "prompt": prompt_text,
        })

    run_limit = len(prompts) if args.limit is None else min(args.limit, len(prompts))

    # Write prompts
    args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.prompts_out, "w") as f:
        for rec in prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} prompts to {args.prompts_out}", flush=True)

    if args.dry_run:
        print("Dry run -- not calling Codex.")
        print(f"\nPrompt summary ({len(prompts)} prompts, run_limit={run_limit}):")
        for p in prompts:
            lines = p["prompt"].count("\n") + 1
            print(f"  {p['step_id']:30s} [{p['title'][:50]:50s}] ~{lines} lines")
        return 0

    # Run through Codex
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(RESPONSE_SCHEMA, sf, ensure_ascii=True, indent=2)
        schema_path = Path(sf.name)

    try:
        with open(args.output, "w") as fout:
            for rec in prompts[:run_limit]:
                print(f"Running {rec['step_id']}...", flush=True)
                step_started = time.time()
                rc, raw_response, stderr_text = run_codex_once(
                    codex_bin=args.codex_bin,
                    model=args.model,
                    cwd=args.repo_root,
                    schema_path=schema_path,
                    prompt_text=rec["prompt"],
                    timeout_sec=args.timeout_sec,
                    codex_home=args.codex_home,
                    retries=args.retries,
                )

                out = {"step_id": rec["step_id"]}
                try:
                    parsed = json.loads(raw_response)
                    if isinstance(parsed, dict):
                        out.update(parsed)
                        # Never trust model-emitted IDs; preserve the run record key.
                        out["step_id"] = rec["step_id"]
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
                status = out.get("status", "parse_error" if out.get("parse_error") else "?")
                elapsed = time.time() - step_started
                print(
                    f"[{processed:02d}/{run_limit}] {rec['step_id']:30s} "
                    f"-> {status} (rc={rc}, {elapsed:.1f}s)"
                )
                sys.stdout.flush()
    finally:
        schema_path.unlink(missing_ok=True)

    print(f"\n--- SUMMARY ---")
    print(f"processed={processed}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
