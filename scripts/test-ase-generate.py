#!/usr/bin/env python3
"""Test the ASE generation pipeline end-to-end without Tickle.

Takes a question (or uses a built-in curriculum question), pulls retrieval
context from the corpus, builds a generation prompt, and pipes it to claude CLI.

Usage:
    # Test with built-in P7 curriculum question #1
    python3 scripts/test-ase-generate.py --question 1

    # Test all 5 curriculum questions
    python3 scripts/test-ase-generate.py --question all

    # Custom question
    python3 scripts/test-ase-generate.py --custom "What is the exact obstruction..."

    # Prompt-only (no API call)
    python3 scripts/test-ase-generate.py --question 1 --prompt-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Codex's 5-question curriculum gate for p7-problem
P7_CURRICULUM = {
    1: {
        "title": "Quantifier lock",
        "question": (
            "Are we proving 'for this fixed Γ' or 'there exists some such Γ' "
            "(or 'for all such Γ')? Specifically: let G be a semi-simple Lie group "
            "with associated symmetric space X = G/K. Let Γ ⊂ G be a uniform lattice "
            "containing elements of order 2. The problem asks whether there exists a "
            "closed aspherical manifold M with π₁(M) ≅ Γ.\n\n"
            "Clarify: is the quantifier over Γ universal, existential, or is Γ "
            "fixed at the outset? What changes if we shift quantifiers?"
        ),
        "node_id": "p7-problem",
        "tags": ["lie-groups", "lattice", "algebraic-topology", "manifolds"],
    },
    2: {
        "title": "Implication vs equivalence lock",
        "question": (
            "The proof uses Q-acyclicity of M to conclude H_*(M;Q) ≅ H_*(Γ;Q). "
            "Is this only an implication (Q-acyclic M ⟹ rational cohomology match) "
            "or is a two-way equivalence being claimed?\n\n"
            "If two-way: what extra hypothesis makes the reverse direction valid? "
            "Is it the asphericity of M, or the Eilenberg-MacLane property, or "
            "something else entirely?"
        ),
        "node_id": "p7-problem",
        "tags": ["algebraic-topology", "homological-algebra", "group-cohomology"],
    },
    3: {
        "title": "Trivial-case boundary",
        "question": (
            "The torsion-free case is 'trivially' positive: if Γ is torsion-free, "
            "then X/Γ is already a closed aspherical manifold with π₁ ≅ Γ.\n\n"
            "Clarify: why does 'Γ has no 2-torsion (but may have odd-order torsion)' "
            "NOT reduce to the same trivial case? What specifically about 2-torsion "
            "(vs odd torsion) breaks the X/Γ quotient argument? Is it the existence "
            "of fixed points under involutions (Smith theory), or something else?"
        ),
        "node_id": "p7-problem",
        "tags": ["lie-groups", "algebraic-topology", "group-theory"],
    },
    4: {
        "title": "Obstruction stack declaration",
        "question": (
            "For the 'conditional yes' path (FH(Q) + surgery upgrade), list the "
            "exact required conditions:\n\n"
            "1. What dimension/category assumptions are needed?\n"
            "2. What is the named obstruction that must vanish (Wall surgery "
            "obstruction in L-groups? A specific element of L_n(ZΓ)?)?\n"
            "3. What role does the Farrell-Jones conjecture play — is it a "
            "hypothesis or a proved ingredient for this class of groups?\n"
            "4. Does the surgery sequence split, or is there a non-trivial "
            "extension problem?"
        ),
        "node_id": "p7-s4",
        "tags": ["algebraic-topology", "manifolds", "surgery-theory"],
    },
    5: {
        "title": "2-torsion mechanism question",
        "question": (
            "Where exactly does 2-torsion obstruct manifold realization?\n\n"
            "Is it:\n"
            "(a) A fixed-point/Smith-type condition — involutions on X have "
            "non-empty fixed point sets, so X/Γ is an orbifold not a manifold?\n"
            "(b) An orbifold fundamental group issue — π₁^orb ≠ Γ?\n"
            "(c) A surgery obstruction compatibility issue — the L-group element "
            "is non-trivial precisely because of 2-torsion contributions?\n\n"
            "Name the concrete theorem or citation that identifies this blocker. "
            "If it is Smith theory, state the precise Smith-theoretic result that "
            "applies (mod-2 vs rational, dimension of fixed set, etc.)."
        ),
        "node_id": "p7-s5",
        "tags": ["algebraic-topology", "group-theory", "surgery-theory"],
    },
}


def load_corpus_context(problem: int) -> dict[str, str]:
    """Load pre-retrieved corpus context for a problem."""
    path = REPO_ROOT / "data" / "first-proof" / f"problem{problem}-corpus-context.jsonl"
    if not path.exists():
        return {}
    context = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            context[rec["node_id"]] = rec["prompt_context"]
    return context


def load_wiring_node(problem: int, node_id: str) -> dict | None:
    """Load a specific node from a wiring diagram."""
    path = REPO_ROOT / "data" / "first-proof" / f"problem{problem}-wiring.json"
    if not path.exists():
        return None
    with path.open() as f:
        wiring = json.load(f)
    for n in wiring["nodes"]:
        if n["id"] == node_id:
            return n
    return None


def build_prompt(q: dict, corpus_ctx: str | None, wiring_node: dict | None) -> str:
    """Build a full generation prompt for a curriculum question."""
    lines = [
        "You are a mathematics expert generating a rigorous Stack Exchange-style "
        "answer to the following question about Frontier Math proof structure.",
        "",
        f"## Question: {q['title']}",
        "",
        q["question"],
        "",
    ]

    if wiring_node:
        lines.extend([
            "## Proof Context (from wiring diagram)",
            "",
            f"**Node**: {wiring_node['id']}",
            f"**Type**: {wiring_node.get('node_type', '?')}",
            f"**Claim**: {wiring_node.get('body_text', '')}",
            "",
        ])

    if corpus_ctx:
        lines.extend([
            corpus_ctx,
            "",
        ])

    lines.extend([
        "## Output Format",
        "",
        "Respond with a JSON object containing:",
        "- `thread_id`: string like `ase-p7-curriculum-q1`",
        "- `title`: concise question title",
        "- `question`: the full question (restate clearly, include LaTeX)",
        "- `answer`: rigorous answer with mathematical precision",
        "- `tags`: list of math tags",
        "- `nodes`: hypergraph nodes (typed: post, term, expression, scope)",
        "  - post nodes: `{id, type: 'post', subtype: 'question'|'answer'}`",
        "  - term nodes: `{id, type: 'term', subtype: '<concept>'}`",
        "  - expression nodes: `{id, type: 'expression', subtype: 'latex'}`",
        "  - scope nodes: `{id, type: 'scope', subtype: 'quant/universal'|'quant/existential'|'conditional'}`",
        "- `edges`: typed edges connecting nodes",
        "  - `{type: 'mention', ends: [post_id, term_id]}`",
        "  - `{type: 'surface', ends: [post_id, expr_id]}`",
        "  - `{type: 'scope', ends: [post_id, scope_id]}`",
        "  - `{type: 'discourse', ends: [question_id, answer_id]}`",
        "  - `{type: 'iatc', ends: [src, tgt], attrs: {performative: 'assert'|'clarify'|'query'}}`",
        "",
        "Aim for 4-8 term nodes, 2-4 expression nodes, 1-3 scope nodes.",
        "Every node should be connected by at least one edge.",
        "Reply with ONLY the JSON object.",
    ])

    return "\n".join(lines)


def call_llm(prompt: str, timeout: int = 120, backend: str = "claude") -> str | None:
    """Call an LLM via CLI and return the response.

    Backends:
        claude — claude -p (Opus 4.6)
        codex  — codex -q (Codex 5.3)
    """
    # Strip CLAUDECODE env var so claude -p doesn't refuse to nest
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    if backend == "claude":
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
    elif backend == "codex":
        cmd = ["codex", "-q", prompt]
    else:
        print(f"  Unknown backend: {backend}", file=sys.stderr)
        return None

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        if result.returncode != 0:
            print(f"  {backend} error: {result.stderr[:300]}", file=sys.stderr)
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"  {backend} timed out after {timeout}s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"  {backend} CLI not found", file=sys.stderr)
        return None


def fix_latex_escapes(text: str) -> str:
    """Fix unescaped LaTeX backslashes in JSON strings.

    LLMs often produce JSON with raw LaTeX like \\Gamma instead of \\\\Gamma.
    This fixes the most common cases without breaking valid JSON escapes.
    """
    import re
    # In JSON, valid escapes after backslash are: " \\ / b f n r t u
    # Anything else is invalid — double the backslash
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def validate_response(response: str) -> dict | None:
    """Try to parse the response as JSON and validate basic structure."""
    # Strip markdown code fences if present
    text = response
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]

    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        # Try fixing LaTeX escapes
        try:
            obj = json.loads(fix_latex_escapes(text.strip()))
            print("  (fixed LaTeX escapes in JSON)")
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return None

    # Basic validation
    required = ["thread_id", "title", "question", "answer"]
    missing = [k for k in required if k not in obj]
    if missing:
        print(f"  Missing required fields: {missing}")
        return None

    nodes = obj.get("nodes", [])
    edges = obj.get("edges", [])
    print(f"  Parsed: {len(nodes)} nodes, {len(edges)} edges")

    # Type distribution
    type_counts = {}
    for n in nodes:
        t = n.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Node types: {type_counts}")

    edge_types = {}
    for e in edges:
        t = e.get("type", "?")
        edge_types[t] = edge_types.get(t, 0) + 1
    print(f"  Edge types: {edge_types}")

    return obj


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--question", default="1",
                        help="Question number (1-5) or 'all'")
    parser.add_argument("--custom", type=str, default=None,
                        help="Custom question text (overrides --question)")
    parser.add_argument("--problem", type=int, default=7)
    parser.add_argument("--prompt-only", action="store_true",
                        help="Print prompt without calling API")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file for results (default: stdout + data/ase-test/)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--backend", choices=["claude", "codex"], default="claude",
                        help="LLM backend: claude (Opus 4.6) or codex (Codex 5.3)")
    args = parser.parse_args()

    # Load context
    corpus_context = load_corpus_context(args.problem)
    if corpus_context:
        print(f"Loaded corpus context: {len(corpus_context)} nodes")
    else:
        print("No corpus context available (proceeding without)")

    # Determine questions to run
    if args.custom:
        questions = [{
            "title": "Custom question",
            "question": args.custom,
            "node_id": f"p{args.problem}-problem",
            "tags": [],
        }]
        question_ids = ["custom"]
    elif args.question == "all":
        questions = [P7_CURRICULUM[i] for i in range(1, 6)]
        question_ids = [str(i) for i in range(1, 6)]
    else:
        qnum = int(args.question)
        if qnum not in P7_CURRICULUM:
            print(f"Question {qnum} not found (valid: 1-5)")
            return 1
        questions = [P7_CURRICULUM[qnum]]
        question_ids = [str(qnum)]

    results = []
    for qid, q in zip(question_ids, questions):
        print(f"\n{'='*70}")
        print(f"Question {qid}: {q['title']}")
        print(f"{'='*70}")

        ctx = corpus_context.get(q["node_id"])
        wiring_node = load_wiring_node(args.problem, q["node_id"])
        prompt = build_prompt(q, ctx, wiring_node)

        print(f"Prompt: {len(prompt)} chars, {prompt.count(chr(10))+1} lines")

        if args.prompt_only:
            print(f"\n--- PROMPT ---\n{prompt}\n--- END ---")
            continue

        print(f"Calling {args.backend}...")
        t0 = time.time()
        response = call_llm(prompt, timeout=args.timeout, backend=args.backend)
        elapsed = time.time() - t0

        if response is None:
            print(f"  Failed ({elapsed:.1f}s)")
            results.append({"question_id": qid, "ok": False, "elapsed": elapsed})
            continue

        print(f"  Response: {len(response)} chars ({elapsed:.1f}s)")
        obj = validate_response(response)

        result = {
            "question_id": qid,
            "question_title": q["title"],
            "ok": obj is not None,
            "elapsed": elapsed,
            "response_length": len(response),
        }
        if obj:
            result["thread_id"] = obj.get("thread_id")
            result["n_nodes"] = len(obj.get("nodes", []))
            result["n_edges"] = len(obj.get("edges", []))
            result["answer_preview"] = obj.get("answer", "")[:200]

        results.append(result)

        # Save individual result
        out_dir = args.output or (REPO_ROOT / "data" / "ase-test")
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"q{qid}-result.json").open("w") as f:
            json.dump(obj or {"raw": response}, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_dir}/q{qid}-result.json")

    if not args.prompt_only and results:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        ok = sum(1 for r in results if r["ok"])
        print(f"  {ok}/{len(results)} successful")
        for r in results:
            status = "OK" if r["ok"] else "FAIL"
            print(f"  Q{r['question_id']}: {status} ({r['elapsed']:.1f}s)"
                  + (f" — {r.get('n_nodes',0)} nodes, {r.get('n_edges',0)} edges" if r["ok"] else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
