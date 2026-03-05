#!/usr/bin/env python3
"""Generate synthetic QA pairs for the Artificial Stack Exchange.

For each proof node, identifies retrieval gaps (topics where the real corpus
is thin), then generates question+answer pairs in hypergraph-native format
— born with typed nodes and edges, immediately FAISS-indexable.

The generator uses existing corpus threads as exemplars: it finds the best
real threads for a topic, then asks the LLM to produce new QA targeting
the specific gap, structured as a hypergraph.

Usage:
    # Identify gaps and generate question prompts (no API calls)
    python3 scripts/generate-synthetic-qa.py --dry-run

    # Generate via API
    python3 scripts/generate-synthetic-qa.py --backend codex

    # Generate for a specific proof node
    python3 scripts/generate-synthetic-qa.py --node-id p7-s4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STORAGE_MATH = Path(os.path.expanduser("~/code/storage/math-processed-gpu"))

# Hypergraph schema for synthetic QA — matches the real pipeline's format
SYNTHETIC_HYPERGRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "thread_id": {"type": "string"},
        "title": {"type": "string"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["post", "term", "expression", "scope"]},
                    "subtype": {"type": "string"},
                    "attrs": {"type": "object"},
                },
                "required": ["id", "type", "subtype"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["iatc", "mention", "discourse", "scope", "surface", "categorical"],
                    },
                    "ends": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "attrs": {"type": "object"},
                },
                "required": ["type", "ends"],
            },
        },
    },
    "required": ["thread_id", "title", "question", "answer", "tags", "nodes", "edges"],
}

# Per-node gap descriptions: what the real corpus is thin on
NODE_GAP_SPECS: dict[str, dict] = {
    "p7-problem": {
        "topic": "uniform lattices with torsion in semi-simple Lie groups",
        "gap": "Few threads discuss the specific interplay between 2-torsion in "
               "lattices and manifold realization. Need QA about when torsion "
               "obstructs or permits pi_1 realization.",
        "seed_tags": ["lie-groups", "lattice", "algebraic-topology", "manifolds"],
    },
    "p7-s1": {
        "topic": "orbifold quotients of symmetric spaces by torsion lattices",
        "gap": "The corpus covers free actions well but is thin on non-free actions "
               "producing orbifolds. Need QA about fixed-point sets of involutions "
               "on symmetric spaces.",
        "seed_tags": ["lie-groups", "differential-topology", "orbifolds"],
    },
    "p7-s2": {
        "topic": "rational Poincare duality for groups with torsion",
        "gap": "Standard PD references assume torsion-free. Need QA about "
               "Bredon/orbifold cohomology giving rational PD for uniform lattices "
               "WITH torsion.",
        "seed_tags": ["group-cohomology", "homological-algebra", "algebraic-topology"],
    },
    "p7-s3": {
        "topic": "equivariant finiteness obstructions and FH(Q)",
        "gap": "Fowler's FH(Q) theorem is not well-represented. Need QA about "
               "finiteness conditions for orbifold groups, Euler-vanishing hypotheses, "
               "and the distinction between FH(Q) and actual manifold realization.",
        "seed_tags": ["algebraic-topology", "homotopy-theory", "group-theory"],
    },
    "p7-s3a": {
        "topic": "arithmetic lattice examples in FH(Q)",
        "gap": "No threads specifically discuss Fowler Section 5 arithmetic lattice "
               "constructions. Need QA about which torsion orders are realizable.",
        "seed_tags": ["algebraic-groups", "number-theory", "group-theory"],
    },
    "p7-s4": {
        "topic": "surgery-theoretic upgrade from finite CW complex to closed manifold",
        "gap": "The delicate step: upgrading FH(Q) to a closed manifold with same "
               "pi_1. Need QA about normal invariants, L-group obstructions, "
               "and Wall surgery exact sequence for groups with torsion.",
        "seed_tags": ["algebraic-topology", "manifolds", "differential-topology"],
    },
    "p7-s5": {
        "topic": "Smith theory: mod-p vs rational acyclicity",
        "gap": "Smith theory is classically mod-p. Need QA clarifying why the "
               "obstruction vanishes over Q and what happens for Z/2 acting freely "
               "on rationally acyclic spaces.",
        "seed_tags": ["algebraic-topology", "group-theory", "homology-cohomology"],
    },
    "p7-s6": {
        "topic": "composing PD + finiteness + surgery for manifold realization",
        "gap": "No threads address the full composition: rational PD group -> "
               "FH(Q) finite complex -> surgery upgrade -> closed manifold. Need QA "
               "about the complete obstruction theory.",
        "seed_tags": ["algebraic-topology", "manifolds", "homotopy-theory"],
    },
}


def load_corpus_context(context_path: Path) -> dict[str, list[dict]]:
    """Load pre-retrieved corpus context per node."""
    result = {}
    if not context_path.exists():
        return result
    with context_path.open() as f:
        for line in f:
            rec = json.loads(line)
            result[rec["node_id"]] = rec.get("retrieved", [])
    return result


def extract_question_components(threads: list[dict]) -> list[dict]:
    """Extract reusable question components from retrieved threads.

    Pulls out sub-questions, key LaTeX expressions, and framing patterns
    that can be recombined into new questions.
    """
    import re

    components = []
    for t in threads:
        q = t.get("question_excerpt", "")
        a = t.get("answer_excerpt", "")
        title = t.get("title", "")
        eid = t.get("entity_id", "")
        source = t.get("retrieval_source", "text")

        # Extract LaTeX expressions
        latex_exprs = re.findall(r'\$[^$]+\$|\$\$[^$]+\$\$', q + " " + a)
        # Deduplicate and take most substantial
        latex_exprs = sorted(set(latex_exprs), key=len, reverse=True)[:5]

        # Extract sub-questions (sentences ending with ?)
        sub_questions = re.findall(r'[^.!?]*\?', q)
        sub_questions = [sq.strip() for sq in sub_questions if len(sq.strip()) > 20]

        # Extract "show that" / "prove that" / "verify" clauses
        proof_tasks = re.findall(
            r'(?:show|prove|verify|check|confirm|determine)\s+that\s+[^.?!]+[.?!]',
            q + " " + a,
            re.IGNORECASE,
        )

        # Extract key assertions from the answer (sentences with "therefore", "hence", etc.)
        conclusions = re.findall(
            r'[^.]*(?:therefore|hence|thus|it follows|this (?:shows|implies|means))[^.]*\.',
            a,
            re.IGNORECASE,
        )

        if latex_exprs or sub_questions or proof_tasks:
            components.append({
                "source_id": eid,
                "source_title": title,
                "retrieval_source": source,
                "latex_expressions": latex_exprs[:3],
                "sub_questions": sub_questions[:3],
                "proof_tasks": proof_tasks[:2],
                "conclusions": conclusions[:2],
            })

    return components


def format_components_for_prompt(components: list[dict]) -> str:
    """Format extracted components as prompt material for question composition."""
    if not components:
        return ""

    lines = [
        "## Question Components (from similar threads)",
        "",
        "Use these fragments from structurally similar threads to compose "
        "your question. Recombine, adapt, and redirect them toward the "
        "specific gap described above. Do NOT copy verbatim — use these "
        "as building blocks.",
        "",
    ]

    for i, c in enumerate(components, 1):
        source_tag = " [structural]" if c["retrieval_source"] == "structural" else ""
        lines.append(f"### Source {i}: {c['source_title'][:60]}{source_tag}")

        if c["sub_questions"]:
            lines.append("**Sub-questions:**")
            for sq in c["sub_questions"]:
                lines.append(f"  - {sq[:150]}")

        if c["latex_expressions"]:
            lines.append("**Key expressions:**")
            for expr in c["latex_expressions"]:
                lines.append(f"  - {expr[:200]}")

        if c["proof_tasks"]:
            lines.append("**Proof tasks:**")
            for pt in c["proof_tasks"]:
                lines.append(f"  - {pt[:200]}")

        if c["conclusions"]:
            lines.append("**Conclusions to build on:**")
            for con in c["conclusions"]:
                lines.append(f"  - {con[:200]}")

        lines.append("")

    return "\n".join(lines)


def build_generation_prompt(
    node_id: str,
    gap_spec: dict,
    exemplar_threads: list[dict],
    wiring_node: dict | None = None,
    question_components: list[dict] | None = None,
) -> str:
    """Build a prompt that generates a synthetic QA pair in hypergraph format.

    When question_components are provided (extracted from FAISS-retrieved
    threads), the prompt instructs the LLM to compose the new question
    by recombining those fragments rather than generating from scratch.
    """

    lines = [
        "You are a mathematics educator creating study materials for "
        "advanced graduate students. Generate a Stack Exchange-style "
        "question and answer pair about the topic below.",
        "",
        "## Topic",
        "",
        f"**Area**: {gap_spec['topic']}",
        f"**Gap to fill**: {gap_spec['gap']}",
        f"**Tags**: {', '.join(gap_spec['seed_tags'])}",
        "",
    ]

    if wiring_node:
        lines.extend([
            "## Proof Context",
            "",
            f"This QA pair should be useful for verifying the following proof step:",
            f"**Node**: {wiring_node['id']}",
            f"**Claim**: {wiring_node.get('body_text', '')}",
            "",
        ])

    if exemplar_threads:
        lines.extend([
            "## Exemplar Threads (from real corpus)",
            "",
            "Use these as style/depth exemplars. Your generated QA should be "
            "at a similar level but target the specific gap described above.",
            "",
        ])
        for i, t in enumerate(exemplar_threads[:3], 1):
            lines.append(f"### Exemplar {i}: {t.get('title', '')}")
            q = t.get("question_excerpt", "")[:300]
            a = t.get("answer_excerpt", "")[:400]
            if q:
                lines.append(f"Q: {q}")
            if a:
                lines.append(f"A: {a}")
            lines.append("")

    if question_components:
        lines.append(format_components_for_prompt(question_components))

    lines.extend([
        "## Output Format",
        "",
        "Generate a JSON object with this structure:",
        "",
        "- `thread_id`: a unique string like `synth-p7-s4-001`",
        "- `title`: a concise question title",
        "- `question`: the full question body (include LaTeX where appropriate)",
        "- `answer`: a detailed answer with mathematical rigor",
        "- `tags`: list of relevant math tags",
        "- `nodes`: typed hypergraph nodes representing the content structure:",
        "  - `{id, type: 'post', subtype: 'question'|'answer'}` for the Q and A",
        "  - `{id, type: 'term', subtype: '<math-concept>'}` for key mathematical terms",
        "  - `{id, type: 'expression', subtype: 'latex'}` for important formulas",
        "  - `{id, type: 'scope', subtype: 'quant/universal'|'quant/existential'|'conditional'}` for logical scopes",
        "- `edges`: typed edges connecting nodes:",
        "  - `{type: 'mention', ends: [post_id, term_id]}` — post mentions a term",
        "  - `{type: 'surface', ends: [post_id, expr_id]}` — post contains expression",
        "  - `{type: 'scope', ends: [post_id, scope_id]}` — post has a logical scope",
        "  - `{type: 'discourse', ends: [question_id, answer_id]}` — answer responds to question",
        "  - `{type: 'iatc', ends: [src_id, tgt_id], attrs: {performative: 'assert'|'clarify'|'query'}}` — speech act",
        "  - `{type: 'categorical', ends: [src_id, tgt_id], attrs: {relation: '<cat-theory-relation>'}}` — if applicable",
        "",
        "Aim for 4-8 term nodes, 2-4 expression nodes, and 1-3 scope nodes.",
        "Every node should be connected by at least one edge.",
        "Reply with ONLY the JSON object.",
    ])

    return "\n".join(lines)


def identify_gaps(
    corpus_context: dict[str, list[dict]],
    wiring: dict,
) -> list[dict]:
    """Identify retrieval gaps per proof node and build generation specs."""
    nodes_by_id = {n["id"]: n for n in wiring["nodes"]}
    specs = []

    for node_id, gap_spec in NODE_GAP_SPECS.items():
        retrieved = corpus_context.get(node_id, [])
        wiring_node = nodes_by_id.get(node_id)

        # Assess gap severity
        text_threads = [r for r in retrieved if r.get("retrieval_source") == "text"]
        structural_threads = [r for r in retrieved if r.get("retrieval_source") == "structural"]
        max_kw_score = max((r.get("retrieval_keyword_score", 0) for r in retrieved), default=0)

        # Higher priority = bigger gap
        gap_severity = "low"
        if max_kw_score < 5:
            gap_severity = "high"
        elif max_kw_score < 8:
            gap_severity = "medium"

        specs.append({
            "node_id": node_id,
            "gap_spec": gap_spec,
            "wiring_node": wiring_node,
            "exemplar_threads": retrieved[:3],
            "gap_severity": gap_severity,
            "n_existing_threads": len(retrieved),
            "max_keyword_score": max_kw_score,
        })

    # Sort: high-severity gaps first
    severity_order = {"high": 0, "medium": 1, "low": 2}
    specs.sort(key=lambda s: severity_order[s["gap_severity"]])
    return specs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--problem", type=int, default=7)
    parser.add_argument("--corpus-context", type=Path, default=None,
                        help="Pre-retrieved context JSONL (from retrieve-proof-context.py)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL for synthetic QA (default: data/synthetic-qa/problem{N}.jsonl)")
    parser.add_argument("--prompts-out", type=Path, default=None,
                        help="Write generation prompts (default: data/synthetic-qa/problem{N}-prompts.jsonl)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Identify gaps and write prompts only, no API calls")
    parser.add_argument("--node-id", action="append", default=[],
                        help="Restrict to specific node IDs (repeatable)")
    parser.add_argument("--n-per-node", type=int, default=2,
                        help="Synthetic QA pairs to generate per node (default: 2)")
    parser.add_argument("--backend", choices=["codex", "claude", "gemini"], default="codex")
    args = parser.parse_args()

    if args.output is None:
        args.output = REPO_ROOT / "data" / "synthetic-qa" / f"problem{args.problem}.jsonl"
    if args.prompts_out is None:
        args.prompts_out = REPO_ROOT / "data" / "synthetic-qa" / f"problem{args.problem}-prompts.jsonl"
    if args.corpus_context is None:
        args.corpus_context = (
            REPO_ROOT / "data" / "first-proof" / f"problem{args.problem}-corpus-context.jsonl"
        )

    # Load wiring
    wiring_path = REPO_ROOT / "data" / "first-proof" / f"problem{args.problem}-wiring.json"
    if not wiring_path.exists():
        print(f"Wiring not found: {wiring_path}", file=sys.stderr)
        return 2
    wiring = json.loads(wiring_path.read_text())

    # Load corpus context
    corpus_context = load_corpus_context(args.corpus_context)
    if not corpus_context:
        print(f"No corpus context at {args.corpus_context}")
        print("Run: python3 scripts/retrieve-proof-context.py first")
        return 2

    # Identify gaps
    specs = identify_gaps(corpus_context, wiring)
    if args.node_id:
        requested = set(args.node_id)
        specs = [s for s in specs if s["node_id"] in requested]

    print(f"Gap analysis for problem {args.problem}:")
    print(f"{'Node':<15s} {'Severity':<10s} {'Existing':>10s} {'Max KW':>8s} Topic")
    print("-" * 80)
    for s in specs:
        print(f"{s['node_id']:<15s} {s['gap_severity']:<10s} "
              f"{s['n_existing_threads']:>10d} {s['max_keyword_score']:>8d} "
              f"{s['gap_spec']['topic'][:40]}")

    # Build prompts — extract question components from retrieved threads
    prompts = []
    for spec in specs:
        components = extract_question_components(spec["exemplar_threads"])
        if components:
            print(f"  {spec['node_id']}: extracted {len(components)} component sources "
                  f"({sum(len(c['sub_questions']) for c in components)} sub-questions, "
                  f"{sum(len(c['latex_expressions']) for c in components)} expressions)")

        for i in range(args.n_per_node):
            prompt = build_generation_prompt(
                node_id=spec["node_id"],
                gap_spec=spec["gap_spec"],
                exemplar_threads=spec["exemplar_threads"],
                wiring_node=spec["wiring_node"],
                question_components=components,
            )
            prompts.append({
                "node_id": spec["node_id"],
                "instance": i,
                "gap_severity": spec["gap_severity"],
                "prompt": prompt,
                "thread_id": f"synth-{spec['node_id']}-{i:03d}",
            })

    # Write prompts
    args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
    with args.prompts_out.open("w") as f:
        for rec in prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(prompts)} generation prompts to {args.prompts_out}")

    if args.dry_run:
        print("\nDry run — prompts written, no API calls.")
        print(f"\nPrompt summary ({len(prompts)} prompts):")
        for p in prompts:
            lines = p["prompt"].count("\n") + 1
            print(f"  {p['thread_id']:25s} [{p['gap_severity']:6s}] ~{lines} lines")
        return 0

    # TODO: API execution (codex/claude/gemini)
    # For now, prompts are written as JSONL and can be fed to any backend
    print(f"\nBackend '{args.backend}' execution not yet implemented.")
    print(f"Prompts are ready at {args.prompts_out}")
    print(f"Feed them to run-stage6-codex.py or equivalent.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
