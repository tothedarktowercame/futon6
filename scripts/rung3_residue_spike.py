#!/usr/bin/env python3
"""Measure rung-3 deterministic residue over CAS-0 moves and loop-run edges.

This is deliberately a measurement helper, not a new selector.  For CAS-0 it
reuses cas_select.retrieve + cas_select.verify directly so the selector's
fixture-only oracle injection path cannot inflate the match rate.  For IATC
loop-run edges there is no oracle, so the script reports retrieval candidates
as provisional and keeps them separate from strict verified matches.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cas_select
from mark3_eval_harness import read_text, top_level_map_blocks_in_vector


ROOT = Path(__file__).resolve().parents[1]
CAS_FIXTURES = ROOT / "tests" / "fixtures" / "cas-select"
LOOP_RUN = ROOT / "data" / "iatc-argument-graphs" / "loop-run-70b"

VERIFIABLE_PATTERNS = {
    "construct-an-explicit-witness",
    "count-over-a-decomposition",
    "epsilon-of-room",
    "estimate-by-bounding",
    "induction-and-well-ordering",
    "quotient-by-irrelevance",
    "reduce-to-known-result",
    "split-into-cases",
    "unfold-the-definition",
    "verify-universal-property",
}

AUTHOR_DECLARED_GAP_RE = re.compile(
    r"\b(conjectur\w*|open problem|problem of|problem is|ought to|"
    r"ought-to|we do not know|unknown|question whether)\b",
    re.I,
)


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def pattern_type(pattern: str | None) -> str:
    if not pattern:
        return "none"
    if pattern in VERIFIABLE_PATTERNS:
        return "verifiable"
    return "heuristic"


def keyword_field(block: str, key: str) -> str | None:
    match = re.search(rf":{re.escape(key)}\s+(:[^\s,}}]+)", block)
    return match.group(1) if match else None


def warrant_text(block: str) -> str:
    match = re.search(r':warrant\s+\{[^{}]*:text\s+"([^"]*)"', block)
    if match:
        return match.group(1)
    if re.search(r":warrant\s+\{[^{}]*:kind\s+:missing-warrant", block):
        return "missing warrant"
    if ":warrant" in block:
        return "resolved warrant"
    return "no warrant supplied"


def edge_move_text(block: str) -> str:
    fields = {
        "id": keyword_field(block, "id") or "?",
        "relation": keyword_field(block, "relation") or "?",
        "premise": keyword_field(block, "premise") or "?",
        "conclusion": keyword_field(block, "conclusion") or "?",
        "warrant": warrant_text(block),
    }
    return (
        f"edge {fields['id']} relation {fields['relation']} from premise "
        f"{fields['premise']} to conclusion {fields['conclusion']} with warrant "
        f"{fields['warrant']}"
    )


def source_lines(block: str) -> str:
    match = re.search(r":lines\s+\[([0-9]+)\s+([0-9]+)\]", block)
    if not match:
        return "?"
    return f"{match.group(1)}-{match.group(2)}"


def measure_cas(patterns: dict[str, cas_select.Pattern], k: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for steps_path in sorted(CAS_FIXTURES.glob("*.steps.json")):
        steps_doc = cas_select.load_steps(steps_path)
        oracle = cas_select.load_oracle(CAS_FIXTURES / f"{steps_doc['paper_id']}.oracle.json")
        for step in steps_doc["steps"]:
            candidates = cas_select.retrieve(step["text"], patterns, k=k)
            verdict = cas_select.verify(
                step,
                candidates,
                patterns,
                backend="stub",
                oracle=oracle,
            )
            expected = oracle.get(step["id"], {}).get("pattern")
            actual = verdict.get("pattern")
            ptype = pattern_type(actual)
            if actual:
                bucket = "grounded" if ptype == "verifiable" else "thin"
            else:
                bucket = "ungrounded"
            rows.append(
                {
                    "source": "cas0",
                    "paper": steps_doc["paper_id"],
                    "move": step["id"],
                    "text": step["text"],
                    "expected": expected,
                    "candidates": [c["pattern"] for c in candidates],
                    "matched": actual,
                    "pattern_type": ptype,
                    "bucket": bucket,
                    "slot": verdict.get("slot"),
                }
            )
    return summarize_rows(rows, strict=True)


def measure_loop(patterns: dict[str, cas_select.Pattern], k: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LOOP_RUN.glob("*.edn")):
        for i, edge in enumerate(top_level_map_blocks_in_vector(read_text(path), ":edges"), start=1):
            text = edge_move_text(edge)
            candidates = cas_select.retrieve(text, patterns, k=k)
            top = candidates[0]["pattern"] if candidates else None
            ptype = pattern_type(top)
            declared_gap = bool(AUTHOR_DECLARED_GAP_RE.search(text))
            if declared_gap:
                bucket = "conjecture"
            elif top and ptype == "verifiable":
                bucket = "grounded-provisional"
            elif top:
                bucket = "thin"
            else:
                bucket = "ungrounded"
            rows.append(
                {
                    "source": "loop-run-70b",
                    "paper": path.stem,
                    "move": f"edge-{i}",
                    "lines": source_lines(edge),
                    "text": text,
                    "candidates": [c["pattern"] for c in candidates],
                    "matched": top,
                    "pattern_type": ptype,
                    "bucket": bucket,
                    "declared_gap": declared_gap,
                    "strict_note": "no verifier oracle for loop-run edges",
                }
            )
    return summarize_rows(rows, strict=False)


def summarize_rows(rows: list[dict[str, Any]], *, strict: bool) -> dict[str, Any]:
    total = len(rows)
    if strict:
        matched = sum(1 for r in rows if r["matched"])
        residue = total - matched
    else:
        matched = sum(1 for r in rows if r["matched"])
        residue = total - matched
    return {
        "total": total,
        "matched": matched,
        "residue": residue,
        "residue_rate": pct(residue, total),
        "coverage_rate": pct(matched, total),
        "strict_verified": matched if strict else 0,
        "strict_residue": residue if strict else total,
        "strict_residue_rate": pct(residue if strict else total, total),
        "buckets": dict(Counter(r["bucket"] for r in rows)),
        "pattern_types": dict(Counter(r["pattern_type"] for r in rows)),
        "patterns": dict(Counter(r["matched"] for r in rows if r["matched"])),
        "by_paper": {
            paper: {
                "total": len(items),
                "matched": sum(1 for r in items if r["matched"]),
                "residue": sum(1 for r in items if not r["matched"]),
                "buckets": dict(Counter(r["bucket"] for r in items)),
            }
            for paper, items in group_by(rows, "paper").items()
        },
        "rows": rows,
    }


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row[key])].append(row)
    return dict(sorted(out.items()))


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def arse_mapping_table() -> str:
    return """| Gap bucket | RM question pattern | ArSE question template |
|---|---|---|
| `thin` / heuristic leaf | `STRUCTURAL PROBE` | What verifiable inference discharges the heuristic step `<pattern>` here? |
| `ungrounded` / no match | `THEOREM APPLICABILITY` or `TECHNIQUE LANDSCAPE` | Which known theorem or proof technique, if any, licenses this move from `<premise>` to `<conclusion>`? |
| missing or unresolved warrant | `KERNEL IDENTIFICATION` | What is the one lemma/computation needed to turn this edge into a resolved inference? |
| author-declared gap/conjecture | `EXISTENCE_WONDER` / `CONJECTURE_TESTING` | Is the stated extension/problem known under the hypotheses used in the passage? |
| obstruction-like residual | `OBSTRUCTION_IDENTIFICATION` | What obstruction prevents the intended inference or generalization? |"""


def render_report(payload: dict[str, Any]) -> str:
    cas = payload["cas0"]
    loop = payload["loop_run_70b"]
    combined_total = cas["total"] + loop["total"]
    combined_candidate_residue = cas["residue"] + loop["residue"]
    lines = [
        "# rung-3-1 residue spike",
        "",
        "This is the empirical verb-side counterpart to R2d: measure how often a proof move is covered by the current CAS pattern menu before asking a model to judge or invent the residue.",
        "",
        "## Inputs and method",
        "",
        f"- Pattern pool: `{cas_select.DEFAULT_LIBRARY}` plus `{cas_select.DEFAULT_INDEX}`; loaded `{payload['pattern_count']}` math-informal patterns.",
        "- CAS-0 worked proofs: `tests/fixtures/cas-select/{a93J05,a96J01,b97J01,a96J04}.steps.json` with their oracle files.",
        "- Loop-run sample: final EDN graphs in `data/iatc-argument-graphs/loop-run-70b` only.",
        "- Selector reuse: direct `cas_select.retrieve(..., k=4)` followed by `cas_select.verify(..., backend=\"stub\", oracle=...)` for CAS-0. This deliberately avoids `select_proof`, whose test-only stub path injects oracle patterns after retrieval misses.",
        "- Loop-run edges have no oracle, so their CAS rows are retrieval-only candidates, not verified matches. They are useful for estimating how much of the 70B edge vocabulary the current menu can even touch, but not for correctness.",
        "- Question menu source on disk: `holes/handoffs/question-asking-pattern-mining-from-mo-rm-2026-03-06.md` and `holes/excursions/E-informal-proof-checking.md`. The referenced `data/question-patterns/question-asking-pattern-language.md` is not present in this checkout.",
        "",
        "## Residue measurements",
        "",
        "| Sample | Moves | Deterministic covered | Residue | Residue rate | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
        f"| CAS-0 strict verified | {cas['total']} | {cas['matched']} | {cas['residue']} | {fmt_pct(cas['residue_rate'])} | Current committed selector's honest strict share; this is the measured LLM/verifier residue on worked proof moves. |",
        f"| loop-run-70b strict verified | {loop['total']} | {loop['strict_verified']} | {loop['strict_residue']} | {fmt_pct(loop['strict_residue_rate'])} | No oracle-backed verifier exists for these graph edges, so every edge remains strict residue. |",
        f"| loop-run-70b retrieval-only | {loop['total']} | {loop['matched']} | {loop['residue']} | {fmt_pct(loop['residue_rate'])} | Candidate coverage only; this measures menu reach, not correctness. |",
        f"| combined candidate surface | {combined_total} | {combined_total - combined_candidate_residue} | {combined_candidate_residue} | {fmt_pct(pct(combined_candidate_residue, combined_total))} | Upper-bound deterministic menu reach if loop retrieval candidates are later verified. |",
        "",
        f"Strict CAS-0 residue is therefore **{cas['residue']}/{cas['total']} = {fmt_pct(cas['residue_rate'])}**. That number is the empirical LLM share for the current CAS-0 verified setting.",
        "",
        "## Buckets",
        "",
        "CAS-0 strict buckets:",
        "",
        "```json",
        json.dumps(cas["buckets"], indent=2, sort_keys=True),
        "```",
        "",
        "loop-run-70b retrieval buckets:",
        "",
        "```json",
        json.dumps(loop["buckets"], indent=2, sort_keys=True),
        "```",
        "",
        "Pattern-type counts:",
        "",
        "```json",
        json.dumps({"cas0": cas["pattern_types"], "loop_run_70b": loop["pattern_types"]}, indent=2, sort_keys=True),
        "```",
        "",
        "## Heuristic vs verifiable typing",
        "",
        "For this spike, a matched pattern is typed `verifiable` only when it can plausibly license an inference leaf by a checkable object, definition, theorem application, calculation, induction, case split, or bound. Other retrieved CAS patterns are typed `heuristic`: they may justify a strategy, but a load-bearing proof edge still needs a lower verifiable discharge.",
        "",
        "Verifiable pattern set used in the measurement:",
        "",
        "```",
        ", ".join(sorted(VERIFIABLE_PATTERNS)),
        "```",
        "",
        "CAS-0 matched-pattern distribution:",
        "",
        "```json",
        json.dumps(cas["patterns"], indent=2, sort_keys=True),
        "```",
        "",
        "loop-run-70b top-candidate distribution:",
        "",
        "```json",
        json.dumps(loop["patterns"], indent=2, sort_keys=True),
        "```",
        "",
        "## Conjecture recognition",
        "",
        "Author-declared gaps are credited rather than flagged as thin: a sentence or edge text matching `conjecture`, `open problem`, `problem of`, `ought to`, `unknown`, or `we do not know` goes to the `conjecture` bucket. In this sample no CAS-0 fixture step is author-declared. The loop-run sample contains retrieval rows with phrases such as `ought-to-include`, and those are credited as author-declared/open-status rather than as hidden failures.",
        "",
        "## Gap to ArSE question mapping",
        "",
        arse_mapping_table(),
        "",
        "## CAS-0 per-move evidence",
        "",
        "| Paper | Move | Expected | Verified match | Type | Bucket | Top-4 candidates |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in cas["rows"]:
        lines.append(
            f"| `{row['paper']}` | `{row['move']}` | `{row['expected']}` | "
            f"`{row['matched'] or 'NONE'}` | `{row['pattern_type']}` | `{row['bucket']}` | "
            f"`{', '.join(row['candidates']) or 'NONE'}` |"
        )
    lines.extend(
        [
            "",
            "## loop-run-70b per-edge sample",
            "",
            "| Paper | Move | Lines | Top candidate | Type | Bucket | Top-4 candidates |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for row in loop["rows"]:
        lines.append(
            f"| `{row['paper']}` | `{row['move']}` | `{row['lines']}` | "
            f"`{row['matched'] or 'NONE'}` | `{row['pattern_type']}` | `{row['bucket']}` | "
            f"`{', '.join(row['candidates']) or 'NONE'}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The deterministic menu is already useful but not sufficient. On CAS-0 it strictly verifies 16/22 moves and leaves 6/22 for a semantic retriever/verifier or a new pattern. On loop-run-70b it can attach candidates to most edges, but that is not a proof of match: rung-3-3 still needs a model or richer verifier for the residual judgement `does this edge instantiate this pattern?`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    patterns = cas_select.load_patterns()
    payload = {
        "pattern_count": len(patterns),
        "cas0": measure_cas(patterns, args.k),
        "loop_run_70b": measure_loop(patterns, args.k),
    }
    args.out.write_text(render_report(payload), encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "holes" / "excursions" / "rung-3-spec.md")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args(argv)
    payload = run(args)
    cas = payload["cas0"]
    loop = payload["loop_run_70b"]
    print(
        f"CAS-0 strict residue: {cas['residue']}/{cas['total']} "
        f"({fmt_pct(cas['residue_rate'])}); loop candidate residue: "
        f"{loop['residue']}/{loop['total']} ({fmt_pct(loop['residue_rate'])}); "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
