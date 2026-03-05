#!/usr/bin/env python3
"""Generate a daily hotspot dashboard for First-Proof + Frontier work.

This script tracks whether we are reducing known failure modes:
- unresolved proof-polish outcomes by problem/node
- Frontier Spec-Lock completion
- Frontier judgement coverage in review JSON files

Usage:
  python3 scripts/hotspot-dashboard.py
  python3 scripts/hotspot-dashboard.py --output-md holes/handoffs/hotspot-dashboard-YYYY-MM-DD.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID = {"verified", "plausible", "gap", "error"}
PROBLEM_RE = re.compile(r"problem(\d+)-codex-results")
SPEC_RE = re.compile(r"`spec_lock_status`:\s*`?([A-Za-z]+)`?")


@dataclass
class ProblemSummary:
    problem_id: str
    rows: int
    verified: int
    plausible: int
    gap: int
    error: int
    parse: int
    stubborn_nodes: list[str]


@dataclass
class ReviewCoverage:
    file: str
    labeled: int
    total: int


@dataclass
class SpecStatus:
    file: str
    status: str


def classify_status(rec: dict[str, Any]) -> str:
    st = rec.get("claim_verified")
    if st in VALID:
        return str(st)
    if rec.get("parse_error"):
        return "parse"
    return "parse"


def load_problem_results(data_dir: Path, min_stubborn_obs: int) -> list[ProblemSummary]:
    files = sorted(data_dir.glob("problem*-codex-results*.jsonl"))
    per_problem_counts: dict[str, Counter[str]] = defaultdict(Counter)
    per_node_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for path in files:
        m = PROBLEM_RE.search(path.name)
        if not m:
            continue
        pid = f"P{int(m.group(1))}"

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                st = classify_status(rec)
                per_problem_counts[pid][st] += 1
                node_id = rec.get("node_id")
                if isinstance(node_id, str) and node_id:
                    per_node_counts[(pid, node_id)][st] += 1

    out: list[ProblemSummary] = []
    for pid in sorted(per_problem_counts.keys(), key=lambda s: int(s[1:])):
        c = per_problem_counts[pid]
        stubborn: list[str] = []
        for (p, node_id), nc in sorted(per_node_counts.items()):
            if p != pid:
                continue
            obs = sum(nc.values())
            if obs < min_stubborn_obs:
                continue
            if nc["verified"] == 0:
                unresolved = nc["plausible"] + nc["gap"] + nc["error"] + nc["parse"]
                if unresolved > 0:
                    stubborn.append(node_id)

        out.append(
            ProblemSummary(
                problem_id=pid,
                rows=sum(c.values()),
                verified=c["verified"],
                plausible=c["plausible"],
                gap=c["gap"],
                error=c["error"],
                parse=c["parse"],
                stubborn_nodes=stubborn,
            )
        )
    return out


def parse_spec_status(state_path: Path) -> str:
    txt = state_path.read_text(encoding="utf-8", errors="ignore")
    m = SPEC_RE.search(txt)
    if not m:
        return "unknown"
    return m.group(1).strip().lower()


def load_frontier_specs(frontier_dir: Path) -> list[SpecStatus]:
    out: list[SpecStatus] = []
    for p in sorted(frontier_dir.glob("FM-*-state.md")):
        out.append(SpecStatus(file=p.name, status=parse_spec_status(p)))
    return out


def load_review_coverage(frontier_dir: Path) -> list[ReviewCoverage]:
    out: list[ReviewCoverage] = []
    for p in sorted(frontier_dir.glob("superpod-frontier-trial*review*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        labeled = 0
        total = 0
        for problem in payload.get("problems", []):
            for rec in problem.get("candidates", []):
                total += 1
                j = (rec.get("judgement") or "").strip().lower()
                if j in {"yes", "no", "unsure"}:
                    labeled += 1
        out.append(ReviewCoverage(file=p.name, labeled=labeled, total=total))
    return out


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def render(
    problems: list[ProblemSummary],
    specs: list[SpecStatus],
    reviews: list[ReviewCoverage],
    min_labels_per_problem: int,
) -> str:
    now = datetime.now(timezone.utc).isoformat()

    all_spec_pass = bool(specs) and all(s.status == "pass" for s in specs)
    any_stubborn = any(p.stubborn_nodes for p in problems)

    # Label gate: require at least min_labels_per_problem per FM problem per review file.
    required = min_labels_per_problem * 3
    label_gate_pass = bool(reviews) and all(r.labeled >= required for r in reviews)

    lines: list[str] = []
    lines.append("# Hotspot Dashboard")
    lines.append("")
    lines.append(f"Generated: `{now}`")
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append(f"- Spec gate (all FM state files pass): `{'PASS' if all_spec_pass else 'FAIL'}`")
    lines.append(
        "- Label gate "
        f"(>= {min_labels_per_problem} labels/problem per review file): `{'PASS' if label_gate_pass else 'FAIL'}`"
    )
    lines.append(f"- Stubborn-node gate (no persistent unresolved nodes): `{'PASS' if not any_stubborn else 'FAIL'}`")
    lines.append("")

    lines.append("## First-Proof Problem Status")
    lines.append("")
    lines.append("| Problem | Rows | Verified | Plausible | Gap | Error | Parse | Unresolved % | Stubborn Nodes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for p in problems:
        unresolved = p.plausible + p.gap + p.error + p.parse
        unresolved_pct = pct(unresolved, p.rows)
        stubborn = ", ".join(p.stubborn_nodes) if p.stubborn_nodes else "-"
        lines.append(
            f"| {p.problem_id} | {p.rows} | {p.verified} | {p.plausible} | {p.gap} | "
            f"{p.error} | {p.parse} | {unresolved_pct:.1f}% | {stubborn} |"
        )
    lines.append("")

    lines.append("## Frontier Spec-Lock")
    lines.append("")
    lines.append("| File | Spec Lock Status |")
    lines.append("|---|---|")
    for s in specs:
        lines.append(f"| {s.file} | {s.status} |")
    lines.append("")

    lines.append("## Frontier Review Coverage")
    lines.append("")
    lines.append("| Review File | Labeled | Total | Coverage | Gate Target |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in reviews:
        cov = pct(r.labeled, r.total)
        lines.append(f"| {r.file} | {r.labeled} | {r.total} | {cov:.1f}% | {required} |")
    lines.append("")

    lines.append("## Immediate Focus")
    lines.append("")
    lines.append("1. Complete Spec-Lock for all FM problems before new retrieval runs.")
    lines.append("2. Label existing Frontier review files to satisfy label gate.")
    lines.append("3. Run targeted proof deep-dives on stubborn nodes only.")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/first-proof"))
    ap.add_argument("--frontier-dir", type=Path, default=Path("data/first-proof/frontiermath-pilot"))
    ap.add_argument("--min-stubborn-obs", type=int, default=3,
                    help="Minimum observations for a node to be considered stubborn")
    ap.add_argument("--min-labels-per-problem", type=int, default=30,
                    help="Required labels per FM problem per review file")
    ap.add_argument("--output-md", type=Path, default=None,
                    help="Optional markdown output path")
    args = ap.parse_args()

    problems = load_problem_results(args.data_dir, args.min_stubborn_obs)
    specs = load_frontier_specs(args.frontier_dir)
    reviews = load_review_coverage(args.frontier_dir)

    out = render(problems, specs, reviews, args.min_labels_per_problem)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(out, encoding="utf-8")
    print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
