#!/usr/bin/env python3
"""Compare two proof-polishing result JSONLs (e.g., wired vs claim-only).

This script summarizes:
- outcome quality (verified/plausible/gap/error/parse)
- latency/retry behavior
- reference source mix (MO vs Math.SE)
- pairwise per-node score deltas
- optional edge-consistency diagnostics using a wiring JSON
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


VALID_STATUSES = {"verified", "plausible", "gap", "error"}
SCORE_MAP = {"verified": 2, "plausible": 1, "gap": 0, "error": 0}


@dataclass
class Summary:
    label: str
    n: int
    verified: int
    plausible: int
    gap: int
    error: int
    parse: int
    score_sum: int
    score_avg: float
    retries_gt1: int
    timed_out: int
    elapsed_mean: float
    elapsed_median: float
    elapsed_max: float
    refs_total: int
    refs_mo: int
    refs_mse: int
    missing_total: int
    missing_mean: float
    improvements_nonempty: int


def rec_status(rec: dict[str, Any]) -> str:
    st = rec.get("claim_verified")
    if st in VALID_STATUSES:
        return str(st)
    if rec.get("parse_error"):
        return "parse"
    return "parse"


def rec_score(rec: dict[str, Any]) -> int:
    return SCORE_MAP.get(rec_status(rec), 0)


def load_latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    by_node: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            node_id = rec.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError(f"{path}:{lineno}: missing/invalid node_id")
            # Keep the most recent row for each node (resume-friendly).
            by_node[node_id] = rec
    return by_node


def summarize(label: str, rows: dict[str, dict[str, Any]]) -> Summary:
    vals = list(rows.values())
    n = len(vals)
    status_counts = {k: 0 for k in ("verified", "plausible", "gap", "error", "parse")}
    retries_gt1 = 0
    timed_out = 0
    elapsed_values: list[float] = []
    refs_total = 0
    refs_mo = 0
    refs_mse = 0
    missing_total = 0
    improvements_nonempty = 0
    score_sum = 0

    for rec in vals:
        st = rec_status(rec)
        status_counts[st] += 1
        score_sum += rec_score(rec)

        attempts = rec.get("attempts", 1)
        if isinstance(attempts, int) and attempts > 1:
            retries_gt1 += 1
        if bool(rec.get("timed_out", False)):
            timed_out += 1

        elapsed = rec.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            elapsed_values.append(float(elapsed))

        refs = rec.get("math_se_references")
        if isinstance(refs, list):
            refs_total += len(refs)
            for r in refs:
                if not isinstance(r, dict):
                    continue
                site = r.get("site")
                if site == "mathoverflow.net":
                    refs_mo += 1
                elif site == "math.stackexchange.com":
                    refs_mse += 1

        missing = rec.get("missing_assumptions")
        if isinstance(missing, list):
            missing_total += len([x for x in missing if isinstance(x, str) and x.strip()])

        suggestion = rec.get("suggested_improvement")
        if isinstance(suggestion, str) and suggestion.strip():
            improvements_nonempty += 1

    if elapsed_values:
        elapsed_mean = mean(elapsed_values)
        elapsed_median = median(elapsed_values)
        elapsed_max = max(elapsed_values)
    else:
        elapsed_mean = 0.0
        elapsed_median = 0.0
        elapsed_max = 0.0

    score_avg = (score_sum / n) if n else 0.0
    missing_mean = (missing_total / n) if n else 0.0

    return Summary(
        label=label,
        n=n,
        verified=status_counts["verified"],
        plausible=status_counts["plausible"],
        gap=status_counts["gap"],
        error=status_counts["error"],
        parse=status_counts["parse"],
        score_sum=score_sum,
        score_avg=score_avg,
        retries_gt1=retries_gt1,
        timed_out=timed_out,
        elapsed_mean=elapsed_mean,
        elapsed_median=elapsed_median,
        elapsed_max=elapsed_max,
        refs_total=refs_total,
        refs_mo=refs_mo,
        refs_mse=refs_mse,
        missing_total=missing_total,
        missing_mean=missing_mean,
        improvements_nonempty=improvements_nonempty,
    )


def edge_consistency(rows: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return (pairs, target_stronger_than_source, hard_jumps)."""
    pairs = 0
    stronger = 0
    hard_jumps = 0
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if not isinstance(s, str) or not isinstance(t, str):
            continue
        if s not in rows or t not in rows:
            continue
        pairs += 1
        s_score = rec_score(rows[s])
        t_score = rec_score(rows[t])
        if t_score > s_score:
            stronger += 1
            if s_score == 0 and t_score == 2:
                hard_jumps += 1
    return pairs, stronger, hard_jumps


def render_summary_table(s: Summary) -> str:
    mo_share = (s.refs_mo / s.refs_total) if s.refs_total else 0.0
    mse_share = (s.refs_mse / s.refs_total) if s.refs_total else 0.0
    return "\n".join(
        [
            f"- rows: {s.n}",
            f"- status: verified={s.verified}, plausible={s.plausible}, gap={s.gap}, error={s.error}, parse={s.parse}",
            f"- score: sum={s.score_sum}, avg={s.score_avg:.3f}",
            f"- retries/timeouts: attempts>1={s.retries_gt1}, timed_out={s.timed_out}",
            f"- elapsed seconds: mean={s.elapsed_mean:.1f}, median={s.elapsed_median:.1f}, max={s.elapsed_max:.1f}",
            f"- references: total={s.refs_total}, mo={s.refs_mo} ({mo_share:.1%}), mse={s.refs_mse} ({mse_share:.1%})",
            f"- missing assumptions: total={s.missing_total}, mean/row={s.missing_mean:.2f}",
            f"- non-empty suggested improvements: {s.improvements_nonempty}",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True, help="Baseline results JSONL")
    ap.add_argument("--candidate", type=Path, required=True, help="Candidate results JSONL")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--candidate-label", default="candidate")
    ap.add_argument("--wiring", type=Path, default=None, help="Optional wiring JSON for edge-consistency diagnostics")
    ap.add_argument("--output-md", type=Path, default=None, help="Optional markdown output path")
    ap.add_argument("--output-json", type=Path, default=None, help="Optional machine-readable output path")
    args = ap.parse_args()

    base_rows = load_latest_rows(args.baseline)
    cand_rows = load_latest_rows(args.candidate)

    base_summary = summarize(args.baseline_label, base_rows)
    cand_summary = summarize(args.candidate_label, cand_rows)

    common = sorted(set(base_rows) & set(cand_rows))
    only_base = sorted(set(base_rows) - set(cand_rows))
    only_cand = sorted(set(cand_rows) - set(base_rows))

    better = 0
    worse = 0
    tie = 0
    changed: list[str] = []
    changed_rows: list[dict[str, str]] = []
    for node_id in common:
        sb = rec_score(base_rows[node_id])
        sc = rec_score(cand_rows[node_id])
        if sc > sb:
            better += 1
        elif sc < sb:
            worse += 1
        else:
            tie += 1
        st_b = rec_status(base_rows[node_id])
        st_c = rec_status(cand_rows[node_id])
        if st_b != st_c:
            changed.append(f"- {node_id}: {st_b} -> {st_c}")
            changed_rows.append({"node_id": node_id, "from": st_b, "to": st_c})

    edge_diag: dict[str, int] | None = None
    lines = [
        "# Proof-Polish Arm Comparison",
        "",
        f"Baseline: `{args.baseline_label}` ({args.baseline})",
        f"Candidate: `{args.candidate_label}` ({args.candidate})",
        "",
        "## Baseline Summary",
        "",
        render_summary_table(base_summary),
        "",
        "## Candidate Summary",
        "",
        render_summary_table(cand_summary),
        "",
        "## Pairwise Node Comparison",
        "",
        f"- common nodes: {len(common)}",
        f"- candidate better: {better}",
        f"- candidate worse: {worse}",
        f"- tie: {tie}",
        f"- only in baseline: {len(only_base)}",
        f"- only in candidate: {len(only_cand)}",
        "",
        "## Node Status Changes",
        "",
    ]
    if changed:
        lines.extend(changed)
    else:
        lines.append("- none")

    if args.wiring:
        wiring = json.loads(args.wiring.read_text(encoding="utf-8"))
        edges = wiring.get("edges", [])
        if not isinstance(edges, list):
            raise ValueError(f"{args.wiring}: 'edges' must be a list")

        b_pairs, b_stronger, b_hard = edge_consistency(base_rows, edges)
        c_pairs, c_stronger, c_hard = edge_consistency(cand_rows, edges)
        edge_diag = {
            "baseline_pairs": b_pairs,
            "candidate_pairs": c_pairs,
            "baseline_target_stronger_than_source": b_stronger,
            "candidate_target_stronger_than_source": c_stronger,
            "baseline_hard_jumps": b_hard,
            "candidate_hard_jumps": c_hard,
        }
        lines.extend(
            [
                "",
                "## Edge-Consistency Diagnostic",
                "",
                f"- evaluated edge pairs: baseline={b_pairs}, candidate={c_pairs}",
                f"- target stronger than source (lower is usually better): baseline={b_stronger}, candidate={c_stronger}",
                f"- hard jumps source<=gap to target=verified (lower is better): baseline={b_hard}, candidate={c_hard}",
            ]
        )

    out = "\n".join(lines) + "\n"
    payload = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline": asdict(base_summary),
        "candidate": asdict(cand_summary),
        "pairwise": {
            "common_nodes": len(common),
            "candidate_better": better,
            "candidate_worse": worse,
            "tie": tie,
            "only_in_baseline": len(only_base),
            "only_in_candidate": len(only_cand),
            "node_status_changes": changed_rows,
        },
    }
    if edge_diag is not None:
        payload["edge_consistency"] = edge_diag
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(out, encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
