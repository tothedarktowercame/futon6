#!/usr/bin/env python3
"""Build Mark8's deterministic, report-only model-call allocation plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "futon6/mark8-allocation/v1"
QUEUE_SCHEMA = "futon6/mark8-precheck-queue/v1"


def load_candidates(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        data = json.loads(path.read_text())
        values = data if isinstance(data, list) else [data]
        if not all(isinstance(value, dict) for value in values):
            raise ValueError(f"{path}: candidate JSON must be an object or list of objects")
        rows.extend(values)
    return rows


def identity(candidate: dict[str, Any]) -> tuple[str, str]:
    item = candidate.get("passage-id") or candidate.get("candidate-id") or candidate.get("id")
    paper = candidate.get("paper-id")
    return str(item or ""), str(paper or "")


def deterministic_defect(candidate: dict[str, Any]) -> str | None:
    """Return only defects knowable before a model call.

    Model-produced graph properties, including cycles, are intentionally absent.
    """
    item, paper = identity(candidate)
    if not item:
        return "missing-item-id"
    if not paper:
        return "missing-paper-id"
    span = candidate.get("window-lines", candidate.get("source-span"))
    if (not isinstance(span, list) or len(span) != 2 or
            not all(isinstance(value, int) for value in span) or
            span[0] < 0 or span[1] < span[0]):
        return "invalid-source-span"
    clause_spans = candidate.get("clause-spans")
    if clause_spans is not None:
        if (not isinstance(clause_spans, list) or not clause_spans or
                any(not isinstance(value, list) or len(value) != 2 or
                    not all(isinstance(point, int) for point in value) or
                    value[0] < span[0] or value[1] < value[0] or value[1] > span[1]
                    for value in clause_spans)):
            return "invalid-clause-spans"
    return None


def build_plan(s3: list[dict[str, Any]], s4: list[dict[str, Any]], *,
               budget: int, policy: str) -> tuple[list[dict], dict, dict]:
    if budget < 0:
        raise ValueError("model-call-budget must be a nonnegative integer")
    tagged = [("S3", row) for row in s3] + [("S4", row) for row in s4]
    keys = [(stage, identity(row)[0]) for stage, row in tagged]
    duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
    if duplicates:
        raise ValueError(f"duplicate candidate identities: {duplicates}")

    queue: list[dict] = []
    for stage, candidate in tagged:
        item, paper = identity(candidate)
        defect = deterministic_defect(candidate)
        queue.append({
            "schema": QUEUE_SCHEMA, "stage": stage, "item-id": item or None,
            "paper-id": paper or None, "precheck": "refused" if defect else "eligible",
            "reason": defect, "baseline-selected": (stage == "S3" or
                bool(candidate.get("baseline-selected", False))),
        })
    queue.sort(key=lambda row: (row["stage"], row["paper-id"] or "", row["item-id"] or ""))

    eligible = [row for row in queue if row["precheck"] == "eligible"]
    baseline = [row for row in eligible if row["baseline-selected"]]
    fill = [row for row in eligible if row["stage"] == "S4" and not row["baseline-selected"]]
    ordered = baseline + fill
    selected_keys = {(row["stage"], row["item-id"]) for row in ordered[:budget]}
    for row in queue:
        key = (row["stage"], row["item-id"])
        if row["precheck"] == "refused":
            row["allocation"] = "precheck-refused"
        elif key in selected_keys:
            row["allocation"] = "selected"
        else:
            row["allocation"] = "deferred"

    selected = [row for row in queue if row["allocation"] == "selected"]
    deferred = [row for row in queue if row["allocation"] == "deferred"]
    unspent = budget - len(selected)
    allocation = {
        "schema": SCHEMA, "allocation-policy": policy, "model-call-budget": budget,
        "selected-calls": len(selected), "unspent-calls": unspent,
        "selected": [{"stage": row["stage"], "item-id": row["item-id"]} for row in selected],
        "deferred": [{"stage": row["stage"], "item-id": row["item-id"]} for row in deferred],
    }
    baseline_s4 = sum(row["stage"] == "S4" and row["baseline-selected"] for row in selected)
    analysis = {
        "schema": SCHEMA, "model-call-budget": budget, "input-items": len(queue),
        "eligible-items": len(eligible),
        "precheck-refused-items": len(queue) - len(eligible),
        "selected-calls": len(selected), "unspent-calls": unspent,
        "selected-by-stage": {stage: sum(row["stage"] == stage for row in selected)
                              for stage in ("S3", "S4")},
        "additional-s4-selections": sum(
            row["stage"] == "S4" and not row["baseline-selected"] for row in selected),
        "baseline-s4-selections-retained": baseline_s4,
        "budget-conserved": len(selected) + unspent == budget,
    }
    assert len(selected) + unspent == budget
    assert not (selected_keys & {(row["stage"], row["item-id"]) for row in deferred})
    return queue, allocation, analysis


def write_outputs(out_dir: Path, queue: list[dict], allocation: dict, analysis: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "precheck-queue.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in queue))
    for name, doc in (("allocation.json", allocation), ("allocation-analysis.json", analysis)):
        (out_dir / name).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3", type=Path, action="append", default=[])
    parser.add_argument("--s4", type=Path, action="append", default=[])
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    write_outputs(args.out_dir, *build_plan(load_candidates(args.s3), load_candidates(args.s4),
                                            budget=args.budget, policy=args.policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
