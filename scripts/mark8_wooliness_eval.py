#!/usr/bin/env python3
"""Evaluate frozen wooliness scores against explicit, keyed post-run records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "futon6/mark8-wooliness-evaluation/v1"
OUTCOME_SCHEMA = "futon6/mark8-frozen-outcomes/v1"
QUOTE_SCHEMA = "futon6/mark8-frozen-quote-agreement/v1"
INPUT_FILES = ("wooliness", "outcomes", "quote-agreement")
MIN_COVERAGE = 0.80
MIN_JOINED = 20
MIN_AUC = 0.65
ATTENTION = {"W": 0.50, "quote-disagreement": 0.30, "baseline-proxy": 0.20}
WOOLINESS_SCHEMA = "futon6/mark8-wooliness/v1"
OUTCOME_STATUSES = frozenset({"accepted", "refused", "rejected", "errored", "deferred"})


def identifier(row: dict, *, paper: bool = False) -> None:
    if not isinstance(row.get("passage-id"), str) or not row["passage-id"]:
        raise ValueError("passage-id must be a nonempty string")
    if paper and (not isinstance(row.get("paper-id"), str) or not row["paper-id"]):
        raise ValueError(f"{row['passage-id']}: paper-id must be a nonempty string")


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite JSON number")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{label} must lie in [0,1]")
    return result


def validate_inputs(wooliness: Any, outcomes: Any, quotes: Any) -> None:
    documents = ((wooliness, WOOLINESS_SCHEMA, "wooliness"),
                 (outcomes, OUTCOME_SCHEMA, "outcomes"),
                 (quotes, QUOTE_SCHEMA, "quote-agreement"))
    for document, schema, label in documents:
        if (not isinstance(document, dict) or document.get("schema") != schema or
                not isinstance(document.get("records"), list) or
                not all(isinstance(row, dict) for row in document["records"])):
            raise ValueError(f"{label} must satisfy schema {schema}")
    if not isinstance(outcomes.get("baseline-name"), str) or not outcomes["baseline-name"].strip():
        raise ValueError("outcomes must name a nonempty baseline proxy")
    for row in wooliness["records"]:
        identifier(row, paper=True)
        for field in ("W", "U", "C", "D"):
            number(row.get(field), f"{row['passage-id']}.{field}")
    for row in outcomes["records"]:
        identifier(row)
        if row.get("status") not in OUTCOME_STATUSES:
            raise ValueError(f"{row['passage-id']}: invalid outcome status")
        if not isinstance(row.get("weak-extraction"), bool):
            raise ValueError(f"{row['passage-id']}: weak-extraction must be a JSON boolean")
        number(row.get("baseline-proxy"), f"{row['passage-id']}.baseline-proxy")
    for row in quotes["records"]:
        identifier(row)
        number(row.get("agrees-share"), f"{row['passage-id']}.agrees-share")


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Pairwise ROC AUC; equal positive/negative scores contribute one half."""
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    credit = sum(1.0 if p > n else 0.5 if p == n else 0.0
                 for p in positive for n in negative)
    return credit / (len(positive) * len(negative))


def index(rows: list[dict], key: str) -> tuple[dict[str, dict], list[str]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    duplicates = sorted(identifier for identifier, values in grouped.items() if len(values) != 1)
    return {identifier: values[0] for identifier, values in grouped.items() if len(values) == 1}, duplicates


def calibration(points: list[dict]) -> list[dict]:
    bins = []
    for number in range(10):
        low, high = number / 10, (number + 1) / 10
        rows = [row for row in points if low <= row["W"] <= high
                and (number == 9 or row["W"] < high)]
        bins.append({"bin": number, "low": low, "high": high,
                     "count": len(rows),
                     "weak-extraction-rate": (sum(row["outcome"] for row in rows) / len(rows)
                                              if rows else None),
                     "mean-W": (sum(row["W"] for row in rows) / len(rows) if rows else None)})
    return bins


def evaluate(wooliness: dict, outcomes: dict, quotes: dict, *, top_k: int = 20) -> dict:
    validate_inputs(wooliness, outcomes, quotes)
    wool, wool_dups = index(wooliness.get("records", []), "passage-id")
    outcome, outcome_dups = index(outcomes.get("records", []), "passage-id")
    quote, quote_dups = index(quotes.get("records", []), "passage-id")
    refused = sorted(identifier for identifier, row in outcome.items()
                     if row.get("status", "accepted") != "accepted")
    accepted_outcome = {identifier: row for identifier, row in outcome.items() if identifier not in refused}
    duplicate_ids = set(wool_dups + outcome_dups + quote_dups)
    joined_ids = sorted((set(wool) & set(accepted_outcome) & set(quote)) - duplicate_ids)
    points = []
    for identifier in joined_ids:
        w, o, q = wool[identifier], accepted_outcome[identifier], quote[identifier]
        values = {name: number(w[name], f"{identifier}.{name}") for name in ("W", "U", "C", "D")}
        values["baseline-proxy"] = number(o["baseline-proxy"], f"{identifier}.baseline-proxy")
        values["agrees-share"] = number(q["agrees-share"], f"{identifier}.agrees-share")
        points.append({"passage-id": identifier, "paper-id": str(w["paper-id"]),
                       **{name: values[name] for name in ("W", "U", "C", "D")},
                       "outcome": o["weak-extraction"],
                       "agrees-share": values["agrees-share"],
                       "baseline-proxy": values["baseline-proxy"]})
    w_auc = auc([row["W"] for row in points], [row["outcome"] for row in points])
    baseline_auc = auc([row["baseline-proxy"] for row in points],
                       [row["outcome"] for row in points])
    coverage = len(points) / len(wool) if wool else 0.0
    adequate = (not duplicate_ids and len(points) >= MIN_JOINED and coverage >= MIN_COVERAGE
                and w_auc is not None and baseline_auc is not None)
    status = ("eligible" if adequate and w_auc >= MIN_AUC and w_auc >= baseline_auc else
              "report-only" if adequate else "insufficient")
    attention = []
    for row in points:
        components = {"W": row["W"], "quote-disagreement": 1 - row["agrees-share"],
                      "baseline-proxy": row["baseline-proxy"]}
        score = sum(ATTENTION[name] * value for name, value in components.items())
        attention.append({"passage-id": row["passage-id"], "paper-id": row["paper-id"],
                          "attention-score": score, "components": components})
    attention.sort(key=lambda row: (-row["attention-score"], row["passage-id"]))
    all_wool, all_outcome, all_quote = set(wool), set(outcome), set(quote)
    return {
        "schema": SCHEMA,
        "gate": {"status": status, "minimum-coverage": MIN_COVERAGE,
                 "minimum-joined": MIN_JOINED, "minimum-W-AUC": MIN_AUC,
                 "requires-W-at-least-baseline": True},
        "join": {"wooliness": len(wool), "outcomes": len(outcome), "quote-agreement": len(quote),
                 "joined": len(points), "coverage": coverage, "refused": len(refused),
                 "duplicate-count": len(duplicate_ids),
                 "duplicates": {"wooliness": wool_dups, "outcomes": outcome_dups,
                                "quote-agreement": quote_dups},
                 "refused-ids": refused,
                 "unmatched": {"wooliness": sorted(all_wool - set(joined_ids)),
                               "outcomes": sorted(all_outcome - set(joined_ids)),
                               "quote-agreement": sorted(all_quote - set(joined_ids))}},
        "auc": {"W": w_auc, "baseline": baseline_auc,
                "baseline-name": str(outcomes.get("baseline-name", "unnamed"))},
        "calibration": calibration(points),
        "C4-points": points,
        "C6-attention": {"score": ATTENTION, "top-k": top_k, "rows": attention[:top_k]},
    }


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wooliness", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--quote-agreement", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")
    report = evaluate(load(args.wooliness), load(args.outcomes), load(args.quote_agreement),
                      top_k=args.top_k)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
