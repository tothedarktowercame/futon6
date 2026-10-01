#!/usr/bin/env python3
"""Evaluate frozen wooliness scores against explicit, keyed post-run records."""
from __future__ import annotations

import argparse
import json
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
    if outcomes.get("schema") != OUTCOME_SCHEMA or quotes.get("schema") != QUOTE_SCHEMA:
        raise ValueError("outcome and quote inputs must use the frozen Mark8 schemas")
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
        values = {name: float(w[name]) for name in ("W", "U", "C", "D")}
        values["baseline-proxy"] = float(o["baseline-proxy"])
        values["agrees-share"] = float(q["agrees-share"])
        if any(not 0 <= value <= 1 for value in values.values()):
            raise ValueError(f"{identifier}: scores and shares must lie in [0,1]")
        points.append({"passage-id": identifier, "paper-id": str(w["paper-id"]),
                       **{name: values[name] for name in ("W", "U", "C", "D")},
                       "outcome": bool(o["weak-extraction"]),
                       "agrees-share": values["agrees-share"],
                       "baseline-proxy": values["baseline-proxy"]})
    w_auc = auc([row["W"] for row in points], [row["outcome"] for row in points])
    baseline_auc = auc([row["baseline-proxy"] for row in points],
                       [row["outcome"] for row in points])
    coverage = len(points) / len(wool) if wool else 0.0
    adequate = len(points) >= MIN_JOINED and coverage >= MIN_COVERAGE and w_auc is not None and baseline_auc is not None
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
