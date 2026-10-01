#!/usr/bin/env python3
"""Freeze Mark7 comprehension and quote measurements under exact candidate passage IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


OUTCOME_SCHEMA = "futon6/mark8-frozen-outcomes/v1"
QUOTE_SCHEMA = "futon6/mark8-frozen-quote-agreement/v1"
AUDIT_SCHEMA = "futon6/mark8-outcome-freeze-audit/v1"
BASELINE_NAME = "candidate-window-line-span-clipped-200/v1"
BASELINE_LINE_CLIP = 200
COMPREHENSION_VERDICTS = frozenset({
    "well-formed", "partial-comprehension", "weak-extraction", "weak-proof",
    "open-problem-bearing",
})
QUOTE_VERDICTS = frozenset({"agrees", "re-anchor", "unclear", "uncheckable"})


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encoded(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a nonfinite value")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_nonfinite(child, f"{label}[{index}]")


def unique(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    out = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{label} has missing/non-string {key}")
        if identifier in out:
            raise ValueError(f"duplicate {label} {key}: {identifier}")
        out[identifier] = row
    return out


def candidates(directory: Path) -> tuple[dict[str, dict], dict[str, str], str]:
    by_pid, by_passage, hashes = {}, {}, []
    for path in sorted(Path(directory).rglob("*.candidate.json"), key=lambda value: str(value)):
        payload = path.read_bytes()
        row = json.loads(payload)
        reject_nonfinite(row, str(path))
        stem = path.name.removesuffix(".candidate.json")
        if stem in by_pid:
            raise ValueError(f"duplicate candidate stem: {stem}")
        if not isinstance(row, dict) or row.get("proof-id") != stem:
            raise ValueError(f"{path}: candidate proof-id must exactly equal filename stem {stem}")
        passage = row.get("passage-id")
        if not isinstance(passage, str) or not passage:
            raise ValueError(f"{path}: missing/non-string passage-id")
        if passage in by_passage:
            raise ValueError(f"duplicate candidate passage-id: {passage}")
        lines = row.get("window-lines")
        if (not isinstance(lines, list) or len(lines) != 2 or
                not all(isinstance(value, int) and not isinstance(value, bool) for value in lines) or
                lines[0] < 1 or lines[1] < lines[0]):
            raise ValueError(f"{path}: invalid window-lines")
        by_pid[stem] = row
        by_passage[passage] = stem
        hashes.append((str(path.relative_to(directory)), digest_bytes(payload)))
    if not by_pid:
        raise ValueError("candidate directory has no candidate JSON")
    aggregate = digest_bytes(encoded({"files": hashes}))
    return by_pid, by_passage, aggregate


def baseline(candidate: dict) -> float:
    """Candidate-only proxy: inclusive source-window line span / 200, clipped to one."""
    lo, hi = candidate["window-lines"]
    return min(1.0, (hi - lo + 1) / BASELINE_LINE_CLIP)


def freeze(candidate_dir: Path, comprehension_path: Path, quote_path: Path) -> tuple[dict, dict, dict]:
    by_pid, by_passage, candidate_hash = candidates(candidate_dir)
    comprehension = read_json(comprehension_path)
    quote = read_json(quote_path)
    reject_nonfinite(comprehension, "comprehension")
    reject_nonfinite(quote, "quote-agreement")
    proof_rows = comprehension.get("proofs")
    graph_rows = quote.get("graphs")
    if not isinstance(proof_rows, list) or not all(isinstance(row, dict) for row in proof_rows):
        raise ValueError("comprehension proofs must be a list of objects")
    if not isinstance(graph_rows, list) or not all(isinstance(row, dict) for row in graph_rows):
        raise ValueError("quote graphs must be a list of objects")
    proofs = unique(proof_rows, "pid", "comprehension")
    graphs = unique(graph_rows, "passage", "quote")
    comprehension_hash = digest_bytes(encoded({
        "proofs": [proofs[pid] for pid in sorted(proofs)],
    }))
    quote_hash = digest_bytes(encoded({
        "graphs": [graphs[passage] for passage in sorted(graphs)],
    }))
    missing_candidates = sorted(set(proofs) - set(by_pid))
    mapped_passages = {by_pid[pid]["passage-id"] for pid in proofs if pid in by_pid}
    missing_quotes = sorted(mapped_passages - set(graphs))
    foreign_quotes = sorted(set(graphs) - mapped_passages)
    unused_candidates = sorted(set(by_pid) - set(proofs))
    if missing_candidates or missing_quotes or foreign_quotes:
        raise ValueError("candidate/comprehension/quote mapping is not exact: " + json.dumps({
            "missing-candidates": missing_candidates,
            "missing-quotes": missing_quotes, "foreign-quotes": foreign_quotes}, sort_keys=True))

    outcome_records, quote_records, zero_checkable = [], [], []
    for pid in sorted(proofs):
        candidate = by_pid[pid]
        passage = candidate["passage-id"]
        proof = proofs[pid]
        verdict = proof.get("verdict")
        if verdict not in COMPREHENSION_VERDICTS:
            raise ValueError(f"{pid}: invalid comprehension verdict {verdict!r}")
        outcome_records.append({"passage-id": passage, "status": "accepted",
                                "weak-extraction": verdict == "weak-extraction",
                                "baseline-proxy": baseline(candidate)})
        nodes = graphs[passage].get("nodes")
        if not isinstance(nodes, list) or not all(isinstance(node, dict) for node in nodes):
            raise ValueError(f"{passage}: quote nodes must be a list of objects")
        verdicts = [node.get("verdict") for node in nodes]
        invalid = sorted({value for value in verdicts if value not in QUOTE_VERDICTS}, key=str)
        if invalid:
            raise ValueError(f"{passage}: invalid quote verdict(s): {invalid}")
        checkable = sum(value != "uncheckable" for value in verdicts)
        agrees = sum(value == "agrees" for value in verdicts)
        if checkable == 0:
            zero_checkable.append(passage)
        else:
            quote_records.append({"passage-id": passage, "agrees-share": agrees / checkable,
                                  "agrees": agrees, "checkable": checkable})
    outcomes = {"schema": OUTCOME_SCHEMA, "baseline-name": BASELINE_NAME,
                "baseline": {"formula": "min(1, inclusive-window-line-span / 200)",
                             "clip-lines": BASELINE_LINE_CLIP, "uses-outcome-labels": False},
                "records": outcome_records}
    quotes = {"schema": QUOTE_SCHEMA, "records": quote_records}
    outcome_bytes, quote_bytes = encoded(outcomes), encoded(quotes)
    audit = {"schema": AUDIT_SCHEMA,
             "source-counts": {"candidates": len(by_pid), "comprehension": len(proofs),
                               "quote-agreement": len(graphs)},
             "joins": {"outcomes": len(outcome_records), "quotes": len(quote_records)},
             "omissions": {"unused-candidates": unused_candidates,
                           "zero-checkable": sorted(zero_checkable)},
             "hashes": {"candidate-set-sha256": candidate_hash,
                        "comprehension-semantic-sha256": comprehension_hash,
                        "quote-agreement-semantic-sha256": quote_hash,
                        "frozen-outcomes-sha256": digest_bytes(outcome_bytes),
                        "frozen-quotes-sha256": digest_bytes(quote_bytes)}}
    return outcomes, quotes, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--comprehension", type=Path, required=True)
    parser.add_argument("--quote-agreement", type=Path, required=True)
    parser.add_argument("--outcomes-out", type=Path, required=True)
    parser.add_argument("--quotes-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    outcomes, quotes, audit = freeze(args.candidates, args.comprehension, args.quote_agreement)
    for path, document in ((args.outcomes_out, outcomes), (args.quotes_out, quotes),
                           (args.audit_out, audit)):
        path.write_bytes(encoded(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
