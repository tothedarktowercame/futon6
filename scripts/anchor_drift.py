"""Locate node text only in the candidate source-window the model was shown.

Coordinates are inclusive candidate labels: source-window.split("\\n")[0]
corresponds to window-lines[0]. No paper reconstruction or line-base inference.
The lexical locator is a heuristic, not a certificate of exact highlighting.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

from edn_format import Keyword, loads


def toks(s):
    return set(w.lower() for w in re.findall(r"[A-Za-z]{4,}", s))


def candidate_lines(graph, candidate):
    for graph_key, candidate_key in (("paper/id", "paper-id"), ("passage/id", "passage-id")):
        if graph.get(Keyword(graph_key)) != candidate.get(candidate_key):
            raise ValueError("candidate identity does not match graph")
    bounds = candidate.get("window-lines")
    text = candidate.get("source-window")
    if not (isinstance(bounds, list) and len(bounds) == 2
            and all(type(n) is int for n in bounds) and bounds[0] <= bounds[1]
            and isinstance(text, str)):
        raise ValueError("candidate lacks valid window-lines/source-window")
    lo, hi = bounds
    lines = text.split("\n")
    if hi - lo >= len(lines):
        raise ValueError("candidate window bounds exceed source-window")
    return [(lo + i, line) for i, line in enumerate(lines) if lo + i <= hi]


def measure(graph, candidate):
    lines = candidate_lines(graph, candidate)
    lo, hi = candidate["window-lines"]
    result = {"total": 0, "located": 0, "invalid": 0, "drift": []}
    for node in graph.get(Keyword("nodes"), []):
        span = node.get(Keyword("source"), {}).get(Keyword("lines"), [])
        if not (len(span) == 2 and all(type(n) is int for n in span)
                and lo <= span[0] <= span[1] <= hi):
            result["invalid"] += 1
            continue
        a, b = span
        tt = toks(node.get(Keyword("text"), ""))
        if len(tt) < 3:
            continue
        result["total"] += 1
        scores = [(len(tt & toks(line)) / len(tt), number) for number, line in lines]
        bestscore = max(score for score, _ in scores)
        best = [number for score, number in scores if score == bestscore]
        # Repeated text does not uniquely identify a line; do not choose one
        # arbitrary occurrence and report its distance as measured drift.
        if bestscore >= 0.6 and len(best) == 1:
            result["located"] += 1
            number = best[0]
            result["drift"].append(0 if a <= number <= b else min(abs(number-a), abs(number-b)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graphs", nargs="?", type=Path, default=Path("data/iatc-argument-graphs/run"))
    ap.add_argument("--candidates-dir", type=Path)
    args = ap.parse_args()
    files = [args.graphs] if args.graphs.is_file() else sorted(args.graphs.glob("*.edn"))
    totals = {"total": 0, "located": 0, "invalid": 0, "drift": []}
    errors = 0
    for file in files:
        if file.name.endswith(".rung2.edn"):
            continue
        candidate = (args.candidates_dir or file.parent.parent / "candidates") / (file.stem + ".candidate.json")
        try:
            result = measure(loads(file.read_text()), json.loads(candidate.read_text()))
        except (OSError, ValueError) as exc:
            errors += 1
            print(f"ERROR {file}: {exc}")
            continue
        for key in totals:
            totals[key] += result[key]
    print("Coordinates: inclusive candidate labels; source-window index 0 = window-lines[0].")
    print(f"Uniquely located by lexical heuristic: {totals['located']}/{totals['total']}; invalid spans: {totals['invalid']}; input errors: {errors}")
    drift = totals["drift"]
    if drift:
        exact = sum(d == 0 for d in drift)
        off = [d for d in drift if d]
        print(f"Anchor covers located line: {exact}/{len(drift)} ({100*exact/len(drift):.0f}%)")
        print(f"Median drift when off: {statistics.median(off) if off else 0} lines")
    print("These heuristic matches do not certify exact source-line highlighting.")
    return 1 if errors or totals["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
