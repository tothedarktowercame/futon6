#!/usr/bin/env python3
"""Census the Mark7 scope fixture used to ground the InkUp grammar."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = (
    HERE.parents[3] / "futon3c/web/scopes/fixtures/mark7-0708.2185.json"
)
DEFAULT_OUTPUT = HERE / "mark7-0708.2185-inventory.json"


def census(path: Path) -> dict:
    raw = path.read_bytes()
    doc = json.loads(raw)
    layer_counts = collections.Counter(span["layer"] for span in doc["spans"])
    kinds_by_layer = {}
    for layer in sorted(layer_counts):
        kinds_by_layer[layer] = dict(sorted(collections.Counter(
            span["kind"] for span in doc["spans"] if span["layer"] == layer
        ).items()))
    relations = collections.Counter(edge["relation"] for edge in doc["edges"])
    return {
        "schema": "inkup.mark7-scope-inventory/v0",
        "paper": doc["turn"].removeprefix("mark7-"),
        "fixture_record": doc["record"],
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "span_count": len(doc["spans"]),
        "edge_count": len(doc["edges"]),
        "layer_counts": dict(sorted(layer_counts.items())),
        "kinds_by_layer": kinds_by_layer,
        "edge_relations": dict(sorted(relations.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(census(args.fixture), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit(f"stale inventory: regenerate {args.output}")
        print(f"inventory current: {args.output}")
    else:
        args.output.write_text(rendered)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
