#!/usr/bin/env python3
"""Build a structural passage index from the audited Mark7 fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from futon6 import inkup_scope_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=inkup_scope_index.DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records = inkup_scope_index.load_index(args.fixture)
    inkup_scope_index.write_jsonl(records, args.output)
    print(f"wrote {len(records)} spans to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
