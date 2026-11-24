#!/usr/bin/env python3
"""Draw a deterministic stratified sample of zero-residual-ink modules."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from futon6.reader_clear_draw import build_draw, write_draw  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--apm-problems", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--controls", type=int, default=5)
    parser.add_argument("--exclude", nargs="+", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        draw = build_draw(args.head_dir, args.apm_problems, args.seed, n=args.n,
                          controls=args.controls, exclude=args.exclude)
        write_draw(args.output, draw)
    except (OSError, ValueError, KeyError) as error:
        print(f"reader-clear-draw: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
