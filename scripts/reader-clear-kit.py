#!/usr/bin/env python3
"""Build reader-clear text kits from a stored sample and render head."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from futon6.reader_clear_kit import build_kits  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--apm-problems", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        kits = build_kits(args.head_dir, args.apm_problems, args.sample, args.out)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"reader-clear-kit: {error}", file=sys.stderr)
        return 2
    print(f"{len(kits)} kits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
