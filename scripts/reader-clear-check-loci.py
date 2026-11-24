#!/usr/bin/env python3
"""Check that reader JSONL loci occur verbatim in the supplied texts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for candidate in (_REPO / "src", _REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from futon6.reader_clear_loci import check_readings, render_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", dest="print_path", required=True, type=Path)
    parser.add_argument("--informal", dest="informal_path", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("readings", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_readings(args.print_path, args.informal_path, args.readings)
    if args.json_output:
        print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(render_text(result))
    return 1 if result.failure_count else 0


if __name__ == "__main__":
    sys.exit(main())
