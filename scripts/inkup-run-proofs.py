#!/usr/bin/env python3
"""Write mechanical proof measures from an extracted Mark7 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from futon6.inkup_run_proofs import write_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_run(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()

