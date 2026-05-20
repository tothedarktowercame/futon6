#!/usr/bin/env python3
"""CLI wrapper for preregistered superpod QC expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from futon6.preregister_superpod_qc import (
    DEFAULT_BASELINE_DIR,
    build_report,
)


def main():
    ap = argparse.ArgumentParser(description="Emit preregistered QC expectations for a superpod manifest.")
    ap.add_argument("manifest", help="Path to manifest.json for the run being evaluated")
    ap.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR),
                    help="Directory of historical mark2 manifests (default: ~/code/storage/mark2/manifests)")
    ap.add_argument("--profile", choices=["broad-arxiv", "mfuton"], default="broad-arxiv",
                    help="Historical lane to use for expectation ranges")
    ap.add_argument("--output", default=None, help="Optional path to write JSON report")
    args = ap.parse_args()

    report = build_report(Path(args.manifest), Path(args.baseline_dir), args.profile)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(payload)


if __name__ == "__main__":
    main()
