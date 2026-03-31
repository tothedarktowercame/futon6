#!/usr/bin/env python3
"""Promote stabilized Lean material from one proof frame into ApmCanaries.Local."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def module_to_path(apm_lean_root: Path, module_name: str) -> Path:
    parts = module_name.split(".")
    if parts[:2] != ["ApmCanaries", "Local"]:
        raise SystemExit("destination module must live under ApmCanaries.Local.*")
    return apm_lean_root.joinpath(*parts).with_suffix(".lean")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspace_metadata", type=Path)
    ap.add_argument("--source", choices=["main", "scratch"], default="main")
    ap.add_argument("--dest-module", required=True,
                    help="Destination module under ApmCanaries.Local.*")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.workspace_metadata.resolve())
    apm_lean_root = Path(metadata["frame/shared-extension-root"]).resolve().parents[1]

    source_key = "lean-main" if args.source == "main" else "lean-scratch"
    src = Path(metadata["artifacts"][source_key]).resolve()
    if not src.exists():
        raise SystemExit(f"missing source file: {src}")

    dst = module_to_path(apm_lean_root, args.dest_module)
    if dst.exists() and not args.force:
        raise SystemExit(f"destination exists: {dst}")

    header = (
        "/-!\n"
        "Promoted from proof frame workspace.\n"
        f"- problem-id: {metadata['proof/problem-id']}\n"
        f"- frame-id: {metadata['frame/id']}\n"
        f"- source: {src}\n"
        f"- promoted-at: {now_utc()}\n"
        "-/\n\n"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(header + src.read_text(encoding="utf-8"), encoding="utf-8")
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
