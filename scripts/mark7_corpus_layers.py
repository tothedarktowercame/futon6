#!/usr/bin/env python3
"""Build Mark7's optional corpus-wide WARP and TAPESTRY layers.

The two phases are separate because WARP must be frozen before paid WEFT mining,
while TAPESTRY needs the run's completed marks and citation resolutions.  Both
write only beneath the run directory named by the immutable run manifest.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import futon6_config as config
import run_manifest


ROOT = Path(__file__).resolve().parents[1]
BASE_STAGES = ("S1a", "S1b", "S1c", "S2", "S3", "S4a", "S4b", "S5", "S4c", "S6t", "S6b")


def enabled(doc: dict, feature: str) -> bool:
    return bool((doc.get("features") or {}).get(feature, False))


def run_command(argv: list[str], env: dict[str, str]) -> None:
    subprocess.run([*config.python_argv(), *argv], cwd=ROOT, env=env, check=True)


def warp_phase(run_dir: Path, doc: dict) -> None:
    if not enabled(doc, "warp"):
        print("WARP disabled by run manifest")
        return
    env = config.child_environment()
    env["FUTON6_CORPUS_IDS"] = str(run_dir / doc["ids"])
    argv = ["scripts/warp_run.py", "--manifest", str(config.warp() / "warp-manifest.json")]
    for stage in BASE_STAGES:
        argv.extend(("--stage", stage))
    run_command(argv, env)
    report = config.warp() / "concept-index-report.md"
    run_command([
        "scripts/sfc_concept_index.py", "--rebuild",
        "--usage", str(config.warp() / "concept-usage.json"),
        "--def-snippets", str(config.warp() / "def-snippets.json"),
        "--defined-index", str(config.warp() / "defined-index.json"),
        "--concept-encyclopedia", str(config.concept_encyclopedia()),
        "--out", str(config.warp() / "concept-index.json"),
        "--report", str(report),
    ], env)


def tapestry_phase(run_dir: Path, doc: dict) -> None:
    if not enabled(doc, "tapestry"):
        print("TAPESTRY disabled by run manifest")
        return
    artifacts = doc["artifacts"]
    marks = run_manifest.contained(run_dir, artifacts["marks"])
    cite_out = run_manifest.contained(run_dir, artifacts["cite-resolution"])
    tapestry_out = run_manifest.contained(run_dir, artifacts["tapestry"])
    tapestry_out.mkdir(parents=True, exist_ok=True)
    env = config.child_environment()
    run_command([
        "scripts/cite_resolve.py", "--ids", str(run_dir / doc["ids"]),
        "--golden-dir", str(marks), "--bib-index", str(config.warp() / "bib-index.json"),
        "--citations", str(config.warp() / "citations.json"), "--out-dir", str(cite_out),
    ], env)
    concept_dir = config.subject_data() / "concept-encyclopedia" / config.subject()
    run_command([
        "scripts/mark3_thread_tapestry.py", "--golden-dir", str(marks),
        "--concept-dir", str(concept_dir), "--cites", str(cite_out),
        "--out", str(tapestry_out / "concept-phylogeny.json"),
    ], env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("warp", "tapestry"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    doc = run_manifest.load(run_dir)
    (warp_phase if args.phase == "warp" else tapestry_phase)(run_dir, doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
