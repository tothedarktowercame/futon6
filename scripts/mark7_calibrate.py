#!/usr/bin/env python3
"""Measure achieved throughput across a concurrency ladder, before booking a window.

The 0919b probe ran single-stream on one GPU and managed ~20 output tok/s. Every
window estimate for the CT-wide run rests on how much batching improves that, and
nothing so far has measured it on the real allocation. This runs the S3 loop over a
slice of candidates at several concurrencies and reports what was actually achieved,
so the window question is settled by a number rather than argued from a guess.

Each rung gets its own output directory: the loop carries prior acceptances, so
sharing one would make every rung after the first measure the filesystem.

Sizing: a rung needs enough items to fill its batch before the rate means anything,
but concurrency 1 does not need 200 of them to give a stable figure. Items per rung
is 4x the concurrency, clamped to [16, --candidates], which keeps the whole ladder
inside the `short` partition's 4h cap.

    scripts/mark7_calibrate.py --candidates-dir <dir> [--ladder 1,8,32,64] [--out cal.json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import futon6_config as config  # noqa: E402

LOOP = ROOT / "scripts" / "mark3_iatc_loop.py"


def rung_size(concurrency: int, available: int, floor: int = 16) -> int:
    return max(floor, min(available, 4 * concurrency))


def output_tokens(outdir: Path) -> int:
    """Model output only: the nodes/steps answers, not the checker's sidecars."""
    attempts = outdir / ".attempts"
    produced = list(attempts.rglob("*.nodes.json")) + list(attempts.rglob("*.steps.json"))
    return sum(p.stat().st_size for p in produced) // 4


def run_rung(candidates: list[Path], concurrency: int, workdir: Path, model: str) -> dict:
    slice_dir = workdir / f"c{concurrency}" / "candidates"
    slice_dir.mkdir(parents=True)
    for candidate in candidates:
        shutil.copy2(candidate, slice_dir / candidate.name)
    outdir = workdir / f"c{concurrency}" / "out"

    env = dict(os.environ)
    env["RUN_ID"] = f"calibrate-c{concurrency}"
    env.pop("FUTON6_STAGE_INVOCATION", None)        # a fresh attempt history per rung
    argv = [*config.python_argv(), str(LOOP), "--candidates", str(slice_dir),
            "--out", str(outdir), "--backend", os.environ.get("CALIBRATE_BACKEND", "openai"),
            "--model", model, "--concurrency", str(concurrency)]

    started = time.monotonic()
    done = subprocess.run(argv, capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - started

    graphs = len([p for p in outdir.glob("*.edn") if not p.name.endswith(".rung2.edn")]) \
        if outdir.exists() else 0
    tokens = output_tokens(outdir) if outdir.exists() else 0
    return {"concurrency": concurrency, "items": len(candidates), "seconds": round(elapsed, 1),
            "graphs": graphs, "output-tokens": tokens,
            "output-tok-per-s": round(tokens / elapsed, 1) if elapsed > 0 else None,
            "items-per-min": round(graphs / elapsed * 60, 2) if elapsed > 0 else None,
            "loop-exit": done.returncode,
            "stderr-tail": done.stderr.strip()[-400:] if done.returncode not in (0, 1) else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--ladder", default="1,8,32,64")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--keep", action="store_true", help="retain per-rung outputs for inspection")
    args = ap.parse_args()

    pool = sorted(Path(args.candidates_dir).glob("*.candidate.json"))[: args.candidates]
    if not pool:
        print(f"no candidates in {args.candidates_dir}", file=sys.stderr)
        return 2
    ladder = [int(c) for c in args.ladder.split(",") if c.strip()]

    inventory = config.hardware()
    gpus = max(1, inventory["count"])
    report = {"schema-version": 1, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "hardware": inventory, "serving": config.serving(),
              "model": config.model(), "endpoint": config.effective()["endpoint"],
              "candidate-pool": len(pool), "rungs": []}

    workdir = Path(tempfile.mkdtemp(prefix="mark7-calibrate-"))
    print(f"== calibrating on {gpus} GPU(s), {len(pool)} candidates available, work in {workdir}")
    try:
        for concurrency in ladder:
            n = rung_size(concurrency, len(pool))
            print(f"-- concurrency {concurrency}: {n} candidates", flush=True)
            result = run_rung(pool[:n], concurrency, workdir, report["model"])
            result["output-tok-per-s-per-gpu"] = (
                round(result["output-tok-per-s"] / gpus, 1) if result["output-tok-per-s"] else None)
            report["rungs"].append(result)
            print(f"   {result['seconds']}s · {result['graphs']} graphs · "
                  f"{result['output-tok-per-s']} tok/s "
                  f"({result['output-tok-per-s-per-gpu']}/GPU)", flush=True)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    base = next((r for r in report["rungs"] if r["concurrency"] == 1), None)
    print(f"\n{'conc':>5} {'items':>6} {'sec':>8} {'tok/s':>9} {'/GPU':>8} {'vs c=1':>8}")
    for r in report["rungs"]:
        speedup = (f"{r['output-tok-per-s'] / base['output-tok-per-s']:.1f}x"
                   if base and base["output-tok-per-s"] and r["output-tok-per-s"] else "-")
        print(f"{r['concurrency']:>5} {r['items']:>6} {r['seconds']:>8} "
              f"{str(r['output-tok-per-s']):>9} {str(r['output-tok-per-s-per-gpu']):>8} {speedup:>8}")
    print(f"\nThe CT-wide bar is 130-166 output tok/s per GPU on 8 GPUs "
          f"(proportionally less on more). Written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
