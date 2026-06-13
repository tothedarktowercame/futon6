#!/usr/bin/env python3
"""Memory-safe batch runner for the DP detector.

The parent process stays light: it does not import ``dp_paper_view`` or the
concept authority stack. Each paper runs in a fresh Python subprocess, writes
its own golden JSON, exits, and releases memory. The parent runs sequentially
by default, logs every done/skipped/failed paper, and sleeps between papers to
avoid load spikes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
GOLDEN_DIR = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
LOG_DIR = ROOT / "data" / "warp" / "logs"
FLAGS = dict(with_ca=True, with_binders=True, with_scopes=True, with_xref=True)


def golden_path(paper: str) -> Path:
    return GOLDEN_DIR / f"fable-{paper}-dp-emacs.json"


def iter_eprint_ids() -> list[str]:
    suffixes = (".tar.gz", ".tex.gz", ".gz", ".tar", ".tex")
    ids: set[str] = set()
    for path in EPRINTS.iterdir():
        if not path.is_file():
            continue
        name = path.name
        for suffix in suffixes:
            if name.endswith(suffix):
                ids.add(name[: -len(suffix)])
                break
    return sorted(ids)


def candidates(limit: int | None) -> tuple[list[str], int, int, int]:
    have = {
        p.name[len("fable-") : -len("-dp-emacs.json")]
        for p in GOLDEN_DIR.glob("fable-*-dp-emacs.json")
    }
    ids = iter_eprint_ids()
    todo = [pid for pid in ids if pid not in have]
    if limit is not None:
        todo = todo[:limit]
    return todo, len(ids), len(have), len(ids) - len(have)


def worker_main(paper: str) -> int:
    sys.path.insert(0, str(SCRIPT.parent))
    import dp_paper_view as dpv

    t0 = time.time()
    data = dpv.build(paper, **FLAGS)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = golden_path(paper)
    out.write_text(json.dumps({k: v for k, v in data.items() if k != "_counts"}), encoding="utf-8")
    counts = data.get("_counts", {})
    result = {
        "paper": paper,
        "state": "done",
        "secs": round(time.time() - t0, 1),
        "marks": len(data.get("marks", [])),
        "refs": counts.get("ref", 0),
        "labels": counts.get("label", 0),
        "cites": counts.get("cite", 0),
        "out": str(out),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def run_one(paper: str, timeout: int) -> dict:
    if golden_path(paper).exists():
        return {"paper": paper, "state": "skipped", "reason": "already in golden"}
    cmd = [sys.executable, str(SCRIPT), "--worker-paper", paper]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "paper": paper,
            "state": "failed",
            "secs": round(time.time() - t0, 1),
            "reason": f"timeout >{timeout}s",
            "stdout_tail": (exc.stdout or "")[-1000:],
            "stderr_tail": (exc.stderr or "")[-1000:],
        }
    stdout = proc.stdout.strip()
    if proc.returncode == 0:
        for line in reversed(stdout.splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("state") == "done":
                return row
    return {
        "paper": paper,
        "state": "failed",
        "secs": round(time.time() - t0, 1),
        "reason": f"exit {proc.returncode}",
        "stdout_tail": stdout[-1000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def format_result(idx: int, total: int, row: dict) -> str:
    state = row["state"]
    paper = row["paper"]
    secs = row.get("secs", 0)
    if state == "done":
        detail = (
            f"refs={row.get('refs', 0)} labels={row.get('labels', 0)} "
            f"cites={row.get('cites', 0)} marks={row.get('marks', 0)}"
        )
    elif state == "skipped":
        detail = row.get("reason", "")
    else:
        detail = f"FAILED {row.get('reason', '')}"
    return f"[{idx}/{total}] {paper:14} {state:7} {secs:6}s  {detail}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30, help="max new papers to process")
    ap.add_argument("--timeout", type=int, default=240, help="per-paper subprocess timeout in seconds")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds to sleep between papers")
    ap.add_argument("--paper", action="append", default=[], help="specific paper id(s); already-existing outputs are skipped unless --force")
    ap.add_argument("--force", action="store_true", help="rebuild --paper targets even if output exists")
    ap.add_argument("--log", type=Path, default=None, help="JSONL run log path")
    ap.add_argument("--worker-paper", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.worker_paper:
        return worker_main(args.worker_paper)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    if args.paper:
        todo = args.paper
        n_total = len(todo)
        n_have = sum(1 for paper in todo if golden_path(paper).exists())
        n_missing = n_total - n_have
    else:
        todo, n_total, n_have, n_missing = candidates(args.limit)
    if args.force and args.paper:
        for paper in args.paper:
            golden_path(paper).unlink(missing_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (LOG_DIR / f"dp-batch-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")

    print(
        f"eprints: {n_total} total · {n_have} already in golden · "
        f"{n_missing} missing · processing {len(todo)} sequentially "
        f"(timeout={args.timeout}s, sleep={args.sleep}s)",
        flush=True,
    )
    print(f"log: {log_path}", flush=True)
    if not todo:
        print("nothing to do.")
        return 0

    counts: Counter[str] = Counter()
    failures: list[dict] = []
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        for idx, paper in enumerate(todo, 1):
            if args.force and args.paper:
                golden_path(paper).unlink(missing_ok=True)
            row = run_one(paper, args.timeout)
            counts[row["state"]] += 1
            if row["state"] == "failed":
                failures.append(row)
            row = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            print(format_result(idx, len(todo), row), flush=True)
            if idx < len(todo) and args.sleep > 0:
                time.sleep(args.sleep)

    elapsed = time.time() - started
    print(
        f"BATCH DONE in {elapsed/60:.1f}min — "
        f"done={counts['done']} skipped={counts['skipped']} failed={counts['failed']} "
        f"attempted={len(todo)}",
        flush=True,
    )
    if failures:
        print("FAILURES:")
        for row in failures:
            print(f"  {row['paper']:14} {row.get('reason', '')}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
