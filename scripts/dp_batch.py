#!/usr/bin/env python3
"""Batch-run the DP detector (dp_paper_view.build) over many math.CT eprints —
the scale-up arm of the structure-mining fleet (claude-4, 2026-06-13).

Generation is embarrassingly parallel: each paper is independent and writes its
own golden JSON, so we fan out across CPU cores. Crashes and hangs are isolated
per worker (try/except + a per-paper SIGALRM wall) and RECORDED, never silently
dropped — the runbook's no-silent-caps discipline. Skips papers already in the
golden dir so re-runs are incremental.

    dp_batch.py                 # next 200 unprocessed CT papers, full flags
    dp_batch.py --limit 50 --jobs 4
    dp_batch.py --paper 0809.2517   # force one paper (ignores skip)

Only this runner is committed; the generated data/ JSON is gitignored.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_paper_view as dpv

EPRINTS = dpv.EPRINTS
GOLDEN_DIR = dpv.GOLDEN_DIR
FLAGS = dict(with_ca=True, with_binders=True, with_scopes=True, with_xref=True)


class _Timeout(Exception):
    pass


def _build_one(paper: str, timeout: int):
    """Worker: build one paper with the full flags, write its golden JSON.
    Returns a small result dict (picklable). Self-aborts after `timeout` s so a
    pathological paper cannot wedge a worker forever."""
    def _alarm(signum, frame):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    t0 = time.time()
    try:
        data = dpv.build(paper, **FLAGS)
        out = GOLDEN_DIR / f"fable-{paper}-dp-emacs.json"
        out.write_text(json.dumps({k: v for k, v in data.items()
                                   if k != "_counts"}))
        c = data["_counts"]
        return {"paper": paper, "state": "done", "secs": round(time.time() - t0, 1),
                "marks": len(data["marks"]),
                "refs": c.get("ref", 0), "labels": c.get("label", 0),
                "cites": c.get("cite", 0)}
    except _Timeout:
        return {"paper": paper, "state": "failed", "secs": round(time.time() - t0, 1),
                "reason": f"timeout >{timeout}s"}
    except BaseException as e:  # SystemExit (no eprint), parse errors, etc.
        return {"paper": paper, "state": "failed", "secs": round(time.time() - t0, 1),
                "reason": f"{type(e).__name__}: {str(e)[:160]}"}
    finally:
        signal.alarm(0)


def _candidates(limit: int):
    """The next `limit` CT papers (sorted, deterministic) that have a .tar.gz
    eprint but no golden JSON yet. Returns (todo, n_total, n_already)."""
    have = {p.name[len("fable-"):-len("-dp-emacs.json")]
            for p in GOLDEN_DIR.glob("fable-*-dp-emacs.json")}
    todo = []
    n_total = 0
    for e in sorted(EPRINTS.glob("*.tar.gz")):
        n_total += 1
        pid = e.name[:-len(".tar.gz")]
        if pid not in have:
            todo.append(pid)
    return todo[:limit], n_total, len(have)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200,
                    help="max papers to process this run (default 200)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel worker processes (default 8)")
    ap.add_argument("--timeout", type=int, default=240,
                    help="per-paper wall-clock seconds before abort (default 240)")
    ap.add_argument("--paper", help="force a single paper id (ignores skip-set)")
    args = ap.parse_args(argv)

    if args.paper:
        todo, n_total, n_have = [args.paper], 1, 0
    else:
        todo, n_total, n_have = _candidates(args.limit)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"eprints: {n_total} total · {n_have} already in golden · "
          f"processing {len(todo)} (jobs={args.jobs}, timeout={args.timeout}s)",
          flush=True)
    if not todo:
        print("nothing to do.")
        return 0

    done, failed = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_build_one, p, args.timeout): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["state"] == "done":
                done.append(r)
                tag = (f"refs={r['refs']} labels={r['labels']} "
                       f"cites={r['cites']} marks={r['marks']}")
            else:
                failed.append(r)
                tag = f"FAILED {r['reason']}"
            print(f"[{i}/{len(todo)}] {r['paper']:14} {r['state']:6} "
                  f"{r['secs']:5}s  {tag}", flush=True)

    dt = time.time() - t0
    print(f"\n{'='*64}")
    print(f"BATCH DONE in {dt/60:.1f}min — {len(done)} ok, {len(failed)} failed, "
          f"{len(todo)} attempted of {n_total} eprints")
    if failed:
        print("FAILURES (paper · reason):")
        # group identical reasons for a compact, honest tally
        from collections import Counter
        by_reason = Counter(f["reason"].split(":")[0] for f in failed)
        for r in failed:
            print(f"  {r['paper']:14} {r['reason']}")
        print("  reason tally:", dict(by_reason))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
