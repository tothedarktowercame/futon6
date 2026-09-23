#!/usr/bin/env python3
"""Corpus check: normalize-math-prose must not manufacture the byte pair "\\$".

Acceptance signal from holes/normalize-math-prose-stranded-backslash-defects.md:
no source file in the APM Lean corpus contains "\\$", so every occurrence in
output is manufactured by the script. The count must be 0.

Each file runs through process_file -- never process_line, which bypasses the
guards that protect display math -- in a subprocess with a memory cap and a
timeout, because an unfixed scanner can take the whole cgroup down rather than
just this process.

    python3 scripts/survey-normalize-generated-dollars.py [corpus-glob]

Exit status is 0 only when nothing hung, nothing blew the cap, and no generated
"\\$" remains.
"""
from __future__ import annotations

import glob
import importlib.util
import multiprocessing as mp
import resource
import shutil
import sys
import tempfile
from pathlib import Path

import futon6_config as config

SCRIPT = Path(__file__).resolve().parent / "normalize-math-prose.py"
DEFAULT_GLOB = str(config.sibling("apm-lean") / "problems/*/informal-solution.md")
MEM_CAP = 2 * 1024**3
TIMEOUT_S = 15


def _work(q, src: str) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP, MEM_CAP))
    spec = importlib.util.spec_from_file_location("nmp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp = tempfile.mkdtemp()
    try:
        staged = Path(tmp) / "f.md"
        shutil.copy(src, staged)
        mod.process_file(staged, write=True)
        out = staged.read_text(encoding="utf-8", errors="replace")
        q.put(("ok", [i for i, l in enumerate(out.split("\n"), 1) if "\\$" in l]))
    except MemoryError:
        q.put(("mem", []))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_one(src: str):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_work, args=(q, src))
    p.start()
    try:
        result = q.get(timeout=TIMEOUT_S)
    except Exception:
        result = ("stall", [])
    finally:
        if p.is_alive():
            p.kill()
        p.join(5)
    return result


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOB
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files matched {pattern}", file=sys.stderr)
        return 2

    stalled, capped, damaged, lines = [], [], [], 0
    for f in files:
        status, bad = run_one(f)
        if status == "stall":
            stalled.append(f)
        elif status == "mem":
            capped.append(f)
        elif bad:
            damaged.append((f, bad))
            lines += len(bad)

    print(f"files scanned            : {len(files)}")
    print(f"hung (>{TIMEOUT_S}s)             : {len(stalled)}")
    print(f"blew the {MEM_CAP // 1024**3} GiB cap        : {len(capped)}")
    print(f"files with generated \\$  : {len(damaged)}   lines: {lines}")
    for f, bad in damaged:
        print(f"    {Path(f).parent.name}: lines {', '.join(map(str, bad))}")

    ok = not (stalled or capped or damaged)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
