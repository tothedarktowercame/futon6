#!/usr/bin/env python3
"""Render dp-demo-quality pages for the gh200 top-cited math.CT papers with the
CURRENT pipeline — IATC free-body inferences (register deductive/body),
expository scopes (incl. the scaffold-less fallback), term-prior concept
normalization, and all render fixes. In-memory re-mine via dp_paper_view.build:
it reads the EPRINT and NEVER writes golden/, so it is safe to run alongside any
mining and cannot race the live store.

Sharded (--shard k/n) for parallelism, resumable (skips existing output unless
--force), and per-paper timeout-protected (one slow top-cited paper can't hang
the shard).

    render_gh200.py [--shard k/n] [--timeout 600] [--list FILE] [--out DIR] [--force]
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_anatomy_html as R
import dp_paper_view as dpv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST = ROOT / "data" / "warp" / "gh200.txt"
DEFAULT_OUT = ROOT / "data" / "showcases" / "ct-anatomy" / "gh200"
FLAGS = dict(with_ca=True, with_binders=True, with_scopes=True, with_xref=True)


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="k/n disjoint partition")
    ap.add_argument("--timeout", type=int, default=600, help="per-paper seconds")
    ap.add_argument("--list", type=Path, default=DEFAULT_LIST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in a.list.read_text().splitlines() if l.strip()]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        ids = [p for i, p in enumerate(ids) if i % n == k]
    signal.signal(signal.SIGALRM, _alarm)
    done = fail = skip = 0
    for pid in ids:
        out = a.out / f"{pid}.html"
        if out.exists() and not a.force:
            skip += 1
            continue
        signal.alarm(a.timeout)
        try:
            d = dpv.build(pid, **FLAGS)
            doc, cov = R.build_html(pid, d)
            out.write_text(doc, encoding="utf8")
            c = cov["coverage"]
            expo = sum(1 for m in d["marks"] if m.get("kind") == "exposition")
            inf = sum(1 for m in d["marks"] if m.get("kind") == "inference")
            print(f"OK {pid} expo={expo} inf={inf} "
                  f"grounded={c.get('symbol_grounded')} tagged={c.get('symbol_tagged')} "
                  f"wf={c.get('wellformed_errors')}", flush=True)
            done += 1
        except _Timeout:
            print(f"TIMEOUT {pid} (>{a.timeout}s)", flush=True)
            fail += 1
        except Exception as exc:  # one bad paper never sinks the shard
            print(f"FAIL {pid}: {type(exc).__name__}: {exc}", flush=True)
            fail += 1
        finally:
            signal.alarm(0)
    print(f"shard {a.shard or 'all'} done: ok={done} fail={fail} skip={skip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
