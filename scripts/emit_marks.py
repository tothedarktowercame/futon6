#!/usr/bin/env python3
"""S1 marks emitter (closes the cold-front fetch->candidates gap).

Runs the deterministic detector (dp_paper_view.build, all flags) per paper and
writes its output as golden/fable-<id>-dp-emacs.json — the exact {paper,text,marks}
shape mark3_extract_candidates reads. Eprints must already be fetched into
DEFAULT_EPRINTS (fetch-arxiv-eprints.py). Pipeline:
  fetch-arxiv-eprints -> emit_marks -> mark3_extract_candidates -> S3 (iatc loop)

Usage:
  futon6/.venv/bin/python scripts/emit_marks.py --list holes/math-ct-200.ids.txt
  futon6/.venv/bin/python scripts/emit_marks.py --papers 0705.4406
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import futon6_config as config

import argparse
from functools import partial
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dp_paper_view as dpv  # noqa: E402
import paper_pool  # noqa: E402

GOLDEN = str(config.marks())


def mark_paper(pid, out, coverage):
    """One paper's marks, written to <out>; runs in a worker process, so failures are returned."""
    try:
        d = dpv.build(pid, with_ca=True, with_binders=True, with_scopes=True)
        with open(os.path.join(out, f"fable-{pid}-dp-emacs.json"), "w") as fh:
            json.dump({"paper": d["paper"], "text": d["text"], "marks": d["marks"]}, fh)
        kinds = {}
        for m in d["marks"]:
            kinds[m.get("kind")] = kinds.get(m.get("kind"), 0) + 1
        result = {"marks": len(d["marks"]), "proof-move": kinds.get("proof-move", 0)}
        if coverage:
            try:
                import metric_harness as mh
                result["coverage"] = mh._interval_coverage(d["marks"], len(d.get("text", "")) or 1)
            except Exception as ee:
                result["coverage-error"] = str(ee)
        return result
    # A helper that ends the process (dp_paper_view.build did so for a paper with no e-print)
    # fails this paper only; uncaught, SystemExit escaped the pool and ended S1 for every paper.
    except (Exception, SystemExit) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list")
    ap.add_argument("--papers", nargs="*")
    ap.add_argument("--out", default=GOLDEN)
    ap.add_argument("--run-dir", help="if set, emit S1 MetricRecords here (INSTANTIATE-GPU)")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    args = ap.parse_args()
    mh = None
    if args.run_dir:
        import metric_harness as mh  # emit_record + _interval_coverage
    ids = args.papers or [l.strip() for l in open(args.list) if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    ok = fail = 0
    # Papers are independent: mark them on every CPU the job may use, and report
    # and record them here in list order, as the serial loop did.
    print(f"marking {len(ids)} papers on up to {config.workers()} worker processes")
    work = partial(mark_paper, out=args.out, coverage=bool(mh))
    for pid, result in paper_pool.map_papers(work, ids):
        if "error" in result:
            print(f"  {pid}: FAIL {result['error']}")
            fail += 1
            continue
        print(f"  {pid}: {result['marks']} marks (proof-move={result['proof-move']})")
        if mh:  # S1 inline metric emit (non-fatal — never abort the marks artifact)
            try:
                if "coverage-error" in result:
                    raise RuntimeError(result["coverage-error"])
                mh.emit_record(args.run_dir, run_id=args.run_id, corpus_id=args.corpus_id,
                               paper_id=pid, stage="S1", metric="any-markup-coverage",
                               axis="completeness", value=round(result["coverage"], 4), computable=True)
            except Exception as ee:
                print(f"    (metric emit skipped: {ee})")
        ok += 1
    print(f"emitted {ok} / failed {fail}")
    sys.exit(1 if fail and not ok else 0)


if __name__ == "__main__":
    main()
