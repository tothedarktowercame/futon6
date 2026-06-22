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
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dp_paper_view as dpv  # noqa: E402

GOLDEN = os.path.join(ROOT, "data/showcases/ct-anatomy/golden")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list")
    ap.add_argument("--papers", nargs="*")
    ap.add_argument("--out", default=GOLDEN)
    args = ap.parse_args()
    ids = args.papers or [l.strip() for l in open(args.list) if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    ok = fail = 0
    for pid in ids:
        try:
            d = dpv.build(pid, with_ca=True, with_binders=True, with_scopes=True)
            with open(os.path.join(args.out, f"fable-{pid}-dp-emacs.json"), "w") as fh:
                json.dump({"paper": d["paper"], "text": d["text"], "marks": d["marks"]}, fh)
            kinds = {}
            for m in d["marks"]:
                kinds[m.get("kind")] = kinds.get(m.get("kind"), 0) + 1
            pm = kinds.get("proof-move", 0)
            print(f"  {pid}: {len(d['marks'])} marks (proof-move={pm})")
            ok += 1
        except Exception as e:
            print(f"  {pid}: FAIL {type(e).__name__}: {e}")
            fail += 1
    print(f"emitted {ok} / failed {fail}")
    sys.exit(1 if fail and not ok else 0)


if __name__ == "__main__":
    main()
