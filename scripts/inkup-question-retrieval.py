#!/usr/bin/env python
"""Rank candidate reader questions from local Math.SE / MathOverflow stores.

Thin CLI over futon6.inkup_question_retrieval. Store locations come from repeated
``--store SITE=PATH`` arguments or from ``INKUP_QA_STORES`` (``SITE=PATH`` entries
separated by the platform path separator, ``:`` on POSIX). There is no default path.

Example:
    INKUP_QA_STORES="math=/data/se-math:mathoverflow=/data/se-mo" \
        scripts/inkup-question-retrieval.py --query query.json --output-dir runs/q1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for candidate in (_REPO / "src", _REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from futon6 import inkup_question_retrieval as retrieval  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", required=True, type=Path,
                        help="query JSON: a list of terms, each with a source note")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="directory for question-candidates.jsonl and manifest.json")
    parser.add_argument("--store", action="append", default=[], metavar="SITE=PATH",
                        help="store directory holding entities.json and embeddings.npy; repeatable")
    parser.add_argument("--limit", type=int, default=retrieval.TOP_PER_METHOD,
                        help="candidates per method per site (default %(default)s)")
    parser.add_argument("--sample-count", type=int, default=retrieval.MIN_ALIGNMENT_SAMPLE,
                        help="rows to re-embed for the row-alignment check (at least %(default)s)")
    parser.add_argument("--model", default=retrieval.MODEL_NAME, help="embedding model")
    parser.add_argument("--device", default=None, help="torch device for the embedder")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Everything that can be refused is refused before the model is loaded.
        stores = retrieval.parse_stores(args.store)
        retrieval.load_query(args.query)
    except (ValueError, OSError) as error:
        print(f"inkup-question-retrieval: {error}", file=sys.stderr)
        return 2
    embedder = retrieval.BgeEmbedder(args.model, device=args.device)
    try:
        outdir = retrieval.run_retrieval(args.query, stores, args.output_dir, embedder,
                                         limit=args.limit, sample_count=args.sample_count,
                                         repo=_REPO)
    except (ValueError, OSError) as error:
        print(f"inkup-question-retrieval: {error}", file=sys.stderr)
        return 2
    print(outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
