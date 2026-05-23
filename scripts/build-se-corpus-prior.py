#!/usr/bin/env python3
r"""Build the SE corpus-frequency prior over math.SE + MathOverflow.

For each canon in the MSC topic prior (the universe of "things that
have PM articles"), count how often its natural-language phrase form
appears in math.SE and MathOverflow question + answer bodies.

The signal: canons that essentially never appear in real
mathematical discourse get a small marginal prior, regardless of
topic. `StableMarriageProblem` ≈ 0 in math.SE corpus; `Functor`,
`AbelianGroup`, `Limit` are heavily mentioned.

Algorithm:
  1. Load canon → phrase mapping from the MSC prior file.
  2. Bucket phrases by token count (1, 2, 3, ... words).
  3. Stream entities (ijson for math.SE's 2.4GB array; in-memory
     for MO's 349MB). For each body, lowercase + tokenize, walk
     n-grams up to max-phrase-length, look up in the per-length set.
  4. Increment per-canon counts; one canon = one count per body it
     appears in (not per-occurrence within a body — we want a
     document-frequency prior, not a term-frequency one).

Usage:
    python scripts/build-se-corpus-prior.py \\
        --msc-prior data/topic-prior-msc.json \\
        --mo-entities /home/joe/code/storage/futon6/mo-processed/entities.json \\
        --mse-entities /home/joe/code/storage/futon6/se-data/math-processed/entities.json \\
        --out data/topic-prior-se-corpus.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6.topic_prior import SECorpusPrior, canon_to_phrase


# Tokeniser: word characters and digits, lowercased.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_phrase_index(canons):
    """phrase tuple -> canon name, plus per-length bucket of phrase tuples.

    Includes single-word phrases so legitimate one-word canons like
    "Functor" or "Category" still pick up corpus signal. Single-word
    collisions with common English words are unavoidable for these
    cases and are exactly what the MSC topic prior handles in parallel
    (Functor → MSC 18 strong; "stable" → no MSC anchor)."""
    phrase_to_canon: dict[tuple[str, ...], str] = {}
    by_length: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    skipped = 0
    for canon in canons:
        phrase = canon_to_phrase(canon)
        toks = tuple(tokenize(phrase))
        if not toks:
            skipped += 1
            continue
        # Skip single-token phrases shorter than 3 chars — these are
        # pathological "canons" like "A", "C", "Pi" that match every
        # variable name in math discourse and flood the index. Real
        # short canons (Z2, O2) inherit the same problem; the cost
        # of dropping them is they get the neutral 1.0 prior.
        if len(toks) == 1 and len(toks[0]) < 3:
            skipped += 1
            continue
        phrase_to_canon.setdefault(toks, canon)
        by_length[len(toks)].add(toks)
    return phrase_to_canon, by_length, skipped


def count_phrases_in_text(
    text: str,
    phrase_to_canon: dict[tuple[str, ...], str],
    by_length: dict[int, set[tuple[str, ...]]],
) -> set[str]:
    """Return set of canons appearing at least once in text."""
    toks = tokenize(text)
    if not toks:
        return set()
    found: set[str] = set()
    max_len = max(by_length.keys(), default=0)
    # Walk every n-gram up to max_len
    for n in range(1, max_len + 1):
        bucket = by_length.get(n)
        if not bucket:
            continue
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i:i + n])
            if ng in bucket:
                found.add(phrase_to_canon[ng])
    return found


def stream_mo(path: Path):
    """MO is small enough to load fully."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for entity in data:
        yield entity


def stream_mse(path: Path):
    """MSE is 2.4GB — stream with ijson."""
    import ijson
    with open(path, "rb") as f:
        for entity in ijson.items(f, "item"):
            yield entity


def process_corpus(name, stream_fn, path, phrase_to_canon, by_length,
                   prior: SECorpusPrior, max_entities: int | None = None):
    print(f"[se-prior] processing {name} from {path}")
    t0 = time.time()
    n = 0
    for entity in stream_fn(path):
        body = (entity.get("question-body") or "") + " " + (entity.get("answer-body") or "")
        title = entity.get("title") or ""
        text = title + " " + body
        if not text.strip():
            continue
        n += 1
        prior.n_documents += 1
        found = count_phrases_in_text(text, phrase_to_canon, by_length)
        for canon in found:
            prior.add(canon, 1)
        if n % 10000 == 0:
            dt = time.time() - t0
            print(f"[se-prior]   {name}: {n:>7d} docs in {dt:.1f}s "
                  f"({n/dt:.0f} docs/s); unique canons hit so far: {len(prior.counts)}")
        if max_entities is not None and n >= max_entities:
            print(f"[se-prior]   {name}: stopping at --max-entities={max_entities}")
            break
    print(f"[se-prior] {name}: {n} docs in {time.time()-t0:.1f}s")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--msc-prior", type=Path, required=True,
                   help="topic-prior-msc.json — gives the canon universe")
    p.add_argument("--mo-entities", type=Path, default=None,
                   help="MathOverflow processed entities.json")
    p.add_argument("--mse-entities", type=Path, default=None,
                   help="math.SE processed entities.json (streamed)")
    p.add_argument("--out", type=Path,
                   default=Path("data/topic-prior-se-corpus.json"))
    p.add_argument("--max-entities", type=int, default=None,
                   help="Cap per-corpus document count (for smoke tests)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    msc_data = json.loads(args.msc_prior.read_text(encoding="utf-8"))
    canons = list(msc_data.get("counts", {}).keys())
    print(f"[se-prior] {len(canons)} canons from {args.msc_prior}")

    phrase_to_canon, by_length, skipped = build_phrase_index(canons)
    print(f"[se-prior] {len(phrase_to_canon)} multi-word phrases; "
          f"skipped {skipped} single-word/empty canons")
    print(f"[se-prior] phrase length distribution: "
          f"{sorted({k: len(v) for k, v in by_length.items()}.items())}")

    prior = SECorpusPrior()

    if args.mo_entities and args.mo_entities.exists():
        process_corpus("MO", stream_mo, args.mo_entities,
                       phrase_to_canon, by_length, prior, args.max_entities)
    if args.mse_entities and args.mse_entities.exists():
        process_corpus("MSE", stream_mse, args.mse_entities,
                       phrase_to_canon, by_length, prior, args.max_entities)

    print(f"[se-prior] {len(prior.counts)} canons matched in corpus; "
          f"{prior.grand_total} doc-occurrences total; "
          f"{prior.n_documents} docs scanned")

    print()
    print("[se-prior] top 20 most-mentioned canons:")
    for canon, n in sorted(prior.counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {canon:40s} {n:>6d}")

    print()
    print("[se-prior] sanity prior values:")
    for canon in ["StableMarriageProblem", "AbelianGroup", "Functor",
                  "RingedSpace", "RiemannHypothesis", "Limit",
                  "CategoryTheory", "BooleanAlgebra"]:
        print(f"  {canon:30s} P={prior.prior(canon):.4f} (n={prior.counts.get(canon, 0)})")

    prior.save(args.out)
    print(f"[se-prior] wrote {args.out}")


if __name__ == "__main__":
    main()
