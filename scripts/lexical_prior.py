#!/usr/bin/env python3
"""Lexical pattern priors over proof PROSE, and the coverage survey built on them.

Two jobs, one scan:

  --survey   which patterns are lexically detectable, and where
  --prior    per-paper pattern priors, to seed extraction rather than follow it

**Why prose.** The existing recogniser runs on IATC-extracted step texts, which
keep the mathematics and discard the argumentative connective tissue. Measured on
the same corpus, `contradict*` appears in 15 of 1,523 source passages and in 0 of
818 extracted steps. So a pattern whose trigger vocabulary is argumentative
(`argue-by-contradiction`, `split-into-cases`) scores zero for a reason that is
about the pipeline, not the mathematics. Any coverage survey has to read the
prose or it will find the same artifact in every arXiv area at once and mistake
it for a fact about the subject.

**Why inverse-pattern-frequency.** The hotword column in patterns-index.tsv has
path tokens leaked into it: `math` and `informal` appear in 38 of the 39
math-informal rows. A raw overlap count therefore fires every pattern on every
mathematical passage. A hotword shared by many patterns cannot discriminate
between them, so each is weighted by 1/(patterns containing it) — the pattern-side
analogue of IDF — and anything above `--max-pattern-df` is dropped outright.

**What a hit is and is not.** A hit says the vocabulary is present, not that the
move was made. Spot-checking the `contradict*` hits on math.CT, 12 of 13 were
genuinely "assume the negation, derive a contradiction", which is high but not
free — and that idiom is unusually specific. Treat the output as a prior to be
checked, never as a label. See `futon3/README-pattern-mining.md` §5.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cas_select as cas  # noqa: E402

WORD = re.compile(r"[a-z][a-z0-9-]{2,}")
# Pattern NAMES are tokenised into hotwords, so English function words arrive as
# high-weight terms: `and` enters from `induction-and-well-ordering`, is claimed
# by no other pattern, scores weight 1.0, and fires on 89% of passages. `right`
# from `find-the-right-abstraction` then matches every "right adjoint" in a
# category-theory corpus. Neither says anything about the move being made.
STOPWORDS = {
    "and", "the", "for", "with", "from", "into", "that", "this", "then", "than",
    "over", "under", "about", "which", "when", "where", "your", "you", "are",
    "not", "but", "all", "any", "its", "it's", "one", "two", "via", "per",
    "right", "left", "well", "case", "cases", "use", "used", "using", "make",
    "made", "get", "gets", "take", "takes", "way", "ways", "work", "works",
    "new", "old", "own", "same", "such", "some", "more", "most", "much",
    "many", "just", "only", "also", "both", "each", "other", "another",
}
# Latex control sequences and math-mode noise: stripped before tokenising, so
# \contradiction-in-a-macro does not read as prose and $\cd_i$ contributes nothing.
TEX = re.compile(r"\\[a-zA-Z]+\s*|\$[^$]{0,400}\$|[{}\\^_~]")


def tokens(text: str) -> set[str]:
    return {w for w in WORD.findall(TEX.sub(" ", text or "").lower())
            if w not in STOPWORDS}


def load_hotwords(max_pattern_df: int, corpus_df: dict[str, int] | None = None,
                  n_docs: int = 0, max_corpus_frac: float = 0.15):
    """{pattern: {hotword: weight}} plus the tokens dropped as non-discriminative.

    Two independent filters, because a hotword can fail to discriminate in two
    different ways:

      pattern-side  shared by many patterns -> says nothing about WHICH pattern
                    (`math`, `informal`: leaked path tokens, in 38 of 39 rows)
      corpus-side   present in most passages -> says nothing about WHICH passage
                    (`proof` in 51%, `category` in 43% of a math.CT corpus)

    The first alone leaves a term unique to one pattern at full weight however
    common it is in the corpus, which is how `and` came to fire on 89% of
    passages. Corpus frequency is measured on the corpus being scanned, so the
    same pattern set self-calibrates to a different area.
    """
    pats = cas.load_patterns()
    raw = {name: {w for w in p.hotwords if w not in STOPWORDS}
           for name, p in pats.items()}
    pdf = collections.Counter(w for ws in raw.values() for w in ws)
    dropped_pattern = sorted(w for w, n in pdf.items() if n > max_pattern_df)
    dropped_corpus = []
    if corpus_df and n_docs:
        ceiling = max_corpus_frac * n_docs
        dropped_corpus = sorted(w for w in pdf if corpus_df.get(w, 0) > ceiling)
    dead = set(dropped_pattern) | set(dropped_corpus)
    weighted = {name: {w: 1.0 / pdf[w] for w in ws if w not in dead}
                for name, ws in raw.items()}
    return weighted, sorted(dead)


def passages(root: str) -> list[tuple[str, str]]:
    """(id, prose) for every candidate passage that carries real text."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, "*.candidate.json"))):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        text = ""
        for v in d.values():
            if isinstance(v, str) and len(v) > 200:
                text = v
                break
        if text:
            out.append((os.path.basename(f).replace(".candidate.json", ""), text))
    return out


def scan(items, weighted, min_score):
    """[(id, [(pattern, score, hits)])] — every pattern scoring above threshold."""
    results = []
    for pid, text in items:
        toks = tokens(text)
        fired = []
        for name, hots in weighted.items():
            hits = toks & set(hots)
            if not hits:
                continue
            score = sum(hots[h] for h in hits)
            if score >= min_score:
                fired.append((name, round(score, 3), sorted(hits)))
        fired.sort(key=lambda t: -t[1])
        results.append((pid, fired))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/iatc-candidates-run")
    ap.add_argument("--max-pattern-df", type=int, default=6,
                    help="drop hotwords shared by more than this many patterns")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--max-corpus-frac", type=float, default=0.15,
                    help="drop hotwords appearing in more than this fraction of passages")
    ap.add_argument("--survey", action="store_true", help="per-pattern detection rates")
    ap.add_argument("--prior", action="store_true", help="per-paper priors as JSON")
    ap.add_argument("--top", type=int, default=3, help="priors kept per paper")
    ap.add_argument("--out")
    a = ap.parse_args()

    items = passages(a.candidates)
    if not items:
        print(f"no candidate passages under {a.candidates}", file=sys.stderr)
        return 1
    cdf: collections.Counter = collections.Counter()
    for _, text in items:
        for tok in tokens(text):
            cdf[tok] += 1
    weighted, dropped = load_hotwords(a.max_pattern_df, cdf, len(items),
                                      a.max_corpus_frac)
    results = scan(items, weighted, a.min_score)

    if a.survey:
        per = collections.Counter()
        papers = collections.defaultdict(set)
        for pid, fired in results:
            paper = pid.split("__")[0]
            for name, _, _ in fired:
                per[name] += 1
                papers[name].add(paper)
        n = len(items)
        npapers = len({p.split("__")[0] for p, _ in results})
        print(f"lexical survey over PROSE — {n} passages, {npapers} papers")
        print(f"  dropped {len(dropped)} non-discriminative hotwords "
              f"(>{a.max_pattern_df} patterns, or >{100*a.max_corpus_frac:.0f}% of "
              f"passages): {', '.join(dropped[:10])}"
              f"{' …' if len(dropped) > 10 else ''}\n")
        print(f"  {'pattern':38} {'passages':>8} {'papers':>7}")
        for name in sorted(weighted, key=lambda k: -per[k]):
            print(f"  {name:38} {per[name]:8d} {len(papers[name]):7d}")
        silent = [k for k in weighted if per[k] == 0]
        print(f"\n  never detected in prose: {len(silent)}/{len(weighted)}")
        for s in sorted(silent):
            print(f"     {s}  (hotwords kept: {len(weighted[s])})")
        nohit = [p for p, f in results if not f]
        print(f"\n  passages with NO pattern above threshold: {len(nohit)} "
              f"({100*len(nohit)/n:.1f}%) — the mining worklist")

    if a.prior:
        by_paper = collections.defaultdict(collections.Counter)
        for pid, fired in results:
            paper = pid.split("__")[0]
            for name, score, _ in fired:
                by_paper[paper][name] += score
        priors = {p: [{"pattern": k, "score": round(v, 3)}
                      for k, v in c.most_common(a.top)]
                  for p, c in by_paper.items()}
        text = json.dumps(priors, indent=2, sort_keys=True)
        if a.out:
            Path(a.out).write_text(text + "\n")
            print(f"  priors for {len(priors)} papers -> {a.out}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
