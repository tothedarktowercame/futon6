#!/usr/bin/env python3
"""Prose-term base-rate prior (E-prior-over-terms): document-frequency of
candidate math terms across a corpus. The concept layer uses it to resolve
OVERFED phrases (trim to the high-df core: "interesting abelian category" ->
"abelian category"), HUNGRY ones (extend to the high-df maximal form: "category
of modules" -> "... over a ring"), and HAPAX junk (df 1 -> drop). It is the
prose-term analogue of build_recognizer_registry.py (which does this for macros)
and the "learn-and-promote" signal — document-frequency IS the promotion
criterion, so no curated list is needed.

MSC-class-repeatable BY DESIGN: re-point --golden-dir and --out per class
(the superpod blast re-points per MSC). Nothing CT-specific is hardcoded.

    build_term_prior.py [--golden-dir DIR] [--min-papers K] [--max-papers N]
                        [--msc NAME] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
DEFAULT_OUT = ROOT / "data" / "term-prior-ct.json"

_WORD = re.compile(r"[a-z][a-z-]*")
# boundary stopwords: an n-gram may not START or END on one (so "of modules"
# and "the category" aren't counted, but "category of modules" is).
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "with", "is",
    "are", "be", "been", "being", "that", "this", "we", "it", "its", "by", "as",
    "from", "at", "if", "then", "which", "such", "any", "all", "each", "every",
    "some", "no", "not", "there", "where", "these", "those", "one", "two",
    "both", "also", "only", "so", "thus", "hence", "let", "given", "when",
}
MAX_N = 4


def ngrams(words):
    """Content-bounded 1..MAX_N-grams (first and last word non-stop)."""
    for i in range(len(words)):
        if words[i] in _STOP:
            continue
        for n in range(1, MAX_N + 1):
            if i + n > len(words):
                break
            seg = words[i:i + n]
            if seg[-1] in _STOP:
                continue
            yield " ".join(seg)


def document_frequencies(texts):
    """Return document frequencies for candidate prose terms in `texts`."""
    df = Counter()
    for text in texts:
        for g in set(ngrams(_WORD.findall(text.lower()))):
            df[g] += 1
    return df


def build_index(df, *, min_papers: int):
    return {g: c for g, c in df.items() if c >= min_papers}


def resolve_phrase(phrase: str, df, *, min_papers: int = 2) -> dict:
    """Resolve one prose-term candidate against the df prior.

    Behaviours:
      - HAPAX: phrase df is below threshold and no sub/superphrase rescues it.
      - OVERFED: phrase is too specific, trim to the highest-df core subphrase.
      - HUNGRY: phrase is a recurring core, extend to the highest-df superphrase.
      - KEPT: phrase already resolves to itself.
    """
    words = _WORD.findall(phrase.lower())
    if not words:
        return {"input": phrase, "resolution": None, "action": "HAPAX", "df": 0}
    grams = [" ".join(words[i:j]) for i in range(len(words))
             for j in range(i + 1, min(len(words), i + MAX_N) + 1)]
    phrase_key = " ".join(words)
    eligible = {g: df.get(g, 0) for g in grams if df.get(g, 0) >= min_papers}
    if not eligible:
        return {"input": phrase, "resolution": None, "action": "HAPAX",
                "df": df.get(phrase_key, 0)}

    phrase_df = df.get(phrase_key, 0)
    if phrase_df < min_papers:
        best = sorted(eligible.items(),
                      key=lambda kv: (-len(kv[0].split()), -kv[1], kv[0]))[0]
        return {"input": phrase, "resolution": best[0], "action": "OVERFED",
                "df": phrase_df, "resolved_df": best[1]}

    superphrases = {
        g: c for g, c in df.items()
        if c >= min_papers and g != phrase_key and (
            g.startswith(phrase_key + " ") or g.endswith(" " + phrase_key)
            or f" {phrase_key} " in f" {g} "
        )
    }
    if superphrases:
        best_super = sorted(superphrases.items(),
                            key=lambda kv: (-len(kv[0].split()), -kv[1], kv[0]))[0]
        if len(best_super[0].split()) > len(words):
            return {"input": phrase, "resolution": best_super[0], "action": "HUNGRY",
                    "df": phrase_df, "resolved_df": best_super[1]}
    return {"input": phrase, "resolution": phrase_key, "action": "KEPT", "df": phrase_df}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--min-papers", type=int, default=3)
    ap.add_argument("--max-papers", type=int, default=4000)
    ap.add_argument("--msc", default="ct")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    files = sorted(a.golden_dir.glob("fable-*-dp-emacs.json"))[:a.max_papers]
    df, n = Counter(), 0
    for f in files:
        try:
            t = json.loads(f.read_text()).get("text", "")
        except Exception:
            continue
        if not t:
            continue
        n += 1
        df.update(document_frequencies([t]))
        if n % 500 == 0:
            print(f"  {n} papers...", file=sys.stderr)
    index = build_index(df, min_papers=a.min_papers)
    a.out.write_text(json.dumps(
        {"_meta": {"msc": a.msc, "papers": n, "min_papers": a.min_papers,
                   "terms": len(index)}, "df": index}))
    print(f"{n} papers; {len(index)} terms (df>={a.min_papers}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
