#!/usr/bin/env python3
"""Build a run's concept lexicon from its own papers with a classical scan.

The staged concept vocabulary is a subject's (math.CT unless FUTON6_SUBJECT
says otherwise). A collection from another classification, or across several,
needs its own; a hand-made concept index can be named with
FUTON6_WARP_LEXICON_IDS, and this builds one when there is none.

Candidate terms come from the background index (data/background-corpus-index.json:
PlanetMath, MathWorld, Wikipedia, the Encyclopedia of Mathematics, nLab, DLMF,
MathHub -- every classification). A term is kept when enough of the run's
papers use it: document frequency at least --min-df (default: 2, or 1% of the
papers if more) and below --max-share of them (near-universal words carry no
signal). Single words must be known to two or more sources; discourse words
("proof", "lemma", "case") are never terms.

The lexicon is written as one paper of definition environments,
<out>/lexicon-built.tex, so WARP reads it exactly as it reads a hand-made
index, plus <out>/lexicon.json: the terms with their document frequency,
sources and MSC codes, and the corpus's MSC profile (which classifications the
collection draws on, weighted by use).

Usage: build_lexicon.py --ids <ids.txt> --out <dir> [--min-df N] [--max-share F]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anatomy_v0_sweep as sweep  # noqa: E402
import futon6_config as config  # noqa: E402
import eprint_corpus  # noqa: E402

LEXICON_ID = "lexicon-built"
DISCOURSE = {
    "proof", "proofs", "theorem", "lemma", "corollary", "proposition", "definition",
    "example", "remark", "claim", "case", "cases", "step", "steps", "problem", "solution",
    "statement", "result", "results", "note", "part", "section", "argument", "fact",
    "conclusion", "hypothesis", "assumption", "condition", "conditions", "property",
    "method", "idea", "way", "order", "form", "means", "point of view", "end",
}
# A term neither begins nor ends with a function word: "such that", "element
# of" and "is equivalent to" are encyclopedia entries, but connectives, not
# concepts (the classical (Adj|Noun)* Noun shape, without a tagger).
FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from", "into",
    "onto", "that", "this", "these", "those", "is", "are", "be", "been", "as", "and", "or",
    "not", "no", "all", "some", "every", "each", "any", "most", "least", "exactly", "one",
    "such", "than", "then", "if", "iff", "it", "its", "we", "you", "your", "our", "must",
    "need", "can", "may", "will", "has", "have", "there", "which", "what", "where", "when",
    "over", "under", "between", "about", "via", "per", "within", "without", "along",
}
# Single words with encyclopedia entries that are ordinary English in prose.
COMMON_WORDS = {
    "right", "left", "choose", "second", "first", "third", "three", "two", "four", "five",
    "term", "value", "equal", "number", "fixed", "point", "unique", "positive", "negative",
    "given", "set", "show", "take", "let", "use", "give", "get", "make", "find", "follow",
    "follows", "large", "small", "true", "false", "general", "simple", "standard", "unit",
    "second", "end", "line", "level", "rank", "degree", "type", "kind", "class", "element",
}
MAX_WORDS = 4


def canon(text: str) -> list[str]:
    """Words of the prose, mathematics and commands removed."""
    text = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"(?<!\\)%[^\n]*", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    return re.findall(r"[a-z][a-z'-]*[a-z]|[a-z]", text.lower())


def vocabulary(index_path: Path) -> dict[str, dict]:
    """Surface form -> {term, sources, msc, url} from the background index."""
    d = json.loads(index_path.read_text())
    out = {}
    for surface, entries in d.get("terms", {}).items():
        words = surface.split()
        if not 1 <= len(words) <= MAX_WORDS or len(surface) < 4 or surface in DISCOURSE:
            continue
        if not re.fullmatch(r"[a-z][a-z' -]*[a-z]", surface):
            continue
        e = entries[0]
        # ct-prior entries are n-grams mined from the math.CT corpus ("complete
        # proof", "you must"), not encyclopedia entries; a built lexicon draws on
        # the encyclopedias alone.
        if e.get("resolution-kind") == "ct-prior":
            continue
        term = (e.get("term") or surface).lower()
        # Judge the surface form and its canonical term alike: "points" is "point".
        for form in {surface, term}:
            fw = form.split()
            if not fw or fw[0] in FUNCTION_WORDS or fw[-1] in FUNCTION_WORDS or \
                    (len(fw) == 1 and form in COMMON_WORDS):
                break
        else:
            form = None
        if form is not None:
            continue
        sources = e.get("domains") or ([e["target"].split("-")[0]] if e.get("target") else [])
        if len(words) == 1 and len(sources) < 2:
            continue
        out[surface] = {"term": term, "sources": sources,
                        "msc": e.get("msc") or [], "url": (e.get("urls") or [None])[0]}
    return out


def paper_text(pid: str) -> str | None:
    archive = eprint_corpus.find_corpus_eprint(pid, config.eprint_roots())
    if archive is None:
        return None
    try:
        files, _meta = sweep.read_eprint_files(archive)
    except Exception:
        return None
    return " ".join(f["text"] for f in files if f["file"].endswith(".tex"))


def scan(ids: list[str], vocab: dict[str, dict]) -> tuple[Counter, int]:
    """Document frequency of each vocabulary term over the papers read."""
    df: Counter = Counter()
    read = 0
    for pid in ids:
        text = paper_text(pid)
        if text is None:
            continue
        read += 1
        words = canon(text)
        seen = set()
        for n in range(1, MAX_WORDS + 1):
            for i in range(len(words) - n + 1):
                g = " ".join(words[i:i + n])
                if g in vocab:
                    seen.add(vocab[g]["term"])
        df.update(seen)
    return df, read


def select(df: Counter, read: int, vocab: dict[str, dict], min_df: int | None,
           max_share: float) -> list[dict]:
    floor = min_df if min_df is not None else max(2, math.ceil(0.01 * read))
    by_term = {v["term"]: v for v in vocab.values()}
    keep = []
    for term, n in df.most_common():
        if n < floor or n > max_share * read or term in DISCOURSE:
            continue
        v = by_term.get(term, {})
        keep.append({"term": term, "df": n, "sources": v.get("sources", []),
                     "msc": v.get("msc", []), "url": v.get("url")})
    return keep


def msc_profile(terms: list[dict]) -> list[tuple[str, int]]:
    """Top-level MSC classes the lexicon draws on, weighted by document frequency."""
    prof: Counter = Counter()
    for t in terms:
        tops = {m[:2] for m in t["msc"] if re.match(r"\d\d", m)}
        for top in tops:
            prof[top] += t["df"]
    return prof.most_common()


def lexicon_tex(terms: list[dict]) -> str:
    def esc(s: str) -> str:
        return re.sub(r"([#$%&_{}~^\\])", r"\\\1", s)
    lines = [r"\documentclass{article}", r"\newtheorem{definition}{Definition}",
             r"\begin{document}", r"\section{Lexicon built from the run's papers}"]
    for t in terms:
        gloss = "; ".join(filter(None, [", ".join(t["sources"]),
                                        "MSC " + " ".join(t["msc"][:4]) if t["msc"] else ""]))
        lines.append(f"\\begin{{definition}}\\emph{{{esc(t['term'])}}}: {esc(gloss)}"
                     f" (used in {t['df']} papers)\\end{{definition}}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def build(ids: list[str], out: Path, min_df: int | None = None, max_share: float = 0.9,
          index: Path | None = None) -> dict:
    vocab = vocabulary(index or config.authority())
    df, read = scan(ids, vocab)
    terms = select(df, read, vocab, min_df, max_share)
    report = {"schema": "built-lexicon/v1", "papers-read": read, "papers-listed": len(ids),
              "candidates": len(vocab), "terms": terms, "msc-profile": msc_profile(terms)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{LEXICON_ID}.tex").write_text(lexicon_tex(terms))
    (out / "lexicon.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-df", type=int, default=None)
    ap.add_argument("--max-share", type=float, default=0.9)
    args = ap.parse_args(argv)
    ids = [l.strip() for l in Path(args.ids).read_text().splitlines()
           if l.strip() and not l.startswith("#")]
    report = build(ids, Path(args.out), args.min_df, args.max_share)
    top = ", ".join(f"{m}:{n}" for m, n in report["msc-profile"][:6])
    print(f"[build-lexicon] {report['papers-read']}/{len(ids)} papers read; "
          f"{len(report['terms'])} terms kept of {report['candidates']} candidates; "
          f"MSC profile {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
