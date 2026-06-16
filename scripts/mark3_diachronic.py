#!/usr/bin/env python3
"""mark3 H12: diachronic / emergence detector for CT terms.

The static term prior is document-frequency only. This script rebuilds the same
term set per dated paper, buckets papers by arXiv month/year, normalizes by the
number of papers in each bucket, and ranks terms whose normalized usage is rising.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
DEFAULT_CONCEPT_DIR = ROOT / "data" / "concept-encyclopedia" / "ct"
DEFAULT_OUT = ROOT / "tmp" / "mark3-diachronic" / "ct-emerging-terms.json"
TEXISH_TOKENS = {
    "align",
    "amsmath",
    "amssymb",
    "amsthm",
    "begin",
    "bibliography",
    "cal",
    "cdot",
    "cite",
    "color",
    "documentclass",
    "end",
    "eqref",
    "graphicx",
    "hyperref",
    "label",
    "lemma",
    "mathbb",
    "mathcal",
    "mathfrak",
    "mathrm",
    "mathrsfs",
    "mathsf",
    "mathtools",
    "newcommand",
    "newtheorem",
    "proof",
    "ref",
    "sep",
    "text",
    "theorem",
    "theoremstyle",
    "tikz",
    "tikzcd",
    "usepackage",
    "usetikzlibrary",
}
PROSE_DRIFT_TOKENS = {
    "actually",
    "center",
    "clearly",
    "could",
    "do",
    "does",
    "done",
    "following",
    "however",
    "indeed",
    "may",
    "needed",
    "observe",
    "rather",
    "red",
    "see",
    "should",
    "since",
    "therefore",
    "via",
    "was",
    "were",
    "would",
}


def _fallback_word_re():
    return re.compile(r"[a-z][a-z-]*")


def _fallback_ngrams(words):
    stop = {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "with",
        "is",
        "are",
        "be",
        "been",
        "being",
        "that",
        "this",
        "we",
        "it",
        "its",
        "by",
        "as",
        "from",
        "at",
        "if",
        "then",
        "which",
        "such",
        "any",
        "all",
        "each",
        "every",
        "some",
        "no",
        "not",
        "there",
        "where",
        "these",
        "those",
        "one",
        "two",
        "both",
        "also",
        "only",
        "so",
        "thus",
        "hence",
        "let",
        "given",
        "when",
    }
    max_n = 4
    for i in range(len(words)):
        if words[i] in stop:
            continue
        for n in range(1, max_n + 1):
            if i + n > len(words):
                break
            seg = words[i : i + n]
            if seg[-1] in stop:
                continue
            yield " ".join(seg)


def load_term_extractor():
    """Load build_term_prior.py's extractor, falling back to an identical copy."""
    path = ROOT / "scripts" / "build_term_prior.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("build_term_prior", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module._WORD, module.ngrams, "scripts/build_term_prior.py"
    return _fallback_word_re(), _fallback_ngrams, "embedded build_term_prior-compatible fallback"


def parse_arxiv_month(arxiv_id: str) -> str:
    """Return YYYY-MM from new-style or old-style arXiv IDs.

    New style: 0705.0452 -> 2007-05.
    Old style: math/9811139 or math__9811139 -> 1998-11.
    """
    raw = arxiv_id
    if raw.startswith("fable-"):
        raw = raw[len("fable-") :]
    raw = raw.replace("-dp-emacs.json", "").replace(".json", "")
    new = re.match(r"^(\d{2})(\d{2})\.\d+", raw)
    if new:
        yy = int(new.group(1))
        month = int(new.group(2))
        if not 1 <= month <= 12:
            raise ValueError(f"invalid arXiv month in {arxiv_id}")
        return f"20{yy:02d}-{month:02d}"
    old = re.match(r"^(?:[a-z-]+(?:/|__|_))?(\d{2})(\d{2})\d+", raw)
    if old:
        yy = int(old.group(1))
        month = int(old.group(2))
        if not 1 <= month <= 12:
            raise ValueError(f"invalid arXiv month in {arxiv_id}")
        year = 1900 + yy if yy >= 91 else 2000 + yy
        return f"{year:04d}-{month:02d}"
    raise ValueError(f"cannot parse arXiv date from {arxiv_id}")


def paper_id_from_path(path: Path) -> str:
    name = path.name
    if name.startswith("fable-") and name.endswith("-dp-emacs.json"):
        return name[len("fable-") : -len("-dp-emacs.json")]
    return path.stem


def bucket_key(month: str, granularity: str) -> str:
    if granularity == "month":
        return month
    if granularity == "year":
        return month[:4]
    raise ValueError(f"unknown granularity: {granularity}")


def extract_terms(text: str, word_re, ngrams) -> set[str]:
    return set(ngrams(word_re.findall(text.lower())))


def load_candidate_terms(concept_dir: Path) -> set[str]:
    """Load known CT concept surfaces from the encyclopedia.

    The temporal signal still comes only from the dated mark files. The
    encyclopedia is used as a candidate vocabulary so the default ranking
    reports mathematical concepts instead of raw TeX/prose drift.
    """
    terms: set[str] = set()
    if not concept_dir.exists():
        return terms
    for path in concept_dir.glob("*.edn"):
        stem = path.stem.replace("-", " ")
        if stem:
            terms.add(stem.lower())
        try:
            text = path.read_text()
        except OSError:
            continue
        for match in re.finditer(r':name\s+"([^"]+)"', text):
            terms.add(match.group(1).lower())
    return terms


def is_math_candidate(term: str, include_texish: bool = False) -> bool:
    if include_texish:
        return True
    parts = term.split()
    return not any(part in TEXISH_TOKENS or part in PROSE_DRIFT_TOKENS for part in parts)


@dataclass(frozen=True)
class TermTrend:
    term: str
    df: int
    emergence_score: float
    first_seen: str
    peak_year: str
    trend: dict


def linear_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def trend_for_series(term: str, counts: dict[str, int], bucket_docs: dict[str, int]) -> TermTrend:
    buckets = sorted(bucket_docs)
    rates = [counts.get(bucket, 0) / bucket_docs[bucket] for bucket in buckets]
    total = int(sum(counts.values()))
    nonzero = [bucket for bucket in buckets if counts.get(bucket, 0) > 0]
    first_seen = nonzero[0] if nonzero else None
    peak = max(buckets, key=lambda b: (counts.get(b, 0) / bucket_docs[b], counts.get(b, 0))) if buckets else None
    n = len(rates)
    split = max(1, n // 3)
    early_mean = float(np.mean(rates[:split])) if rates else 0.0
    recent_mean = float(np.mean(rates[-split:])) if rates else 0.0
    slope = linear_slope(rates)
    recent_count = int(sum(counts.get(b, 0) for b in buckets[-split:]))
    early_count = int(sum(counts.get(b, 0) for b in buckets[:split]))
    ratio = (recent_mean + 1e-6) / (early_mean + 1e-6)
    support = math.log1p(total)
    slope_component = max(0.0, slope) * len(buckets) * 100.0
    ratio_component = max(0.0, math.log(ratio))
    late_support = math.log1p(recent_count)
    score = support * (slope_component + ratio_component) * (1.0 + 0.15 * late_support)
    return TermTrend(
        term=term,
        df=total,
        emergence_score=float(score),
        first_seen=first_seen or "",
        peak_year=peak or "",
        trend={
            "slope": slope,
            "early_mean": early_mean,
            "recent_mean": recent_mean,
            "recent_to_early": ratio,
            "early_count": early_count,
            "recent_count": recent_count,
            "nonzero_buckets": len(nonzero),
        },
    )


def build_diachronic_index(golden_dir: Path, granularity: str, max_papers: int | None = None):
    word_re, ngrams, source = load_term_extractor()
    files = sorted(golden_dir.glob("fable-*-dp-emacs.json"))
    if max_papers and len(files) > max_papers:
        # Validation samples should preserve the time axis. Taking the first N
        # sorted arXiv IDs collapses to the earliest years, so sample evenly.
        idxs = np.linspace(0, len(files) - 1, max_papers, dtype=int)
        files = [files[int(i)] for i in idxs]
    term_counts: dict[str, Counter] = defaultdict(Counter)
    bucket_docs: Counter = Counter()
    papers = 0
    skipped = []
    for path in files:
        paper_id = paper_id_from_path(path)
        try:
            month = parse_arxiv_month(paper_id)
            bucket = bucket_key(month, granularity)
            data = json.loads(path.read_text())
            text = data.get("text", "")
        except Exception as exc:
            skipped.append({"paper": paper_id, "error": str(exc)})
            continue
        if not text:
            continue
        papers += 1
        bucket_docs[bucket] += 1
        for term in extract_terms(text, word_re, ngrams):
            term_counts[term][bucket] += 1
    return term_counts, dict(bucket_docs), papers, skipped, source


def rank_emerging_terms(
    term_counts: dict[str, Counter],
    bucket_docs: dict[str, int],
    min_df: int,
    top_n: int,
    include_texish: bool = False,
    candidate_terms: set[str] | None = None,
) -> list[TermTrend]:
    trends = []
    for term, counts in term_counts.items():
        if candidate_terms is not None and term not in candidate_terms:
            continue
        if not is_math_candidate(term, include_texish=include_texish):
            continue
        total = sum(counts.values())
        if total < min_df:
            continue
        trend = trend_for_series(term, dict(counts), bucket_docs)
        if trend.emergence_score > 0:
            trends.append(trend)
    trends.sort(key=lambda t: (-t.emergence_score, -t.df, t.term))
    return trends[:top_n]


def run(args: argparse.Namespace) -> dict:
    term_counts, bucket_docs, papers, skipped, extractor_source = build_diachronic_index(
        args.golden_dir,
        args.granularity,
        args.max_papers,
    )
    candidate_terms = None
    if args.candidate_source == "encyclopedia":
        candidate_terms = load_candidate_terms(args.concept_dir)
    ranked = rank_emerging_terms(
        term_counts,
        bucket_docs,
        args.min_df,
        args.top_n,
        args.include_texish,
        candidate_terms,
    )
    date_range = [min(bucket_docs), max(bucket_docs)] if bucket_docs else [None, None]
    result = {
        "meta": {
            "papers": papers,
            "terms": len(term_counts),
            "date_range": date_range,
            "buckets": len(bucket_docs),
            "granularity": args.granularity,
            "min_df": args.min_df,
            "include_texish": args.include_texish,
            "candidate_source": args.candidate_source,
            "candidate_terms": len(candidate_terms) if candidate_terms is not None else None,
            "extractor_source": extractor_source,
            "skipped": skipped[:20],
            "data_limit_note": (
                "This run uses the local dated CT mark-file corpus. Full diachronic power "
                "comes from running the same code over all arXiv/MSC mark files."
            ),
        },
        "emerging_terms": [asdict(t) for t in ranked],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def self_test() -> None:
    assert parse_arxiv_month("0705.0452") == "2007-05"
    assert parse_arxiv_month("2401.14311") == "2024-01"
    assert parse_arxiv_month("math/9811139") == "1998-11"
    assert parse_arxiv_month("math__0210114") == "2002-10"
    buckets = {str(y): 10 for y in range(2000, 2006)}
    rising = {str(y): max(0, y - 2001) for y in range(2000, 2006)}
    flat = {str(y): 2 for y in range(2000, 2006)}
    rising_trend = trend_for_series("rising", rising, buckets)
    flat_trend = trend_for_series("flat", flat, buckets)
    assert rising_trend.trend["slope"] > 0
    assert abs(flat_trend.trend["slope"]) < 1e-12
    assert rising_trend.emergence_score > flat_trend.emergence_score
    print("self-test ok")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mark3 H12 diachronic emergence detector")
    p.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--concept-dir", type=Path, default=DEFAULT_CONCEPT_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--granularity", choices=["year", "month"], default="year")
    p.add_argument("--min-df", type=int, default=4)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument(
        "--candidate-source",
        choices=["encyclopedia", "all"],
        default="encyclopedia",
        help="Rank known encyclopedia concepts by default; use all for raw extracted ngrams",
    )
    p.add_argument("--include-texish", action="store_true", help="Do not filter LaTeX/layout drift terms")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
