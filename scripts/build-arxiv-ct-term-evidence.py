#!/usr/bin/env python3
"""Build open-term evidence from the local arXiv math.CT corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import edn_format


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from futon6.theorem_extraction import extract_from_tarball


DEFAULT_REPO_DATA_ROOT = ROOT / "data"
DEFAULT_STORAGE_DATA_ROOT = Path.home() / "code" / "storage" / "futon6" / "data"
DEFAULT_PM_SEED = ROOT / "data" / "dictionary" / "entries-pm-seed.edn"
DEFAULT_NLAB_SEED = ROOT / "data" / "dictionary" / "entries-nlab-seed.edn"
DEFAULT_OUT_DIR = ROOT / "data" / "dictionary"
DEFAULT_NNEXUS_STOPWORDS = Path.home() / "code" / "nnexus" / "lib" / "NNexus" / "StopWordList.pm"
DEFAULT_NNEXUS_SNAPSHOT = Path.home() / "code" / "nnexus" / "lib" / "NNexus" / "resources" / "database" / "snapshot-6-2014.sqlite"
PROGRESS_EVERY = 250
MAX_CONTEXT_CHARS = 500

DEFINITIONAL_SOURCES = {
    "called-as",
    "is-called",
    "defined-as",
    "definition-of",
    "definition-block-subject",
}

LOCAL_DEFINITIONAL_RE = re.compile(
    r"\b(?:is\s+defined\s+as|defined\s+to\s+be|we\s+call|is\s+called|definition\s+of)\b",
    re.IGNORECASE,
)

GENERIC_SINGLE_WORDS = {
    "free",
    "internal",
    "external",
    "standard",
    "connected",
    "coherent",
    "reduced",
    "cartesian",
    "monoidal",
    "simplicial",
    "cubical",
    "abelian",
    "braided",
    "symmetric",
    "weak",
    "strict",
    "finite",
    "small",
    "large",
    "complete",
    "cocomplete",
    "exact",
    "closed",
    "pointed",
    "presentable",
    "thin",
    "left",
    "right",
    "upper",
    "lower",
    "objects",
    "object",
    "spaces",
    "space",
    "maps",
    "map",
    "morphism",
    "morphisms",
    "cells",
    "cell",
    "arrows",
    "arrow",
    "sheaves",
    "sheaf",
    "functors",
    "functor",
    "what",
}

PHRASE_BAN_TOKENS = {
    "this",
    "these",
    "that",
    "those",
    "smallest",
    "largest",
    "least",
    "greatest",
}

ADJECTIVAL_SUFFIXES = (
    "al",
    "ial",
    "ical",
    "ic",
    "ary",
    "ory",
    "ive",
    "less",
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LOAD_ARXIV = load_module("load_arxiv_ct", ROOT / "scripts" / "load-arxiv-ct.py")
PM_SEED = load_module("seed_dictionary_from_pm", ROOT / "scripts" / "seed-dictionary-from-pm.py")
SUPERPOD_JOB = load_module("superpod_job_term_discovery", ROOT / "scripts" / "superpod-job.py")


def default_data_root() -> Path:
    if (DEFAULT_STORAGE_DATA_ROOT / "arxiv-math-ct-metadata.jsonl").exists():
        return DEFAULT_STORAGE_DATA_ROOT
    return DEFAULT_REPO_DATA_ROOT


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--pm-seed", type=Path, default=DEFAULT_PM_SEED)
    parser.add_argument("--nlab-seed", type=Path, default=DEFAULT_NLAB_SEED)
    parser.add_argument("--nnexus-stopwords", type=Path, default=DEFAULT_NNEXUS_STOPWORDS)
    parser.add_argument("--nnexus-snapshot", type=Path, default=DEFAULT_NNEXUS_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-candidates-per-paper", type=int, default=64)
    parser.add_argument("--max-lhs-contexts", type=int, default=3)
    parser.add_argument("--max-rhs-contexts", type=int, default=5)
    parser.add_argument("--timestamp", help="Stable UTC timestamp for deterministic outputs.")
    return parser.parse_args(argv)


def load_known_term_lowers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    raw = edn_format.loads(path.read_text(encoding="utf-8"))
    entries = raw[edn_format.Keyword("dictionary/entries")]
    out = set()
    for entry in entries:
        lower = entry.get(edn_format.Keyword("term/lower"))
        if lower:
            out.add(str(lower))
    return out


def load_nnexus_stopwords(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"return \[qw/(.*?)/\];", text, re.DOTALL)
    if not match:
        return set()
    return {token.strip().lower() for token in match.group(1).split() if token.strip()}


def load_nnexus_concept_lowers(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    out = set()
    pat = re.compile(r'^INSERT INTO "concepts" VALUES\(\d+,\'([^\']*)\',\'([^\']*)\',')
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pat.match(line)
            if not match:
                continue
            firstword, tailwords = match.groups()
            concept = " ".join(part for part in [firstword, tailwords] if part).strip()
            if concept:
                out.add(concept.lower())
    return out


def normalize_text(text: str) -> str:
    return PM_SEED.collapse_whitespace(PM_SEED.latex_to_text(text))


def contains_term(term_lower: str, text: str) -> bool:
    normalized = normalize_text(text).lower()
    return term_lower in normalized


def trim_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    collapsed = normalize_text(text)
    return collapsed[:max_chars]


def context_looks_definitional(text: str) -> bool:
    return bool(LOCAL_DEFINITIONAL_RE.search(normalize_text(text)))


def is_single_word_quality_term(
    term_lower: str,
    *,
    known_in_pm_seed: bool,
    known_in_nlab_seed: bool,
    entity_count: int,
    rhs_support_counts: dict[str, int],
    nnexus_stopwords: set[str],
) -> bool:
    if known_in_pm_seed or known_in_nlab_seed:
        return True
    if " " in term_lower:
        return True
    if term_lower in nnexus_stopwords or term_lower in GENERIC_SINGLE_WORDS:
        return False
    if any(term_lower.endswith(suffix) for suffix in ADJECTIVAL_SUFFIXES):
        return False
    if entity_count < 2:
        return False
    if not (rhs_support_counts.get("definition-env", 0) or rhs_support_counts.get("local-definitional-context", 0)):
        return False
    return True


def is_multiword_quality_term(
    term_lower: str,
    *,
    known_in_pm_seed: bool,
    known_in_nlab_seed: bool,
    nnexus_stopwords: set[str],
) -> bool:
    if known_in_pm_seed or known_in_nlab_seed:
        return True
    tokens = term_lower.split()
    if any(token in PHRASE_BAN_TOKENS for token in tokens):
        return False
    stopword_count = sum(1 for token in tokens if token in nnexus_stopwords)
    if stopword_count > 1:
        return False
    return True


def iter_index_rows(data_root: Path, *, limit: int | None = None) -> Iterator[dict]:
    index_path = data_root / "arxiv-math-ct-file-index.jsonl"
    produced = 0
    for row in LOAD_ARXIV._iter_jsonl(index_path):
        if limit is not None and produced >= limit:
            break
        if not row.get("has_local_file"):
            continue
        yield row
        produced += 1


def rhs_contexts_for_paper(paper_id: str, local_path: Path) -> list[dict]:
    result = extract_from_tarball(str(local_path), paper_id)
    contexts = []
    for definition in result.definitions:
        contexts.append({
            "kind": "definition-env",
            "label": definition.get("label") or None,
            "section": definition.get("section") or None,
            "text": trim_context(definition.get("content", "")),
        })
    for theorem in result.theorems:
        contexts.append({
            "kind": "theorem-statement",
            "label": theorem.label or None,
            "section": theorem.section or None,
            "text": trim_context(theorem.statement),
        })
    return [ctx for ctx in contexts if ctx["text"]]


def seed_membership(term_lower: str, pm_lowers: set[str], nlab_lowers: set[str]) -> dict[str, bool | str]:
    in_pm = term_lower in pm_lowers
    in_nlab = term_lower in nlab_lowers
    return {
        "known_in_pm_seed": in_pm,
        "known_in_nlab_seed": in_nlab,
        "novel_vs_seed": "novel" if not (in_pm or in_nlab) else "known",
    }


def extended_seed_membership(term_lower: str, pm_lowers: set[str], nlab_lowers: set[str], nnexus_lowers: set[str]) -> dict[str, bool | str]:
    in_pm = term_lower in pm_lowers
    in_nlab = term_lower in nlab_lowers
    in_nnexus = term_lower in nnexus_lowers
    return {
        "known_in_pm_seed": in_pm,
        "known_in_nlab_seed": in_nlab,
        "known_in_nnexus_snapshot": in_nnexus,
        "novel_vs_seed": "novel" if not (in_pm or in_nlab or in_nnexus) else "known",
    }


def add_context_limited(bucket: list[dict], row: dict, limit: int) -> None:
    if len(bucket) >= limit:
        return
    key = (row.get("paper_id"), row.get("kind"), row.get("text"))
    existing = {
        (item.get("paper_id"), item.get("kind"), item.get("text"))
        for item in bucket
    }
    if key not in existing:
        bucket.append(row)


def build_term_evidence(args: argparse.Namespace) -> dict:
    started = time.time()
    timestamp_iso = args.timestamp or PM_SEED.iso_utc_now()
    pm_lowers = load_known_term_lowers(args.pm_seed)
    nlab_lowers = load_known_term_lowers(args.nlab_seed)
    nnexus_stopwords = load_nnexus_stopwords(args.nnexus_stopwords)
    nnexus_lowers = load_nnexus_concept_lowers(args.nnexus_snapshot)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    aggregates: dict[str, dict] = {}
    papers_processed = 0
    papers_with_candidates = 0
    total_candidates = 0
    rhs_context_total = 0
    rhs_match_total = 0
    source_counter = Counter()

    for row in iter_index_rows(args.data_root, limit=args.limit):
        papers_processed += 1
        local_file = row.get("local_file") or row.get("path")
        local_path = LOAD_ARXIV._resolve_local_path(local_file, args.data_root)
        body_text = LOAD_ARXIV._read_payload(local_path)
        if not body_text:
            continue

        paper_id = row["id"]
        candidates = SUPERPOD_JOB.extract_open_ner_candidates(
            body_text, max_per_entity=args.max_candidates_per_paper
        )
        if candidates:
            papers_with_candidates += 1
        rhs_contexts = rhs_contexts_for_paper(paper_id, local_path)
        rhs_context_total += len(rhs_contexts)

        for term_lower, source, lhs_context in candidates:
            total_candidates += 1
            source_counter[source] += 1
            record = aggregates.setdefault(
                term_lower,
                {
                    "term_lower": term_lower,
                    "candidate_count": 0,
                    "entity_ids": set(),
                    "sources": Counter(),
                    "lhs_contexts": [],
                    "rhs_support_counts": Counter(),
                    "supporting_contexts": [],
                    **extended_seed_membership(term_lower, pm_lowers, nlab_lowers, nnexus_lowers),
                },
            )
            record["candidate_count"] += 1
            record["entity_ids"].add(paper_id)
            record["sources"][source] += 1

            add_context_limited(
                record["lhs_contexts"],
                {
                    "paper_id": paper_id,
                    "source": source,
                    "kind": "lhs-candidate-context",
                    "text": lhs_context[:MAX_CONTEXT_CHARS],
                },
                args.max_lhs_contexts,
            )

            if source in DEFINITIONAL_SOURCES or context_looks_definitional(lhs_context):
                record["rhs_support_counts"]["local-definitional-context"] += 1
                add_context_limited(
                    record["supporting_contexts"],
                    {
                        "paper_id": paper_id,
                        "source": source,
                        "kind": "local-definitional-context",
                        "text": lhs_context[:MAX_CONTEXT_CHARS],
                    },
                    args.max_rhs_contexts,
                )
                rhs_match_total += 1

            for context in rhs_contexts:
                if not contains_term(term_lower, context["text"]):
                    continue
                record["rhs_support_counts"][context["kind"]] += 1
                add_context_limited(
                    record["supporting_contexts"],
                    {
                        "paper_id": paper_id,
                        "source": source,
                        "kind": context["kind"],
                        "label": context.get("label"),
                        "section": context.get("section"),
                        "text": context["text"],
                    },
                    args.max_rhs_contexts,
                )
                rhs_match_total += 1

        if papers_processed % PROGRESS_EVERY == 0:
            print(
                f"Processed {papers_processed} papers; {len(aggregates)} unique candidate terms so far...",
                flush=True,
            )

    rows = []
    novel_terms = 0
    known_pm = 0
    known_nlab = 0
    known_nnexus = 0
    rhs_supported_terms = 0
    filtered_out_terms = 0
    prefilter_unique_terms = len(aggregates)
    for term_lower, record in aggregates.items():
        entity_ids = sorted(record.pop("entity_ids"))
        rhs_support_counts = dict(sorted(record["rhs_support_counts"].items()))
        if " " in term_lower:
            keep = is_multiword_quality_term(
                term_lower,
                known_in_pm_seed=record["known_in_pm_seed"],
                known_in_nlab_seed=record["known_in_nlab_seed"],
                nnexus_stopwords=nnexus_stopwords,
            )
        else:
            keep = is_single_word_quality_term(
                term_lower,
                known_in_pm_seed=record["known_in_pm_seed"],
                known_in_nlab_seed=record["known_in_nlab_seed"],
                entity_count=len(entity_ids),
                rhs_support_counts=rhs_support_counts,
                nnexus_stopwords=nnexus_stopwords,
            )
        if not keep:
            filtered_out_terms += 1
            continue
        row = {
            "term_lower": term_lower,
            "candidate_count": record["candidate_count"],
            "entity_count": len(entity_ids),
            "paper_ids": entity_ids[:25],
            "sources": dict(sorted(record["sources"].items())),
            "lhs_contexts": record["lhs_contexts"],
            "rhs_support_counts": rhs_support_counts,
            "supporting_contexts": record["supporting_contexts"],
            "known_in_pm_seed": record["known_in_pm_seed"],
            "known_in_nlab_seed": record["known_in_nlab_seed"],
            "known_in_nnexus_snapshot": record["known_in_nnexus_snapshot"],
            "novel_vs_seed": record["novel_vs_seed"],
        }
        if row["novel_vs_seed"] == "novel":
            novel_terms += 1
        if row["known_in_pm_seed"]:
            known_pm += 1
        if row["known_in_nlab_seed"]:
            known_nlab += 1
        if row["known_in_nnexus_snapshot"]:
            known_nnexus += 1
        if row["rhs_support_counts"]:
            rhs_supported_terms += 1
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["novel_vs_seed"] != "novel",
            -row["entity_count"],
            -row["candidate_count"],
            -sum(row["rhs_support_counts"].values()),
            row["term_lower"],
        )
    )

    evidence_path = args.out_dir / "arxiv-ct-open-term-evidence.jsonl"
    summary_path = args.out_dir / "arxiv-ct-open-term-evidence-summary.json"
    with evidence_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = round(time.time() - started, 3)
    summary = {
        "timestamp": timestamp_iso,
        "data_root": str(args.data_root),
        "pm_seed": str(args.pm_seed),
        "nlab_seed": str(args.nlab_seed),
        "nnexus_snapshot": str(args.nnexus_snapshot) if args.nnexus_snapshot else None,
        "nnexus_stopwords": str(args.nnexus_stopwords) if args.nnexus_stopwords else None,
        "nnexus_concept_count": len(nnexus_lowers),
        "nnexus_stopword_count": len(nnexus_stopwords),
        "papers_processed": papers_processed,
        "papers_with_candidates": papers_with_candidates,
        "prefilter_unique_candidate_terms": prefilter_unique_terms,
        "unique_candidate_terms": len(rows),
        "filtered_out_terms": filtered_out_terms,
        "total_candidates": total_candidates,
        "rhs_context_total": rhs_context_total,
        "rhs_match_total": rhs_match_total,
        "novel_terms": novel_terms,
        "known_in_pm_seed": known_pm,
        "known_in_nlab_seed": known_nlab,
        "known_in_nnexus_snapshot": known_nnexus,
        "rhs_supported_terms": rhs_supported_terms,
        "candidate_source_counts": dict(sorted(source_counter.items())),
        "output_jsonl": str(evidence_path),
        "elapsed_seconds": elapsed,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(rows)} arXiv CT term-evidence rows from {papers_processed} papers in {elapsed:.3f}s.",
        flush=True,
    )
    return {
        "rows": rows,
        "summary": summary,
        "evidence_path": evidence_path,
        "summary_path": summary_path,
    }


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv or sys.argv[1:])
    return build_term_evidence(args)


if __name__ == "__main__":
    main()
