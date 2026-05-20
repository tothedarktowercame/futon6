#!/usr/bin/env python3
"""Build a daisychain audit of sentences not covered by Stage 5 discourse.

This supports a "Distributed Proofreaders" workflow:
  1. choose fresh papers not previously used in the audit chain
  2. extract uncovered sentences under the current discourse detector
  3. hand-review those residuals and add only general structure patterns
  4. advance to a new paper set on the next iteration
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import random
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_TAR = Path.home() / "code" / "storage" / "mark2" / "inbox" / "batch-008.tar.gz"
DEFAULT_OUT_DIR = ROOT / "data" / "showcases" / "distributed-proofreaders"
DEFAULT_LEDGER = DEFAULT_OUT_DIR / "daisychain-ledger.json"
DEFAULT_RUN_JSON = DEFAULT_OUT_DIR / "latest-audit.json"
DEFAULT_RUN_HTML = DEFAULT_OUT_DIR / "latest-audit.html"
DEFAULT_NER_KERNEL = Path.home() / "code" / "storage" / "futon6" / "data" / "ner-kernel" / "terms.tsv"
STRUCTURE_CUE_WORDS = {
    "we", "let", "define", "denote", "write", "show", "prove", "obtain", "apply",
    "study", "consider", "introduce", "recall", "if", "then", "assume", "suppose",
    "where", "when", "for", "any", "every", "there", "exists", "be",
    "that", "and", "or", "not", "only", "particular", "consist", "depend",
    "turn", "focus", "choose", "work",
}
STRUCTURE_CUE_LEMMAS = {
    "shows": "show",
    "proved": "prove",
    "proves": "prove",
    "obtains": "obtain",
    "obtained": "obtain",
    "applies": "apply",
    "applied": "apply",
    "studies": "study",
    "considered": "consider",
    "considers": "consider",
    "introduced": "introduce",
    "introduces": "introduce",
    "recalled": "recall",
    "recalls": "recall",
    "depends": "depend",
    "consists": "consist",
    "chooses": "choose",
    "chose": "choose",
    "worked": "work",
    "is": "be",
    "are": "be",
    "was": "be",
    "were": "be",
    "being": "be",
    "been": "be",
}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VIEWER = load_module("batch008_qc_audit_viewer", ROOT / "scripts" / "build-batch-008-qc-viewer.py")
LOAD_ARXIV = load_module("load_arxiv_ct_audit", ROOT / "scripts" / "load-arxiv-ct.py")
NLAB_WIRING = load_module("nlab_wiring_audit", ROOT / "scripts" / "nlab-wiring.py")
SUPERPOD_JOB = load_module("superpod_job_structure_audit", ROOT / "scripts" / "superpod-job.py")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-tar", type=Path, default=DEFAULT_BATCH_TAR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_RUN_JSON)
    parser.add_argument("--out-html", type=Path, default=DEFAULT_RUN_HTML)
    parser.add_argument("--ner-kernel", type=Path, default=DEFAULT_NER_KERNEL)
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument("--paper-count", type=int, default=3)
    parser.add_argument("--ct-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--max-uncovered", type=int, default=30)
    parser.add_argument("--min-sentence-chars", type=int, default=40)
    parser.add_argument("--advance-ledger", action="store_true", default=True)
    parser.add_argument("--no-advance-ledger", dest="advance_ledger", action="store_false")
    parser.add_argument(
        "--seed-signatures-json",
        type=Path,
        default=None,
        help=(
            "Path to a prior audit JSON; structure_seed_candidates from that file are "
            "loaded as seed signatures. Each residual in this run gets matched against "
            "them via in-order subsequence; the matched prior is recorded as "
            "matched_prior_signature on the residual. This is how cross-batch firings "
            "are counted."
        ),
    )
    return parser.parse_args(argv)


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    out = []
    for match in re.finditer(r"[^.!?\n][^.!?\n]*(?:[.!?](?=\s|$)|$)", text):
        start, end = match.span()
        snippet = text[start:end].strip()
        if end > start and snippet:
            out.append((start, end, snippet))
    return out


def record_position_key(record: dict) -> tuple[int, str]:
    if not isinstance(record, dict):
        return (10**12, "")
    content = record.get("hx/content") or {}
    pos = content.get("position")
    if not isinstance(pos, int):
        pos = 10**12
    return (pos, record.get("hx/id") or "")


def extract_sentence_term_features(
    sentence: str,
    singles: dict,
    multi_index: dict,
) -> dict:
    hits = SUPERPOD_JOB.spot_terms_entity(sentence, singles, multi_index)
    unique_terms = []
    seen = set()
    for row in hits:
        term = (row.get("term") or row.get("term_lower") or "").strip()
        if not term:
            continue
        term_lower = (row.get("term_lower") or term.lower()).strip().lower()
        if term_lower in seen:
            continue
        seen.add(term_lower)
        unique_terms.append({
            "term": term,
            "term_lower": term_lower,
            "canon": row.get("canon"),
        })
    return {
        "known_term_hit_count": len(unique_terms),
        "known_term_hits": unique_terms,
    }


def normalize_structure_seed_text(
    sentence: str,
    known_term_hits: list[dict],
) -> str:
    normalized = sentence
    for item in sorted(
        known_term_hits,
        key=lambda row: len(row.get("term_lower", "")),
        reverse=True,
    ):
        term = (item.get("term") or item.get("term_lower") or "").strip()
        if not term:
            continue
        variants = [term]
        if not term.endswith("s"):
            variants.append(f"{term}s")
        for variant in variants:
            pattern = re.compile(rf"\b{re.escape(variant)}\b", re.IGNORECASE)
            normalized = pattern.sub("<TERM>", normalized)
    normalized = re.sub(r"\$[^$]+\$", "<MATH>", normalized)
    normalized = re.sub(r"\\cite\{[^}]+\}", "<CITE>", normalized)
    normalized = re.sub(r"\[[^\]]+\]", "<CITE>", normalized)
    normalized = re.sub(r"\\[A-Za-z]+", "<CMD>", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def structure_seed_skeleton(normalized_template: str) -> str:
    tokens = re.findall(r"<[a-z]+>|[a-z]+", normalized_template)
    kept = []
    for token in tokens:
        if token.startswith("<") and token.endswith(">"):
            kept.append(token)
            continue
        lemma = STRUCTURE_CUE_LEMMAS.get(token, token)
        if lemma in STRUCTURE_CUE_WORDS:
            kept.append(lemma)
    collapsed = []
    for token in kept:
        if collapsed and collapsed[-1] == token and token.startswith("<"):
            continue
        collapsed.append(token)
    return " ".join(collapsed)


def extract_uncovered_sentences(
    text: str,
    records: list[dict],
    singles: dict,
    multi_index: dict,
    *,
    min_sentence_chars: int = 40,
    max_uncovered: int = 30,
) -> list[dict]:
    merged = VIEWER.merge_spans([
        span for record in records if (span := VIEWER.scope_span(record))
    ])
    rows = []
    for idx, (start, end, sentence) in enumerate(sentence_spans(text)):
        if len(sentence) < min_sentence_chars:
            continue
        if any(not (m_end <= start or m_start >= end) for m_start, m_end in merged):
            continue
        term_features = extract_sentence_term_features(sentence, singles, multi_index)
        structure_seed_template = normalize_structure_seed_text(
            sentence,
            term_features["known_term_hits"],
        )
        structure_seed_signature = structure_seed_skeleton(structure_seed_template)
        rows.append({
            "index": idx,
            "start": start,
            "end": end,
            "text": sentence,
            "has_math": ("$" in sentence or "\\" in sentence),
            "has_citation": ("\\cite" in sentence or "[" in sentence and "]" in sentence),
            **term_features,
            "structure_seed_template": structure_seed_template,
            "structure_seed_signature": structure_seed_signature,
            "annotation_status": "unreviewed",
            "proposed_structure": None,
            "notes": None,
        })
    rows.sort(
        key=lambda row: (
            row["known_term_hit_count"],
            row["has_math"],
            len(row["text"]),
        ),
        reverse=True,
    )
    return rows[:max_uncovered]


def summarize_uncovered_cues(uncovered_rows: list[dict]) -> list[dict]:
    counter = Counter()
    for row in uncovered_rows:
        text = row["text"]
        normalized = re.sub(r"\$[^$]*\$", "<math>", text)
        normalized = re.sub(r"\\[A-Za-z]+", "<cmd>", normalized)
        words = re.findall(r"[A-Za-z<>-]+", normalized.lower())
        if not words:
            continue
        cue = " ".join(words[:4])
        counter[cue] += 1
    return [{"cue": cue, "count": count} for cue, count in counter.most_common(12)]


def summarize_term_dense_uncovered(uncovered_rows: list[dict]) -> list[dict]:
    rows = [row for row in uncovered_rows if row.get("known_term_hit_count", 0) > 0]
    rows.sort(
        key=lambda row: (
            row.get("known_term_hit_count", 0),
            len(row.get("text", "")),
        ),
        reverse=True,
    )
    out = []
    for row in rows[:12]:
        out.append({
            "index": row["index"],
            "known_term_hit_count": row["known_term_hit_count"],
            "known_terms": [item["term_lower"] for item in row.get("known_term_hits", [])[:8]],
            "text": row["text"],
        })
    return out


def summarize_structure_seed_candidates(paper_reports: list[dict]) -> list[dict]:
    buckets = {}
    for paper in paper_reports:
        for row in paper.get("uncovered_sentences", []):
            if row.get("known_term_hit_count", 0) <= 0:
                continue
            template = row.get("structure_seed_signature") or ""
            if not template:
                continue
            bucket = buckets.setdefault(template, {
                "signature": template,
                "count": 0,
                "paper_ids": set(),
                "example_sentences": [],
                "max_known_term_hit_count": 0,
            })
            bucket["count"] += 1
            bucket["paper_ids"].add(paper.get("paper_id"))
            bucket["max_known_term_hit_count"] = max(
                bucket["max_known_term_hit_count"],
                row.get("known_term_hit_count", 0),
            )
            if len(bucket["example_sentences"]) < 3:
                bucket["example_sentences"].append({
                    "paper_id": paper.get("paper_id"),
                    "index": row.get("index"),
                    "text": row.get("text"),
                    "known_terms": [
                        item["term_lower"] for item in row.get("known_term_hits", [])[:8]
                    ],
                })
    rows = []
    for bucket in buckets.values():
        rows.append({
            "signature": bucket["signature"],
            "count": bucket["count"],
            "paper_ids": sorted(bucket["paper_ids"]),
            "paper_count": len(bucket["paper_ids"]),
            "max_known_term_hit_count": bucket["max_known_term_hit_count"],
            "example_sentences": bucket["example_sentences"],
        })
    rows.sort(
        key=lambda row: (
            row["paper_count"],
            row["count"],
            row["max_known_term_hit_count"],
            len(row["signature"]),
        ),
        reverse=True,
    )
    return rows[:24]


def select_daisychain_papers(
    batch_meta: dict[str, dict],
    available_ids: set[str],
    ledger: dict,
    *,
    paper_count: int,
    ct_count: int,
    seed: int,
) -> list[str]:
    previously_used = set()
    for run in ledger.get("runs", []):
        previously_used.update(run.get("paper_ids", []))
    unseen = [raw_id for raw_id in sorted(available_ids) if raw_id not in previously_used]
    if len(unseen) < paper_count:
        raise RuntimeError(
            f"Need {paper_count} unseen papers but only found {len(unseen)}. "
            "Use explicit --paper-id overrides or rotate the ledger."
        )

    run_index = len(ledger.get("runs", []))
    rng = random.Random(seed + run_index)
    ct_ids = [raw_id for raw_id in unseen if "math.CT" in (batch_meta.get(raw_id, {}).get("categories") or [])]
    non_ct_ids = [raw_id for raw_id in unseen if raw_id not in ct_ids]
    rng.shuffle(ct_ids)
    rng.shuffle(non_ct_ids)

    chosen = ct_ids[: min(ct_count, paper_count)]
    needed = paper_count - len(chosen)
    chosen.extend(non_ct_ids[:needed])
    if len(chosen) < paper_count:
        remainder = [raw_id for raw_id in unseen if raw_id not in chosen]
        chosen.extend(remainder[: paper_count - len(chosen)])
    return chosen[:paper_count]


def load_eprint_text(batch_tar: Path, raw_id: str) -> str:
    payload, suffix = VIEWER.extract_batch_eprint(batch_tar, raw_id)
    if payload is None or suffix is None:
        raise RuntimeError(f"Missing eprint payload for {raw_id}")
    with tempfile.NamedTemporaryFile(prefix="audit-", suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        return LOAD_ARXIV._read_payload(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def build_paper_audit(
    batch_tar: Path,
    batch_meta: dict[str, dict],
    raw_id: str,
    args: argparse.Namespace,
    singles: dict,
    multi_index: dict,
    seed_signatures: list | None = None,
) -> dict:
    text = load_eprint_text(batch_tar, raw_id)
    scopes = NLAB_WIRING.detect_scopes(raw_id, text) or []
    wires = NLAB_WIRING.detect_wires(raw_id, text) or []
    ports = NLAB_WIRING.detect_ports(raw_id, text) or []
    labels = NLAB_WIRING.detect_labels(raw_id, text) or []
    discourse = sorted([*scopes, *wires, *ports, *labels], key=record_position_key)
    scope_coverage = VIEWER.scope_coverage_stats(text, scopes)
    discourse_coverage = VIEWER.scope_coverage_stats(text, discourse)
    uncovered_rows = extract_uncovered_sentences(
        text,
        discourse,
        singles,
        multi_index,
        min_sentence_chars=args.min_sentence_chars,
        max_uncovered=args.max_uncovered,
    )
    uncovered_with_known_terms = sum(
        1 for row in uncovered_rows if row.get("known_term_hit_count", 0) > 0
    )
    residuals_with_seed_match = 0
    if seed_signatures:
        for row in uncovered_rows:
            if row.get("known_term_hit_count", 0) <= 0:
                continue
            matched = SUPERPOD_JOB._match_structure_seed_signature(
                row.get("structure_seed_signature") or "",
                seed_signatures,
            )
            row["matched_prior_signature"] = matched
            if matched:
                residuals_with_seed_match += 1
    return {
        "paper_id": raw_id,
        "title": batch_meta.get(raw_id, {}).get("title", ""),
        "categories": batch_meta.get(raw_id, {}).get("categories") or [],
        "scope_count": len(scopes),
        "wire_count": len(wires),
        "port_count": len(ports),
        "label_count": len(labels),
        "scope_coverage": scope_coverage,
        "discourse_coverage": discourse_coverage,
        "coverage_lift": {
            "char": round(discourse_coverage["char_coverage"] - scope_coverage["char_coverage"], 4),
            "sentence": round(discourse_coverage["sentence_coverage"] - scope_coverage["sentence_coverage"], 4),
        },
        "known_term_feature_source": str(args.ner_kernel),
        "uncovered_sentence_count": len(uncovered_rows),
        "uncovered_sentences_with_known_terms": uncovered_with_known_terms,
        "uncovered_sentence_known_term_ratio": (
            round(uncovered_with_known_terms / len(uncovered_rows), 4)
            if uncovered_rows else 0.0
        ),
        "residuals_with_seed_match": residuals_with_seed_match,
        "uncovered_sentences": uncovered_rows,
        "top_uncovered_cues": summarize_uncovered_cues(uncovered_rows),
        "term_dense_uncovered_sentences": summarize_term_dense_uncovered(uncovered_rows),
    }


def render_html(report: dict) -> str:
    structure_seed_html = "".join(
        "<li>"
        f"<code>{html.escape(row['signature'])}</code> "
        f"(papers={row['paper_count']}, hits={row['count']}, max-known-terms={row['max_known_term_hit_count']})<br>"
        + "".join(
            f"{html.escape(example['paper_id'])}#{example['index']}: "
            f"{html.escape(example['text'])}<br>"
            for example in row["example_sentences"]
        )
        + "</li>"
        for row in report.get("structure_seed_candidates", [])
    ) or "<li>No cross-paper structure seed candidates.</li>"
    papers_html = []
    for paper in report["papers"]:
        uncovered = "".join(
            "<li>"
            f"<code>{row['index']}</code> "
            f"{html.escape(row['text'])}"
            "</li>"
            for row in paper["uncovered_sentences"]
        ) or "<li>No uncovered sentences under current filter.</li>"
        cues = "".join(
            f"<li><code>{html.escape(cue['cue'])}</code> × {cue['count']}</li>"
            for cue in paper["top_uncovered_cues"]
        ) or "<li>No repeated cues.</li>"
        term_dense = "".join(
            "<li>"
            f"<code>{row['index']}</code> "
            f"[{row['known_term_hit_count']} known terms] "
            f"{html.escape(', '.join(row['known_terms']))}<br>"
            f"{html.escape(row['text'])}"
            "</li>"
            for row in paper["term_dense_uncovered_sentences"]
        ) or "<li>No known-term-dense uncovered sentences.</li>"
        papers_html.append(
            f"""
            <section>
              <h2>{html.escape(paper['paper_id'])}</h2>
              <p><strong>{html.escape(paper['title'])}</strong><br>{html.escape(', '.join(paper['categories']))}</p>
              <p>
                Scope sentence coverage: {paper['scope_coverage']['sentence_coverage']:.1%}<br>
                Discourse sentence coverage: {paper['discourse_coverage']['sentence_coverage']:.1%}<br>
                Remaining uncovered sentences: {paper['uncovered_sentence_count']}<br>
                Uncovered with known terms: {paper['uncovered_sentences_with_known_terms']} ({paper['uncovered_sentence_known_term_ratio']:.1%})
              </p>
              <h3>Top Uncovered Cues</h3>
              <ul>{cues}</ul>
              <h3>Known-Term-Dense Residuals</h3>
              <ol>{term_dense}</ol>
              <h3>Uncovered Sentences</h3>
              <ol>{uncovered}</ol>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Distributed Proofreaders Audit</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 1100px; line-height: 1.45; color: #17202a; }}
    section {{ border-top: 1px solid #d5dbdb; padding-top: 1rem; margin-top: 1.5rem; }}
    code {{ background: #f4f6f6; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Distributed Proofreaders Audit</h1>
  <p>Generated {html.escape(report['generated_at'])}. Run index: <code>{report['run_index']}</code>.</p>
  <p>Method: choose fresh papers from the daisychain ledger, measure current discourse coverage, then surface only the residual uncovered sentences for manual structure review.</p>
  <p>Feature enrichment: each uncovered sentence is also annotated with ordinary Stage 5 known-term hits from <code>{html.escape(report['ner_kernel'])}</code>, so structure review can prioritize term-dense residual prose.</p>
  <h2>Cross-Paper Structure Seed Candidates</h2>
  <ol>{structure_seed_html}</ol>
  {''.join(papers_html)}
</body>
</html>"""


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)
    batch_meta = VIEWER.load_batch_metadata(args.batch_tar)
    available_ids = VIEWER.load_available_eprint_ids(args.batch_tar)
    ledger = read_json(args.ledger, {"runs": []})
    selected_papers = args.paper_ids or select_daisychain_papers(
        batch_meta,
        available_ids,
        ledger,
        paper_count=args.paper_count,
        ct_count=args.ct_count,
        seed=args.seed,
    )
    seed_signatures = SUPERPOD_JOB._load_structure_seed_signatures(args.seed_signatures_json)
    paper_reports = [
        build_paper_audit(
            args.batch_tar, batch_meta, raw_id, args, singles, multi_index,
            seed_signatures=seed_signatures,
        )
        for raw_id in selected_papers
    ]
    seed_matches_applied = sum(p.get("residuals_with_seed_match", 0) for p in paper_reports)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_index": len(ledger.get("runs", [])),
        "ner_kernel": str(args.ner_kernel),
        "paper_ids": selected_papers,
        "seed_signatures_json": str(args.seed_signatures_json) if args.seed_signatures_json else None,
        "seed_signatures_loaded": len(seed_signatures),
        "seed_matches_applied": seed_matches_applied,
        "papers": paper_reports,
        "structure_seed_candidates": summarize_structure_seed_candidates(paper_reports),
    }
    write_json(args.out_json, report)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(render_html(report), encoding="utf-8")

    if args.advance_ledger:
        ledger.setdefault("runs", []).append({
            "generated_at": report["generated_at"],
            "paper_ids": selected_papers,
            "out_json": str(args.out_json),
            "out_html": str(args.out_html),
        })
        write_json(args.ledger, ledger)

    print(f"Selected papers: {', '.join(selected_papers)}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_html}")
    if seed_signatures:
        print(
            f"Seed replay: {len(seed_signatures)} prior signatures loaded, "
            f"{seed_matches_applied} cross-batch matches applied"
        )
    if args.advance_ledger:
        print(f"Advanced ledger: {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
