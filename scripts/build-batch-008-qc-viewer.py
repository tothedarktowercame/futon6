#!/usr/bin/env python3
"""Build a small NNexus-Glasses-style QC viewer over Rob's batch-008 results.

Reads:
  - /home/joe/code/storage/mark2/inbox/batch-008.tar.gz
  - /home/joe/code/storage/mark2/outbox/results-008.tar.gz

Outputs:
  - data/showcases/batch-008-math-ct-qc.html
  - data/showcases/batch-008-math-ct-qc.json

The viewer is intentionally narrow: it picks one or two math.CT papers from the
returned result set, then shows:
  - returned scopes from results-008 over the result text
  - returned hypergraph / reverse-morphogenesis stats
  - local scope detection over the raw eprint
  - local filtered open-term evidence over the raw eprint
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_BATCH_TAR = Path.home() / "code" / "storage" / "mark2" / "inbox" / "batch-008.tar.gz"
DEFAULT_RESULTS_TAR = Path.home() / "code" / "storage" / "mark2" / "outbox" / "results-008.tar.gz"
DEFAULT_OUT_HTML = ROOT / "data" / "showcases" / "batch-008-math-ct-qc.html"
DEFAULT_OUT_JSON = ROOT / "data" / "showcases" / "batch-008-math-ct-qc.json"
DEFAULT_PM_SEED = ROOT / "data" / "dictionary" / "entries-pm-seed.edn"
DEFAULT_NLAB_SEED = ROOT / "data" / "dictionary" / "entries-nlab-seed.edn"
DEFAULT_NNEXUS_STOPWORDS = Path.home() / "code" / "nnexus" / "lib" / "NNexus" / "StopWordList.pm"
DEFAULT_NNEXUS_SNAPSHOT = Path.home() / "code" / "nnexus" / "lib" / "NNexus" / "resources" / "database" / "snapshot-6-2014.sqlite"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NLAB_WIRING = load_module("nlab_wiring_batch008_qc", ROOT / "scripts" / "nlab-wiring.py")
TERM_EVIDENCE = load_module("build_arxiv_ct_term_evidence_qc", ROOT / "scripts" / "build-arxiv-ct-term-evidence.py")
SUPERPOD_JOB = TERM_EVIDENCE.SUPERPOD_JOB
from futon6.theorem_extraction import extract_from_tarball


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-tar", type=Path, default=DEFAULT_BATCH_TAR)
    parser.add_argument("--results-tar", type=Path, default=DEFAULT_RESULTS_TAR)
    parser.add_argument("--out-html", type=Path, default=DEFAULT_OUT_HTML)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--paper-id", action="append", dest="paper_ids",
                        help="Raw batch paper id like 0710.3853v1 or math/0606735v1. Repeatable.")
    parser.add_argument("--paper-count", type=int, default=2)
    parser.add_argument("--pm-seed", type=Path, default=DEFAULT_PM_SEED)
    parser.add_argument("--nlab-seed", type=Path, default=DEFAULT_NLAB_SEED)
    parser.add_argument("--nnexus-stopwords", type=Path, default=DEFAULT_NNEXUS_STOPWORDS)
    parser.add_argument("--nnexus-snapshot", type=Path, default=DEFAULT_NNEXUS_SNAPSHOT)
    parser.add_argument("--max-local-terms", type=int, default=12)
    return parser.parse_args(argv)


def raw_to_entity_id(raw_id: str) -> str:
    return f"arxiv-{raw_id}"


def entity_to_raw_id(entity_id: str) -> str:
    if not entity_id.startswith("arxiv-"):
        raise ValueError(f"unexpected entity id: {entity_id}")
    return entity_id[len("arxiv-"):]


def load_batch_metadata(batch_tar: Path) -> dict[str, dict]:
    with tarfile.open(batch_tar, "r:gz") as tf:
        fh = tf.extractfile("batch-008/batch-008.jsonl")
        assert fh is not None
        return {row["id"]: row for row in map(json.loads, fh)}


def load_results(results_tar: Path) -> dict[str, dict]:
    with tarfile.open(results_tar, "r:gz") as tf:
        entities = json.load(tf.extractfile("output/entities.json"))
        scopes = json.load(tf.extractfile("output/scopes.json"))
        ner = json.load(tf.extractfile("output/ner-terms.json"))
        paper_hg = json.load(tf.extractfile("output/paper-hypergraphs.json"))
        reverse = json.load(tf.extractfile("output/reverse-morphogenesis.json"))
        manifest = json.load(tf.extractfile("output/manifest.json"))
    return {
        "entities": {row["entity/id"]: row for row in entities},
        "scopes": {row["entity_id"]: row for row in scopes},
        "ner_terms": {row["entity_id"]: row for row in ner},
        "paper_hypergraphs": {row["paper_id"]: row for row in paper_hg},
        "reverse_morphogenesis": {row["entity_id"]: row for row in reverse},
        "manifest": manifest,
    }


def pick_default_papers(batch_meta: dict[str, dict], results: dict[str, dict], paper_count: int) -> list[str]:
    rows = []
    for raw_id, meta in batch_meta.items():
        if "math.CT" not in (meta.get("categories") or []):
            continue
        entity_id = raw_to_entity_id(raw_id)
        scope_row = results["scopes"].get(entity_id, {})
        hg_row = results["paper_hypergraphs"].get(entity_id, {})
        rev_row = results["reverse_morphogenesis"].get(entity_id, {})
        scopes = scope_row.get("count", 0)
        terms = results["ner_terms"].get(entity_id, {}).get("count", 0)
        edges = len(hg_row.get("edges") or [])
        nodes = len(hg_row.get("nodes") or [])
        score = (10 if scopes > 0 else 0) + min(scopes, 25) + min(edges, 120) + min(terms, 40)
        rows.append({
            "raw_id": raw_id,
            "entity_id": entity_id,
            "title": meta.get("title", ""),
            "scope_count": scopes,
            "term_count": terms,
            "nodes": nodes,
            "edges": edges,
            "reverse_status": rev_row.get("status"),
            "score": score,
        })
    rows.sort(key=lambda row: (row["score"], row["scope_count"], row["edges"], row["term_count"]), reverse=True)
    chosen = [row["raw_id"] for row in rows if row["scope_count"] > 0][:paper_count]
    if len(chosen) < paper_count:
        for row in rows:
            if row["raw_id"] in chosen:
                continue
            chosen.append(row["raw_id"])
            if len(chosen) >= paper_count:
                break
    return chosen


def find_member(tf: tarfile.TarFile, path: str) -> tarfile.TarInfo | None:
    try:
        return tf.getmember(path)
    except KeyError:
        return None


def extract_batch_eprint(batch_tar: Path, raw_id: str) -> tuple[bytes | None, str | None]:
    with tarfile.open(batch_tar, "r:gz") as tf:
        for suffix in (".tar.gz", ".bin"):
            member = find_member(tf, f"batch-008/eprints/{raw_id}{suffix}")
            if member is None:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                return None, None
            return fh.read(), suffix
    return None, None


def load_eprint_text_and_theorems(raw_id: str, payload: bytes, suffix: str) -> tuple[str, object]:
    with tempfile.NamedTemporaryFile(prefix="batch-008-", suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        theorem_result = extract_from_tarball(str(tmp_path), raw_id)
        text = TERM_EVIDENCE.LOAD_ARXIV._read_payload(tmp_path)
        return text, theorem_result
    finally:
        tmp_path.unlink(missing_ok=True)


def top_edge_types(paper_hg: dict) -> list[tuple[str, int]]:
    counter = Counter()
    for edge in paper_hg.get("edges") or []:
        label = edge.get("type") or "?"
        subtype = edge.get("subtype")
        if subtype:
            label = f"{label}/{subtype}"
        counter[label] += 1
    return counter.most_common(8)


def top_node_types(paper_hg: dict) -> list[tuple[str, int]]:
    counter = Counter()
    for node in paper_hg.get("nodes") or []:
        label = node.get("type") or "?"
        subtype = node.get("subtype")
        if subtype:
            label = f"{label}/{subtype}"
        counter[label] += 1
    return counter.most_common(8)


def scope_snippet(text: str, scopes: list[dict], window: int = 180) -> tuple[str, int]:
    if not text:
        return "", 0
    positions = [s.get("hx/content", {}).get("position") for s in scopes if isinstance(s.get("hx/content", {}).get("position"), int)]
    if not positions:
        start = 0
        return text[: min(len(text), window * 2)], start
    start = max(0, min(positions) - window)
    max_end = max(s.get("hx/content", {}).get("end", p) for s, p in zip(scopes, positions))
    end = min(len(text), max_end + window)
    return text[start:end], start


def local_scope_snippet(text: str, scopes: list[dict], window: int = 260) -> tuple[str, int]:
    return scope_snippet(text, scopes, window=window)


def render_scope_markup(text: str, scopes: list[dict], offset: int = 0, limit: int = 8) -> str:
    spans = []
    for scope in scopes[:limit]:
        content = scope.get("hx/content", {})
        start = content.get("position")
        end = content.get("end")
        if not isinstance(start, int):
            continue
        if not isinstance(end, int):
            match = content.get("match", "")
            end = start + len(match)
        spans.append({
            "start": max(0, start - offset),
            "end": max(0, end - offset),
            "label": scope.get("hx/type", "?"),
        })
    spans.sort(key=lambda row: (row["start"], row["end"]))
    out = []
    cursor = 0
    for span in spans:
        start = max(0, min(len(text), span["start"]))
        end = max(start, min(len(text), span["end"]))
        if start < cursor:
            continue
        out.append(html.escape(text[cursor:start]))
        frag = html.escape(text[start:end])
        out.append(
            f'<mark class="scope"><span class="scope-label">{html.escape(span["label"])}</span>{frag}</mark>'
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def pick_local_terms(
    text: str,
    raw_id: str,
    theorem_result,
    pm_lowers: set[str],
    nlab_lowers: set[str],
    nnexus_lowers: set[str],
    nnexus_stopwords: set[str],
    max_terms: int,
) -> list[dict]:
    rhs_contexts = []
    for definition in theorem_result.definitions:
        rhs_contexts.append({
            "kind": "definition-env",
            "text": TERM_EVIDENCE.trim_context(definition.get("content", "")),
        })
    for theorem in theorem_result.theorems:
        rhs_contexts.append({
            "kind": "theorem-statement",
            "text": TERM_EVIDENCE.trim_context(theorem.statement),
        })

    records = []
    for term_lower, source, lhs_context in SUPERPOD_JOB.extract_open_ner_candidates(text, max_per_entity=80):
        membership = TERM_EVIDENCE.extended_seed_membership(term_lower, pm_lowers, nlab_lowers, nnexus_lowers)
        rhs_counts = Counter()
        if source in TERM_EVIDENCE.DEFINITIONAL_SOURCES or TERM_EVIDENCE.context_looks_definitional(lhs_context):
            rhs_counts["local-definitional-context"] += 1
        for context in rhs_contexts:
            if TERM_EVIDENCE.contains_term(term_lower, context["text"]):
                rhs_counts[context["kind"]] += 1
        if " " in term_lower:
            keep = TERM_EVIDENCE.is_multiword_quality_term(
                term_lower,
                known_in_pm_seed=membership["known_in_pm_seed"],
                known_in_nlab_seed=membership["known_in_nlab_seed"],
                nnexus_stopwords=nnexus_stopwords,
            )
        else:
            keep = TERM_EVIDENCE.is_single_word_quality_term(
                term_lower,
                known_in_pm_seed=membership["known_in_pm_seed"],
                known_in_nlab_seed=membership["known_in_nlab_seed"],
                entity_count=1,
                rhs_support_counts=dict(rhs_counts),
                nnexus_stopwords=nnexus_stopwords,
            )
        if not keep:
            continue
        records.append({
            "term_lower": term_lower,
            "source": source,
            "lhs_context": TERM_EVIDENCE.trim_context(lhs_context, max_chars=220),
            "rhs_support_counts": dict(sorted(rhs_counts.items())),
            **membership,
        })

    deduped = {}
    for row in records:
        deduped.setdefault(row["term_lower"], row)
    rows = list(deduped.values())
    rows.sort(
        key=lambda row: (
            row["novel_vs_seed"] != "novel",
            -sum(row["rhs_support_counts"].values()),
            row["term_lower"],
        )
    )
    return rows[:max_terms]


def build_paper_view(
    raw_id: str,
    batch_meta: dict[str, dict],
    results: dict[str, dict],
    args: argparse.Namespace,
    pm_lowers: set[str],
    nlab_lowers: set[str],
    nnexus_lowers: set[str],
    nnexus_stopwords: set[str],
) -> dict:
    entity_id = raw_to_entity_id(raw_id)
    meta = batch_meta[raw_id]
    entity = results["entities"][entity_id]
    scope_row = results["scopes"].get(entity_id, {"scopes": [], "count": 0})
    ner_row = results["ner_terms"].get(entity_id, {"terms": [], "count": 0})
    paper_hg = results["paper_hypergraphs"].get(entity_id, {"nodes": [], "edges": [], "sectional": {}, "meta": {}})
    reverse = results["reverse_morphogenesis"].get(entity_id, {})

    payload, suffix = extract_batch_eprint(args.batch_tar, raw_id)
    if not payload or not suffix:
        raise FileNotFoundError(f"missing batch-008 eprint payload for {raw_id}")
    eprint_text, theorem_result = load_eprint_text_and_theorems(raw_id, payload, suffix)
    local_scopes = NLAB_WIRING.detect_scopes(entity_id, eprint_text)
    local_terms = pick_local_terms(
        eprint_text,
        raw_id,
        theorem_result,
        pm_lowers,
        nlab_lowers,
        nnexus_lowers,
        nnexus_stopwords,
        args.max_local_terms,
    )

    result_text = entity.get("question-body") or ""
    result_snippet, result_offset = scope_snippet(result_text, scope_row.get("scopes") or [])
    local_snippet, local_offset = local_scope_snippet(eprint_text, local_scopes)

    return {
        "raw_id": raw_id,
        "entity_id": entity_id,
        "title": meta.get("title", ""),
        "categories": meta.get("categories") or [],
        "authors": meta.get("authors") or [],
        "result_text": result_text,
        "result_scope_count": scope_row.get("count", 0),
        "result_scopes": scope_row.get("scopes") or [],
        "result_scope_markup": render_scope_markup(result_snippet, scope_row.get("scopes") or [], offset=result_offset),
        "result_scope_types": Counter(scope.get("hx/type", "?") for scope in scope_row.get("scopes") or []).most_common(8),
        "returned_terms": (ner_row.get("terms") or [])[:18],
        "paper_hypergraph": {
            "node_count": len(paper_hg.get("nodes") or []),
            "edge_count": len(paper_hg.get("edges") or []),
            "node_types": top_node_types(paper_hg),
            "edge_types": top_edge_types(paper_hg),
            "section_count": len(paper_hg.get("sectional") or []),
            "block_count": (paper_hg.get("meta") or {}).get("n_blocks", 0),
        },
        "reverse_morphogenesis": {
            "status": reverse.get("status"),
            "slot_distinctness": (reverse.get("slot_distinctness") or {}).get("status"),
            "collapsed_pairs": (reverse.get("slot_distinctness") or {}).get("collapsed_pairs") or [],
        },
        "eprint_text_length": len(eprint_text),
        "local_scope_count": len(local_scopes),
        "local_scope_types": Counter(scope.get("hx/type", "?") for scope in local_scopes).most_common(8),
        "local_scope_markup": render_scope_markup(local_snippet, local_scopes, offset=local_offset),
        "local_theorem_stats": theorem_result.stats,
        "local_terms": local_terms,
    }


def render_html(papers: list[dict], manifest: dict, out_json: Path) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cards = []
    for paper in papers:
        returned_terms = "".join(
            f"<li><code>{html.escape(term['term_lower'])}</code> "
            f"<span class=\"canon\">{html.escape(str(term.get('canon') or ''))}</span></li>"
            for term in paper["returned_terms"]
        ) or "<li>No returned NER terms</li>"
        local_terms = "".join(
            f"<li><code>{html.escape(term['term_lower'])}</code> "
            f"<span class=\"badge {html.escape(term['novel_vs_seed'])}\">{html.escape(term['novel_vs_seed'])}</span> "
            f"<span class=\"rhs\">rhs={html.escape(json.dumps(term['rhs_support_counts'], ensure_ascii=False))}</span></li>"
            for term in paper["local_terms"]
        ) or "<li>No local filtered terms</li>"
        cards.append(
            f"""
            <section class="paper-card">
              <h2>{html.escape(paper['title'])}</h2>
              <p class="meta"><code>{html.escape(paper['raw_id'])}</code> | {html.escape(', '.join(paper['categories']))}</p>
              <div class="stats-grid">
                <div><strong>Returned scopes</strong><br>{paper['result_scope_count']}</div>
                <div><strong>Local scopes</strong><br>{paper['local_scope_count']}</div>
                <div><strong>Hypergraph</strong><br>{paper['paper_hypergraph']['node_count']} nodes / {paper['paper_hypergraph']['edge_count']} edges</div>
                <div><strong>Theorem pass</strong><br>{paper['local_theorem_stats'].get('theorems', 0)} theorems / {paper['local_theorem_stats'].get('definitions', 0)} definitions</div>
              </div>
              <div class="two-up">
                <div class="panel">
                  <h3>Returned scopes over result text</h3>
                  <p class="sub">This confirms scopes/constructions are present in <code>results-008</code> for this paper.</p>
                  <pre>{paper['result_scope_markup']}</pre>
                  <p class="tiny">Top returned scope types: {html.escape(json.dumps(paper['result_scope_types'], ensure_ascii=False))}</p>
                </div>
                <div class="panel">
                  <h3>Local scopes over raw eprint text</h3>
                  <p class="sub">Fresh pass from the raw eprint using <code>nlab-wiring.detect_scopes</code>.</p>
                  <pre>{paper['local_scope_markup']}</pre>
                  <p class="tiny">Top local scope types: {html.escape(json.dumps(paper['local_scope_types'], ensure_ascii=False))}</p>
                </div>
              </div>
              <div class="two-up">
                <div class="panel">
                  <h3>Returned paper graph summary</h3>
                  <ul>
                    <li>Sections: {paper['paper_hypergraph']['section_count']}</li>
                    <li>Normalized blocks: {paper['paper_hypergraph']['block_count']}</li>
                    <li>Node types: <code>{html.escape(json.dumps(paper['paper_hypergraph']['node_types'], ensure_ascii=False))}</code></li>
                    <li>Edge types: <code>{html.escape(json.dumps(paper['paper_hypergraph']['edge_types'], ensure_ascii=False))}</code></li>
                    <li>Reverse morphogenesis: <code>{html.escape(str(paper['reverse_morphogenesis']['status']))}</code>, slot distinctness <code>{html.escape(str(paper['reverse_morphogenesis']['slot_distinctness']))}</code></li>
                  </ul>
                </div>
                <div class="panel">
                  <h3>Local filtered term evidence</h3>
                  <p class="sub">This is the extra upgrade layer on top of Rob's batch output.</p>
                  <ul>{local_terms}</ul>
                </div>
              </div>
              <div class="two-up">
                <div class="panel">
                  <h3>Returned NER terms</h3>
                  <ul>{returned_terms}</ul>
                </div>
                <div class="panel">
                  <h3>Artifact link</h3>
                  <p class="sub">Structured companion report: <code>{html.escape(str(out_json))}</code></p>
                </div>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Batch 008 Math.CT QC Viewer</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --paper: #fffdf8;
      --ink: #1d1a16;
      --muted: #6b6258;
      --line: #d7cec2;
      --accent: #0f766e;
      --scope: #fbd38d;
      --scope2: #fee2e2;
      --novel: #7c2d12;
      --known: #14532d;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #f2eadf 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", serif;
    }}
    h1, h2, h3 {{ margin: 0 0 10px 0; }}
    .lead {{ color: var(--muted); max-width: 1000px; }}
    .paper-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 18px 12px;
      margin: 22px 0;
      box-shadow: 0 14px 30px rgba(60, 42, 18, 0.07);
    }}
    .two-up {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin: 14px 0;
    }}
    .panel {{
      background: #fffaf1;
      border: 1px solid #e8dccf;
      border-radius: 14px;
      padding: 12px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin: 14px 0;
    }}
    .stats-grid > div {{
      background: #f7f0e7;
      border: 1px solid #eadccf;
      border-radius: 12px;
      padding: 10px;
    }}
    pre {{
      white-space: pre-wrap;
      background: #fff;
      border: 1px solid #ece3d6;
      padding: 10px;
      border-radius: 10px;
      line-height: 1.45;
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    .scope {{
      background: linear-gradient(90deg, var(--scope), var(--scope2));
      padding: 0 1px;
      border-radius: 3px;
    }}
    .scope-label {{
      display: inline-block;
      margin-right: 6px;
      padding: 0 4px;
      background: rgba(29, 26, 22, 0.1);
      border-radius: 999px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .meta, .sub, .tiny {{ color: var(--muted); }}
    .tiny {{ font-size: 12px; }}
    .badge {{
      display: inline-block;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 12px;
      margin-left: 6px;
      border: 1px solid currentColor;
    }}
    .badge.novel {{ color: var(--novel); }}
    .badge.known {{ color: var(--known); }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    ul {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 900px) {{
      .two-up, .stats-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>Batch 008 Math.CT QC Viewer</h1>
  <p class="lead">
    Generated {html.escape(generated)}. This viewer joins Rob's <code>results-008.tar.gz</code>
    with raw eprints from <code>batch-008.tar.gz</code>. It is meant as a direct quality-control
    check for whether scopes/constructions are present in the returned run artifacts and how much
    extra signal the local term-evidence pass can add.
  </p>
  <p class="lead">
    Batch manifest summary: entity_count={manifest.get('entity_count')} |
    stage5 scope coverage={manifest.get('stage5_stats', {}).get('scope_coverage')} |
    hypergraphs={manifest.get('stage9a_stats', {}).get('hypergraphs_produced')}
  </p>
  {''.join(cards)}
</body>
</html>
"""


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv or sys.argv[1:])
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    batch_meta = load_batch_metadata(args.batch_tar)
    results = load_results(args.results_tar)
    pm_lowers = TERM_EVIDENCE.load_known_term_lowers(args.pm_seed)
    nlab_lowers = TERM_EVIDENCE.load_known_term_lowers(args.nlab_seed)
    nnexus_lowers = TERM_EVIDENCE.load_nnexus_concept_lowers(args.nnexus_snapshot)
    nnexus_stopwords = TERM_EVIDENCE.load_nnexus_stopwords(args.nnexus_stopwords)

    selected_raw_ids = args.paper_ids or pick_default_papers(batch_meta, results, args.paper_count)
    papers = [
        build_paper_view(
            raw_id,
            batch_meta,
            results,
            args,
            pm_lowers,
            nlab_lowers,
            nnexus_lowers,
            nnexus_stopwords,
        )
        for raw_id in selected_raw_ids
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch_tar": str(args.batch_tar),
        "results_tar": str(args.results_tar),
        "selected_papers": [
            {
                "raw_id": paper["raw_id"],
                "title": paper["title"],
                "result_scope_count": paper["result_scope_count"],
                "local_scope_count": paper["local_scope_count"],
                "paper_hypergraph": paper["paper_hypergraph"],
                "reverse_morphogenesis": paper["reverse_morphogenesis"],
                "local_theorem_stats": paper["local_theorem_stats"],
                "local_terms": paper["local_terms"],
            }
            for paper in papers
        ],
        "manifest_summary": {
            "entity_count": results["manifest"].get("entity_count"),
            "stage5_stats": results["manifest"].get("stage5_stats"),
            "stage9a_stats": results["manifest"].get("stage9a_stats"),
        },
    }
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_html.write_text(render_html(papers, results["manifest"], args.out_json), encoding="utf-8")
    print(f"Wrote {args.out_html}")
    print(f"Wrote {args.out_json}")
    return report


if __name__ == "__main__":
    main()
