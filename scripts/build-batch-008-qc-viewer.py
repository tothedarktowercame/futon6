#!/usr/bin/env python3
"""Build a more representative NNexus-Glasses-style QC viewer over batch-008.

Reads:
  - /home/joe/code/storage/mark2/inbox/batch-008.tar.gz
  - /home/joe/code/storage/mark2/outbox/results-008.tar.gz

Outputs:
  - data/showcases/batch-008-math-ct-qc.html           (index page)
  - data/showcases/batch-008-math-ct-qc.json           (summary report)
  - data/showcases/batch-008-math-ct-qc-pages/*.html   (per-paper pages)

The viewer deliberately separates:
  - returned scopes from results-008 over the result text
  - fresh local scope detection over the raw eprint
  - clustered local windows so the detector's actual density is visible
  - local filtered open-term evidence over the raw eprint
  - paper hypergraph / reverse-morphogenesis summaries
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
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
DEFAULT_NER_KERNEL = Path.home() / "code" / "storage" / "futon6" / "data" / "ner-kernel" / "terms.tsv"


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
    parser.add_argument("--out-page-dir", type=Path, default=None,
                        help="Directory for per-paper HTML pages. Default: <out-html stem>-pages/")
    parser.add_argument("--paper-id", action="append", dest="paper_ids",
                        help="Raw batch paper id like 0710.3853v1 or math/0606735v1. Repeatable.")
    parser.add_argument("--paper-count", type=int, default=4)
    parser.add_argument("--ct-count", type=int, default=2,
                        help="Default number of Category Theory papers when auto-selecting")
    parser.add_argument("--non-ct-count", type=int, default=2,
                        help="Default number of non-Category Theory papers when auto-selecting")
    parser.add_argument("--pm-seed", type=Path, default=DEFAULT_PM_SEED)
    parser.add_argument("--nlab-seed", type=Path, default=DEFAULT_NLAB_SEED)
    parser.add_argument("--nnexus-stopwords", type=Path, default=DEFAULT_NNEXUS_STOPWORDS)
    parser.add_argument("--nnexus-snapshot", type=Path, default=DEFAULT_NNEXUS_SNAPSHOT)
    parser.add_argument("--ner-kernel", type=Path, default=DEFAULT_NER_KERNEL,
                        help="Live NER kernel TSV used for inline term overlay markup.")
    parser.add_argument("--max-local-terms", type=int, default=12)
    parser.add_argument("--max-local-windows", type=int, default=6,
                        help="Max clustered local-scope windows per paper")
    parser.add_argument("--window-chars", type=int, default=1800,
                        help="Approximate local window size in characters")
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


def paper_score(raw_id: str, meta: dict, results: dict[str, dict]) -> dict:
    entity_id = raw_to_entity_id(raw_id)
    scope_row = results["scopes"].get(entity_id, {})
    hg_row = results["paper_hypergraphs"].get(entity_id, {})
    rev_row = results["reverse_morphogenesis"].get(entity_id, {})
    scopes = scope_row.get("count", 0)
    terms = results["ner_terms"].get(entity_id, {}).get("count", 0)
    edges = len(hg_row.get("edges") or [])
    nodes = len(hg_row.get("nodes") or [])
    score = (10 if scopes > 0 else 0) + min(scopes, 25) + min(edges, 120) + min(terms, 40)
    return {
        "raw_id": raw_id,
        "entity_id": entity_id,
        "title": meta.get("title", ""),
        "categories": meta.get("categories") or [],
        "scope_count": scopes,
        "term_count": terms,
        "nodes": nodes,
        "edges": edges,
        "reverse_status": rev_row.get("status"),
        "score": score,
    }


def pick_default_papers(batch_meta: dict[str, dict], results: dict[str, dict], paper_count: int) -> list[str]:
    rows = []
    for raw_id, meta in batch_meta.items():
        if "math.CT" not in (meta.get("categories") or []):
            continue
        rows.append(paper_score(raw_id, meta, results))
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


def pick_representative_papers(
    batch_meta: dict[str, dict],
    results: dict[str, dict],
    *,
    ct_count: int,
    non_ct_count: int,
    available_eprint_ids: set[str] | None = None,
) -> list[str]:
    rows = [
        paper_score(raw_id, meta, results)
        for raw_id, meta in batch_meta.items()
        if available_eprint_ids is None or raw_id in available_eprint_ids
    ]
    rows.sort(key=lambda row: (row["score"], row["scope_count"], row["edges"], row["term_count"]), reverse=True)

    def pick(predicate, count):
        chosen = [row["raw_id"] for row in rows if predicate(row) and row["scope_count"] > 0][:count]
        if len(chosen) < count:
            for row in rows:
                if not predicate(row) or row["raw_id"] in chosen:
                    continue
                chosen.append(row["raw_id"])
                if len(chosen) >= count:
                    break
        return chosen

    ct_rows = pick(lambda row: "math.CT" in row["categories"], ct_count)
    non_ct_rows = pick(lambda row: "math.CT" not in row["categories"], non_ct_count)
    return ct_rows + [raw_id for raw_id in non_ct_rows if raw_id not in ct_rows]


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


def load_available_eprint_ids(batch_tar: Path) -> set[str]:
    out = set()
    with tarfile.open(batch_tar, "r:gz") as tf:
        for member in tf.getmembers():
            name = member.name
            if not name.startswith("batch-008/eprints/"):
                continue
            tail = name[len("batch-008/eprints/"):]
            for suffix in (".tar.gz", ".bin"):
                if tail.endswith(suffix):
                    out.add(tail[: -len(suffix)])
    return out


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


def scope_span(scope: dict) -> tuple[int, int] | None:
    content = scope.get("hx/content", {})
    start = content.get("position")
    end = content.get("end")
    if not isinstance(start, int):
        return None
    if not isinstance(end, int):
        match = content.get("match", "")
        end = start + len(match)
    if end < start:
        end = start
    return start, end


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def scope_coverage_stats(text: str, scopes: list[dict]) -> dict:
    spans = [span for scope in scopes if (span := scope_span(scope))]
    merged = merge_spans(spans)
    total_chars = len(text)
    covered_chars = sum(max(0, end - start) for start, end in merged)
    char_ratio = (covered_chars / total_chars) if total_chars else 0.0

    sentence_spans = []
    for match in re.finditer(r"[^.!?\n][^.!?\n]*(?:[.!?](?=\s|$)|$)", text):
        start, end = match.span()
        if end > start and text[start:end].strip():
            sentence_spans.append((start, end))
    covered_sentences = 0
    for start, end in sentence_spans:
        if any(not (m_end <= start or m_start >= end) for m_start, m_end in merged):
            covered_sentences += 1
    sentence_ratio = (covered_sentences / len(sentence_spans)) if sentence_spans else 0.0

    return {
        "scope_count": len(scopes),
        "merged_span_count": len(merged),
        "covered_chars": covered_chars,
        "total_chars": total_chars,
        "char_coverage": round(char_ratio, 4),
        "sentences": len(sentence_spans),
        "sentences_touched": covered_sentences,
        "sentence_coverage": round(sentence_ratio, 4),
    }


def scope_density_bins(text: str, scopes: list[dict], *, bins: int = 40) -> list[dict]:
    total = max(1, len(text))
    width = max(1, total // bins)
    rows = []
    for idx in range(bins):
        start = idx * width
        end = total if idx == bins - 1 else min(total, (idx + 1) * width)
        count = 0
        for scope in scopes:
            span = scope_span(scope)
            if span is None:
                continue
            s0, s1 = span
            if not (s1 <= start or s0 >= end):
                count += 1
        rows.append({"start": start, "end": end, "count": count})
    return rows


def pick_scope_windows(text: str, scopes: list[dict], *, max_windows: int, window_chars: int) -> list[dict]:
    if not text:
        return []
    bins = scope_density_bins(text, scopes, bins=max(12, min(60, len(text) // max(1, window_chars // 2))))
    candidates = []
    for row in bins:
        if row["count"] <= 0:
            continue
        start = max(0, row["start"] - window_chars // 4)
        end = min(len(text), start + window_chars)
        local_scopes = []
        for scope in scopes:
            span = scope_span(scope)
            if span is None:
                continue
            s0, s1 = span
            if not (s1 <= start or s0 >= end):
                local_scopes.append(scope)
        if not local_scopes:
            continue
        snippet = text[start:end]
        coverage = scope_coverage_stats(snippet, [
            {
                **scope,
                "hx/content": {
                    **(scope.get("hx/content", {})),
                    "position": max(0, (scope_span(scope) or (0, 0))[0] - start),
                    "end": max(0, (scope_span(scope) or (0, 0))[1] - start),
                },
            }
            for scope in local_scopes
            if scope_span(scope)
        ])
        candidates.append({
            "start": start,
            "end": end,
            "scope_count": len(local_scopes),
            "coverage": coverage,
            "scopes": local_scopes,
            "text": snippet,
        })
    candidates.sort(
        key=lambda row: (
            row["scope_count"],
            row["coverage"]["char_coverage"],
            row["coverage"]["sentence_coverage"],
            -row["start"],
        ),
        reverse=True,
    )
    chosen = []
    for row in candidates:
        if any(not (row["end"] <= prev["start"] or row["start"] >= prev["end"]) for prev in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= max_windows:
            break
    chosen.sort(key=lambda row: row["start"])
    return chosen


def find_kernel_term_positions(text: str, singles: dict, multi_index: dict) -> list[tuple[int, int, str, str | None]]:
    """For every term the NER kernel spots in `text`, locate all occurrences.

    Returns (start, end, term_lower, canon) tuples relative to `text`. Overlapping
    occurrences are resolved by preferring the longest match starting earliest,
    which lets multi-word terms like "monoidal category" win over their
    constituent words.
    """
    hits = SUPERPOD_JOB.spot_terms_entity(text, singles, multi_index)
    positions: list[tuple[int, int, str, str | None]] = []
    for hit in hits:
        term_lower = hit.get("term_lower") or ""
        canon = hit.get("canon")
        if not term_lower:
            continue
        try:
            pattern = re.compile(rf"\b{re.escape(term_lower)}\b", re.IGNORECASE)
        except re.error:
            continue
        for m in pattern.finditer(text):
            positions.append((m.start(), m.end(), term_lower, canon))
    # Longest-first at each start, then non-overlapping greedy.
    positions.sort(key=lambda row: (row[0], -(row[1] - row[0])))
    deduped: list[tuple[int, int, str, str | None]] = []
    last_end = -1
    for start, end, tl, canon in positions:
        if start >= last_end:
            deduped.append((start, end, tl, canon))
            last_end = end
    return deduped


def classify_kernel_terms(text: str, scopes: list[dict], singles: dict, multi_index: dict) -> dict:
    """Count kernel-term occurrences by relation to scope coverage.

    Returns {inhabited, outer, straddled, total, depth_distribution}:
    - inhabited: term fully inside some scope (any scope; the frontier
      semantics doesn't care whether other scopes straddle)
    - outer: term completely disjoint from all scopes (scope-development
      candidate — what the structure-learning loop should target)
    - straddled: term overlaps some scope boundary AND isn't contained in
      any scope (ambiguous edges)
    - depth_distribution: map of {depth -> count} for inhabited terms,
      where depth=1 is "inside a top-level scope," depth=2 is "inside a
      scope inside a scope," etc. Computed via the tree renderer's view of
      the same data; terms placed inside scopes the tree had to drop due
      to mutual straddling don't contribute to this map but are still
      counted as inhabited above.
    """
    term_positions = find_kernel_term_positions(text, singles, multi_index)
    span_records: list[dict] = []
    for scope in scopes:
        content = scope.get("hx/content", {}) or {}
        start = content.get("position")
        end = content.get("end")
        if not isinstance(start, int):
            continue
        if not isinstance(end, int):
            end = start + len(content.get("match", ""))
        span_records.append({
            "start": start,
            "end": end,
            "label": scope.get("hx/type", "?"),
        })

    inhabited = outer = straddled = 0
    inhabited_terms: list[tuple[int, int, str, str | None]] = []
    for term in term_positions:
        ts, te = term[0], term[1]
        contained = False
        overlaps_any = False
        for sp in span_records:
            ss, se = sp["start"], sp["end"]
            if ss <= ts and te <= se:
                contained = True
            elif not (se <= ts or ss >= te):
                overlaps_any = True
        if contained:
            inhabited += 1
            inhabited_terms.append(term)
        elif overlaps_any:
            straddled += 1
        else:
            outer += 1

    tree = build_scope_tree(span_records, inhabited_terms)
    depth_dist: dict[int, int] = {}

    def walk(node: dict) -> None:
        for child in node["children"]:
            term_count = len(child["terms"])
            if term_count:
                depth_dist[child["depth"]] = depth_dist.get(child["depth"], 0) + term_count
            walk(child)

    walk(tree)
    return {
        "inhabited": inhabited,
        "outer": outer,
        "straddled": straddled,
        "total": len(term_positions),
        "depth_distribution": dict(sorted(depth_dist.items())),
    }


def _term_span_html(text: str, start: int, end: int, canon: str | None, *, inhabited: bool) -> str:
    title = f' title="canon={html.escape(canon)}"' if canon else ""
    klass = "term-kernel inhabited" if inhabited else "term-kernel"
    return f'<span class="{klass}"{title}>{html.escape(text[start:end])}</span>'


def build_scope_tree(
    scope_spans: list[dict],
    term_positions: list[tuple[int, int, str, str | None]],
) -> dict:
    """Arrange flat scope spans into a containment tree, with terms at deepest scope.

    Each scope is either fully contained in another (becomes a child) or sits as
    a sibling at the top level. Scopes that straddle another scope's boundary
    (partial overlap, neither containing nor contained) are dropped — they
    can't be tree-arranged and would produce broken nesting. Each term is
    placed in the deepest scope that fully contains it; terms not contained
    by any scope sit at the root; terms straddling a scope boundary are
    dropped.

    The root node uses sentinel start=-1 and end=10**18 so any real span fits
    inside it. Its `label` is "$root" and is never rendered; only its children
    and terms.
    """
    root: dict = {
        "start": -1,
        "end": 10**18,
        "label": "$root",
        "children": [],
        "terms": [],
        "depth": 0,
    }
    # Outer-first ordering: earlier start wins; on tie, larger span wins.
    sorted_scopes = sorted(scope_spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    stack: list[dict] = [root]
    for sp in sorted_scopes:
        # Pop scopes that ended before this one starts.
        while stack[-1] is not root and stack[-1]["end"] <= sp["start"]:
            stack.pop()
        top = stack[-1]
        # If the new scope fully fits inside the current top, nest it.
        if top["start"] <= sp["start"] and sp["end"] <= top["end"]:
            node = {
                "start": sp["start"],
                "end": sp["end"],
                "label": sp["label"],
                "children": [],
                "terms": [],
                "depth": top["depth"] + 1,
            }
            top["children"].append(node)
            stack.append(node)
        # else: scope straddles the top span — drop it.

    def find_deepest(node: dict, ts: int, te: int) -> dict | None:
        if not (node["start"] <= ts and te <= node["end"]):
            return None
        for child in node["children"]:
            deeper = find_deepest(child, ts, te)
            if deeper is not None:
                return deeper
        return node

    def straddles_any_input_scope(ts: int, te: int) -> bool:
        # A term that overlaps a scope without being fully contained in it is
        # ambiguous — drop it. Check against the original input scope set so a
        # term doesn't sneak into the root just because the straddled scope was
        # dropped from the tree.
        for sp in scope_spans:
            ss, se = sp["start"], sp["end"]
            if se <= ts or ss >= te:
                continue  # disjoint
            if ss <= ts and te <= se:
                continue  # fully contained
            return True
        return False

    for (ts, te, tl, canon) in term_positions:
        if straddles_any_input_scope(ts, te):
            continue  # straddled — drop
        deepest = find_deepest(root, ts, te)
        if deepest is not None:
            deepest["terms"].append((ts, te, tl, canon))
    return root


def render_tree_node(text: str, node: dict, *, is_root: bool) -> str:
    """Recursively emit nested mark elements with deepest-scope term placement.

    Terms at the root render with the .term-kernel class (outer); terms at any
    non-root scope render with .term-kernel.inhabited (purple). Nested scopes
    produce nested <mark> elements.
    """
    events: list[tuple[int, int, str, object]] = []
    for child in node["children"]:
        events.append((child["start"], child["end"], "scope", child))
    for (ts, te, tl, canon) in node["terms"]:
        events.append((ts, te, "term", (tl, canon)))
    events.sort(key=lambda e: (e[0], e[1]))

    cursor = 0 if is_root else node["start"]
    end_pos = len(text) if is_root else node["end"]
    out: list[str] = []
    for start, end, kind, payload in events:
        if start < cursor:
            continue
        clipped_start = max(cursor, min(len(text), start))
        clipped_end = max(clipped_start, min(len(text), end))
        out.append(html.escape(text[cursor:clipped_start]))
        if kind == "scope":
            out.append(render_tree_node(text, payload, is_root=False))
        else:
            _tl, canon = payload
            out.append(_term_span_html(text, clipped_start, clipped_end, canon, inhabited=not is_root))
        cursor = clipped_end
    out.append(html.escape(text[cursor:min(len(text), end_pos)]))

    if is_root:
        return "".join(out)
    label_html = f'<span class="scope-label">{html.escape(node["label"])}</span>'
    return f'<mark class="scope">{label_html}{"".join(out)}</mark>'


def render_overlay_markup(
    text: str,
    scopes: list[dict],
    singles: dict,
    multi_index: dict,
    offset: int = 0,
    scope_limit: int | None = None,
) -> str:
    """Tree-aware scope + kernel-term overlay.

    Scopes nest as a containment tree: an inner scope renders inside its
    parent scope's mark element. Each kernel term gets placed in its deepest
    containing scope (or at the root if no scope contains it). Terms in any
    non-root scope render as inhabited (purple); terms at the root render as
    outer (teal). Straddling scopes and straddling terms are dropped.

    scope_limit truncates the input span list before tree-building; default
    None renders all scopes (the snippet itself bounds total volume).
    """
    all_scope_spans: list[dict] = []
    for scope in scopes:
        content = scope.get("hx/content", {}) or {}
        start = content.get("position")
        end = content.get("end")
        if not isinstance(start, int):
            continue
        if not isinstance(end, int):
            end = start + len(content.get("match", ""))
        all_scope_spans.append({
            "start": max(0, start - offset),
            "end": max(0, end - offset),
            "label": scope.get("hx/type", "?"),
        })
    all_scope_spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    if scope_limit is not None:
        all_scope_spans = all_scope_spans[:scope_limit]

    term_positions = find_kernel_term_positions(text, singles, multi_index)
    tree = build_scope_tree(all_scope_spans, term_positions)
    return render_tree_node(text, tree, is_root=True)


def render_scope_markup(text: str, scopes: list[dict], offset: int = 0, limit: int | None = 8) -> str:
    spans = []
    sorted_scopes = sorted(
        scopes,
        key=lambda scope: (
            (scope.get("hx/content", {}) or {}).get("position", 10**18),
            (scope.get("hx/content", {}) or {}).get("end", 10**18),
        ),
    )
    if limit is not None:
        sorted_scopes = sorted_scopes[:limit]
    for scope in sorted_scopes:
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
    singles: dict,
    multi_index: dict,
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
    local_coverage = scope_coverage_stats(eprint_text, local_scopes)
    local_bins = scope_density_bins(eprint_text, local_scopes)
    local_windows = pick_scope_windows(
        eprint_text,
        local_scopes,
        max_windows=args.max_local_windows,
        window_chars=args.window_chars,
    )
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

    rendered_windows = []
    for idx, window in enumerate(local_windows, start=1):
        rendered_windows.append({
            "ordinal": idx,
            "start": window["start"],
            "end": window["end"],
            "scope_count": window["scope_count"],
            "coverage": window["coverage"],
            "scope_types": Counter(scope.get("hx/type", "?") for scope in window["scopes"]).most_common(6),
            "markup": render_overlay_markup(
                window["text"], window["scopes"], singles, multi_index,
                offset=window["start"], scope_limit=None,
            ),
        })

    # Whole-paper term classification. The `outer` count is the
    # scope-development frontier: each term in residual prose is a hit
    # the kernel already knows but the structure detector has not yet
    # wrapped in a scope. That count should fall as structure-learning
    # patterns land.
    term_stats = classify_kernel_terms(eprint_text, local_scopes, singles, multi_index)
    local_term_overlay_count = term_stats["total"]

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
        "local_scope_coverage": local_coverage,
        "local_scope_types": Counter(scope.get("hx/type", "?") for scope in local_scopes).most_common(8),
        "local_scope_bins": local_bins,
        "local_scope_markup": render_overlay_markup(
            local_snippet, local_scopes, singles, multi_index, offset=local_offset,
        ),
        "local_scope_windows": rendered_windows,
        "local_theorem_stats": theorem_result.stats,
        "local_terms": local_terms,
        "local_term_overlay_count": local_term_overlay_count,
        "local_term_inhabited_count": term_stats["inhabited"],
        "local_term_outer_count": term_stats["outer"],
        "local_term_straddled_count": term_stats["straddled"],
        "local_term_depth_distribution": term_stats["depth_distribution"],
    }


def coverage_badge(ratio: float) -> tuple[str, str]:
    if ratio >= 0.75:
        return "strong", "Strong"
    if ratio >= 0.4:
        return "mixed", "Mixed"
    return "sparse", "Sparse"


def render_density_bar(bins: list[dict]) -> str:
    max_count = max((row["count"] for row in bins), default=1)
    bars = []
    for row in bins:
        level = row["count"] / max_count if max_count else 0.0
        bars.append(
            f'<span class="density-bin" style="opacity:{0.15 + 0.85 * level:.3f}" '
            f'title="count={row["count"]}; chars {row["start"]}-{row["end"]}"></span>'
        )
    return "".join(bars)


def paper_filename(raw_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw_id)
    return f"{safe}.html"


def render_paper_page(paper: dict, *, report_path: Path, back_href: str) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    local_cov = paper["local_scope_coverage"]
    cov_class, cov_label = coverage_badge(local_cov["char_coverage"])
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
    windows = "".join(
        f"""
        <section class="window-card">
          <h3>Local scope window {window['ordinal']}</h3>
          <p class="tiny">
            chars {window['start']}-{window['end']} |
            scopes {window['scope_count']} |
            char coverage {window['coverage']['char_coverage']:.1%} |
            sentence coverage {window['coverage']['sentence_coverage']:.1%}
          </p>
          <pre>{window['markup']}</pre>
          <p class="tiny">Top window scope types: {html.escape(json.dumps(window['scope_types'], ensure_ascii=False))}</p>
        </section>
        """
        for window in paper["local_scope_windows"]
    ) or "<p class='tiny'>No clustered local windows.</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(paper['title'])} — Batch 008 QC</title>
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
      --strong: #166534;
      --mixed: #9a3412;
      --sparse: #991b1b;
    }}
    body {{ margin: 0; padding: 24px; background: linear-gradient(180deg, #f2eadf 0%, var(--bg) 100%); color: var(--ink); font-family: Georgia, "Iowan Old Style", serif; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; }}
    .topnav {{ margin-bottom: 18px; font-family: system-ui, sans-serif; font-size: 14px; }}
    .topnav a {{ color: #0f766e; text-decoration: none; }}
    .hero, .panel, .window-card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 16px; padding: 16px; box-shadow: 0 12px 28px rgba(60, 42, 18, 0.06); }}
    .hero {{ margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }}
    .triple {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px; }}
    .stat {{ background: #f7f0e7; border: 1px solid #eadccf; border-radius: 12px; padding: 10px; }}
    .stat .k {{ font: 700 0.72rem/1 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .05em; color: #6b6258; }}
    .stat .v {{ margin-top: 4px; font: 700 1.2rem/1.15 system-ui, sans-serif; }}
    .density {{ display: grid; grid-template-columns: repeat(40, 1fr); gap: 2px; margin-top: 10px; }}
    .density-bin {{ display: block; height: 16px; background: #0f766e; border-radius: 3px; }}
    pre {{ white-space: pre-wrap; background: #fff; border: 1px solid #ece3d6; padding: 10px; border-radius: 10px; line-height: 1.45; font-size: 14px; overflow-wrap: anywhere; }}
    .scope {{ background: linear-gradient(90deg, var(--scope), var(--scope2)); padding: 0 1px; border-radius: 3px; }}
    .scope-label {{ display: inline-block; margin-right: 6px; padding: 0 4px; background: rgba(29, 26, 22, 0.1); border-radius: 999px; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
    .term-kernel {{ background: rgba(15, 118, 110, 0.12); border-bottom: 1px solid rgba(15, 118, 110, 0.45); padding: 0 1px; border-radius: 2px; cursor: help; }}
    .term-kernel:hover {{ background: rgba(15, 118, 110, 0.22); }}
    .term-kernel.inhabited {{ background: rgba(124, 58, 237, 0.18); border-bottom: 1px solid rgba(124, 58, 237, 0.6); }}
    .term-kernel.inhabited:hover {{ background: rgba(124, 58, 237, 0.30); }}
    .markup-legend {{ font: 12px/1.4 system-ui, sans-serif; color: var(--muted); margin: 8px 0 6px 0; }}
    .markup-legend .swatch {{ display: inline-block; padding: 0 6px; border-radius: 3px; margin-right: 4px; }}
    .markup-legend .swatch.scope {{ background: linear-gradient(90deg, var(--scope), var(--scope2)); }}
    .markup-legend .swatch.term {{ background: rgba(15, 118, 110, 0.18); border-bottom: 1px solid rgba(15, 118, 110, 0.55); }}
    .markup-legend .swatch.term-inhabited {{ background: rgba(124, 58, 237, 0.20); border-bottom: 1px solid rgba(124, 58, 237, 0.6); }}
    .badge {{ display: inline-block; padding: 1px 6px; border-radius: 999px; font-size: 12px; margin-left: 6px; border: 1px solid currentColor; }}
    .badge.novel {{ color: var(--novel); }}
    .badge.known {{ color: var(--known); }}
    .coverage-badge {{ display: inline-block; padding: 4px 8px; border-radius: 999px; border: 1px solid currentColor; font: 700 12px/1 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .05em; }}
    .coverage-badge.strong {{ color: var(--strong); }}
    .coverage-badge.mixed {{ color: var(--mixed); }}
    .coverage-badge.sparse {{ color: var(--sparse); }}
    .tiny, .sub, .meta {{ color: var(--muted); }}
    .tiny {{ font-size: 12px; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    ul {{ margin: 0; padding-left: 18px; }}
    h1, h2, h3 {{ margin: 0 0 10px 0; }}
    @media (max-width: 980px) {{ .grid, .triple {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topnav"><a href="{html.escape(back_href)}">Back to index</a></div>
    <section class="hero">
      <h1>{html.escape(paper['title'])}</h1>
      <p class="meta"><code>{html.escape(paper['raw_id'])}</code> | {html.escape(', '.join(paper['categories']))}</p>
      <p class="sub">Generated {html.escape(generated)}. Local scope detector: <code>nlab-wiring.detect_scopes</code>, which is a general mathematical construction detector rather than a Category-Theory-only grammar.</p>
      <div class="triple">
        <div class="stat"><div class="k">Returned scopes</div><div class="v">{paper['result_scope_count']}</div></div>
        <div class="stat"><div class="k">Local scopes</div><div class="v">{paper['local_scope_count']}</div></div>
        <div class="stat"><div class="k">Hypergraph</div><div class="v">{paper['paper_hypergraph']['node_count']} / {paper['paper_hypergraph']['edge_count']}</div></div>
      </div>
      <div class="triple">
        <div class="stat"><div class="k">Char coverage</div><div class="v">{paper['local_scope_coverage']['char_coverage']:.1%}</div></div>
        <div class="stat"><div class="k">Sentence coverage</div><div class="v">{paper['local_scope_coverage']['sentence_coverage']:.1%}</div></div>
        <div class="stat"><div class="k">Coverage verdict</div><div class="v"><span class="coverage-badge {cov_class}">{cov_label}</span></div></div>
      </div>
      <div class="density">{render_density_bar(paper['local_scope_bins'])}</div>
      <p class="tiny">This heat strip shows where scope detections cluster across the full eprint. A sparse strip means the detector is construction-oriented rather than sentence-complete.</p>
    </section>
    <div class="grid">
      <section class="panel">
        <h2>Returned scopes over result text</h2>
        <p class="sub">This is what Rob's returned run already contained for the abstract/result surface.</p>
        <pre>{paper['result_scope_markup']}</pre>
        <p class="tiny">Top returned scope types: {html.escape(json.dumps(paper['result_scope_types'], ensure_ascii=False))}</p>
      </section>
      <section class="panel">
        <h2>Old local viewer snippet</h2>
        <p class="sub">Retained here only to show why the previous mockup was misleading: it used a near-full-paper snippet and highlighted too few scopes. The viewer now also overlays NER-kernel term hits inline.</p>
        <p class="markup-legend">
          <span class="swatch scope">scope</span> structural construction
          &nbsp;·&nbsp;
          <span class="swatch term">term</span> kernel term in residual prose (= <strong>scope-development candidate</strong>)
          &nbsp;·&nbsp;
          <span class="swatch term-inhabited">term</span> kernel term <em>inhabiting</em> a scope
        </p>
        <p class="markup-legend">
          Full-eprint term counts:
          <code>{paper.get('local_term_inhabited_count', 0)}</code> inhabited
          / <code>{paper.get('local_term_outer_count', 0)}</code> outer (scope-development frontier)
          / <code>{paper.get('local_term_straddled_count', 0)}</code> straddled
          = <code>{paper.get('local_term_overlay_count', 0)}</code> total kernel hits.
          Outer count should fall as learned scope patterns land.
        </p>
        <p class="markup-legend">
          Inhabited-term depth distribution (tree-aware):
          {' '.join(f'depth&nbsp;{d}:&nbsp;<code>{c}</code>' for d, c in (paper.get('local_term_depth_distribution') or {}).items()) or '<em>no nested terms</em>'}.
          Higher depths mean richer structural nesting around the term.
        </p>
        <pre>{paper['local_scope_markup']}</pre>
        <p class="tiny">Top local scope types: {html.escape(json.dumps(paper['local_scope_types'], ensure_ascii=False))}</p>
      </section>
    </div>
    <section class="panel">
      <h2>Representative local scope windows</h2>
      <p class="sub">These windows are chosen from dense local clusters in the raw eprint, so the markup now reflects what the detector actually found instead of showing a mostly blank giant excerpt. Scopes and NER-kernel term hits are layered together; scopes take precedence on overlap.</p>
      {windows}
    </section>
    <div class="grid">
      <section class="panel">
        <h2>Returned paper graph summary</h2>
        <ul>
          <li>Sections: {paper['paper_hypergraph']['section_count']}</li>
          <li>Normalized blocks: {paper['paper_hypergraph']['block_count']}</li>
          <li>Node types: <code>{html.escape(json.dumps(paper['paper_hypergraph']['node_types'], ensure_ascii=False))}</code></li>
          <li>Edge types: <code>{html.escape(json.dumps(paper['paper_hypergraph']['edge_types'], ensure_ascii=False))}</code></li>
          <li>Reverse morphogenesis: <code>{html.escape(str(paper['reverse_morphogenesis']['status']))}</code>, slot distinctness <code>{html.escape(str(paper['reverse_morphogenesis']['slot_distinctness']))}</code></li>
        </ul>
      </section>
      <section class="panel">
        <h2>Local filtered term evidence</h2>
        <ul>{local_terms}</ul>
      </section>
    </div>
    <div class="grid">
      <section class="panel">
        <h2>Returned NER terms</h2>
        <ul>{returned_terms}</ul>
      </section>
      <section class="panel">
        <h2>Artifact link</h2>
        <p class="sub">Structured companion report: <code>{html.escape(str(report_path))}</code></p>
      </section>
    </div>
  </div>
</body>
</html>"""


def render_index_html(papers: list[dict], manifest: dict, out_json: Path, page_dir: Path, index_path: Path) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cards = []
    for paper in papers:
        cov_class, cov_label = coverage_badge(paper["local_scope_coverage"]["char_coverage"])
        page_href = os.path.relpath(Path(paper["page_path"]), start=index_path.parent)
        cards.append(
            f"""
            <section class="paper-card">
              <h2><a href="{html.escape(page_href)}">{html.escape(paper['title'])}</a></h2>
              <p class="meta"><code>{html.escape(paper['raw_id'])}</code> | {html.escape(', '.join(paper['categories']))}</p>
              <div class="stats-grid">
                <div><strong>Returned scopes</strong><br>{paper['result_scope_count']}</div>
                <div><strong>Local scopes</strong><br>{paper['local_scope_count']}</div>
                <div><strong>Char coverage</strong><br>{paper['local_scope_coverage']['char_coverage']:.1%}</div>
                <div><strong>Sentence coverage</strong><br>{paper['local_scope_coverage']['sentence_coverage']:.1%}</div>
                <div><strong>Coverage</strong><br><span class="coverage-badge {cov_class}">{cov_label}</span></div>
                <div><strong>Hypergraph</strong><br>{paper['paper_hypergraph']['node_count']} nodes / {paper['paper_hypergraph']['edge_count']} edges</div>
              </div>
              <div class="density">{render_density_bar(paper['local_scope_bins'])}</div>
              <p class="tiny">Top local scope types: {html.escape(json.dumps(paper['local_scope_types'], ensure_ascii=False))}</p>
              <p class="tiny">Kernel terms:
                <code>{paper.get('local_term_inhabited_count', 0)}</code> inhabited
                / <code>{paper.get('local_term_outer_count', 0)}</code> outer
                <em>(scope-development frontier)</em>
                / <code>{paper.get('local_term_overlay_count', 0)}</code> total.
              </p>
              <p class="tiny"><a href="{html.escape(page_href)}">Open full paper demo</a></p>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Batch 008 Stage 5 QC Index</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --paper: #fffdf8;
      --ink: #1d1a16;
      --muted: #6b6258;
      --line: #d7cec2;
      --accent: #0f766e;
      --strong: #166534;
      --mixed: #9a3412;
      --sparse: #991b1b;
    }}
    body {{ margin: 0; padding: 24px; background: linear-gradient(180deg, #f2eadf 0%, var(--bg) 100%); color: var(--ink); font-family: Georgia, "Iowan Old Style", serif; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    .lead, .tiny, .meta {{ color: var(--muted); }}
    .paper-card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 18px; padding: 18px; margin: 22px 0; box-shadow: 0 14px 30px rgba(60, 42, 18, 0.07); }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 14px 0; }}
    .stats-grid > div {{ background: #f7f0e7; border: 1px solid #eadccf; border-radius: 12px; padding: 10px; }}
    .density {{ display: grid; grid-template-columns: repeat(40, 1fr); gap: 2px; margin-top: 10px; }}
    .density-bin {{ display: block; height: 16px; background: #0f766e; border-radius: 3px; }}
    .coverage-badge {{ display: inline-block; padding: 4px 8px; border-radius: 999px; border: 1px solid currentColor; font: 700 12px/1 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .05em; }}
    .coverage-badge.strong {{ color: var(--strong); }}
    .coverage-badge.mixed {{ color: var(--mixed); }}
    .coverage-badge.sparse {{ color: var(--sparse); }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    a {{ color: var(--accent); text-decoration: none; }}
    @media (max-width: 980px) {{ .stats-grid {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Batch 008 Stage 5 QC Index</h1>
    <p class="lead">
      Generated {html.escape(generated)}. This revised demo is intentionally stricter than the earlier mockup:
      it shows 2 Category Theory papers and 2 non-CT papers, splits them into separate pages, and reports
      explicit full-paper coverage metrics instead of implying that a large scope count means near-complete annotation.
    </p>
    <p class="lead">
      Local detector: <code>nlab-wiring.detect_scopes</code>. This is a general mathematical construction detector,
      not a Category-Theory-only scope grammar. If coverage is sparse here, that is a real capability limit, not a viewer artifact.
    </p>
    <p class="lead">
      Batch manifest summary: entity_count={manifest.get('entity_count')} |
      stage5 scope coverage={manifest.get('stage5_stats', {}).get('scope_coverage')} |
      hypergraphs={manifest.get('stage9a_stats', {}).get('hypergraphs_produced')} |
      report={html.escape(str(out_json))}
    </p>
    {''.join(cards)}
  </div>
</body>
</html>"""


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv or sys.argv[1:])
    if args.out_page_dir is None:
        args.out_page_dir = args.out_html.parent / f"{args.out_html.stem}-pages"
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_page_dir.mkdir(parents=True, exist_ok=True)

    batch_meta = load_batch_metadata(args.batch_tar)
    available_eprint_ids = load_available_eprint_ids(args.batch_tar)
    results = load_results(args.results_tar)
    pm_lowers = TERM_EVIDENCE.load_known_term_lowers(args.pm_seed)
    nlab_lowers = TERM_EVIDENCE.load_known_term_lowers(args.nlab_seed)
    nnexus_lowers = TERM_EVIDENCE.load_nnexus_concept_lowers(args.nnexus_snapshot)
    nnexus_stopwords = TERM_EVIDENCE.load_nnexus_stopwords(args.nnexus_stopwords)
    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)

    selected_raw_ids = args.paper_ids or pick_representative_papers(
        batch_meta,
        results,
        ct_count=args.ct_count,
        non_ct_count=args.non_ct_count,
        available_eprint_ids=available_eprint_ids,
    )[:args.paper_count]
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
            singles,
            multi_index,
        )
        for raw_id in selected_raw_ids
    ]

    for paper in papers:
        page_path = args.out_page_dir / paper_filename(paper["raw_id"])
        page_html = render_paper_page(
            paper,
            report_path=args.out_json,
            back_href=os.path.relpath(args.out_html, start=args.out_page_dir),
        )
        page_path.write_text(page_html, encoding="utf-8")
        paper["page_path"] = str(page_path)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch_tar": str(args.batch_tar),
        "results_tar": str(args.results_tar),
        "index_html": str(args.out_html),
        "page_dir": str(args.out_page_dir),
        "selected_papers": [
            {
                "raw_id": paper["raw_id"],
                "title": paper["title"],
                "categories": paper["categories"],
                "page_path": paper["page_path"],
                "result_scope_count": paper["result_scope_count"],
                "local_scope_count": paper["local_scope_count"],
                "local_scope_coverage": paper["local_scope_coverage"],
                "local_scope_types": paper["local_scope_types"],
                "window_count": len(paper["local_scope_windows"]),
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
    args.out_html.write_text(
        render_index_html(papers, results["manifest"], args.out_json, args.out_page_dir, args.out_html),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_html}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_page_dir}")
    return report


if __name__ == "__main__":
    main()
