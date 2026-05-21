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
from futon6 import structure_seed as _ss
from futon6 import math_ast as _ma
from futon6 import symbol_grounding as _sg


def _make_kernel_phrase_lookup(singles: dict, multi_index: dict):
    """Build a phrase→canon lookup for the symbol-grounding strategies.

    `singles` maps single-word term_lower → (term_orig, canon).
    `multi_index` maps first_word → list of (term_lower, term_orig, canon).
    Returns a function that, given a phrase like "abelian group", returns
    the kernel's canon name, or None if the phrase isn't known.
    """
    def lookup(phrase: str) -> str | None:
        phrase = (phrase or "").lower().strip()
        if not phrase:
            return None
        if phrase in singles:
            return singles[phrase][1]
        first_word = phrase.split()[0] if phrase else ""
        if first_word in multi_index:
            for term_lower, _orig, canon in multi_index[first_word]:
                if term_lower == phrase:
                    return canon
        return None
    return lookup


def _math_atoms_for_grounding(text: str):
    """Yield (atom_text, abs_start, abs_end) for each atom we'd like to
    look up in the SymbolEnvironment.

    Atoms are: (a) each single letter inside chars nodes within math
    envelopes, and (b) full macro-token texts like `\\mathcal{C}`. Letter
    atoms are emitted one at a time so juxtapositions like `XY` become
    candidates `X` and `Y` separately. Macros are emitted as a whole so
    `\\mathcal{C}` matches a Let-binding that captured the same literal
    string.
    """
    for env_start, env_end, int_start, int_end, _kind in _ma.find_math_envelopes(text):
        interior = text[int_start:int_end]
        nodes = _ma.parse_math(interior, base_offset=int_start)
        yield from _walk_atoms(nodes)


def _walk_atoms(nodes):
    for node in nodes:
        if node.kind == "chars":
            for i, ch in enumerate(node.text):
                if ch.isalpha():
                    yield (ch, node.start + i, node.start + i + 1)
        elif node.kind == "macro":
            yield (node.text, node.start, node.end)
        for arg in node.args:
            yield from _walk_atoms(arg["nodes"])


def detect_grounded_symbols(
    entity_id: str,
    text: str,
    singles: dict,
    multi_index: dict,
):
    """Run grounding strategies; return (scope records, env, strategy summary).

    Each emitted scope is a `math/grounded-symbol` record positioned at an
    atom that matched a binding in the per-paper SymbolEnvironment. The
    record carries the canon name + originating strategy in hx/content so
    downstream rendering can surface it.
    """
    kernel_lookup = _make_kernel_phrase_lookup(singles, multi_index)
    ctx = _sg.StrategyContext(
        paper_id=entity_id,
        paper_text=text,
        kernel_lookup=kernel_lookup,
    )
    env = _sg.run_strategies(ctx, _sg.default_strategies())

    records = []
    rec_idx = 0
    grounded_atom_count = 0
    for atom_text, start, end in _math_atoms_for_grounding(text):
        binding = env.lookup(atom_text, start)
        if binding is None:
            continue
        # Gate by strategy. NewcommandStrategy always emits — its canon
        # is a body-derived fallback when no kernel hit, which is still
        # informative (e.g. `\RR` -> "R"). Prose strategies only emit
        # when a kernel canon was found; without it the regex's
        # phrasal capture is noisy ("first assertion is", "components
        # in the ", …) and would pollute the viewer.
        if binding.strategy != "newcommand" and not binding.canon:
            continue
        if not binding.canon and not binding.type_phrase:
            continue
        grounded_atom_count += 1
        canon_or_fallback = binding.canon or binding.type_phrase[:24]
        role = _ma.classify_atom_role(atom_text)
        records.append({
            "hx/id": f"{entity_id}:grounded-{rec_idx:05d}",
            "hx/role": "scope",
            "hx/type": "math/grounded-symbol",
            "hx/parent": None,
            "hx/content": {
                "match": atom_text,
                "position": start,
                "end": end,
                "canon": binding.canon,
                "type_phrase": binding.type_phrase,
                "strategy": binding.strategy,
                "syntax_role": role,
            },
            "hx/labels": [
                "scope", "math", "grounded",
                f"strategy-{binding.strategy}",
                f"canon-{canon_or_fallback}",
            ],
        })
        rec_idx += 1

    # Per-strategy emission counts for the QC summary
    strategy_emit_counts: dict[str, int] = {}
    for b in env.all_bindings:
        strategy_emit_counts[b.strategy] = strategy_emit_counts.get(b.strategy, 0) + 1
    strategy_active_counts: dict[str, int] = {}
    for b in env.all_active():
        strategy_active_counts[b.strategy] = strategy_active_counts.get(b.strategy, 0) + 1

    summary = {
        "total_bindings_emitted": len(env.all_bindings),
        "active_bindings": len(env.all_active()),
        "grounded_atom_count": grounded_atom_count,
        "strategy_emit_counts": dict(sorted(strategy_emit_counts.items())),
        "strategy_active_counts": dict(sorted(strategy_active_counts.items())),
    }
    return records, env, summary


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
    """Wrapper around futon6.structure_seed.find_kernel_term_positions that
    feeds in this script's spot_terms_entity (sourced from superpod-job)."""
    return _ss.find_kernel_term_positions(text, SUPERPOD_JOB.spot_terms_entity, singles, multi_index)


def classify_kernel_terms(text: str, scopes: list[dict], singles: dict, multi_index: dict) -> dict:
    """Wrapper around futon6.structure_seed.classify_kernel_terms_from_positions
    that handles this script's spot-terms step + scope-record conversion.

    Returns flat inhabited/outer/straddled/total plus tree-aware
    depth_distribution. See the shared module for semantics.
    """
    positions = find_kernel_term_positions(text, singles, multi_index)
    span_records = _ss.scope_records_to_spans(scopes)
    return _ss.classify_kernel_terms_from_positions(positions, span_records)


def _term_span_html(text: str, start: int, end: int, canon: str | None, *, inhabited: bool) -> str:
    title = f' title="canon={html.escape(canon)}"' if canon else ""
    klass = "term-kernel inhabited" if inhabited else "term-kernel"
    return f'<span class="{klass}"{title}>{html.escape(text[start:end])}</span>'


# build_scope_tree is re-exported from futon6.structure_seed.
build_scope_tree = _ss.build_scope_tree


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
    # Depth-aware visual class so nested scopes are distinguishable. Cap at
    # depth-5+ so the palette stays bounded for arbitrarily deep nesting.
    depth_class = f'depth-{min(node["depth"], 5)}'
    # Type-slug class lets CSS override the palette per scope type (e.g.,
    # comment/unreachable becomes .comment-unreachable). Forward slashes and
    # whitespace get normalized to dashes for class-name validity.
    type_class = node["label"].replace("/", "-").replace(" ", "-")
    # Hover tooltip: surface enrichment from hx/content where present.
    # math/grounded-symbol scopes carry the canon name + originating strategy,
    # so a reader can hover any purple symbol and see "AbelianGroup (let-binding)".
    title_attr = ""
    content = node.get("content") or {}
    role_class = ""
    if node["label"] == "math/grounded-symbol":
        canon = content.get("canon")
        strategy = content.get("strategy") or "?"
        type_phrase = content.get("type_phrase") or ""
        syntax_role = content.get("syntax_role") or "variable"
        # If the strategy didn't get a kernel canon, fall back to a
        # truncated type_phrase so the badge stays informative. The full
        # type_phrase lives in the tooltip below.
        display = canon or (type_phrase[:18] if type_phrase else "?")
        title_text = f"{display} (via {strategy}; role={syntax_role}"
        if type_phrase and type_phrase != display:
            title_text += f": {type_phrase}"
        title_text += ")"
        title_attr = f' title="{html.escape(title_text)}"'
        label_text = display
        role_class = f" role-{syntax_role}"
    else:
        label_text = node["label"]
    label_html = f'<span class="scope-label">{html.escape(label_text)}</span>'
    return f'<mark class="scope {type_class} {depth_class}{role_class}"{title_attr}>{label_html}{"".join(out)}</mark>'


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
            "content": content,  # propagated so render_tree_node can surface enrichment (canon, strategy, ...)
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
    # Merge LaTeX-comment scopes so commented-out source counts toward
    # coverage (and so the kernel terms inside don't get reported as
    # scope-development frontier). Math sub-scopes (math/typed-arrow,
    # math/named-functor, etc.) fire inside $...$ blocks and nest under
    # any outer math scope (relation-expression, bind/typed) via the
    # shared scope-tree builder — that's the visible payoff for the
    # symbol-grounding mission (M-symbol-grounding.md).
    grounded_scopes, symbol_env, grounding_summary = detect_grounded_symbols(
        entity_id, eprint_text, singles, multi_index,
    )
    local_scopes = [
        *local_scopes,
        *NLAB_WIRING.detect_comments(entity_id, eprint_text),
        *NLAB_WIRING.detect_math_scopes(entity_id, eprint_text),
        *NLAB_WIRING.detect_math_scopes_ast(entity_id, eprint_text),
        *grounded_scopes,
    ]
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
        "local_symbol_grounding": grounding_summary,
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
    .scope {{ padding: 0 1px; border-radius: 3px; }}
    /* Depth-aware scope coloring. Outer (d1) = amber; nested levels shift hue
       through rose, violet, indigo, slate. d5+ caps the palette so arbitrarily
       deep nesting doesn't run out of distinguishable colors. */
    .scope.depth-1 {{ background: linear-gradient(90deg, #fbd38d, #fee2e2); }}
    .scope.depth-2 {{ background: linear-gradient(90deg, #fbcfe8, #fce7f3); }}
    .scope.depth-3 {{ background: linear-gradient(90deg, #ddd6fe, #ede9fe); }}
    .scope.depth-4 {{ background: linear-gradient(90deg, #c7d2fe, #e0e7ff); }}
    .scope.depth-5 {{ background: linear-gradient(90deg, #a5b4fc, #cbd5e1); outline: 1px dashed rgba(71, 85, 105, 0.4); outline-offset: -2px; }}
    /* LaTeX comment scope overrides the depth palette: dim background plus
       strikethrough makes "this content is unreachable in the PDF" obvious. */
    .scope.comment-unreachable {{ background: rgba(120, 113, 108, 0.16); color: rgba(60, 60, 60, 0.55); text-decoration: line-through; text-decoration-color: rgba(120, 113, 108, 0.5); outline: 1px dotted rgba(120, 113, 108, 0.45); outline-offset: -1px; }}
    .scope.comment-unreachable .scope-label {{ background: rgba(120, 113, 108, 0.25); text-decoration: none; }}
    /* Grounded math symbol — per-paper SymbolEnvironment match. Saturated
       purple to read against the depth palette and pair with the inhabited
       term overlay. The label badge shows the strategy that bound it. */
    .scope.math-grounded-symbol {{ background: rgba(124, 58, 237, 0.34); outline: 1px solid rgba(124, 58, 237, 0.8); outline-offset: -1px; }}
    .scope.math-grounded-symbol .scope-label {{ background: rgba(124, 58, 237, 0.55); color: white; }}
    /* First Proof syntactic-role palette (math-proofread-style.sty v0.9).
       Applied to the grounded-symbol mark's label so the reader sees the
       role at a glance: Greek = Mulberry, named-op = BurntOrange, etc.
       The mark BACKGROUND stays purple (grounding signal); the LABEL
       chip carries the role color so the two signals can coexist. */
    .scope.math-grounded-symbol.role-greek .scope-label {{ background: #c92a82; }}
    .scope.math-grounded-symbol.role-binop .scope-label {{ background: #8b008b; }}
    .scope.math-grounded-symbol.role-bridge .scope-label {{ background: #2e8b57; }}
    .scope.math-grounded-symbol.role-relation .scope-label {{ background: #7851a9; }}
    .scope.math-grounded-symbol.role-comparison .scope-label {{ background: #004225; }}
    .scope.math-grounded-symbol.role-large-op .scope-label {{ background: #8a2be2; }}
    .scope.math-grounded-symbol.role-arrow .scope-label {{ background: #008080; }}
    .scope.math-grounded-symbol.role-function .scope-label {{ background: #da70d6; }}
    .scope.math-grounded-symbol.role-delimiter .scope-label {{ background: #ff00ff; }}
    .scope.math-grounded-symbol.role-named-op .scope-label {{ background: #cc5500; }}
    .scope.math-grounded-symbol.role-number .scope-label {{ background: #b22222; }}
    .scope.math-grounded-symbol.role-variable .scope-label {{ background: rgba(124, 58, 237, 0.55); }}
    .scope-label {{ display: inline-block; margin-right: 6px; padding: 0 4px; background: rgba(29, 26, 22, 0.1); border-radius: 999px; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
    .term-kernel {{ background: rgba(15, 118, 110, 0.12); border-bottom: 1px solid rgba(15, 118, 110, 0.45); padding: 0 1px; border-radius: 2px; cursor: help; }}
    .term-kernel:hover {{ background: rgba(15, 118, 110, 0.22); }}
    .term-kernel.inhabited {{ background: rgba(124, 58, 237, 0.18); border-bottom: 1px solid rgba(124, 58, 237, 0.6); }}
    .term-kernel.inhabited:hover {{ background: rgba(124, 58, 237, 0.30); }}
    .markup-legend {{ font: 12px/1.4 system-ui, sans-serif; color: var(--muted); margin: 8px 0 6px 0; }}
    .markup-legend .swatch {{ display: inline-block; padding: 0 6px; border-radius: 3px; margin-right: 4px; }}
    .markup-legend .swatch.scope.depth-1 {{ background: linear-gradient(90deg, #fbd38d, #fee2e2); }}
    .markup-legend .swatch.scope.depth-2 {{ background: linear-gradient(90deg, #fbcfe8, #fce7f3); }}
    .markup-legend .swatch.scope.depth-3 {{ background: linear-gradient(90deg, #ddd6fe, #ede9fe); }}
    .markup-legend .swatch.scope.depth-4 {{ background: linear-gradient(90deg, #c7d2fe, #e0e7ff); }}
    .markup-legend .swatch.scope.depth-5 {{ background: linear-gradient(90deg, #a5b4fc, #cbd5e1); outline: 1px dashed rgba(71, 85, 105, 0.4); outline-offset: -2px; }}
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
          <span class="swatch scope depth-1">d1</span>
          <span class="swatch scope depth-2">d2</span>
          <span class="swatch scope depth-3">d3</span>
          <span class="swatch scope depth-4">d4</span>
          <span class="swatch scope depth-5">d5+</span>
          scope nesting depth (outer → inner)
          &nbsp;·&nbsp;
          <span class="swatch term">term</span> kernel term in residual prose (= <strong>scope-development candidate</strong>)
          &nbsp;·&nbsp;
          <span class="swatch term-inhabited">term</span> kernel term <em>inhabiting</em> a scope
          &nbsp;·&nbsp;
          <span class="swatch scope math-grounded-symbol" style="padding: 0 6px;">X</span> math symbol <em>grounded</em> per-paper by the symbol-grounding strategies
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
        <p class="markup-legend">
          Symbol-grounding strategies fired (per-paper, defeasible):
          <code>{(paper.get('local_symbol_grounding') or {}).get('total_bindings_emitted', 0)}</code> bindings emitted,
          <code>{(paper.get('local_symbol_grounding') or {}).get('active_bindings', 0)}</code> still active at end of paper,
          <code>{(paper.get('local_symbol_grounding') or {}).get('grounded_atom_count', 0)}</code> math-atom occurrences grounded.
          By strategy:
          {' '.join(f'{html.escape(s)}:&nbsp;<code>{n}</code>' for s, n in ((paper.get('local_symbol_grounding') or {}).get('strategy_emit_counts') or {}).items()) or '<em>none</em>'}.
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
                "local_symbol_grounding": paper.get("local_symbol_grounding"),
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
