"""Shared structure-seed primitives used by the audit, viewer, and superpod runners.

Three runners need consistent versions of these primitives:

- `scripts/superpod-job.py` (Stage 5 emits learned-structure-candidates.json,
  applies replay matcher, runs end-of-job audit)
- `scripts/build-uncovered-sentence-audit.py` (daily-use audit producing
  candidate signatures + learned-discourse-patterns.json)
- `scripts/build-batch-008-qc-viewer.py` (per-paper QC visualization)

Each previously inlined its own copy with slight divergences. This module is
the single source of truth.

Conventions:
- A `<TERM>` placeholder represents a NER-kernel-known term occurrence.
- `<MATH>` / `<CITE>` / `<CMD>` / `<NUM>` are other placeholder kinds.
- A signature is a space-separated token stream where placeholders use
  lowercase `<term>` form.
- "Coarse" signature = the discourse-verb + structural-connective backbone,
  used as the clustering key for cross-paper aggregation.
- "Full" signature = the per-residual signature including placeholders.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# ============================================================
# Language tables
# ============================================================

STRUCTURE_CUE_WORDS = frozenset({
    "we", "let", "define", "denote", "write", "show", "prove", "obtain", "apply",
    "study", "consider", "introduce", "recall", "if", "then", "assume", "suppose",
    "where", "when", "for", "any", "every", "there", "exists", "be",
    "that", "and", "or", "not", "only", "particular", "consist", "depend",
    "turn", "focus", "choose", "work",
})

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

# Discourse-verb taxonomy. Drives the prefilter (candidate must contain at
# least one of these) and the predicted_kind heuristic.
DISCOURSE_VERB_KIND = {
    # scope: binding or assumption
    "let": "scope",
    "define": "scope",
    "denote": "scope",
    "write": "scope",
    "fix": "scope",
    "assume": "scope",
    "suppose": "scope",
    # label: rhetorical / strategic move
    "prove": "label",
    "show": "label",
    "obtain": "label",
    "derive": "label",
    "study": "label",
    "consider": "label",
    "introduce": "label",
    "recall": "label",
    "apply": "label",
    # wire: discourse connective
    "then": "wire",
    "therefore": "wire",
    "notice": "wire",
    "observe": "wire",
}
DISCOURSE_VERBS = frozenset(DISCOURSE_VERB_KIND.keys())

# Coarse clustering key: drop placeholders and content tokens, keep only the
# discourse-verb + structural-connective backbone. Two residuals with the same
# coarse signature share the same discourse role even if their full token
# sequences differ in noun-phrase positioning.
COARSE_STRUCTURAL_CUES = DISCOURSE_VERBS | frozenset({
    "and", "or", "not", "that", "be", "exists", "there", "if", "then",
    "where", "when", "we", "for", "any", "every",
})


# ============================================================
# Signature normalization & skeletonization
# ============================================================

def normalize_structure_seed_text(sentence, known_term_hits):
    """Replace kernel-known terms with `<TERM>`, math with `<MATH>`, etc.

    Returns a lowercase whitespace-normalized template ready for skeletonization.
    `known_term_hits` is a list of dicts with keys term/term_lower (the kernel
    spotter shape).
    """
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


def structure_seed_skeleton(normalized_template):
    """Reduce a normalized template to its cue-token + placeholder backbone.

    Tokens go through STRUCTURE_CUE_LEMMAS, then are kept only if they're in
    STRUCTURE_CUE_WORDS or are placeholders. Adjacent identical placeholders
    collapse (e.g. `<term> <term>` → `<term>`).
    """
    tokens = re.findall(r"<[a-z]+>|[a-z]+", normalized_template or "")
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


def coarse_discourse_signature(full_signature):
    """Reduce a full skeleton to the discourse-verb + structural-connective backbone.

    Placeholders are dropped. Only tokens in COARSE_STRUCTURAL_CUES survive.
    Used as the bucketing key for cross-paper candidate aggregation.
    """
    tokens = (full_signature or "").split()
    kept = [t for t in tokens if t in COARSE_STRUCTURAL_CUES]
    return " ".join(kept)


# ============================================================
# Discourse-verb classification
# ============================================================

def signature_has_discourse_verb(signature):
    return any(token in DISCOURSE_VERBS for token in (signature or "").split())


def predict_kind_from_signature(signature):
    """Classify a signature by its strongest discourse verb.

    Preference order: scope > label > wire. Returns None if no discourse verb
    appears (those candidates are dropped from gated patterns).
    """
    kinds = [
        DISCOURSE_VERB_KIND[token]
        for token in (signature or "").split()
        if token in DISCOURSE_VERB_KIND
    ]
    if not kinds:
        return None
    for preferred in ("scope", "label", "wire"):
        if preferred in kinds:
            return preferred
    return None


# ============================================================
# Cross-paper candidate aggregation
# ============================================================

def summarize_structure_seed_candidates(
    rows,
    *,
    min_signature_freq=1,
    max_candidates=1000,
):
    """Aggregate per-residual rows into cross-paper candidate signatures.

    Each row should carry `structure_seed_signature` (full per-residual
    signature), `paper_id`, `known_term_hits`, `known_term_hit_count`,
    `text`, `index`. Rows without a discourse verb are filtered upfront.
    Bucketing is by COARSE signature so analogous constructions across
    papers cluster together. Each bucket carries `full_signatures` (the
    full per-residual signatures observed) so the replay matcher still has
    precise priors to subsequence-match against.

    Sorted by (paper_count, count, max_known_term_hit_count, len(signature))
    descending.
    """
    buckets = {}
    for row in rows:
        if row.get("known_term_hit_count", 0) <= 0:
            continue
        full_signature = row.get("structure_seed_signature") or ""
        if not full_signature:
            continue
        if not signature_has_discourse_verb(full_signature):
            continue
        coarse = coarse_discourse_signature(full_signature)
        if not coarse:
            continue
        bucket = buckets.setdefault(coarse, {
            "signature": coarse,
            "count": 0,
            "paper_ids": set(),
            "full_signatures": set(),
            "example_sentences": [],
            "max_known_term_hit_count": 0,
        })
        bucket["count"] += 1
        bucket["paper_ids"].add(row.get("paper_id"))
        bucket["full_signatures"].add(full_signature)
        bucket["max_known_term_hit_count"] = max(
            bucket["max_known_term_hit_count"],
            row.get("known_term_hit_count", 0),
        )
        if len(bucket["example_sentences"]) < 3:
            bucket["example_sentences"].append({
                "paper_id": row.get("paper_id"),
                "index": row.get("index"),
                "text": row.get("text"),
                "full_signature": full_signature,
                "known_terms": [
                    item["term_lower"] for item in row.get("known_term_hits", [])[:8]
                ],
            })

    out = []
    for bucket in buckets.values():
        if bucket["count"] < min_signature_freq:
            continue
        out.append({
            "signature": bucket["signature"],
            "count": bucket["count"],
            "paper_ids": sorted(bucket["paper_ids"]),
            "paper_count": len(bucket["paper_ids"]),
            "full_signatures": sorted(bucket["full_signatures"]),
            "max_known_term_hit_count": bucket["max_known_term_hit_count"],
            "predicted_kind": predict_kind_from_signature(bucket["signature"]),
            "example_sentences": bucket["example_sentences"],
        })
    out.sort(
        key=lambda row: (
            row["paper_count"],
            row["count"],
            row["max_known_term_hit_count"],
            len(row["signature"]),
        ),
        reverse=True,
    )
    return out[:max_candidates]


# ============================================================
# Replay matcher (subsequence over signature tokens)
# ============================================================

STRUCTURE_SEED_MIN_TOKENS = 3


def signature_tokens(signature):
    return tuple(re.findall(r"<[a-z]+>|[a-z]+", signature or ""))


def is_subsequence(needle, haystack):
    if not needle:
        return False
    i = 0
    for token in haystack:
        if token == needle[i]:
            i += 1
            if i == len(needle):
                return True
    return False


def match_structure_seed_signature(new_signature, prior_signatures, min_tokens=STRUCTURE_SEED_MIN_TOKENS):
    """Return the longest prior signature that appears as a subsequence of new.

    `prior_signatures` is a list of (signature_str, token_tuple). Priors with
    fewer than `min_tokens` are skipped to suppress degenerate matches.
    """
    new_tokens = signature_tokens(new_signature)
    if not new_tokens:
        return None
    best = None
    best_len = 0
    for prior_sig, prior_tokens in prior_signatures:
        if len(prior_tokens) < min_tokens:
            continue
        if len(prior_tokens) > len(new_tokens):
            continue
        if is_subsequence(prior_tokens, new_tokens):
            if len(prior_tokens) > best_len:
                best = prior_sig
                best_len = len(prior_tokens)
    return best


def load_structure_seed_signatures(path):
    """Load prior signatures for the replay matcher.

    Prefers each candidate's `full_signatures` list (post-refactor schema);
    falls back to top-level `signature` for older single-signature outputs.
    Returns list of (signature_str, token_tuple) sorted longest-first.
    """
    if path is None or not Path(path).exists():
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict):
        rows = payload.get("structure_seed_candidates") or payload.get("candidates") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        return []
    out = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_sigs = list(row.get("full_signatures") or [])
        if not candidate_sigs:
            fallback = (row.get("signature") or row.get("structure_seed_signature") or "").strip()
            if fallback:
                candidate_sigs = [fallback]
        for sig in candidate_sigs:
            signature = (sig or "").strip()
            if not signature or signature in seen:
                continue
            seen.add(signature)
            tokens = signature_tokens(signature)
            if not tokens:
                continue
            out.append((signature, tokens))
    out.sort(key=lambda item: len(item[1]), reverse=True)
    return out


# ============================================================
# Scope tree (containment hierarchy)
# ============================================================

def build_scope_tree(scope_spans, term_positions):
    """Arrange flat scope spans into a containment tree.

    `scope_spans` is a list of dicts with keys start, end, label.
    `term_positions` is a list of (start, end, ...) tuples.

    Returns a root node with sentinel start=-1, end=10**18, label="$root".
    Each non-root node has start, end, label, depth (1 for root's direct
    children), children (list), terms (list of tuples placed at this depth).

    Scopes that straddle a tree-ancestor's boundary are dropped (can't be
    tree-arranged). Terms that straddle any input scope are dropped. Each
    surviving term is placed in its deepest containing scope, or at the
    root if no scope contains it.
    """
    root = {
        "start": -1,
        "end": 10**18,
        "label": "$root",
        "children": [],
        "terms": [],
        "depth": 0,
    }
    sorted_scopes = sorted(scope_spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    stack = [root]
    for sp in sorted_scopes:
        while stack[-1] is not root and stack[-1]["end"] <= sp["start"]:
            stack.pop()
        top = stack[-1]
        if top["start"] <= sp["start"] and sp["end"] <= top["end"]:
            node = {
                "start": sp["start"],
                "end": sp["end"],
                "label": sp["label"],
                "content": sp.get("content"),  # preserved enrichment fields
                "children": [],
                "terms": [],
                "depth": top["depth"] + 1,
            }
            top["children"].append(node)
            stack.append(node)

    def find_deepest(node, ts, te):
        if not (node["start"] <= ts and te <= node["end"]):
            return None
        for child in node["children"]:
            deeper = find_deepest(child, ts, te)
            if deeper is not None:
                return deeper
        return node

    def straddles_any_input_scope(ts, te):
        for sp in scope_spans:
            ss, se = sp["start"], sp["end"]
            if se <= ts or ss >= te:
                continue
            if ss <= ts and te <= se:
                continue
            return True
        return False

    for term in term_positions:
        ts, te = term[0], term[1]
        if straddles_any_input_scope(ts, te):
            continue
        deepest = find_deepest(root, ts, te)
        if deepest is not None:
            deepest["terms"].append(term)
    return root


def scope_records_to_spans(records):
    """Convert detector records (hx/content with position+end) to scope_spans dicts.

    Skips records without a usable position. The full `hx/content` dict is
    preserved on the span as `content` so downstream renderers can surface
    enrichment fields (canon, strategy, matched_prior_signature, ...) as
    tooltips or sub-labels without having to re-look-up the original
    record.
    """
    out = []
    for rec in records:
        content = rec.get("hx/content", {}) or {}
        start = content.get("position")
        end = content.get("end")
        if not isinstance(start, int):
            continue
        if not isinstance(end, int):
            end = start + len(content.get("match", ""))
        out.append({
            "start": start,
            "end": end,
            "label": rec.get("hx/type", "?"),
            "content": content,
        })
    return out


# ============================================================
# Kernel-term classification (frontier metric)
# ============================================================

def classify_kernel_terms_from_positions(term_positions, scope_spans):
    """Classify kernel term positions against scope spans.

    `term_positions`: list of (start, end, ...) tuples (caller supplies; the
    spot-terms step is runner-specific).
    `scope_spans`: list of {start, end, label} dicts (the input shape that
    `build_scope_tree` consumes).

    Returns {inhabited, outer, straddled, total, depth_distribution}.
    Inhabited / outer / straddled use FLAT semantics (any-scope containment)
    so the frontier metric is order-independent. depth_distribution comes
    from the tree builder; terms inside scopes the tree dropped count as
    inhabited but don't show up in depth_distribution (best-effort caveat).
    """
    inhabited = outer = straddled = 0
    inhabited_terms = []
    for term in term_positions:
        ts, te = term[0], term[1]
        contained = False
        overlaps_any = False
        for sp in scope_spans:
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

    tree = build_scope_tree(scope_spans, inhabited_terms)
    depth_dist = {}

    def walk(node):
        for child in node["children"]:
            count = len(child["terms"])
            if count:
                depth_dist[child["depth"]] = depth_dist.get(child["depth"], 0) + count
            walk(child)

    walk(tree)
    return {
        "inhabited": inhabited,
        "outer": outer,
        "straddled": straddled,
        "total": inhabited + outer + straddled,
        "depth_distribution": dict(sorted(depth_dist.items())),
    }


def find_kernel_term_positions(text, spot_terms_fn, singles, multi_index):
    """Locate every kernel-term occurrence in text.

    `spot_terms_fn(text, singles, multi_index) -> list[dict with term_lower]`
    is the runner-specific spotter. Term positions are deduplicated by
    longest-first non-overlapping greedy.

    Returns (start, end, term_lower, canon) tuples.
    """
    hits = spot_terms_fn(text, singles, multi_index)
    positions = []
    for hit in hits:
        tl = hit.get("term_lower") or hit.get("term") or ""
        if not tl:
            continue
        try:
            pattern = re.compile(rf"\b{re.escape(tl)}\b", re.IGNORECASE)
        except re.error:
            continue
        canon = hit.get("canon")
        for m in pattern.finditer(text):
            positions.append((m.start(), m.end(), tl, canon))
    positions.sort(key=lambda r: (r[0], -(r[1] - r[0])))
    deduped = []
    last_end = -1
    for ts, te, tl, canon in positions:
        if ts >= last_end:
            deduped.append((ts, te, tl, canon))
            last_end = te
    return deduped
