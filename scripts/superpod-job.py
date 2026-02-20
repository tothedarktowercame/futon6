#!/usr/bin/env python3
"""Superpod batch job: process math.stackexchange into F6 artefacts.

Self-contained job that reads a SE data dump and produces a single output
directory with all computed artefacts. Designed for GPU-accelerated batch
processing on a multi-GPU machine, but CPU-only stages run fine on a laptop.

Download + run (self-contained):
    python scripts/superpod-job.py --download math --data-dir ./se-data
    python scripts/superpod-job.py ./se-data/math.stackexchange.com/Posts.xml \\
        --output-dir ./math-se-processed --site math.stackexchange

    python scripts/superpod-job.py --download mathoverflow --data-dir ./se-data
    python scripts/superpod-job.py ./se-data/mathoverflow.net/Posts.xml \\
        --output-dir ./mo-processed --site mathoverflow.net

    # CPU-only (no GPU required):
    python scripts/superpod-job.py ./se-data/math.stackexchange.com/Posts.xml \\
        --skip-embeddings --skip-llm --skip-clustering \\
        --site math.stackexchange --output-dir ./math-se-processed

    # Dry run (shows plan, downloads nothing, processes nothing):
    python scripts/superpod-job.py ./se-data/math.stackexchange.com/Posts.xml --dry-run

    # Moist run (CPU stages + prompt files for Codex/Claude handoff):
    python scripts/superpod-job.py ./se-data/math.stackexchange.com/Posts.xml \\
        --moist-run --site math.stackexchange

All stages:
    0. Download + extract (optional, --download)
    1. Parse XML -> QA pairs (CPU, streaming)
    2. Dense embeddings (GPU, bge-large-en-v1.5)
    3. LLM pattern tagging (GPU, Llama-3-8B)
    4. Clustering (CPU, HDBSCAN on embeddings)
    5. NER term spotting + scope detection (CPU, classical)
    6. Reverse morphogenesis S<-Q<-A (LLM / prompt generation)
    7. Thread wiring diagrams + IATC performatives (CPU, classical + LLM)
    8. Expression surface parsing (CPU, LaTeX -> s-exp)
    9a. Hypergraph assembly (CPU, typed hypergraph from wiring + expressions)
    9b. Graph embedding (GPU, R-GCN contrastive learning on hypergraphs)
    10. FAISS structural similarity index (CPU)

Each stage writes its output independently -- if a stage fails, earlier
outputs are still usable.

Stage 5 output uses futon4-compatible hyperedge format for scope records:
  :hx/type, :hx/ends (with roles), :hx/content, :hx/labels
This enables direct ingest into futon1/XTDB via the standard relation->hx
conversion path. See futon1/apps/graph-memory for schema.
"""

import argparse
import hashlib
import importlib
import itertools
import json
import os
import random
import re
import sys
import tarfile
import time
import uuid
from collections import Counter
from pathlib import Path

import shutil
import subprocess

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# CT-backed wiring modules (from scripts/)
_assemble_wiring = None
_nlab_wiring = None
_scope_detector = None
_scope_detector_name = None

def _load_ct_modules():
    """Lazy-load CT wiring modules (only needed for Stage 7 CT path)."""
    global _assemble_wiring, _nlab_wiring
    if _assemble_wiring is None:
        _assemble_wiring = importlib.import_module("assemble-wiring")
        _nlab_wiring = importlib.import_module("nlab-wiring")
    return _assemble_wiring, _nlab_wiring


def _load_scope_detector(prefer_nlab=True):
    """Return (detect_scopes_fn, detector_name), preferring nlab-wiring."""
    global _scope_detector, _scope_detector_name
    if _scope_detector is not None:
        return _scope_detector, _scope_detector_name

    if prefer_nlab:
        try:
            nw = importlib.import_module("nlab-wiring")
            _scope_detector = nw.detect_scopes
            _scope_detector_name = "nlab-wiring.detect_scopes"
            return _scope_detector, _scope_detector_name
        except Exception as exc:
            print(f"       Scope detector fallback: nlab-wiring unavailable ({exc})")

    _scope_detector = detect_scopes_entity
    _scope_detector_name = "superpod.detect_scopes_entity"
    return _scope_detector, _scope_detector_name

from futon6.stackexchange import (
    build_qa_pairs_streaming,
    build_threads_streaming,
    load_arxiv_pairs,
    qa_to_entity,
    qa_to_relations,
    tag_entities,
    corpus_stats,
    compute_qa_embeddings,
)
from futon6.thread_performatives import (
    build_thread_wiring_diagram,
    build_thread_performative_prompt,
    process_threads_to_diagrams,
    diagram_to_dict,
    merge_llm_edges,
    write_thread_wiring_json,
)
from futon6.latex_sexp import parse as sexp_parse, parse_all
from futon6.hypergraph import assemble as assemble_hypergraph
from futon6.faiss_index import build_index, save_index

_GRAPH_EMBED_IMPORT_ERROR = None
try:
    from futon6.graph_embed import train as train_gnn, embed_hypergraphs, save_model
except Exception as exc:
    train_gnn = None
    embed_hypergraphs = None
    save_model = None
    _GRAPH_EMBED_IMPORT_ERROR = exc

# --- Downloadable SE data dumps (Internet Archive) ---

SE_DUMPS = {
    "math": {
        "url": "https://archive.org/download/stackexchange/math.stackexchange.com.7z",
        "site": "math.stackexchange",
        "dirname": "math.stackexchange.com",
        "description": "Math StackExchange (~3.4 GB compressed, ~567K QA pairs)",
    },
    "mathoverflow": {
        "url": "https://archive.org/download/stackexchange/mathoverflow.net.7z",
        "site": "mathoverflow.net",
        "dirname": "mathoverflow.net",
        "description": "MathOverflow (~500 MB compressed, ~100K QA pairs, research-level)",
    },
    "physics": {
        "url": "https://archive.org/download/stackexchange/physics.stackexchange.com.7z",
        "site": "physics.stackexchange",
        "dirname": "physics.stackexchange.com",
        "description": "Physics StackExchange (~1 GB compressed, ~114K QA pairs)",
    },
}


def download_and_extract(dump_key, data_dir):
    """Download a SE dump from Internet Archive and extract it.

    Uses wget for download and 7z for extraction. Both must be installed.
    """
    if dump_key not in SE_DUMPS:
        print(f"Unknown dump '{dump_key}'. Available: {', '.join(SE_DUMPS.keys())}")
        sys.exit(1)

    dump = SE_DUMPS[dump_key]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_name = dump["url"].split("/")[-1]
    archive_path = data_dir / archive_name
    extract_dir = data_dir / dump["dirname"]
    posts_path = extract_dir / "Posts.xml"

    print("=" * 64)
    print(f"DOWNLOAD: {dump['description']}")
    print("=" * 64)

    # Check if already extracted
    if posts_path.exists():
        size = posts_path.stat().st_size / 1e9
        print(f"\n  Already extracted: {posts_path} ({size:.1f} GB)")
        print(f"  To re-download, remove {extract_dir}")
        print(f"\n  Run with:")
        print(f"    python scripts/superpod-job.py {posts_path} \\")
        print(f"      --site {dump['site']} --output-dir {data_dir}/{dump_key}-processed")
        return str(posts_path)

    # Check tools
    for tool in ["wget", "7z"]:
        if not shutil.which(tool):
            print(f"\n  ERROR: '{tool}' not found. Install it:")
            if tool == "wget":
                print(f"    apt install wget  # or: brew install wget")
            else:
                print(f"    apt install p7zip-full  # or: brew install p7zip")
            sys.exit(1)

    # Download
    if archive_path.exists():
        size = archive_path.stat().st_size / 1e9
        print(f"\n  Archive exists: {archive_path} ({size:.2f} GB)")
    else:
        print(f"\n  Downloading {dump['url']}")
        print(f"  To: {archive_path}")
        print()
        rc = subprocess.run(
            ["wget", "-c", "--progress=bar:force:noscroll",
             "-O", str(archive_path), dump["url"]],
        ).returncode
        if rc != 0:
            print(f"\n  ERROR: wget failed (exit {rc})")
            sys.exit(1)
        size = archive_path.stat().st_size / 1e9
        print(f"\n  Downloaded {size:.2f} GB")

    # Extract
    print(f"\n  Extracting to {extract_dir}/")
    extract_dir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        ["7z", "x", "-y", f"-o{extract_dir}", str(archive_path)],
    ).returncode
    if rc != 0:
        print(f"\n  ERROR: 7z extraction failed (exit {rc})")
        sys.exit(1)

    if posts_path.exists():
        size = posts_path.stat().st_size / 1e9
        print(f"\n  Extracted: {posts_path} ({size:.1f} GB)")
    else:
        print(f"\n  WARNING: Posts.xml not found after extraction")
        print(f"  Contents of {extract_dir}:")
        for f in sorted(extract_dir.iterdir()):
            print(f"    {f.name}")
        sys.exit(1)

    # Suggest next command
    print(f"\n  Run with:")
    print(f"    python scripts/superpod-job.py {posts_path} \\")
    print(f"      --site {dump['site']} --output-dir {data_dir}/{dump_key}-processed")
    print()
    print(f"  CPU-only (no GPU):")
    print(f"    python scripts/superpod-job.py {posts_path} \\")
    print(f"      --skip-embeddings --skip-llm --skip-clustering \\")
    print(f"      --site {dump['site']} --output-dir {data_dir}/{dump_key}-processed")

    return str(posts_path)


# --- The 25 math-informal patterns (baked in for self-containment) ---

PATTERNS = [
    ("try-a-simpler-case", "Reduce parameters to the smallest non-trivial value and solve that case first"),
    ("work-examples-first", "Compute concrete examples before attempting a general proof"),
    ("argue-by-contradiction", "Assume the negation and derive an absurdity"),
    ("find-the-right-abstraction", "Seek a higher-level framework where the proof becomes natural"),
    ("quotient-by-irrelevance", "Identify what doesn't matter and collapse it via equivalence relation"),
    ("check-the-extreme-cases", "Test the conjecture at zero, infinity, empty set, trivial case"),
    ("exploit-symmetry", "Use symmetry (WLOG, orbits, invariants) to reduce work"),
    ("construct-an-explicit-witness", "Prove existence by building the object"),
    ("dualise-the-problem", "Reverse arrows, take complements, or transpose — solve in dual setting"),
    ("reduce-to-known-result", "Transform until hypotheses match a known theorem, then invoke it"),
    ("pass-to-a-subsequence", "Extract convergent subsequence via compactness"),
    ("induction-and-well-ordering", "Prove base case and inductive step"),
    ("local-to-global", "Prove on local pieces, assemble global result via patching"),
    ("the-diagonal-argument", "Defeat enumeration by constructing object differing from every listed item"),
    ("encode-as-algebra", "Translate problem into algebraic language and solve there"),
    ("use-probabilistic-method", "Show random object has desired property with positive probability"),
    ("unfold-the-definition", "Expand defined terms — write out what the words mean"),
    ("construct-auxiliary-object", "Introduce purpose-built object mediating hypothesis and conclusion"),
    ("estimate-by-bounding", "Replace with simpler bound via standard inequalities"),
    ("split-into-cases", "Partition into qualitatively different cases, handle each"),
    ("verify-universal-property", "Check existence and uniqueness of mediating morphism"),
    ("optimise-a-free-parameter", "Choose parameter value giving tightest bound"),
    ("transport-across-isomorphism", "Prove in easier setting, transport via isomorphism"),
    ("show-both-inequalities", "Prove equality via inequality in both directions"),
    ("monotone-approximation", "Approximate by monotone sequence, pass to limit"),
]

PATTERN_NAMES = [p[0] for p in PATTERNS]


def derive_default_output_dir(site):
    """Pick a stable default output directory from site name."""
    normalized = (site or "").strip().lower()
    known = {
        "math.stackexchange": "./math-se-processed",
        "math.stackexchange.com": "./math-se-processed",
        "mathoverflow": "./mo-processed",
        "mathoverflow.net": "./mo-processed",
        "physics.stackexchange": "./physics-se-processed",
        "physics.stackexchange.com": "./physics-se-processed",
    }
    if normalized in known:
        return known[normalized]

    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if not slug:
        slug = "se-site"
    return f"./{slug}-processed"


def write_json(path, data):
    """Write JSON with reasonable formatting."""
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"       Written {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


# --- Stage 5: NER term spotting + scope detection (CPU, classical) ---

# Scope detection regexes (from data/ner-kernel/scope-patterns.edn).
# Baked in for self-containment, same as patterns above.
SCOPE_REGEXES = [
    ("let-binding", r"\bLet\s+\$([^$]+)\$\s+(be|denote)\s+([^.,$]+)"),
    ("define", r"\bDefine\s+\$([^$]+)\$\s*(:=|=|\\equiv)\s*([^.,$]+)"),
    ("assume", r"\b(Assume|Suppose)\s+(that\s+)?\$([^$]+)\$"),
    ("consider", r"\bConsider\s+(a|an|the|some)?\s*\$?([^$.]{1,60})"),
    ("for-any", r"\b(?:for\s+)?(any|every|each|all)\s+\$([^$]+)\$"),
    ("where-binding", r"\bwhere\s+\$([^$]+)\$\s+(is|denotes|represents)\s+([^.,$]+)"),
    ("set-notation", r"\$([^$]*\\in\s+[^$]+)\$"),
]

CLASSICAL_TO_METATHEORY = {
    "let-binding": "bind/let",
    "define": "bind/define",
    "assume": "assume/explicit",
    "consider": "assume/consider",
    "for-any": "quant/universal",
    "where-binding": "constrain/where",
    "set-notation": "constrain/such-that",
}

BINDER_TYPE_BY_COMMAND = {
    "sum": "bind/summation",
    "prod": "bind/product",
    "coprod": "bind/coprod",
    "bigcup": "bind/big-union",
    "bigcap": "bind/big-intersection",
}

SYMBOL_RE = re.compile(r"[A-Za-z](?:_[A-Za-z0-9]+)?")
QUANTIFIER_CMD_RE = re.compile(
    r"\\(forall|exists)\s*(?:\{)?\s*([A-Za-z](?:_[A-Za-z0-9]+)?)"
)
AGGREGATE_BINDER_RE = re.compile(
    r"\\(sum|prod|coprod|bigcup|bigcap)\s*(?:_\{([^}]*)\}|_([A-Za-z](?:_[A-Za-z0-9]+)?))?"
)
INTEGRAL_BINDER_RE = re.compile(
    r"\\int(?:\s*(?:_\{[^}]*\}|_[^\s^{}]+))?(?:\s*(?:\^\{[^}]*\}|\^[^\s{}]+))?"
)
PROSE_ENV_TYPE_MAP = {
    "definition": "env/definition",
    "defn": "env/definition",
    "theorem": "env/theorem",
    "lemma": "env/lemma",
    "proposition": "env/proposition",
    "prop": "env/proposition",
    "corollary": "env/corollary",
    "remark": "env/remark",
    "example": "env/example",
    "proof": "env/proof",
    "notation": "env/definition",
}
LATEX_ENV_STYLE = {
    "defn": "env/definition",
    "definition": "env/definition",
    "theorem": "env/theorem",
    "thm": "env/theorem",
    "prop": "env/proposition",
    "proposition": "env/proposition",
    "lemma": "env/lemma",
    "cor": "env/corollary",
    "corollary": "env/corollary",
    "remark": "env/remark",
    "rmk": "env/remark",
    "example": "env/example",
    "proof": "env/proof",
    "note": "env/note",
    "notation": "env/definition",
}
LATEX_ENV_OPEN_RE = re.compile(r"\\begin\{(\w+)\}")
PROSE_ENV_HEADING_RE = re.compile(
    r"(?m)^\s*(Definition|Defn|Theorem|Lemma|Proposition|Prop|Corollary|Remark|Example|Proof|Notation)\b[:.]?"
)
LATEX_EMPH_RE = re.compile(r"\\(?:emph|textit|textbf)\{([^{}]{3,120})\}")
CALLED_AS_RE = re.compile(
    r"\b(?:called|known as|termed)\s+(?:the\s+)?([A-Za-z][A-Za-z\- ]{2,80})",
    re.IGNORECASE,
)
IS_CALLED_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\- ]{2,80})\s+is\s+(?:called|known as|termed)\b",
    re.IGNORECASE,
)
DEFINED_AS_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\- ]{2,80})\s+is\s+(?:defined as|defined to be)\b",
    re.IGNORECASE,
)
DEFINITION_OF_RE = re.compile(
    r"\bdefinition\s+of\s+([A-Za-z][A-Za-z\- ]{2,80})",
    re.IGNORECASE,
)
LATEX_DEFINITION_BLOCK_RE = re.compile(
    r"\\begin\{definition\}(.{0,320}?)\\end\{definition\}",
    re.IGNORECASE | re.DOTALL,
)
DEF_BLOCK_SUBJECT_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\- ]{2,80})\s+is\b",
    re.IGNORECASE,
)
DISCOVERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "its", "of", "on", "or", "our", "that", "the", "their", "then",
    "there", "these", "this", "to", "we", "with", "if", "let", "any", "every",
    "each", "all", "such",
}
DISCOVERY_GENERIC = {
    "definition", "proof", "theorem", "lemma", "proposition", "corollary",
    "remark", "example", "notation", "equation", "result",
    "weak", "strong", "which", "morphisms", "cells", "condition", "define",
    "acknowledgements",
}


def _compact_context(text, start, end, window=120):
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    snippet = text[lo:hi]
    rel_start = max(0, start - lo)
    rel_end = max(rel_start, min(len(snippet), end - lo))
    s = f"{snippet[:rel_start]}<<{snippet[rel_start:rel_end]}>>{snippet[rel_end:]}"
    s = " ".join(s.split())
    if lo > 0:
        s = "... " + s
    if hi < len(text):
        s = s + " ..."
    return s


def _normalize_discovered_term(raw):
    if not raw:
        return None
    s = raw
    s = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", s)  # remove latex commands
    s = s.replace("{", " ").replace("}", " ").replace("$", " ")
    s = re.sub(r"[^A-Za-z\-\s']", " ", s)
    s = " ".join(s.split()).lower().strip(" -'")
    if not s:
        return None

    toks = [t for t in s.split() if t]
    while toks and toks[0] in DISCOVERY_STOPWORDS:
        toks = toks[1:]
    while toks and toks[-1] in DISCOVERY_STOPWORDS:
        toks = toks[:-1]
    if not toks:
        return None

    if len(toks) > 4:
        return None
    if all(t in DISCOVERY_STOPWORDS for t in toks):
        return None
    if len(toks) == 1 and (toks[0] in DISCOVERY_STOPWORDS or toks[0] in DISCOVERY_GENERIC):
        return None
    if toks[0] in DISCOVERY_GENERIC and len(toks) == 1:
        return None
    if any(len(t) == 1 for t in toks):
        return None
    if not any(len(t) >= 4 for t in toks):
        return None

    term = " ".join(toks)
    if len(term) < 3 or len(term) > 80:
        return None
    return term


def extract_open_ner_candidates(text, max_per_entity=64):
    """Extract open-world candidate terms from definitional/marked contexts."""
    out = []
    seen = set()

    def _add(raw, source, start, end):
        term = _normalize_discovered_term(raw)
        if not term:
            return
        key = (term, source)
        if key in seen:
            return
        seen.add(key)
        out.append((term, source, _compact_context(text, start, end)))

    for m in LATEX_EMPH_RE.finditer(text):
        _add(m.group(1), "latex-emph", m.start(1), m.end(1))
        if len(out) >= max_per_entity:
            return out

    for m in CALLED_AS_RE.finditer(text):
        _add(m.group(1), "called-as", m.start(1), m.end(1))
        if len(out) >= max_per_entity:
            return out

    for m in IS_CALLED_RE.finditer(text):
        _add(m.group(1), "is-called", m.start(1), m.end(1))
        if len(out) >= max_per_entity:
            return out

    for m in DEFINED_AS_RE.finditer(text):
        _add(m.group(1), "defined-as", m.start(1), m.end(1))
        if len(out) >= max_per_entity:
            return out

    for m in DEFINITION_OF_RE.finditer(text):
        _add(m.group(1), "definition-of", m.start(1), m.end(1))
        if len(out) >= max_per_entity:
            return out

    for m in LATEX_DEFINITION_BLOCK_RE.finditer(text):
        block = m.group(1)
        if not block:
            continue
        for sm in DEF_BLOCK_SUBJECT_RE.finditer(block):
            abs_start = m.start(1) + sm.start(1)
            abs_end = m.start(1) + sm.end(1)
            _add(sm.group(1), "definition-block-subject", abs_start, abs_end)
            if len(out) >= max_per_entity:
                return out

    return out


def load_ner_kernel(path):
    """Load NER kernel terms from TSV.

    Returns (single_terms_set, multi_index, multi_count).
    multi_index maps first content word -> [(term_lower, term_original, canon_id)].
    """
    singles = {}      # term_lower -> (term_orig, canon_id)
    multi_index = {}   # first_content_word -> [(term_lower, term_orig, canon_id)]
    multi_count = 0
    skip_prefixes = ("$", "(", "\"", "-")

    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            term_lower = parts[0].strip()
            term_orig = parts[1].strip()
            canon = parts[3].strip() if len(parts) > 3 else term_lower

            if not term_lower or any(term_lower.startswith(p) for p in skip_prefixes):
                continue
            if len(term_lower) < 3:
                continue

            words = term_lower.split()
            if len(words) == 1:
                singles[term_lower] = (term_orig, canon)
            else:
                first_key = None
                for w in words:
                    if len(w) >= 3:
                        first_key = w
                        break
                if first_key is None:
                    first_key = words[0]
                if first_key not in multi_index:
                    multi_index[first_key] = []
                multi_index[first_key].append((term_lower, term_orig, canon))
                multi_count += 1

    return singles, multi_index, multi_count


def spot_terms_entity(text, singles, multi_index):
    """Spot NER terms in entity text.

    Returns list of {term, canon, source} dicts.
    """
    text_lower = text.lower()
    words = text_lower.split()
    hits = {}  # term_lower -> (term_orig, canon)

    # Single-word matches
    for w in set(words):
        clean = w.strip(".,;:!?()[]\"'")
        if clean in singles:
            hits[clean] = singles[clean]

    # Multi-word matches via inverted index
    for w in set(words):
        clean = w.strip(".,;:!?()[]\"'")
        if clean in multi_index:
            for term_lower, term_orig, canon in multi_index[clean]:
                if term_lower not in hits and term_lower in text_lower:
                    hits[term_lower] = (term_orig, canon)

    return [{"term": orig, "term_lower": tl, "canon": canon}
            for tl, (orig, canon) in sorted(hits.items())]


def detect_scopes_entity(entity_id, text):
    """Detect scope openers in text, returning futon4-compatible hyperedges.

    Each scope record is a hyperedge with:
      hx/type  — metatheory type (bind/*, quant/*, assume/*, constrain/*)
      hx/ends  — endpoints with roles (entity, symbol, type/condition)
      hx/content — match text and position
      hx/labels — tags for filtering
    """
    scopes = []
    scope_idx = 0

    for stype, pattern in SCOPE_REGEXES:
        for m in re.finditer(pattern, text):
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1

            ends = [{"role": "entity", "ident": entity_id}]

            if stype == "let-binding":
                # Groups: symbol, be/denote, type description
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "type", "text": m.group(3).strip()[:80]})

            elif stype == "define":
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "value", "text": m.group(3).strip()[:80]})

            elif stype == "assume":
                ends.append({"role": "condition",
                             "latex": m.group(3).strip()})

            elif stype == "consider":
                obj = (m.group(2) or "").strip()
                if obj:
                    ends.append({"role": "object", "text": obj[:80]})

            elif stype == "for-any":
                ends.append({"role": "quantifier", "text": m.group(1)})
                ends.append({"role": "symbol", "latex": m.group(2).strip()})

            elif stype == "where-binding":
                ends.append({"role": "symbol", "latex": m.group(1).strip()})
                ends.append({"role": "description",
                             "text": m.group(3).strip()[:80]})

            elif stype == "set-notation":
                ends.append({"role": "membership", "latex": m.group(1).strip()})

            meta_type = CLASSICAL_TO_METATHEORY.get(stype, f"scope/{stype}")
            scopes.append({
                "hx/id": scope_id,
                "hx/type": meta_type,
                "hx/ends": ends,
                "hx/content": {"match": m.group()[:120],
                               "position": m.start()},
                "hx/labels": ["scope", stype],
            })

    env_scopes = _detect_environment_scopes(entity_id, text, start_idx=scope_idx)
    scopes.extend(env_scopes)
    scope_idx += len(env_scopes)

    scopes.extend(_detect_symbolic_binders(entity_id, text, start_idx=scope_idx))
    return scopes


def _iter_math_fragments(text):
    """Yield (fragment, absolute_position) for common LaTeX math delimiters."""
    blocked = []

    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        blocked.append((m.start(), m.end()))
        yield m.group(1), m.start(1)

    for m in re.finditer(r"\\\[(.+?)\\\]", text, re.DOTALL):
        blocked.append((m.start(), m.end()))
        yield m.group(1), m.start(1)

    for m in re.finditer(r"\$([^$\n]+?)\$", text):
        if any(a <= m.start() < b for a, b in blocked):
            continue
        yield m.group(1), m.start(1)

    for m in re.finditer(r"\\\((.+?)\\\)", text, re.DOTALL):
        yield m.group(1), m.start(1)


def _extract_bound_symbol(subscript):
    """Best-effort extraction of a bound variable from a subscript."""
    if not subscript:
        return None

    m = re.search(r"([A-Za-z](?:_[A-Za-z0-9]+)?)\s*(?:=|\\in|∈)", subscript)
    if m:
        return m.group(1)

    m = SYMBOL_RE.search(subscript)
    return m.group(0) if m else None


def _extract_integral_symbol(fragment_tail):
    """Best-effort extraction of integration variable from trailing fragment."""
    m = re.search(
        r"(?:\\,|\\;|\\!|\s)*(?:d|\\mathrm\{d\})\s*([A-Za-z](?:_[A-Za-z0-9]+)?)",
        fragment_tail,
    )
    if m:
        return m.group(1)
    return None


def _detect_symbolic_binders(entity_id, text, start_idx=0):
    """Detect binder-like symbolic operators in LaTeX math fragments."""
    scopes = []
    scope_idx = start_idx

    for fragment, frag_pos in _iter_math_fragments(text):
        for m in QUANTIFIER_CMD_RE.finditer(fragment):
            quant_cmd = m.group(1)
            symbol = m.group(2)
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1

            scope_type = "quant/universal" if quant_cmd == "forall" else "quant/existential"
            scopes.append({
                "hx/id": scope_id,
                "hx/type": scope_type,
                "hx/ends": [
                    {"role": "entity", "ident": entity_id},
                    {"role": "binder", "latex": f"\\{quant_cmd}"},
                    {"role": "symbol", "latex": symbol},
                ],
                "hx/content": {
                    "match": fragment[m.start():m.end()][:120],
                    "position": frag_pos + m.start(),
                },
                "hx/labels": ["scope", "symbolic-binder", quant_cmd],
            })

        for m in AGGREGATE_BINDER_RE.finditer(fragment):
            cmd = m.group(1)
            subscript = m.group(2) or m.group(3) or ""
            bound_symbol = _extract_bound_symbol(subscript)
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1

            ends = [
                {"role": "entity", "ident": entity_id},
                {"role": "binder", "latex": f"\\{cmd}"},
            ]
            if bound_symbol:
                ends.append({"role": "symbol", "latex": bound_symbol})
            if subscript:
                ends.append({"role": "subscript", "latex": subscript[:80]})

            scopes.append({
                "hx/id": scope_id,
                "hx/type": BINDER_TYPE_BY_COMMAND.get(cmd, "bind/operator"),
                "hx/ends": ends,
                "hx/content": {
                    "match": fragment[m.start():m.end()][:120],
                    "position": frag_pos + m.start(),
                },
                "hx/labels": ["scope", "symbolic-binder", cmd],
            })

        for m in INTEGRAL_BINDER_RE.finditer(fragment):
            tail = fragment[m.end():m.end() + 64]
            symbol = _extract_integral_symbol(tail)
            scope_id = f"{entity_id}:scope-{scope_idx:03d}"
            scope_idx += 1

            ends = [
                {"role": "entity", "ident": entity_id},
                {"role": "binder", "latex": "\\int"},
            ]
            if symbol:
                ends.append({"role": "symbol", "latex": symbol})

            scopes.append({
                "hx/id": scope_id,
                "hx/type": "bind/integral",
                "hx/ends": ends,
                "hx/content": {
                    "match": fragment[m.start():min(len(fragment), m.end() + 32)][:120],
                    "position": frag_pos + m.start(),
                },
                "hx/labels": ["scope", "symbolic-binder", "integral"],
            })

    return scopes


def _detect_environment_scopes(entity_id, text, start_idx=0):
    """Detect theorem-like environments as discourse scopes."""
    scopes = []
    scope_idx = start_idx

    for m in LATEX_ENV_OPEN_RE.finditer(text):
        env_name = m.group(1).lower()
        env_type = LATEX_ENV_STYLE.get(env_name)
        if not env_type:
            continue
        scope_id = f"{entity_id}:scope-{scope_idx:03d}"
        scope_idx += 1
        scopes.append({
            "hx/id": scope_id,
            "hx/type": env_type,
            "hx/ends": [
                {"role": "entity", "ident": entity_id},
                {"role": "environment", "name": env_name},
            ],
            "hx/content": {
                "match": m.group()[:120],
                "position": m.start(),
            },
            "hx/labels": ["scope", "environment", env_name],
        })

    for m in PROSE_ENV_HEADING_RE.finditer(text):
        label = m.group(1).lower()
        env_type = PROSE_ENV_TYPE_MAP.get(label)
        if not env_type:
            continue
        scope_id = f"{entity_id}:scope-{scope_idx:03d}"
        scope_idx += 1
        scopes.append({
            "hx/id": scope_id,
            "hx/type": env_type,
            "hx/ends": [
                {"role": "entity", "ident": entity_id},
                {"role": "environment", "name": label},
            ],
            "hx/content": {
                "match": m.group()[:120],
                "position": m.start(),
            },
            "hx/labels": ["scope", "environment", label],
        })

    return scopes


def run_stage5_ner_scopes(
    entities,
    pairs,
    ner_kernel_path,
    outdir,
    scope_detector=None,
    scope_detector_name="superpod.detect_scopes_entity",
    discover_terms=False,
    discover_terms_min_freq=3,
    discover_terms_max=2000,
    discover_terms_max_per_entity=64,
    discover_terms_eprint_dir=None,
    discover_terms_eprint_max_chars=240_000,
    discover_terms_eprint_max_tex_members=4,
):
    """Run Stage 5: NER term spotting + scope detection.

    Memory-safe: streams results directly to disk (one JSON object per line
    inside a JSON array), never accumulating all results in RAM.

    Returns stats dict.
    """
    from collections import Counter

    if scope_detector is None:
        scope_detector = detect_scopes_entity
        scope_detector_name = "superpod.detect_scopes_entity"

    singles, multi_index, multi_count = load_ner_kernel(ner_kernel_path)
    print(f"       NER kernel: {len(singles)} single + {multi_count} multi-word terms")
    print(f"       Scope detector: {scope_detector_name}")

    known_terms = None
    discovery_total = 0
    discovery_unknown = 0
    discovery_occ = Counter()
    discovery_entities = Counter()
    discovery_sources = {}
    discovery_example = {}
    discovery_eprint_ok = 0
    discovery_eprint_missing = 0
    discovery_eprint_status = Counter()
    if discover_terms:
        known_terms = set(singles.keys())
        for rows in multi_index.values():
            for term_lower, _, _ in rows:
                known_terms.add(term_lower)
        print("       Open-world term discovery: enabled "
              f"(min_freq={discover_terms_min_freq}, max={discover_terms_max})")
        if discover_terms_eprint_dir is not None:
            print("       Open-world term discovery text source: eprints "
                  f"({discover_terms_eprint_dir})")

    ner_path = outdir / "ner-terms.json"
    scope_path = outdir / "scopes.json"

    total_ner_hits = 0
    total_scopes = 0
    entities_with_ner = 0
    entities_with_scopes = 0
    stype_freq = Counter()
    n = len(entities)

    with open(ner_path, "w") as ner_f, open(scope_path, "w") as scope_f:
        ner_f.write("[\n")
        scope_f.write("[\n")

        for i, (entity, pair) in enumerate(zip(entities, pairs)):
            eid = entity["entity/id"]
            full_text = (pair.question.body_text + " " + pair.answer.body_text)

            # NER term spotting
            terms = spot_terms_entity(full_text, singles, multi_index)
            if terms:
                entities_with_ner += 1
                total_ner_hits += len(terms)

            # Scope detection
            scopes = scope_detector(eid, full_text)
            if scopes:
                entities_with_scopes += 1
                total_scopes += len(scopes)
                for s in scopes:
                    stype_freq[s["hx/type"]] += 1

            # Open-world term discovery
            if discover_terms:
                discovery_text = full_text
                if discover_terms_eprint_dir is not None:
                    eprint_text, e_meta = _load_eprint_text_for_entity(
                        eprint_dir=discover_terms_eprint_dir,
                        entity_id=eid,
                        max_chars=discover_terms_eprint_max_chars,
                        max_members=discover_terms_eprint_max_tex_members,
                    )
                    estatus = e_meta.get("status", "unknown")
                    discovery_eprint_status[estatus] += 1
                    if eprint_text:
                        discovery_text = eprint_text
                        discovery_eprint_ok += 1
                    else:
                        discovery_eprint_missing += 1
                candidates = extract_open_ner_candidates(
                    discovery_text,
                    max_per_entity=discover_terms_max_per_entity,
                )
                seen_terms = set()
                seen_pair = set()
                for term, source, context in candidates:
                    discovery_total += 1
                    if term in known_terms:
                        continue
                    discovery_unknown += 1
                    key = (term, source)
                    if key in seen_pair:
                        continue
                    seen_pair.add(key)
                    discovery_occ[term] += 1
                    if term not in discovery_sources:
                        discovery_sources[term] = Counter()
                    discovery_sources[term][source] += 1
                    if term not in discovery_example:
                        discovery_example[term] = {
                            "entity_id": eid,
                            "source": source,
                            "context": context,
                        }
                    seen_terms.add(term)
                for term in seen_terms:
                    discovery_entities[term] += 1

            # Write to disk immediately, then discard
            sep = ",\n" if i > 0 else ""
            ner_f.write(sep + json.dumps(
                {"entity_id": eid, "terms": terms, "count": len(terms)},
                ensure_ascii=False))
            scope_f.write(sep + json.dumps(
                {"entity_id": eid, "scopes": scopes, "count": len(scopes)},
                ensure_ascii=False))

            if (i + 1) % 10000 == 0 or (i + 1) == n:
                print(f"       [{i+1}/{n}] "
                      f"NER: {total_ner_hits} hits, "
                      f"scopes: {total_scopes} records")

        ner_f.write("\n]")
        scope_f.write("\n]")

    print(f"       Written {ner_path} ({os.path.getsize(ner_path) / 1e6:.1f} MB)")
    print(f"       Written {scope_path} ({os.path.getsize(scope_path) / 1e6:.1f} MB)")

    open_ner_stats = None
    if discover_terms:
        cands_path = outdir / "candidate-new-terms.jsonl"
        summary_path = outdir / "candidate-new-terms-summary.json"
        rows = []
        for term, occ in discovery_occ.items():
            entity_n = discovery_entities.get(term, 0)
            if entity_n < discover_terms_min_freq:
                continue
            src_counter = discovery_sources.get(term, Counter())
            definitional = sum(
                src_counter.get(k, 0) for k in (
                    "is-called",
                    "called-as",
                    "defined-as",
                    "definition-of",
                    "definition-block-subject",
                    "latex-emph",
                )
            )
            score = occ + 0.5 * entity_n + 0.2 * definitional
            ex = discovery_example.get(term, {})
            rows.append({
                "term_lower": term,
                "candidate_count": int(occ),
                "entity_count": int(entity_n),
                "score": round(score, 4),
                "sources": dict(src_counter),
                "example_entity_id": ex.get("entity_id"),
                "example_source": ex.get("source"),
                "example_context": ex.get("context"),
            })
        rows.sort(key=lambda r: (-r["score"], -r["entity_count"], -r["candidate_count"], r["term_lower"]))
        rows = rows[:discover_terms_max]

        with open(cands_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        open_ner_stats = {
            "enabled": True,
            "min_freq": discover_terms_min_freq,
            "max_candidates": discover_terms_max,
            "max_per_entity": discover_terms_max_per_entity,
            "eprint_mode": discover_terms_eprint_dir is not None,
            "eprint_dir": str(discover_terms_eprint_dir) if discover_terms_eprint_dir is not None else None,
            "eprint_text_used": discovery_eprint_ok,
            "eprint_text_missing": discovery_eprint_missing,
            "eprint_status_counts": dict(discovery_eprint_status),
            "total_extracted": discovery_total,
            "total_unknown_extracted": discovery_unknown,
            "unique_unknown_terms": len(discovery_occ),
            "candidates_written": len(rows),
            "output_jsonl": str(cands_path),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(open_ner_stats, f, indent=2, ensure_ascii=False)

        print(f"       Written {cands_path} ({os.path.getsize(cands_path) / 1e6:.1f} MB)")
        print(f"       Written {summary_path} ({os.path.getsize(summary_path) / 1e6:.1f} MB)")

    return {
        "ner_kernel_terms": len(singles) + multi_count,
        "scope_detector": scope_detector_name,
        "entities_processed": n,
        "total_ner_hits": total_ner_hits,
        "entities_with_ner": entities_with_ner,
        "ner_coverage": entities_with_ner / n if n else 0,
        "total_scopes": total_scopes,
        "entities_with_scopes": entities_with_scopes,
        "scope_coverage": entities_with_scopes / n if n else 0,
        "scope_type_freq": dict(stype_freq.most_common()),
        "open_ner": open_ner_stats,
    }


def _load_distinctor_mit_module():
    """Load the shared MIT heuristics from pilot-planetmath-distinctors.py."""
    return importlib.import_module("pilot-planetmath-distinctors")


def _safe_arxiv_id(arxiv_id):
    s = arxiv_id.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _read_tex_members_from_tar(path, max_chars=240_000, max_members=4):
    """Extract and concatenate up to max_members .tex files from a tar archive."""
    try:
        with tarfile.open(path, "r:*") as tf:
            members = [
                m for m in tf.getmembers()
                if m.isfile() and m.name.lower().endswith(".tex")
            ]
            members.sort(key=lambda m: m.size, reverse=True)
            selected = members[:max_members]
            if not selected:
                return "", {"status": "no-tex-members", "path": str(path)}

            chunks = []
            member_names = []
            total_chars = 0
            for m in selected:
                remaining = max_chars - total_chars
                if remaining <= 0:
                    break
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                raw = fh.read(max(4096, remaining * 2))
                text = raw.decode("utf-8", errors="ignore")
                if len(text) > remaining:
                    text = text[:remaining]
                if not text.strip():
                    continue
                chunks.append(text)
                member_names.append(m.name)
                total_chars += len(text)
            joined = "\n\n".join(chunks)
            return joined[:max_chars], {
                "status": "ok",
                "path": str(path),
                "members": member_names,
            }
    except (tarfile.TarError, OSError, EOFError) as exc:
        return "", {"status": "tar-read-error", "path": str(path), "error": str(exc)}


def _load_eprint_text_for_entity(eprint_dir, entity_id, max_chars=240_000, max_members=4):
    """Load TeX text for arXiv entity id (arxiv-<id>) if available."""
    if not entity_id.startswith("arxiv-"):
        return None, {"status": "non-arxiv-entity"}
    arxiv_id = entity_id[len("arxiv-"):]
    sid = _safe_arxiv_id(arxiv_id)
    cands = [p for p in eprint_dir.glob(f"{sid}*") if p.is_file()]
    if not cands:
        return None, {"status": "missing", "id": arxiv_id}

    def _pri(path):
        name = path.name.lower()
        if name.endswith(".tex"):
            return 0
        if name.endswith(".tar.gz"):
            return 1
        if name.endswith(".tar"):
            return 2
        if name.endswith(".bin"):
            return 3
        return 9

    for path in sorted(cands, key=_pri):
        name = path.name.lower()
        if name.endswith(".tex"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                return text[:max_chars], {"status": "ok", "path": str(path), "members": [path.name]}
            except OSError:
                continue
        if name.endswith(".tar.gz") or name.endswith(".tar"):
            text, meta = _read_tex_members_from_tar(path, max_chars=max_chars, max_members=max_members)
            if text:
                return text, meta
            continue
        if name.endswith(".bin"):
            # Some payloads are mislabeled; try tar decode first, then plain text.
            text, meta = _read_tex_members_from_tar(path, max_chars=max_chars, max_members=max_members)
            if text:
                return text, meta
            try:
                raw = path.read_bytes()[: max(8192, max_chars * 2)]
                guess = raw.decode("utf-8", errors="ignore")
                if "\\documentclass" in guess or "\\begin{" in guess or "$" in guess:
                    return guess[:max_chars], {"status": "ok", "path": str(path), "members": [path.name]}
            except OSError:
                continue

    return None, {"status": "unusable", "id": arxiv_id}


def run_stage5b_distinctor_mit(
    entities,
    pairs,
    outdir,
    scope_detector,
    scope_detector_name,
    entity_limit=0,
    max_hits=500,
    seed=13,
    eprint_dir=None,
    eprint_max_chars=240_000,
    eprint_max_tex_members=4,
):
    """Run distinctor/MIT pilot on entity text using binder-aware scopes."""
    mod = _load_distinctor_mit_module()
    n_total = len(entities)
    indices = list(range(n_total))
    sampled = False
    if entity_limit and entity_limit > 0 and n_total > entity_limit:
        sampled = True
        rnd = random.Random(seed)
        indices = sorted(rnd.sample(indices, entity_limit))

    n_eval = len(indices)
    pair_events = Counter()
    unresolved_events = Counter()
    explicit_eq_events = Counter()
    explicit_neq_events = Counter()
    scope_type_freq = Counter()
    status_counts = Counter()
    entries_with_binder_scopes = 0
    total_binder_scopes = 0
    eprint_text_used = 0
    eprint_text_missing = 0
    eprint_status = Counter()
    hits = []

    hits_path = outdir / "distinctor-mit-hits.jsonl"
    summary_path = outdir / "distinctor-mit-summary.json"
    findings_json_path = outdir / "distinctor-mit-findings.json"
    findings_md_path = outdir / "distinctor-mit-findings.md"
    use_eprints = bool(eprint_dir)

    for i, idx in enumerate(indices, start=1):
        entity = entities[idx]
        pair = pairs[idx]
        eid = entity["entity/id"]
        full_text = (pair.question.body_text + " " + pair.answer.body_text)
        if use_eprints:
            eprint_text, e_meta = _load_eprint_text_for_entity(
                eprint_dir=eprint_dir,
                entity_id=eid,
                max_chars=eprint_max_chars,
                max_members=eprint_max_tex_members,
            )
            estatus = e_meta.get("status", "unknown")
            eprint_status[estatus] += 1
            if eprint_text:
                full_text = eprint_text
                eprint_text_used += 1
            else:
                eprint_text_missing += 1

        scopes = scope_detector(eid, full_text)
        binder_scopes = [s for s in scopes if mod.is_binderish(s)]
        if binder_scopes:
            entries_with_binder_scopes += 1
        total_binder_scopes += len(binder_scopes)
        for s in binder_scopes:
            scope_type_freq[s.get("hx/type", "?")] += 1

        exprs = mod.extract_math_expressions(full_text)

        for scope in binder_scopes:
            scope_symbols = mod.extract_scope_symbols(scope)
            if len(scope_symbols) < 2:
                continue

            scope_exprs = mod._scope_expressions(scope, exprs)
            eq_pairs = set()
            neq_pairs = set()
            for ex in scope_exprs:
                ex_eq, ex_neq = mod.parse_relation_pairs(ex["latex"])
                eq_pairs.update(ex_eq)
                neq_pairs.update(ex_neq)

            for a, b in itertools.combinations(sorted(scope_symbols), 2):
                if not mod._pair_compatible(a, b):
                    continue
                sym_pair = tuple(sorted((a, b)))

                has_nontrivial_context = any(
                    mod._is_nontrivial_pair_context(sym_pair, ex["latex"])
                    for ex in scope_exprs
                )
                if (
                    not has_nontrivial_context
                    and sym_pair not in eq_pairs
                    and sym_pair not in neq_pairs
                ):
                    continue

                status = "unresolved"
                pair_events[sym_pair] += 1
                if sym_pair in neq_pairs:
                    status = "explicit-distinct"
                    explicit_neq_events[sym_pair] += 1
                elif sym_pair in eq_pairs:
                    status = "explicit-equal"
                    explicit_eq_events[sym_pair] += 1
                else:
                    unresolved_events[sym_pair] += 1
                status_counts[status] += 1

                if len(hits) < max_hits:
                    support_ex = mod._best_supporting_expr(sym_pair, scope_exprs)
                    c = scope.get("hx/content", {})
                    support_start = support_ex.get("position") if support_ex else None
                    support_end = support_ex.get("end") if support_ex else None
                    scope_start = c.get("position")
                    scope_end = c.get("end")
                    hit = {
                        "hit_id": f"{eid}:{scope.get('hx/id', '')}:{sym_pair[0]}:{sym_pair[1]}:{len(hits) + 1}",
                        "status": status,
                        "entity_id": eid,
                        "title": pair.question.title,
                        "pair": [sym_pair[0], sym_pair[1]],
                        "scope_id": scope.get("hx/id", ""),
                        "scope_type": scope.get("hx/type", ""),
                        "scope_start": scope_start,
                        "scope_end": scope_end,
                        "scope_match": (c.get("match") or "")[:140],
                        "scope_symbols": scope_symbols,
                        "support_latex": (support_ex["latex"] if support_ex else "")[:300],
                        "support_expr_start": support_start,
                        "support_expr_end": support_end,
                        "scope_context": mod._context_excerpt(full_text, scope_start, scope_end),
                        "support_context": mod._context_excerpt(full_text, support_start, support_end),
                        "has_nontrivial_context": bool(
                            support_ex and mod._is_nontrivial_pair_context(sym_pair, support_ex["latex"])
                        ),
                    }
                    hit.update(mod.assess_mit_for_hit(hit))
                    hits.append(hit)

        if i % 1000 == 0 or i == n_eval:
            status_tail = ""
            if use_eprints:
                status_tail = (
                    f" eprint_ok={eprint_text_used}"
                    f" eprint_missing={eprint_text_missing}"
                )
            print(
                f"       [MIT {i}/{n_eval}] binder_scopes={total_binder_scopes} "
                f"candidate={sum(pair_events.values())} hits={len(hits)}{status_tail}"
            )

    with open(hits_path, "w", encoding="utf-8") as f:
        for row in hits:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mit_counts = Counter(h.get("mit_label", "unclear") for h in hits)
    mit_avg_conf = (
        sum(float(h.get("mit_confidence", 0.0)) for h in hits) / len(hits)
        if hits else 0.0
    )

    summary = {
        "entity_count_total": n_total,
        "entity_count_evaluated": n_eval,
        "sampled_entities": sampled,
        "entity_limit": entity_limit,
        "seed": seed,
        "scope_detector": scope_detector_name,
        "eprint_mode": use_eprints,
        "eprint_dir": str(eprint_dir) if eprint_dir else None,
        "eprint_text_used": eprint_text_used,
        "eprint_text_missing": eprint_text_missing,
        "eprint_status_counts": dict(eprint_status),
        "entries_with_binder_scopes": entries_with_binder_scopes,
        "entry_binder_coverage": round(entries_with_binder_scopes / n_eval, 4) if n_eval else 0.0,
        "total_binder_scopes": total_binder_scopes,
        "candidate_pair_events": sum(pair_events.values()),
        "unresolved_pair_events": sum(unresolved_events.values()),
        "explicit_equal_pair_events": sum(explicit_eq_events.values()),
        "explicit_distinct_pair_events": sum(explicit_neq_events.values()),
        "top_candidate_pairs": [[a, b, n] for (a, b), n in pair_events.most_common(30)],
        "top_unresolved_pairs": [[a, b, n] for (a, b), n in unresolved_events.most_common(30)],
        "scope_type_freq": dict(scope_type_freq.most_common()),
        "hit_status_counts": dict(status_counts),
        "hits_written": len(hits),
        "mit_counts": dict(mit_counts),
        "mit_avg_confidence": round(mit_avg_conf, 4),
        "output_hits": str(hits_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    findings = {
        "generated_from": str(hits_path),
        "total_hits": len(hits),
        "mit_counts": dict(mit_counts),
        "mit_avg_confidence": round(mit_avg_conf, 4),
        "top_likely_distinctor": [
            h for h in hits if h.get("mit_label") == "likely-distinctor"
        ][:120],
        "top_unclear": [
            h for h in hits if h.get("mit_label") == "unclear"
        ][:120],
    }
    with open(findings_json_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Distinctor MIT Findings",
        "",
        f"- entities_evaluated: {n_eval}",
        f"- sampled: {sampled}",
        f"- candidate_pair_events: {summary['candidate_pair_events']}",
        f"- unresolved_pair_events: {summary['unresolved_pair_events']}",
        f"- explicit_equal_pair_events: {summary['explicit_equal_pair_events']}",
        f"- explicit_distinct_pair_events: {summary['explicit_distinct_pair_events']}",
        f"- benign-cooccurrence: {mit_counts.get('benign-cooccurrence', 0)}",
        f"- likely-distinctor: {mit_counts.get('likely-distinctor', 0)}",
        f"- unclear: {mit_counts.get('unclear', 0)}",
        f"- avg_confidence: {round(mit_avg_conf, 4)}",
        "",
        "## Likely Distinctor Candidates",
    ]
    likely_rows = [h for h in hits if h.get("mit_label") == "likely-distinctor"]
    if likely_rows:
        for i, h in enumerate(likely_rows[:80], start=1):
            md_lines.append(
                f"{i}. {h.get('entity_id', '')} "
                f"pair=({', '.join(h.get('pair', []))}) "
                f"scope={h.get('scope_type', '')} "
                f"conf={h.get('mit_confidence', 0.0)} "
                f"rationale={','.join(h.get('mit_rationale', []))}"
            )
    else:
        md_lines.append("- none")
    md_lines.append("")
    md_lines.append("## Unclear Candidates")
    unclear_rows = [h for h in hits if h.get("mit_label") == "unclear"]
    if unclear_rows:
        for i, h in enumerate(unclear_rows[:80], start=1):
            md_lines.append(
                f"{i}. {h.get('entity_id', '')} "
                f"pair=({', '.join(h.get('pair', []))}) "
                f"scope={h.get('scope_type', '')} "
                f"conf={h.get('mit_confidence', 0.0)} "
                f"rationale={','.join(h.get('mit_rationale', []))}"
            )
    else:
        md_lines.append("- none")
    with open(findings_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {
        **summary,
        "summary_path": str(summary_path),
        "findings_json_path": str(findings_json_path),
        "findings_md_path": str(findings_md_path),
    }


# --- Stage 3: LLM pattern tagging ---

def build_pattern_prompt(question_title, question_text, answer_text):
    """Build the prompt for pattern tagging."""
    pattern_list = "\n".join(
        f"  {i+1}. {name}: {desc}" for i, (name, desc) in enumerate(PATTERNS)
    )

    # Truncate to fit context
    q = question_text[:800]
    a = answer_text[:1200]

    return f"""You are a mathematics education researcher analysing proof strategies.

Task: identify which informal reasoning patterns the ANSWER uses.

Here are the 25 patterns to check:
{pattern_list}

Reply with ONLY a JSON list of pattern numbers (1-25) that the answer clearly uses. Example: [3, 10, 17]
If none apply clearly, reply: []

Now analyze this Q&A from math.stackexchange:
Question: {question_title}
{q}

Answer:
{a}"""


def _create_llm_pipeline(model_name, batch_size=8):
    """Create a reusable LLM pipeline for text generation.

    Returns (pipe, tokenizer) tuple. The pipeline is created without
    default max_new_tokens — callers pass it per-call.
    """
    from transformers import pipeline, AutoTokenizer
    import torch

    print(f"       Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        do_sample=False,
        batch_size=batch_size,
    )
    return pipe, tokenizer


def tag_patterns_llm_batch(pairs, model_name=None, batch_size=8, device="cuda",
                           pipe=None, tokenizer=None, entry_ids=None):
    """Tag QA pairs with reasoning patterns using a local LLM.

    Uses transformers pipeline for batched inference. If pipe/tokenizer
    are provided, reuses them; otherwise creates a new pipeline.
    """
    if pipe is None:
        pipe, tokenizer = _create_llm_pipeline(model_name, batch_size)

    results = []
    total = len(pairs)
    if entry_ids is not None and len(entry_ids) != total:
        raise ValueError(f"entry_ids length ({len(entry_ids)}) != pairs length ({total})")

    for start in range(0, total, batch_size):
        batch = pairs[start:start + batch_size]
        prompts = []
        for pair in batch:
            prompt = build_pattern_prompt(
                pair.question.title,
                pair.question.body_text,
                pair.answer.body_text,
            )
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted)

        outputs = pipe(prompts, return_full_text=False, max_new_tokens=64)

        for i, out in enumerate(outputs):
            text = out[0]["generated_text"].strip()
            pattern_ids = _parse_pattern_response(text)
            entry_id = (entry_ids[start + i]
                        if entry_ids is not None
                        else f"se-math-{batch[i].question.id}")
            results.append({
                "entry_id": entry_id,
                "patterns": [PATTERN_NAMES[pid - 1] for pid in pattern_ids
                             if 1 <= pid <= 25],
                "raw": text,
            })

        done = min(start + batch_size, total)
        if done % 1000 < batch_size or done == total:
            elapsed = time.time() - _llm_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"       [{done}/{total}] {rate:.0f} pairs/s, "
                  f"ETA {eta/60:.0f} min")

    return results


def _parse_pattern_response(text):
    """Extract list of integers from LLM response."""
    import re
    # Try to find a JSON list
    match = re.search(r"\[[\d,\s]*\]", text)
    if match:
        try:
            nums = json.loads(match.group())
            return [int(n) for n in nums if isinstance(n, (int, float))]
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: find all integers
    nums = re.findall(r"\b(\d{1,2})\b", text)
    return [int(n) for n in nums if 1 <= int(n) <= 25]


def _parse_json_object_response(text):
    """Extract a JSON object from LLM response text."""
    start = text.find('{')
    if start == -1:
        return {"raw": text, "parse_error": "no JSON object found"}

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return {"raw": text[start:i+1], "parse_error": "invalid JSON"}

    return {"raw": text, "parse_error": "unclosed JSON object"}


def _parse_json_array_response(text):
    """Extract a JSON array from LLM response text."""
    start = text.find('[')
    if start == -1:
        return []

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:i+1])
                    return result if isinstance(result, list) else []
                except json.JSONDecodeError:
                    return []

    return []


# --- Stage 6: Reverse morphogenesis LLM inference ---

def run_reverse_morphogenesis_llm_batch(pairs, entities, pipe, tokenizer,
                                        batch_size=4):
    """Run reverse morphogenesis analysis using a local LLM.

    For each QA pair, generates the S←Q←A prompt, runs inference, and
    parses the JSON response. Smaller batch_size than Stage 3 because
    prompts and responses are both larger.

    Returns list of result dicts.
    """
    results = []
    total = len(pairs)
    t_start = time.time()

    for start in range(0, total, batch_size):
        batch_pairs = pairs[start:start + batch_size]
        batch_entities = entities[start:start + batch_size]
        prompts = []
        for pair in batch_pairs:
            prompt = build_reverse_morphogenesis_prompt(
                pair.question.title,
                pair.question.body_text,
                pair.answer.body_text,
            )
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted)

        outputs = pipe(prompts, return_full_text=False, max_new_tokens=512)

        for i, out in enumerate(outputs):
            text = out[0]["generated_text"].strip()
            parsed = _parse_json_object_response(text)
            results.append({
                "entity_id": batch_entities[i]["entity/id"],
                "question_id": batch_pairs[i].question.id,
                "analysis": parsed,
                "raw": text,
            })

        done = min(start + batch_size, total)
        if done % 500 < batch_size or done == total:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"       [{done}/{total}] {rate:.0f} pairs/s, "
                  f"ETA {eta/60:.0f} min")

    return results


# --- Stage 7: Thread performative LLM classification ---

def classify_thread_performatives_llm_batch(diagrams, pipe, tokenizer,
                                            batch_size=2):
    """Run LLM-based performative classification on thread wiring diagrams.

    Enhances classical detection: for each diagram, generates a prompt,
    runs the LLM, parses the JSON response, and merges LLM-classified
    edge types back into the diagram (overriding classical/structural).

    Smaller batch_size because thread prompts can be very long.

    Returns count of diagrams where LLM provided new classifications.
    """
    total = len(diagrams)
    llm_enhanced = 0
    t_start = time.time()

    for start in range(0, total, batch_size):
        batch = diagrams[start:start + batch_size]
        prompts = []
        for diagram in batch:
            prompt = build_thread_performative_prompt(diagram)
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted)

        outputs = pipe(prompts, return_full_text=False, max_new_tokens=512)

        for i, out in enumerate(outputs):
            text = out[0]["generated_text"].strip()
            llm_edges = _parse_json_array_response(text)
            if llm_edges:
                merge_llm_edges(batch[i], llm_edges)
                llm_enhanced += 1

        done = min(start + batch_size, total)
        if done % 100 < batch_size or done == total:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"       [{done}/{total}] {rate:.0f} threads/s, "
                  f"ETA {eta/60:.0f} min, {llm_enhanced} enhanced")

    return llm_enhanced


# --- Stage 7 (CT-backed): Thread wiring with categorical annotation ---

def run_stage7_ct_wiring(threads, reference, singles, multi_index,
                          site="", outdir=None):
    """Stage 7: CT-backed thread wiring assembly.

    Replaces process_threads_to_diagrams() when CT reference is available.
    Streams results to disk (one thread at a time), never accumulating all
    wirings in RAM.

    Returns (stats_dict, output_path).
    """
    aw, _ = _load_ct_modules()

    wiring_path = outdir / "thread-wiring-ct.json"
    wiring_jsonl_path = outdir / "thread-wiring-ct.jsonl"
    total = len(threads)
    stats = Counter()
    cat_types = Counter()
    iatc_types = Counter()

    with open(wiring_path, "w") as f, open(wiring_jsonl_path, "w") as fj:
        f.write("[\n")
        for i, thread in enumerate(threads):
            thread_dict = aw.sethread_to_dict(thread, site=site)
            wiring = aw.build_thread_graph(thread_dict, reference, singles, multi_index)

            # Stream to JSON array
            if i > 0:
                f.write(",\n")
            json.dump(wiring, f, ensure_ascii=False)

            # Stream to JSONL (one record per line — fast multiprocessing)
            fj.write(json.dumps(wiring, ensure_ascii=False))
            fj.write("\n")

            # Accumulate stats
            s = wiring["stats"]
            stats["threads_processed"] += 1
            stats["total_nodes"] += s["n_nodes"]
            stats["total_edges"] += s["n_edges"]
            stats["n_categorical"] += s["n_categorical"]
            stats["n_diagrams"] += s["n_diagrams"]
            stats["n_port_matches"] += s["n_port_matches"]
            for t, c in s.get("categorical_types", {}).items():
                cat_types[t] += c
            for t, c in s.get("iatc_types", {}).items():
                iatc_types[t] += c

            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"       [{i+1}/{total}] nodes={stats['total_nodes']} "
                      f"edges={stats['total_edges']} "
                      f"cat={stats['n_categorical']} "
                      f"ports={stats['n_port_matches']}")

        f.write("\n]")

    # Compute derived stats
    n_threads = stats["threads_processed"]
    n_edges = stats["total_edges"]
    default_edges = iatc_types.get("assert", 0) + iatc_types.get("clarify", 0)
    classical_edges = n_edges - default_edges

    result_stats = dict(stats)
    result_stats.update({
        "categorical_types": dict(cat_types.most_common()),
        "iatc_types": dict(iatc_types.most_common()),
        "performative_freq": dict(iatc_types.most_common()),
        "unique_performatives": len(iatc_types),
        "classical_edges": classical_edges,
        "structural_edges": default_edges,
        "classical_edge_rate": classical_edges / n_edges if n_edges else 0,
        "threads_with_classical": n_threads,  # all threads get IATC in new pipeline
        "ct_backed": True,
    })

    return result_stats, wiring_path


# ---------------------------------------------------------------------------
# Stage 8: Expression surface parsing (CPU)
# ---------------------------------------------------------------------------

def run_stage8_expression_surfaces(threads, outdir):
    """Stage 8: Parse all LaTeX expressions in thread bodies to s-expressions.

    CPU-only, streaming one thread at a time. Extracts $...$ and $$...$$
    from question/answer/comment HTML, parses each to a typed s-exp.

    Returns (stats_dict, output_path).
    """
    out_path = outdir / "expression-surfaces.json"
    total = len(threads)
    n_exprs = 0
    n_parsed = 0
    n_fallback = 0

    with open(out_path, "w") as f:
        f.write("[\n")
        for i, thread in enumerate(threads):
            # Gather all HTML bodies from the thread
            bodies = []
            q = thread.question
            bodies.append((f"q-{q.id}", q.body))
            for ans in thread.answers:
                bodies.append((f"a-{ans.id}", ans.body))
            for post_id, comments in thread.comments.items():
                for c in comments:
                    bodies.append((f"c-{c.id}", c.text))

            # Parse expressions from each body
            thread_exprs = []
            for post_id, html in bodies:
                if not html:
                    continue
                parsed = parse_all(html)
                for p in parsed:
                    is_fallback = (p["sexp"].startswith('"')
                                   and p["sexp"].endswith('"'))
                    thread_exprs.append({
                        "post_id": post_id,
                        "latex": p["latex"],
                        "sexp": p["sexp"],
                        "display": p["display"],
                        "fallback": is_fallback,
                    })
                    n_exprs += 1
                    if is_fallback:
                        n_fallback += 1
                    else:
                        n_parsed += 1

            record = {
                "thread_id": q.id,
                "n_expressions": len(thread_exprs),
                "expressions": thread_exprs,
            }

            if i > 0:
                f.write(",\n")
            json.dump(record, f, ensure_ascii=False)

            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"       [{i+1}/{total}] exprs={n_exprs} "
                      f"parsed={n_parsed} fallback={n_fallback}")

        f.write("\n]")

    stats = {
        "threads_processed": total,
        "total_expressions": n_exprs,
        "parsed": n_parsed,
        "fallback": n_fallback,
        "parse_rate": n_parsed / n_exprs if n_exprs else 0,
    }
    return stats, out_path


def _load_stage5_entity_map(path, key):
    """Load Stage 5 JSON array file into entity_id -> list payload map."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        eid = row.get("entity_id")
        if not eid:
            continue
        payload = row.get(key, [])
        out[eid] = payload if isinstance(payload, list) else []
    return out


def _extract_math_snippets_for_hypergraph(text, max_exprs=160, max_latex_len=512):
    """Extract unique LaTeX snippets from text for paper-level hypergraphs."""
    out = []
    seen = set()
    blocked = []

    def _add(raw, display, pos):
        tex = " ".join((raw or "").strip().split())
        if not tex or len(tex) > max_latex_len:
            return False
        if tex in seen:
            return False
        seen.add(tex)
        out.append({
            "latex": tex,
            "display": bool(display),
            "position": int(pos),
        })
        return len(out) >= max_exprs

    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        blocked.append((m.start(), m.end()))
        if _add(m.group(1), True, m.start(1)):
            return out

    for m in re.finditer(r"\\\[(.+?)\\\]", text, re.DOTALL):
        blocked.append((m.start(), m.end()))
        if _add(m.group(1), True, m.start(1)):
            return out

    for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", text):
        if any(a <= m.start() < b for a, b in blocked):
            continue
        if _add(m.group(1), False, m.start(1)):
            return out

    for m in re.finditer(r"\\\((.+?)\\\)", text, re.DOTALL):
        if _add(m.group(1), False, m.start(1)):
            return out

    out.sort(key=lambda r: r["position"])
    return out


def _build_arxiv_paper_hypergraph(entity, pair, terms, scopes, text, max_exprs=160):
    """Build a paper-level typed hypergraph using Stage 5 outputs + expressions."""
    eid = entity["entity/id"]
    post_id = f"paper:{eid}"
    nodes = {}
    edges = []

    def _add_node(nid, ntype, subtype, attrs):
        if nid not in nodes:
            nodes[nid] = {
                "id": nid,
                "type": ntype,
                "subtype": subtype,
                "attrs": attrs,
            }

    _add_node(post_id, "post", "paper", {
        "title": pair.question.title,
        "tags": pair.tags or [],
        "date": pair.question.creation_date,
    })

    # NER terms -> term nodes + mention edges
    seen_term_edges = set()
    for t in terms:
        if not isinstance(t, dict):
            continue
        canon = (t.get("canon") or t.get("term_lower") or t.get("term") or "").strip().lower()
        surface = (t.get("term") or canon).strip()
        if not canon:
            continue
        term_id = f"term:{canon}"
        _add_node(term_id, "term", canon, {"surface_forms": set()})
        nodes[term_id]["attrs"]["surface_forms"].add(surface)
        if term_id not in seen_term_edges:
            edges.append({
                "type": "mention",
                "ends": [post_id, term_id],
                "attrs": {"surface": surface},
            })
            seen_term_edges.add(term_id)

    # Scope records -> scope nodes + scope edges
    for idx, s in enumerate(scopes):
        if not isinstance(s, dict):
            continue
        scope_id = s.get("hx/id") or f"{eid}:scope-{idx:04d}"
        scope_type = s.get("hx/type", "scope/unknown")
        content = s.get("hx/content") or {}
        _add_node(scope_id, "scope", scope_type, {
            "match": content.get("match", ""),
            "position": content.get("position"),
            "end": content.get("end"),
            "labels": s.get("hx/labels", []),
        })
        edges.append({
            "type": "scope",
            "ends": [scope_id, post_id],
            "attrs": {"binding_type": scope_type},
        })

    # Expressions -> expression nodes + surface edges
    expr_rows = _extract_math_snippets_for_hypergraph(text, max_exprs=max_exprs)
    for ex in expr_rows:
        latex = ex["latex"]
        sexp = sexp_parse(latex)
        h = hashlib.sha1(latex.encode("utf-8")).hexdigest()[:12]
        expr_id = f"expr:{eid}:{h}"
        _add_node(expr_id, "expression", "math", {
            "latex": latex,
            "sexp": sexp,
            "display": ex["display"],
        })
        edges.append({
            "type": "surface",
            "ends": [expr_id, post_id],
            "attrs": {"position": ex["position"]},
        })

    # JSON-serializable attrs
    for n in nodes.values():
        for k, v in list(n.get("attrs", {}).items()):
            if isinstance(v, set):
                n["attrs"][k] = sorted(v)

    node_list = sorted(nodes.values(), key=lambda n: n["id"])
    edge_types = Counter(e["type"] for e in edges)
    return {
        "thread_id": eid,
        "nodes": node_list,
        "edges": edges,
        "meta": {
            "n_nodes": len(node_list),
            "n_edges": len(edges),
            "n_posts": sum(1 for n in node_list if n["type"] == "post"),
            "n_terms": sum(1 for n in node_list if n["type"] == "term"),
            "n_expressions": sum(1 for n in node_list if n["type"] == "expression"),
            "n_scopes": sum(1 for n in node_list if n["type"] == "scope"),
            "edge_types": dict(edge_types),
        },
    }


def run_stage9a_arxiv_paper_hypergraphs(
    entities,
    pairs,
    outdir,
    paper_hg_eprint_dir=None,
    paper_hg_text_max_chars=240_000,
    paper_hg_max_tex_members=4,
    paper_hg_max_expressions=160,
):
    """Stage 9a (arXiv): assemble paper-level hypergraphs from Stage 5 outputs."""
    total = len(entities)
    ner_map = _load_stage5_entity_map(outdir / "ner-terms.json", "terms")
    scope_map = _load_stage5_entity_map(outdir / "scopes.json", "scopes")
    if not ner_map:
        print("       Warning: ner-terms.json missing or empty; term nodes may be sparse")
    if not scope_map:
        print("       Warning: scopes.json missing or empty; scope nodes may be sparse")

    out_path = outdir / "hypergraphs.json"
    n_hg = 0
    n_nodes = 0
    n_edges = 0
    n_terms = 0
    n_scopes = 0
    n_exprs = 0
    edge_type_freq = Counter()
    eprint_ok = 0
    eprint_missing = 0
    eprint_status = Counter()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, (entity, pair) in enumerate(zip(entities, pairs)):
            eid = entity["entity/id"]
            full_text = pair.question.body_text + " " + pair.answer.body_text
            if paper_hg_eprint_dir is not None:
                eprint_text, e_meta = _load_eprint_text_for_entity(
                    eprint_dir=paper_hg_eprint_dir,
                    entity_id=eid,
                    max_chars=paper_hg_text_max_chars,
                    max_members=paper_hg_max_tex_members,
                )
                estatus = e_meta.get("status", "unknown")
                eprint_status[estatus] += 1
                if eprint_text:
                    full_text = eprint_text
                    eprint_ok += 1
                else:
                    eprint_missing += 1
            elif paper_hg_text_max_chars and len(full_text) > paper_hg_text_max_chars:
                full_text = full_text[:paper_hg_text_max_chars]

            hg = _build_arxiv_paper_hypergraph(
                entity=entity,
                pair=pair,
                terms=ner_map.get(eid, []),
                scopes=scope_map.get(eid, []),
                text=full_text,
                max_exprs=paper_hg_max_expressions,
            )

            if i > 0:
                f.write(",\n")
            json.dump(hg, f, ensure_ascii=False)

            n_hg += 1
            n_nodes += hg["meta"]["n_nodes"]
            n_edges += hg["meta"]["n_edges"]
            n_terms += hg["meta"]["n_terms"]
            n_scopes += hg["meta"]["n_scopes"]
            n_exprs += hg["meta"]["n_expressions"]
            edge_type_freq.update(hg["meta"].get("edge_types", {}))

            if (i + 1) % 500 == 0 or (i + 1) == total:
                extra = ""
                if paper_hg_eprint_dir is not None:
                    extra = f" eprint_ok={eprint_ok} missing={eprint_missing}"
                print(f"       [{i+1}/{total}] hypergraphs={n_hg} "
                      f"nodes={n_nodes} edges={n_edges}{extra}")

        f.write("\n]")

    stats = {
        "threads_processed": total,  # compatibility with downstream health gates
        "papers_processed": total,
        "hypergraphs_produced": n_hg,
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "avg_nodes": n_nodes / n_hg if n_hg else 0,
        "avg_edges": n_edges / n_hg if n_hg else 0,
        "total_terms": n_terms,
        "total_scopes": n_scopes,
        "total_expressions": n_exprs,
        "edge_type_freq": dict(edge_type_freq.most_common()),
        "paper_text_source": ("eprints" if paper_hg_eprint_dir is not None else "metadata"),
        "paper_hg_max_expressions": paper_hg_max_expressions,
        "paper_hg_text_max_chars": paper_hg_text_max_chars,
        "eprint_text_used": eprint_ok,
        "eprint_text_missing": eprint_missing,
        "eprint_status_counts": dict(eprint_status),
    }
    return stats, out_path


# ---------------------------------------------------------------------------
# Stage 9a: Hypergraph assembly (CPU)
# ---------------------------------------------------------------------------

def run_stage9a_hypergraphs(threads, wiring_path, surfaces_path, outdir):
    """Stage 9a: Assemble typed hypergraphs from wiring + expression surfaces.

    Reads Stage 7 wiring and Stage 8 expression surfaces, combines into
    a per-thread hypergraph with nodes (post, term, expression, scope)
    and edges (iatc, mention, discourse, scope, surface, categorical).

    Returns (stats_dict, output_path).
    """
    # Load wiring dicts (streamed from Stage 7)
    with open(wiring_path) as f:
        wirings = json.load(f)

    # Index wiring by thread_id
    wiring_by_id = {}
    for w in wirings:
        tid = w.get("thread_id") or w.get("question_id")
        if tid:
            wiring_by_id[tid] = w

    # Load expression surfaces (from Stage 8)
    surfaces_by_id = {}
    if surfaces_path and Path(surfaces_path).exists():
        with open(surfaces_path) as f:
            surfaces = json.load(f)
        for s in surfaces:
            surfaces_by_id[s["thread_id"]] = s

    out_path = outdir / "hypergraphs.json"
    total = len(threads)
    n_nodes = 0
    n_edges = 0
    n_hg = 0

    with open(out_path, "w") as f:
        f.write("[\n")
        for i, thread in enumerate(threads):
            qid = thread.question.id
            wiring = wiring_by_id.get(qid)
            if not wiring:
                continue

            # Build the raw dict expected by assemble_hypergraph
            raw = _sethread_to_raw(thread)
            # Extract wiring nodes/edges
            w_nodes = wiring.get("nodes", [])
            w_edges = wiring.get("edges", [])
            wiring_dict = {"nodes": w_nodes, "edges": w_edges}

            hg = assemble_hypergraph(raw, wiring_dict)

            if i > 0:
                f.write(",\n")
            json.dump(hg, f, ensure_ascii=False)

            n_nodes += hg["meta"]["n_nodes"]
            n_edges += hg["meta"]["n_edges"]
            n_hg += 1

            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"       [{i+1}/{total}] hypergraphs={n_hg} "
                      f"nodes={n_nodes} edges={n_edges}")

        f.write("\n]")

    stats = {
        "threads_processed": total,
        "hypergraphs_produced": n_hg,
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "avg_nodes": n_nodes / n_hg if n_hg else 0,
        "avg_edges": n_edges / n_hg if n_hg else 0,
    }
    return stats, out_path


def _sethread_to_raw(thread) -> dict:
    """Convert an SEThread dataclass to the raw dict format expected by hypergraph.assemble()."""
    q = thread.question
    raw = {
        "question": {
            "id": q.id,
            "title": q.title,
            "score": q.score,
            "tags": q.tags,
            "body_html": q.body,
        },
        "answers": [],
        "comments_q": [],
        "comments_a": {},
    }
    for ans in thread.answers:
        raw["answers"].append({
            "id": ans.id,
            "score": ans.score,
            "is_accepted": (q.accepted_answer_id == ans.id),
            "body_html": ans.body,
        })
    # Comments on question
    for c in thread.comments.get(q.id, []):
        raw["comments_q"].append({
            "id": c.id,
            "score": c.score,
            "text": c.text,
        })
    # Comments on answers
    for ans in thread.answers:
        ans_comments = thread.comments.get(ans.id, [])
        if ans_comments:
            raw["comments_a"][str(ans.id)] = [
                {"id": c.id, "score": c.score, "text": c.text}
                for c in ans_comments
            ]
    return raw


# ---------------------------------------------------------------------------
# Stage 9b: Hypergraph embedding via R-GCN (GPU)
# ---------------------------------------------------------------------------

def run_stage9b_graph_embedding(hg_path, outdir, embed_dim=128, hidden_dim=128,
                                 n_layers=2, epochs=50, batch_size=512,
                                 device=None, num_workers=4):
    """Stage 9b: Train R-GCN on thread hypergraphs, produce embeddings.

    GPU-accelerated contrastive learning on the typed hypergraph structure.
    Uses a .pt tensor cache to avoid re-parsing multi-GB JSON on rerun.

    Returns (stats_dict, embeddings_path, model_path, thread_ids).
    """
    if train_gnn is None:
        raise RuntimeError(
            f"Graph embedding dependencies unavailable: {_GRAPH_EMBED_IMPORT_ERROR}"
        )
    tensor_cache = outdir / "hypergraph-tensors.pt"

    # If tensor cache exists, skip JSON load entirely
    if tensor_cache.exists():
        print(f"       Tensor cache found: {tensor_cache}")
        from futon6.graph_embed import load_tensor_cache
        _, cached_thread_ids = load_tensor_cache(str(tensor_cache))
        hypergraphs = []  # not needed — train() will load from cache
        thread_ids = cached_thread_ids
    else:
        with open(hg_path) as f:
            hypergraphs = json.load(f)
        thread_ids = [hg.get("thread_id", i) for i, hg in enumerate(hypergraphs)]

    model, embeddings, train_stats = train_gnn(
        hypergraphs, dim=embed_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, epochs=epochs, batch_size=batch_size,
        device=device, verbose=True, num_workers=num_workers,
        tensor_cache_path=str(tensor_cache))

    emb_path = outdir / "hypergraph-embeddings.npy"
    model_path = outdir / "graph-gnn-model.pt"
    ids_path = outdir / "hypergraph-thread-ids.json"

    np.save(str(emb_path), embeddings)
    save_model(model, str(model_path))
    with open(ids_path, "w") as f:
        json.dump(thread_ids, f)

    stats = {
        "n_threads": len(thread_ids),
        "n_embedded": embeddings.shape[0],
        "embed_dim": embeddings.shape[1],
        "epochs": epochs,
        "device": str(device or "auto"),
        "train_metrics": train_stats,
    }
    return stats, emb_path, model_path, thread_ids


# ---------------------------------------------------------------------------
# Stage 10: FAISS structural similarity index (CPU)
# ---------------------------------------------------------------------------

def run_stage10_faiss_index(embeddings_path, thread_ids, outdir,
                             index_type="flat"):
    """Stage 10: Build FAISS index from hypergraph embeddings.

    Returns (stats_dict, index_path).
    """
    embeddings = np.load(str(embeddings_path))

    index, ids = build_index(embeddings, thread_ids, index_type=index_type)

    index_path = outdir / "structural-similarity-index"
    save_index(index, ids, str(index_path))

    stats = {
        "n_vectors": len(ids),
        "dimension": embeddings.shape[1],
        "index_type": index_type,
        "has_faiss": True,
    }

    # Quick self-check: query a random thread
    try:
        from futon6.faiss_index import query as faiss_query
        if len(ids) >= 2:
            results = faiss_query(index, ids, embeddings[0], k=5, exclude_id=ids[0])
            stats["sample_query"] = {
                "query_thread": ids[0],
                "top_5": [r["thread_id"] for r in results],
                "top_5_sim": [round(r["similarity"], 3) for r in results],
            }
    except Exception:
        pass

    return stats, index_path


def build_ct_performative_prompt(wiring_dict):
    """Build LLM performative classification prompt from CT-backed wiring dict.

    Generates the same kind of prompt as build_thread_performative_prompt() but
    from the richer wiring dict instead of the old ThreadWiringDiagram dataclass.
    """
    lines = []
    lines.append("You are analysing a math StackExchange thread to classify "
                 "the argumentative moves between posts.\n")
    lines.append("For each edge, classify the argumentative move as one of:")
    lines.append("  assert, challenge, query, clarify, reform, exemplify, "
                 "reference, agree, retract")
    lines.append("Reply as JSON array:")
    lines.append('[{"source": "<id>", "target": "<id>", "performative": "<type>", '
                 '"reasoning": "<brief>"}]')
    lines.append("")
    lines.append("Now classify this thread payload.")
    lines.append(f"Thread: \"{wiring_dict.get('title', '')}\"")
    lines.append(f"Topic: {wiring_dict.get('topic', 'mathematics')}\n")

    lines.append("Posts:")
    for node in wiring_dict["nodes"]:
        ntype = node["type"].upper()
        nid = node["id"]
        text = ""
        # Get text from discourse or reconstruct
        if node.get("title"):
            text = node["title"][:200]
        elif node.get("text_length", 0) > 0:
            # Use port labels as proxy for content
            ports = node.get("input_ports", []) + node.get("output_ports", [])
            text = "; ".join(p.get("label", "")[:50] for p in ports[:3])
        cat_info = ""
        if node.get("categorical"):
            cat_types_str = ", ".join(c["hx/type"] for c in node["categorical"][:3])
            cat_info = f" [CT: {cat_types_str}]"
        lines.append(f"  [{nid}] ({ntype}, score={node.get('score',0)}{cat_info}): "
                     f"{text[:200]}")

    lines.append("\nEdges (current classification):")
    for edge in wiring_dict["edges"]:
        lines.append(f"  {edge['from']} → {edge['to']}: {edge['iatc']} ({edge['type']})")

    return "\n".join(lines)


# --- Stage 4: Clustering ---

def cluster_embeddings(embeddings, min_cluster_size=50, max_clusters=500):
    """Cluster embeddings using HDBSCAN (or KMeans fallback)."""
    if len(embeddings) == 0:
        print("       No embeddings to cluster (0 rows)")
        return [], 0, 0

    try:
        from sklearn.cluster import HDBSCAN
        print(f"       Using HDBSCAN (min_cluster_size={min_cluster_size})...")
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="cosine",
            n_jobs=-1,
        )
        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels) - {-1})
        n_noise = int(np.sum(labels == -1))
        print(f"       {n_clusters} clusters, {n_noise} noise points")
    except (ImportError, Exception) as e:
        print(f"       HDBSCAN failed ({e}), falling back to KMeans...")
        try:
            from sklearn.cluster import MiniBatchKMeans
            n_clusters = max(1, min(max_clusters, len(embeddings) // 100))
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=4096,
                n_init=3,
            )
            labels = kmeans.fit_predict(embeddings)
            n_noise = 0
            print(f"       {n_clusters} clusters (KMeans)")
        except (ImportError, Exception) as kmeans_err:
            # Last-resort fallback: keep pipeline running even without sklearn.
            print(f"       KMeans unavailable ({kmeans_err}); assigning one cluster")
            labels = np.zeros(len(embeddings), dtype=int)
            n_clusters = 1
            n_noise = 0

    return labels.tolist(), n_clusters, n_noise


def build_reverse_morphogenesis_prompt(question_title, question_text, answer_text):
    r"""Build the S <- Q <- A prompt for reverse morphogenesis.

    Given a Q/A pair, asks the LLM to:
    1. Identify the mathematical form (象) and desired understanding (香)
    2. Classify question quality by failure mode (wrong 象, wrong 香, wrong ←)
    3. Induce a fictitious situation S to which Q is the natural question
    4. Verify by checking that S naturally produces Q

    This is double reverse morphogenesis: S ← Q ← A.
    """
    q = question_text[:800]
    a = answer_text[:1200]

    return f"""You are a mathematics education researcher studying how questions arise from situations.

Perform reverse morphogenesis analysis for a Q&A pair.

Tasks:

1. IDENTIFY THE ← STRUCTURE:
   - 象 (form): What is the mathematical object/structure the question is about?
   - 香 (salience): What understanding does the questioner seek? What makes this worth asking?
   - ← (constraint): What constraint does the question infer — i.e., what would you need to know/prove/construct for the form to yield that understanding?

2. CLASSIFY QUESTION QUALITY:
   Rate each dimension (good/weak/broken):
   - 象 quality: Is the mathematical form well-specified?
   - 香 quality: Is the salience signal grounded (does the questioner know WHY they want to know)?
   - ← quality: Does the question actually connect form to understanding?

3. INDUCE SITUATION S (reverse morphogenesis of Q):
   Construct a concrete, vivid situation (could be fictional, pedagogical, or applied) from which this question would NATURALLY arise. The situation should make someone who encounters it think "hmm, I wonder..." and arrive at exactly this question.

4. VERIFY (← round-trip):
   Given your situation S, would a student/researcher encountering S naturally ask Q (or something equivalent)? If not, revise S.

Reply as JSON:
{{
  "xiang_form": "<the mathematical form>",
  "xiang_salience": "<what understanding is sought>",
  "arrow_constraint": "<what the question infers>",
  "quality": {{"form": "good|weak|broken", "salience": "good|weak|broken", "arrow": "good|weak|broken"}},
  "situation_S": "<the induced situation>",
  "roundtrip_check": "<does S -> Q hold? brief assessment>"
}}

Now analyze this Q&A pair from math.stackexchange:
Question: {question_title}
{q}

Answer:
{a}"""


def generate_moist_prompts(pairs, entities, outdir, stages=None, thread_diagrams=None):
    """Generate prompt files for LLM stages (Codex/Claude handoff).

    Instead of running Llama locally, writes JSONL files with one prompt per
    line, ready for batch submission to Codex, Claude, or any API.

    stages: list of stage names to generate. Default: all LLM stages.
    thread_diagrams: list of ThreadWiringDiagram objects for stage 7.
    """
    if stages is None:
        stages = ["pattern_tagging", "reverse_morphogenesis"]

    prompt_dir = outdir / "moist-prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    generated = {}

    if "pattern_tagging" in stages:
        path = prompt_dir / "stage3-pattern-tagging.jsonl"
        count = 0
        with open(path, "w") as f:
            for pair, entity in zip(pairs, entities):
                prompt = build_pattern_prompt(
                    pair.question.title,
                    pair.question.body_text,
                    pair.answer.body_text,
                )
                record = {
                    "entity_id": entity["entity/id"],
                    "question_id": pair.question.id,
                    "stage": "pattern_tagging",
                    "prompt": prompt,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        size_mb = os.path.getsize(path) / 1e6
        print(f"       Written {path} ({count} prompts, {size_mb:.1f} MB)")
        generated["pattern_tagging"] = {"path": str(path), "count": count}

    if "reverse_morphogenesis" in stages:
        path = prompt_dir / "stage6-reverse-morphogenesis.jsonl"
        count = 0
        with open(path, "w") as f:
            for pair, entity in zip(pairs, entities):
                prompt = build_reverse_morphogenesis_prompt(
                    pair.question.title,
                    pair.question.body_text,
                    pair.answer.body_text,
                )
                record = {
                    "entity_id": entity["entity/id"],
                    "question_id": pair.question.id,
                    "stage": "reverse_morphogenesis",
                    "prompt": prompt,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        size_mb = os.path.getsize(path) / 1e6
        print(f"       Written {path} ({count} prompts, {size_mb:.1f} MB)")
        generated["reverse_morphogenesis"] = {"path": str(path), "count": count}

    if "thread_performatives" in stages and thread_diagrams:
        path = prompt_dir / "stage7-thread-performatives.jsonl"
        count = 0
        with open(path, "w") as f:
            for diagram in thread_diagrams:
                prompt = build_thread_performative_prompt(diagram)
                record = {
                    "thread_id": diagram.thread_id,
                    "stage": "thread_performatives",
                    "n_nodes": len(diagram.nodes),
                    "n_edges": len(diagram.edges),
                    "prompt": prompt,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        size_mb = os.path.getsize(path) / 1e6
        print(f"       Written {path} ({count} prompts, {size_mb:.1f} MB)")
        generated["thread_performatives"] = {"path": str(path), "count": count}

    # Write/update a manifest for the moist-run (merge stages across calls).
    manifest_path = prompt_dir / "manifest.json"
    merged_stages = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                prev_manifest = json.load(f)
            if isinstance(prev_manifest, dict):
                prev_stages = prev_manifest.get("stages", {})
                if isinstance(prev_stages, dict):
                    merged_stages.update(prev_stages)
        except (OSError, json.JSONDecodeError):
            pass
    merged_stages.update(generated)

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "moist-run",
        "note": "Prompt files for Codex/Claude batch submission. "
                "Each JSONL line has entity_id, question_id, stage, prompt.",
        "stages": merged_stages,
        "usage": {
            "codex": "codex --prompt-file <path.jsonl>",
            "claude_api": "for line in open(path): send(json.loads(line)['prompt'])",
            "manual": "Read prompt field, paste into LLM, save response",
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"       Written {manifest_path}")

    return generated


def print_dry_run(args):
    """Print execution plan without running anything."""
    posts = Path(args.posts_xml)
    posts_size = posts.stat().st_size / 1e9 if posts.exists() else 0

    # Estimate QA pair count from file size (~6KB per QA pair for physics.SE)
    est_pairs = int(posts_size * 1e9 / 6000) if posts_size else "?"
    if args.limit:
        est_pairs = min(est_pairs, args.limit) if isinstance(est_pairs, int) else args.limit

    # Estimate output sizes (based on physics.SE ratios: 114K pairs → 438MB entities)
    if isinstance(est_pairs, int):
        est_entities_mb = est_pairs * 438 / 114000
        est_embeddings_gb = est_pairs * 1024 * 4 / 1e9  # 1024-dim float32
        est_total_gb = est_entities_mb / 1000 + est_embeddings_gb + 0.5  # +0.5 for misc
    else:
        est_entities_mb = est_embeddings_gb = est_total_gb = "?"

    # Estimate time (based on physics.SE: ~15min for 114K on GPU)
    if isinstance(est_pairs, int):
        est_stage1_min = est_pairs / 114000 * 2
        est_stage2_min = est_pairs / 114000 * 5  # embeddings
        est_stage3_min = est_pairs / 114000 * 8  # LLM inference (slowest)
        est_stage4_min = est_pairs / 114000 * 1  # clustering
        est_stage5_min = est_pairs / 114000 * 2  # NER + scope
        est_stage5b_min = est_pairs / 114000 * 1.5  # MIT binder-pair pass
        est_stage6_min = est_stage1_min  # rough placeholder for reverse morphogenesis
        est_stage7_min = est_stage1_min * 1.5
    else:
        est_stage1_min = est_stage2_min = est_stage3_min = "?"
        est_stage4_min = est_stage5_min = est_stage5b_min = est_stage6_min = est_stage7_min = "?"

    skip_embeddings = args.skip_embeddings or args.moist_run
    skip_clustering = args.skip_clustering or args.moist_run
    llm_inference_active = (not args.skip_llm) and (not args.moist_run)

    stage2_active = not skip_embeddings
    stage3_active = args.moist_run or (not args.skip_llm)
    stage4_active = not skip_clustering
    stage5_active = not args.skip_ner
    stage5b_active = args.run_distinctor_mit
    stage6_active = args.moist_run or (not args.skip_llm)
    stage7_active = not args.skip_threads

    print("=" * 64)
    print("SUPERPOD JOB — DRY RUN (nothing will be executed)")
    print("=" * 64)
    print()
    print(f"  Input:        {args.posts_xml}")
    print(f"  Input size:   {posts_size:.2f} GB" if posts_size else f"  Input:        {args.posts_xml} (NOT FOUND)")
    print(f"  Site:         {args.site}")
    print(f"  Min score:    {args.min_score}")
    print(f"  Limit:        {args.limit or 'none'}")
    print(f"  Output dir:   {args.output_dir}")
    print()
    print(f"  Est. QA pairs:  ~{est_pairs:,}" if isinstance(est_pairs, int) else f"  Est. QA pairs:  {est_pairs}")
    print()

    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)

    print("  STAGE PLAN:")
    print(f"  {'Stage':<42s} {'Model/Tool':<36s} {'Est. Time':>10s}")
    print(f"  {'-'*42} {'-'*36} {'-'*10}")
    print(f"  {'1. Parse XML → QA pairs (CPU)':<42s} {'streaming XML parser':<36s} {fmt(est_stage1_min)+' min':>10s}")

    if not stage2_active:
        print(f"  {'2. Embeddings':<42s} {'SKIPPED':>10s}")
    else:
        print(f"  {'2. Dense embeddings (GPU)':<42s} {args.embed_model:<36s} {fmt(est_stage2_min)+' min':>10s}")

    if not stage3_active:
        print(f"  {'3. LLM pattern tagging':<42s} {'SKIPPED':>10s}")
    elif args.moist_run:
        print(f"  {'3. Pattern tagging prompts (CPU)':<42s} {'prompt generation':<36s} {fmt(est_stage1_min)+' min':>10s}")
    else:
        print(f"  {'3. LLM pattern tagging (GPU)':<42s} {args.llm_model:<36s} {fmt(est_stage3_min)+' min':>10s}")

    if not stage4_active:
        print(f"  {'4. Clustering':<42s} {'SKIPPED':>10s}")
    else:
        print(f"  {'4. Clustering (CPU)':<42s} {'HDBSCAN/KMeans':<36s} {fmt(est_stage4_min)+' min':>10s}")

    if not stage5_active:
        print(f"  {'5. NER + scope detection':<42s} {'SKIPPED':>10s}")
    else:
        print(f"  {'5. NER + scope detection (CPU)':<42s} {args.ner_kernel:<36s} {fmt(est_stage5_min)+' min':>10s}")

    if not stage5b_active:
        print(f"  {'5b. Distinctor MIT pilot':<42s} {'SKIPPED':>10s}")
    else:
        lim = args.distinctor_entity_limit or "all"
        src = "eprints" if args.distinctor_eprint_dir else "qa/metadata"
        tool = f"binder-pair MIT {src} (limit={lim})"
        print(f"  {'5b. Distinctor MIT pilot (CPU)':<42s} {tool:<36s} {fmt(est_stage5b_min)+' min':>10s}")

    if not stage6_active:
        print(f"  {'6. Reverse morphogenesis S←Q←A':<42s} {'SKIPPED':>10s}")
    elif args.moist_run:
        print(f"  {'6. Reverse morphogenesis prompts':<42s} {'prompt generation':<36s} {fmt(est_stage6_min)+' min':>10s}")
    else:
        print(f"  {'6. Reverse morphogenesis S←Q←A (LLM)':<42s} {args.llm_model:<36s} {fmt(est_stage6_min)+' min':>10s}")

    # Stage 7: Thread wiring diagrams
    ct_ref_exists = Path(args.ct_reference).exists()
    if not stage7_active:
        print(f"  {'7. Thread wiring diagrams':<42s} {'SKIPPED':>10s}")
    elif ct_ref_exists:
        print(f"  {'7. CT-backed wiring + IATC + cat (CPU)':<42s} {'CT ref + IATC + ports':<36s} {fmt(est_stage7_min)+' min':>10s}")
    else:
        print(f"  {'7. Thread wiring + performatives (CPU)':<42s} {'IATC regex bank (legacy)':<36s} {fmt(est_stage7_min)+' min':>10s}")

    # Stage 8: Expression surfaces
    stage8_active = not args.skip_expressions
    if not stage8_active:
        print(f"  {'8. Expression surface parsing':<42s} {'SKIPPED':>10s}")
    else:
        est_stage8_min = est_stage1_min * 0.5 if isinstance(est_stage1_min, (int, float)) else "?"
        print(f"  {'8. Expression surfaces (CPU)':<42s} {'latex_sexp parser':<36s} {fmt(est_stage8_min)+' min':>10s}")

    # Stage 9a: Hypergraph assembly
    stage9a_active = not args.skip_hypergraphs
    if not stage9a_active:
        print(f"  {'9a. Hypergraph assembly':<42s} {'SKIPPED':>10s}")
    else:
        est_stage9a_min = est_stage1_min * 0.3 if isinstance(est_stage1_min, (int, float)) else "?"
        print(f"  {'9a. Hypergraph assembly (CPU)':<42s} {'typed hypergraph builder':<36s} {fmt(est_stage9a_min)+' min':>10s}")

    # Stage 9b: Graph embedding
    stage9b_active = not args.skip_graph_embed
    if not stage9b_active:
        print(f"  {'9b. Graph embedding':<42s} {'SKIPPED':>10s}")
    else:
        est_stage9b_min = est_stage1_min * 2.0 if isinstance(est_stage1_min, (int, float)) else "?"
        print(f"  {'9b. Graph embedding (GPU, R-GCN)':<42s} {f'{args.graph_embed_dim}d, {args.graph_embed_epochs}ep, bs={args.graph_embed_batch_size}':<36s} {fmt(est_stage9b_min)+' min':>10s}")

    # Stage 10: FAISS index
    stage10_active = not args.skip_faiss
    if not stage10_active:
        print(f"  {'10. FAISS similarity index':<42s} {'SKIPPED':>10s}")
    else:
        est_stage10_min = est_stage1_min * 0.1 if isinstance(est_stage1_min, (int, float)) else "?"
        print(f"  {'10. FAISS similarity index (CPU)':<42s} {'inner product search':<36s} {fmt(est_stage10_min)+' min':>10s}")

    print(f"  {'-'*42} {'-'*36} {'-'*10}")

    active = (1 + int(stage2_active) + int(stage3_active) + int(stage4_active)
              + int(stage5_active) + int(stage5b_active)
              + int(stage6_active) + int(stage7_active)
              + int(stage8_active) + int(stage9a_active)
              + int(stage9b_active) + int(stage10_active))
    if isinstance(est_stage1_min, (int, float)):
        est_total_min = est_stage1_min
        if stage2_active:
            est_total_min += est_stage2_min
        if stage3_active:
            est_total_min += est_stage1_min if args.moist_run else est_stage3_min
        if stage4_active:
            est_total_min += est_stage4_min
        if stage5_active:
            est_total_min += est_stage5_min
        if stage5b_active:
            est_total_min += est_stage5b_min
        if stage6_active:
            est_total_min += est_stage6_min
        if stage7_active:
            est_total_min += est_stage7_min
        if stage8_active:
            est_total_min += est_stage8_min
        if stage9a_active:
            est_total_min += est_stage9a_min
        if stage9b_active:
            est_total_min += est_stage9b_min
        if stage10_active:
            est_total_min += est_stage10_min
    else:
        est_total_min = "?"
    total_stages = 11 + int(stage5b_active)
    print(f"  {'TOTAL':<42s} {f'{active}/{total_stages} stages active':<36s} {fmt(est_total_min)+' min':>10s}")
    print()

    print("  ESTIMATED OUTPUT:")
    if isinstance(est_entities_mb, (int, float)):
        print(f"    entities.json         ~{fmt(est_entities_mb)} MB")
        if stage2_active:
            print(f"    embeddings.npy        ~{fmt(est_embeddings_gb)} GB")
        if llm_inference_active:
            print(f"    pattern-tags.json     ~{fmt(est_entities_mb * 0.3)} MB")
        if stage4_active:
            print(f"    clusters.json         ~{fmt(est_entities_mb * 0.05)} MB")
        if not args.skip_ner:
            print(f"    ner-terms.json        ~{fmt(est_entities_mb * 0.4)} MB")
            print(f"    scopes.json           ~{fmt(est_entities_mb * 0.2)} MB")
            if args.discover_terms:
                print(f"    candidate-new-terms.jsonl ~{fmt(est_entities_mb * 0.05)} MB")
        if stage5b_active:
            print(f"    distinctor-mit-hits.jsonl   ~{fmt(est_entities_mb * 0.08)} MB")
            print(f"    distinctor-mit-summary.json ~{fmt(est_entities_mb * 0.01)} MB")
        if not args.skip_threads:
            print(f"    thread-wiring.json    ~{fmt(est_entities_mb * 0.5)} MB")
        if not args.skip_expressions:
            print(f"    expression-surfaces.json ~{fmt(est_entities_mb * 0.3)} MB")
        if not args.skip_hypergraphs:
            print(f"    hypergraphs.json      ~{fmt(est_entities_mb * 0.8)} MB")
        if not args.skip_graph_embed:
            est_emb_mb = est_pairs * args.graph_embed_dim * 4 / 1e6 if isinstance(est_pairs, int) else "?"
            print(f"    hypergraph-embeddings.npy ~{fmt(est_emb_mb)} MB")
        if not args.skip_faiss:
            print(f"    structural-similarity-index  ~{fmt(est_emb_mb) if not args.skip_graph_embed else '?'} MB")
        print(f"    {'':24s} --------")
        print(f"    {'TOTAL':24s} ~{fmt(est_total_gb)} GB")
    else:
        print(f"    (cannot estimate — input file not found)")
    print()

    print("  GPU REQUIREMENTS:")
    if stage2_active or llm_inference_active:
        print(f"    Embedding model:  {args.embed_model}" if stage2_active else "")
        print(f"    LLM model:        {args.llm_model}" if llm_inference_active else "")
        vram = 18 if llm_inference_active else 2  # bge-large ~2GB, Llama-3-8B ~16GB
        print(f"    Est. VRAM needed: ~{vram} GB")
    else:
        print(f"    None (CPU-only stages)")
    print()

    print("  PATTERNS (25 math-informal):")
    for i, (name, desc) in enumerate(PATTERNS):
        print(f"    {i+1:2d}. {name:<35s} {desc[:50]}")
    print()

    print("  TO RUN FOR REAL:")
    cmd_parts = [f"python scripts/superpod-job.py {args.posts_xml}"]
    cmd_parts.append(f"  --output-dir {args.output_dir}")
    cmd_parts.append(f"  --site {args.site}")
    if args.limit:
        cmd_parts.append(f"  --limit {args.limit}")
    if args.skip_embeddings:
        cmd_parts.append(f"  --skip-embeddings")
    if args.skip_llm:
        cmd_parts.append(f"  --skip-llm")
    if args.skip_clustering:
        cmd_parts.append(f"  --skip-clustering")
    if args.skip_ner:
        cmd_parts.append(f"  --skip-ner")
    if args.discover_terms:
        cmd_parts.append(f"  --discover-terms")
        if args.discover_terms_min_freq != 3:
            cmd_parts.append(f"  --discover-terms-min-freq {args.discover_terms_min_freq}")
        if args.discover_terms_max != 2000:
            cmd_parts.append(f"  --discover-terms-max {args.discover_terms_max}")
        if args.discover_terms_max_per_entity != 64:
            cmd_parts.append(f"  --discover-terms-max-per-entity {args.discover_terms_max_per_entity}")
        if args.discover_terms_eprint_dir:
            cmd_parts.append(f"  --discover-terms-eprint-dir {args.discover_terms_eprint_dir}")
        if args.discover_terms_eprint_max_chars != 240000:
            cmd_parts.append(f"  --discover-terms-eprint-max-chars {args.discover_terms_eprint_max_chars}")
        if args.discover_terms_eprint_max_tex_members != 4:
            cmd_parts.append(f"  --discover-terms-eprint-max-tex-members {args.discover_terms_eprint_max_tex_members}")
    if args.run_distinctor_mit:
        cmd_parts.append(f"  --run-distinctor-mit")
        if args.distinctor_entity_limit:
            cmd_parts.append(f"  --distinctor-entity-limit {args.distinctor_entity_limit}")
        if args.distinctor_max_hits != 500:
            cmd_parts.append(f"  --distinctor-max-hits {args.distinctor_max_hits}")
        if args.distinctor_seed != 13:
            cmd_parts.append(f"  --distinctor-seed {args.distinctor_seed}")
        if args.distinctor_eprint_dir:
            cmd_parts.append(f"  --distinctor-eprint-dir {args.distinctor_eprint_dir}")
        if args.distinctor_eprint_max_chars != 240000:
            cmd_parts.append(f"  --distinctor-eprint-max-chars {args.distinctor_eprint_max_chars}")
        if args.distinctor_eprint_max_tex_members != 4:
            cmd_parts.append(f"  --distinctor-eprint-max-tex-members {args.distinctor_eprint_max_tex_members}")
    if args.skip_threads:
        cmd_parts.append(f"  --skip-threads")
    if args.thread_limit:
        cmd_parts.append(f"  --thread-limit {args.thread_limit}")
    if args.skip_hypergraphs:
        cmd_parts.append(f"  --skip-hypergraphs")
    if args.paper_hg_eprint_dir:
        cmd_parts.append(f"  --paper-hg-eprint-dir {args.paper_hg_eprint_dir}")
    if args.paper_hg_max_expressions != 160:
        cmd_parts.append(f"  --paper-hg-max-expressions {args.paper_hg_max_expressions}")
    if args.paper_hg_text_max_chars != 240000:
        cmd_parts.append(f"  --paper-hg-text-max-chars {args.paper_hg_text_max_chars}")
    if args.paper_hg_max_tex_members != 4:
        cmd_parts.append(f"  --paper-hg-max-tex-members {args.paper_hg_max_tex_members}")
    print(f"    {' \\\\\n    '.join(cmd_parts)}")
    print()
    print(f"  MOIST RUN (CPU stages + prompt files for Codex handoff):")
    moist_parts = [f"python scripts/superpod-job.py {args.posts_xml}"]
    moist_parts.append(f"  --moist-run")
    moist_parts.append(f"  --output-dir {args.output_dir}")
    moist_parts.append(f"  --site {args.site}")
    if args.limit:
        moist_parts.append(f"  --limit {args.limit}")
    if args.discover_terms:
        moist_parts.append(f"  --discover-terms")
        if args.discover_terms_min_freq != 3:
            moist_parts.append(f"  --discover-terms-min-freq {args.discover_terms_min_freq}")
        if args.discover_terms_max != 2000:
            moist_parts.append(f"  --discover-terms-max {args.discover_terms_max}")
        if args.discover_terms_max_per_entity != 64:
            moist_parts.append(f"  --discover-terms-max-per-entity {args.discover_terms_max_per_entity}")
        if args.discover_terms_eprint_dir:
            moist_parts.append(f"  --discover-terms-eprint-dir {args.discover_terms_eprint_dir}")
        if args.discover_terms_eprint_max_chars != 240000:
            moist_parts.append(f"  --discover-terms-eprint-max-chars {args.discover_terms_eprint_max_chars}")
        if args.discover_terms_eprint_max_tex_members != 4:
            moist_parts.append(f"  --discover-terms-eprint-max-tex-members {args.discover_terms_eprint_max_tex_members}")
    if args.run_distinctor_mit:
        moist_parts.append(f"  --run-distinctor-mit")
        if args.distinctor_entity_limit:
            moist_parts.append(f"  --distinctor-entity-limit {args.distinctor_entity_limit}")
        if args.distinctor_max_hits != 500:
            moist_parts.append(f"  --distinctor-max-hits {args.distinctor_max_hits}")
        if args.distinctor_seed != 13:
            moist_parts.append(f"  --distinctor-seed {args.distinctor_seed}")
        if args.distinctor_eprint_dir:
            moist_parts.append(f"  --distinctor-eprint-dir {args.distinctor_eprint_dir}")
        if args.distinctor_eprint_max_chars != 240000:
            moist_parts.append(f"  --distinctor-eprint-max-chars {args.distinctor_eprint_max_chars}")
        if args.distinctor_eprint_max_tex_members != 4:
            moist_parts.append(f"  --distinctor-eprint-max-tex-members {args.distinctor_eprint_max_tex_members}")
    if args.skip_hypergraphs:
        moist_parts.append(f"  --skip-hypergraphs")
    if args.paper_hg_eprint_dir:
        moist_parts.append(f"  --paper-hg-eprint-dir {args.paper_hg_eprint_dir}")
    if args.paper_hg_max_expressions != 160:
        moist_parts.append(f"  --paper-hg-max-expressions {args.paper_hg_max_expressions}")
    if args.paper_hg_text_max_chars != 240000:
        moist_parts.append(f"  --paper-hg-text-max-chars {args.paper_hg_text_max_chars}")
    if args.paper_hg_max_tex_members != 4:
        moist_parts.append(f"  --paper-hg-max-tex-members {args.paper_hg_max_tex_members}")
    print(f"    {' \\\\\n    '.join(moist_parts)}")
    print()
    print(f"    Generates: moist-prompts/stage3-pattern-tagging.jsonl")
    print(f"               moist-prompts/stage6-reverse-morphogenesis.jsonl")
    print(f"               moist-prompts/stage7-thread-performatives.jsonl")
    print(f"    Each line is {{entity_id, question_id, stage, prompt}}")
    print(f"    Feed to Codex, Claude API, or any LLM batch runner.")
    print()
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="Superpod batch job: SE dump -> F6 artefacts",
        epilog="Available downloads: " + ", ".join(
            f"{k} ({v['description']})" for k, v in SE_DUMPS.items()))
    parser.add_argument("posts_xml", nargs="?", default=None,
                        help="Path to Posts.xml (not needed with --download or --arxiv-jsonl)")

    # Download options
    parser.add_argument("--download", choices=list(SE_DUMPS.keys()),
                        help="Download + extract a SE dump from Internet Archive")
    parser.add_argument("--data-dir", default="./se-data",
                        help="Directory for downloaded dumps (default: ./se-data)")

    # ArXiv input (alternative to SE Posts.xml)
    parser.add_argument("--arxiv-jsonl", default=None,
                        help="ArXiv metadata JSONL (from harvest-arxiv-ct.py). "
                             "Alternative to Posts.xml for processing ArXiv papers.")

    # Laptop mode
    parser.add_argument("--laptop", action="store_true",
                        help="Laptop-friendly defaults: MiniLM embeddings, CPU device, "
                             "moist-run for LLM stages, small batch sizes")

    parser.add_argument("--input-dir", "-i", default=None,
                        help="Base directory for input data (Posts.xml, 7z files). "
                             "Use when data lives on a different filesystem (e.g. /scratch/).")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for all artefacts (default: auto from --site)")
    parser.add_argument("--site", default="math.stackexchange",
                        help="SE site name (or ArXiv category e.g. 'arxiv.math-ct')")
    parser.add_argument("--min-score", type=int, default=1,
                        help="Minimum post score")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max QA pairs to index in Stage 1 (pilot mode)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show execution plan without running anything")
    parser.add_argument("--moist-run", action="store_true",
                        help="Run CPU stages, output prompts for LLM stages (for Codex handoff)")

    # Embedding options
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5",
                        help="Embedding model")
    parser.add_argument("--embed-batch-size", type=int, default=8192,
                        help="Embedding batch size (default: 8192, tune for GPU VRAM)")
    parser.add_argument("--embed-device", default="cuda",
                        help="Device for embeddings")
    parser.add_argument("--skip-embeddings", action="store_true")

    # LLM options
    parser.add_argument("--llm-model",
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="LLM for pattern tagging")
    parser.add_argument("--llm-batch-size", type=int, default=16,
                        help="LLM inference batch size (default: 16, safe for 80GB GPUs)")
    parser.add_argument("--skip-llm", action="store_true")

    # Clustering
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--min-cluster-size", type=int, default=50)

    # NER + scope detection (Stage 5)
    parser.add_argument("--ner-kernel",
                        default="data/ner-kernel/terms.tsv",
                        help="Path to NER kernel TSV")
    parser.add_argument("--skip-ner", action="store_true",
                        help="Skip NER term spotting and scope detection")
    parser.add_argument("--discover-terms", action="store_true",
                        help="Enable open-world technical term discovery in Stage 5")
    parser.add_argument("--discover-terms-min-freq", type=int, default=3,
                        help="Min entity frequency for discovered terms (default: 3)")
    parser.add_argument("--discover-terms-max", type=int, default=2000,
                        help="Max discovered terms to write (default: 2000)")
    parser.add_argument("--discover-terms-max-per-entity", type=int, default=64,
                        help="Max raw discovery candidates scanned per entity (default: 64)")
    parser.add_argument("--discover-terms-eprint-dir", default=None,
                        help="Optional arXiv eprint directory for full-text term discovery")
    parser.add_argument("--discover-terms-eprint-max-chars", type=int, default=240000,
                        help="Per-entity text cap for eprint-backed term discovery (default: 240000)")
    parser.add_argument("--discover-terms-eprint-max-tex-members", type=int, default=4,
                        help="Max .tex members per tar for eprint-backed term discovery (default: 4)")
    parser.add_argument("--run-distinctor-mit", action="store_true",
                        help="Run binder-pair MIT/distinctor pilot (Stage 5b)")
    parser.add_argument("--distinctor-entity-limit", type=int, default=0,
                        help="Optional cap for Stage 5b entities (0 = all)")
    parser.add_argument("--distinctor-max-hits", type=int, default=500,
                        help="Max inspectable MIT hit rows to write (default: 500)")
    parser.add_argument("--distinctor-seed", type=int, default=13,
                        help="Sampling seed for Stage 5b when --distinctor-entity-limit is used")
    parser.add_argument("--distinctor-eprint-dir", default=None,
                        help="Optional arXiv eprint directory (.tar/.tar.gz/.tex) for Stage 5b full-text MIT")
    parser.add_argument("--distinctor-eprint-max-chars", type=int, default=240000,
                        help="Per-entity text cap when reading eprints (default: 240000)")
    parser.add_argument("--distinctor-eprint-max-tex-members", type=int, default=4,
                        help="Max .tex members per tar archive in Stage 5b (default: 4)")

    # Thread wiring diagrams (Stage 7)
    parser.add_argument("--comments-xml", default=None,
                        help="Path to Comments.xml (auto-detected from Posts.xml dir)")
    parser.add_argument("--skip-threads", action="store_true",
                        help="Skip thread wiring diagram construction")
    parser.add_argument("--thread-limit", type=int, default=None,
                        help="Max threads to process in Stage 7")
    parser.add_argument("--ct-reference", default="data/nlab-ct-reference.json",
                        help="CT reference dictionary for CT-backed wiring (Stage 7)")

    # Expression surfaces + hypergraphs (Stages 8-9a)
    parser.add_argument("--skip-expressions", action="store_true",
                        help="Skip expression surface parsing (Stage 8)")
    parser.add_argument("--skip-hypergraphs", action="store_true",
                        help="Skip hypergraph assembly (Stage 9a)")
    parser.add_argument("--paper-hg-eprint-dir", default=None,
                        help="Optional arXiv eprint dir for paper-level hypergraphs (Stage 9a arXiv mode)")
    parser.add_argument("--paper-hg-max-expressions", type=int, default=160,
                        help="Max expressions parsed per arXiv paper hypergraph (default: 160)")
    parser.add_argument("--paper-hg-text-max-chars", type=int, default=240000,
                        help="Text cap per arXiv paper for hypergraph assembly (default: 240000)")
    parser.add_argument("--paper-hg-max-tex-members", type=int, default=4,
                        help="Max .tex members read per arXiv tar for paper hypergraphs (default: 4)")

    # Graph embedding + FAISS index (Stages 9b-10)
    parser.add_argument("--skip-graph-embed", action="store_true",
                        help="Skip graph GNN embedding (Stage 9b, GPU)")
    parser.add_argument("--skip-faiss", action="store_true",
                        help="Skip FAISS index construction (Stage 10)")
    parser.add_argument("--graph-embed-dim", type=int, default=128,
                        help="Hypergraph embedding dimension (default: 128)")
    parser.add_argument("--graph-embed-epochs", type=int, default=50,
                        help="GNN training epochs (default: 50)")
    parser.add_argument("--graph-embed-batch-size", type=int, default=512,
                        help="GNN training batch size (default: 512, was 64)")
    parser.add_argument("--graph-embed-workers", type=int, default=4,
                        help="DataLoader workers for GNN training (default: 4, 0=inline)")

    # Health gates
    parser.add_argument("--preflight", action="store_true",
                        help="Strict health gates: abort on warnings (use for pre-flight validation runs)")

    # Sharding (for parallel execution on multi-GPU machines)
    parser.add_argument("--shard-index", type=int, default=None,
                        help="Shard index (0-based) for parallel runs")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards")

    args = parser.parse_args()

    # --laptop: sensible CPU defaults
    if args.laptop:
        if args.embed_model == parser.get_default("embed_model"):
            args.embed_model = "all-MiniLM-L6-v2"
        if args.embed_device == parser.get_default("embed_device"):
            args.embed_device = "cpu"
        if args.embed_batch_size == parser.get_default("embed_batch_size"):
            args.embed_batch_size = 256
        if args.graph_embed_batch_size == parser.get_default("graph_embed_batch_size"):
            args.graph_embed_batch_size = 64
        if args.graph_embed_workers == parser.get_default("graph_embed_workers"):
            args.graph_embed_workers = 0
        if args.graph_embed_epochs == parser.get_default("graph_embed_epochs"):
            args.graph_embed_epochs = 10
        if not args.moist_run:
            args.moist_run = True
        print("  Laptop mode: MiniLM embeddings, CPU, moist-run for LLM stages")

    # --input-dir: resolve relative input paths against a base directory
    if args.input_dir:
        input_base = Path(args.input_dir)
        if args.posts_xml and not Path(args.posts_xml).is_absolute():
            args.posts_xml = str(input_base / args.posts_xml)
        if args.arxiv_jsonl and not Path(args.arxiv_jsonl).is_absolute():
            args.arxiv_jsonl = str(input_base / args.arxiv_jsonl)
        if args.comments_xml and not Path(args.comments_xml).is_absolute():
            args.comments_xml = str(input_base / args.comments_xml)
        if args.paper_hg_eprint_dir and not Path(args.paper_hg_eprint_dir).is_absolute():
            args.paper_hg_eprint_dir = str(input_base / args.paper_hg_eprint_dir)
        if args.discover_terms_eprint_dir and not Path(args.discover_terms_eprint_dir).is_absolute():
            args.discover_terms_eprint_dir = str(input_base / args.discover_terms_eprint_dir)
        if args.distinctor_eprint_dir and not Path(args.distinctor_eprint_dir).is_absolute():
            args.distinctor_eprint_dir = str(input_base / args.distinctor_eprint_dir)
        # data-dir for downloads also goes to input location
        if args.data_dir == parser.get_default("data_dir"):
            args.data_dir = str(input_base / "se-data")
        print(f"  Input dir: {args.input_dir}")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")
    if args.discover_terms_min_freq <= 0:
        parser.error("--discover-terms-min-freq must be > 0")
    if args.discover_terms_max <= 0:
        parser.error("--discover-terms-max must be > 0")
    if args.discover_terms_max_per_entity <= 0:
        parser.error("--discover-terms-max-per-entity must be > 0")
    if args.discover_terms_eprint_max_chars <= 0:
        parser.error("--discover-terms-eprint-max-chars must be > 0")
    if args.discover_terms_eprint_max_tex_members <= 0:
        parser.error("--discover-terms-eprint-max-tex-members must be > 0")
    if args.paper_hg_max_expressions <= 0:
        parser.error("--paper-hg-max-expressions must be > 0")
    if args.paper_hg_text_max_chars <= 0:
        parser.error("--paper-hg-text-max-chars must be > 0")
    if args.paper_hg_max_tex_members <= 0:
        parser.error("--paper-hg-max-tex-members must be > 0")
    if args.distinctor_entity_limit < 0:
        parser.error("--distinctor-entity-limit must be >= 0")
    if args.distinctor_max_hits <= 0:
        parser.error("--distinctor-max-hits must be > 0")
    if args.distinctor_eprint_max_chars <= 0:
        parser.error("--distinctor-eprint-max-chars must be > 0")
    if args.distinctor_eprint_max_tex_members <= 0:
        parser.error("--distinctor-eprint-max-tex-members must be > 0")

    # Shard validation
    if (args.shard_index is not None) != (args.num_shards is not None):
        parser.error("--shard-index and --num-shards must be used together")
    if args.num_shards is not None:
        if args.num_shards < 1:
            parser.error("--num-shards must be >= 1")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            parser.error(f"--shard-index must be in [0, {args.num_shards - 1}]")
        # In shard mode, skip stages that need the full corpus
        args.skip_clustering = True
        args.skip_graph_embed = True
        args.skip_faiss = True
        print(f"  Shard mode: shard {args.shard_index}/{args.num_shards}"
              f" (auto-skipping clustering, graph-embed, faiss)")

    auto_thread_limit = False
    if args.limit is not None and args.thread_limit is None:
        args.thread_limit = args.limit
        auto_thread_limit = True

    if train_gnn is None and not args.skip_graph_embed:
        args.skip_graph_embed = True
        if not args.skip_faiss:
            args.skip_faiss = True
        print("  Graph embedding disabled: optional dependencies missing")
        print(f"    reason: {_GRAPH_EMBED_IMPORT_ERROR}")
        print("    auto-skipping stages 9b (graph-embed) and 10 (faiss)")

    # Dry-run for download-only mode: infer expected Posts.xml path with no side effects.
    if args.dry_run and args.download and not args.posts_xml:
        dump = SE_DUMPS[args.download]
        if args.site == parser.get_default("site"):
            args.site = dump["site"]
        args.posts_xml = str(Path(args.data_dir) / dump["dirname"] / "Posts.xml")
        print(f"  Dry run note: inferred posts path from --download {args.download}:")
        print(f"    {args.posts_xml}")

    if not args.output_dir:
        args.output_dir = derive_default_output_dir(args.site)

    # ========== Dry Run ==========
    # Run before download logic to guarantee no network/filesystem side effects.
    if args.dry_run:
        if not args.posts_xml:
            parser.error("posts_xml is required for --dry-run unless used with --download")
        print_dry_run(args)
        return

    # ========== Download ==========
    if args.download:
        posts_path = download_and_extract(args.download, args.data_dir)
        if not args.posts_xml:
            # Just downloading, not processing
            return
        # If posts_xml was also given, continue to processing

    if not args.posts_xml and not args.arxiv_jsonl:
        parser.error("posts_xml or --arxiv-jsonl is required (or use --download to fetch data first)")

    if auto_thread_limit:
        print(f"  Pilot mode: --thread-limit defaulted to --limit ({args.limit})")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.arxiv_jsonl and args.paper_hg_eprint_dir is None:
        if args.distinctor_eprint_dir:
            args.paper_hg_eprint_dir = args.distinctor_eprint_dir
            print(f"  Paper hypergraphs: defaulting --paper-hg-eprint-dir to --distinctor-eprint-dir ({args.paper_hg_eprint_dir})")
        elif args.discover_terms_eprint_dir:
            args.paper_hg_eprint_dir = args.discover_terms_eprint_dir
            print(f"  Paper hypergraphs: defaulting --paper-hg-eprint-dir to --discover-terms-eprint-dir ({args.paper_hg_eprint_dir})")

    if args.distinctor_eprint_dir and not Path(args.distinctor_eprint_dir).exists():
        print(f"  WARNING: --distinctor-eprint-dir not found: {args.distinctor_eprint_dir}")
        print("           Stage 5b will fall back to metadata/QA text.")
    if args.discover_terms_eprint_dir and not Path(args.discover_terms_eprint_dir).exists():
        print(f"  WARNING: --discover-terms-eprint-dir not found: {args.discover_terms_eprint_dir}")
        print("           Open-world term discovery will use metadata/QA text.")
    if args.paper_hg_eprint_dir and not Path(args.paper_hg_eprint_dir).exists():
        print(f"  WARNING: --paper-hg-eprint-dir not found: {args.paper_hg_eprint_dir}")
        print("           Paper hypergraphs will use metadata/QA text.")

    # ========== Moist Run ==========
    # Moist run: execute CPU stages normally, generate prompt files for LLM
    # stages. This lets you run Stage 1 (parse) and Stage 5 (NER) on a laptop,
    # then hand the prompt files to Codex/Claude for Stage 3 and Stage 6.
    if args.moist_run:
        args.skip_embeddings = True
        args.skip_clustering = True
        # We still parse (Stage 1) and run NER (Stage 5), but skip_llm
        # is handled specially below — we generate prompts instead.
        print("=" * 64)
        print("MOIST RUN: CPU stages + prompt generation for Codex handoff")
        print("  Stages 2,4 (GPU): skipped")
        print("  Stage 3 (LLM):    prompt files generated")
        print("  Stage 6 (S←Q←A):  prompt files generated")
        print("  Stage 7 (threads): classical + prompt files generated")
        print("=" * 64)
        print()

    t0 = time.time()

    # Health gate: warn or abort depending on --preflight
    def health_gate(stage: str, condition: bool, message: str):
        """Print health warning; abort if --preflight is set."""
        if not condition:
            return
        if args.preflight:
            print(f"\n[HEALTH GATE FAIL] {stage}: {message}")
            print(f"  Aborting (--preflight mode). Fix the issue and re-run.")
            sys.exit(1)
        else:
            print(f"  [HEALTH WARNING] {stage}: {message}")

    n_stages = 11 + (1 if args.run_distinctor_mit else 0)  # + optional 5b

    # Auto-detect Comments.xml if not provided
    if not args.comments_xml and args.posts_xml:
        comments_candidate = Path(args.posts_xml).parent / "Comments.xml"
        if comments_candidate.exists():
            args.comments_xml = str(comments_candidate)
            print(f"  Auto-detected Comments.xml: {args.comments_xml}")
        else:
            print(f"  No Comments.xml found at {comments_candidate}")

    # ========== Stage 1: Parse ==========
    arxiv_id_map = {}  # question.id → arxiv string ID (only for ArXiv input)
    if args.arxiv_jsonl:
        # ArXiv JSONL input
        print(f"[Stage 1/{n_stages}] Loading ArXiv papers from {args.arxiv_jsonl}...")
        pairs, arxiv_id_map = load_arxiv_pairs(args.arxiv_jsonl, limit=args.limit)
        # Default site name for ArXiv
        if args.site == "math.stackexchange":
            args.site = "arxiv.math-ct"
        # Thread stages don't apply to single-document papers
        if not args.skip_threads:
            args.skip_threads = True
            print(f"       Auto-skipping thread wiring (single-document papers)")
    else:
        print(f"[Stage 1/{n_stages}] Parsing {args.posts_xml}...")
        pairs = build_qa_pairs_streaming(
            args.posts_xml,
            min_score=args.min_score,
            question_limit=args.limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        if args.limit and len(pairs) > args.limit:
            print(f"       Parsed {len(pairs)} pairs, limiting to {args.limit}")
            pairs = pairs[:args.limit]

    stats = corpus_stats(pairs)
    print(f"       {stats['qa_pairs']} QA pairs, "
          f"{stats['unique_tags']} tags, "
          f"{stats['with_latex']} with LaTeX")

    # Write entities (without embeddings — those go in .npy)
    entities = []
    all_relations = []
    for pair in pairs:
        if arxiv_id_map:
            arxiv_str_id = arxiv_id_map.get(pair.question.id, str(pair.question.id))
            entity_id = f"arxiv-{arxiv_str_id}"
        else:
            entity_id = f"se-{args.site.split('.')[0]}-{pair.question.id}"
        entity = qa_to_entity(pair, site=args.site, entity_id=entity_id)
        entities.append(entity)
        rels = qa_to_relations(pair, site=args.site, entity_id=entity_id)
        all_relations.extend(rels)

    tags = tag_entities(pairs)

    write_json(outdir / "entities.json", entities)
    write_json(outdir / "relations.json", all_relations)
    write_json(outdir / "tags.json", tags)
    write_json(outdir / "stats.json", stats)

    print(f"       Stage 1 done in {time.time()-t0:.0f}s")

    # ========== Stage 2: Embeddings ==========
    if not args.skip_embeddings:
        t2 = time.time()
        print(f"\n[Stage 2/{n_stages}] Embeddings ({args.embed_model})...")
        embeddings = compute_qa_embeddings(
            pairs,
            model_name=args.embed_model,
            batch_size=args.embed_batch_size,
            device=args.embed_device,
        )
        emb_path = outdir / "embeddings.npy"
        np.save(emb_path, embeddings)
        print(f"       Shape: {embeddings.shape}, "
              f"saved {os.path.getsize(emb_path)/1e9:.2f} GB")
        print(f"       Stage 2 done in {time.time()-t2:.0f}s")
    else:
        print(f"\n[Stage 2/{n_stages}] Skipped (--skip-embeddings)")
        embeddings = None

    # ========== Shared LLM pipeline (lazy, created once) ==========
    llm_pipe = None
    llm_tokenizer = None

    def _ensure_llm_pipeline():
        nonlocal llm_pipe, llm_tokenizer
        if llm_pipe is None:
            llm_pipe, llm_tokenizer = _create_llm_pipeline(
                args.llm_model, args.llm_batch_size)
        return llm_pipe, llm_tokenizer

    # ========== Stage 3: LLM pattern tagging ==========
    if args.moist_run:
        # Moist mode: generate prompts, don't run LLM
        t3 = time.time()
        print(f"\n[Stage 3/{n_stages}] Generating pattern-tagging prompts (moist-run)...")
        moist_result = generate_moist_prompts(
            pairs, entities, outdir, stages=["pattern_tagging"])
        print(f"       Stage 3 (moist) done in {time.time()-t3:.0f}s")
    elif not args.skip_llm:
        t3 = time.time()
        global _llm_start
        _llm_start = t3
        print(f"\n[Stage 3/{n_stages}] LLM pattern tagging ({args.llm_model})...")
        pipe, tok = _ensure_llm_pipeline()
        pattern_tags = tag_patterns_llm_batch(
            pairs,
            batch_size=args.llm_batch_size,
            pipe=pipe, tokenizer=tok,
            entry_ids=[e["entity/id"] for e in entities],
        )
        write_json(outdir / "pattern-tags.json", pattern_tags)
        print(f"       Stage 3 done in {time.time()-t3:.0f}s")

        # Pattern frequency summary
        from collections import Counter
        freq = Counter()
        for pt in pattern_tags:
            freq.update(pt["patterns"])
        print(f"       Pattern frequency (top 10):")
        for name, count in freq.most_common(10):
            print(f"         {name}: {count}")
    else:
        print(f"\n[Stage 3/{n_stages}] Skipped (--skip-llm)")

    # ========== Stage 4: Clustering ==========
    if not args.skip_clustering and embeddings is not None:
        t4 = time.time()
        print(f"\n[Stage 4/{n_stages}] Clustering...")
        labels, n_clusters, n_noise = cluster_embeddings(
            embeddings, min_cluster_size=args.min_cluster_size)

        # Attach cluster labels to entity IDs
        cluster_data = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "assignments": [
                {"entity_id": entities[i]["entity/id"], "cluster": labels[i]}
                for i in range(len(labels))
            ],
        }
        write_json(outdir / "clusters.json", cluster_data)
        print(f"       Stage 4 done in {time.time()-t4:.0f}s")
    else:
        print(f"\n[Stage 4/{n_stages}] Skipped")

    # ========== Stage 5: NER + Scope Detection ==========
    stage5_stats = None
    stage5b_stats = None
    scope_detector, scope_detector_name = _load_scope_detector(prefer_nlab=True)
    if not args.skip_ner:
        ner_path = Path(args.ner_kernel)
        if not ner_path.exists():
            print(f"\n[Stage 5/{n_stages}] Skipped (NER kernel not found: {ner_path})")
        else:
            t5 = time.time()
            print(f"\n[Stage 5/{n_stages}] NER term spotting + scope detection...")
            print(f"       Kernel: {ner_path}")
            stage5_stats = run_stage5_ner_scopes(
                entities, pairs, str(ner_path), outdir,
                scope_detector=scope_detector,
                scope_detector_name=scope_detector_name,
                discover_terms=args.discover_terms,
                discover_terms_min_freq=args.discover_terms_min_freq,
                discover_terms_max=args.discover_terms_max,
                discover_terms_max_per_entity=args.discover_terms_max_per_entity,
                discover_terms_eprint_dir=(Path(args.discover_terms_eprint_dir)
                                           if args.discover_terms_eprint_dir else None),
                discover_terms_eprint_max_chars=args.discover_terms_eprint_max_chars,
                discover_terms_eprint_max_tex_members=args.discover_terms_eprint_max_tex_members)

            print(f"       NER coverage: {stage5_stats['ner_coverage']:.0%} "
                  f"({stage5_stats['entities_with_ner']}/{stage5_stats['entities_processed']})")
            print(f"       Scope coverage: {stage5_stats['scope_coverage']:.0%} "
                  f"({stage5_stats['entities_with_scopes']}/{stage5_stats['entities_processed']})")
            if stage5_stats['scope_type_freq']:
                print(f"       Scope types:")
                for stype, count in stage5_stats['scope_type_freq'].items():
                    print(f"         {stype}: {count}")
            open_ner = stage5_stats.get("open_ner")
            if open_ner:
                print("       Open-world NER: "
                      f"{open_ner['candidates_written']} candidates "
                      f"(unknown_extracted={open_ner['total_unknown_extracted']}, "
                      f"unique={open_ner['unique_unknown_terms']})")
                if open_ner.get("eprint_mode"):
                    print("         eprint text coverage: "
                          f"{open_ner.get('eprint_text_used', 0)}/"
                          f"{stage5_stats['entities_processed']} "
                          f"(missing={open_ner.get('eprint_text_missing', 0)})")
            print(f"       Stage 5 done in {time.time()-t5:.0f}s")
    else:
        print(f"\n[Stage 5/{n_stages}] Skipped (--skip-ner)")

    # ========== Stage 5b: Distinctor MIT pilot ==========
    if args.run_distinctor_mit:
        t5b = time.time()
        print(f"\n[Stage 5b/{n_stages}] Distinctor MIT pilot (CPU)...")
        print(f"       Scope detector: {scope_detector_name}")
        print(f"       Entity limit: {args.distinctor_entity_limit or 'all'}")
        print(f"       Max hits: {args.distinctor_max_hits}")
        eprint_dir = Path(args.distinctor_eprint_dir) if args.distinctor_eprint_dir else None
        if eprint_dir:
            print(f"       Eprint text: {eprint_dir} "
                  f"(max_chars={args.distinctor_eprint_max_chars}, "
                  f"max_tex_members={args.distinctor_eprint_max_tex_members})")
        stage5b_stats = run_stage5b_distinctor_mit(
            entities=entities,
            pairs=pairs,
            outdir=outdir,
            scope_detector=scope_detector,
            scope_detector_name=scope_detector_name,
            entity_limit=args.distinctor_entity_limit,
            max_hits=args.distinctor_max_hits,
            seed=args.distinctor_seed,
            eprint_dir=eprint_dir,
            eprint_max_chars=args.distinctor_eprint_max_chars,
            eprint_max_tex_members=args.distinctor_eprint_max_tex_members,
        )
        print(f"       Binder coverage: {stage5b_stats['entry_binder_coverage']:.0%} "
              f"({stage5b_stats['entries_with_binder_scopes']}/{stage5b_stats['entity_count_evaluated']})")
        if stage5b_stats.get("eprint_mode"):
            print(f"       Eprint text coverage: {stage5b_stats['eprint_text_used']}/"
                  f"{stage5b_stats['entity_count_evaluated']} "
                  f"(missing={stage5b_stats['eprint_text_missing']})")
        print(f"       Candidate pairs: {stage5b_stats['candidate_pair_events']} "
              f"(unresolved={stage5b_stats['unresolved_pair_events']}, "
              f"eq={stage5b_stats['explicit_equal_pair_events']}, "
              f"neq={stage5b_stats['explicit_distinct_pair_events']})")
        print(f"       MIT labels: {stage5b_stats['mit_counts']}")
        print(f"       Outputs: {Path(stage5b_stats['summary_path']).name}, "
              f"{Path(stage5b_stats['findings_json_path']).name}, "
              f"{Path(stage5b_stats['findings_md_path']).name}, "
              f"{Path(stage5b_stats['output_hits']).name}")
        print(f"       Stage 5b done in {time.time()-t5b:.0f}s")
    else:
        print(f"\n[Stage 5b/{n_stages}] Skipped (--run-distinctor-mit not set)")

    # ========== Stage 6: Reverse morphogenesis S←Q←A ==========
    stage6_stats = None
    if args.moist_run:
        t6 = time.time()
        print(f"\n[Stage 6/{n_stages}] Generating reverse morphogenesis prompts (moist-run)...")
        moist_s6 = generate_moist_prompts(
            pairs, entities, outdir, stages=["reverse_morphogenesis"])
        stage6_stats = {
            "mode": "moist-run",
            "prompts_generated": moist_s6.get("reverse_morphogenesis", {}).get("count", 0),
        }
        print(f"       Stage 6 (moist) done in {time.time()-t6:.0f}s")
    elif not args.skip_llm:
        t6 = time.time()
        print(f"\n[Stage 6/{n_stages}] Reverse morphogenesis S←Q←A ({args.llm_model})...")
        pipe, tok = _ensure_llm_pipeline()
        rm_results = run_reverse_morphogenesis_llm_batch(
            pairs, entities, pipe, tok,
            batch_size=max(1, args.llm_batch_size // 2),
        )
        write_json(outdir / "reverse-morphogenesis.json", rm_results)

        # Quality summary
        n_parsed = sum(1 for r in rm_results if "parse_error" not in r["analysis"])
        stage6_stats = {
            "mode": "llm-inference",
            "total": len(rm_results),
            "parsed_ok": n_parsed,
            "parse_rate": n_parsed / len(rm_results) if rm_results else 0,
        }
        print(f"       {n_parsed}/{len(rm_results)} responses parsed as JSON")
        print(f"       Stage 6 done in {time.time()-t6:.0f}s")
    else:
        print(f"\n[Stage 6/{n_stages}] Skipped (--skip-llm)")

    # ========== Stage 7: Thread wiring diagrams + performatives ==========
    stage7_stats = None
    thread_diagrams = None
    ct_wiring_path = None
    if not args.skip_threads:
        t7 = time.time()

        # Build full threads (3-pass streaming)
        print(f"\n[Stage 7/{n_stages}] Building threads from {args.posts_xml}...")
        print(f"       (streaming XML, thread_limit={args.thread_limit or 'none'}, "
              f"this may take a few minutes on large dumps)")
        threads = build_threads_streaming(
            args.posts_xml,
            comments_xml_path=args.comments_xml,
            min_score=args.min_score,
            thread_limit=args.thread_limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        print(f"       {len(threads)} threads built in {time.time()-t7:.0f}s")

        if not threads:
            print(f"\n[Stage 7/{n_stages}] No threads built (0 qualifying questions)")
            stage7_stats = {"threads_processed": 0}

        # --- CT-backed path (preferred when reference exists) ---
        elif Path(args.ct_reference).exists():
            print(f"\n[Stage 7/{n_stages}] CT-backed thread wiring + IATC + categorical...")
            print(f"       CT reference: {args.ct_reference}")

            with open(args.ct_reference) as f:
                ct_reference = json.load(f)

            _, nw = _load_ct_modules()
            ner_path = Path(args.ner_kernel)
            if ner_path.exists():
                ct_singles, ct_multi, ct_ncount = nw.load_ner_kernel(ner_path)
                print(f"       NER kernel: {len(ct_singles)} single + {ct_ncount} multi terms")
            else:
                ct_singles, ct_multi = None, None
                print(f"       NER kernel not found at {ner_path}, skipping NER")

            stage7_stats, ct_wiring_path = run_stage7_ct_wiring(
                threads, ct_reference, ct_singles, ct_multi,
                site=args.site, outdir=outdir)

            print(f"       {stage7_stats['threads_processed']} threads, "
                  f"{stage7_stats['total_nodes']} nodes, "
                  f"{stage7_stats['total_edges']} edges")
            print(f"       Categorical: {stage7_stats['n_categorical']} "
                  f"({dict(list(stage7_stats.get('categorical_types', {}).items())[:5])})")
            print(f"       Port matches: {stage7_stats['n_port_matches']}")
            if stage7_stats.get('iatc_types'):
                print(f"       IATC types: {dict(list(stage7_stats['iatc_types'].items())[:6])}")
            print(f"       Written {ct_wiring_path} "
                  f"({os.path.getsize(ct_wiring_path) / 1e6:.1f} MB)")

            # Moist-run: generate LLM prompts from CT wiring dicts
            if args.moist_run:
                print(f"       Generating CT-backed performative prompts (moist-run)...")
                aw, _ = _load_ct_modules()
                prompt_dir = outdir / "moist-prompts"
                prompt_dir.mkdir(parents=True, exist_ok=True)
                prompt_path = prompt_dir / "stage7-thread-performatives.jsonl"
                n_prompts = 0
                with open(ct_wiring_path) as wf, open(prompt_path, "w") as pf:
                    wirings = json.load(wf)
                    for w in wirings:
                        prompt = build_ct_performative_prompt(w)
                        record = {
                            "thread_id": w["thread_id"],
                            "stage": "thread_performatives",
                            "n_nodes": w["stats"]["n_nodes"],
                            "n_edges": w["stats"]["n_edges"],
                            "prompt": prompt,
                        }
                        pf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_prompts += 1
                print(f"       Written {prompt_path} ({n_prompts} prompts)")

            print(f"       Stage 7 done in {time.time()-t7:.0f}s")

        # --- Fallback: old thread_performatives path ---
        else:
            print(f"\n[Stage 7/{n_stages}] Thread wiring diagrams + IATC performatives...")
            print(f"       (CT reference not found at {args.ct_reference}, "
                  f"using legacy pipeline)")

            # Classical performative detection + wiring diagram construction
            thread_diagrams, stage7_stats = process_threads_to_diagrams(threads)

            print(f"       {stage7_stats['threads_processed']} threads, "
                  f"{stage7_stats['total_nodes']} nodes, "
                  f"{stage7_stats['total_edges']} edges")
            print(f"       Classical detection rate: "
                  f"{stage7_stats['classical_edge_rate']:.0%} of edges")
            print(f"       Threads with classical: "
                  f"{stage7_stats['threads_with_classical']}/{stage7_stats['threads_processed']}")
            if stage7_stats['performative_freq']:
                print(f"       Performative types ({stage7_stats['unique_performatives']}):")
                for ptype, count in stage7_stats['performative_freq'].items():
                    print(f"         {ptype}: {count}")

            # LLM enhancement (if available, runs after classical)
            if not args.skip_llm and not args.moist_run:
                print(f"       Running LLM performative classification...")
                pipe, tok = _ensure_llm_pipeline()
                llm_enhanced = classify_thread_performatives_llm_batch(
                    thread_diagrams, pipe, tok,
                    batch_size=max(1, args.llm_batch_size // 4),
                )
                stage7_stats["llm_enhanced_threads"] = llm_enhanced
                stage7_stats["llm_enhance_rate"] = (
                    llm_enhanced / len(thread_diagrams) if thread_diagrams else 0)
                print(f"       LLM enhanced {llm_enhanced}/{len(thread_diagrams)} threads")

            # Write final wiring diagrams (with both classical + LLM edges)
            wiring_path = outdir / "thread-wiring.json"
            write_thread_wiring_json(thread_diagrams, str(wiring_path))
            print(f"       Written {wiring_path} "
                  f"({os.path.getsize(wiring_path) / 1e6:.1f} MB)")

            # Moist-run: generate LLM prompts for thread performatives
            if args.moist_run:
                print(f"       Generating thread performative prompts (moist-run)...")
                generate_moist_prompts(
                    pairs, entities, outdir,
                    stages=["thread_performatives"],
                    thread_diagrams=thread_diagrams)

            print(f"       Stage 7 done in {time.time()-t7:.0f}s")
    else:
        print(f"\n[Stage 7/{n_stages}] Skipped (--skip-threads)")

    # ========== Stage 8: Expression surface parsing ==========
    stage8_stats = None
    surfaces_path = None
    if not args.skip_expressions and not args.skip_threads and threads:
        t8 = time.time()
        print(f"\n[Stage 8/{n_stages}] Expression surface parsing (LaTeX → s-exp)...")
        stage8_stats, surfaces_path = run_stage8_expression_surfaces(
            threads, outdir)
        print(f"       {stage8_stats['total_expressions']} expressions from "
              f"{stage8_stats['threads_processed']} threads")
        print(f"       Parse rate: {stage8_stats['parse_rate']:.1%} "
              f"({stage8_stats['parsed']} parsed, "
              f"{stage8_stats['fallback']} fallback)")
        print(f"       Written {surfaces_path} "
              f"({os.path.getsize(surfaces_path) / 1e6:.1f} MB)")
        print(f"       Stage 8 done in {time.time()-t8:.0f}s")

        # Health gates
        health_gate("Stage 8", stage8_stats['parse_rate'] < 0.50,
                     f"parse rate {stage8_stats['parse_rate']:.1%} < 50% — parser is broken")
        health_gate("Stage 8", stage8_stats['parse_rate'] < 0.80,
                     f"parse rate {stage8_stats['parse_rate']:.1%} < 80% — parser needs more construct coverage")
        health_gate("Stage 8", stage8_stats['total_expressions'] == 0,
                     "zero expressions found — is the input LaTeX-free?")
    elif args.skip_expressions:
        print(f"\n[Stage 8/{n_stages}] Skipped (--skip-expressions)")
    else:
        print(f"\n[Stage 8/{n_stages}] Skipped (no threads from Stage 7)")

    # ========== Stage 9a: Hypergraph assembly ==========
    stage9a_stats = None
    if not args.skip_hypergraphs and args.arxiv_jsonl:
        t9 = time.time()
        print(f"\n[Stage 9a/{n_stages}] Paper-level hypergraph assembly (arXiv mode)...")
        paper_hg_eprint_dir = Path(args.paper_hg_eprint_dir) if args.paper_hg_eprint_dir else None
        if paper_hg_eprint_dir:
            print(f"       Paper text source: {paper_hg_eprint_dir} "
                  f"(max_chars={args.paper_hg_text_max_chars}, "
                  f"max_tex_members={args.paper_hg_max_tex_members})")
        stage9a_stats, hg_path = run_stage9a_arxiv_paper_hypergraphs(
            entities=entities,
            pairs=pairs,
            outdir=outdir,
            paper_hg_eprint_dir=paper_hg_eprint_dir,
            paper_hg_text_max_chars=args.paper_hg_text_max_chars,
            paper_hg_max_tex_members=args.paper_hg_max_tex_members,
            paper_hg_max_expressions=args.paper_hg_max_expressions,
        )
        print(f"       {stage9a_stats['hypergraphs_produced']} paper hypergraphs, "
              f"{stage9a_stats['total_nodes']} nodes, "
              f"{stage9a_stats['total_edges']} edges")
        print(f"       Avg: {stage9a_stats['avg_nodes']:.0f} nodes, "
              f"{stage9a_stats['avg_edges']:.0f} edges per paper")
        if stage9a_stats.get("paper_text_source") == "eprints":
            print(f"       Eprint text coverage: {stage9a_stats.get('eprint_text_used', 0)}/"
                  f"{stage9a_stats.get('papers_processed', 0)} "
                  f"(missing={stage9a_stats.get('eprint_text_missing', 0)})")
        print(f"       Written {hg_path} "
              f"({os.path.getsize(hg_path) / 1e6:.1f} MB)")
        print(f"       Stage 9a done in {time.time()-t9:.0f}s")

        assembly_rate = (stage9a_stats['hypergraphs_produced']
                         / stage9a_stats.get('papers_processed', 0)
                         if stage9a_stats.get('papers_processed', 0) else 0)
        health_gate("Stage 9a", assembly_rate < 0.90,
                    f"assembly rate {assembly_rate:.1%} < 90% — hypergraph schema too rigid")
        health_gate("Stage 9a", stage9a_stats['avg_nodes'] < 3,
                    f"avg {stage9a_stats['avg_nodes']:.1f} nodes/paper — hypergraphs are trivial")
    elif (not args.skip_hypergraphs and not args.skip_threads
            and threads and (ct_wiring_path or thread_diagrams)):
        t9 = time.time()
        print(f"\n[Stage 9a/{n_stages}] Hypergraph assembly...")

        # Determine wiring path (CT-backed or legacy)
        wiring_src = ct_wiring_path
        if not wiring_src:
            # Legacy path: write thread diagrams to temp file
            wiring_src = outdir / "thread-wiring.json"

        stage9a_stats, hg_path = run_stage9a_hypergraphs(
            threads, wiring_src, surfaces_path, outdir)
        print(f"       {stage9a_stats['hypergraphs_produced']} hypergraphs, "
              f"{stage9a_stats['total_nodes']} nodes, "
              f"{stage9a_stats['total_edges']} edges")
        print(f"       Avg: {stage9a_stats['avg_nodes']:.0f} nodes, "
              f"{stage9a_stats['avg_edges']:.0f} edges per thread")
        print(f"       Written {hg_path} "
              f"({os.path.getsize(hg_path) / 1e6:.1f} MB)")
        print(f"       Stage 9a done in {time.time()-t9:.0f}s")

        # Health gates
        assembly_rate = (stage9a_stats['hypergraphs_produced']
                         / stage9a_stats['threads_processed']
                         if stage9a_stats['threads_processed'] else 0)
        health_gate("Stage 9a", assembly_rate < 0.90,
                     f"assembly rate {assembly_rate:.1%} < 90% — hypergraph schema too rigid")
        health_gate("Stage 9a", stage9a_stats['avg_nodes'] < 3,
                     f"avg {stage9a_stats['avg_nodes']:.1f} nodes/thread — hypergraphs are trivial")
    elif args.skip_hypergraphs:
        print(f"\n[Stage 9a/{n_stages}] Skipped (--skip-hypergraphs)")
    else:
        print(f"\n[Stage 9a/{n_stages}] Skipped (no wiring from Stage 7)")

    # ========== Stage 9b: Graph embedding (GPU) ==========
    stage9b_stats = None
    hg_embeddings_path = None
    hg_thread_ids = None
    hg_path = outdir / "hypergraphs.json"
    if (not args.skip_graph_embed and stage9a_stats is not None
            and hg_path.exists()):
        t9b = time.time()
        print(f"\n[Stage 9b/{n_stages}] Graph embedding "
              f"(R-GCN, {args.graph_embed_dim}d, {args.graph_embed_epochs} epochs, "
              f"bs={args.graph_embed_batch_size}, workers={args.graph_embed_workers})...")
        stage9b_stats, hg_embeddings_path, model_path, hg_thread_ids = \
            run_stage9b_graph_embedding(
                hg_path, outdir,
                embed_dim=args.graph_embed_dim,
                epochs=args.graph_embed_epochs,
                batch_size=args.graph_embed_batch_size,
                num_workers=args.graph_embed_workers,
            )
        print(f"       {stage9b_stats['n_embedded']} thread embeddings "
              f"({stage9b_stats['embed_dim']}d) on {stage9b_stats['device']}")
        tm = stage9b_stats.get("train_metrics") or {}
        if tm.get("val_acc1_final") is not None:
            print("       Validation retrieval: "
                  f"Acc@1={tm['val_acc1_final']:.3f} "
                  f"Acc@5={tm['val_acc5_final']:.3f} "
                  f"MRR={tm['val_mrr_final']:.3f} "
                  f"(best Acc@1={tm.get('best_val_acc1', 0.0):.3f} "
                  f"@ epoch {tm.get('best_val_epoch')})")
        print(f"       Written {hg_embeddings_path} "
              f"({os.path.getsize(hg_embeddings_path) / 1e6:.1f} MB)")
        print(f"       Model: {model_path}")

        # Inline embedding quality check
        _emb = np.load(str(hg_embeddings_path))
        if len(_emb) >= 10:
            _sample = _emb[:min(500, len(_emb))]
            _norms = np.linalg.norm(_sample, axis=1)
            _normed = _sample / (_norms[:, None] + 1e-8)
            _cos = (_normed @ _normed.T)
            _mask = ~np.eye(len(_sample), dtype=bool)
            _avg_cos = float(_cos[_mask].mean())
            print(f"       Embedding health: avg_pairwise_cosine={_avg_cos:.4f} "
                  f"(want ~0, degenerate if >0.9)")
            stage9b_stats["avg_pairwise_cosine"] = round(_avg_cos, 4)

            health_gate("Stage 9b", _avg_cos > 0.9,
                         f"DEGENERATE embeddings (avg cosine {_avg_cos:.3f}) — training failed")
            health_gate("Stage 9b", _avg_cos > 0.5,
                         f"high avg cosine ({_avg_cos:.3f}) — embeddings may be collapsing")

        print(f"       Stage 9b done in {time.time()-t9b:.0f}s")
    elif args.skip_graph_embed:
        print(f"\n[Stage 9b/{n_stages}] Skipped (--skip-graph-embed)")
    else:
        print(f"\n[Stage 9b/{n_stages}] Skipped (no hypergraphs from Stage 9a)")

    # ========== Stage 10: FAISS similarity index ==========
    stage10_stats = None
    if (not args.skip_faiss and hg_embeddings_path is not None
            and hg_thread_ids is not None):
        t10 = time.time()
        print(f"\n[Stage 10/{n_stages}] Building structural similarity index...")
        stage10_stats, index_path = run_stage10_faiss_index(
            hg_embeddings_path, hg_thread_ids, outdir)
        print(f"       {stage10_stats['n_vectors']} vectors "
              f"({stage10_stats['dimension']}d)")
        if stage10_stats.get("sample_query"):
            sq = stage10_stats["sample_query"]
            print(f"       Sample: thread {sq['query_thread']} → "
                  f"neighbors {sq['top_5'][:3]}")
        print(f"       Written {index_path}.*")
        print(f"       Stage 10 done in {time.time()-t10:.0f}s")
    elif args.skip_faiss:
        print(f"\n[Stage 10/{n_stages}] Skipped (--skip-faiss)")
    else:
        print(f"\n[Stage 10/{n_stages}] Skipped (no embeddings from Stage 9b)")

    # ========== Manifest ==========
    elapsed = time.time() - t0
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.site,
        "posts_xml": args.posts_xml,
        "arxiv_jsonl": args.arxiv_jsonl,
        "min_score": args.min_score,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "embed_model": args.embed_model if not args.skip_embeddings else None,
        "llm_model": args.llm_model if not args.skip_llm else None,
        "stats": stats,
        "entity_count": len(entities),
        "tag_count": len(tags),
        "relation_count": len(all_relations),
        "elapsed_seconds": round(elapsed),
        "moist_run": args.moist_run,
        "discover_terms": args.discover_terms,
        "discover_terms_eprint_dir": args.discover_terms_eprint_dir,
        "run_distinctor_mit": args.run_distinctor_mit,
        "distinctor_eprint_dir": args.distinctor_eprint_dir,
        "paper_hg_eprint_dir": args.paper_hg_eprint_dir,
        "stages_completed": [
            "parse",
            *([] if args.skip_embeddings else ["embeddings"]),
            *(["llm_pattern_tags_moist"] if args.moist_run else
              ([] if args.skip_llm else ["llm_pattern_tags"])),
            *([] if args.skip_clustering or embeddings is None else ["clustering"]),
            *([] if args.skip_ner or stage5_stats is None else ["ner_scopes"]),
            *([] if (not args.run_distinctor_mit) or stage5b_stats is None else ["distinctor_mit"]),
            *([] if stage6_stats is None else ["reverse_morphogenesis"]),
            *([] if stage7_stats is None else ["thread_wiring"]),
            *([] if stage8_stats is None else ["expression_surfaces"]),
            *([] if stage9a_stats is None else ["hypergraphs"]),
            *([] if stage9b_stats is None else ["graph_embedding"]),
            *([] if stage10_stats is None else ["faiss_index"]),
        ],
        "stage5_stats": stage5_stats,
        "stage5b_stats": stage5b_stats,
        "stage6_stats": stage6_stats,
        "stage7_stats": stage7_stats,
        "stage8_stats": stage8_stats,
        "stage9a_stats": stage9a_stats,
        "stage9b_stats": stage9b_stats,
        "stage10_stats": stage10_stats,
        "output_files": [f.name for f in outdir.iterdir() if f.is_file()],
        "patterns": PATTERN_NAMES,
    }
    write_json(outdir / "manifest.json", manifest)

    print(f"\n{'='*60}")
    print(f"Job complete in {elapsed/60:.1f} min")
    print(f"Output: {outdir}/")
    for f in sorted(outdir.iterdir()):
        if f.is_file():
            print(f"  {f.name:30s} {os.path.getsize(f)/1e6:8.1f} MB")
        elif f.is_dir():
            sub_files = list(f.iterdir())
            sub_size = sum(sf.stat().st_size for sf in sub_files if sf.is_file())
            print(f"  {f.name + '/':30s} {sub_size/1e6:8.1f} MB ({len(sub_files)} files)")
    if args.moist_run:
        prompt_dir = outdir / "moist-prompts"
        if prompt_dir.exists():
            print(f"\nMoist-run prompt files:")
            for f in sorted(prompt_dir.iterdir()):
                if f.suffix == ".jsonl":
                    n_lines = sum(1 for _ in open(f))
                    print(f"  {f.name:40s} {n_lines:>8,} prompts")
            print(f"\nTo submit to Codex:")
            print(f"  codex --prompt-file {prompt_dir}/stage3-pattern-tagging.jsonl")
            print(f"  codex --prompt-file {prompt_dir}/stage6-reverse-morphogenesis.jsonl")
            print(f"  codex --prompt-file {prompt_dir}/stage7-thread-performatives.jsonl")
    print(f"\nTo upload: tar czf {outdir.name}.tar.gz {outdir.name}/")


_llm_start = 0  # module-level for rate tracking

if __name__ == "__main__":
    main()
