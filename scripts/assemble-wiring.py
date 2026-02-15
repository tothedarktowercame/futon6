#!/usr/bin/env python3
"""Hierarchical Wiring Assembly for SE/MO Threads.

Assembles three-level nested wiring diagrams from StackExchange/MathOverflow
threads using the CT reference extracted from nLab:

  Level 1 (Thread): posts as nodes, IATC performative edges
  Level 2 (Categorical): CT patterns detected in each post
  Level 3 (Diagram): commutative diagrams extracted from post LaTeX

Usage:
    python scripts/assemble-wiring.py assemble \
        --threads data/stackexchange-samples/ \
        --reference data/nlab-ct-reference.json \
        --output-dir data/thread-wiring

    python scripts/assemble-wiring.py inspect \
        --wiring data/thread-wiring/mathoverflow.net__category-theory.json \
        --thread-id 178778

Memory-safe, CPU-only, laptop-scale.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Import from nlab-wiring.py
sys.path.insert(0, str(Path(__file__).parent))
import importlib
nlab_wiring = importlib.import_module("nlab-wiring")

NER_KERNEL = Path("data/ner-kernel/terms.tsv")


# ============================================================
# IATC Performative Detection
# ============================================================

IATC_PATTERNS = {
    "assert": re.compile(
        r"\b(?:we have|it follows|this shows|this proves|indeed|clearly|"
        r"one can show|it is easy to see|the result follows|QED|"
        r"we conclude|this gives|this yields)\b", re.IGNORECASE),
    "challenge": re.compile(
        r"\b(?:but (?:what|why|how|this)|however|is this|are you sure|"
        r"I don't think|I don't see|this seems wrong|doesn't this|"
        r"what about|isn't it|I disagree|that can't be right)\b", re.IGNORECASE),
    "query": re.compile(
        r"\b(?:what (?:is|are|about)|how (?:do|does|can|would)|"
        r"why (?:is|does|do|should)|is there|can (?:you|we|one)|"
        r"could you|do you mean|I wonder)\b", re.IGNORECASE),
    "clarify": re.compile(
        r"\b(?:to clarify|more precisely|what I mean|in other words|"
        r"that is(?: to say)?|to be precise|I should clarify|"
        r"let me rephrase|to put it differently)\b", re.IGNORECASE),
    "reform": re.compile(
        r"\b(?:alternatively|another way|can be rephrased|equivalently|"
        r"a different approach|one could also|rephrasing|"
        r"another proof|a simpler argument)\b", re.IGNORECASE),
    "exemplify": re.compile(
        r"\b(?:for example|for instance|e\.g\.|consider the case|"
        r"as an example|take for instance|a concrete example|"
        r"to illustrate)\b", re.IGNORECASE),
    "reference": re.compile(
        r"\b(?:see |cf\.|as shown in|as proved in|by theorem|"
        r"according to|it is known that|as in \[|"
        r"this is proved in|following \w+ \()\b", re.IGNORECASE),
    "agree": re.compile(
        r"\b(?:yes[,.]|right[,.]|exactly|I agree|that's correct|good point|"
        r"nice|that's right|indeed so|you're right)\b", re.IGNORECASE),
    "retract": re.compile(
        r"\b(?:actually.*wrong|I was mistaken|correction[:\s]|erratum|"
        r"my mistake|I take (?:that|it) back|"
        r"I stand corrected|upon reflection)\b", re.IGNORECASE),
}


def detect_iatc(text):
    """Detect IATC performative types in text.

    Returns list of (type, match_text, position) sorted by position.
    """
    hits = []
    for iatc_type, pattern in IATC_PATTERNS.items():
        for m in pattern.finditer(text):
            hits.append((iatc_type, m.group(), m.start()))
    hits.sort(key=lambda x: x[2])
    return hits


def classify_edge_iatc(source_text, source_type):
    """Classify the IATC type of an edge based on source node's text.

    Returns the dominant IATC type or a default based on node type.
    """
    hits = detect_iatc(source_text)
    if hits:
        # Count by type, return most frequent
        counts = Counter(h[0] for h in hits)
        return counts.most_common(1)[0][0]
    # Defaults
    if source_type == "answer":
        return "assert"
    elif source_type == "comment":
        return "clarify"
    return "assert"


# ============================================================
# Port Extraction from Discourse Scopes
# ============================================================

def extract_ports(text, entity_id):
    """Extract input and output ports from text using discourse scope analysis.

    Input ports: assumptions, let-bindings, given conditions (before conclusions)
    Output ports: conclusions, established results (after consequential wires)

    Returns (input_ports, output_ports) as lists of port dicts.
    """
    scopes = nlab_wiring.detect_scopes(entity_id, text)
    wires = nlab_wiring.detect_wires(entity_id, text)

    input_ports = []
    output_ports = []

    # Wire positions mark the boundary between assumption and conclusion zones
    consequential_positions = []
    for w in wires:
        if w["hx/type"] in ("wire/consequential", "wire/causal"):
            consequential_positions.append(w["hx/content"]["position"])

    earliest_conclusion = min(consequential_positions) if consequential_positions else len(text)

    for scope in scopes:
        pos = scope["hx/content"]["position"]
        scope_type = scope["hx/type"]
        match_text = scope["hx/content"]["match"][:80]

        # Extract a label from the scope
        label = _scope_to_label(scope)

        port = {
            "id": scope["hx/id"],
            "type": scope_type,
            "label": label,
            "text": match_text,
            "position": pos,
        }

        # Scopes before consequential wires are inputs (assumptions)
        # Scopes after are outputs (conclusions)
        if pos < earliest_conclusion:
            if scope_type in ("bind/let", "bind/define", "assume/explicit",
                              "assume/consider", "quant/universal",
                              "constrain/where", "constrain/such-that"):
                input_ports.append(port)
        else:
            output_ports.append(port)

    return input_ports, output_ports


def _scope_to_label(scope):
    """Extract a human-readable label from a scope record."""
    ends = scope.get("hx/ends", [])
    parts = []
    for end in ends:
        if end.get("role") == "symbol":
            parts.append(end.get("latex", end.get("text", "")))
        elif end.get("role") in ("type", "description", "value"):
            parts.append(end.get("text", end.get("latex", "")))
        elif end.get("role") == "condition":
            parts.append(end.get("latex", ""))
        elif end.get("role") == "object":
            parts.append(end.get("text", ""))
    return " : ".join(parts) if parts else scope["hx/content"]["match"][:40]


def match_ports(source_outputs, target_inputs, reference):
    """Find port matches between source output ports and target input ports.

    Uses term overlap between port labels, boosted by CT reference term weights.
    Returns list of (source_port_id, target_port_id, match_score).
    """
    link_weights = reference.get("link_weights", {})
    matches = []

    for out_port in source_outputs:
        out_terms = set(out_port["label"].lower().split())
        for in_port in target_inputs:
            in_terms = set(in_port["label"].lower().split())
            overlap = out_terms & in_terms
            if not overlap:
                continue
            # Score: count + weight from reference
            score = len(overlap)
            for term in overlap:
                if term in link_weights:
                    score += 0.5  # bonus for CT-relevant term
            if score > 0:
                matches.append((out_port["id"], in_port["id"], round(score, 2)))

    matches.sort(key=lambda x: -x[2])
    return matches


# ============================================================
# Per-node Categorical Annotation (adapted for SE text)
# ============================================================

def _compute_term_idf(reference):
    """Compute inverse document frequency for terms across patterns.

    Terms appearing in many patterns are generic (low IDF);
    terms unique to 1-2 patterns are discriminative (high IDF).
    """
    patterns = reference.get("patterns", {})
    n_patterns = len(patterns)
    if n_patterns == 0:
        return {}

    # Count how many patterns each term appears in
    term_df = Counter()
    for pat_data in patterns.values():
        seen = set()
        for link in pat_data.get("required_links", []):
            seen.add(link.lower())
        for link in pat_data.get("typical_links", []):
            seen.add(link.lower())
        for t in seen:
            term_df[t] += 1

    # IDF-like weight: terms in 1 pattern get weight 1.0,
    # terms in all patterns get weight ~0
    idf = {}
    for term, df in term_df.items():
        idf[term] = max(0.0, 1.0 - (df - 1) / max(n_patterns - 1, 1))
    return idf


def detect_categorical_for_se(text, tags, reference):
    """Detect categorical patterns in SE answer/question text.

    Adapted from nlab-wiring.detect_categorical_patterns() for SE text
    which uses $...$ LaTeX instead of [[wiki links]].

    Uses term IDF to discount ubiquitous terms (category, functor) and
    reward pattern-specific terms (adjoint, monad, Kan extension).
    """
    patterns = reference.get("patterns", {})
    plain = nlab_wiring.strip_nlab_markup(text)
    text_lower = plain.lower()

    # Pre-compute term IDF if not cached
    if not hasattr(detect_categorical_for_se, "_idf_cache"):
        detect_categorical_for_se._idf_cache = _compute_term_idf(reference)
    idf = detect_categorical_for_se._idf_cache

    hyperedges = []
    for cat_type, pat_data in patterns.items():
        score = 0.0
        evidence = []
        has_discriminative = False

        # Check required_links as terms in text (IDF-weighted)
        for link in pat_data.get("required_links", []):
            if link.lower() in text_lower:
                w = idf.get(link.lower(), 0.5)
                score += 2 * w
                evidence.append(f"req:{link}")
                if w >= 0.5:
                    has_discriminative = True

        # Check typical_links as terms in text (IDF-weighted)
        for link in pat_data.get("typical_links", []):
            if link.lower() in text_lower:
                w = idf.get(link.lower(), 0.5)
                score += 1 * w
                evidence.append(f"typ:{link}")
                if w >= 0.5:
                    has_discriminative = True

        # Check question tags — only boost if tag is specific to this pattern
        cat_key = cat_type.split("/")[1].replace("-", " ")
        for tag in (tags or []):
            tag_lower = tag.lower().replace("-", " ")
            if cat_key in tag_lower or tag_lower in cat_key:
                score += 2
                evidence.append(f"tag:{tag}")
                has_discriminative = True

        # Text signals from nlab-wiring CAT_PATTERNS
        nlab_pattern = nlab_wiring.CAT_PATTERNS.get(cat_type, {})
        for sig_re in nlab_pattern.get("text_signals", []):
            if re.search(sig_re, plain, re.IGNORECASE):
                score += 1.5
                evidence.append(f"text:{sig_re[:25]}")
                has_discriminative = True

        # Require at least one discriminative signal (not just generic CT terms)
        min_signals = max(nlab_pattern.get("min_signals", 3), 3)
        if score >= min_signals and has_discriminative:
            hyperedges.append({
                "hx/type": cat_type,
                "hx/content": {"evidence": evidence, "score": round(score, 1)},
            })

    return hyperedges


# ============================================================
# Per-node Diagram Extraction
# ============================================================

def extract_diagrams_from_text(text, entity_id):
    """Extract tikzcd and array diagrams from SE post text."""
    # SE posts use \begin{tikzcd}...\end{tikzcd} and \array{...}
    diagrams = nlab_wiring.extract_diagrams(entity_id, text, [])
    return diagrams


# ============================================================
# Thread Graph Construction
# ============================================================

def build_thread_graph(thread, reference, singles=None, multi_index=None):
    """Build a hierarchical wiring diagram from an SE/MO thread.

    Args:
        thread: normalised thread dict from load_threads()
            For raw JSONL, call load_raw_thread() first.
        reference: CT reference dictionary
        singles, multi_index: NER kernel

    Returns nested wiring diagram dict.
    """
    q = thread
    q_id = q["id"]
    q_text = q.get("body", "")
    q_title = q.get("title", "")
    q_tags = q.get("tags", [])
    site = q.get("site", "")
    topic = q.get("topic", "")
    answers = q.get("answers", [])
    comments = q.get("comments", {})

    nodes = []
    edges = []

    # --- Question node ---
    q_full = q_title + "\n" + q_text
    q_input_ports, q_output_ports = extract_ports(q_full, f"q-{q_id}")
    q_categorical = detect_categorical_for_se(q_full, q_tags, reference)
    q_diagrams = extract_diagrams_from_text(q_full, f"q-{q_id}")
    q_discourse = _extract_discourse(q_full, f"q-{q_id}")

    # NER terms
    q_ner = []
    if singles:
        q_ner = nlab_wiring.spot_terms(q_full, singles, multi_index or {})

    nodes.append({
        "id": f"q-{q_id}",
        "type": "question",
        "title": q_title,
        "score": q.get("score", 0),
        "tags": q_tags,
        "input_ports": q_input_ports,
        "output_ports": q_output_ports,
        "categorical": q_categorical,
        "diagrams": q_diagrams,
        "discourse": q_discourse,
        "ner_terms": q_ner,
        "text_length": len(q_full),
    })

    # --- Answer nodes ---
    for answer in answers:
        a_id = answer["id"]
        a_text = answer.get("body", "")
        is_accepted = answer.get("is_accepted", False)
        entity_id = f"a-{a_id}"

        a_input_ports, a_output_ports = extract_ports(a_text, entity_id)
        a_categorical = detect_categorical_for_se(a_text, q_tags, reference)
        a_diagrams = extract_diagrams_from_text(a_text, entity_id)
        a_discourse = _extract_discourse(a_text, entity_id)
        a_ner = []
        if singles:
            a_ner = nlab_wiring.spot_terms(a_text, singles, multi_index or {})

        iatc = classify_edge_iatc(a_text, "answer")

        nodes.append({
            "id": entity_id,
            "type": "answer",
            "score": answer.get("score", 0),
            "is_accepted": is_accepted,
            "input_ports": a_input_ports,
            "output_ports": a_output_ports,
            "categorical": a_categorical,
            "diagrams": a_diagrams,
            "discourse": a_discourse,
            "ner_terms": a_ner,
            "text_length": len(a_text),
        })

        # Edge: answer → question
        port_matches = match_ports(a_output_ports, q_output_ports, reference)
        edges.append({
            "from": entity_id,
            "to": f"q-{q_id}",
            "type": "responds-to",
            "iatc": iatc,
            "port_matches": port_matches[:3],  # top 3
        })

    # --- Comment nodes ---
    # Handle both dict format (Codex) and list format
    comment_list = _flatten_comments(comments)
    for comment in comment_list:
        c_id = comment.get("id", comment.get("comment_id"))
        c_text = comment.get("text", comment.get("body_text", ""))
        parent_id = comment.get("post_id", comment.get("parent_id"))
        entity_id = f"c-{c_id}"

        c_input_ports, c_output_ports = extract_ports(c_text, entity_id)
        iatc = classify_edge_iatc(c_text, "comment")
        c_discourse = _extract_discourse(c_text, entity_id)

        nodes.append({
            "id": entity_id,
            "type": "comment",
            "score": comment.get("score", 0),
            "input_ports": c_input_ports,
            "output_ports": c_output_ports,
            "categorical": [],  # comments rarely have full CT patterns
            "diagrams": [],
            "discourse": c_discourse,
            "text_length": len(c_text),
        })

        # Edge: comment → parent post
        parent_node_id = _find_node_id(parent_id, q_id, answers)
        edges.append({
            "from": entity_id,
            "to": parent_node_id,
            "type": "comments-on",
            "iatc": iatc,
            "port_matches": [],
        })

    # --- Aggregate statistics ---
    all_categorical = []
    all_diagrams = []
    for n in nodes:
        all_categorical.extend(n.get("categorical", []))
        all_diagrams.extend(n.get("diagrams", []))

    cat_summary = Counter(c["hx/type"] for c in all_categorical)

    return {
        "thread_id": q_id,
        "site": site,
        "topic": topic,
        "title": q_title,
        "level": "thread",
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_answers": sum(1 for n in nodes if n["type"] == "answer"),
            "n_comments": sum(1 for n in nodes if n["type"] == "comment"),
            "n_categorical": len(all_categorical),
            "n_diagrams": len(all_diagrams),
            "categorical_types": dict(cat_summary),
            "n_port_matches": sum(len(e.get("port_matches", [])) for e in edges),
            "iatc_types": dict(Counter(e["iatc"] for e in edges)),
        },
    }


def _extract_discourse(text, entity_id):
    """Extract discourse records (scopes + wires + ports + labels) from text."""
    records = []
    records.extend(nlab_wiring.detect_scopes(entity_id, text))
    records.extend(nlab_wiring.detect_wires(entity_id, text))
    records.extend(nlab_wiring.detect_ports(entity_id, text))
    records.extend(nlab_wiring.detect_labels(entity_id, text))
    return records


def _flatten_comments(comments):
    """Flatten the comments structure from SE thread format.

    Comments can be:
    - A dict with {"question": [...], "answers": {id: [...], ...}}
    - A list of comment dicts
    - Empty
    """
    if isinstance(comments, list):
        return comments
    if isinstance(comments, dict):
        flat = []
        for c in comments.get("question", []):
            flat.append(c)
        answers_comments = comments.get("answers", {})
        if isinstance(answers_comments, dict):
            for aid, clist in answers_comments.items():
                if isinstance(clist, list):
                    flat.extend(clist)
        elif isinstance(answers_comments, list):
            flat.extend(answers_comments)
        return flat
    return []


def _find_node_id(parent_post_id, question_id, answers):
    """Map a parent_post_id to a node ID in our graph."""
    if parent_post_id == question_id or str(parent_post_id) == str(question_id):
        return f"q-{question_id}"
    for a in answers:
        if str(a["id"]) == str(parent_post_id):
            return f"a-{a['id']}"
    return f"post-{parent_post_id}"


# ============================================================
# Thread Loading (from JSONL with full comments)
# ============================================================

def load_thread_with_comments(path):
    """Load threads from JSONL preserving comment structure.

    Unlike nlab-wiring.load_threads() which normalises for flat evaluation,
    this preserves the full comment tree needed for wiring assembly.
    """
    threads = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            q = raw["question"]
            accepted_id = q.get("accepted_answer_id")
            answers = []
            for a in raw.get("answers", []):
                a_id = a.get("id")
                answers.append({
                    "id": a_id,
                    "body": a.get("body_text", a.get("body", "")),
                    "score": a.get("score", 0),
                    "is_accepted": str(a_id) == str(accepted_id) if accepted_id else False,
                })
            threads.append({
                "id": q.get("id", raw.get("thread_id", "unknown")),
                "body": q.get("body_text", q.get("body", "")),
                "title": q.get("title", ""),
                "tags": q.get("tags", []),
                "score": q.get("score", 0),
                "site": raw.get("site", ""),
                "topic": raw.get("topic", ""),
                "answers": answers,
                "comments": raw.get("comments", {}),
            })
    return threads


# ============================================================
# Subcommand: assemble
# ============================================================

def cmd_assemble(args):
    """Assemble nested wiring diagrams from SE/MO threads."""
    threads_path = Path(args.threads)
    ref_path = Path(args.reference)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reference from {ref_path}...")
    with open(ref_path) as f:
        reference = json.load(f)

    # Load NER kernel
    ner_path = Path(args.ner_kernel)
    singles, multi_index = None, None
    if ner_path.exists():
        singles, multi_index, _ = nlab_wiring.load_ner_kernel(ner_path)
        print(f"NER kernel: {len(singles)} single + many multi-word terms")

    # Collect input files
    if threads_path.is_dir():
        input_files = sorted(threads_path.glob("*.jsonl"))
    else:
        input_files = [threads_path]

    t0 = time.time()
    grand_stats = Counter()

    for input_file in input_files:
        label = input_file.stem
        print(f"\nAssembling {label}...")
        threads = load_thread_with_comments(input_file)
        print(f"  {len(threads)} threads")

        wirings = []
        file_stats = Counter()

        for thread in threads:
            wiring = build_thread_graph(thread, reference, singles, multi_index)
            wirings.append(wiring)

            s = wiring["stats"]
            file_stats["threads"] += 1
            file_stats["nodes"] += s["n_nodes"]
            file_stats["edges"] += s["n_edges"]
            file_stats["categorical"] += s["n_categorical"]
            file_stats["diagrams"] += s["n_diagrams"]
            file_stats["port_matches"] += s["n_port_matches"]

        # Write output
        out_path = outdir / f"{label}.json"
        with open(out_path, "w") as f:
            json.dump(wirings, f, indent=2, ensure_ascii=False)

        print(f"  Nodes: {file_stats['nodes']}, Edges: {file_stats['edges']}")
        print(f"  Categorical: {file_stats['categorical']}, Diagrams: {file_stats['diagrams']}")
        print(f"  Port matches: {file_stats['port_matches']}")
        print(f"  Output: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

        grand_stats += file_stats

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Assembly complete in {elapsed:.0f}s")
    print(f"  Total threads: {grand_stats['threads']}")
    print(f"  Total nodes: {grand_stats['nodes']}")
    print(f"  Total edges: {grand_stats['edges']}")
    print(f"  Total categorical: {grand_stats['categorical']}")
    print(f"  Total diagrams: {grand_stats['diagrams']}")
    print(f"  Total port matches: {grand_stats['port_matches']}")


# ============================================================
# Subcommand: inspect
# ============================================================

def cmd_inspect(args):
    """Inspect a specific thread wiring diagram."""
    wiring_path = Path(args.wiring)
    target_id = args.thread_id

    with open(wiring_path) as f:
        wirings = json.load(f)

    for w in wirings:
        if str(w["thread_id"]) == str(target_id):
            print(f"Thread {w['thread_id']}: \"{w['title']}\"")
            print(f"  Site: {w['site']}, Topic: {w['topic']}")
            print(f"  Nodes: {w['stats']['n_nodes']} "
                  f"({w['stats']['n_answers']} answers, "
                  f"{w['stats']['n_comments']} comments)")
            print(f"  Edges: {w['stats']['n_edges']}")
            print(f"  Categorical: {w['stats']['n_categorical']}")
            print(f"  Diagrams: {w['stats']['n_diagrams']}")
            print(f"  Port matches: {w['stats']['n_port_matches']}")
            print(f"  IATC types: {w['stats']['iatc_types']}")
            print(f"  Categorical types: {w['stats']['categorical_types']}")

            print(f"\n  Nodes:")
            for n in w["nodes"]:
                acc = " *ACCEPTED*" if n.get("is_accepted") else ""
                n_cat = len(n.get("categorical", []))
                n_diag = len(n.get("diagrams", []))
                n_in = len(n.get("input_ports", []))
                n_out = len(n.get("output_ports", []))
                print(f"    {n['id']} [{n['type']}] score={n.get('score',0)}{acc}"
                      f" cat={n_cat} diag={n_diag} ports={n_in}in/{n_out}out"
                      f" chars={n.get('text_length',0)}")
                for cat in n.get("categorical", []):
                    print(f"      {cat['hx/type']} score={cat['hx/content']['score']}")
                for p in n.get("input_ports", [])[:3]:
                    print(f"      IN: {p['type']} — {p['label'][:60]}")
                for p in n.get("output_ports", [])[:3]:
                    print(f"      OUT: {p['type']} — {p['label'][:60]}")

            print(f"\n  Edges:")
            for e in w["edges"]:
                pm = f" ports={len(e.get('port_matches',[]))}" if e.get("port_matches") else ""
                print(f"    {e['from']} →[{e['iatc']}]→ {e['to']} ({e['type']}){pm}")
                for pm_entry in e.get("port_matches", [])[:2]:
                    print(f"      match: {pm_entry}")
            return

    print(f"Thread {target_id} not found in {wiring_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Wiring Assembly for SE/MO Threads")
    sub = parser.add_subparsers(dest="command")

    # assemble
    asm_p = sub.add_parser("assemble", help="Build nested wiring from threads")
    asm_p.add_argument("--threads", required=True,
                       help="Path to JSONL file or directory of JSONL files")
    asm_p.add_argument("--reference", default="data/nlab-ct-reference.json")
    asm_p.add_argument("--ner-kernel", default=str(NER_KERNEL))
    asm_p.add_argument("--output-dir", "-o", default="data/thread-wiring")

    # inspect
    ins_p = sub.add_parser("inspect", help="Inspect a thread wiring diagram")
    ins_p.add_argument("--wiring", required=True, help="Path to wiring JSON file")
    ins_p.add_argument("--thread-id", required=True, help="Thread ID to inspect")

    args = parser.parse_args()

    if args.command == "assemble":
        cmd_assemble(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
