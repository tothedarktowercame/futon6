#!/usr/bin/env python3
"""Build the golden-50 test suite for superpod pipeline validation.

Selects 50 physics.SE QA pairs stratified by difficulty, annotates each with:
- NER term hits (from the 19K-term kernel)
- Pattern tags (25 math-informal patterns via hotword scoring)
- Scope records (Let/Define/Assume/Consider/where detection)

Output: tests/golden/golden-50.json
  A self-contained file that ships with the superpod job.
  The superpod runs these 50 first as a smoke test.

Usage:
    python scripts/build-golden-50.py
"""

import json
import os
import re
import sys
import random
from collections import Counter
from pathlib import Path

# ---------- Config ----------

DATA = Path("data")
SE_JSON = DATA / "se-physics.json"
TERMS_TSV = DATA / "ner-kernel" / "terms.tsv"
SCOPE_PATTERNS_EDN = DATA / "ner-kernel" / "scope-patterns.edn"
OUTPUT = Path("tests/golden/golden-50.json")

# Stratification: 17 easy (score >= 20), 17 medium (5-19), 16 hard (1-4, where
# answers are longer and more technical — "hard" means less community signal,
# forcing the pipeline to work harder)
STRATA = [
    ("easy", lambda e: e["score"] >= 20, 17),
    ("medium", lambda e: 5 <= e["score"] < 20, 17),
    ("hard", lambda e: 1 <= e["score"] < 5, 16),
]

SEED = 42  # reproducible

# ---------- The 25 math-informal patterns (same as superpod-job.py) ----------

PATTERNS = [
    ("try-a-simpler-case", "Reduce parameters to the smallest non-trivial value",
     ["simpler", "simplest", "trivial case", "degenerate", "small case", "toy"]),
    ("work-examples-first", "Compute concrete examples before general proof",
     ["example", "for instance", "e.g.", "consider the case", "illustrate"]),
    ("argue-by-contradiction", "Assume negation and derive absurdity",
     ["contradiction", "absurd", "suppose not", "assume the contrary", "impossible"]),
    ("find-the-right-abstraction", "Seek higher-level framework",
     ["abstraction", "generalise", "generalize", "framework", "viewpoint", "perspective"]),
    ("quotient-by-irrelevance", "Collapse via equivalence relation",
     ["quotient", "equivalence class", "mod ", "modulo", "up to"]),
    ("check-the-extreme-cases", "Test at zero, infinity, boundary",
     ["extreme", "boundary", "limiting case", "zero", "infinity", "degenerate"]),
    ("exploit-symmetry", "Use symmetry to reduce work",
     ["symmetry", "symmetric", "WLOG", "without loss of generality", "invariant"]),
    ("construct-an-explicit-witness", "Prove existence by building",
     ["construct", "explicit", "witness", "exhibit", "build"]),
    ("dualise-the-problem", "Reverse arrows, take complements",
     ["dual", "adjoint", "complement", "transpose", "contravariant"]),
    ("reduce-to-known-result", "Transform until matching known theorem",
     ["by theorem", "known result", "well-known", "classical result", "it is known"]),
    ("pass-to-a-subsequence", "Extract convergent subsequence",
     ["subsequence", "compactness", "Bolzano", "convergent sub"]),
    ("induction-and-well-ordering", "Base case and inductive step",
     ["induction", "inductive", "base case", "inductive step", "well-ordering"]),
    ("local-to-global", "Prove locally, assemble globally",
     ["local", "global", "locally", "globally", "patching", "sheaf", "cover"]),
    ("the-diagonal-argument", "Defeat enumeration by differing",
     ["diagonal", "Cantor", "diagonalisation", "uncountable"]),
    ("encode-as-algebra", "Translate into algebraic language",
     ["algebra", "algebraic", "polynomial", "ring", "module", "encode"]),
    ("use-probabilistic-method", "Random object with positive probability",
     ["probability", "random", "expected value", "probabilistic", "Erdos"]),
    ("unfold-the-definition", "Expand defined terms",
     ["by definition", "definition of", "unfolding", "means that", "recall that"]),
    ("construct-auxiliary-object", "Introduce mediating object",
     ["auxiliary", "helper", "introduce", "consider the function", "define the map"]),
    ("estimate-by-bounding", "Replace with simpler bound",
     ["bound", "inequality", "estimate", "at most", "at least", "upper bound", "lower bound"]),
    ("split-into-cases", "Partition into cases",
     ["case 1", "case 2", "cases:", "two cases", "three cases", "if ... then ... otherwise"]),
    ("verify-universal-property", "Check mediating morphism",
     ["universal property", "unique morphism", "commutative diagram", "mediating"]),
    ("optimise-a-free-parameter", "Choose parameter for tightest bound",
     ["optimise", "optimize", "choose", "pick", "free parameter", "minimise", "maximize"]),
    ("transport-across-isomorphism", "Prove in easier setting, transport",
     ["isomorphism", "isomorphic", "equivalent", "transport", "identify"]),
    ("show-both-inequalities", "Prove equality via two inequalities",
     ["both directions", "reverse inequality", "≤ and ≥", "other direction"]),
    ("monotone-approximation", "Approximate by monotone sequence, take limit",
     ["monotone", "increasing sequence", "decreasing sequence", "approximate", "limit"]),
]

# ---------- NER term loading (inverted index, same as O-0) ----------

def load_ner_kernel(path):
    """Load NER kernel terms. Returns (single_terms_set, multi_index, multi_count)."""
    singles = set()
    multi_index = {}  # first_content_word -> [(full_term_lower, term_original)]
    multi_count = 0
    skip_prefixes = ("$", "(", "\"", "-")

    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            term_lower = parts[0].strip()
            term_orig = parts[1].strip()

            if not term_lower or any(term_lower.startswith(p) for p in skip_prefixes):
                continue
            if len(term_lower) < 3:
                continue

            words = term_lower.split()
            if len(words) == 1:
                singles.add(term_lower)
            else:
                # Index by first content word (skip short function words)
                first_key = None
                for w in words:
                    if len(w) >= 3:
                        first_key = w
                        break
                if first_key is None:
                    first_key = words[0]
                if first_key not in multi_index:
                    multi_index[first_key] = []
                multi_index[first_key].append((term_lower, term_orig))
                multi_count += 1

    return singles, multi_index, multi_count


def spot_terms(text, singles, multi_index):
    """Spot NER terms in text. Returns list of matched terms."""
    text_lower = text.lower()
    words = text_lower.split()
    hits = []

    # Single-word matches
    for w in set(words):
        clean = w.strip(".,;:!?()[]\"'")
        if clean in singles:
            hits.append(clean)

    # Multi-word matches via inverted index
    for i, w in enumerate(words):
        clean = w.strip(".,;:!?()[]\"'")
        if clean in multi_index:
            for term_lower, term_orig in multi_index[clean]:
                term_words = term_lower.split()
                # Check if full term appears starting near position i
                term_str = term_lower
                if term_str in text_lower:
                    if term_lower not in hits:
                        hits.append(term_lower)

    return sorted(set(hits))


# ---------- Pattern matching (hotword scoring, same as tag-patterns.bb) ----------

def tag_patterns(text):
    """Score text against 25 patterns via hotword matching. Returns list of (name, score)."""
    text_lower = text.lower()
    matches = []
    for name, desc, hotwords in PATTERNS:
        score = 0
        matched_hotwords = []
        for hw in hotwords:
            if hw.lower() in text_lower:
                score += 2.0
                matched_hotwords.append(hw)
        if score > 0:
            matches.append({"pattern": name, "score": score, "hotwords": matched_hotwords})
    return sorted(matches, key=lambda x: -x["score"])


# ---------- Scope detection ----------

SCOPE_REGEXES = [
    ("let-binding", r"\bLet\s+\$[^$]+\$\s+(be|denote)"),
    ("define", r"\bDefine\s+\$[^$]+\$\s*(:=|=|\\equiv)"),
    ("assume", r"\b(Assume|Suppose)\s+(that\s+)?\$"),
    ("consider", r"\bConsider\s+(a|an|the|some)?\s*\$?[^$.]{0,60}"),
    ("for-any", r"\b(for\s+)?(any|every|each|all)\s+\$[^$]+\$"),
    ("where-binding", r"\bwhere\s+\$[^$]+\$\s+(is|denotes|represents)"),
    ("set-notation", r"\$[^$]*\\in\s+[^$]+\$"),
]


def detect_scopes(text):
    """Detect scope openers in text. Returns list of (type, match_text)."""
    scopes = []
    for stype, pattern in SCOPE_REGEXES:
        for m in re.finditer(pattern, text):
            scopes.append({"type": stype, "match": m.group()[:80]})
    return scopes


# ---------- Main ----------

def stream_entities_sample(path, seed=42, sample_per_stratum=200):
    """Stream entities from se-physics.json, keeping only a small sample.

    Memory-safe: reads file in chunks, uses bracket-counting to extract
    individual entity objects. Never holds the full 114K-entity array.
    Keeps top-200 candidates per stratum via heap.
    """
    import heapq

    strata_pools = {"easy": [], "medium": [], "hard": []}
    rng = random.Random(seed)
    count = 0

    CHUNK = 65536

    with open(path) as f:
        # Scan forward until we find "entities": [
        buf = ""
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                print("  ERROR: never found 'entities' key")
                return {"easy": [], "medium": [], "hard": []}
            buf += chunk
            marker = buf.find('"entities"')
            if marker != -1:
                # Find the opening [ after "entities":
                bracket = buf.find("[", marker)
                if bracket != -1:
                    # Start parsing from just after the [
                    remainder = buf[bracket + 1:]
                    break
                # [ might be in the next chunk
        del buf

        # String-aware bracket-counting parser.
        # Tracks whether we're inside a JSON string to avoid counting
        # braces in LaTeX content like \{ \} or literal { } in text.
        depth = 0
        obj_chars = []
        in_obj = False
        in_string = False
        escape_next = False

        def process_chars(chars):
            nonlocal depth, obj_chars, in_obj, in_string, escape_next, count
            for ch in chars:
                if in_obj:
                    obj_chars.append(ch)

                if escape_next:
                    escape_next = False
                    continue

                if ch == '\\' and in_string:
                    escape_next = True
                    continue

                if ch == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                # Outside strings: track brackets
                if ch == '{':
                    if depth == 0:
                        in_obj = True
                        obj_chars = [ch]
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and in_obj:
                        try:
                            entity = json.loads(''.join(obj_chars))
                        except json.JSONDecodeError:
                            obj_chars = []
                            in_obj = False
                            continue
                        count += 1
                        score = entity.get("score", 0)
                        if score >= 20:
                            stratum = "easy"
                        elif score >= 5:
                            stratum = "medium"
                        else:
                            stratum = "hard"

                        has_latex = 1 if entity.get("answer-latex") else 0
                        ans_len = len(entity.get("answer-body", ""))
                        priority = (has_latex, ans_len, rng.random())

                        pool = strata_pools[stratum]
                        if len(pool) < sample_per_stratum:
                            heapq.heappush(pool, (priority, entity))
                        else:
                            heapq.heappushpop(pool, (priority, entity))

                        obj_chars = []
                        in_obj = False

                        if count % 20000 == 0:
                            print(f"    streamed {count} entities...")
                elif ch == ']' and depth == 0:
                    return True
            return False

        # Process the remainder from the initial scan
        done = process_chars(remainder)
        del remainder

        # Continue reading chunks
        while not done:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            done = process_chars(chunk)

    print(f"  Streamed {count} entities total")
    result = {}
    for stratum, pool in strata_pools.items():
        result[stratum] = [entity for (_, entity) in pool]
        rng.shuffle(result[stratum])
    return result


def main():
    random.seed(SEED)

    print(f"Streaming entities from {SE_JSON} (memory-safe)...")
    strata_pools = stream_entities_sample(SE_JSON, seed=SEED)

    # Load NER kernel
    print(f"Loading NER kernel from {TERMS_TSV}...")
    singles, multi_index, multi_count = load_ner_kernel(TERMS_TSV)
    print(f"  {len(singles)} single-word terms, {multi_count} multi-word terms")

    # Select from pre-stratified pools
    selected = []
    for stratum_name, _, count in STRATA:
        pool = strata_pools.get(stratum_name, [])
        chosen = pool[:count]
        for e in chosen:
            e["_stratum"] = stratum_name
        selected.extend(chosen)
        print(f"  Stratum '{stratum_name}': {len(pool)} in pool, selected {len(chosen)}")

    print(f"\nSelected {len(selected)} entries. Annotating...")

    golden = []
    pattern_counter = Counter()
    scope_counter = Counter()

    for i, entity in enumerate(selected):
        eid = entity["entity/id"]
        text = (entity.get("question-body", "") + " " +
                entity.get("answer-body", ""))
        answer_text = entity.get("answer-body", "")

        # NER term spotting
        ner_hits = spot_terms(text, singles, multi_index)

        # Pattern tagging (on answer text, as that's where reasoning patterns live)
        patterns = tag_patterns(answer_text)
        for p in patterns:
            pattern_counter[p["pattern"]] += 1

        # Scope detection (on answer text)
        scopes = detect_scopes(answer_text)
        for s in scopes:
            scope_counter[s["type"]] += 1

        golden.append({
            "id": eid,
            "stratum": entity["_stratum"],
            "title": entity.get("title", ""),
            "tags": entity.get("tags", []),
            "score": entity.get("score", 0),
            "answer_score": entity.get("answer-score", 0),
            "question_body": entity.get("question-body", ""),
            "answer_body": answer_text,
            "question_latex": entity.get("question-latex", []),
            "answer_latex": entity.get("answer-latex", []),
            "created": entity.get("created", ""),
            # --- Golden annotations ---
            "golden_ner_terms": ner_hits,
            "golden_ner_count": len(ner_hits),
            "golden_patterns": patterns,
            "golden_pattern_names": [p["pattern"] for p in patterns],
            "golden_scopes": scopes,
            "golden_scope_count": len(scopes),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(selected)}] annotated")

    # Summary statistics
    summary = {
        "total": len(golden),
        "strata": {name: sum(1 for g in golden if g["stratum"] == name)
                   for name, _, _ in STRATA},
        "avg_ner_terms": sum(g["golden_ner_count"] for g in golden) / len(golden),
        "avg_patterns": sum(len(g["golden_pattern_names"]) for g in golden) / len(golden),
        "avg_scopes": sum(g["golden_scope_count"] for g in golden) / len(golden),
        "pattern_frequency": dict(pattern_counter.most_common()),
        "scope_frequency": dict(scope_counter.most_common()),
        "entries_with_ner": sum(1 for g in golden if g["golden_ner_count"] > 0),
        "entries_with_patterns": sum(1 for g in golden if g["golden_pattern_names"]),
        "entries_with_scopes": sum(1 for g in golden if g["golden_scope_count"] > 0),
    }

    output = {
        "version": "golden-50-v1",
        "generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  __import__("time").gmtime()),
        "source": "physics.stackexchange",
        "seed": SEED,
        "summary": summary,
        "entries": golden,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"\nWritten {OUTPUT} ({size_mb:.1f} MB)")
    print(f"\nSummary:")
    print(f"  Entries:        {summary['total']}")
    print(f"  Strata:         {summary['strata']}")
    print(f"  Avg NER terms:  {summary['avg_ner_terms']:.1f}")
    print(f"  Avg patterns:   {summary['avg_patterns']:.1f}")
    print(f"  Avg scopes:     {summary['avg_scopes']:.1f}")
    print(f"  With NER:       {summary['entries_with_ner']}/{summary['total']}")
    print(f"  With patterns:  {summary['entries_with_patterns']}/{summary['total']}")
    print(f"  With scopes:    {summary['entries_with_scopes']}/{summary['total']}")
    print(f"\n  Pattern freq (top 10):")
    for name, count in pattern_counter.most_common(10):
        print(f"    {name}: {count}")
    print(f"\n  Scope freq:")
    for name, count in scope_counter.most_common():
        print(f"    {name}: {count}")


if __name__ == "__main__":
    main()
