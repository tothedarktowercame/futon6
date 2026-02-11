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
        --output-dir ./mo-processed --site mathoverflow

    # CPU-only (no GPU required):
    python scripts/superpod-job.py ./se-data/math.stackexchange.com/Posts.xml \\
        --skip-embeddings --skip-llm --skip-clustering \\
        --site math.stackexchange

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

Each stage writes its output independently -- if a stage fails, earlier
outputs are still usable.

Stage 5 output uses futon4-compatible hyperedge format for scope records:
  :hx/type, :hx/ends (with roles), :hx/content, :hx/labels
This enables direct ingest into futon1/XTDB via the standard relation->hx
conversion path. See futon1/apps/graph-memory for schema.
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import shutil
import subprocess

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from futon6.stackexchange import (
    build_qa_pairs_streaming,
    qa_to_entity,
    qa_to_relations,
    tag_entities,
    corpus_stats,
    compute_qa_embeddings,
)

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
        "site": "mathoverflow",
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
      hx/type  — scope type (let-binding, define, assume, etc.)
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

            scopes.append({
                "hx/id": scope_id,
                "hx/type": f"scope/{stype}",
                "hx/ends": ends,
                "hx/content": {"match": m.group()[:120],
                               "position": m.start()},
                "hx/labels": ["scope", stype],
            })

    return scopes


def run_stage5_ner_scopes(entities, pairs, ner_kernel_path, outdir):
    """Run Stage 5: NER term spotting + scope detection.

    Memory-safe: streams results directly to disk (one JSON object per line
    inside a JSON array), never accumulating all results in RAM.

    Returns stats dict.
    """
    from collections import Counter

    singles, multi_index, multi_count = load_ner_kernel(ner_kernel_path)
    print(f"       NER kernel: {len(singles)} single + {multi_count} multi-word terms")

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
            answer_text = pair.answer.body_text

            # NER term spotting
            terms = spot_terms_entity(full_text, singles, multi_index)
            if terms:
                entities_with_ner += 1
                total_ner_hits += len(terms)

            # Scope detection
            scopes = detect_scopes_entity(eid, answer_text)
            if scopes:
                entities_with_scopes += 1
                total_scopes += len(scopes)
                for s in scopes:
                    stype_freq[s["hx/type"]] += 1

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

    return {
        "ner_kernel_terms": len(singles) + multi_count,
        "entities_processed": n,
        "total_ner_hits": total_ner_hits,
        "entities_with_ner": entities_with_ner,
        "ner_coverage": entities_with_ner / n if n else 0,
        "total_scopes": total_scopes,
        "entities_with_scopes": entities_with_scopes,
        "scope_coverage": entities_with_scopes / n if n else 0,
        "scope_type_freq": dict(stype_freq.most_common()),
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

Given this Q&A from math.stackexchange, identify which informal reasoning patterns the ANSWER uses.

Question: {question_title}
{q}

Answer:
{a}

Here are the 25 patterns to check:
{pattern_list}

Reply with ONLY a JSON list of pattern numbers (1-25) that the answer clearly uses. Example: [3, 10, 17]
If none apply clearly, reply: []"""


def tag_patterns_llm_batch(pairs, model_name, batch_size=8, device="cuda"):
    """Tag QA pairs with reasoning patterns using a local LLM.

    Uses transformers pipeline for batched inference.
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
        device_map="auto",  # spread across available GPUs
        max_new_tokens=64,
        do_sample=False,
        batch_size=batch_size,
    )

    results = []
    total = len(pairs)

    for start in range(0, total, batch_size):
        batch = pairs[start:start + batch_size]
        prompts = []
        for pair in batch:
            prompt = build_pattern_prompt(
                pair.question.title,
                pair.question.body_text,
                pair.answer.body_text,
            )
            # Format for instruct model
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted)

        outputs = pipe(prompts, return_full_text=False)

        for i, out in enumerate(outputs):
            text = out[0]["generated_text"].strip()
            # Parse the JSON list from the response
            pattern_ids = _parse_pattern_response(text)
            results.append({
                "entry_id": f"se-math-{batch[i].question.id}",
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


# --- Stage 4: Clustering ---

def cluster_embeddings(embeddings, min_cluster_size=50, max_clusters=500):
    """Cluster embeddings using HDBSCAN (or KMeans fallback)."""
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
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(max_clusters, len(embeddings) // 100)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=4096,
            n_init=3,
        )
        labels = kmeans.fit_predict(embeddings)
        n_noise = 0
        print(f"       {n_clusters} clusters (KMeans)")

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

Given this Q&A pair from math.stackexchange, perform reverse morphogenesis analysis.

Question: {question_title}
{q}

Answer:
{a}

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
}}"""


def generate_moist_prompts(pairs, entities, outdir, stages=None):
    """Generate prompt files for LLM stages (Codex/Claude handoff).

    Instead of running Llama locally, writes JSONL files with one prompt per
    line, ready for batch submission to Codex, Claude, or any API.

    stages: list of stage names to generate. Default: all LLM stages.
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

    # Write a manifest for the moist-run
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "moist-run",
        "note": "Prompt files for Codex/Claude batch submission. "
                "Each JSONL line has entity_id, question_id, stage, prompt.",
        "stages": generated,
        "usage": {
            "codex": "codex --prompt-file <path.jsonl>",
            "claude_api": "for line in open(path): send(json.loads(line)['prompt'])",
            "manual": "Read prompt field, paste into LLM, save response",
        },
    }
    manifest_path = prompt_dir / "manifest.json"
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
        est_total_min = est_stage1_min + est_stage2_min + est_stage3_min + est_stage4_min + est_stage5_min
    else:
        est_stage1_min = est_stage2_min = est_stage3_min = "?"
        est_stage4_min = est_stage5_min = est_total_min = "?"

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

    if args.skip_embeddings:
        print(f"  {'2. Embeddings':<42s} {'SKIPPED':>10s}")
    else:
        print(f"  {'2. Dense embeddings (GPU)':<42s} {args.embed_model:<36s} {fmt(est_stage2_min)+' min':>10s}")

    if args.skip_llm:
        print(f"  {'3. LLM pattern tagging':<42s} {'SKIPPED':>10s}")
    else:
        print(f"  {'3. LLM pattern tagging (GPU)':<42s} {args.llm_model:<36s} {fmt(est_stage3_min)+' min':>10s}")

    if args.skip_clustering:
        print(f"  {'4. Clustering':<42s} {'SKIPPED':>10s}")
    else:
        print(f"  {'4. Clustering (CPU)':<42s} {'HDBSCAN/KMeans':<36s} {fmt(est_stage4_min)+' min':>10s}")

    if args.skip_ner:
        print(f"  {'5. NER + scope detection':<42s} {'SKIPPED':>10s}")
    else:
        print(f"  {'5. NER + scope detection (CPU)':<42s} {args.ner_kernel:<36s} {fmt(est_stage5_min)+' min':>10s}")

    # Stage 6 is always present (CPU — prompt generation or LLM)
    est_stage6_min = est_stage1_min  # roughly same as parse (just string formatting)
    print(f"  {'6. Reverse morphogenesis S←Q←A (LLM)':<42s} {args.llm_model:<36s} {fmt(est_stage6_min)+' min':>10s}")

    print(f"  {'-'*42} {'-'*36} {'-'*10}")

    skipped = sum([args.skip_embeddings, args.skip_llm, args.skip_clustering, args.skip_ner])
    active = 6 - skipped
    if isinstance(est_total_min, (int, float)):
        est_total_min += est_stage6_min
    print(f"  {'TOTAL':<42s} {f'{active}/6 stages active':<36s} {fmt(est_total_min)+' min':>10s}")
    print()

    print("  ESTIMATED OUTPUT:")
    if isinstance(est_entities_mb, (int, float)):
        print(f"    entities.json         ~{fmt(est_entities_mb)} MB")
        if not args.skip_embeddings:
            print(f"    embeddings.npy        ~{fmt(est_embeddings_gb)} GB")
        if not args.skip_llm:
            print(f"    pattern-tags.json     ~{fmt(est_entities_mb * 0.3)} MB")
        if not args.skip_clustering:
            print(f"    clusters.json         ~{fmt(est_entities_mb * 0.05)} MB")
        if not args.skip_ner:
            print(f"    ner-terms.json        ~{fmt(est_entities_mb * 0.4)} MB")
            print(f"    scopes.json           ~{fmt(est_entities_mb * 0.2)} MB")
        print(f"    {'':24s} --------")
        print(f"    {'TOTAL':24s} ~{fmt(est_total_gb)} GB")
    else:
        print(f"    (cannot estimate — input file not found)")
    print()

    print("  GPU REQUIREMENTS:")
    if not args.skip_embeddings or not args.skip_llm:
        print(f"    Embedding model:  {args.embed_model}" if not args.skip_embeddings else "")
        print(f"    LLM model:        {args.llm_model}" if not args.skip_llm else "")
        vram = 2 if args.skip_llm else 18  # bge-large ~2GB, Llama-3-8B ~16GB
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
    print(f"    {' \\\\\n    '.join(cmd_parts)}")
    print()
    print(f"  MOIST RUN (CPU stages + prompt files for Codex handoff):")
    moist_parts = [f"python scripts/superpod-job.py {args.posts_xml}"]
    moist_parts.append(f"  --moist-run")
    moist_parts.append(f"  --output-dir {args.output_dir}")
    moist_parts.append(f"  --site {args.site}")
    if args.limit:
        moist_parts.append(f"  --limit {args.limit}")
    print(f"    {' \\\\\n    '.join(moist_parts)}")
    print()
    print(f"    Generates: moist-prompts/stage3-pattern-tagging.jsonl")
    print(f"               moist-prompts/stage6-reverse-morphogenesis.jsonl")
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
                        help="Path to Posts.xml (not needed with --download)")

    # Download options
    parser.add_argument("--download", choices=list(SE_DUMPS.keys()),
                        help="Download + extract a SE dump from Internet Archive")
    parser.add_argument("--data-dir", default="./se-data",
                        help="Directory for downloaded dumps (default: ./se-data)")

    parser.add_argument("--output-dir", "-o", default="./math-se-processed",
                        help="Output directory for all artefacts")
    parser.add_argument("--site", default="math.stackexchange",
                        help="SE site name")
    parser.add_argument("--min-score", type=int, default=1,
                        help="Minimum post score")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max QA pairs to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show execution plan without running anything")
    parser.add_argument("--moist-run", action="store_true",
                        help="Run CPU stages, output prompts for LLM stages (for Codex handoff)")

    # Embedding options
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5",
                        help="Embedding model")
    parser.add_argument("--embed-batch-size", type=int, default=2048,
                        help="Embedding batch size")
    parser.add_argument("--embed-device", default="cuda",
                        help="Device for embeddings")
    parser.add_argument("--skip-embeddings", action="store_true")

    # LLM options
    parser.add_argument("--llm-model",
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="LLM for pattern tagging")
    parser.add_argument("--llm-batch-size", type=int, default=8,
                        help="LLM inference batch size")
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

    args = parser.parse_args()

    # ========== Download ==========
    if args.download:
        posts_path = download_and_extract(args.download, args.data_dir)
        if not args.posts_xml:
            # Just downloading, not processing
            return
        # If posts_xml was also given, continue to processing

    if not args.posts_xml:
        parser.error("posts_xml is required (or use --download to fetch data first)")

    # ========== Dry Run ==========
    if args.dry_run:
        print_dry_run(args)
        return

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        print("=" * 64)
        print()

    t0 = time.time()

    n_stages = 6
    # ========== Stage 1: Parse ==========
    print(f"[Stage 1/{n_stages}] Parsing {args.posts_xml}...")
    pairs = build_qa_pairs_streaming(args.posts_xml, min_score=args.min_score)
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
        entity = qa_to_entity(pair)
        entity["entity/source"] = args.site
        entity["entity/id"] = f"se-{args.site.split('.')[0]}-{pair.question.id}"
        entities.append(entity)
        all_relations.extend(qa_to_relations(pair))

    tags = tag_entities(pairs)

    write_json(outdir / "entities.json", entities)
    write_json(outdir / "relations.json", all_relations)
    write_json(outdir / "tags.json", tags)
    write_json(outdir / "stats.json", stats)

    print(f"       Stage 1 done in {time.time()-t0:.0f}s")

    # ========== Stage 2: Embeddings ==========
    if not args.skip_embeddings:
        t2 = time.time()
        print(f"\n[Stage 2/6] Embeddings ({args.embed_model})...")
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
        print(f"\n[Stage 2/6] Skipped (--skip-embeddings)")
        embeddings = None

    # ========== Stage 3: LLM pattern tagging ==========
    if args.moist_run:
        # Moist mode: generate prompts, don't run LLM
        t3 = time.time()
        print(f"\n[Stage 3/6] Generating pattern-tagging prompts (moist-run)...")
        moist_result = generate_moist_prompts(
            pairs, entities, outdir, stages=["pattern_tagging"])
        print(f"       Stage 3 (moist) done in {time.time()-t3:.0f}s")
    elif not args.skip_llm:
        t3 = time.time()
        global _llm_start
        _llm_start = t3
        print(f"\n[Stage 3/6] LLM pattern tagging ({args.llm_model})...")
        pattern_tags = tag_patterns_llm_batch(
            pairs,
            model_name=args.llm_model,
            batch_size=args.llm_batch_size,
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
        print(f"\n[Stage 3/6] Skipped (--skip-llm)")

    # ========== Stage 4: Clustering ==========
    if not args.skip_clustering and embeddings is not None:
        t4 = time.time()
        print(f"\n[Stage 4/6] Clustering...")
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
        print(f"\n[Stage 4/6] Skipped")

    # ========== Stage 5: NER + Scope Detection ==========
    stage5_stats = None
    if not args.skip_ner:
        ner_path = Path(args.ner_kernel)
        if not ner_path.exists():
            print(f"\n[Stage 5/6] Skipped (NER kernel not found: {ner_path})")
        else:
            t5 = time.time()
            print(f"\n[Stage 5/6] NER term spotting + scope detection...")
            print(f"       Kernel: {ner_path}")
            stage5_stats = run_stage5_ner_scopes(
                entities, pairs, str(ner_path), outdir)

            print(f"       NER coverage: {stage5_stats['ner_coverage']:.0%} "
                  f"({stage5_stats['entities_with_ner']}/{stage5_stats['entities_processed']})")
            print(f"       Scope coverage: {stage5_stats['scope_coverage']:.0%} "
                  f"({stage5_stats['entities_with_scopes']}/{stage5_stats['entities_processed']})")
            if stage5_stats['scope_type_freq']:
                print(f"       Scope types:")
                for stype, count in stage5_stats['scope_type_freq'].items():
                    print(f"         {stype}: {count}")
            print(f"       Stage 5 done in {time.time()-t5:.0f}s")
    else:
        print(f"\n[Stage 5/6] Skipped (--skip-ner)")

    # ========== Stage 6: Reverse morphogenesis S←Q←A ==========
    stage6_stats = None
    if args.moist_run:
        t6 = time.time()
        print(f"\n[Stage 6/6] Generating reverse morphogenesis prompts (moist-run)...")
        moist_s6 = generate_moist_prompts(
            pairs, entities, outdir, stages=["reverse_morphogenesis"])
        stage6_stats = {
            "mode": "moist-run",
            "prompts_generated": moist_s6.get("reverse_morphogenesis", {}).get("count", 0),
        }
        print(f"       Stage 6 (moist) done in {time.time()-t6:.0f}s")
    elif not args.skip_llm:
        # Full run: would use LLM for S←Q←A inference
        # For now, generate prompts even in full mode (Stage 6 is new)
        t6 = time.time()
        print(f"\n[Stage 6/6] Reverse morphogenesis S←Q←A...")
        print(f"       (LLM inference for Stage 6 not yet implemented;")
        print(f"        generating prompt files for manual/API submission)")
        moist_s6 = generate_moist_prompts(
            pairs, entities, outdir, stages=["reverse_morphogenesis"])
        stage6_stats = {
            "mode": "prompt-generation",
            "prompts_generated": moist_s6.get("reverse_morphogenesis", {}).get("count", 0),
        }
        print(f"       Stage 6 done in {time.time()-t6:.0f}s")
    else:
        print(f"\n[Stage 6/6] Skipped (--skip-llm)")

    # ========== Manifest ==========
    elapsed = time.time() - t0
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.site,
        "posts_xml": args.posts_xml,
        "min_score": args.min_score,
        "embed_model": args.embed_model if not args.skip_embeddings else None,
        "llm_model": args.llm_model if not args.skip_llm else None,
        "stats": stats,
        "entity_count": len(entities),
        "tag_count": len(tags),
        "relation_count": len(all_relations),
        "elapsed_seconds": round(elapsed),
        "moist_run": args.moist_run,
        "stages_completed": [
            "parse",
            *([] if args.skip_embeddings else ["embeddings"]),
            *(["llm_pattern_tags_moist"] if args.moist_run else
              ([] if args.skip_llm else ["llm_pattern_tags"])),
            *([] if args.skip_clustering or embeddings is None else ["clustering"]),
            *([] if args.skip_ner or stage5_stats is None else ["ner_scopes"]),
            *([] if stage6_stats is None else ["reverse_morphogenesis"]),
        ],
        "stage5_stats": stage5_stats,
        "stage6_stats": stage6_stats,
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
    print(f"\nTo upload: tar czf {outdir.name}.tar.gz {outdir.name}/")


_llm_start = 0  # module-level for rate tracking

if __name__ == "__main__":
    main()
