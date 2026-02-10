#!/usr/bin/env python3
"""Superpod batch job: process math.stackexchange into F6 artefacts.

Self-contained job that reads a SE data dump and produces a single output
directory with all computed artefacts. Designed for GPU-accelerated batch
processing on a multi-GPU machine.

Setup:
    # Clone and install
    git clone <futon6-repo> && cd futon6
    pip install -e ".[gpu]"
    # or: pip install sentence-transformers torch

    # Extract SE dump
    7z x math.stackexchange.com.7z -o./math-se-raw/

Usage:
    python scripts/superpod-job.py ./math-se-raw/Posts.xml \
        --output-dir ./math-se-processed \
        --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
        --site math.stackexchange

    # Then tar and upload:
    tar czf math-se-processed.tar.gz math-se-processed/

All stages:
    1. Parse XML → QA pairs (CPU, streaming)
    2. Dense embeddings (GPU, bge-large-en-v1.5)
    3. LLM pattern tagging (GPU, Llama-3-8B)
    4. Clustering (CPU, HDBSCAN on embeddings)
    5. NER term spotting + scope detection (CPU, classical)

Each stage writes its output independently — if a stage fails, earlier
outputs are still usable.

Stage 5 output uses futon4-compatible hyperedge format for scope records:
  :hx/type, :hx/ends (with roles), :hx/content, :hx/labels
This enables direct ingest into futon1/XTDB via the standard relation→hx
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


def main():
    parser = argparse.ArgumentParser(
        description="Superpod batch job: SE dump → F6 artefacts")
    parser.add_argument("posts_xml", help="Path to Posts.xml")
    parser.add_argument("--output-dir", "-o", default="./math-se-processed",
                        help="Output directory for all artefacts")
    parser.add_argument("--site", default="math.stackexchange",
                        help="SE site name")
    parser.add_argument("--min-score", type=int, default=1,
                        help="Minimum post score")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max QA pairs to process (for dry runs)")

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

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ========== Stage 1: Parse ==========
    print(f"[Stage 1/5] Parsing {args.posts_xml}...")
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
        print(f"\n[Stage 2/5] Embeddings ({args.embed_model})...")
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
        print(f"\n[Stage 2/5] Skipped (--skip-embeddings)")
        embeddings = None

    # ========== Stage 3: LLM pattern tagging ==========
    if not args.skip_llm:
        t3 = time.time()
        global _llm_start
        _llm_start = t3
        print(f"\n[Stage 3/5] LLM pattern tagging ({args.llm_model})...")
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
        print(f"\n[Stage 3/5] Skipped (--skip-llm)")

    # ========== Stage 4: Clustering ==========
    if not args.skip_clustering and embeddings is not None:
        t4 = time.time()
        print(f"\n[Stage 4/5] Clustering...")
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
        print(f"\n[Stage 4/5] Skipped")

    # ========== Stage 5: NER + Scope Detection ==========
    stage5_stats = None
    if not args.skip_ner:
        ner_path = Path(args.ner_kernel)
        if not ner_path.exists():
            print(f"\n[Stage 5/5] Skipped (NER kernel not found: {ner_path})")
        else:
            t5 = time.time()
            print(f"\n[Stage 5/5] NER term spotting + scope detection...")
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
        print(f"\n[Stage 5/5] Skipped (--skip-ner)")

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
        "stages_completed": [
            "parse",
            *([] if args.skip_embeddings else ["embeddings"]),
            *([] if args.skip_llm else ["llm_pattern_tags"]),
            *([] if args.skip_clustering or embeddings is None else ["clustering"]),
            *([] if args.skip_ner or stage5_stats is None else ["ner_scopes"]),
        ],
        "stage5_stats": stage5_stats,
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
    print(f"\nTo upload: tar czf {outdir.name}.tar.gz {outdir.name}/")


_llm_start = 0  # module-level for rate tracking

if __name__ == "__main__":
    main()
