#!/usr/bin/env python3
"""Process a StackExchange data dump into F6 entities with embeddings.

Usage:
    # CPU (small dataset / testing):
    python scripts/process-stackexchange.py /path/to/Posts.xml --output data/se-physics.edn

    # GPU superpod (large dataset):
    python scripts/process-stackexchange.py /path/to/Posts.xml \
        --output data/se-math.edn \
        --device cuda \
        --batch-size 1024 \
        --model BAAI/bge-large-en-v1.5 \
        --min-score 1 \
        --workers 16

    # Multi-GPU (splits across all visible GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/process-stackexchange.py ...

Superpod profile (128 CPU, 2TB RAM, 8 GPU):
    --workers 32 --batch-size 2048 --device cuda --model BAAI/bge-large-en-v1.5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from futon6.stackexchange import (
    iter_posts,
    load_posts,
    build_qa_pairs,
    qa_to_entity,
    qa_to_relations,
    tag_entities,
    corpus_stats,
    compute_qa_embeddings,
    SEQAPair,
)


def edn_str(obj) -> str:
    """Minimal Python->EDN serialisation for F6 output."""
    if isinstance(obj, dict):
        pairs = " ".join(f":{k} {edn_str(v)}" for k, v in obj.items())
        return "{" + pairs + "}"
    elif isinstance(obj, (list, tuple)):
        return "[" + " ".join(edn_str(x) for x in obj) + "]"
    elif isinstance(obj, str):
        escaped = obj.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif isinstance(obj, (int, float)):
        return str(obj)
    elif obj is None:
        return "nil"
    else:
        return f'"{obj}"'


def main():
    parser = argparse.ArgumentParser(description="Process SE dump into F6 entities")
    parser.add_argument("posts_xml", help="Path to Posts.xml")
    parser.add_argument("--output", "-o", default="data/se-output.edn",
                        help="Output EDN file")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Minimum post score to include")
    parser.add_argument("--site", default="physics.stackexchange",
                        help="SE site name for entity IDs")

    # Embedding options
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model name")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Embedding batch size (increase for GPU)")
    parser.add_argument("--device", default=None,
                        help="Device: cuda, cpu, or auto-detect (None)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding computation")

    # Performance
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for preprocessing")

    args = parser.parse_args()

    t0 = time.time()

    # --- Stage 1: Parse XML ---
    print(f"[1/4] Parsing {args.posts_xml} (min_score={args.min_score})...")
    posts = load_posts(args.posts_xml, min_score=args.min_score)
    print(f"       Loaded {len(posts)} posts in {time.time()-t0:.1f}s")

    # --- Stage 2: Build QA pairs ---
    t1 = time.time()
    print(f"[2/4] Building QA pairs...")
    pairs = build_qa_pairs(posts)
    stats = corpus_stats(pairs)
    print(f"       {stats['qa_pairs']} QA pairs, "
          f"{stats['unique_tags']} tags, "
          f"{stats['with_latex']} with LaTeX, "
          f"{stats['total_latex_fragments']} LaTeX fragments")
    print(f"       Avg question score: {stats['avg_q_score']:.1f}, "
          f"avg answer score: {stats['avg_a_score']:.1f}")
    print(f"       Built in {time.time()-t1:.1f}s")

    # --- Stage 3: Embeddings ---
    embeddings = None
    if not args.no_embeddings:
        t2 = time.time()
        print(f"[3/4] Computing embeddings (model={args.model}, "
              f"batch_size={args.batch_size}, device={args.device})...")
        embeddings = compute_qa_embeddings(
            pairs,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"       Embeddings shape: {embeddings.shape}, "
              f"computed in {time.time()-t2:.1f}s")
    else:
        print(f"[3/4] Skipping embeddings (--no-embeddings)")

    # --- Stage 4: Convert to entities and write ---
    t3 = time.time()
    print(f"[4/4] Converting to F6 entities and writing {args.output}...")

    entities = []
    relations = []
    for i, pair in enumerate(pairs):
        entity = qa_to_entity(pair)
        # Override site name
        entity["entity/source"] = args.site
        entity["entity/id"] = f"se-{args.site.split('.')[0]}-{pair.question.id}"

        if embeddings is not None:
            entity["embedding-model"] = args.model
            entity["embedding-dim"] = int(embeddings.shape[1])
            # Store embedding as list of floats (for EDN)
            entity["embedding"] = embeddings[i].tolist()

        entities.append(entity)
        relations.extend(qa_to_relations(pair))

    tags = tag_entities(pairs)

    output = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.site,
        "posts-xml": args.posts_xml,
        "min-score": args.min_score,
        "model": args.model if not args.no_embeddings else "none",
        "stats": stats,
        "entity-count": len(entities),
        "tag-count": len(tags),
        "relation-count": len(relations),
    }

    # Write as JSON (more practical for large datasets than EDN)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".json":
        with open(out_path, "w") as f:
            json.dump({**output, "entities": entities, "tags": tags,
                       "relations": relations}, f)
    else:
        # EDN output (no embeddings inline — too large)
        with open(out_path, "w") as f:
            f.write(edn_str({**output,
                             "entities": [{k: v for k, v in e.items()
                                           if k != "embedding"}
                                          for e in entities],
                             "tags": tags,
                             "relations": relations}))

    # Save embeddings separately as numpy
    if embeddings is not None:
        import numpy as np
        emb_path = out_path.with_suffix(".embeddings.npy")
        np.save(emb_path, embeddings)
        print(f"       Embeddings saved to {emb_path}")

    elapsed = time.time() - t0
    print(f"       Written {len(entities)} entities, {len(tags)} tags, "
          f"{len(relations)} relations")
    print(f"       Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
