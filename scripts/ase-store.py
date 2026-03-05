#!/usr/bin/env python3
"""Artificial Stack Exchange store.

A minimal, file-based store for synthetic QA threads in hypergraph-native
format. Mirrors the structure of storage/math-processed-gpu/ so the
retrieval pipeline can query both real and synthetic corpora uniformly.

Operations:
    ingest   — Add new synthetic QA threads (from generate-synthetic-qa.py output)
    reindex  — Rebuild FAISS index from current store
    stats    — Show store statistics
    query    — Search the store (keyword + FAISS)

The store lives at storage/ase/ by default.

Usage:
    # Ingest synthetic QA from API output
    python3 scripts/ase-store.py ingest data/synthetic-qa/problem7.jsonl

    # Rebuild indexes after ingestion
    python3 scripts/ase-store.py reindex

    # Show stats
    python3 scripts/ase-store.py stats

    # Query
    python3 scripts/ase-store.py query "surgery obstruction torsion lattice"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = Path(os.path.expanduser("~/code/storage/ase"))


def ensure_store(store_dir: Path) -> None:
    """Create store directory structure if needed."""
    store_dir.mkdir(parents=True, exist_ok=True)


def load_entities(store_dir: Path) -> list[dict]:
    path = store_dir / "entities.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def save_entities(store_dir: Path, entities: list[dict]) -> None:
    with (store_dir / "entities.json").open("w") as f:
        json.dump(entities, f, ensure_ascii=False, indent=1)


def load_hypergraphs(store_dir: Path) -> list[dict]:
    path = store_dir / "hypergraphs.jsonl"
    if not path.exists():
        return []
    hgs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                hgs.append(json.loads(line))
    return hgs


def save_hypergraphs(store_dir: Path, hypergraphs: list[dict]) -> None:
    with (store_dir / "hypergraphs.jsonl").open("w") as f:
        for hg in hypergraphs:
            f.write(json.dumps(hg, ensure_ascii=False) + "\n")


def load_manifest(store_dir: Path) -> dict:
    path = store_dir / "manifest.json"
    if not path.exists():
        return {
            "source": "artificial-stack-exchange",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entity_count": 0,
            "ingestion_log": [],
        }
    with path.open() as f:
        return json.load(f)


def save_manifest(store_dir: Path, manifest: dict) -> None:
    with (store_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def ingest(store_dir: Path, input_path: Path) -> int:
    """Ingest synthetic QA threads from JSONL."""
    ensure_store(store_dir)
    entities = load_entities(store_dir)
    hypergraphs = load_hypergraphs(store_dir)
    manifest = load_manifest(store_dir)

    existing_ids = {e.get("entity/id", e.get("thread_id", "")) for e in entities}
    new_count = 0
    dupe_count = 0

    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            thread_id = rec.get("thread_id", "")
            if not thread_id:
                continue
            if thread_id in existing_ids:
                dupe_count += 1
                continue

            # Build entity (matches real corpus format)
            entity = {
                "entity/id": thread_id,
                "entity/type": "QAPair",
                "entity/source": "artificial-stack-exchange",
                "title": rec.get("title", ""),
                "question-body": rec.get("question", ""),
                "answer-body": rec.get("answer", ""),
                "tags": rec.get("tags", []),
                "score": 0,
                "answer-score": 0,
                "synthetic": True,
                "source_node": rec.get("source_node", ""),
                "source_problem": rec.get("source_problem", ""),
            }
            entities.append(entity)
            existing_ids.add(thread_id)

            # Build hypergraph (if nodes/edges provided)
            if rec.get("nodes") and rec.get("edges"):
                hg = {
                    "thread_id": thread_id,
                    "nodes": rec["nodes"],
                    "edges": rec["edges"],
                }
                hypergraphs.append(hg)

            new_count += 1

    save_entities(store_dir, entities)
    save_hypergraphs(store_dir, hypergraphs)

    manifest["entity_count"] = len(entities)
    manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["ingestion_log"].append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(input_path),
        "new": new_count,
        "duplicates": dupe_count,
    })
    save_manifest(store_dir, manifest)

    print(f"Ingested {new_count} new threads ({dupe_count} duplicates skipped)")
    print(f"Store now has {len(entities)} entities, {len(hypergraphs)} hypergraphs")
    return 0


def reindex(store_dir: Path) -> int:
    """Rebuild text embeddings and FAISS index."""
    entities = load_entities(store_dir)
    if not entities:
        print("Store is empty, nothing to index.")
        return 0

    hypergraphs = load_hypergraphs(store_dir)

    print(f"Reindexing {len(entities)} entities...")

    # Text embeddings via sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed, skipping text embeddings")
        return 1

    print("  Loading BGE model...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    texts = []
    for e in entities:
        text = f"{e.get('title', '')} {e.get('question-body', '')} {e.get('answer-body', '')}"
        texts.append(text[:1000])

    print(f"  Encoding {len(texts)} texts...")
    t0 = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"  Encoded in {time.time()-t0:.1f}s")

    np.save(store_dir / "embeddings.npy", embeddings)
    print(f"  Saved embeddings: {embeddings.shape}")

    # GNN embeddings + FAISS if we have hypergraphs
    if hypergraphs:
        gnn_model_path = Path(os.path.expanduser(
            "~/code/storage/math-processed-gpu/graph-gnn-model.pt"
        ))
        if gnn_model_path.exists():
            print("  Computing GNN embeddings...")
            try:
                _build_gnn_index(store_dir, hypergraphs, gnn_model_path)
            except Exception as e:
                print(f"  GNN indexing failed: {e}")
        else:
            print("  No GNN model found, skipping structural index")

    # Update manifest
    manifest = load_manifest(store_dir)
    manifest["last_indexed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["index_stats"] = {
        "text_embeddings": list(embeddings.shape),
        "n_hypergraphs": len(hypergraphs),
    }
    save_manifest(store_dir, manifest)

    print("Reindex complete.")
    return 0


def _build_gnn_index(store_dir: Path, hypergraphs: list[dict], model_path: Path) -> None:
    """Build FAISS index from GNN embeddings of hypergraphs."""
    import torch
    import faiss

    # Add src to path for futon6 imports
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from futon6.graph_embed import (
        ThreadGNN, hypergraph_to_tensors, collate_graphs, embed_hypergraphs,
    )

    # Load model
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ThreadGNN(
        hidden_dim=config["hidden_dim"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_relations=config["n_relations"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Convert hypergraphs to tensors
    graphs = []
    thread_ids = []
    for hg in hypergraphs:
        try:
            x, ei = hypergraph_to_tensors(hg)
            graphs.append((x, ei))
            thread_ids.append(hg["thread_id"])
        except Exception:
            continue

    if not graphs:
        print("    No valid hypergraphs to embed")
        return

    # Embed
    t0 = time.time()
    with torch.no_grad():
        emb_tensor = embed_hypergraphs(model, graphs, "cpu", batch_size=64)
    embeddings = emb_tensor.numpy()
    print(f"    GNN embeddings: {embeddings.shape}, {time.time()-t0:.1f}s")

    # Normalize and build FAISS index
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(store_dir / "structural-similarity-index.faiss"))
    with (store_dir / "structural-similarity-index.ids.json").open("w") as f:
        json.dump(thread_ids, f)

    np.save(store_dir / "hypergraph-embeddings.npy", embeddings)
    print(f"    FAISS index: {index.ntotal} vectors")


def stats(store_dir: Path) -> int:
    """Show store statistics."""
    manifest = load_manifest(store_dir)
    entities = load_entities(store_dir)
    hypergraphs = load_hypergraphs(store_dir)

    print(f"ASE Store: {store_dir}")
    print(f"  Entities:    {len(entities)}")
    print(f"  Hypergraphs: {len(hypergraphs)}")
    print(f"  Created:     {manifest.get('created', '?')}")
    print(f"  Updated:     {manifest.get('last_updated', 'never')}")
    print(f"  Indexed:     {manifest.get('last_indexed', 'never')}")

    emb_path = store_dir / "embeddings.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        print(f"  Text embeddings: {emb.shape}")

    faiss_path = store_dir / "structural-similarity-index.faiss"
    if faiss_path.exists():
        import faiss
        idx = faiss.read_index(str(faiss_path))
        print(f"  FAISS index: {idx.ntotal} vectors, dim={idx.d}")

    if manifest.get("ingestion_log"):
        print(f"\n  Ingestion history:")
        for entry in manifest["ingestion_log"][-5:]:
            print(f"    {entry['timestamp']}: +{entry['new']} from {Path(entry['source']).name}")

    # Source distribution
    if entities:
        from collections import Counter
        sources = Counter(e.get("source_node", "unknown") for e in entities)
        print(f"\n  By source node:")
        for node, count in sources.most_common():
            print(f"    {node}: {count}")

    return 0


def query(store_dir: Path, query_text: str, top_k: int = 5) -> int:
    """Search the store by keyword."""
    entities = load_entities(store_dir)
    if not entities:
        print("Store is empty.")
        return 0

    query_lower = query_text.lower()
    terms = query_lower.split()

    scored = []
    for i, e in enumerate(entities):
        text = f"{e.get('title', '')} {e.get('question-body', '')} {e.get('answer-body', '')}".lower()
        score = sum(1 for t in terms if t in text)
        if score > 0:
            scored.append((score, i))

    scored.sort(reverse=True)
    if not scored:
        print(f"No results for '{query_text}'")
        return 0

    print(f"Top {min(top_k, len(scored))} results for '{query_text}':")
    for score, idx in scored[:top_k]:
        e = entities[idx]
        print(f"\n  [{score}] {e.get('entity/id', '?')}: {e.get('title', '')}")
        print(f"      tags: {e.get('tags', [])[:5]}")
        q = e.get("question-body", "")[:200]
        if q:
            print(f"      Q: {q}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE,
                        help=f"Store directory (default: {DEFAULT_STORE})")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Add synthetic QA threads")
    p_ingest.add_argument("input", type=Path, help="JSONL file of synthetic QA")

    sub.add_parser("reindex", help="Rebuild embeddings and FAISS index")
    sub.add_parser("stats", help="Show store statistics")

    p_query = sub.add_parser("query", help="Search the store")
    p_query.add_argument("text", help="Query text")
    p_query.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "ingest":
        return ingest(args.store, args.input)
    elif args.command == "reindex":
        return reindex(args.store)
    elif args.command == "stats":
        return stats(args.store)
    elif args.command == "query":
        return query(args.store, args.text, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
