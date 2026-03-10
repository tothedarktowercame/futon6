#!/usr/bin/env python3
"""Cluster MO reverse morphogenesis situations into candidate question patterns."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

SITUATION_RE = re.compile(r"situation\s*s[^\w]?", re.IGNORECASE)
SECTION_SPLIT_RE = re.compile(r"\n\s*(?:classify|identify|verify|analysis|question)\b", re.IGNORECASE)


def extract_situation(raw_text: str, window: int) -> str:
    if not raw_text:
        return ""
    match = SITUATION_RE.search(raw_text)
    if match:
        snippet = raw_text[match.start(): match.start() + window]
    else:
        snippet = raw_text[:window]
    # Stop at the next section header if one exists within the window
    split_match = SECTION_SPLIT_RE.search(snippet)
    if split_match:
        snippet = snippet[: split_match.start()]
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet.strip()


def load_mo_situations(path: Path, limit: int | None, window: int) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = []
    for idx, entry in enumerate(tqdm(data, desc="reading situations")):
        if limit and idx >= limit:
            break
        raw = entry.get("raw") or entry.get("analysis", {}).get("raw", "")
        snippet = extract_situation(raw, window)
        if not snippet:
            continue
        entries.append(
            {
                "entity_id": entry.get("entity_id"),
                "question_id": entry.get("question_id"),
                "dataset_index": idx,
                "situation": snippet,
            }
        )
    return entries


def embed_situations(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def choose_cluster_count(
    embeddings: np.ndarray,
    candidates: Sequence[int],
    sample_size: int,
    random_state: int,
) -> int:
    rng = np.random.default_rng(random_state)
    n = embeddings.shape[0]
    if n <= 2 or not candidates:
        return max(2, candidates[-1] if candidates else 25)
    sample_size = min(sample_size, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample = embeddings[sample_idx]
    best_k = None
    best_score = -math.inf
    for k in candidates:
        if k >= sample.shape[0]:
            continue
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=1024,
            n_init="auto",
            max_iter=200,
        )
        labels = km.fit_predict(sample)
        # Silhouette score requires at least 2 clusters with >1 samples
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(sample, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k or candidates[-1]


def summarize_clusters(
    entries: Sequence[dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_k: int = 5,
) -> List[dict]:
    clusters = []
    label_to_indices: dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[int(label)].append(idx)
    for cluster_id, indices in sorted(label_to_indices.items()):
        cluster_vecs = embeddings[indices]
        centroid = cluster_vecs.mean(axis=0)
        distances = np.linalg.norm(cluster_vecs - centroid, axis=1)
        order = np.argsort(distances)[:top_k]
        reps = []
        for rel_idx in order:
            abs_idx = indices[int(rel_idx)]
            reps.append(
                {
                    "entity_id": entries[abs_idx]["entity_id"],
                    "question_id": entries[abs_idx]["question_id"],
                    "situation": entries[abs_idx]["situation"],
                }
            )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "centroid": centroid.tolist(),
                "representatives": reps,
            }
        )
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster MO situations into candidate question patterns")
    parser.add_argument("--dataset", type=Path, default=Path("/home/joe/code/storage/mo-processed-gpu/reverse-morphogenesis.json"))
    parser.add_argument("--output", type=Path, default=Path("data/question-patterns/mo-situation-clusters.json"))
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries for testing")
    parser.add_argument("--window", type=int, default=1200, help="Character window for situation extraction")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--cluster-count", type=int, default=None)
    parser.add_argument("--cluster-candidates", type=int, nargs="*", default=[30, 45, 60, 80, 100])
    parser.add_argument("--silhouette-sample", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entries = load_mo_situations(args.dataset, args.limit, args.window)
    if not entries:
        raise SystemExit("No situations extracted")

    if args.embedding_cache and args.embedding_cache.exists():
        cache = np.load(args.embedding_cache)
        max_index = max(e["dataset_index"] for e in entries)
        if cache.shape[0] <= max_index:
            raise SystemExit("embedding cache is smaller than dataset slice")
        embeddings = np.stack([cache[e["dataset_index"]] for e in entries])
    else:
        embeddings = embed_situations([e["situation"] for e in entries], args.model, args.batch_size, args.device)

    if args.cluster_count:
        num_clusters = args.cluster_count
    else:
        num_clusters = choose_cluster_count(embeddings, args.cluster_candidates, args.silhouette_sample, args.seed)
    km = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=args.seed,
        batch_size=2048,
        n_init="auto",
        max_iter=300,
    )
    labels = km.fit_predict(embeddings)

    clusters = summarize_clusters(entries, embeddings, labels)
    assignments = []
    for idx, entry in enumerate(entries):
        assignments.append(
            {
                "entity_id": entry["entity_id"],
                "question_id": entry["question_id"],
                "cluster_id": int(labels[idx]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "num_entries": len(entries),
        "num_clusters": num_clusters,
        "clusters": clusters,
        "assignments": assignments,
    }
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"wrote {args.output} ({len(entries)} entries, {num_clusters} clusters)")


if __name__ == "__main__":
    main()
