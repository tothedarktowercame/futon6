#!/usr/bin/env python3
"""A2: Run Stage 4 clustering on merged embeddings.

HDBSCAN clustering was skipped in shard mode during the superpod run.
This script runs post-merge clustering on the existing embeddings.npy files.

CPU-only, fits in RAM (<4GB for 900K × 1024 float32).

Usage:
    python3 scripts/run-stage4-clustering.py --source math
    python3 scripts/run-stage4-clustering.py --source mo
    python3 scripts/run-stage4-clustering.py --source both
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

STORAGE = {
    "math": Path(os.path.expanduser("~/code/storage/math-processed-gpu")),
    "mo": Path(os.path.expanduser("~/code/storage/mo-processed-gpu")),
}


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    max_clusters: int = 500,
) -> tuple[np.ndarray, dict]:
    """Cluster embeddings using HDBSCAN with KMeans fallback."""
    n = len(embeddings)
    stats = {"n_embeddings": n, "method": None}

    try:
        from sklearn.cluster import HDBSCAN

        print(f"  Using HDBSCAN (min_cluster_size={min_cluster_size})...")
        t0 = time.time()
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="cosine",
            n_jobs=-1,
        )
        labels = clusterer.fit_predict(embeddings)
        elapsed = time.time() - t0

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        stats.update({
            "method": "hdbscan",
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": n_noise / n,
            "elapsed_seconds": round(elapsed, 1),
        })
        print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise ({n_noise/n:.1%}), "
              f"{elapsed:.1f}s")
        return labels, stats

    except Exception as e:
        print(f"  HDBSCAN failed ({e}), falling back to KMeans...")

    from sklearn.cluster import MiniBatchKMeans

    k = min(max_clusters, max(10, int(np.sqrt(n / 2))))
    print(f"  Using MiniBatchKMeans (k={k})...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42, n_init=3)
    labels = km.fit_predict(embeddings)
    elapsed = time.time() - t0

    stats.update({
        "method": "kmeans",
        "n_clusters": k,
        "n_noise": 0,
        "noise_fraction": 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "inertia": float(km.inertia_),
    })
    print(f"  KMeans: {k} clusters, {elapsed:.1f}s, inertia={km.inertia_:.0f}")
    return labels, stats


def cluster_source(source: str) -> dict:
    outdir = STORAGE[source]
    emb_path = outdir / "embeddings.npy"
    if not emb_path.exists():
        print(f"  Embeddings not found: {emb_path}")
        return {"error": f"missing {emb_path}"}

    print(f"\n=== Clustering {source} ===")
    embeddings = np.load(emb_path)
    print(f"  Loaded {embeddings.shape[0]} × {embeddings.shape[1]} embeddings")

    labels, stats = cluster_embeddings(embeddings)

    # Save results
    labels_path = outdir / "cluster-labels.npy"
    np.save(labels_path, labels)
    print(f"  Saved cluster labels to {labels_path}")

    stats_path = outdir / "cluster-stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved cluster stats to {stats_path}")

    # Cluster size distribution
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(counts):
        print(f"  Cluster sizes: min={counts.min()}, median={int(np.median(counts))}, "
              f"max={counts.max()}, mean={counts.mean():.0f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="A2: Post-merge HDBSCAN clustering")
    parser.add_argument("--source", choices=["math", "mo", "both"], default="both")
    args = parser.parse_args()

    sources = ["math", "mo"] if args.source == "both" else [args.source]
    results = {}
    for src in sources:
        results[src] = cluster_source(src)

    print("\n=== Summary ===")
    for src, stats in results.items():
        if "error" in stats:
            print(f"  {src}: {stats['error']}")
        else:
            print(f"  {src}: {stats['method']}, {stats['n_clusters']} clusters, "
                  f"{stats.get('noise_fraction', 0):.1%} noise, "
                  f"{stats['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
