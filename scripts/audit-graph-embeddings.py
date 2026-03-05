#!/usr/bin/env python3
"""A3: Audit Stage 9b graph embeddings for quality and discrimination.

Checks:
1. Embedding collapse — are all vectors near-identical?
2. Cosine similarity distribution — random pairs vs. self-similarity
3. Review-50 alignment — do human judgments correlate with embedding similarity?
4. Nearest-neighbor quality — what do top-k neighbors look like?

Usage:
    python scripts/audit-graph-embeddings.py /path/to/processed-gpu/

Reads: hypergraph-embeddings.npy, structural-similarity-index.ids.json,
       review-50.json, stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_data(data_dir: Path):
    emb_path = data_dir / "hypergraph-embeddings.npy"
    ids_path = data_dir / "structural-similarity-index.ids.json"
    review_path = data_dir / "review-50.json"
    stats_path = data_dir / "stats.json"

    embeddings = np.load(str(emb_path))
    with open(ids_path) as f:
        thread_ids = json.load(f)
    with open(review_path) as f:
        review = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)

    return embeddings, thread_ids, review, stats


def check_collapse(embeddings: np.ndarray):
    """Check if embeddings have collapsed to near-identical vectors."""
    print("\n=== 1. Embedding Collapse Check ===")
    n, d = embeddings.shape
    print(f"Shape: {n} vectors × {d} dimensions")

    # Check norms
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norm range: [{norms.min():.4f}, {norms.max():.4f}], mean={norms.mean():.4f}, std={norms.std():.6f}")

    # Check variance per dimension
    dim_var = embeddings.var(axis=0)
    print(f"Per-dim variance: mean={dim_var.mean():.6f}, min={dim_var.min():.6f}, max={dim_var.max():.6f}")
    low_var_dims = (dim_var < 1e-6).sum()
    print(f"Dead dimensions (var < 1e-6): {low_var_dims}/{d}")

    # Mean vector — if collapsed, all vectors ≈ mean
    mean_vec = embeddings.mean(axis=0)
    mean_vec_norm = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
    cosines_to_mean = embeddings @ mean_vec_norm
    print(f"Cosine to mean vector: mean={cosines_to_mean.mean():.4f}, std={cosines_to_mean.std():.4f}")

    if cosines_to_mean.std() < 0.01:
        print("⚠ WARNING: Very low spread around mean — possible collapse!")
    elif cosines_to_mean.std() < 0.05:
        print("⚠ CAUTION: Low spread around mean — embeddings may be poorly discriminative")
    else:
        print("✓ Spread looks healthy")

    return dim_var, cosines_to_mean


def check_similarity_distribution(embeddings: np.ndarray, n_samples=10000):
    """Sample random pairs and compute cosine similarity distribution."""
    print("\n=== 2. Random-Pair Cosine Similarity Distribution ===")
    n = len(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-12)

    # Sample random pairs
    rng = np.random.default_rng(42)
    i1 = rng.integers(0, n, size=n_samples)
    i2 = rng.integers(0, n, size=n_samples)
    # Ensure no self-pairs
    mask = i1 == i2
    i2[mask] = (i2[mask] + 1) % n

    cosines = (normed[i1] * normed[i2]).sum(axis=1)

    percentiles = np.percentile(cosines, [5, 25, 50, 75, 95])
    print(f"Random-pair cosine similarity (n={n_samples}):")
    print(f"  5th pctile: {percentiles[0]:.4f}")
    print(f"  25th pctile: {percentiles[1]:.4f}")
    print(f"  Median:      {percentiles[2]:.4f}")
    print(f"  75th pctile: {percentiles[3]:.4f}")
    print(f"  95th pctile: {percentiles[4]:.4f}")
    print(f"  Mean: {cosines.mean():.4f}, Std: {cosines.std():.4f}")

    if cosines.std() < 0.02:
        print("⚠ WARNING: Nearly no variation — all pairs equally similar")
    elif percentiles[2] > 0.9:
        print("⚠ WARNING: Median similarity very high — may indicate collapse")
    else:
        print("✓ Distribution looks discriminative")

    return cosines


def check_review50(embeddings: np.ndarray, thread_ids: list, review: list | dict):
    """Check if review-50 human judgments correlate with embedding similarity."""
    print("\n=== 3. Review-50 Human Judgment Alignment ===")

    # Handle both list and dict formats
    pairs = review if isinstance(review, list) else review.get("pairs", [])
    if not pairs:
        print("No review pairs found")
        return

    # Build thread_id -> index mapping
    tid_to_idx = {tid: i for i, tid in enumerate(thread_ids)}

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-12)

    yes_sims = []
    no_sims = []
    unsure_sims = []
    missing = 0

    for pair in pairs:
        judgment = pair.get("judgement", pair.get("judgment", pair.get("label", "")))
        if not judgment:
            continue

        # Handle nested format: {thread_a: {idx: ...}} or flat {thread_a: id}
        ta = pair.get("thread_a", pair.get("id_a"))
        tb = pair.get("thread_b", pair.get("id_b"))
        id_a = ta.get("idx") if isinstance(ta, dict) else ta
        id_b = tb.get("idx") if isinstance(tb, dict) else tb
        if id_a is None or id_b is None:
            missing += 1
            continue

        idx_a = tid_to_idx.get(id_a)
        idx_b = tid_to_idx.get(id_b)
        if idx_a is None or idx_b is None:
            missing += 1
            continue

        sim = float(normed[idx_a] @ normed[idx_b])

        if judgment == "yes":
            yes_sims.append(sim)
        elif judgment == "no":
            no_sims.append(sim)
        elif judgment == "unsure":
            unsure_sims.append(sim)

    print(f"Pairs resolved: yes={len(yes_sims)}, no={len(no_sims)}, unsure={len(unsure_sims)}, missing/unresolved={missing}")

    if yes_sims:
        print(f"  'yes' mean sim:    {np.mean(yes_sims):.4f} (std={np.std(yes_sims):.4f})")
    if no_sims:
        print(f"  'no' mean sim:     {np.mean(no_sims):.4f} (std={np.std(no_sims):.4f})")
    if unsure_sims:
        print(f"  'unsure' mean sim: {np.mean(unsure_sims):.4f} (std={np.std(unsure_sims):.4f})")

    if missing > 0:
        total = len(pairs)
        print(f"  NOTE: {missing}/{total} pairs had thread IDs not in embedding index")

    if yes_sims and no_sims:
        gap = np.mean(yes_sims) - np.mean(no_sims)
        print(f"  Gap (yes - no): {gap:.4f}")
        if gap > 0.05:
            print("✓ Embeddings discriminate between yes/no pairs")
        elif gap > 0:
            print("⚠ CAUTION: Small positive gap — weak discrimination")
        else:
            print("⚠ WARNING: No positive gap — embeddings don't align with judgments")
    elif missing > len(pairs) * 0.5:
        print("⚠ WARNING: Too many review pairs missing from index to draw conclusions")


def check_nearest_neighbors(embeddings: np.ndarray, thread_ids: list, k=5, n_probes=10):
    """Spot-check nearest neighbors for a few random threads."""
    print(f"\n=== 4. Nearest-Neighbor Spot Check (k={k}, {n_probes} probes) ===")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-12)

    rng = np.random.default_rng(42)
    probe_indices = rng.integers(0, len(embeddings), size=n_probes)

    all_nn_sims = []
    for i, idx in enumerate(probe_indices):
        sims = normed[idx] @ normed.T
        # Exclude self
        sims[idx] = -1
        top_k = np.argsort(-sims)[:k]
        top_sims = sims[top_k]
        all_nn_sims.extend(top_sims.tolist())
        if i < 3:  # Show first 3 probes
            print(f"  Thread {thread_ids[idx]}:")
            for rank, (nn_idx, s) in enumerate(zip(top_k, top_sims), 1):
                print(f"    #{rank}: thread {thread_ids[nn_idx]}, sim={s:.4f}")

    nn_sims = np.array(all_nn_sims)
    print(f"\n  Top-{k} neighbor similarity stats across {n_probes} probes:")
    print(f"    Mean: {nn_sims.mean():.4f}, Std: {nn_sims.std():.4f}")
    print(f"    Min:  {nn_sims.min():.4f}, Max: {nn_sims.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Audit graph embeddings quality")
    parser.add_argument("data_dir", type=Path, help="Path to processed-gpu directory")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Auditing graph embeddings in {args.data_dir}")
    embeddings, thread_ids, review, stats = load_data(args.data_dir)
    print(f"Corpus: {stats.get('qa_pairs', '?')} QA pairs")

    check_collapse(embeddings)
    check_similarity_distribution(embeddings)
    check_review50(embeddings, thread_ids, review)
    check_nearest_neighbors(embeddings, thread_ids)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
