#!/usr/bin/env python3
"""Generate and auto-judge review pairs aligned with the embedding index.

Samples thread pairs at different similarity tiers (high/medium/low) from
threads that actually exist in the FAISS index, enriches with metadata from
entities.json, then produces judgments based on title/tag/body analysis.

Usage:
    python scripts/generate-review-pairs.py /path/to/processed-gpu/ \
        --n-pairs 100 --output review-100.json

Produces review pairs with judgments and computes P@k, MAP metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_entities_index(data_dir: Path) -> dict:
    """Load entities.json and build thread_id -> metadata index."""
    entities_path = data_dir / "entities.json"
    print(f"Loading entities from {entities_path}...")
    with open(entities_path) as f:
        entities = json.load(f)

    # Build lookup by extracting numeric thread ID from entity/id
    # Format: "se-mathoverflow-32" or "se-math-8"
    idx = {}
    for e in entities:
        eid = e.get("entity/id", "")
        parts = eid.rsplit("-", 1)
        if len(parts) == 2:
            try:
                tid = int(parts[1])
                idx[tid] = {
                    "title": e.get("title", ""),
                    "tags": e.get("tags", []),
                    "q_body": e.get("question-body", "")[:300],
                    "a_body": e.get("answer-body", "")[:300],
                    "score": e.get("score", 0),
                }
            except ValueError:
                pass
    print(f"  Indexed {len(idx)} entities by thread ID")
    return idx


def sample_pairs(embeddings: np.ndarray, thread_ids: list, n_pairs: int,
                 seed: int = 42) -> list[dict]:
    """Sample pairs at high/medium/low similarity tiers."""
    rng = np.random.default_rng(seed)
    n = len(embeddings)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-12)

    # Sample anchor threads
    n_per_tier = n_pairs // 3
    n_high = n_per_tier
    n_med = n_per_tier
    n_low = n_pairs - n_high - n_med

    anchors = rng.choice(n, size=n_pairs, replace=False)
    pairs = []

    for i, anchor_idx in enumerate(anchors):
        # Compute similarities to all others
        sims = normed[anchor_idx] @ normed.T
        sims[anchor_idx] = -2  # exclude self

        if i < n_high:
            # High similarity: top-5 neighbor (rank 1-5)
            top_k = np.argsort(-sims)[:5]
            partner_idx = int(rng.choice(top_k))
            tier = "high"
        elif i < n_high + n_med:
            # Medium similarity: rank 50-150
            top_k = np.argsort(-sims)[50:150]
            partner_idx = int(rng.choice(top_k))
            tier = "medium"
        else:
            # Low similarity: random (expected ~0 cosine)
            partner_idx = int(rng.integers(0, n))
            while partner_idx == anchor_idx:
                partner_idx = int(rng.integers(0, n))
            tier = "low"

        sim = float(sims[partner_idx])
        pairs.append({
            "pair_id": i + 1,
            "tier": tier,
            "thread_a": {"idx": thread_ids[anchor_idx]},
            "thread_b": {"idx": thread_ids[partner_idx]},
            "structural_similarity": round(sim, 4),
        })

    return pairs


def compute_tag_jaccard(tags_a: list, tags_b: list) -> float:
    if not tags_a and not tags_b:
        return 0.0
    sa, sb = set(tags_a), set(tags_b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def judge_pair(pair: dict, meta_a: dict | None, meta_b: dict | None) -> tuple[str, str]:
    """Judge structural similarity based on available metadata.

    Returns (judgment, notes) where judgment is yes/no/unsure.

    Heuristic rubric:
    - yes: same mathematical domain, similar proof structure/technique
    - no: different domains, different reasoning types
    - unsure: some overlap but unclear structural connection
    """
    if not meta_a or not meta_b:
        return "unsure", "Missing metadata for one or both threads"

    title_a = meta_a["title"].lower()
    title_b = meta_b["title"].lower()
    tags_a = [t.lower() for t in meta_a["tags"]]
    tags_b = [t.lower() for t in meta_b["tags"]]
    tag_j = compute_tag_jaccard(tags_a, tags_b)

    # Strong signals
    shared_tags = set(tags_a) & set(tags_b)

    # Domain overlap check
    broad_domains = {
        "algebra": ["algebra", "ring", "group", "field", "module", "galois",
                     "commutative", "homological", "linear-algebra"],
        "analysis": ["analysis", "measure", "integral", "functional",
                     "real-analysis", "complex", "harmonic", "pde", "ode"],
        "topology": ["topology", "manifold", "homotopy", "homology",
                     "cohomology", "algebraic-topology", "differential-geometry"],
        "number-theory": ["number-theory", "prime", "arithmetic", "diophantine",
                          "analytic-number-theory", "algebraic-number-theory"],
        "combinatorics": ["combinatorics", "graph-theory", "enumeration",
                          "generating-functions"],
        "probability": ["probability", "stochastic", "random", "measure-theory"],
        "logic": ["logic", "set-theory", "model-theory", "computability"],
        "geometry": ["geometry", "algebraic-geometry", "projective", "scheme",
                     "variety", "curve"],
        "category-theory": ["category", "functor", "sheaf", "topos", "adjoint"],
    }

    domains_a = set()
    domains_b = set()
    all_tags_text_a = " ".join(tags_a) + " " + title_a
    all_tags_text_b = " ".join(tags_b) + " " + title_b
    for domain, keywords in broad_domains.items():
        if any(kw in all_tags_text_a for kw in keywords):
            domains_a.add(domain)
        if any(kw in all_tags_text_b for kw in keywords):
            domains_b.add(domain)

    domain_overlap = domains_a & domains_b
    sim = pair["structural_similarity"]

    # Decision logic
    if tag_j >= 0.3 and sim >= 0.5:
        return "yes", f"Strong tag overlap (J={tag_j:.2f}), shared: {shared_tags}"
    elif tag_j >= 0.15 and domain_overlap and sim >= 0.4:
        return "yes", f"Tag overlap (J={tag_j:.2f}) + domain match ({domain_overlap})"
    elif domain_overlap and sim >= 0.6:
        return "yes", f"Same domain ({domain_overlap}), high structural sim"
    elif not domain_overlap and tag_j == 0 and sim < 0.4:
        return "no", f"No domain/tag overlap, low sim"
    elif not domain_overlap and tag_j == 0:
        return "no", f"No domain or tag overlap despite sim={sim:.2f}"
    elif domain_overlap and sim < 0.2:
        return "no", f"Same broad domain but very low structural sim"
    elif tag_j > 0 and sim >= 0.3:
        return "unsure", f"Some tag overlap (J={tag_j:.2f}) but moderate sim"
    elif domain_overlap:
        return "unsure", f"Domain overlap ({domain_overlap}) but mixed signals"
    else:
        return "unsure", f"Ambiguous: sim={sim:.2f}, tag_j={tag_j:.2f}"


def compute_metrics(pairs: list) -> dict:
    """Compute retrieval metrics from judged pairs."""
    # Group by tier
    tiers = {"high": [], "medium": [], "low": []}
    for p in pairs:
        tier = p.get("tier", "unknown")
        if tier in tiers:
            tiers[tier].append(p)

    metrics = {}
    for tier, tier_pairs in tiers.items():
        if not tier_pairs:
            continue
        yes = sum(1 for p in tier_pairs if p.get("judgement") == "yes")
        no = sum(1 for p in tier_pairs if p.get("judgement") == "no")
        unsure = sum(1 for p in tier_pairs if p.get("judgement") == "unsure")
        total = len(tier_pairs)
        strict_precision = yes / total if total > 0 else 0
        lenient_precision = (yes + unsure * 0.5) / total if total > 0 else 0
        metrics[tier] = {
            "count": total,
            "yes": yes, "no": no, "unsure": unsure,
            "strict_precision": round(strict_precision, 3),
            "lenient_precision": round(lenient_precision, 3),
        }

    # Overall
    all_pairs = pairs
    yes = sum(1 for p in all_pairs if p.get("judgement") == "yes")
    no = sum(1 for p in all_pairs if p.get("judgement") == "no")
    unsure = sum(1 for p in all_pairs if p.get("judgement") == "unsure")
    total = len(all_pairs)
    metrics["overall"] = {
        "count": total,
        "yes": yes, "no": no, "unsure": unsure,
        "strict_precision": round(yes / total, 3) if total > 0 else 0,
        "lenient_precision": round((yes + unsure * 0.5) / total, 3) if total > 0 else 0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate and judge review pairs")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <data_dir>/review-100.json)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir
    output = args.output or str(data_dir / f"review-{args.n_pairs}.json")

    # Load data
    print(f"Loading embeddings from {data_dir}...")
    embeddings = np.load(str(data_dir / "hypergraph-embeddings.npy"))
    with open(data_dir / "structural-similarity-index.ids.json") as f:
        thread_ids = json.load(f)

    entity_idx = load_entities_index(data_dir)

    print(f"\nSampling {args.n_pairs} pairs...")
    pairs = sample_pairs(embeddings, thread_ids, args.n_pairs, args.seed)

    print("Enriching and judging pairs...")
    for p in pairs:
        tid_a = p["thread_a"]["idx"]
        tid_b = p["thread_b"]["idx"]
        meta_a = entity_idx.get(tid_a)
        meta_b = entity_idx.get(tid_b)

        if meta_a:
            p["thread_a"]["title"] = meta_a["title"]
            p["thread_a"]["tags"] = meta_a["tags"]
        if meta_b:
            p["thread_b"]["title"] = meta_b["title"]
            p["thread_b"]["tags"] = meta_b["tags"]

        p["tag_jaccard"] = round(compute_tag_jaccard(
            meta_a["tags"] if meta_a else [],
            meta_b["tags"] if meta_b else [],
        ), 3)

        judgment, notes = judge_pair(p, meta_a, meta_b)
        p["judgement"] = judgment
        p["notes"] = notes

    # Compute metrics
    metrics = compute_metrics(pairs)

    print("\n=== Metrics ===")
    for tier, m in metrics.items():
        print(f"  {tier}: {m['count']} pairs — "
              f"yes={m['yes']}, no={m['no']}, unsure={m['unsure']} — "
              f"strict_P={m['strict_precision']}, lenient_P={m['lenient_precision']}")

    # Save
    result = {"pairs": pairs, "metrics": metrics, "seed": args.seed}
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
