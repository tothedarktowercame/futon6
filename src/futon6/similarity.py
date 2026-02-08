"""Embedding-based similarity for mathematical entries.

Uses sentence-transformers to compute embeddings from entry titles and
keywords, then finds similar entries across or within datasets.

This module is optional — it requires sentence-transformers and a model
download. The rest of futon6 works without it.
"""

import numpy as np


def _build_text(entry: dict) -> str:
    """Build a text representation of an entry for embedding."""
    parts = []
    if entry.get("title"):
        parts.append(entry["title"])
    if entry.get("keywords"):
        parts.append(", ".join(entry["keywords"][:10]))
    if entry.get("defines"):
        parts.append("defines: " + ", ".join(entry["defines"][:5]))
    return ". ".join(parts) if parts else entry.get("entity/id", "")


def compute_embeddings(entries: list[dict], model_name: str = "all-MiniLM-L6-v2"):
    """Compute embeddings for a list of entries.

    Args:
        entries: list of entry dicts (must have title/keywords)
        model_name: sentence-transformers model name

    Returns:
        numpy array of shape (len(entries), embedding_dim)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [_build_text(e) for e in entries]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def find_similar(
    entries: list[dict],
    embeddings: np.ndarray,
    query_idx: int,
    top_k: int = 5,
    threshold: float = 0.3,
) -> list[dict]:
    """Find entries most similar to the entry at query_idx.

    Returns list of {entry, score, idx} dicts, excluding self.
    """
    query_vec = embeddings[query_idx]
    # Cosine similarity (embeddings are already normalized by sentence-transformers)
    scores = embeddings @ query_vec
    ranked = np.argsort(-scores)

    results = []
    for idx in ranked:
        idx = int(idx)
        if idx == query_idx:
            continue
        score = float(scores[idx])
        if score < threshold:
            break
        results.append({
            "entry": entries[idx],
            "score": score,
            "idx": idx,
        })
        if len(results) >= top_k:
            break
    return results


def cross_source_matches(
    entries_a: list[dict],
    entries_b: list[dict],
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    threshold: float = 0.7,
) -> list[dict]:
    """Find matching entries across two sources (e.g., PlanetMath vs nLab).

    Returns list of {a, b, score} dicts where score >= threshold.
    """
    # Cosine similarity matrix
    sim_matrix = embeddings_a @ embeddings_b.T

    matches = []
    for i in range(len(entries_a)):
        for j in range(len(entries_b)):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                matches.append({
                    "a": entries_a[i],
                    "b": entries_b[j],
                    "score": score,
                })

    matches.sort(key=lambda m: -m["score"])
    return matches
