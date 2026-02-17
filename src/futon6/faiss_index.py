"""Stage 10: FAISS index for structural similarity search.

Builds a FAISS index from thread hypergraph embeddings (Stage 9b),
enabling nearest-neighbor queries for "structurally related questions".

    >>> from futon6.faiss_index import build_index, query
    >>> index, thread_ids = build_index(embeddings, ids)
    >>> neighbors = query(index, thread_ids, query_vec, k=10)

When faiss is not installed, falls back to brute-force numpy search
(same API, slower for large corpora).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Try to import faiss; fall back to numpy brute force
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray, thread_ids: list[int],
                index_type: str = "flat",
                nlist: int = 100) -> tuple:
    """Build a FAISS index from embeddings.

    Parameters
    ----------
    embeddings : (N, D) float32 array, L2-normalized
    thread_ids : list of N thread IDs (for mapping index→thread)
    index_type : "flat" (exact, small corpus) or "ivf" (approximate, large)
    nlist : number of IVF cells (only for index_type="ivf")

    Returns
    -------
    (index, thread_ids) — the index object and the ID mapping
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, d = embeddings.shape

    if HAS_FAISS:
        if index_type == "ivf" and n > nlist * 10:
            # IVF index for large corpora
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, min(nlist, n // 10),
                                       faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(10, nlist)
        else:
            # Flat index (exact search, fine for < 1M vectors)
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
    else:
        # Numpy fallback
        index = NumpyIndex(embeddings)

    return index, list(thread_ids)


class NumpyIndex:
    """Brute-force nearest neighbor search via numpy (faiss fallback)."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.ntotal = len(embeddings)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Returns (distances, indices) arrays of shape (n_queries, k).
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        # Inner product (embeddings are L2-normalized, so this = cosine sim)
        sims = query @ self.embeddings.T  # (n_queries, N)
        # Top-k per query
        k = min(k, self.ntotal)
        top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]
        # Sort the top-k by similarity
        rows = np.arange(sims.shape[0])[:, None]
        top_k_sims = sims[rows, top_k_idx]
        sort_order = np.argsort(-top_k_sims, axis=1)
        sorted_idx = top_k_idx[rows, sort_order]
        sorted_sims = top_k_sims[rows, sort_order]
        return sorted_sims, sorted_idx


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(index, thread_ids: list[int],
          query_embedding: np.ndarray, k: int = 10,
          exclude_id: int | None = None) -> list[dict]:
    """Find k most structurally similar threads to a query embedding.

    Parameters
    ----------
    index : FAISS index or NumpyIndex
    thread_ids : ID mapping from build_index
    query_embedding : (D,) or (1, D) float32 array
    k : number of neighbors
    exclude_id : thread ID to exclude (e.g., the query thread itself)

    Returns
    -------
    List of dicts: [{thread_id, similarity, rank}, ...]
    """
    q = np.ascontiguousarray(query_embedding, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    # Search for k+1 to allow excluding self
    sims, idxs = index.search(q, min(k + 1, len(thread_ids)))

    results = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx < 0 or idx >= len(thread_ids):
            continue
        tid = thread_ids[idx]
        if tid == exclude_id:
            continue
        results.append({
            "thread_id": tid,
            "similarity": float(sim),
            "rank": len(results) + 1,
        })
        if len(results) >= k:
            break

    return results


def batch_query(index, thread_ids: list[int],
                query_embeddings: np.ndarray, k: int = 10,
                ) -> list[list[dict]]:
    """Batch query: find neighbors for multiple embeddings at once.

    Returns list of result lists, one per query.
    """
    q = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    sims, idxs = index.search(q, min(k + 1, len(thread_ids)))

    all_results = []
    for i in range(len(q)):
        results = []
        query_tid = thread_ids[i] if i < len(thread_ids) else None
        for sim, idx in zip(sims[i], idxs[i]):
            if idx < 0 or idx >= len(thread_ids):
                continue
            tid = thread_ids[idx]
            if tid == query_tid:
                continue
            results.append({
                "thread_id": tid,
                "similarity": float(sim),
                "rank": len(results) + 1,
            })
            if len(results) >= k:
                break
        all_results.append(results)

    return all_results


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_index(index, thread_ids: list[int], path: str) -> None:
    """Save index and thread ID mapping to disk."""
    p = Path(path)
    if HAS_FAISS and not isinstance(index, NumpyIndex):
        faiss.write_index(index, str(p.with_suffix(".faiss")))
    else:
        np.save(str(p.with_suffix(".npy")), index.embeddings)

    with open(str(p.with_suffix(".ids.json")), "w") as f:
        json.dump(thread_ids, f)


def load_index(path: str) -> tuple:
    """Load index and thread ID mapping from disk."""
    p = Path(path)
    ids_path = p.with_suffix(".ids.json")
    with open(ids_path) as f:
        thread_ids = json.load(f)

    faiss_path = p.with_suffix(".faiss")
    npy_path = p.with_suffix(".npy")

    if HAS_FAISS and faiss_path.exists():
        index = faiss.read_index(str(faiss_path))
    elif npy_path.exists():
        embeddings = np.load(str(npy_path))
        if HAS_FAISS:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
        else:
            index = NumpyIndex(embeddings)
    else:
        raise FileNotFoundError(f"No index file at {faiss_path} or {npy_path}")

    return index, thread_ids
