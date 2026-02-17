"""Tests for FAISS index module (Stage 10)."""

import json
import os
import pytest
import numpy as np

from futon6.faiss_index import (
    build_index,
    query,
    batch_query,
    save_index,
    load_index,
    NumpyIndex,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    embeddings = np.random.randn(50, 32).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    thread_ids = list(range(1000, 1050))
    return embeddings, thread_ids


def test_build_index(sample_data):
    embeddings, thread_ids = sample_data
    index, ids = build_index(embeddings, thread_ids)
    assert index.ntotal == 50
    assert len(ids) == 50


def test_query(sample_data):
    embeddings, thread_ids = sample_data
    index, ids = build_index(embeddings, thread_ids)
    results = query(index, ids, embeddings[0], k=5, exclude_id=1000)
    assert len(results) == 5
    assert all(r["thread_id"] != 1000 for r in results)
    # Similarities should be descending
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_batch_query(sample_data):
    embeddings, thread_ids = sample_data
    index, ids = build_index(embeddings, thread_ids)
    results = batch_query(index, ids, embeddings[:3], k=3)
    assert len(results) == 3
    assert all(len(r) == 3 for r in results)


def test_save_load(sample_data, tmp_path):
    embeddings, thread_ids = sample_data
    index, ids = build_index(embeddings, thread_ids)

    save_index(index, ids, str(tmp_path / "test-index"))
    index2, ids2 = load_index(str(tmp_path / "test-index"))

    assert ids == ids2
    # Same query results
    r1 = query(index, ids, embeddings[0], k=5, exclude_id=1000)
    r2 = query(index2, ids2, embeddings[0], k=5, exclude_id=1000)
    assert [r["thread_id"] for r in r1] == [r["thread_id"] for r in r2]


def test_numpy_fallback():
    """Directly test the NumpyIndex fallback."""
    embeddings = np.eye(5, dtype=np.float32)
    index = NumpyIndex(embeddings)
    sims, idxs = index.search(embeddings[:1], k=3)
    assert idxs[0][0] == 0  # self is most similar
    assert abs(sims[0][0] - 1.0) < 1e-5
