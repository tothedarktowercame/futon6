"""Regression tests for structural embedding/thread-id alignment invariants."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_eval_module(root: Path):
    path = root / "scripts" / "evaluate-superpod-run.py"
    spec = importlib.util.spec_from_file_location("evaluate_superpod_run", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_embeddings_aligns_by_thread_id(tmp_path: Path):
    root = Path(__file__).parent.parent
    mod = _load_eval_module(root)

    # Text embeddings are in entity row order (101, 102, 103).
    text = np.array(
        [
            [1.0, 0.0],  # thread 101
            [0.0, 1.0],  # thread 102
            [0.7, 0.7],  # thread 103
        ],
        dtype=np.float32,
    )
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    np.save(tmp_path / "embeddings.npy", text)

    # Structural embeddings are row-ordered as (102, 101).
    struct = np.array(
        [
            [0.0, 1.0],  # thread 102
            [1.0, 0.0],  # thread 101
        ],
        dtype=np.float32,
    )
    np.save(tmp_path / "hypergraph-embeddings.npy", struct)

    entities = [
        {"entity/id": "se-math-101", "tags": ["alpha"]},
        {"entity/id": "se-math-102", "tags": ["beta"]},
        {"entity/id": "se-math-103", "tags": ["gamma"]},
    ]
    (tmp_path / "entities.json").write_text(json.dumps(entities), encoding="utf-8")

    # Legacy mismatch case: extra ID entry should be truncated safely.
    (tmp_path / "hypergraph-thread-ids.json").write_text(
        json.dumps([102, 101, 999]), encoding="utf-8"
    )

    result = mod.compare_embeddings(tmp_path, n_sample=2, k=1)
    assert isinstance(result, tuple)
    report, _candidates = result

    assert report["ok"] is True
    assert report["n_struct_rows"] == 2
    assert report["n_struct_aligned_rows"] == 2
    assert report.get("warnings")
    assert "mismatch" in report["warnings"][0]


def test_stage10_rejects_id_embedding_mismatch(tmp_path: Path):
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root / "scripts"))
    mod = importlib.import_module("superpod-job")

    emb = np.ones((2, 4), dtype=np.float32)
    emb_path = tmp_path / "hypergraph-embeddings.npy"
    np.save(emb_path, emb)

    with pytest.raises(ValueError, match="thread ID count does not match embedding rows"):
        mod.run_stage10_faiss_index(emb_path, [11, 12, 13], tmp_path)
