"""Tests for graph embedding module (Stage 9b)."""

import json
import os
import pytest

torch = pytest.importorskip("torch")

from futon6.graph_embed import (
    hypergraph_to_tensors,
    collate_graphs,
    augment_graph,
    ThreadGNN,
    info_nce_loss,
    train,
    save_tensor_cache,
    load_tensor_cache,
)


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "first-proof")


@pytest.fixture
def sample_hg():
    path = os.path.join(FIXTURES_DIR, "thread-633512-hypergraph.json")
    if not os.path.exists(path):
        pytest.skip("thread-633512-hypergraph.json not found")
    with open(path) as f:
        return json.load(f)


def test_hypergraph_to_tensors(sample_hg):
    x, ei = hypergraph_to_tensors(sample_hg)
    assert x.shape[0] == sample_hg["meta"]["n_nodes"]
    assert x.shape[1] == 2
    assert len(ei) > 0
    total_edges = sum(v.shape[1] for v in ei.values())
    assert total_edges > 0


def test_augment_graph(sample_hg):
    x, ei = hypergraph_to_tensors(sample_hg)
    x2, ei2 = augment_graph(x, ei, node_drop=0.2, edge_drop=0.3)
    # Should have fewer nodes
    assert x2.shape[0] <= x.shape[0]
    assert x2.shape[0] >= 2


def test_collate_graphs(sample_hg):
    x, ei = hypergraph_to_tensors(sample_hg)
    batch = collate_graphs([(x, ei), (x, ei)])
    assert batch.x.shape[0] == 2 * x.shape[0]
    assert batch.n_graphs == 2
    assert batch.batch.max().item() == 1


def test_model_forward(sample_hg):
    x, ei = hypergraph_to_tensors(sample_hg)
    batch = collate_graphs([(x, ei)])
    model = ThreadGNN(hidden_dim=32, embed_dim=16, n_layers=2)
    emb = model.embed(batch)
    assert emb.shape == (1, 16)
    # Should be L2-normalized
    assert abs(emb.norm().item() - 1.0) < 1e-5


def test_info_nce_loss():
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    loss = info_nce_loss(z1, z2)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_train_smoke(sample_hg):
    import copy
    corpus = [copy.deepcopy(sample_hg) for _ in range(4)]
    model, embeddings = train(
        corpus, dim=16, hidden_dim=32, n_layers=1,
        epochs=2, batch_size=2, verbose=False)
    assert embeddings.shape == (4, 16)


def test_minimal_graph():
    """Test with a minimal 2-node graph."""
    hg = {
        "nodes": [
            {"id": "a", "type": "post", "subtype": "question", "attrs": {}},
            {"id": "b", "type": "post", "subtype": "answer", "attrs": {}},
        ],
        "edges": [
            {"type": "iatc", "ends": ["b", "a"], "attrs": {}},
        ],
        "meta": {"n_nodes": 2, "n_edges": 1},
    }
    x, ei = hypergraph_to_tensors(hg)
    assert x.shape == (2, 2)
    assert 0 in ei  # iatc edge type


def test_tensor_cache_roundtrip(sample_hg, tmp_path):
    """Test save/load of pre-tensorized hypergraphs."""
    x, ei = hypergraph_to_tensors(sample_hg)
    graph_tensors = [(x, ei)]
    thread_ids = ["thread-633512"]

    cache_path = str(tmp_path / "cache.pt")
    save_tensor_cache(graph_tensors, thread_ids, cache_path)

    loaded_tensors, loaded_ids = load_tensor_cache(cache_path)
    assert len(loaded_tensors) == 1
    assert loaded_ids == thread_ids

    lx, lei = loaded_tensors[0]
    assert torch.equal(lx, x)
    for k in ei:
        assert torch.equal(lei[k], ei[k])


def test_train_with_tensor_cache(sample_hg, tmp_path):
    """Test that train() can save and reload a tensor cache."""
    import copy
    corpus = [copy.deepcopy(sample_hg) for _ in range(4)]
    cache_path = str(tmp_path / "tensors.pt")

    # First run: creates cache
    model1, emb1 = train(
        corpus, dim=16, hidden_dim=32, n_layers=1,
        epochs=2, batch_size=2, verbose=False,
        tensor_cache_path=cache_path)
    assert os.path.exists(cache_path)

    # Second run: loads from cache (empty hypergraphs list)
    model2, emb2 = train(
        [], dim=16, hidden_dim=32, n_layers=1,
        epochs=2, batch_size=2, verbose=False,
        tensor_cache_path=cache_path)
    assert emb2.shape == emb1.shape
