import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mark3_embed", ROOT / "scripts" / "mark3_embed.py")
mark3_embed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mark3_embed
SPEC.loader.exec_module(mark3_embed)


def item(item_id, label):
    return mark3_embed.EmbedItem(
        item_id=item_id,
        item_type="concept",
        msc="ct",
        label=label,
        text=f"{label} category theory functor morphism structure",
    )


def test_hard_negative_mining_excludes_synonym_cluster_and_prefers_close_terms():
    items = [
        item("concept:monoidal-category", "monoidal category"),
        item("concept:monoidal-categories", "monoidal categories"),
        item("concept:monoidal-functor", "monoidal functor"),
        item("concept:abelian-category", "abelian category"),
    ]
    clusters = [["concept:monoidal-category", "concept:monoidal-categories"]]

    triples = mark3_embed.mine_hard_negatives(items, clusters, negatives_per_pair=2)

    assert triples
    assert all(t.negative_id not in clusters[0] for t in triples)
    assert {t.anchor_id for t in triples} == set(clusters[0])
    # "monoidal functor" shares the discriminating token and should be surfaced
    # before the unrelated abelian category negative for at least one anchor.
    assert any(t.negative_id == "concept:monoidal-functor" for t in triples)


def test_batching_shards_and_hash_embeddings_are_deterministic():
    items = [item(f"concept:x-{idx}", f"x {idx}") for idx in range(5)]
    batches = list(mark3_embed.batched(items, 2))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert mark3_embed.shard_ranges(5, 2) == [(0, 3), (3, 5)]

    emb1 = mark3_embed.stable_hash_embeddings(["monoidal category", "abelian category"], 16)
    emb2 = mark3_embed.stable_hash_embeddings(["monoidal category", "abelian category"], 16)
    np.testing.assert_allclose(emb1, emb2)
    np.testing.assert_allclose(np.linalg.norm(emb1, axis=1), np.ones(2), rtol=1e-6)
