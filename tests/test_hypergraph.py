"""Tests for thread hypergraph assembler (Stage 9a)."""

import json
import os
import pytest
from futon6.hypergraph import assemble


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "first-proof")


@pytest.fixture
def thread_633512():
    raw_path = os.path.join(FIXTURES_DIR, "thread-633512-raw.json")
    wiring_path = os.path.join(FIXTURES_DIR, "thread-633512-wiring.json")
    if not os.path.exists(raw_path):
        pytest.skip("thread-633512-raw.json not found")
    with open(raw_path) as f:
        raw = json.load(f)
    with open(wiring_path) as f:
        wiring = json.load(f)
    return raw, wiring


def test_assemble_returns_valid_structure(thread_633512):
    raw, wiring = thread_633512
    hg = assemble(raw, wiring)

    assert "thread_id" in hg
    assert "nodes" in hg
    assert "edges" in hg
    assert "meta" in hg
    assert hg["thread_id"] == 633512


def test_node_counts(thread_633512):
    raw, wiring = thread_633512
    hg = assemble(raw, wiring)
    meta = hg["meta"]

    # 1 question + 1 answer + 6 q-comments + 8 a-comments = 16 posts
    assert meta["n_posts"] == 16
    # 19 unique NER canonical terms across question + answer nodes
    assert meta["n_terms"] == 19
    # 4 scope bindings (2 let, 1 forall, 1 where)
    assert meta["n_scopes"] == 4
    # Expressions should be > 20 (question + answer have many math expressions)
    assert meta["n_expressions"] > 20


def test_edge_types(thread_633512):
    raw, wiring = thread_633512
    hg = assemble(raw, wiring)
    edge_types = hg["meta"]["edge_types"]

    assert edge_types["iatc"] == 15  # 15 edges in WIRING
    assert edge_types["mention"] > 20  # many NER mentions
    assert edge_types["scope"] == 4
    assert edge_types["categorical"] == 3
    assert edge_types["surface"] > 20


def test_all_nodes_have_required_fields(thread_633512):
    raw, wiring = thread_633512
    hg = assemble(raw, wiring)

    for n in hg["nodes"]:
        assert "id" in n
        assert "type" in n
        assert "subtype" in n
        assert "attrs" in n
        assert n["type"] in ("post", "term", "expression", "scope")


def test_all_edges_have_required_fields(thread_633512):
    raw, wiring = thread_633512
    hg = assemble(raw, wiring)

    for e in hg["edges"]:
        assert "type" in e
        assert "ends" in e
        assert "attrs" in e
        assert isinstance(e["ends"], list)
        assert len(e["ends"]) >= 1


def test_expression_nodes_have_sexp(thread_633512):
    raw, wiring = thread_633512
    hg = assemble(raw, wiring)

    expr_nodes = [n for n in hg["nodes"] if n["type"] == "expression"]
    for n in expr_nodes:
        assert "latex" in n["attrs"]
        assert "sexp" in n["attrs"]
        assert n["attrs"]["sexp"]  # not empty


def test_json_serializable(thread_633512):
    raw, wiring = thread_633512
    hg = assemble(raw, wiring)

    # Should not raise
    result = json.dumps(hg, ensure_ascii=False)
    assert len(result) > 1000


def test_minimal_thread():
    """Test with a minimal synthetic thread."""
    raw = {
        "question": {
            "id": 1,
            "title": "Test",
            "score": 1,
            "tags": ["test"],
            "body_html": "<p>What is $x = 1$?</p>",
        },
        "answers": [],
        "comments_q": [],
        "comments_a": {},
    }
    wiring = {
        "nodes": [
            {
                "id": "q-1",
                "type": "question",
                "ner_terms": [{"term": "test", "canon": "Test"}],
                "discourse": [],
                "categorical": [],
            }
        ],
        "edges": [],
    }

    hg = assemble(raw, wiring)
    assert hg["thread_id"] == 1
    assert hg["meta"]["n_posts"] == 1
    assert hg["meta"]["n_terms"] == 1
    assert hg["meta"]["n_expressions"] == 1

    # Check the expression parsed
    expr = [n for n in hg["nodes"] if n["type"] == "expression"][0]
    assert expr["attrs"]["sexp"] == "(= x 1)"
