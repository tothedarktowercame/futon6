import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build-batch-008-qc-viewer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_batch_008_qc_viewer_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_raw_and_entity_id_round_trip():
    mod = load_module()
    raw_id = "0710.3853v1"
    entity_id = mod.raw_to_entity_id(raw_id)
    assert entity_id == "arxiv-0710.3853v1"
    assert mod.entity_to_raw_id(entity_id) == raw_id


def test_pick_default_papers_prefers_scopeful_math_ct_rows():
    mod = load_module()
    batch_meta = {
        "p1": {"id": "p1", "title": "Paper 1", "categories": ["math.CT"]},
        "p2": {"id": "p2", "title": "Paper 2", "categories": ["math.CT"]},
        "p3": {"id": "p3", "title": "Paper 3", "categories": ["math.CT"]},
    }
    results = {
        "scopes": {
            "arxiv-p1": {"count": 0},
            "arxiv-p2": {"count": 4},
            "arxiv-p3": {"count": 2},
        },
        "ner_terms": {
            "arxiv-p1": {"count": 5},
            "arxiv-p2": {"count": 3},
            "arxiv-p3": {"count": 20},
        },
        "paper_hypergraphs": {
            "arxiv-p1": {"edges": [1] * 40, "nodes": [1] * 40},
            "arxiv-p2": {"edges": [1] * 25, "nodes": [1] * 50},
            "arxiv-p3": {"edges": [1] * 10, "nodes": [1] * 20},
        },
        "reverse_morphogenesis": {},
    }
    chosen = mod.pick_default_papers(batch_meta, results, paper_count=2)
    assert chosen == ["p2", "p3"]


def test_render_scope_markup_inserts_scope_label():
    mod = load_module()
    text = "Let f : X -> Y be a morphism."
    scopes = [{
        "hx/type": "bind/let",
        "hx/content": {"position": 0, "end": 14},
    }]
    markup = mod.render_scope_markup(text, scopes)
    assert "scope-label" in markup
    assert "bind/let" in markup
    assert "Let f : X -&gt;" in markup
