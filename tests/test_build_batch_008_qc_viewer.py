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


# ============================================================
# Tree-aware scope renderer tests
# ============================================================

def _scope(start, end, label):
    return {"start": start, "end": end, "label": label}


def test_build_scope_tree_nests_inner_under_outer():
    mod = load_module()
    spans = [
        _scope(0, 100, "env/proof"),
        _scope(20, 60, "bind/typed"),
    ]
    tree = mod.build_scope_tree(spans, [])
    assert tree["label"] == "$root"
    assert len(tree["children"]) == 1
    outer = tree["children"][0]
    assert outer["label"] == "env/proof"
    assert outer["depth"] == 1
    assert len(outer["children"]) == 1
    inner = outer["children"][0]
    assert inner["label"] == "bind/typed"
    assert inner["depth"] == 2


def test_build_scope_tree_keeps_disjoint_scopes_as_siblings():
    mod = load_module()
    spans = [
        _scope(0, 40, "bind/let"),
        _scope(50, 90, "constrain/relation"),
    ]
    tree = mod.build_scope_tree(spans, [])
    assert len(tree["children"]) == 2
    assert {c["label"] for c in tree["children"]} == {"bind/let", "constrain/relation"}


def test_build_scope_tree_drops_straddling_scope():
    mod = load_module()
    # Second scope straddles the first's right edge — can't be tree-arranged.
    spans = [
        _scope(0, 50, "env/proof"),
        _scope(40, 90, "constrain/relation"),
    ]
    tree = mod.build_scope_tree(spans, [])
    assert len(tree["children"]) == 1
    assert tree["children"][0]["label"] == "env/proof"


def test_term_placed_at_deepest_containing_scope():
    mod = load_module()
    spans = [
        _scope(0, 100, "env/proof"),
        _scope(20, 60, "bind/typed"),
    ]
    # Term at [30, 35] is inside both; should land in inner (bind/typed).
    terms = [(30, 35, "monad", "Monad")]
    tree = mod.build_scope_tree(spans, terms)
    outer = tree["children"][0]
    inner = outer["children"][0]
    assert inner["terms"] == [(30, 35, "monad", "Monad")]
    assert outer["terms"] == []


def test_term_at_root_when_no_scope_contains_it():
    mod = load_module()
    spans = [_scope(0, 50, "env/proof")]
    terms = [(80, 85, "functor", "Functor")]
    tree = mod.build_scope_tree(spans, terms)
    assert tree["terms"] == [(80, 85, "functor", "Functor")]


def test_term_straddling_scope_is_dropped():
    mod = load_module()
    spans = [_scope(0, 50, "env/proof")]
    terms = [(45, 60, "monad", "Monad")]  # crosses right edge
    tree = mod.build_scope_tree(spans, terms)
    # Term doesn't fit anywhere — not at root (it overlaps a scope), not in scope.
    assert tree["terms"] == []
    assert tree["children"][0]["terms"] == []


def test_render_tree_node_produces_nested_marks():
    mod = load_module()
    text = "AAAA proof body BBBB"
    spans = [
        _scope(0, len(text), "env/proof"),
        _scope(5, 15, "bind/typed"),
    ]
    tree = mod.build_scope_tree(spans, [])
    html_out = mod.render_tree_node(text, tree, is_root=True)
    # Outer mark should contain inner mark (nested).
    outer_pos = html_out.find("env/proof")
    inner_pos = html_out.find("bind/typed")
    end_inner = html_out.find("</mark>", inner_pos)
    end_outer = html_out.find("</mark>", end_inner + 1)
    assert outer_pos < inner_pos < end_inner < end_outer


def test_classify_kernel_terms_reports_depth_distribution():
    mod = load_module()
    # Text with whitespace tokenization so spot_terms_entity can find "monad".
    # Place "monad" inside the inner (nested) scope at depth 2.
    prefix = "a " * 30  # 60 chars
    middle = "monad "  # 6 chars; "monad" sits at offset 60-65
    suffix = "b " * 100  # 200 chars
    text = prefix + middle + suffix
    proof_end = len(text)
    scopes = [
        {"hx/type": "env/proof", "hx/content": {"position": 0, "end": proof_end, "match": text}},
        {"hx/type": "bind/typed", "hx/content": {"position": 50, "end": 100, "match": text[50:100]}},
    ]
    singles = {"monad": ("monad", "Monad")}
    multi_index: dict = {}
    stats = mod.classify_kernel_terms(text, scopes, singles, multi_index)
    assert stats["total"] == 1
    assert stats["inhabited"] == 1
    assert stats["outer"] == 0
    assert stats["straddled"] == 0
    # depth=2 since the term lands inside the inner (nested) scope.
    assert stats["depth_distribution"] == {2: 1}
