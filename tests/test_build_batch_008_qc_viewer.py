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


# ============================================================
# Symbol grounding wiring (Task 48)
# ============================================================

def _toy_kernel():
    """Mimic (singles, multi_index) shape returned by load_ner_kernel."""
    singles = {
        "abelian group": ("abelian group", "AbelianGroup"),  # ner_kernel does store multi-word in singles for direct lookup in the viewer's path
        "category": ("category", "Category"),
    }
    multi_index = {
        "abelian": [("abelian group", "abelian group", "AbelianGroup")],
    }
    return singles, multi_index


def test_kernel_phrase_lookup_handles_singles_and_multi_word():
    mod = load_module()
    singles, multi_index = _toy_kernel()
    lookup = mod._make_kernel_phrase_lookup(singles, multi_index)
    # exact match in singles
    assert lookup("category") == "Category"
    # multi-word resolved via multi_index
    assert lookup("abelian group") == "AbelianGroup"
    # unknown phrase
    assert lookup("frobnicator") is None
    # case + whitespace tolerance
    assert lookup("  ABELIAN GROUP  ") == "AbelianGroup"


def test_walk_atoms_yields_single_letters_inside_chars():
    mod = load_module()
    from futon6 import math_ast as ma
    nodes = ma.parse_math("XYZ", base_offset=0)
    atoms = list(mod._walk_atoms(nodes))
    # Each letter is its own atom
    texts = [a[0] for a in atoms]
    assert texts == ["X", "Y", "Z"]


def test_walk_atoms_yields_full_macro_text():
    mod = load_module()
    from futon6 import math_ast as ma
    nodes = ma.parse_math(r"\mathcal{C}", base_offset=0)
    atoms = list(mod._walk_atoms(nodes))
    # Macro itself emits one atom (full text); then recurses into args ('C')
    macro_atoms = [a for a in atoms if a[0].startswith("\\")]
    assert any(a[0] == r"\mathcal{C}" for a in macro_atoms)
    # And the interior 'C' is also yielded by recursion
    inner = [a for a in atoms if a[0] == "C"]
    assert inner


def test_detect_grounded_symbols_grounds_x_from_let_binding():
    mod = load_module()
    singles, multi_index = _toy_kernel()
    text = "Let $X$ be an abelian group. The group $X$ has identity element."
    records, env, summary = mod.detect_grounded_symbols(
        "test-entity", text, singles, multi_index,
    )
    # Every "X" inside $...$ after the declaration should ground to AbelianGroup.
    grounded_texts = [r["hx/content"]["match"] for r in records]
    assert "X" in grounded_texts
    # Canon attached
    canons = {r["hx/content"]["canon"] for r in records}
    assert "AbelianGroup" in canons
    # Strategy attribution present
    strats = {r["hx/content"]["strategy"] for r in records}
    assert "let-binding" in strats
    # Summary block populated
    assert summary["total_bindings_emitted"] >= 1
    assert summary["grounded_atom_count"] >= 1


def test_detect_grounded_symbols_records_have_math_grounded_symbol_type():
    mod = load_module()
    singles, multi_index = _toy_kernel()
    text = "Let $X$ be an abelian group. So $X$."
    records, _, _ = mod.detect_grounded_symbols("e", text, singles, multi_index)
    assert all(r["hx/type"] == "math/grounded-symbol" for r in records)
    assert all(r["hx/role"] == "scope" for r in records)


def test_detect_grounded_symbols_returns_empty_for_no_bindings():
    mod = load_module()
    singles, multi_index = _toy_kernel()
    text = "No declarations here. Just $Z$ floating around."
    records, env, summary = mod.detect_grounded_symbols("e", text, singles, multi_index)
    assert records == []
    assert summary["grounded_atom_count"] == 0


# ============================================================
# Tooltip + canon-label rendering on grounded-symbol marks
# ============================================================

def test_render_tree_node_emits_canon_label_and_tooltip_for_grounded_symbol():
    mod = load_module()
    from futon6 import structure_seed as ss
    record = {
        "hx/id": "t:g-0", "hx/role": "scope", "hx/type": "math/grounded-symbol",
        "hx/parent": None,
        "hx/content": {
            "match": "X", "position": 0, "end": 1,
            "canon": "AbelianGroup",
            "type_phrase": "abelian group",
            "strategy": "let-binding",
        },
        "hx/labels": ["scope", "math", "grounded"],
    }
    spans = ss.scope_records_to_spans([record])
    tree = ss.build_scope_tree(spans, [])
    node = tree["children"][0]
    out = mod.render_tree_node("X", node, is_root=False)
    # Tooltip surfaces canon + strategy + role + type phrase
    assert "AbelianGroup" in out
    assert "let-binding" in out
    assert "abelian group" in out
    assert "role=" in out
    # Badge text shows the canon name, not the verbose type
    assert ">AbelianGroup<" in out
    # CSS class still includes math-grounded-symbol + role-* so styling applies
    assert "math-grounded-symbol" in out
    assert "role-" in out


def test_render_tree_node_default_label_for_non_grounded_scope():
    mod = load_module()
    from futon6 import structure_seed as ss
    record = {
        "hx/id": "t:s-0", "hx/role": "scope", "hx/type": "env/proof",
        "hx/parent": None,
        "hx/content": {"position": 0, "end": 5, "match": "proof"},
        "hx/labels": ["scope"],
    }
    spans = ss.scope_records_to_spans([record])
    tree = ss.build_scope_tree(spans, [])
    node = tree["children"][0]
    out = mod.render_tree_node("proof", node, is_root=False)
    # Non-grounded scopes keep their original label
    assert ">env/proof<" in out
    # No title attribute on non-grounded scopes
    assert "title=" not in out


def test_detect_grounded_symbols_emits_newcommand_with_body_fallback_canon():
    """\\newcommand{\\RR}{\\mathbb R} -> ground \\RR in math envelopes.

    The canon may be a body-derived fallback (no kernel hit on "real
    numbers" in the toy kernel), but the binding still emits and the
    label-on-mark is non-empty.
    """
    mod = load_module()
    singles, multi_index = _toy_kernel()
    text = r"\newcommand{\RR}{{\mathbb R}}" + "\nLet $\\RR$ denote the reals."
    records, _, summary = mod.detect_grounded_symbols("e", text, singles, multi_index)
    nc_records = [r for r in records if r["hx/content"]["strategy"] == "newcommand"]
    assert nc_records, "expected at least one newcommand-grounded atom"
    rec = nc_records[0]
    assert rec["hx/content"]["match"] == r"\RR"
    # Canon is body-fallback "R" (toy kernel has no "real numbers" entry)
    # or whatever the kernel returns; either way it's truthy.
    assert rec["hx/content"]["canon"]
    # Role is enriched on the record
    assert "syntax_role" in rec["hx/content"]


def test_detect_grounded_symbols_role_enrichment_present():
    mod = load_module()
    singles, multi_index = _toy_kernel()
    text = "Let $X$ be an abelian group. The value $X$ is fixed."
    records, _, _ = mod.detect_grounded_symbols("e", text, singles, multi_index)
    assert records
    for r in records:
        assert "syntax_role" in r["hx/content"]
        assert r["hx/content"]["syntax_role"] in {
            "greek", "binop", "bridge", "relation", "comparison",
            "large-op", "arrow", "function", "delimiter", "named-op",
            "number", "variable",
        }


def test_detect_grounded_symbols_skips_uncanon_prose_strategy_bindings():
    """LetBindingStrategy w/ no kernel canon shouldn't pollute the records.

    Reason: the prose regex captures noisy phrasal residue when the
    kernel doesn't recognise the phrase; rendering those would emit
    spurious marks. NewcommandStrategy is the exception (its body
    fallback IS informative).
    """
    mod = load_module()
    singles, multi_index = _toy_kernel()
    # "frobnicator" is not in the toy kernel; let-binding fires but
    # canon stays None.
    text = "Let $W$ be a frobnicator. So $W$ is well-defined."
    records, _, _ = mod.detect_grounded_symbols("e", text, singles, multi_index)
    # Filter: only newcommand bindings should pass when canon is missing.
    let_w = [r for r in records if r["hx/content"]["match"] == "W"]
    assert not let_w, "let-binding without kernel canon shouldn't emit"
