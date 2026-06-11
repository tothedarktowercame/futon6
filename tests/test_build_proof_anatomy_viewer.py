import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build-proof-anatomy-viewer.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_proof_anatomy_viewer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_annotate_text_segments_nested_scopes_and_expressions():
    builder = load_builder()
    text = "Let x < y & z."
    scopes = [
        {
            "hx/type": "bind/let",
            "hx/content": {"position": 0, "end": len(text), "match": text},
        },
        {
            "hx/type": "constrain/relation",
            "hx/content": {"position": 6, "end": 11, "match": "< y &"},
        },
    ]
    expressions = [
        {
            "expr": "x < y",
            "position": 4,
            "type": "relation",
            "grade": "floating",
        }
    ]

    out = builder.annotate_text(text, scopes, expressions)

    assert "scope-label binder-let" in out
    assert "scope-label binder-constrain" in out
    assert "depth-2" in out
    assert "expr-type-relation" in out
    assert "expr-grade-floating" in out
    assert "&lt;" in out
    assert "&amp;" in out
    assert "< y &" not in out


def test_slug_for_writeup_is_problem_page_name():
    builder = load_builder()
    assert builder.slug_for_writeup("problem7-writeup.md") == "problem7"
    assert builder.slug_for_writeup("problem7-solution-full.tex") == "problem7-full"


def test_index_html_pairs_summary_and_full_registers():
    builder = load_builder()
    summary = [{
        "writeup": "problem1-writeup.md",
        "expr-count": 2,
        "scope-count": 3,
        "floating-expr-count": 1,
        "floating-expr-pct": 50.0,
        "free-symbols": ["x"],
        "vacuous-count": 0,
        "externally-bound-count": 0,
        "orphan-count": 0,
        "vacuous-scopes": [],
        "expressions": [{"type": "number", "grade": "weak"}],
        "scopes": [{"hx/type": "bind/let", "hx/content": {"position": 0, "end": 1}}],
    }]
    full = [{
        "writeup": "problem1-solution-full.tex",
        "expr-count": 5,
        "scope-count": 4,
        "floating-expr-count": 1,
        "floating-expr-pct": 20.0,
        "free-symbols": [],
        "vacuous-count": 1,
        "vacuous-scopes": [],
        "expressions": [{"type": "operator", "grade": "strict", "gold-types": ["operator"]}],
        "scopes": [{"hx/type": "section/body", "hx/content": {"position": 0, "end": 1}}],
        "gold-agreement-rate": 75.0,
        "gold-annotated-count": 4,
        "gold-agree-count": 3,
        "gold-disagreements": [],
    }]

    html = builder.index_html(summary, full, {"number", "operator"}, {"weak", "strict"}, {"bind/let", "section/body"}, "now")

    assert "Summary Expr" in html
    assert "Full Expr" in html
    assert 'href="problem1-full.html">5</a>' in html
    assert "75.0%" in html
