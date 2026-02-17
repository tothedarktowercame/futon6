"""Tests for LaTeX → s-expression parser (Stage 8)."""

import pytest
from futon6.latex_sexp import parse, parse_tree, Atom, App


# -- Thread #633512 expressions --

THREAD_633512_CASES = [
    (r"\Gamma=(V,E,s,t)", "(= Γ (tuple V E s t))"),
    (r"s(X(e))=X(s(e))", "(= (s (X e)) (X (s e)))"),
    (r"t(X(e))=X(t(e))", "(= (t (X e)) (X (t e)))"),
    (r"e : v \to w", "(: e (→ v w))"),
    (r"X(e) : X(v) \to X(w)", "(: (X e) (→ (X v) (X w)))"),
    (r"\mathcal{C}", "𝒞"),
    (r"\mathcal{A}", "𝒜"),
    (r"\mathcal{I}", "ℐ"),
    (r"X(v) \in \mathcal{C}", "(∈ (X v) 𝒞)"),
    (r"v \in V", "(∈ v V)"),
    (r"e \in E", "(∈ e E)"),
    (r"X \circ s = X \circ t", "(= (∘ X s) (∘ X t))"),
    (r"\gamma = \beta \circ e", "(= γ (∘ β e))"),
    (r"X(\gamma)", "(X γ)"),
    (r"s=t", "(= s t)"),
    (r"f:V\to W", "(: f (→ V W))"),
    (r"D:\mathcal{I}\rightarrow\mathcal{A}", "(: D (→ ℐ 𝒜))"),
    (r"\mathsf{Path}(\Gamma)", "(Path Γ)"),
    (r"\mathsf{Path}(\Gamma) \to \mathcal{C}", "(→ (Path Γ) 𝒞)"),
    (r"\Gamma \to U(\mathcal{C})", "(→ Γ (U 𝒞))"),
    (r"U(-)", "(U □)"),
    (r"\bar\Gamma", "(bar Γ)"),
]


@pytest.mark.parametrize("latex,expected", THREAD_633512_CASES,
                         ids=[c[0][:30] for c in THREAD_633512_CASES])
def test_thread_633512(latex, expected):
    assert parse(latex) == expected


def test_frac():
    assert parse(r"\frac{a}{b}") == "(/ a b)"


def test_subscript():
    assert parse(r"x_i") == "(sub x i)"


def test_superscript():
    assert parse(r"x^2") == "(sup x 2)"


def test_prime():
    assert parse(r"\gamma'") == "(prime γ)"


def test_tuple_in_parens():
    assert parse(r"(a, b, c)") == "(tuple a b c)"


def test_greek():
    assert parse(r"\alpha") == "α"
    assert parse(r"\Omega") == "Ω"


def test_definition():
    result = parse(r"X(\gamma):=\mathrm{id}_{X(v)}")
    assert result == "(:= (X γ) (sub id (X v)))"


def test_array_square():
    tex = (r"\begin{array}{c} \bullet & \rightarrow & \bullet"
           r" \\ \downarrow && \downarrow"
           r" \\ \bullet & \rightarrow & \bullet  \end{array}")
    result = parse(tex)
    assert "graph" in result
    assert "→" in result
    assert "•" in result


def test_array_double_arrows():
    tex = (r"\begin{array}{c} \bullet & \rightrightarrows & \bullet"
           r" \\ \downarrow && \downarrow"
           r" \\ \bullet & \rightrightarrows & \bullet  \end{array}")
    result = parse(tex)
    assert "graph" in result
    assert "⇉" in result


def test_empty():
    assert parse("") == '""'


def test_fallback():
    """Unparseable input should return the LaTeX in quotes, not crash."""
    result = parse(r"\undefinedcommand{lots}{of}{args}")
    assert isinstance(result, str)


def test_parse_tree_returns_ast():
    tree = parse_tree(r"a = b")
    assert isinstance(tree, App)
    assert tree.op == "="
    assert len(tree.args) == 2
