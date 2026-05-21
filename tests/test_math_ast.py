"""Tests for the symbol-grounding Layer 2 AST parser (futon6.math_ast)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6 import math_ast as ma


# ---- find_math_envelopes ----

def test_find_inline_math():
    text = "before $X + Y$ after"
    envs = list(ma.find_math_envelopes(text))
    assert len(envs) == 1
    start, end, ints, inte, kind = envs[0]
    assert kind == "inline"
    assert text[ints:inte] == "X + Y"


def test_find_display_math_double_dollar():
    text = "before $$X + Y$$ after"
    envs = list(ma.find_math_envelopes(text))
    assert len(envs) == 1
    _, _, ints, inte, kind = envs[0]
    assert kind == "display"
    assert text[ints:inte] == "X + Y"


def test_find_paren_math():
    text = r"before \(X + Y\) after"
    envs = list(ma.find_math_envelopes(text))
    assert len(envs) == 1
    _, _, ints, inte, kind = envs[0]
    assert kind == "paren"
    assert text[ints:inte] == "X + Y"


def test_find_bracket_math():
    text = r"before \[X = Y\] after"
    envs = list(ma.find_math_envelopes(text))
    assert len(envs) == 1
    _, _, ints, inte, kind = envs[0]
    assert kind == "bracket"
    assert text[ints:inte] == "X = Y"


def test_find_equation_environment():
    text = (
        "Some text. "
        r"\begin{equation}"
        "F(x) = x^2"
        r"\end{equation}"
        " more text."
    )
    envs = list(ma.find_math_envelopes(text))
    assert len(envs) == 1
    _, _, ints, inte, kind = envs[0]
    assert kind == "environment"
    assert text[ints:inte] == "F(x) = x^2"


def test_does_not_match_escaped_dollar():
    text = r"The price is 95\$ (escaped). No math here."
    envs = list(ma.find_math_envelopes(text))
    assert envs == []


def test_inline_doesnt_span_newline():
    text = "$X\nY$"  # spurious dollars across newline
    envs = list(ma.find_math_envelopes(text))
    assert envs == []


# ---- parse_math ----

def test_parse_simple_macro_no_args():
    nodes = ma.parse_math(r"\to")
    assert len(nodes) == 1
    assert nodes[0].kind == "macro"
    assert nodes[0].name == "to"
    assert nodes[0].args == []


def test_parse_macro_with_two_args():
    nodes = ma.parse_math(r"\Hom{A}{B}")
    assert len(nodes) == 1
    n = nodes[0]
    assert n.kind == "macro"
    assert n.name == "Hom"
    assert len(n.args) == 2
    assert n.args[0]["interior"] == "A"
    assert n.args[1]["interior"] == "B"
    # Whole node text spans the entire macro call
    assert n.text == r"\Hom{A}{B}"
    # End position includes both braces
    assert n.end - n.start == len(r"\Hom{A}{B}")


def test_parse_nested_macro_in_argument():
    nodes = ma.parse_math(r"\Hom{\Hom{A}{B}}{C}")
    outer = nodes[0]
    assert outer.name == "Hom"
    assert len(outer.args) == 2
    inner = outer.args[0]["nodes"][0]
    assert inner.kind == "macro"
    assert inner.name == "Hom"
    assert len(inner.args) == 2


def test_parse_subscript_with_braces():
    nodes = ma.parse_math(r"X_{\mathrm{op}}")
    # X (chars), then sub
    sub = [n for n in nodes if n.kind == "sub"]
    assert sub
    assert sub[0].text == r"_{\mathrm{op}}"


def test_parse_superscript_single_char():
    nodes = ma.parse_math(r"x^2")
    sup = [n for n in nodes if n.kind == "sup"]
    assert sup
    assert sup[0].text == "^2"


def test_parse_bare_group_with_internal_macro():
    nodes = ma.parse_math(r"{X \to Y}")
    grp = [n for n in nodes if n.kind == "group"]
    assert grp
    inner = grp[0].args[0]["nodes"]
    inner_names = [(n.kind, n.name) for n in inner if n.kind == "macro"]
    assert ("macro", "to") in inner_names


def test_parse_positions_are_absolute():
    interior = r"\Hom{A}{B}"
    nodes = ma.parse_math(interior, base_offset=100)
    n = nodes[0]
    assert n.start == 100
    assert n.end == 100 + len(interior)
    assert n.args[0]["start"] == 100 + len(r"\Hom")
    assert n.args[0]["interior_start"] == 100 + len(r"\Hom{")


# ---- walk_math_ast ----

def test_walk_yields_nested_depth():
    nodes = ma.parse_math(r"\Hom{\Hom{A}{B}}{C}")
    visited = list(ma.walk_math_ast(nodes))
    depths = [d for _, d in visited]
    assert max(depths) >= 2  # outer macro at d=0, inner macro at d>=1
