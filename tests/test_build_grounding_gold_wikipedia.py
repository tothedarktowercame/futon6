"""Tests for build-grounding-gold-wikipedia.py helpers."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module():
    spec = spec_from_file_location(
        "build_grounding_gold_wikipedia",
        ROOT / "scripts" / "build-grounding-gold-wikipedia.py",
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_strip_wiki_markup_math_becomes_dollar():
    mod = _load_module()
    out = mod._strip_wiki_markup("Let <math>X</math> be a thing.")
    assert "$X$" in out
    assert "<math>" not in out


def test_strip_wiki_markup_plain_link_keeps_target_text():
    mod = _load_module()
    out = mod._strip_wiki_markup("See [[abelian group]] for more.")
    assert "abelian group" in out
    assert "[[" not in out


def test_strip_wiki_markup_pipe_link_keeps_display():
    mod = _load_module()
    out = mod._strip_wiki_markup("See [[Abelian group|groups]] for context.")
    assert "groups" in out
    assert "Abelian group" not in out


def test_strip_wiki_markup_strips_section_anchor():
    mod = _load_module()
    out = mod._strip_wiki_markup("See [[Group (mathematics)#Examples|examples]].")
    assert "examples" in out
    assert "#" not in out


def test_normalize_canon_singleword():
    mod = _load_module()
    assert mod._normalize_canon("group") == "Group"


def test_normalize_canon_multiword_camelcased():
    mod = _load_module()
    assert mod._normalize_canon("abelian group") == "AbelianGroup"


def test_normalize_canon_strips_section_anchor():
    mod = _load_module()
    assert mod._normalize_canon("Topological group#Examples") == "TopologicalGroup"


def test_extract_gold_strict_pattern():
    mod = _load_module()
    text = "<math>X</math> is an [[Abelian group]] of finite order."
    result = mod.extract_gold_from_page("Test", text)
    assert result is not None
    raw, gold = result
    assert any(g["symbol"] == "X" and g["canon"] == "AbelianGroup" for g in gold)
    assert "$X$" in raw
    assert "Abelian group" in raw


def test_extract_gold_let_fix_pattern():
    mod = _load_module()
    text = "Let <math>G</math> be a [[topological group]]. Then..."
    result = mod.extract_gold_from_page("Test", text)
    assert result is not None
    raw, gold = result
    assert any(g["symbol"] == "G" and g["canon"] == "TopologicalGroup" for g in gold)


def test_extract_gold_rejects_complex_lhs():
    """Multi-symbol LHS shouldn't end up as gold — the engine quotes
    those as constructor declarations, not single bindings."""
    mod = _load_module()
    text = "Let <math>X = (Y, Z)</math> be an [[ordered pair]]."
    result = mod.extract_gold_from_page("Test", text)
    if result is None:
        return
    _, gold = result
    assert not any(g["symbol"] == "X = (Y, Z)" for g in gold)


def test_extract_gold_returns_none_when_no_match():
    mod = _load_module()
    text = "Plain prose with no math or links."
    assert mod.extract_gold_from_page("Test", text) is None


def test_extract_gold_dedupes_same_pair_from_multiple_patterns():
    mod = _load_module()
    text = (
        "Let <math>X</math> be a [[group]]. "
        "<math>X</math> is a [[group]] of order 2."
    )
    result = mod.extract_gold_from_page("Test", text)
    assert result is not None
    _, gold = result
    seen = [(g["symbol"], g["canon"]) for g in gold]
    assert seen.count(("X", "Group")) == 1
