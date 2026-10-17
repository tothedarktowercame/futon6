"""S11 reads a line holding several formulae as each of them."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sfc_struct_canon as s  # noqa: E402


def test_side_by_side_formulae_are_split():
    assert s.split_formula(r"P_{M}^{2}=P_{M},\qquad P_{M}^{\ast{}}=P_{M},") == [r"P_{M}^{2}=P_{M}", r"P_{M}^{\ast{}}=P_{M}"]
    assert s.split_formula(r"e_{1} = a,\ e_{2} = c") == ["e_{1} = a", "e_{2} = c"]


def test_boxed_tag_and_text_are_not_formulae():
    assert s.split_formula(r"\boxed{H_{0}(M) = \mathbb Z,\quad H_{2}(M) = 0}") == [r"H_{0}(M) = \mathbb Z", "H_{2}(M) = 0"]
    assert s.split_formula(r"a(u_{n},v)=\ell(v) \tag{\ensuremath{\dagger}}") == [r"a(u_{n},v)=\ell(v)"]


def test_bracket_colon_notation_becomes_a_two_place_operator():
    assert s.split_formula("n = [G:H]") == [r"n = \operatorname{bracket}(G,H)"]
    assert s.split_formula(r"f(x) = [0,1]") == ["f(x) = [0,1]"]          # an interval is left alone


def test_commas_inside_brackets_do_not_split():
    assert s.split_formula(r"f(x,y) = g(x, y),\quad x \in A") == ["f(x,y) = g(x, y)", r"x \in A"]
