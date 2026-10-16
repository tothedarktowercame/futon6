"""Code spans that hold mathematics become mathematics; Lean and shell stay code.

The APM informal proofs put formulae in backticks: `\\sigma_a = \\rho(a)`, `f`,
`S₅`, `f_n ∈ K`. They were set as \\texttt with literal backslashes (2,886
formulae and 4,663 single symbols over 501 papers, 2026-10-16), and merged
formulae swallowed sentence punctuation and prose brackets.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

FILTER = Path(__file__).resolve().parent.parent / "scripts" / "pandoc-mathify.lua"

pytestmark = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")


def convert(markdown: str) -> str:
    return subprocess.run(
        ["pandoc", "--from=gfm-superscript-subscript", "--to=latex", "--wrap=preserve",
         f"--lua-filter={FILTER}"],
        input=markdown, capture_output=True, text=True, check=True, timeout=60).stdout


@pytest.mark.parametrize("span, expect", [
    (r"\sigma_a = \rho(a)", r"\(\sigma_{a} = \rho(a)\)"),
    ("f", r"\(f\)"),
    ("C,D", r"\(C,D\)"),
    ("ξ", r"\(\xi\)"),
    ("f_n ∈ K", r"\(f_{n}"),
    ("X × Y", r"\(X \times Y\)"),
    ("10V", r"V\)"),
])
def test_math_in_backticks_is_mathematics(span, expect):
    out = convert(f"we have `{span}` here\n")
    assert expect in out and "texttt" not in out


@pytest.mark.parametrize("span", ["Real.sq_pi", "norm_num [Real.pi_pos]", "lake build",
                                  'rg -n "indicator" Mathlib/MeasureTheory', "Main.lean",
                                  "MeasureTheory.Memℒp f 2 μ", "sorry"])
def test_lean_and_shell_stay_code(span):
    assert "texttt" in convert(f"we ran `{span}` here\n")


def test_merged_formulae_leave_sentence_punctuation_outside():
    out = convert(r"spans `B = Q\,D\,Q^*`, `|G| = |G|/2`." + "\n")
    assert r"Q^{\mDualStar}\), \(" in out and out.rstrip().endswith(r"\).")


def test_a_bracket_the_formula_never_opened_goes_back_to_the_sentence():
    out = convert("For |t| ≥ δ (with 0 < δ < 1), (1-t²)^{n/2} ≤ x; also (exponent p = 2). "
                  "The interval $(0,1]$ stays.\n")
    spans = re.findall(r"\\\((.*?)\\\)", out)
    assert spans
    for s in spans:
        depth = sum(s.count(c) for c in "([") - sum(s.count(c) for c in ")]")
        assert depth == 0, s
    assert r"\(({\mNumber{0}},{\mNumber{1}}]\)" in out


def test_latex_authored_formula_keeps_in_as_a_product():
    # "$1/(2\pi in^{2})$" (APM a95A07) became 1/(2\pi \in^{2}); plain-text "x in A" is still \in.
    out = convert(r"the residue is $1/(2\pi in^{2})$ and plain x in A." + "\n")
    assert r"\pi in^{" in out and r"\in^" not in out
    assert r"\(x \in A\)" in out


def test_merging_keeps_a_command_apart_from_a_following_letter():
    # "$\neg$UC" merged as the undefined command \negUC (APM a93A01).
    out = convert(r"Assume $\neg$UC: $\exists\varepsilon > 0$: done." + "\n")
    assert r"\negUC" not in out and r"\neg UC" in out
