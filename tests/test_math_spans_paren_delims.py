"""math_spans reads LaTeX's \\(...\\) as the same inline scope as $...$.

The APM informal proofs write \\(...\\) throughout; reading only dollars, S1 found
12 clause units in a00J02 against 53 for the same text in dollars.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import anatomy_v0_sweep as sweep  # noqa: E402


def spans(text):
    return [(d, b) for _s, _e, d, b in sweep.math_spans(text)]


def test_paren_delimiters_are_inline_math():
    assert spans(r"Let \(f \in L^2\) and $g$ hold.") == [("\\(", "f \\in L^2"), ("$", "g")]


def test_offsets_cover_the_delimiters():
    text = r"a \(x\) b"
    (start, end, delim, body), = sweep.math_spans(text)
    assert text[start:end] == r"\(x\)" and text[start + len(delim):end - 2] == body


def test_line_break_then_parenthesis_is_not_math():
    assert spans(r"line\\(not math) and \(x\)") == [("\\(", "x")]


def test_comment_and_unclosed_are_skipped():
    assert spans("% \\(c\\)\n\\(y\\) then \\(open") == [("\\(", "y")]
