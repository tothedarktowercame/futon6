"""Definition snippets hold whole formulae, and S11 reads formulae whole.

Fixed-width windows started inside a formula, so pairing their dollars took
the prose between formulae for formulae (apm-501-xi-r1: 157 extracted, 0 parsed).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import anatomy_v0_sweep as sweep  # noqa: E402
import def_formulae_extract as dfe  # noqa: E402
import warp_def_snippets as wds  # noqa: E402

TEXT = (r"We have $\|f\|_p \leq \|f\|_\infty$. $\blacksquare$ Now an \emph{upper bound} "
        r"means $\|f\|_p \leq \|f\|_q \cdot \mu(X)^{1/p-1/q}$ for finite measure, and \[ g = h \circ k \] holds.")


def test_math_extents_cover_inline_and_display():
    bodies = [b.strip() for _s, _e, b in sweep.math_extents(TEXT)]
    assert r"\|f\|_p \leq \|f\|_\infty" in bodies and r"g = h \circ k" in bodies


def test_a_window_is_widened_to_whole_formulae():
    at = TEXT.index("upper bound")
    cut = wds.window(TEXT, at, at + 11, pad=40)                  # a fixed cut lands inside two formulae
    assert cut.startswith(r"\|_\infty$") and cut.endswith(r"\mu(")
    snip = wds.window(TEXT, at, at + 11, pad=40, extents=sweep.math_extents(TEXT))
    assert snip.startswith(r"$\|f\|_p \leq \|f\|_\infty$")
    assert snip.endswith(r"\mu(X)^{1/p-1/q}$")


def test_extractor_takes_the_papers_whole_formulae_for_an_old_cut_snippet():
    at = TEXT.index("upper bound")
    old_snippet = wds.window(TEXT, at, at + 11, pad=40)         # as written before the fix
    with patch.object(dfe.dp, "read_text", return_value=TEXT):
        near = dfe.Papers().formulae_near("p1", old_snippet)
    assert near is not None
    assert any(r"\mu(X)^{1/p-1/q}" in f for f in near)
    assert all("upper bound" not in f for f in near)             # no prose taken for a formula
