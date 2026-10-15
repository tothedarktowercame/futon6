"""The word "int" becomes an integral sign only where something is integrated.

pandoc-mathify.lua turns a bare "int" in mathematics into \\Integral, which is
right for "int_0^1 f" and wrong for "e^{int}", where i, n and t are a product
in an exponent (APM a95J03: "a(n)e^{int}" came out as a(n)e^{\\int}).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FILTER = Path(__file__).resolve().parent.parent / "scripts" / "pandoc-mathify.lua"

pytestmark = pytest.mark.skipif(
    shutil.which("pandoc") is None, reason="pandoc not installed"
)


def convert(markdown: str) -> str:
    return subprocess.run(
        ["pandoc", "--from=gfm", "--to=latex", f"--lua-filter={FILTER}"],
        input=markdown,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout


def test_exponent_int_is_a_product():
    out = convert("The curve is $\\gamma(t) = a(n)e^{int}$ here.\n")
    assert "Integral" not in out
    assert "e^{int}" in out


def test_int_with_limits_is_an_integral():
    out = convert("Then $\\int_C\\alpha = int_0^1 f(x)\\,dx$ holds.\n")
    assert "\\Integral_" in out


def test_int_before_integrand_is_an_integral():
    out = convert("We have $F = int f(x) dx$ by definition.\n")
    assert "\\Integral" in out


def test_int_inside_a_word_is_untouched():
    out = convert("Take $x \\in \\mathrm{interior}(A)$ and $print(y)$.\n")
    assert "Integral" not in out
