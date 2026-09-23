"""A fenced block that names its language must survive as code.

An untagged fence in these proofs is usually display mathematics, so
pandoc-mathify.lua sets it as math. A ```lean fence is not: it is source, and
setting it as math turns `theorem foo :` into \text{theorem} foo_{:} inside an
aligned environment. Before the classes guard, Lean blocks were rescued only by
accident -- most contain a "#check" or an "@[simp]" that trips the algorithm
heuristic -- so a block without one was silently mathified.
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


# No "#", no "@", no "for"/"if"/"return": nothing the heuristics would catch.
LEAN = """\
```lean
theorem foo (a b : ℕ) : a + b = b + a := by
  omega
```
"""


def test_tagged_fence_stays_code():
    out = convert(LEAN)
    assert "aligned" not in out
    assert "theorem foo" in out


def test_untagged_fence_still_becomes_math():
    out = convert("```\nx^2 + y^2 = z^2\n```\n")
    assert "\\(" in out or "\\[" in out
