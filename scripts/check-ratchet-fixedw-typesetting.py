#!/usr/bin/env python3
"""Ratchet check for fixed-W fractional-ideal sentence rendering."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


SNIPPET = (
    "By Section 3a below, the integrals over V (for fixed W = W_0) generate "
    "the full fractional ideal I = L(s, Pi x pi) * C[q_F^s, q_F^{-s}].\n"
)


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    normalizer = repo / "scripts" / "normalize-math-prose.py"
    lua_filter = repo / "scripts" / "pandoc-mathify.lua"
    style_file = repo / "data" / "first-proof" / "latex" / "math-proofread-style.sty"

    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        md = tmp / "snippet.md"
        tex = tmp / "snippet.tex"
        md.write_text(SNIPPET, encoding="utf-8")

        run(
            [
                "python3",
                str(normalizer),
                "--write",
                "--allow-in-place",
                str(md),
            ],
            cwd=repo,
        )
        run(
            [
                "pandoc",
                str(md),
                "-f",
                "gfm-superscript-subscript",
                "-t",
                "latex",
                "--wrap=preserve",
                "--lua-filter",
                str(lua_filter),
                "-o",
                str(tex),
            ],
            cwd=repo,
        )
        out = tex.read_text(encoding="utf-8")

    if r"integrals over \(V\)" not in out:
        failures.append("missing inline math for V in 'integrals over V'")

    if r"(for fixed \(W = W_0\))" not in out and r"(for fixed \(W = W_{0}\))" not in out:
        failures.append("missing inline math for 'W = W_0'")

    target_eq_variants = [
        r"\(I = L(s, \Pi \times \pi)\,\ast\,\mathbb{C}[q_{F}^{s}, q_{F}^{-s}]\)",
        r"\(I = L(s, \Pi \times \pi)\,\\ast\,\mathbb{C}[q_{F}^{s}, q_{F}^{-s}]\)",
        r"\(I = L(s, \Pi \times \pi)\,\ast\,\mathbb{C}[q_F^s, q_F^{-s}]\)",
        r"\(I = L(s, \Pi \times \pi)\,\\ast\,\mathbb{C}[q_F^s, q_F^{-s}]\)",
    ]
    if not any(v in out for v in target_eq_variants):
        failures.append("missing fully normalized ideal equation with \\times, \\ast, and \\mathbb{C}")

    if r"\(\Pi\) x pi" in out:
        failures.append("found broken '\\(\\Pi\\) x pi' pattern")

    style = style_file.read_text(encoding="utf-8")
    if r"\let\MP@orig@ast\ast" not in style:
        failures.append("math-proofread style does not preserve original \\ast")
    if r"\renewcommand{\ast}{\mOperator{\MP@orig@ast}}" not in style:
        failures.append("math-proofread style does not colorize \\ast as operator")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1

    print("Ratchet check passed: fixed-W sentence is normalized and color-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
