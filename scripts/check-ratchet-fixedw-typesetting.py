#!/usr/bin/env python3
"""Ratchet checks for prose-to-math normalization and syntax-class rendering."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


FIXED_W_SNIPPET = (
    "By Section 3a below, the integrals over V (for fixed W = W_0) generate "
    "the full fractional ideal I = L(s, Pi x pi) * C[q_F^s, q_F^{-s}].\n"
)
W_PI_PSI_SNIPPET = "In the psi-Whittaker model W(pi, psi), choose V in W(pi,psi).\n"
SYNTAX_CLASS_SNIPPET = (
    "hence P in R.\n"
    "equal to c * qF^{-ks}.\n"
    "r in GLn+1 x GLn.\n"
)
DIAG_SNIPPET = "    I(s, W, V) = int_{N_n\\GL_n(F)} W(diag(g,1)) V(g) |det g|^{s-1/2} dg\n"


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def render_snippet(repo: Path, normalizer: Path, lua_filter: Path, snippet: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        md = tmp / "snippet.md"
        tex = tmp / "snippet.tex"
        md.write_text(snippet, encoding="utf-8")

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
        normalized = md.read_text(encoding="utf-8")
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
        rendered = tex.read_text(encoding="utf-8")
    return normalized, rendered


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    normalizer = repo / "scripts" / "normalize-math-prose.py"
    lua_filter = repo / "scripts" / "pandoc-mathify.lua"
    style_file = repo / "data" / "first-proof" / "latex" / "math-proofread-style.sty"

    failures: list[str] = []

    _, out = render_snippet(repo, normalizer, lua_filter, FIXED_W_SNIPPET)
    normalized_w, out_w = render_snippet(repo, normalizer, lua_filter, W_PI_PSI_SNIPPET)
    normalized_syntax, out_syntax = render_snippet(
        repo, normalizer, lua_filter, SYNTAX_CLASS_SNIPPET
    )
    _, out_diag = render_snippet(repo, normalizer, lua_filter, DIAG_SNIPPET)

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

    if "$W(\\pi, \\psi)$" not in normalized_w:
        failures.append("normalizer did not wrap W(pi, psi) as $W(\\pi, \\psi)$")
    if "$\\psi$-Whittaker" not in normalized_w:
        failures.append("normalizer did not wrap psi-Whittaker as $\\psi$-Whittaker")
    if r"\(W(\pi, \psi)\)" not in out_w:
        failures.append("rendered output missing inline math for W(\\pi, \\psi)")
    if "W(pi, psi)" in out_w or "W(pi,psi)" in out_w:
        failures.append("rendered output still contains raw W(pi, psi)")

    if r"$P \in R$" not in normalized_syntax:
        failures.append("normalizer did not convert 'P in R' to '$P \\in R$'")
    if r"$c \ast q_F^{-ks}$" not in normalized_syntax:
        failures.append("normalizer did not convert 'c * qF^{-ks}' to '$c \\ast q_F^{-ks}$'")
    if r"$r \in \mathup{GL}_{n+1} \times \mathup{GL}_{n}$" not in normalized_syntax:
        failures.append("normalizer did not convert GLn+1 x GLn into \\mathup{GL} product math")

    if r"\(P \in R\)" not in out_syntax:
        failures.append("rendered output missing inline math for 'P \\in R'")
    if r"\(c \ast q_{F}^{-ks}\)" not in out_syntax and r"\(c \ast q_F^{-ks}\)" not in out_syntax:
        failures.append("rendered output missing inline math for 'c \\ast q_F^{-ks}'")
    if r"\operatorname{diag}(g,1)" not in out_diag:
        failures.append("diag(g,1) was not rendered as \\operatorname{diag}(g,1)")
    gl_variants = [
        r"\(r \in \mathup{GL}_{n+1} \times \mathup{GL}_{n}\)",
        r"\(r \in \mathup{GL}_{n + 1} \times \mathup{GL}_{n}\)",
    ]
    if not any(v in out_syntax for v in gl_variants):
        failures.append("rendered output missing 'r \\in \\mathup{GL}_{n+1} \\times \\mathup{GL}_{n}'")

    style = style_file.read_text(encoding="utf-8")
    if r"\let\MP@orig@ast\ast" not in style:
        failures.append("math-proofread style does not preserve original \\ast")
    if r"\renewcommand{\ast}{\mOperator{\MP@orig@ast}}" not in style:
        failures.append("math-proofread style does not colorize \\ast as operator")
    if r"\renewcommand{\pi}{\mGreek{\MP@orig@pi}}" not in style:
        failures.append("math-proofread style does not colorize \\pi as Greek")
    if r"\renewcommand{\pi}{\mNumber{\MP@orig@pi}}" in style:
        failures.append("math-proofread style still categorizes \\pi as Number")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1

    print("Ratchet check passed: fixed-W, W(pi,psi), and syntax-class normalization are color-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
