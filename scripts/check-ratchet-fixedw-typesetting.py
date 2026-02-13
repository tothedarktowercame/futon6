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
NON_COLOR_SNIPPET = (
    "realized in W(Pi, psi^{-1}); map normalizes psi; "
    "|det g_0|^{1/2-s}; W_0(diag(g,1) u_Q); (j != i); 3 x 4 matrices.\n"
)
MAP_CLASS_SNIPPET = (
    "Consider the map Phi: K(Pi)|_{GL_n} → (fractional ideals of R) defined by "
    "Phi(phi) = { I(s, phi, V) : V in W(pi, psi) } · R. "
    "By the JPSS theory, ∪_{phi} Phi(phi) generates L(s, Pi x pi) · R.\n"
)
PHI_ACTION_SNIPPET = "so Phi(R(g_0)phi) and Phi(phi) generate the same ideal.\n"
HYBRID_SUFFIX_SNIPPET = (
    "Phi is GL_n-equivariant and GL_n-translates span K(Pi)|_{GL_n}. "
    "Work in the psi^{-1}-Whittaker model.\n"
)
PRIME_AND_SINGLE_SNIPPET = (
    "for any nonzero phi' in K(Pi)|_{GL_n}. "
    "for each V, there exists I such that I(s, W, V) = c * qF^{-ks}.\n"
)


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
    normalized_noncolor, out_noncolor = render_snippet(
        repo, normalizer, lua_filter, NON_COLOR_SNIPPET
    )
    normalized_map, out_map = render_snippet(repo, normalizer, lua_filter, MAP_CLASS_SNIPPET)
    normalized_phi_action, out_phi_action = render_snippet(
        repo, normalizer, lua_filter, PHI_ACTION_SNIPPET
    )
    normalized_hybrid_suffix, out_hybrid_suffix = render_snippet(
        repo, normalizer, lua_filter, HYBRID_SUFFIX_SNIPPET
    )
    normalized_prime_single, out_prime_single = render_snippet(
        repo, normalizer, lua_filter, PRIME_AND_SINGLE_SNIPPET
    )

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

    if "$W(\\Pi, \\psi^{-1})$" not in normalized_noncolor:
        failures.append("normalizer did not convert W(Pi, psi^{-1}) into math")
    if "normalizes $\\psi$" not in normalized_noncolor:
        failures.append("normalizer did not convert bare psi to math in 'normalizes psi'")
    if "$|\\det g_0|^{1/2-s}$" not in normalized_noncolor:
        failures.append("normalizer did not convert |det g_0|^{1/2-s} into math")
    if "$W_0(\\operatorname{diag}(g,1)u_Q)$" not in normalized_noncolor:
        failures.append("normalizer did not convert W_0(diag(g,1) u_Q) into math")
    if "$(j \\neq i)$" not in normalized_noncolor:
        failures.append("normalizer did not convert (j != i) into math")
    if "$3 \\times 4$ matrices" not in normalized_noncolor:
        failures.append("normalizer did not convert numeric dimension '3 x 4' into math")

    if r"\(W(\Pi, \psi^{-1})\)" not in out_noncolor:
        failures.append("rendered output missing inline math W(\\Pi, \\psi^{-1})")
    if r"normalizes \(\psi\)" not in out_noncolor:
        failures.append("rendered output missing inline math for psi in 'normalizes psi'")
    if r"\(|\det g_0|^{1/2-s}\)" not in out_noncolor:
        failures.append("rendered output missing inline math for |det g_0|^{1/2-s}")
    if r"\(W_0(\operatorname{diag}(g,1)u_{Q})\)" not in out_noncolor and r"\(W_0(\operatorname{diag}(g,1)u_Q)\)" not in out_noncolor:
        failures.append("rendered output missing inline math W_0(\\operatorname{diag}(g,1)u_Q)")
    if r"\((j \neq i)\)" not in out_noncolor:
        failures.append("rendered output missing inline math for (j \\neq i)")
    if r"\(3 \times 4\) matrices" not in out_noncolor:
        failures.append("rendered output missing inline math for 3 \\times 4")

    if r"$\Phi: K(\Pi)|_{\mathup{GL}_{n}} \to (\text{fractional ideals of } R)$" not in normalized_map:
        failures.append("normalizer did not convert map signature to inline math")
    if r"$\Phi(\phi) = \{ I(s, \phi, V) : V \in W(\pi, \psi) \} \cdot R$" not in normalized_map:
        failures.append("normalizer did not convert map set-expression to inline math")
    if r"$\bigcup_{\phi}$" not in normalized_map:
        failures.append("normalizer did not convert ∪_{phi} into \\bigcup_{\\phi}")

    if r"\(\Phi: K(\Pi)|_{\mathup{GL}_{n}} \to (\text{fractional ideals of } R)\)" not in out_map:
        failures.append("rendered output missing inline math map signature")
    if r"\(\Phi(\phi) = \{ I(s, \phi, V) : V \in W(\pi, \psi) \} \cdot R\)" not in out_map:
        failures.append("rendered output missing inline math map set-expression")
    if r"\(\bigcup_{\phi}\)" not in out_map:
        failures.append("rendered output missing inline math for \\bigcup_{\\phi}")

    if r"$\Phi(R(g_0)\,\phi)$" not in normalized_phi_action:
        failures.append("normalizer did not convert Phi(R(g_0)phi) into stable inline math")
    if r"\(\Phi(R(g_0)\,\phi)\)" not in out_phi_action:
        failures.append("rendered output missing inline math for Phi(R(g_0)\\,\\phi)")
    if r"\backslash \phi" in out_phi_action:
        failures.append("rendered output regressed to '\\backslash \\phi' for Phi(R(g_0)phi)")

    if r"$\mathup{GL}_{n}$-equivariant" not in normalized_hybrid_suffix:
        failures.append("normalizer did not split GL_n-equivariant as math-prefix plus prose suffix")
    if r"$\mathup{GL}_{n}$-translates" not in normalized_hybrid_suffix:
        failures.append("normalizer did not split GL_n-translates as math-prefix plus prose suffix")
    if r"$\psi^{-1}$-Whittaker" not in normalized_hybrid_suffix:
        failures.append("normalizer did not split psi^{-1}-Whittaker as math-prefix plus prose suffix")
    if r"\(\mathup{GL}_{n}-equivariant\)" in out_hybrid_suffix:
        failures.append("rendered output still has non-math word inside GL_n-equivariant math span")
    if r"\(\mathup{GL}_{n}-translates\)" in out_hybrid_suffix:
        failures.append("rendered output still has non-math word inside GL_n-translates math span")
    if r"\(\psi^{-1}-Whittaker\)" in out_hybrid_suffix:
        failures.append("rendered output still has non-math word inside psi^{-1}-Whittaker math span")
    if r"\(\mathup{GL}_{n}\)-equivariant" not in out_hybrid_suffix:
        failures.append("rendered output missing split form '(GL_n)-equivariant'")
    if r"\(\mathup{GL}_{n}\)-translates" not in out_hybrid_suffix:
        failures.append("rendered output missing split form '(GL_n)-translates'")
    if r"\(\psi^{-1}\)-Whittaker" not in out_hybrid_suffix:
        failures.append("rendered output missing split form '(psi^{-1})-Whittaker'")

    if r"for any nonzero $\phi'$ in $K(\Pi)|_{\mathup{GL}_{n}}$." not in normalized_prime_single:
        failures.append("normalizer did not convert phi' and K(Pi)|_{GL_n} in quantifier sentence")
    if r"for each $V$, there exists $I$ such that $I(s, W, V)$ = $c \ast q_F^{-ks}$." not in normalized_prime_single:
        failures.append("normalizer did not convert quantifier single letters and I(s,W,V) call")
    prime_render_variants = [
        r"for any nonzero \(\phi'\) in \(K(\Pi)|_{\mathup{GL}_{n}}\).",
        r"for any nonzero \(\phi' \in K(\Pi)|_{\mathup{GL}_{n}}\).",
    ]
    if not any(v in out_prime_single for v in prime_render_variants):
        failures.append("rendered output missing inline math for phi' / K(Pi)|_{GL_n}")
    if r"for each \(V\), there exists \(I\) such that \(I(s, W, V)\) = \(c \ast q_{F}^{-ks}\)." not in out_prime_single and r"for each \(V\), there exists \(I\) such that \(I(s, W, V)\) = \(c \ast q_F^{-ks}\)." not in out_prime_single:
        failures.append("rendered output missing inline math for V/I/I(s,W,V) sentence")

    style = style_file.read_text(encoding="utf-8")
    if r"\let\MP@orig@ast\ast" not in style:
        failures.append("math-proofread style does not preserve original \\ast")
    if r"\renewcommand{\ast}{\mOperator{\MP@orig@ast}}" not in style:
        failures.append("math-proofread style does not colorize \\ast as operator")
    if r"\renewcommand{\pi}{\mGreek{\MP@orig@pi}}" not in style:
        failures.append("math-proofread style does not colorize \\pi as Greek")
    if r"\renewcommand{\pi}{\mNumber{\MP@orig@pi}}" in style:
        failures.append("math-proofread style still categorizes \\pi as Number")
    if r"\renewcommand{\det}{\mFunction{\MP@orig@det}}" not in style:
        failures.append("math-proofread style does not colorize \\det as function")
    if r"\renewcommand{\prime}{\mOperator{\MP@orig@prime}}" not in style:
        failures.append("math-proofread style does not colorize \\prime as operator")
    if r"\renewcommand{\text}[1]{\mMathText{\MP@orig@text{##1}}}" not in style:
        failures.append("math-proofread style does not colorize \\text payloads")
    if r"\renewcommand{\mathit}[1]{\mMathItalic{\MP@orig@mathit{##1}}}" not in style:
        failures.append("math-proofread style does not colorize \\mathit payloads")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1

    print("Ratchet check passed: fixed-W, W(pi,psi), and non-coloring syntax leaks are color-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
