#!/usr/bin/env python3
"""Ratchet checks for prose-to-math normalization and syntax-class rendering."""

from __future__ import annotations

import re
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
COMPARISON_UNIFY_SNIPPET = (
    "a = b and a <= b and a != b. "
    "$a$ $=$ $b$. "
    "$x+y$ $\\le$ $z+w$.\n"
)
PLUS_STAR_UNIFY_SNIPPET = (
    "a + b and c * d and n + 12 and 3 * k. "
    "$a$ $+$ $b$. "
    "$c$ $*$ $d$.\n"
)
COMPACT_PLUS_SNIPPET = (
    "the vacancy at site j+1 and position (n, n+1).\n"
)
OPNAME_MERGE_SNIPPET = (
    "L_i = span(e_{i-1,i}, e_{i,i+1}) and ell = ker(omega|_L). "
    "Then omega(e_i, e_{i+1}) = 0.\n"
)
UNICODE_ESCAPE_SNIPPET = (
    "In R^4 = C^2, we have u ∈ V → W and A ⊂ B. "
    "Also R = C[q_F^s, q_F^{-s}].\n"
)
ALGO_BLOCK_SNIPPET = (
    "```text\n"
    "if ||r_new|| <= eps * ||b||: break\n"
    "Y = K_tau @ (Wprime @ Z) + lambda * (K_tau @ V)\n"
    "return vec(Y)\n"
    "```\n"
)
COTANGENT_STAR_SNIPPET = (
    "The graph lies in T^*R^2 and dual vectors live in T^*V.\n"
)
MU_DMU_INTEGRAL_SNIPPET = (
    "Yes. The measures µ and Tψ∗ µare equivalent. "
    "Use dmu_0 and integral_{T^3}. "
    "Also n < m and m > 0. Consider [n] and {0,1}.\n"
)
ESCAPED_SCRIPT_SNIPPET = (
    r"Q\^{}(a,b)\_\{i,j\} and p ⊞\_n q and omega\textbar\_\{L_i\}."
    "\n"
)
HAT_TOKEN_SNIPPET = (
    "Set Ahat = Phat and use phat in the preconditioner.\n"
)
PROSE_CONNECTOR_MATH_SNIPPET = (
    "$x = y for each z and there exists w such that w = z$.\n"
)


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def strip_mnumber(text: str) -> str:
    return re.sub(r"\\mNumber\{([^{}]+)\}", r"\1", text)


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
    _, out_compare = render_snippet(
        repo, normalizer, lua_filter, COMPARISON_UNIFY_SNIPPET
    )
    _, out_plus_star = render_snippet(
        repo, normalizer, lua_filter, PLUS_STAR_UNIFY_SNIPPET
    )
    normalized_compact_plus, out_compact_plus = render_snippet(
        repo, normalizer, lua_filter, COMPACT_PLUS_SNIPPET
    )
    _, out_opname_merge = render_snippet(
        repo, normalizer, lua_filter, OPNAME_MERGE_SNIPPET
    )
    normalized_unicode_escape, out_unicode_escape = render_snippet(
        repo, normalizer, lua_filter, UNICODE_ESCAPE_SNIPPET
    )
    _, out_algo_block = render_snippet(
        repo, normalizer, lua_filter, ALGO_BLOCK_SNIPPET
    )
    normalized_cotangent, out_cotangent = render_snippet(
        repo, normalizer, lua_filter, COTANGENT_STAR_SNIPPET
    )
    normalized_mu_dmu, out_mu_dmu = render_snippet(
        repo, normalizer, lua_filter, MU_DMU_INTEGRAL_SNIPPET
    )
    _, out_escaped_script = render_snippet(
        repo, normalizer, lua_filter, ESCAPED_SCRIPT_SNIPPET
    )
    _, out_hat = render_snippet(
        repo, normalizer, lua_filter, HAT_TOKEN_SNIPPET
    )
    _, out_prose_math = render_snippet(
        repo, normalizer, lua_filter, PROSE_CONNECTOR_MATH_SNIPPET
    )
    out_plus_star_raw = out_plus_star

    out = strip_mnumber(out)
    normalized_w = strip_mnumber(normalized_w)
    out_w = strip_mnumber(out_w)
    normalized_syntax = strip_mnumber(normalized_syntax)
    out_syntax = strip_mnumber(out_syntax)
    out_diag = strip_mnumber(out_diag)
    normalized_noncolor = strip_mnumber(normalized_noncolor)
    out_noncolor = strip_mnumber(out_noncolor)
    normalized_map = strip_mnumber(normalized_map)
    out_map = strip_mnumber(out_map)
    normalized_phi_action = strip_mnumber(normalized_phi_action)
    out_phi_action = strip_mnumber(out_phi_action)
    normalized_hybrid_suffix = strip_mnumber(normalized_hybrid_suffix)
    out_hybrid_suffix = strip_mnumber(out_hybrid_suffix)
    normalized_prime_single = strip_mnumber(normalized_prime_single)
    out_prime_single = strip_mnumber(out_prime_single)
    out_compare = strip_mnumber(out_compare)
    out_plus_star = strip_mnumber(out_plus_star)
    normalized_compact_plus = strip_mnumber(normalized_compact_plus)
    out_compact_plus = strip_mnumber(out_compact_plus)
    out_opname_merge = strip_mnumber(out_opname_merge)
    normalized_unicode_escape = strip_mnumber(normalized_unicode_escape)
    out_unicode_escape = strip_mnumber(out_unicode_escape)
    normalized_cotangent = strip_mnumber(normalized_cotangent)
    out_cotangent = strip_mnumber(out_cotangent)
    normalized_mu_dmu = strip_mnumber(normalized_mu_dmu)
    out_mu_dmu = strip_mnumber(out_mu_dmu)
    out_escaped_script = strip_mnumber(out_escaped_script)
    out_hat = strip_mnumber(out_hat)
    out_prose_math = strip_mnumber(out_prose_math)

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
    if r"\(\bigcup_{\phi}\)" not in out_map and r"\(\bigcup_{\phi}" not in out_map:
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
    if (
        r"for each \(V\), there exists \(I\) such that \(I(s, W, V)\) = \(c \ast q_{F}^{-ks}\)." not in out_prime_single
        and r"for each \(V\), there exists \(I\) such that \(I(s, W, V)\) = \(c \ast q_F^{-ks}\)." not in out_prime_single
        and r"for each \(V\), there exists \(I\) such that \(I(s, W, V) = c \ast q_{F}^{-ks}\)." not in out_prime_single
        and r"for each \(V\), there exists \(I\) such that \(I(s, W, V) = c \ast q_F^{-ks}\)." not in out_prime_single
    ):
        failures.append("rendered output missing inline math for V/I/I(s,W,V) sentence")

    if r"\(a = b\)" not in out_compare:
        failures.append("rendered output missing inline math for equality comparison")
    if r"\(a \le b\)" not in out_compare:
        failures.append("rendered output missing inline math for <= comparison")
    if r"\(a \neq b\)" not in out_compare:
        failures.append("rendered output missing inline math for != comparison")
    if r"\(a\) \(=\) \(b\)" in out_compare:
        failures.append("comparison unifier regressed to split '$a$ $=$ $b$' form")
    if r"\(x+y \le z+w\)" not in out_compare and r"\(x + y \le z + w\)" not in out_compare:
        failures.append("comparison unifier did not merge math globs around \\le")
    plus_variants = [
        r"\(a + b\)",
        r"\(a \mBridgeOperator{+} b\)",
    ]
    if not any(v in out_plus_star for v in plus_variants):
        failures.append("rendered output missing inline math for plus expression")
    if r"\(c \ast d\)" not in out_plus_star:
        failures.append("rendered output missing inline math for star expression")
    if r"\mNumber{12}" not in out_plus_star_raw or r"\mNumber{3}" not in out_plus_star_raw:
        failures.append("rendered output missing integer-to-\\mNumber wrapping")
    if r"\(a\) \(+\) \(b\)" in out_plus_star:
        failures.append("operator unifier regressed to split '$a$ $+$ $b$' form")
    if r"\(c\) \(\ast\) \(d\)" in out_plus_star:
        failures.append("operator unifier regressed to split '$c$ $*$ $d$' form")
    if "site $j + 1$" not in normalized_compact_plus:
        failures.append("normalizer did not convert compact plus token 'j+1' into inline math")
    if "position $(n, n + 1)$" not in normalized_compact_plus:
        failures.append("normalizer did not convert tuple token '(n, n+1)' into inline math")
    compact_plus_render_variants = [
        r"site \(j + 1\) and position \((n, n + 1)\).",
        r"site \(j + 1\) and position \((n, n+1)\).",
    ]
    if not any(v in out_compact_plus for v in compact_plus_render_variants):
        failures.append("rendered output missing compact-plus normalization for j+1 / (n, n+1)")
    if r"\mOpName{span}" not in out_opname_merge:
        failures.append("rendered output missing \\mOpName{span} for span(...)")
    if r"\mOpName{ker}" not in out_opname_merge:
        failures.append("rendered output missing \\mOpName{ker} for ker(...)")
    if r"\(\omega\)\(" in out_opname_merge:
        failures.append("math-fragment merger regressed to nested '\\(\\omega\\)(' output")
    if r"\(omega(e_i, e_{i+1}) = 0\)" in out_opname_merge:
        failures.append("omega(...) expression was not TeX-normalized to \\omega")
    omega_variants = [
        r"\(\omega(e_i, e_{i+1}) = 0\)",
        r"\(\omega(e_i, e_{i + 1}) = 0\)",
        r"\(\omega(e_{i}, e_{i+1}) = 0\)",
        r"\(\omega(e_{i}, e_{i + 1}) = 0\)",
    ]
    if not any(v in out_opname_merge for v in omega_variants):
        failures.append("rendered output missing merged omega(e_i, e_{i+1}) = 0 expression")
    if r"$\mathbb{R}^{4}$" not in normalized_unicode_escape:
        failures.append("normalizer did not convert R^4 into $\\mathbb{R}^{4}$")
    if r"$\mathbb{C}^{2}$" not in normalized_unicode_escape:
        failures.append("normalizer did not convert C^2 into $\\mathbb{C}^{2}$")
    if r"$\mathbb{C}[q_F^s, q_F^{-s}]$" not in normalized_unicode_escape:
        failures.append("normalizer did not convert C[q_F^s, q_F^{-s}] into mathbb form")
    if any(ch in out_unicode_escape for ch in ("∈", "→", "⊂")):
        failures.append("rendered output still contains raw unicode math symbols")
    if "C{[}" in out_unicode_escape or "{]}" in out_unicode_escape:
        failures.append("rendered output still contains escaped bracket artifacts")
    if r"\begin{aligned}" in out_algo_block:
        failures.append("algorithm-like code block was incorrectly converted into display math")
    if r"$T^{\mDualStar} \mathbb{R}^{2}$" not in normalized_cotangent:
        failures.append("normalizer did not convert T^*R^2 into dual-star cotangent math")
    if r"$T^{\mDualStar} V$" not in normalized_cotangent:
        failures.append("normalizer did not convert T^*V into dual-star cotangent math")
    if r"\(T^{\mDualStar} \mathbb{R}^{2}\)" not in out_cotangent:
        failures.append("rendered output missing inline math for T^*R^2 cotangent expression")
    if r"\(T^{\mDualStar} V\)" not in out_cotangent:
        failures.append("rendered output missing inline math for T^*V dual expression")
    if r"$\mu$ are equivalent" not in normalized_mu_dmu:
        failures.append("normalizer did not repair missing space after mu (muare -> mu are)")
    if r"$d\mu_0$" not in normalized_mu_dmu:
        failures.append("normalizer did not convert dmu_0 into d\\mu_0 math")
    if r"$\Integral_{T^3}$" not in normalized_mu_dmu:
        failures.append("normalizer did not convert integral_{...} into \\Integral_{...} math")
    if r"$n < m$" not in normalized_mu_dmu or r"$m > 0$" not in normalized_mu_dmu:
        failures.append("normalizer did not convert < and > comparisons into math")
    if r"\muare" in out_mu_dmu:
        failures.append("rendered output still contains glued \\muare token")
    if r"\(d\mu_{0}\)" not in out_mu_dmu and r"\(d\mu_0\)" not in out_mu_dmu:
        failures.append("rendered output missing inline math for d\\mu_0")
    if r"\(\Integral_{T^{3}}\)" not in out_mu_dmu and r"\(\Integral_{T^3}\)" not in out_mu_dmu:
        failures.append("rendered output missing inline math for \\Integral_{T^3}")
    if r"\(n < m\)" not in out_mu_dmu or r"\(m > 0\)" not in out_mu_dmu:
        failures.append("rendered output missing inline math for < or > comparisons")
    if r"\([n]\)" not in out_mu_dmu:
        failures.append("rendered output missing inline math for bracket token [n]")
    if r"\(\{0,1\}\)" not in out_mu_dmu and r"\(\{0, 1\}\)" not in out_mu_dmu:
        failures.append("rendered output missing inline math for brace token {0,1}")
    q_variants = [
        r"\(Q^{a,b}_{i,j}\)",
        r"\(Q^{(a,b)}_{i,j}\)",
        r"\(Q^{a,b}_{i, j}\)",
        r"\(Q^{(a,b)}_{i, j}\)",
    ]
    if not any(v in out_escaped_script for v in q_variants):
        failures.append("escaped script artifacts were not normalized to a clean Q superscript/subscript form")
    if r"\(\boxplus_n\)" not in out_escaped_script and r"\(\boxplus_{n}\)" not in out_escaped_script:
        failures.append("escaped unicode-script token ⊞\\_n was not normalized to a math \\boxplus subscript")
    if r"\(\omega|_{L_i}\)" not in out_escaped_script and r"\(\omega|_{L_{i}}\)" not in out_escaped_script:
        failures.append("escaped omega|_{L_i} token was not normalized to math")
    if r"\_" in out_escaped_script:
        failures.append("rendered output still contains literal \\_ escape artifact")
    if r"\^{}" in out_escaped_script:
        failures.append("rendered output still contains empty escaped-caret artifact \\^{}")
    if "textasciicircum" in out_escaped_script:
        failures.append("rendered output still contains textasciicircum escape artifact")
    if r"\hat{A}" not in out_hat:
        failures.append("Ahat token was not normalized to \\hat{A}")
    if r"\hat{P}" not in out_hat:
        failures.append("Phat token was not normalized to \\hat{P}")
    if r"\hat{p}" not in out_hat:
        failures.append("phat token was not normalized to \\hat{p}")
    if r"\text{for}" not in out_prose_math:
        failures.append("math prose connector 'for' was not wrapped as \\text{for}")
    if r"\text{each}" not in out_prose_math:
        failures.append("math prose connector 'each' was not wrapped as \\text{each}")
    if r"\text{and}" not in out_prose_math:
        failures.append("math prose connector 'and' was not wrapped as \\text{and}")
    if r"\text{there}" not in out_prose_math:
        failures.append("math prose connector 'there' was not wrapped as \\text{there}")
    if r"\text{exists}" not in out_prose_math:
        failures.append("math prose connector 'exists' was not wrapped as \\text{exists}")
    if r"\text{such}" not in out_prose_math:
        failures.append("math prose connector 'such' was not wrapped as \\text{such}")
    if r"\text{that}" not in out_prose_math:
        failures.append("math prose connector 'that' was not wrapped as \\text{that}")

    style = style_file.read_text(encoding="utf-8")
    if r"\let\MP@orig@ast\ast" not in style:
        failures.append("math-proofread style does not preserve original \\ast")
    if r"\renewcommand{\ast}{\mBridgeOperator{\MP@orig@ast}}" not in style:
        failures.append("math-proofread style does not colorize \\ast as bridge operator")
    if r'\begingroup\catcode`\+=\active\gdef+{{\color{MPSyntaxBridgeOperatorColor}\mathchar"202B}}\endgroup' not in style:
        failures.append("math-proofread style does not colorize raw '+' as bridge operator")
    if r'\mathcode`+="8000' not in style:
        failures.append("math-proofread style does not activate raw '+' in math mode")
    if r'\mathcode`+="202B' not in style:
        failures.append("math-proofread style does not restore default raw '+' mathcode on disable")
    if r"\colorlet{MPSyntaxBridgeOperatorColor}{SeaGreen}" not in style:
        failures.append("math-proofread style does not define medium green bridge-operator color")
    if r"\colorlet{MPSyntaxNamedOperatorColor}{BurntOrange}" not in style:
        failures.append("math-proofread style does not define named-operator color class")
    if r"\newcommand{\mOpName}[1]" not in style:
        failures.append("math-proofread style does not define \\mOpName macro")
    if r"\providecommand{\Integral}{\int}" not in style:
        failures.append("math-proofread style does not provide \\Integral alias")
    if r"\colorlet{MPSyntaxDualMarkerColor}{Turquoise}" not in style:
        failures.append("math-proofread style does not define dual-marker color class")
    if r"\newcommand{\mDualStar}" not in style:
        failures.append("math-proofread style does not define \\mDualStar macro")
    if r"\renewcommand{\pi}{\mGreek{\MP@orig@pi}}" not in style:
        failures.append("math-proofread style does not colorize \\pi as Greek")
    if r"\renewcommand{\pi}{\mNumber{\MP@orig@pi}}" in style:
        failures.append("math-proofread style still categorizes \\pi as Number")
    if r"\renewcommand{\det}{\mFunction{\MP@orig@det}}" not in style:
        failures.append("math-proofread style does not colorize \\det as function")
    if r"\definecolor{MPSyntaxComparisonColor}{RGB}{0,66,37}" not in style:
        failures.append("math-proofread style does not define comparison color (British racing green)")
    if r"\renewcommand{\le}{\mCompare{\MP@orig@le}}" not in style:
        failures.append("math-proofread style does not colorize \\le as comparison")
    if r"\renewcommand{\neq}{\mCompare{\MP@orig@neq}}" not in style:
        failures.append("math-proofread style does not colorize \\neq as comparison")
    if r"\renewcommand{\prime}{\mOperator{\MP@orig@prime}}" not in style:
        failures.append("math-proofread style does not colorize \\prime as operator")
    if r"\renewcommand{\text}[1]{\mMathText{\MP@orig@text{##1}}}" not in style:
        failures.append("math-proofread style does not colorize \\text payloads")
    if r"\renewcommand{\mathit}[1]{\mMathItalic{\MP@orig@mathit{##1}}}" not in style:
        failures.append("math-proofread style does not colorize \\mathit payloads")
    if r"\colorlet{MPSyntaxNumberColor}{Red}" not in style:
        failures.append("math-proofread style does not set integer/number color to Red")
    if r"\colorlet{MPSyntaxDelimiterColor}{Magenta}" not in style:
        failures.append("math-proofread style does not set delimiters to Magenta")
    if r"\definecolor{MPSyntaxBlueBlack}{RGB}{10,24,48}" not in style:
        failures.append("math-proofread style does not define blue-black default math color")
    if r"\colorlet{MPSyntaxDefaultMathColor}{MPSyntaxBlueBlack}" not in style:
        failures.append("math-proofread style does not set default math color to blue-black")
    if r"\let\MP@orig@textlbrace\{" not in style:
        failures.append("math-proofread style does not preserve original text left brace (\\{)")
    if r"\let\MP@orig@textrbrace\}" not in style:
        failures.append("math-proofread style does not preserve original text right brace (\\})")
    if r"\renewcommand{\{}{{\color{MPSyntaxDelimiterColor}\MP@orig@textlbrace}}" not in style:
        failures.append("math-proofread style does not colorize text left brace (\\{) as delimiter")
    if r"\renewcommand{\}}{{\color{MPSyntaxDelimiterColor}\MP@orig@textrbrace}}" not in style:
        failures.append("math-proofread style does not colorize text right brace (\\}) as delimiter")
    if r"\let\{\MP@orig@textlbrace" not in style:
        failures.append("math-proofread style does not restore original text left brace on disable")
    if r"\let\}\MP@orig@textrbrace" not in style:
        failures.append("math-proofread style does not restore original text right brace on disable")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1

    print("Ratchet check passed: fixed-W, W(pi,psi), and non-coloring syntax leaks are color-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
