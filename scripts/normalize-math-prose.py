#!/usr/bin/env python3
"""Conservative Markdown prose-math normalizer.

Scope of automatic edits (high-confidence only):
- Bare Big-O terms: O(...) -> $O(...)$
- Hyphenated variable adjectives: N-dependent -> $N$-dependent

Protected regions (never rewritten):
- fenced code blocks
- indented code blocks
- inline code spans
- inline math spans ($...$)
- display math blocks ($$...$$)

Safety:
- In-place file rewrites require both --write and --allow-in-place.
"""

from __future__ import annotations

import argparse
import glob
import re
import tempfile
from pathlib import Path

UNICODE_TO_TEX = {
    "≤": r"\le",
    "≥": r"\ge",
    "≠": r"\ne",
    "×": r"\times",
    "⊗": r"\otimes",
    "⊕": r"\oplus",
    "∘": r"\circ",
    "→": r"\to",
    "←": r"\leftarrow",
    "∧": r"\wedge",
    "⊂": r"\subset",
    "∈": r"\in",
    "∞": r"\infty",
    "μ": r"\mu",
    "µ": r"\mu",
    "ψ": r"\psi",
    "Ψ": r"\Psi",
    "π": r"\pi",
    "Π": r"\Pi",
    "∗": r"\ast",
}
UNICODE_MATH_CHAR_RE = re.compile("|".join(re.escape(k) for k in UNICODE_TO_TEX))

N_DEP_RE = re.compile(r"\b([A-Z])-(dependent|independent)\b")
WHITTAKER_W_RE = re.compile(r"\bW\(\s*pi\s*,\s*psi\s*\)")
WHITTAKER_W_ANY_RE = re.compile(
    r"\bW\(\s*(Pi|pi)\s*,\s*psi(?:\^\{(-?1)\}|\^(-?1))?\s*\)"
)
PSI_WHITTAKER_RE = re.compile(r"\bpsi-Whittaker\b")
PSI_INV_WHITTAKER_RE = re.compile(r"\bpsi(?:\^\{-?1\}|\^-?1)-Whittaker\b")
PSI_CALL_RE = re.compile(r"\bpsi\(([^()]+)\)")
NORMALIZES_PSI_RE = re.compile(r"\bnormalizes psi\b")
CHARACTER_PSI_RE = re.compile(r"\bcharacter psi\b")
DET_ABS_RE = re.compile(r"\|det\s+([A-Za-z0-9_]+)\|\^\{([^}]+)\}")
DIAG_CALL_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\(\s*diag\(\s*([^()]*)\s*\)\s*([A-Za-z0-9_\\^{}+\-]+)\s*\)"
)
NEQ_PAREN_RE = re.compile(r"\((\s*[A-Za-z0-9_]+\s*)!=(\s*[A-Za-z0-9_]+\s*)\)")
DIM_NUM_RE = re.compile(r"\b(\d+)\s+[xX]\s+(\d+)\b")
QF_RE = re.compile(r"\bqF(\^\{[^}]+\}|\^-?[A-Za-z0-9]+)")
STAR_QF_RE = re.compile(r"\b([A-Za-z])\s*\*\s*(q_F(?:\^\{[^}]+\}|\^-?[A-Za-z0-9]+))")
SINGLE_IN_RE = re.compile(r"\b([A-Za-z])\s+in\s+([A-Za-z])\b([.,;:]?)")
GL_TOKEN_RE = r"GL(?:_\{?[A-Za-z0-9+\-]+\}?|[A-Za-z0-9+\-]+)"
GL_MEMBERSHIP_PRODUCT_RE = re.compile(
    rf"\b([A-Za-z])\s+in\s+({GL_TOKEN_RE})\s*[xX]\s*({GL_TOKEN_RE})\b"
)
GL_PRODUCT_RE = re.compile(rf"\b({GL_TOKEN_RE})\s*[xX]\s*({GL_TOKEN_RE})\b")
GL_HYPHEN_WORD_RE = re.compile(rf"\b({GL_TOKEN_RE})-([A-Za-z][A-Za-z-]+)\b")
FIXED_W_RE = re.compile(r"\bfor fixed\s+W\s*=\s*W_0\b")
IDEAL_EQ_RE = re.compile(
    r"\bI\s*=\s*L\(s,\s*Pi\s*x\s*pi\)\s*\*\s*C\[q_F\^s,\s*q_F\^{-s}\]"
)
L_FACTOR_RE = re.compile(r"L\(s,\s*Pi\s*x\s*pi\)")
GREEK_PRIME_TOKEN_RE = re.compile(r"\b(Phi|Pi|Psi|phi|pi|psi)(['*])(?=[^A-Za-z0-9_]|$)")
EXISTS_SINGLE_RE = re.compile(r"\b(there exists)\s+([A-Za-z])\s+such that\b", re.IGNORECASE)
FOR_EACH_SINGLE_RE = re.compile(r"\b(for each)\s+([A-Za-z])\b", re.IGNORECASE)
I_S_CALL_RE = re.compile(
    r"\bI\(\s*s\s*,\s*([A-Za-z][A-Za-z0-9_']*)\s*,\s*([A-Za-z][A-Za-z0-9_']*)\s*\)"
)
Z_SQRT_RING_RE = re.compile(r"\bZ\[\s*(?:sqrt\(\s*([0-9]+)\s*\)|√\s*([0-9]+))\s*\]")
Z_GAMMA_RING_RE = re.compile(r"\bZ\[\s*(?:Gamma|Γ)\s*\]")
KIRILLOV_MODEL_RE = re.compile(
    r"K\(\s*Pi\s*\)\s*\|\s*(?:_\{\s*([^}]+)\s*\}|([A-Za-z0-9_+\-]+))"
)
MAP_SIGNATURE_RE = re.compile(
    r"Phi:\s*K\(\s*Pi\s*\)\s*\|\s*(?:_\{\s*([^}]+)\s*\}|([A-Za-z0-9_+\-]+))"
    r"\s*→\s*\(fractional ideals of\s+([A-Za-z])\)"
)
UNION_SUB_RE = re.compile(r"∪_\{\s*([^}]+)\s*\}")
PHI_CALL_RE = re.compile(
    r"(?<!\\)\bPhi\(\s*([A-Za-z0-9_\\]+(?:\([^()]*\)[A-Za-z0-9_\\]*)?)\s*\)"
)
PHI_SET_RE = re.compile(
    r"Phi\(\s*phi\s*\)\s*=\s*\{\s*I\(\s*s\s*,\s*phi\s*,\s*V\s*\)\s*:\s*"
    r"V\s+in\s+W\(\s*pi\s*,\s*psi\s*\)\s*\}\s*·\s*([A-Za-z])"
)
COMPLEX_RING_RE = re.compile(r"\bC\[\s*([^\]]+)\]")
FIELD_POWER_RE = re.compile(r"\b([RCZQ])\^([0-9]+|[A-Za-z])\b")
COTANGENT_FIELD_RE = re.compile(r"\bT\^\\?\*\s*([RCZQ])\^([0-9]+|[A-Za-z])\b")
COTANGENT_TOKEN_RE = re.compile(r"\bT\^\\?\*\s*([A-Za-z][A-Za-z0-9_]*)\b")
OP_CALL_TEXT_RE = re.compile(r"\b(span|ker|rank|dim|codim|trace)\(([^()]+)\)")
OMEGA_RESTRICT_RE = re.compile(r"\bomega\|_([A-Za-z0-9]+)\b")
OMEGA_EQ_RE = re.compile(r"\bomega\(([^()]+)\)\s*=\s*([0-9]+)\b")
MU_GLUE_RE = re.compile(r"([μµ])([A-Za-z])")
DMU_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])dmu(?:_\{[^}]+\}|_[A-Za-z0-9+\-]+)?(?=[^A-Za-z0-9_]|$)"
)
INTEGRAL_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])integral(?:_\{[^}]+\}|_[A-Za-z0-9+\-]+)?(?=[^A-Za-z0-9_]|$)"
)
SIMPLE_COMPARISON_RE = re.compile(
    r"(?<![$\\])\b([A-Za-z][A-Za-z0-9_]*|\d+)\s*([<>])\s*([A-Za-z][A-Za-z0-9_]*|\d+)\b"
)
COMPACT_PLUS_TUPLE_RE = re.compile(
    r"(?<![$\\{_])\(\s*([A-Za-z][A-Za-z0-9_]*|\d+)\s*,\s*([A-Za-z])\+(\d+)\s*\)"
)
COMPACT_PLUS_RE = re.compile(
    r"(?<![$\\_^{}.,])\b([A-Za-z])\+(\d+)\b"
)
TPSI_STAR_RE = re.compile(r"T([ψΨ])([∗*])")
SN_ALL_PERMS_RE = re.compile(
    r"\bS_?n\(\s*(?:lambda|\\lambda|λ)\s*\)\s*=\s*\{\s*all\s+permutations\s+of\s+the\s+parts\s+of\s*(?:lambda|\\lambda|λ)\s*\}\s*([.?!,;:]?)",
    re.IGNORECASE,
)


def _unicode_math_char_repl(m: re.Match[str]) -> str:
    return f"${UNICODE_TO_TEX[m.group(0)]}$"


def _complex_ring_repl(m: re.Match[str]) -> str:
    inner = m.group(1).strip()
    inner = inner.replace("qF", "q_F")
    return rf"$\mathbb{{C}}[{inner}]$"


def _field_power_repl(m: re.Match[str]) -> str:
    letter = m.group(1)
    exp = m.group(2)
    return rf"$\mathbb{{{letter}}}^{{{exp}}}$"


def _cotangent_field_repl(m: re.Match[str]) -> str:
    letter = m.group(1)
    exp = m.group(2)
    return rf"$T^{{\mDualStar}} \mathbb{{{letter}}}^{{{exp}}}$"


def _cotangent_token_repl(m: re.Match[str]) -> str:
    token = _texify_token(m.group(1))
    return rf"$T^{{\mDualStar}} {token}$"


def _op_call_text_repl(m: re.Match[str]) -> str:
    op = m.group(1)
    args = m.group(2)
    args = re.sub(r"\bomega\|_([A-Za-z0-9]+)\b", r"\\omega|_{\1}", args)
    return rf"$\mOpName{{{op}}}({args})$"


def _dmu_token_repl(m: re.Match[str]) -> str:
    tok = m.group(0)
    suffix = tok[3:]
    return rf"$d\mu{suffix}$"


def _integral_token_repl(m: re.Match[str]) -> str:
    tok = m.group(0)
    suffix = tok[len("integral") :]
    return rf"$\Integral{suffix}$"


def _simple_comparison_repl(m: re.Match[str]) -> str:
    lhs = _texify_token(m.group(1))
    op = m.group(2)
    rhs = _texify_token(m.group(3))
    return rf"${lhs} {op} {rhs}$"


def _compact_plus_tuple_repl(m: re.Match[str]) -> str:
    lhs = _texify_token(m.group(1))
    rhs_left = _texify_token(m.group(2))
    rhs_right = _texify_token(m.group(3))
    return rf"$({lhs}, {rhs_left} + {rhs_right})$"


def _compact_plus_repl(m: re.Match[str]) -> str:
    lhs = _texify_token(m.group(1))
    rhs = _texify_token(m.group(2))
    return rf"${lhs} + {rhs}$"


def _tpsi_star_repl(m: re.Match[str]) -> str:
    psi = r"\Psi" if m.group(1) == "Ψ" else r"\psi"
    return rf"$T_{{{psi}}}^{{\mDualStar}}$"


def _sn_all_perms_repl(m: re.Match[str]) -> str:
    punct = m.group(1) or ""
    return r"$S_n(\lambda) = \{\text{all permutations of the parts of } \lambda\}$" + punct


def split_inline_code(s: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "`":
            j = s.find("`", i + 1)
            if j < 0:
                out.append(("text", s[i:]))
                break
            out.append(("code", s[i : j + 1]))
            i = j + 1
        else:
            j = s.find("`", i)
            if j < 0:
                out.append(("text", s[i:]))
                break
            out.append(("text", s[i:j]))
            i = j
    return out


def split_inline_math_dollar(s: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "$" and (i == 0 or s[i - 1] != "\\"):
            j = i + 1
            while True:
                j = s.find("$", j)
                if j < 0:
                    out.append(("text", s[i:]))
                    return out
                if s[j - 1] != "\\":
                    break
                j += 1
            out.append(("math", s[i : j + 1]))
            i = j + 1
        else:
            j = s.find("$", i)
            if j < 0:
                out.append(("text", s[i:]))
                return out
            out.append(("text", s[i:j]))
            i = j
    return out


def _normalize_math_expr(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    for src, dst in UNICODE_TO_TEX.items():
        s = s.replace(src, dst)
    return s


def _wrap_big_o_balanced(s: str) -> str:
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith("O(", i) and (i == 0 or not s[i - 1].isalnum()):
            j = i + 2
            depth = 1
            while j < n and depth > 0:
                ch = s[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                expr = _normalize_math_expr(s[i:j])
                out.append(f"${expr}$")
                i = j
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _n_dep_repl(m: re.Match[str]) -> str:
    return f"${m.group(1)}$-{m.group(2)}"


def _whittaker_any_repl(m: re.Match[str]) -> str:
    rep = m.group(1)
    inv = m.group(2) or m.group(3)
    rep_tex = r"\Pi" if rep == "Pi" else r"\pi"
    psi_tex = r"\psi^{-1}" if inv else r"\psi"
    return rf"$W({rep_tex}, {psi_tex})$"


def _diag_call_repl(m: re.Match[str]) -> str:
    fn, args, rhs = m.group(1), m.group(2), m.group(3)
    return rf"${fn}(\operatorname{{diag}}({args}){rhs})$"


def _normalize_gl_token(token: str) -> str:
    if token.startswith("GL_{") and token.endswith("}"):
        idx = token[4:-1]
    elif token.startswith("GL_"):
        idx = token[3:]
    elif token.startswith("GL"):
        idx = token[2:]
    else:
        return token
    return rf"\mathup{{GL}}_{{{idx}}}"


def _single_in_repl(m: re.Match[str]) -> str:
    lhs, rhs, punct = m.group(1), m.group(2), m.group(3)
    return rf"${lhs} \in {rhs}$" + punct


def _star_qf_repl(m: re.Match[str]) -> str:
    lhs, rhs = m.group(1), m.group(2)
    return rf"${lhs} \ast {rhs}$"


def _gl_membership_product_repl(m: re.Match[str]) -> str:
    var = m.group(1)
    left = _normalize_gl_token(m.group(2))
    right = _normalize_gl_token(m.group(3))
    return rf"${var} \in {left} \times {right}$"


def _gl_product_repl(m: re.Match[str]) -> str:
    left = _normalize_gl_token(m.group(1))
    right = _normalize_gl_token(m.group(2))
    return rf"${left} \times {right}$"


def _gl_hyphen_word_repl(m: re.Match[str]) -> str:
    gl, word = m.group(1), m.group(2)
    return rf"${_normalize_gl_token(gl)}$-{word}"


def _exists_single_repl(m: re.Match[str]) -> str:
    lead, var = m.group(1), m.group(2)
    return f"{lead} ${var}$ such that"


def _for_each_single_repl(m: re.Match[str]) -> str:
    lead, var = m.group(1), m.group(2)
    return f"{lead} ${var}$"


def _texify_symbol(token: str) -> str:
    token = token.strip()
    m = re.fullmatch(r"([A-Za-z]+)(['`]*)", token)
    if not m:
        return token
    base, primes = m.group(1), m.group(2)
    greek = {
        "Phi": r"\Phi",
        "Pi": r"\Pi",
        "Psi": r"\Psi",
        "phi": r"\phi",
        "pi": r"\pi",
        "psi": r"\psi",
    }
    if base in greek:
        return greek[base] + primes
    return base + primes


def _texify_token(token: str) -> str:
    token = _texify_symbol(token.strip())
    token = re.sub(r"\bPi\b", r"\\Pi", token)
    token = re.sub(r"\bPhi\b", r"\\Phi", token)
    token = re.sub(r"\bPsi\b", r"\\Psi", token)
    token = re.sub(r"\bpi\b", r"\\pi", token)
    token = re.sub(r"\bphi\b", r"\\phi", token)
    token = re.sub(r"\bpsi\b", r"\\psi", token)
    return token


def _greek_prime_repl(m: re.Match[str]) -> str:
    base = _texify_symbol(m.group(1))
    # Treat bare '*' suffix as ascii-art prime.
    return rf"${base}'$"


def _i_s_call_repl(m: re.Match[str]) -> str:
    lhs = _texify_token(m.group(1))
    rhs = _texify_token(m.group(2))
    return rf"$I(s, {lhs}, {rhs})$"


def _kirillov_model_repl(m: re.Match[str]) -> str:
    gl = m.group(1) or m.group(2) or ""
    return rf"$K(\Pi)|_{{{_normalize_gl_token(gl)}}}$"


def _map_signature_repl(m: re.Match[str]) -> str:
    gl = m.group(1) or m.group(2) or ""
    ring = m.group(3)
    return (
        rf"$\Phi: K(\Pi)|_{{{_normalize_gl_token(gl)}}} \to "
        rf"(\text{{fractional ideals of }} {ring})$"
    )


def _union_sub_repl(m: re.Match[str]) -> str:
    idx = _texify_symbol(m.group(1))
    return rf"$\bigcup_{{{idx}}}$"


def _phi_call_repl(m: re.Match[str]) -> str:
    expr = m.group(1).strip()
    expr = re.sub(r"\bPi\b", r"\\Pi", expr)
    expr = re.sub(r"\bphi\b", r"\\phi", expr)
    expr = re.sub(r"\bpsi\b", r"\\psi", expr)
    expr = re.sub(r"\bpi\b", r"\\pi", expr)
    expr = re.sub(r"R\(\s*([A-Za-z0-9_]+)\s*\)\s*\\phi", r"R(\1)\\,\\phi", expr)
    return rf"$\Phi({expr})$"


def _phi_set_repl(m: re.Match[str]) -> str:
    ring = m.group(1)
    return (
        r"$\Phi(\phi) = \{ I(s, \phi, V) : V \in W(\pi, \psi) \} "
        rf"\cdot {ring}$"
    )


def process_plain_text_segment(s: str) -> str:
    s = SN_ALL_PERMS_RE.sub(_sn_all_perms_repl, s)
    s = MAP_SIGNATURE_RE.sub(_map_signature_repl, s)
    s = PHI_SET_RE.sub(_phi_set_repl, s)
    s = Z_SQRT_RING_RE.sub(
        lambda m: rf"$\mathbb{{Z}}[\sqrt{{{m.group(1) or m.group(2)}}}]$", s
    )
    s = Z_GAMMA_RING_RE.sub(lambda _m: r"$\mathbb{Z}[\Gamma]$", s)
    s = _wrap_big_o_balanced(s)
    s = N_DEP_RE.sub(_n_dep_repl, s)
    s = PSI_INV_WHITTAKER_RE.sub(r"$\\psi^{-1}$-Whittaker", s)
    s = PSI_WHITTAKER_RE.sub(r"$\\psi$-Whittaker", s)
    s = WHITTAKER_W_RE.sub(r"$W(\\pi, \\psi)$", s)
    s = WHITTAKER_W_ANY_RE.sub(_whittaker_any_repl, s)
    s = DIAG_CALL_RE.sub(_diag_call_repl, s)
    s = DET_ABS_RE.sub(r"$|\\det \1|^{\2}$", s)
    s = PSI_CALL_RE.sub(r"$\\psi(\1)$", s)
    s = NORMALIZES_PSI_RE.sub(r"normalizes $\\psi$", s)
    s = CHARACTER_PSI_RE.sub(r"character $\\psi$", s)
    s = NEQ_PAREN_RE.sub(r"$(\1\\neq\2)$", s)
    s = DIM_NUM_RE.sub(r"$\1 \\times \2$", s)
    s = QF_RE.sub(r"q_F\1", s)
    s = STAR_QF_RE.sub(_star_qf_repl, s)
    s = SINGLE_IN_RE.sub(_single_in_repl, s)
    s = re.sub(r"\$([A-Za-z]) \\in ([A-Za-z])\$\^([A-Za-z0-9]+)", r"$\1 \\in \2^\3$", s)
    s = EXISTS_SINGLE_RE.sub(_exists_single_repl, s)
    s = FOR_EACH_SINGLE_RE.sub(_for_each_single_repl, s)
    s = I_S_CALL_RE.sub(_i_s_call_repl, s)
    s = GL_MEMBERSHIP_PRODUCT_RE.sub(_gl_membership_product_repl, s)
    s = GL_PRODUCT_RE.sub(_gl_product_repl, s)
    s = GL_HYPHEN_WORD_RE.sub(_gl_hyphen_word_repl, s)
    s = KIRILLOV_MODEL_RE.sub(_kirillov_model_repl, s)
    s = PHI_CALL_RE.sub(_phi_call_repl, s)
    s = UNION_SUB_RE.sub(_union_sub_repl, s)
    s = GREEK_PRIME_TOKEN_RE.sub(_greek_prime_repl, s)
    s = re.sub(r"\bover V\b", r"over $V$", s)
    s = FIXED_W_RE.sub(r"for fixed $W = W_0$", s)
    s = IDEAL_EQ_RE.sub(
        r"$I = L(s, \\Pi \\times \\pi)\\,\\ast\\,\\mathbb{C}[q_F^s, q_F^{-s}]$",
        s,
    )
    s = L_FACTOR_RE.sub(r"$L(s, \\Pi \\times \\pi)$", s)
    s = re.sub(r"\bfractional ideal I\b", r"fractional ideal $I$", s)
    s = re.sub(r"\bI is a free rank-1 module\b", r"$I$ is a free rank-1 module", s)
    s = re.sub(r"\bof I of the form\b", r"of $I$ of the form", s)
    s = s.replace("{[}", "[").replace("{]}", "]")
    s = s.replace("\\textbar{}", "|").replace("\\textbar", "|")
    s = s.replace("\\textless{}", "<").replace("\\textgreater{}", ">")
    s = s.replace("\\textless", "<").replace("\\textgreater", ">")
    s = MU_GLUE_RE.sub(r"\1 \2", s)
    s = TPSI_STAR_RE.sub(_tpsi_star_repl, s)
    s = re.sub(r"\^\\?\*(?=[A-Za-z0-9(])", r"^\\*", s)
    s = DMU_TOKEN_RE.sub(_dmu_token_repl, s)
    s = INTEGRAL_TOKEN_RE.sub(_integral_token_repl, s)
    s = COMPACT_PLUS_TUPLE_RE.sub(_compact_plus_tuple_repl, s)
    s = COMPACT_PLUS_RE.sub(_compact_plus_repl, s)
    s = SIMPLE_COMPARISON_RE.sub(_simple_comparison_repl, s)
    s = COMPLEX_RING_RE.sub(_complex_ring_repl, s)
    s = COTANGENT_FIELD_RE.sub(_cotangent_field_repl, s)
    s = COTANGENT_TOKEN_RE.sub(_cotangent_token_repl, s)
    s = FIELD_POWER_RE.sub(_field_power_repl, s)
    s = OP_CALL_TEXT_RE.sub(_op_call_text_repl, s)
    s = OMEGA_RESTRICT_RE.sub(r"$\\omega|_{\1}$", s)
    s = OMEGA_EQ_RE.sub(r"$\\omega(\1) = \2$", s)
    s = UNICODE_MATH_CHAR_RE.sub(_unicode_math_char_repl, s)
    return s


def process_line(line: str) -> str:
    out_parts: list[str] = []
    for kind_code, part in split_inline_code(line):
        if kind_code == "code":
            out_parts.append(part)
            continue
        for kind_math, sub in split_inline_math_dollar(part):
            if kind_math == "math":
                out_parts.append(sub)
            else:
                out_parts.append(process_plain_text_segment(sub))
    return "".join(out_parts)


def _count_unescaped_dollar2(line: str) -> int:
    return len(re.findall(r"(?<!\\)\$\$", line))


def process_file(path: Path, write: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    in_fence = False
    in_display_dollar = False
    out_lines: list[str] = []

    for ln in lines:
        stripped = ln.lstrip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            out_lines.append(ln)
            continue

        if in_fence:
            out_lines.append(ln)
            continue

        if in_display_dollar or _count_unescaped_dollar2(ln) > 0:
            out_lines.append(ln)
            if _count_unescaped_dollar2(ln) % 2 == 1:
                in_display_dollar = not in_display_dollar
            continue

        if ln.startswith("    ") or ln.startswith("\t"):
            stripped_indent = ln.lstrip(" \t")
            if SN_ALL_PERMS_RE.search(stripped_indent):
                indent_len = len(ln) - len(stripped_indent)
                prefix = ln[:indent_len]
                out_lines.append(prefix + process_line(stripped_indent))
                continue
            # Keep true code-like blocks untouched.
            code_like = re.match(
                r"(?:```|~~~|[#>{}\[\]$\\]|[A-Za-z_][A-Za-z0-9_]*\s*[:=]|"
                r"[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\s*=|"
                r"(?:def|class|for|while|if|elif|else|return|import|from)\b)",
                stripped_indent,
            ) is not None
            has_formula_symbol = re.search(r"(?:\||->|<-|→|←|\^|_)", stripped_indent) is not None
            if (
                code_like
                or ("\\" in stripped_indent)
                or ("=" in stripped_indent)
                or has_formula_symbol
            ):
                out_lines.append(ln)
                continue
            # Indented markdown continuations in lists are prose, not code.
            indent_len = len(ln) - len(stripped_indent)
            prefix = ln[:indent_len]
            out_lines.append(prefix + process_line(stripped_indent))
            continue

        out_lines.append(process_line(ln))

    updated = "".join(out_lines)
    changed = updated != original
    if changed and write:
        path.write_text(updated, encoding="utf-8")
    return changed


def run_self_test() -> None:
    cases = [
        ("Total O(q r) work.", r"Total $O(q r)$ work."),
        ("N-dependent bounds", r"$N$-dependent bounds"),
        ("CG needs only y = A_tau x.", "CG needs only y = A_tau x."),
        ("Use lambda_{abgd} and a,b,g,d.", "Use lambda_{abgd} and a,b,g,d."),
        (
            "By Section 3a below, the integrals over V (for fixed W = W_0) "
            "generate the full fractional ideal I = L(s, Pi x pi) * C[q_F^s, q_F^{-s}].",
            "By Section 3a below, the integrals over $V$ (for fixed $W = W_0$) "
            "generate the full fractional ideal "
            "$I = L(s, \\Pi \\times \\pi)\\,\\ast\\,\\mathbb{C}[q_F^s, q_F^{-s}]$.",
        ),
        (
            "In the psi-Whittaker model W(pi, psi), choose V in W(pi,psi).",
            "In the $\\psi$-Whittaker model $W(\\pi, \\psi)$, choose V in $W(\\pi, \\psi)$.",
        ),
        (
            "realized in psi^{-1}-Whittaker; Phi is GL_n-equivariant with GL_n-translates.",
            r"realized in $\psi^{-1}$-Whittaker; Phi is $\mathup{GL}_{n}$-equivariant with $\mathup{GL}_{n}$-translates.",
        ),
        ("for each V, there exists I such that I(s, W, V) = c * qF^{-ks}.",
         r"for each $V$, there exists $I$ such that $I(s, W, V)$ = $c \ast q_F^{-ks}$."),
        ("for any nonzero phi' in K(Pi)|_{GL_n}.",
         r"for any nonzero $\phi'$ in $K(\Pi)|_{\mathup{GL}_{n}}$."),
        ("choose Q in F^x.", r"choose $Q \in F^x$."),
        ("hence P in R.", r"hence $P \in R$."),
        ("equal to c * qF^{-ks}", r"equal to $c \ast q_F^{-ks}$"),
        ("r in GLn+1 x GLn", r"$r \in \mathup{GL}_{n+1} \times \mathup{GL}_{n}$"),
        ("for GL_{n+1} x GL_n is", r"for $\mathup{GL}_{n+1} \times \mathup{GL}_{n}$ is"),
        (
            "realized in W(Pi, psi^{-1}) and some V in W(pi, psi).",
            r"realized in $W(\Pi, \psi^{-1})$ and some V in $W(\pi, \psi)$.",
        ),
        ("which normalizes psi.", r"which normalizes $\psi$."),
        ("the generic character psi of N_n.", r"the generic character $\psi$ of N_n."),
        ("psi(u_Q) = psi(Q)", r"$\psi(u_Q)$ = $\psi(Q)$"),
        ("|det g_0|^{1/2-s}", r"$|\det g_0|^{1/2-s}$"),
        ("W_0(diag(g,1) u_Q)", r"$W_0(\operatorname{diag}(g,1)u_Q)$"),
        ("(j != i)", r"$(j \neq i)$"),
        ("3 x 4 matrices", r"$3 \times 4$ matrices"),
        ("site j+1", r"site $j + 1$"),
        ("position (n, n+1)", r"position $(n, n + 1)$"),
        ("rotation matrices over Z[√2]", r"rotation matrices over $\mathbb{Z}[\sqrt{2}]$"),
        ("theta = 0 in L_8(Z[Γ])", r"theta = 0 in L_8($\mathbb{Z}[\Gamma]$)"),
        ("graphs in T^*R^2 are exact.", r"graphs in $T^{\mDualStar} \mathbb{R}^{2}$ are exact."),
        (
            "Yes. The measures µ and Tψ∗ µare equivalent. Use dmu_0 and integral_{T^3}. n < m and m > 0.",
            r"Yes. The measures $\mu$ and $T_{\psi}^{\mDualStar}$ $\mu$ are equivalent. Use $d\mu_0$ and $\Integral_{T^3}$. $n < m$ and $m > 0$.",
        ),
        (
            "Consider the map Phi: K(Pi)|_{GL_n} → (fractional ideals of R) defined by "
            "Phi(phi) = { I(s, phi, V) : V in W(pi, psi) } · R. "
            "By the JPSS theory, ∪_{phi} Phi(phi) generates L(s, Pi x pi) · R.",
            r"Consider the map $\Phi: K(\Pi)|_{\mathup{GL}_{n}} \to (\text{fractional ideals of } R)$ defined by "
            r"$\Phi(\phi) = \{ I(s, \phi, V) : V \in W(\pi, \psi) \} \cdot R$. "
            r"By the JPSS theory, $\bigcup_{\phi}$ $\Phi(\phi)$ generates $L(s, \Pi \times \pi)$ · R.",
        ),
        (
            "S_n(lambda) = {all permutations of the parts of lambda}.",
            r"$S_n(\lambda) = \{\text{all permutations of the parts of } \lambda\}$.",
        ),
    ]
    for raw, want in cases:
        got = process_plain_text_segment(raw)
        if got != want:
            raise AssertionError(
                f"self-test failed:\nraw:  {raw}\nwant: {want}\ngot:  {got}"
            )

    sample = """line before\n$$\nF(lambda_{abgd} Q^(abgd) : a,b,g,d in [n]) = 0\n$$\nline after\n"""
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=True, encoding="utf-8") as tmp:
        tmp.write(sample)
        tmp.flush()
        changed = process_file(Path(tmp.name), write=True)
        got = Path(tmp.name).read_text(encoding="utf-8")
        if changed or got != sample:
            raise AssertionError("self-test failed: display-math block should remain unchanged")

    indented = (
        "1. item\n"
        "   - note with `L_n(Z[Gamma]) tensor Q`\n"
        "     rotation matrices over Z[√2]\n"
    )
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=True, encoding="utf-8") as tmp:
        tmp.write(indented)
        tmp.flush()
        process_file(Path(tmp.name), write=True)
        got = Path(tmp.name).read_text(encoding="utf-8")
        if "rotation matrices over $\\mathbb{Z}[\\sqrt{2}]$" not in got:
            raise AssertionError("self-test failed: indented prose continuation was not normalized")
        if "`L_n(Z[Gamma]) tensor Q`" not in got:
            raise AssertionError("self-test failed: inline code in indented prose was modified")

    indented_equation = "1. item\n    S_n(lambda) = {all permutations of the parts of lambda}.\n"
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=True, encoding="utf-8") as tmp:
        tmp.write(indented_equation)
        tmp.flush()
        process_file(Path(tmp.name), write=True)
        got = Path(tmp.name).read_text(encoding="utf-8")
        want = "    $S_n(\\lambda) = \\{\\text{all permutations of the parts of } \\lambda\\}$.\n"
        if want not in got:
            raise AssertionError("self-test failed: indented S_n(lambda) set-definition was not normalized")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["data/first-proof/problem*-solution.md"])
    ap.add_argument("--write", action="store_true")
    ap.add_argument(
        "--allow-in-place",
        action="store_true",
        help="required with --write; prevents accidental source-file rewrites",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        print("Self-test passed.")
        return 0

    if args.write and not args.allow_in_place:
        ap.error("--write requires --allow-in-place")

    files: list[Path] = []
    for p in args.paths:
        matches = sorted(glob.glob(p))
        if matches:
            files.extend(Path(m) for m in matches if Path(m).is_file())
        elif Path(p).is_file():
            files.append(Path(p))

    changed_files: list[Path] = []
    for f in files:
        if process_file(f, write=args.write):
            changed_files.append(f)

    if changed_files:
        print("Changed:")
        for f in changed_files:
            print(f"  {f}")
    else:
        print("No changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
