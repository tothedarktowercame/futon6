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
}

N_DEP_RE = re.compile(r"\b([A-Z])-(dependent|independent)\b")
WHITTAKER_W_RE = re.compile(r"\bW\(\s*pi\s*,\s*psi\s*\)")
PSI_WHITTAKER_RE = re.compile(r"\bpsi-Whittaker\b")
QF_RE = re.compile(r"\bqF(\^\{[^}]+\}|\^-?[A-Za-z0-9]+)")
STAR_QF_RE = re.compile(r"\b([A-Za-z])\s*\*\s*(q_F(?:\^\{[^}]+\}|\^-?[A-Za-z0-9]+))")
SINGLE_IN_RE = re.compile(r"\b([A-Za-z])\s+in\s+([A-Za-z])\b([.,;:]?)")
GL_TOKEN_RE = r"GL(?:_\{?[A-Za-z0-9+\-]+\}?|[A-Za-z0-9+\-]+)"
GL_MEMBERSHIP_PRODUCT_RE = re.compile(
    rf"\b([A-Za-z])\s+in\s+({GL_TOKEN_RE})\s*[xX]\s*({GL_TOKEN_RE})\b"
)
GL_PRODUCT_RE = re.compile(rf"\b({GL_TOKEN_RE})\s*[xX]\s*({GL_TOKEN_RE})\b")
FIXED_W_RE = re.compile(r"\bfor fixed\s+W\s*=\s*W_0\b")
IDEAL_EQ_RE = re.compile(
    r"\bI\s*=\s*L\(s,\s*Pi\s*x\s*pi\)\s*\*\s*C\[q_F\^s,\s*q_F\^{-s}\]"
)
L_FACTOR_RE = re.compile(r"L\(s,\s*Pi\s*x\s*pi\)")


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


def process_plain_text_segment(s: str) -> str:
    s = _wrap_big_o_balanced(s)
    s = N_DEP_RE.sub(_n_dep_repl, s)
    s = PSI_WHITTAKER_RE.sub(r"$\\psi$-Whittaker", s)
    s = WHITTAKER_W_RE.sub(r"$W(\\pi, \\psi)$", s)
    s = QF_RE.sub(r"q_F\1", s)
    s = STAR_QF_RE.sub(_star_qf_repl, s)
    s = SINGLE_IN_RE.sub(_single_in_repl, s)
    s = GL_MEMBERSHIP_PRODUCT_RE.sub(_gl_membership_product_repl, s)
    s = GL_PRODUCT_RE.sub(_gl_product_repl, s)
    s = re.sub(r"\bover V\b", r"over $V$", s)
    s = FIXED_W_RE.sub(r"for fixed $W = W_0$", s)
    s = IDEAL_EQ_RE.sub(
        r"$I = L(s, \\Pi \\times \\pi)\\,\\ast\\,\\mathbb{C}[q_F^s, q_F^{-s}]$",
        s,
    )
    s = L_FACTOR_RE.sub(r"$L(s, \\Pi \\times \\pi)$", s)
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
            out_lines.append(ln)
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
        ("hence P in R.", r"hence $P \in R$."),
        ("equal to c * qF^{-ks}", r"equal to $c \ast q_F^{-ks}$"),
        ("r in GLn+1 x GLn", r"$r \in \mathup{GL}_{n+1} \times \mathup{GL}_{n}$"),
        ("for GL_{n+1} x GL_n is", r"for $\mathup{GL}_{n+1} \times \mathup{GL}_{n}$ is"),
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
