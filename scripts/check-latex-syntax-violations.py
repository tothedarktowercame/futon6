#!/usr/bin/env python3
r"""Detect math syntax-class violations in generated LaTeX.

Targets:
- Bare Greek names (`pi`, `psi`, ...) not written as TeX commands.
- Bare `diag(` instead of `\operatorname{diag}(`.
- Bare `GL...` group tokens instead of `\mathup{GL}_{...}` (or similar).
- Bare infix operators (`in`, `x`, `*`) between math-like tokens.
- Non-math prose leaks such as `P in R` and `qF^{-s}`.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MATH_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "aligned",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "split",
    "math",
    "displaymath",
    "array",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "cases",
}

BEGIN_ENV_RE = re.compile(r"\\begin\{([^}]+)\}")
END_ENV_RE = re.compile(r"\\end\{([^}]+)\}")

GREEK_NAMES = (
    "Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Pi|"
    "Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega|"
    "alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|"
    "lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega"
)

MATH_BARE_GREEK_RE = re.compile(rf"(?<!\\)\b(?:{GREEK_NAMES})\b")
MATH_BARE_DIAG_RE = re.compile(r"(?<!\\)\bdiag\s*\(")
MATH_BARE_GL_RE = re.compile(r"(?<![\\{])\bGL(?:_\{[^}]+\}|_[A-Za-z0-9+\-]+|n[+\-]?\d*)\b")
MATH_BARE_OPERATOR_WORD_RE = re.compile(r"(?<!\\)\b(?:sum|prod|lim)\b")
MATH_BARE_IN_RE = re.compile(r"([A-Za-z\\][A-Za-z0-9_{}\\^]*)\s+in\s+([A-Za-z\\][A-Za-z0-9_{}\\^]*)")
MATH_BARE_STAR_RE = re.compile(r"([A-Za-z0-9}\\)])\s*\*\s*([A-Za-z0-9\\{(])")
MATH_X_CAND_RE = re.compile(r"(\S+)\s+x\s+(\S+)")

NONMATH_SINGLE_IN_RE = re.compile(r"\b([A-Za-z])\s+in\s+([A-Za-z])\b")
NONMATH_QF_RE = re.compile(r"\bqF(?:\^\{[^}]+\}|\^-?[A-Za-z0-9]+)\b")
NONMATH_DIAG_RE = re.compile(r"\bdiag\s*\(")
NONMATH_GL_X_RE = re.compile(
    r"\bGL(?:_\{?[A-Za-z0-9+\-]+\}?|n[+\-]?\d*)\s*[xX]\s*GL(?:_\{?[A-Za-z0-9+\-]+\}?|n[+\-]?\d*)\b"
)
NONMATH_OPERATOR_SCRIPT_RE = re.compile(
    r"\b(?:sum|prod|lim)\s*(?:\\_|_|\\\^|\^)\s*(?:\\?\{[^}]*|[A-Za-z0-9])"
)
NONMATH_ESCAPED_SCRIPT_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*\\[_^](?:\\?\{|[A-Za-z0-9])"
)
NONMATH_GREEK_SCRIPT_RE = re.compile(
    rf"\b(?:{GREEK_NAMES})(?:_\{{[^}}]+\}}|_[A-Za-z0-9]+|\^\{{[^}}]+\}}|\^[A-Za-z0-9]+|\([^)]+\))"
)

MATH_WORD_STOP = {
    "and",
    "or",
    "for",
    "all",
    "the",
    "is",
    "are",
    "mod",
    "let",
}


@dataclass
class MathState:
    in_display_bracket: bool = False
    in_display_dollar: bool = False
    in_inline_paren: bool = False
    in_inline_dollar: bool = False
    math_env_depth: int = 0


@dataclass
class Finding:
    path: Path
    line: int
    scope: str
    kind: str
    message: str


def expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        matches = sorted(glob.glob(p, recursive=True))
        if matches:
            out.extend(Path(m) for m in matches if Path(m).is_file())
        elif Path(p).is_file():
            out.append(Path(p))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def strip_comments(line: str) -> str:
    out: list[str] = []
    escaped = False
    for ch in line:
        if ch == "%" and not escaped:
            break
        out.append(ch)
        escaped = ch == "\\" and not escaped
        if ch != "\\":
            escaped = False
    return "".join(out)


def split_math_nonmath(line: str, st: MathState) -> tuple[str, str]:
    math_out: list[str] = []
    nonmath_out: list[str] = []
    i = 0
    n = len(line)

    while i < n:
        mb = BEGIN_ENV_RE.match(line, i)
        if mb:
            if mb.group(1) in MATH_ENVS:
                st.math_env_depth += 1
            i = mb.end()
            continue

        me = END_ENV_RE.match(line, i)
        if me:
            if me.group(1) in MATH_ENVS and st.math_env_depth > 0:
                st.math_env_depth -= 1
            i = me.end()
            continue

        in_math = (
            st.math_env_depth > 0
            or st.in_display_bracket
            or st.in_display_dollar
            or st.in_inline_paren
            or st.in_inline_dollar
        )

        if st.in_display_bracket and line.startswith(r"\]", i):
            st.in_display_bracket = False
            i += 2
            continue
        if st.in_display_dollar and line.startswith("$$", i):
            st.in_display_dollar = False
            i += 2
            continue
        if st.in_inline_paren and line.startswith(r"\)", i):
            st.in_inline_paren = False
            i += 2
            continue
        if st.in_inline_dollar and line[i] == "$" and (i == 0 or line[i - 1] != "\\"):
            st.in_inline_dollar = False
            i += 1
            continue

        if not in_math:
            if line.startswith(r"\[", i):
                st.in_display_bracket = True
                i += 2
                continue
            if line.startswith("$$", i):
                st.in_display_dollar = True
                i += 2
                continue
            if line.startswith(r"\(", i):
                st.in_inline_paren = True
                i += 2
                continue
            if line[i] == "$" and (i == 0 or line[i - 1] != "\\"):
                st.in_inline_dollar = True
                i += 1
                continue

        ch = line[i]
        if in_math:
            math_out.append(ch)
            nonmath_out.append(" ")
        else:
            nonmath_out.append(ch)
            math_out.append(" ")
        i += 1

    return "".join(math_out), "".join(nonmath_out)


def collect_findings(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    st = MathState()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for lineno, raw in enumerate(lines, 1):
        if "\\texorpdfstring{" in raw or raw.lstrip().startswith("\\hypertarget{"):
            continue
        line = strip_comments(raw)
        math_text, nonmath_text = split_math_nonmath(line, st)

        if MATH_BARE_GREEK_RE.search(math_text):
            findings.append(Finding(path, lineno, "math", "bare-greek", "bare Greek name in math"))
        if MATH_BARE_DIAG_RE.search(math_text):
            findings.append(Finding(path, lineno, "math", "bare-diag", "diag(...) should be \\operatorname{diag}(...)"))
        if MATH_BARE_GL_RE.search(math_text):
            findings.append(Finding(path, lineno, "math", "bare-gl", "GL token should be \\mathup{GL}_{...}"))
        if MATH_BARE_OPERATOR_WORD_RE.search(math_text):
            findings.append(Finding(path, lineno, "math", "bare-operator-word", "bare operator word in math"))
        if MATH_BARE_IN_RE.search(math_text):
            findings.append(Finding(path, lineno, "math", "bare-in", "bare 'in' should be \\in"))
        if has_bare_x(math_text):
            findings.append(Finding(path, lineno, "math", "bare-x", "bare 'x' should be \\times"))
        if MATH_BARE_STAR_RE.search(math_text):
            findings.append(Finding(path, lineno, "math", "bare-star", "bare '*' should be \\ast"))

        if NONMATH_SINGLE_IN_RE.search(nonmath_text):
            findings.append(Finding(path, lineno, "text", "single-in", "single-letter membership should be math"))
        if NONMATH_QF_RE.search(nonmath_text):
            findings.append(Finding(path, lineno, "text", "qf-token", "qF token should be q_F in math"))
        if NONMATH_DIAG_RE.search(nonmath_text):
            findings.append(Finding(path, lineno, "text", "diag-text", "diag(...) appears outside math"))
        if NONMATH_GL_X_RE.search(nonmath_text):
            findings.append(Finding(path, lineno, "text", "gl-x-text", "GL... x GL... should be math with \\times"))
        if NONMATH_OPERATOR_SCRIPT_RE.search(nonmath_text):
            findings.append(
                Finding(path, lineno, "text", "operator-script-text", "operator token with script appears outside math")
            )
        if NONMATH_ESCAPED_SCRIPT_RE.search(nonmath_text):
            findings.append(
                Finding(path, lineno, "text", "escaped-script-text", "escaped _/^ token appears outside math")
            )
        if NONMATH_GREEK_SCRIPT_RE.search(nonmath_text):
            findings.append(
                Finding(path, lineno, "text", "greek-script-text", "Greek/script token appears outside math")
            )

    return findings


def _strip_edge(tok: str) -> str:
    tok = tok.strip("()[]{}.,;:")
    return tok


def _is_relation(tok: str) -> bool:
    return tok in {
        r"\in",
        r"\notin",
        r"\le",
        r"\leq",
        r"\ge",
        r"\geq",
        r"\neq",
        r"\subseteq",
        r"\approx",
        r"\to",
        r"\leftarrow",
        r"\Rightarrow",
        r"\mapsto",
        r"\leftrightarrow",
        r"\Leftrightarrow",
    }


def _is_mathish(tok: str) -> bool:
    if not tok:
        return False
    if _is_relation(tok):
        return False
    if re.search(r"[\\_^{}0-9]", tok):
        return True
    if re.fullmatch(r"[A-Za-z]", tok):
        return tok.lower() not in MATH_WORD_STOP
    if re.fullmatch(r"[A-Za-z]{2,3}", tok):
        return tok.lower() not in MATH_WORD_STOP
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*\([^)]*\)", tok):
        return True
    return False


def _is_mul_side(tok: str) -> bool:
    if not tok or _is_relation(tok):
        return False
    if tok in {r"\ast", r"\times", "+", "-", "=", r"\cdot"}:
        return False
    if re.search(r"[\\_^{}0-9]", tok):
        return True
    if re.fullmatch(r"[nmrqNMRQ]", tok):
        return True
    if re.fullmatch(r"(?:nr|rn|nm|mn|qr|rq|mr|rm|nM|Nm|nR|Nr|nQ|Nq)", tok):
        return True
    return False


def has_bare_x(math_text: str) -> bool:
    for m in MATH_X_CAND_RE.finditer(math_text):
        left = _strip_edge(m.group(1))
        right = _strip_edge(m.group(2))
        if _is_mul_side(left) and _is_mul_side(right):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        default=["data/first-proof/latex/full/problem*-solution-full.tex"],
        help="files/globs to scan",
    )
    ap.add_argument("--max-findings", type=int, default=500, help="truncate report output")
    ap.add_argument("--summary-only", action="store_true", help="print only kind counts")
    args = ap.parse_args()

    files = expand(args.paths)
    if not files:
        print("No files matched.", file=sys.stderr)
        return 1

    all_findings: list[Finding] = []
    for path in files:
        all_findings.extend(collect_findings(path))

    if not all_findings:
        print("No syntax-class violations found.")
        return 0

    if not args.summary_only:
        for f in all_findings[: args.max_findings]:
            print(f"{f.path}:{f.line}: [{f.scope}:{f.kind}] {f.message}")
        if len(all_findings) > args.max_findings:
            print(
                f"... truncated {len(all_findings) - args.max_findings} additional findings "
                f"(max-findings={args.max_findings})"
            )
    counts = Counter(f"{f.scope}:{f.kind}" for f in all_findings)
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{key}: {count}", file=sys.stderr)
    print(f"Total findings: {len(all_findings)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
