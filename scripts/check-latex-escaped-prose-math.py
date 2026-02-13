#!/usr/bin/env python3
r"""Flag likely prose math that is not typeset in math mode.

Heuristics on non-math text spans:
- escaped math markers (`\^{}` / `\_`)
- bare inequalities (`<=`, `>=`, `!=`)
- bare complexity terms (`O(...)`)
- Greek/script-like symbol tokens (e.g., `lambda_{abgd}`, `Lambda(2)`)
- comma-separated variable lists (`a,b,g,d`)
- short equation snippets (`y = A_tau x`)
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ESCAPED_MARKER_RE = re.compile(r"\\\^\{\}|\\_")
GREEK_BARE_RE = re.compile(
    r"\b(?:alpha|beta|gamma|delta|lambda|mu|nu|pi|sigma|tau|phi|psi|omega|theta|kappa|chi)\b(?=\s+(?:in|=|is)\b)",
)
GREEK_SCRIPT_BARE_RE = re.compile(
    r"\b(?:Alpha|Beta|Gamma|Delta|Theta|Lambda|Pi|Sigma|Phi|Omega|"
    r"alpha|beta|gamma|delta|theta|lambda|pi|sigma|tau|phi|chi|omega|mu|kappa|psi)"
    r"(?:_\{[^}]+\}|_[A-Za-z0-9]+|\^\{[^}]+\}|\^[A-Za-z0-9]+|\([^)]+\))+"
)
VARLIST_BARE_RE = re.compile(r"\b[a-z](?:,[a-z]){2,}\b")
INEQ_BARE_RE = re.compile(
    r"\b[A-Za-z0-9_(){}\\]+(?:\s+[A-Za-z0-9_(){}\\]+){0,4}\s*(?:<=|>=|!=)\s*"
    r"[A-Za-z0-9_(){}\\]+(?:\s+[A-Za-z0-9_(){}\\]+){0,6}\b"
)
BIG_O_BARE_RE = re.compile(r"\bO\([^\n)$]+\)")
DIM_X_BARE_RE = re.compile(r"\(?[A-Za-z0-9]{1,4}\)?\s+[xX]\s+\(?[A-Za-z0-9]{1,4}\)?")
BINARY_OP_CAND_RE = re.compile(r"(?<!\\)([A-Za-z0-9_{}\\^()]+)\s*([+*=<>])\s*([A-Za-z0-9_{}\\^()]+)")
BEGIN_ENV_RE = re.compile(r"\\begin\{([^}]+)\}")
END_ENV_RE = re.compile(r"\\end\{([^}]+)\}")

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


@dataclass
class MathState:
    in_display_bracket: bool = False
    in_display_dollar: bool = False
    math_env_depth: int = 0


def strip_comments(line: str) -> str:
    out = []
    esc = False
    for ch in line:
        if ch == "%" and not esc:
            break
        out.append(ch)
        esc = ch == "\\" and not esc
        if ch != "\\":
            esc = False
    return "".join(out)


def _find_unescaped_dollar(line: str, start: int) -> int:
    i = start
    while True:
        i = line.find("$", i)
        if i < 0:
            return -1
        if i == 0 or line[i - 1] != "\\":
            return i
        i += 1


def strip_math(line: str, st: MathState) -> str:
    out: list[str] = []
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

        if st.math_env_depth > 0:
            i += 1
            continue

        if st.in_display_bracket:
            if line.startswith(r"\]", i):
                st.in_display_bracket = False
                i += 2
            else:
                i += 1
            continue

        if st.in_display_dollar:
            if line.startswith("$$", i):
                st.in_display_dollar = False
                i += 2
            else:
                i += 1
            continue

        if line.startswith(r"\[", i):
            st.in_display_bracket = True
            i += 2
            continue

        if line.startswith("$$", i):
            st.in_display_dollar = True
            i += 2
            continue

        if line.startswith(r"\(", i):
            j = line.find(r"\)", i + 2)
            if j < 0:
                break
            i = j + 2
            continue

        ch = line[i]
        if ch == "$" and (i == 0 or line[i - 1] != "\\"):
            j = _find_unescaped_dollar(line, i + 1)
            if j < 0:
                break
            i = j + 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        m = sorted(glob.glob(p, recursive=True))
        if m:
            out.extend(Path(x) for x in m if Path(x).is_file())
        elif Path(p).is_file():
            out.append(Path(p))
    seen = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _is_math_operand(tok: str) -> bool:
    tok = tok.strip("()[]{}.,;:")
    if not tok:
        return False
    if tok.lower() in {
        "a",
        "an",
        "and",
        "are",
        "by",
        "for",
        "from",
        "if",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "over",
        "the",
        "then",
        "to",
        "under",
        "where",
        "with",
    }:
        return False
    if re.search(r"[0-9\\_^]", tok):
        return True
    if re.fullmatch(r"[A-Za-z]", tok):
        return True
    if re.fullmatch(r"[A-Z]{2,}[A-Za-z0-9]*", tok):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", tok) and not tok.islower():
        return True
    return False


def has_binary_op_outside_math(line: str) -> bool:
    for m in BINARY_OP_CAND_RE.finditer(line):
        left, _op, right = m.group(1), m.group(2), m.group(3)
        if _is_math_operand(left) and _is_math_operand(right):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        default=["data/first-proof/latex/full/problem*-solution-full.tex"],
    )
    ap.add_argument("--min-markers", type=int, default=2)
    args = ap.parse_args()

    files = expand(args.paths)
    if not files:
        print("No files matched.", file=sys.stderr)
        return 1

    findings = 0
    for path in files:
        st = MathState()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, raw in enumerate(lines, 1):
            if "\\texorpdfstring{" in raw or raw.lstrip().startswith("\\hypertarget{"):
                continue

            line = strip_math(strip_comments(raw), st)
            if not line.strip():
                continue

            markers = len(ESCAPED_MARKER_RE.findall(line))
            if markers >= args.min_markers:
                findings += 1
                print(f"{path}:{lineno}: likely escaped prose-math ({markers} markers)")
                continue

            if GREEK_BARE_RE.search(line):
                findings += 1
                print(f"{path}:{lineno}: bare Greek variable name likely outside math mode")

            if GREEK_SCRIPT_BARE_RE.search(line):
                findings += 1
                print(f"{path}:{lineno}: Greek/script token likely outside math mode")

            if VARLIST_BARE_RE.search(line):
                findings += 1
                print(f"{path}:{lineno}: comma-separated variable list likely outside math mode")

            if INEQ_BARE_RE.search(line):
                findings += 1
                print(f"{path}:{lineno}: inequality likely outside math mode")

            if BIG_O_BARE_RE.search(line):
                findings += 1
                print(f"{path}:{lineno}: Big-O term likely outside math mode")


            if DIM_X_BARE_RE.search(line):
                findings += 1
                print(f"{path}:{lineno}: dimension expression with 'x' likely outside math mode")

            if has_binary_op_outside_math(line):
                findings += 1
                print(f"{path}:{lineno}: binary operator expression likely outside math mode")

    if findings:
        print(f"Found {findings} likely prose-math lines.", file=sys.stderr)
        return 1

    print("No likely escaped prose-math lines found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
