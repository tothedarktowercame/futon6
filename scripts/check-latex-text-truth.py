#!/usr/bin/env python3
"""Lightweight linter for LaTeX prose/math boundary issues.

Checks (high-signal):
- Unescaped '^' or '_' outside math mode.
- Unicode math symbols outside math mode (warning-level).
- Suspicious math-like content inside \texttt{...}.
- Section titles containing '^'/'_' outside math mode.

Exit code:
- 0: no blocking errors (and no warnings in --strict mode)
- 1: at least one blocking error (or warning in --strict mode)
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

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

VERBATIM_ENVS = {
    "verbatim",
    "lstlisting",
    "minted",
    "Verbatim",
}

UNICODE_MATH = set("√∞≤≥≠→←⊞∧∩⊂⊕⊗πΓΦμλωχθ")

TEXTTT_MATHY_RE = re.compile(
    r"(_|\^|=|<=|>=|->|\\to|\\le|\\ge|\\neq|\\in|\\rtimes|\\otimes|\\oplus)"
)

PATHY_RE = re.compile(r"(/|\.md$|\.py$|\.json$|\.jsonl$)")
BEGIN_ENV_RE = re.compile(r"\\begin\{([^}]+)\}")
END_ENV_RE = re.compile(r"\\end\{([^}]+)\}")
TEXTTT_RE = re.compile(r"\\texttt\{([^{}]*)\}")
TITLE_RE = re.compile(r"\\(?:sub)*section\*?\{([^{}]*)\}")


@dataclass
class Finding:
    path: Path
    line: int
    col: int
    kind: str
    message: str
    severity: str  # error|warn


def strip_comments(line: str) -> str:
    out = []
    escaped = False
    for ch in line:
        if ch == "%" and not escaped:
            break
        out.append(ch)
        escaped = (ch == "\\" and not escaped)
        if ch != "\\":
            escaped = False
    return "".join(out)


def _mask_braced_arg(line: str, cmd: str) -> str:
    pat = re.compile(rf"\\{cmd}\{{[^{{}}]*\}}")

    def repl(m: re.Match[str]) -> str:
        return " " * (m.end() - m.start())

    return pat.sub(repl, line)


def mask_nonmath_spans(line: str) -> str:
    masked = line
    for cmd in ["label", "ref", "cref", "eqref", "url", "path", "texttt"]:
        masked = _mask_braced_arg(masked, cmd)

    masked = re.sub(
        r"\\href\{[^{}]*\}\{[^{}]*\}",
        lambda m: " " * (m.end() - m.start()),
        masked,
    )

    masked = re.sub(
        r"\\hypertarget\{[^{}]*\}\{?",
        lambda m: " " * (m.end() - m.start()),
        masked,
    )
    masked = re.sub(
        r"\\texorpdfstring\{[^{}]*\}\{[^{}]*\}",
        lambda m: " " * (m.end() - m.start()),
        masked,
    )
    return masked


def has_unmath_caret_or_underscore(title: str) -> bool:
    # Ignore texorpdfstring payloads in headings; fallback text often includes
    # plain underscores for bookmark strings.
    title = re.sub(r"\\texorpdfstring\{[^{}]*\}\{[^{}]*\}", "", title)

    in_math = False
    for i, ch in enumerate(title):
        if ch == "$" and not (i > 0 and title[i - 1] == "\\"):
            in_math = not in_math
            continue
        if ch in {"^", "_"} and not in_math and not (i > 0 and title[i - 1] == "\\"):
            return True
    return False



class State:
    def __init__(self) -> None:
        self.env_stack: List[str] = []
        self.inline_dollar = False
        self.display_dollar = False
        self.inline_paren = False  # \( ... \)
        self.display_bracket = False  # \[ ... \]

    def in_verbatim(self) -> bool:
        return any(env in VERBATIM_ENVS for env in self.env_stack)

    def in_math(self) -> bool:
        if self.inline_dollar or self.display_dollar or self.inline_paren or self.display_bracket:
            return True
        return any(env in MATH_ENVS for env in self.env_stack)


def scan_file(path: Path, warn_unicode: bool = False) -> List[Finding]:
    findings: List[Finding] = []
    st = State()

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for lineno, raw in enumerate(lines, start=1):
        line = strip_comments(raw)
        line_scan = mask_nonmath_spans(line)

        for tm in TITLE_RE.finditer(line):
            title = tm.group(1)
            if has_unmath_caret_or_underscore(title):
                findings.append(
                    Finding(
                        path,
                        lineno,
                        tm.start(1) + 1,
                        "title-math-char-outside-math",
                        "section title has '^' or '_' outside math mode",
                        "error",
                    )
                )

        for m in TEXTTT_RE.finditer(line):
            payload = m.group(1)
            if PATHY_RE.search(payload):
                continue
            if TEXTTT_MATHY_RE.search(payload):
                findings.append(
                    Finding(
                        path,
                        lineno,
                        m.start(1) + 1,
                        "texttt-mathy",
                        f"math-like token in \\texttt{{...}}: {payload}",
                        "warn",
                    )
                )

        i = 0
        while i < len(line_scan):
            mb = BEGIN_ENV_RE.match(line_scan, i)
            if mb:
                st.env_stack.append(mb.group(1))
                i = mb.end()
                continue
            me = END_ENV_RE.match(line_scan, i)
            if me:
                env = me.group(1)
                if env in st.env_stack:
                    for j in range(len(st.env_stack) - 1, -1, -1):
                        if st.env_stack[j] == env:
                            del st.env_stack[j]
                            break
                i = me.end()
                continue

            if st.in_verbatim():
                i += 1
                continue

            if line_scan.startswith(r"\(", i):
                st.inline_paren = True
                i += 2
                continue
            if line_scan.startswith(r"\)", i):
                st.inline_paren = False
                i += 2
                continue
            if line_scan.startswith(r"\[", i):
                st.display_bracket = True
                i += 2
                continue
            if line_scan.startswith(r"\]", i):
                st.display_bracket = False
                i += 2
                continue

            ch = line_scan[i]

            if ch == "$":
                if i > 0 and line_scan[i - 1] == "\\":
                    i += 1
                    continue
                if i + 1 < len(line_scan) and line_scan[i + 1] == "$":
                    st.display_dollar = not st.display_dollar
                    i += 2
                    continue
                st.inline_dollar = not st.inline_dollar
                i += 1
                continue

            if not st.in_math():
                if ch in {"^", "_"}:
                    if not (i > 0 and line_scan[i - 1] == "\\"):
                        findings.append(
                            Finding(
                                path,
                                lineno,
                                i + 1,
                                "math-char-outside-math",
                                f"'{ch}' appears outside math mode",
                                "error",
                            )
                        )
                elif warn_unicode and ch in UNICODE_MATH:
                    findings.append(
                        Finding(
                            path,
                            lineno,
                            i + 1,
                            "unicode-math-outside-math",
                            f"unicode math symbol '{ch}' outside math mode",
                            "warn",
                        )
                    )

            i += 1

    if st.inline_dollar or st.display_dollar or st.inline_paren or st.display_bracket:
        findings.append(
            Finding(
                path,
                len(lines),
                1,
                "unbalanced-math-delimiter",
                "math delimiter appears unbalanced at EOF",
                "warn",
            )
        )

    return findings


def expand_inputs(inputs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        matches = sorted(glob.glob(item, recursive=True))
        if matches:
            out.extend(Path(m) for m in matches if Path(m).is_file())
        else:
            p = Path(item)
            if p.is_file():
                out.append(p)
    seen = set()
    uniq: List[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="Check LaTeX prose/math boundary issues")
    ap.add_argument(
        "paths",
        nargs="*",
        default=["data/first-proof/latex/**/*.tex"],
        help="files/globs to scan (default: data/first-proof/latex/**/*.tex)",
    )
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--warn-unicode", action="store_true", help="emit unicode math warnings outside math mode")
    args = ap.parse_args()

    files = expand_inputs(args.paths)
    if not files:
        print("No files matched.", file=sys.stderr)
        return 1

    all_findings: List[Finding] = []
    for p in files:
        all_findings.extend(scan_file(p, warn_unicode=args.warn_unicode))

    errors = [f for f in all_findings if f.severity == "error"]
    warns = [f for f in all_findings if f.severity == "warn"]

    for f in all_findings:
        print(f"{f.path}:{f.line}:{f.col}: {f.severity}: {f.kind}: {f.message}")

    print(
        f"Scanned {len(files)} files | errors={len(errors)} warnings={len(warns)}",
        file=sys.stderr,
    )

    if errors:
        return 1
    if args.strict and warns:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
