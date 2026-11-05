#!/usr/bin/env python3
"""Highlight fenced Lean blocks in LaTeXML HTML with the proof palette.

LaTeXML deliberately receives Pandoc's plain verbatim blocks because it does
not understand Pandoc's Highlighting environment.  Run this after LaTeXML;
Pygments supplies the Lean lexer and this adapter maps its tokens onto the
same semantic colours as math-proofread-style.sty.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment, Generic, Keyword, Name, Number, String


CSS = """
<style id="lean-proofread-highlighting">
pre.lean-highlight { background:#0A1830; border:1px solid #243A5A;
  border-left:3px solid #AD5CFF; border-radius:4px;
  padding:1em; overflow-x:auto; line-height:1.35; color:#E6EDF7; }
pre.lean-highlight code { font-family:DejaVu Sans Mono,monospace; }
.lean-kw { color:#1A0DF5; font-weight:600; }
.lean-decl { color:#AD5CFF; font-weight:600; }
.lean-num { color:#FF0000; }
.lean-delim { color:#FF00FF; }
.lean-op { color:#4FFF80; font-weight:600; }
.lean-rel { color:#004225; font-weight:600; }
.lean-comment { color:#A7B4C8; font-style:italic; }
.lean-string { color:#A314FA; }
.lean-error { color:#FF0000; text-decoration:underline wavy; }
</style>
""".strip()

DELIMITERS = set("()[]{}⟨⟩‹›⦃⦄,:")
RELATIONS = {"=", "≠", "<", ">", "≤", "≥", "∈", "∉", "⊂", "⊆", "⊃", "⊇", "→", "↦", ":="}
OPERATORS = {"+", "-", "*", "/", "^", "∑", "∫", "↑", "|", "‖", "'"}


def css_class(token, text: str) -> str | None:
    if token in Generic.Error:
        return "lean-error"
    if token in Comment or token in String.Doc:
        return "lean-comment"
    if token in String:
        return "lean-string"
    if token in Number:
        return "lean-num"
    if token in Keyword.Declaration or token in Keyword.Namespace:
        return "lean-decl"
    if token in Keyword:
        return "lean-kw"
    if text in RELATIONS:
        return "lean-rel"
    if text in DELIMITERS:
        return "lean-delim"
    if text in OPERATORS:
        return "lean-op"
    return None


def highlight(code: str) -> str:
    parts: list[str] = []
    for token, text in lex(code, get_lexer_by_name("lean")):
        escaped = html.escape(text, quote=False)
        cls = css_class(token, text)
        parts.append(f'<span class="{cls}">{escaped}</span>' if cls else escaped)
    return '<pre class="ltx_verbatim ltx_font_typewriter lean-highlight"><code>' + "".join(parts) + "</code></pre>"


def lean_blocks(markdown: str) -> list[str]:
    return re.findall(r"^```lean\s*\n(.*?)^```\s*$", markdown, flags=re.MULTILINE | re.DOTALL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("html", type=Path)
    args = parser.parse_args()

    document = args.html.read_text()
    blocks = lean_blocks(args.markdown.read_text())
    replaced = 0
    for code in blocks:
        expected = code.rstrip("\n")
        pattern = re.compile(r'<pre class="ltx_verbatim ltx_font_typewriter">(.*?)</pre>', re.DOTALL)
        match = next((m for m in pattern.finditer(document)
                      if html.unescape(m.group(1)).strip("\n").replace("’", "'") == expected), None)
        if match is None:
            raise SystemExit(f"could not locate Lean block {replaced + 1} in {args.html}")
        document = document[:match.start()] + highlight(expected) + document[match.end():]
        replaced += 1

    if replaced and 'id="lean-proofread-highlighting"' not in document:
        document = document.replace("</head>", CSS + "\n</head>", 1)
    args.html.write_text(document)
    print(f"highlighted {replaced} Lean block(s) in {args.html}")


if __name__ == "__main__":
    main()
