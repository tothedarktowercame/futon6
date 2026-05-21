"""Tiny LaTeX math AST parser for symbol grounding Layer 2.

Scope is deliberately narrow: parse math content between math envelopes
(inline `$...$`, display `$$...$$`, `\\(...\\)`, `\\[...\\]`, math
environments like `\\begin{equation}...\\end{equation}`) into a tree of
nodes that bounds macro arguments, subscripts, superscripts, and grouped
sub-expressions.

This is NOT a general LaTeX parser. It handles enough to give the
scope-tree builder argument-bounded macro scopes (so `\\Hom{A}{B}` produces
one outer scope containing two child argument scopes) instead of the
point-token scopes Layer 1's regex produces.

If pylatexenc lands in the environment later, swap this out — the AST
node shape is intentionally compatible: each node has start, end, kind,
text, name (for macros), and args (list of child node groups).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class MathNode:
    """A node in the math AST.

    Attributes:
        start, end: absolute positions in the ORIGINAL outer text
        kind: one of 'chars', 'macro', 'group', 'sub', 'sup', 'envelope'
        text: raw source spanning [start, end)
        name: macro name for kind='macro' (e.g., 'Hom' for `\\Hom`), else None
        args: list of MacroArg dicts, each with start, end, interior, nodes
        envelope_kind: for kind='envelope', one of 'inline', 'display',
            'paren', 'bracket', 'environment'
    """
    start: int
    end: int
    kind: str
    text: str
    name: str | None = None
    args: list = field(default_factory=list)
    envelope_kind: str | None = None


# Math-envelope detection
# - inline `$...$` (not `$$`, and not preceded by `\` for escape)
# - display `$$...$$`
# - paren `\(...\)`
# - bracket `\[...\]`
# - environment `\begin{equation/align/gather/multline/displaymath/eqnarray/math}...\end{...}`
#
# We scan envelopes in a single pass with a stateful tokenizer so `$$` is
# not mis-parsed as two inline `$` tokens, and so escaped `\$` is skipped.

_MATH_ENV_NAMES = (
    "equation", "align", "gather", "multline", "displaymath",
    "eqnarray", "math", "alignat", "flalign",
)
_MATH_ENV_BEGIN_RE = re.compile(
    r"\\begin\{(" + "|".join(_MATH_ENV_NAMES) + r")\*?\}",
)


def find_math_envelopes(text: str) -> Iterator[tuple[int, int, int, int, str]]:
    """Yield (envelope_start, envelope_end, interior_start, interior_end, kind)
    for each math envelope in `text`.

    `kind` is one of 'inline', 'display', 'paren', 'bracket', 'environment'.
    Positions are absolute in `text`. interior_start/interior_end bound the
    inner math content (without the delimiters).
    """
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            # Could be \(, \[, or \begin{env}, or just an escaped char.
            if i + 1 < n and text[i + 1] == "(":
                close = text.find(r"\)", i + 2)
                if close != -1:
                    yield i, close + 2, i + 2, close, "paren"
                    i = close + 2
                    continue
            if i + 1 < n and text[i + 1] == "[":
                close = text.find(r"\]", i + 2)
                if close != -1:
                    yield i, close + 2, i + 2, close, "bracket"
                    i = close + 2
                    continue
            m = _MATH_ENV_BEGIN_RE.match(text, i)
            if m:
                env_name = m.group(1)
                # Find matching \end{<env>} or \end{<env>*}
                end_pattern = re.compile(
                    r"\\end\{" + re.escape(env_name) + r"\*?\}",
                )
                end_match = end_pattern.search(text, m.end())
                if end_match:
                    yield m.start(), end_match.end(), m.end(), end_match.start(), "environment"
                    i = end_match.end()
                    continue
            i += 2  # skip escaped char
            continue
        if c == "$":
            # Possibly $$ display or $ inline
            if i + 1 < n and text[i + 1] == "$":
                # Display $$...$$
                close = text.find("$$", i + 2)
                if close != -1:
                    yield i, close + 2, i + 2, close, "display"
                    i = close + 2
                    continue
                # Unbalanced, advance
                i += 2
                continue
            # Inline $...$ — find next unescaped $
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if text[j] == "$":
                    break
                if text[j] == "\n":
                    # bail — inline math doesn't usually span newlines
                    break
                j += 1
            if j < n and text[j] == "$":
                yield i, j + 1, i + 1, j, "inline"
                i = j + 1
                continue
            i += 1
            continue
        i += 1


_MACRO_NAME_RE = re.compile(r"\\([A-Za-z]+|.)")


def _find_matching_brace(text: str, open_pos: int) -> int | None:
    """Given text[open_pos] == '{', return index of matching '}', or None."""
    if open_pos >= len(text) or text[open_pos] != "{":
        return None
    depth = 1
    i = open_pos + 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def parse_math(interior: str, *, base_offset: int = 0) -> list[MathNode]:
    """Parse math content (the text BETWEEN math delimiters) into nodes.

    `base_offset` is the absolute position in the outer text where this
    interior began (so emitted nodes have absolute positions).
    """
    nodes: list[MathNode] = []
    i = 0
    n = len(interior)
    while i < n:
        c = interior[i]
        if c == "\\":
            m = _MACRO_NAME_RE.match(interior, i)
            if not m:
                i += 1
                continue
            name = m.group(1)
            macro_start = i
            cursor = i + len(m.group(0))
            args: list[dict] = []
            # Greedily consume {...} arguments
            while cursor < n and interior[cursor] == "{":
                close = _find_matching_brace(interior, cursor)
                if close is None:
                    break
                arg_interior = interior[cursor + 1:close]
                arg_nodes = parse_math(arg_interior, base_offset=base_offset + cursor + 1)
                args.append({
                    "start": base_offset + cursor,
                    "end": base_offset + close + 1,
                    "interior_start": base_offset + cursor + 1,
                    "interior_end": base_offset + close,
                    "interior": arg_interior,
                    "nodes": arg_nodes,
                })
                cursor = close + 1
            nodes.append(MathNode(
                start=base_offset + macro_start,
                end=base_offset + cursor,
                kind="macro",
                text=interior[macro_start:cursor],
                name=name,
                args=args,
            ))
            i = cursor
        elif c == "{":
            close = _find_matching_brace(interior, i)
            if close is None:
                i += 1
                continue
            group_interior = interior[i + 1:close]
            group_nodes = parse_math(group_interior, base_offset=base_offset + i + 1)
            nodes.append(MathNode(
                start=base_offset + i,
                end=base_offset + close + 1,
                kind="group",
                text=interior[i:close + 1],
                args=[{
                    "start": base_offset + i,
                    "end": base_offset + close + 1,
                    "interior_start": base_offset + i + 1,
                    "interior_end": base_offset + close,
                    "interior": group_interior,
                    "nodes": group_nodes,
                }],
            ))
            i = close + 1
        elif c in ("_", "^"):
            kind = "sub" if c == "_" else "sup"
            script_start = i
            i += 1
            if i >= n:
                break
            if interior[i] == "{":
                close = _find_matching_brace(interior, i)
                if close is None:
                    continue
                arg_interior = interior[i + 1:close]
                arg_nodes = parse_math(arg_interior, base_offset=base_offset + i + 1)
                end = close + 1
            elif interior[i] == "\\":
                m = _MACRO_NAME_RE.match(interior, i)
                if not m:
                    continue
                end = i + len(m.group(0))
                arg_interior = interior[i:end]
                arg_nodes = [MathNode(
                    start=base_offset + i,
                    end=base_offset + end,
                    kind="chars",
                    text=arg_interior,
                )]
            else:
                end = i + 1
                arg_interior = interior[i:end]
                arg_nodes = [MathNode(
                    start=base_offset + i,
                    end=base_offset + end,
                    kind="chars",
                    text=arg_interior,
                )]
            nodes.append(MathNode(
                start=base_offset + script_start,
                end=base_offset + end,
                kind=kind,
                text=interior[script_start:end],
                args=[{
                    "start": base_offset + script_start + 1,
                    "end": base_offset + end,
                    "interior_start": base_offset + i + (1 if interior[i] == "{" else 0),
                    "interior_end": base_offset + end - (1 if interior[i] == "{" else 0),
                    "interior": arg_interior,
                    "nodes": arg_nodes,
                }],
            ))
            i = end
        else:
            char_start = i
            while i < n and interior[i] not in r"\{}_^$":
                i += 1
            if i > char_start:
                nodes.append(MathNode(
                    start=base_offset + char_start,
                    end=base_offset + i,
                    kind="chars",
                    text=interior[char_start:i],
                ))
            else:
                # Defensive: shouldn't reach here, but advance to avoid infinite loop.
                i += 1
    return nodes


def walk_math_ast(nodes: list[MathNode], depth: int = 0) -> Iterator[tuple[MathNode, int]]:
    """Pre-order traversal yielding (node, depth)."""
    for node in nodes:
        yield node, depth
        for arg in node.args:
            yield from walk_math_ast(arg["nodes"], depth + 1)
