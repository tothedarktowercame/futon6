#!/usr/bin/env python3
"""Model answers in JSON whose text fields hold LaTeX.

A JSON string allows only " \\ / b f n r t u after a backslash, and guided
decoding enforces that grammar, so most LaTeX commands cannot be written in
one: the decoder forces a legal letter and the model completes a different
command (S3 measured \\Sigma becoming \\triangle). What reached us on the APM run
(apm-501-xi-r1, 2026-10-16) was worse: a raw TAB byte where \\chi, \\mathbb,
\\cup or \\mapsto belonged, which left the JSON unparseable.

Two steps, one each side of the model:

allow_latex(schema)  gives every free text field the pattern TEXT. With it the
    model writes LaTeX as it would in a paper, one backslash and any command
    (`$g = \\chi_{[-h,h]}$`), and the pattern admits no control character and no
    unescaped quote, so a string can neither carry a raw TAB nor end early.

loads(raw)  parses that. Before json.loads, every backslash that is not a JSON
    escape is doubled, which is how JSON spells a backslash, so it decodes to
    the backslash the model wrote. A JSON escape is \\" \\\\ \\/ \\uXXXX, or
    \\b \\f \\n \\r \\t not followed by another letter. So \\theta, \\frac,
    \\nabla, \\beta, \\rho, \\{ and \\, are LaTeX, and a model that does escape
    properly (\\\\chi) still decodes to \\chi.

Checked against mark4-70b on vLLM 0.10.1.1: S4 regions that had come back
with a TAB for \\chi and \\cup decode to `$g = \\chi_{[-h,h]}$` and
`$({0},{1}) \\cup ({2},{3})$`.
"""
from __future__ import annotations

import json
import re

TEXT = '^([^\\u0000-\\u001f"\\\\]|\\\\[^\\u0000-\\u001f])*$'
# A backslash and what follows it, matched as a unit so that \\ is one escape:
# group 1 is a JSON escape, kept; an empty group 1 is LaTeX's backslash.
BACKSLASH = re.compile(r'\\(["\\/]|u[0-9a-fA-F]{4}|[bfnrt](?![A-Za-z]))?')


def allow_latex(schema):
    """The schema with TEXT on every free string (no enum, const or pattern, not empty)."""
    if isinstance(schema, list):
        return [allow_latex(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: allow_latex(v) for k, v in schema.items()}
    if (out.get("type") == "string" and not {"enum", "const", "pattern"} & out.keys()
            and out.get("maxLength") != 0):
        out["pattern"] = TEXT
    return out


def escape_latex(raw: str) -> str:
    """raw with each backslash that is not a JSON escape doubled, so it decodes to itself."""
    return BACKSLASH.sub(lambda m: m.group(0) if m.group(1) else "\\\\", raw)


def loads(raw: str):
    """json.loads, reading a backslash that is not a JSON escape as LaTeX's."""
    return json.loads(escape_latex(raw))
