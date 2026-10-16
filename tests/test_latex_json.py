"""LaTeX in model JSON: the schema lets the model write it, loads reads it."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import latex_json  # noqa: E402


@pytest.mark.parametrize("raw, text", [
    # What mark4-70b wrote for the APM regions that had come back with a TAB.
    (r'{"a": "$g = \chi_{[-h,h]}$"}', r"$g = \chi_{[-h,h]}$"),
    (r'{"a": "$({0},{1}) \cup ({2},{3})$"}', r"$({0},{1}) \cup ({2},{3})$"),
    (r'{"a": "$f^{\prime}$"}', r"$f^{\prime}$"),
    # Commands that begin with a JSON escape letter are LaTeX too.
    (r'{"a": "$\theta \frac{1}{2} \nabla f \beta \rho$"}', r"$\theta \frac{1}{2} \nabla f \beta \rho$"),
    (r'{"a": "$\{x\}\, \!$"}', r"$\{x\}\, \!$"),
    # Real JSON escapes keep their meaning, and \\ is one escape, not two.
    (r'{"a": "\\chi"}', r"\chi"),
    (r'{"a": "say \"hi\"\n\t x é \/"}', 'say "hi"\n\t x é /'),
    ('{"a": "caf\\u00e9"}', "caf\u00e9"),        # a unicode escape stays one
    # The price: an escape letter followed by a word reads as a command, so a
    # line break written straight before a word ("\nThen") comes back as \nThen.
    (r'{"a": "one\nThen"}', r"one\nThen"),
])
def test_loads_reads_latex_backslashes(raw, text):
    assert latex_json.loads(raw)["a"] == text


def test_text_pattern_admits_latex_and_refuses_control_characters_and_bare_quotes():
    rx = re.compile(latex_json.TEXT.replace("\\u0000", "\x00").replace("\\u001f", "\x1f"))
    assert rx.fullmatch(r"$g = \chi_{[-h,h]}$ and \"quoted\"")
    assert not rx.fullmatch("a\tb")              # the raw TAB that broke the APM answers
    assert not rx.fullmatch('ends "early')       # an unescaped quote would close the string


def test_allow_latex_marks_only_free_text():
    schema = latex_json.allow_latex({"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["a"]},
        "text": {"type": "string", "maxLength": 300},
        "held": {"type": "string", "maxLength": 0},
        "id": {"type": "string", "pattern": "^L[0-9]+$"},
        "n": {"type": "integer"}}})
    props = schema["properties"]
    assert props["text"]["pattern"] == latex_json.TEXT
    assert "pattern" not in props["kind"] and "pattern" not in props["held"]
    assert props["id"]["pattern"] == "^L[0-9]+$" and "pattern" not in props["n"]
    json.dumps(schema)
