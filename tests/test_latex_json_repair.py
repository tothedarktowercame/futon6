"""LaTeX commands delivered as control characters are restored, or refused.

The raw cases are the S4 answers saved on apm-501-xi-r1 (2026-10-16), byte
for byte; ⇥ there is a TAB.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import latex_json_repair as R  # noqa: E402


def fill(text: str) -> str:
    return '{"fill": "' + text + '"}'


@pytest.mark.parametrize("damaged, restored", [
    ("$f \tsto \tint_{0}^{1} f g$", r"$f \mapsto \int_{0}^{1} f g$"),
    ("$S^{2} \to \tathbb{RP}^{2}$", r"$S^{2} \to \mathbb{RP}^{2}$"),
    ("$|| \tightarrow\tinfty \tleq M$", r"$|| \rightarrow\infty \leq M$"),
    ("$C_{n} \tubseteq U$", r"$C_{n} \subseteq U$"),
    ("$g = \tchi_{[-h,h]}$", r"$g = \chi_{[-h,h]}$"),
    ("the $L^{p} \times L^{p}$ norm", r"the $L^{p} \times L^{p}$ norm"),
    ("on a \tph{dense} subset", r"on a \emph{dense} subset"),
])
def test_raw_bytes_observed_on_apm_are_restored(damaged, restored):
    raw = fill(damaged)
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    doc, sanitized, repairs = R.parse(raw)
    assert doc["fill"] == restored
    assert sanitized is not None and repairs


def test_ambiguity_is_recorded():
    _doc, _s, repairs = R.parse(fill("$x \tightarrow y$"))
    assert repairs[0]["restored-command"] == r"\rightarrow"
    assert repairs[0]["alternatives"] == [r"\Rightarrow"]


def test_a_paper_macro_is_restored_only_when_the_source_uses_it():
    raw = fill("$S_{3}, \bbZ/{2}$")
    with pytest.raises(json.JSONDecodeError):
        R.parse(raw)                                    # \bbZ is no standard command
    doc, _s, _r = R.parse(raw, source=r"\newcommand{\bbZ}{\mathbb{Z}} ... $\bbZ/2$")
    assert doc["fill"] == r"$S_{3}, \bbZ/{2}$"


def test_a_lost_command_with_no_letters_left_is_refused():
    with pytest.raises(json.JSONDecodeError):
        R.parse(fill("$({0},{1}) \t({2},{3})$"))


def test_decoded_escapes_become_their_commands():
    raw = json.dumps({"a": "$\theta + \beta$", "b": "$\nabla f \neq 0$", "c": "$\frac{1}{2}$"})
    doc, sanitized, repairs = R.parse(raw)
    assert sanitized is None                            # it parsed; nothing raw to repair
    assert doc == {"a": r"$\theta + \beta$", "b": r"$\nabla f \neq 0$", "c": r"$\frac{1}{2}$"}
    assert all(r["decoded"] for r in repairs) and len(repairs) == 5


def test_line_breaks_in_prose_are_left_alone():
    doc, _s, repairs = R.parse(json.dumps({"a": "first line\nin the proof", "b": "x\nu y"}))
    assert doc == {"a": "first line\nin the proof", "b": "x\nu y"} and repairs == []
    with pytest.raises(json.JSONDecodeError):          # a raw line break in prose is not \in
        R.parse('{"a": "first line\nin the proof"}')


def test_unknown_decoded_word_is_left_for_the_stage_to_refuse():
    doc, _s, repairs = R.parse(json.dumps({"a": "two\twords"}))
    assert doc["a"] == "two\twords" and repairs == []
