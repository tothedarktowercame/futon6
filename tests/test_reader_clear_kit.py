from __future__ import annotations

import json
from pathlib import Path

import pytest

from futon6.reader_clear_kit import (
    build_kits,
    prepare_informal,
    replace_leanrefs,
    step_count,
)


def test_replaces_nested_and_empty_leanrefs() -> None:
    text = (r"A \leanref{X_{a{b}}}{target{x}} B \leanref{}{gone} C "
            r"\leanref{outer \leanref{inner}{target}}{target}")
    assert replace_leanrefs(text) == r"A X_{a{b}} B  C outer inner"


@pytest.mark.parametrize("heading", [
    "**Stage 3 — LEAN FORMALIZATION**\n",
    "## Stage 3\n",
])
def test_informal_cut_and_leading_html_comment(heading: str) -> None:
    text = "<!-- generated -->\nKept.\n" + heading + "Dropped.\n"
    assert prepare_informal(text) == ("Kept.\n", True)


def test_informal_without_stage_three_is_not_cut() -> None:
    assert prepare_informal("Whole file.\n") == ("Whole file.\n", False)


def test_step_count_ignores_period_inside_math() -> None:
    assert step_count(r"First \(x. y\) continues. Second") == 2


def _fixture(tmp_path: Path, *, module: str = "problems.p01.lean.Main") -> tuple[Path, Path, Path]:
    head = tmp_path / "head-demo"
    head.mkdir()
    rows = [
        {"module": module, "declaration": "First", "body": r"B \leanref{}{target}",
         "proof": "One. Two"},
        {"module": module, "declaration": "Second", "body": "Body", "proof": "Done"},
    ]
    (head / "render.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    problems = tmp_path / "problems"
    source = problems / "p01"
    source.mkdir(parents=True)
    (source / "problem.md").write_text("Problem\n", encoding="utf-8")
    (source / "informal-solution.md").write_text("Informal\n", encoding="utf-8")
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({
        "schema": "reader-clear.sample/v0",
        "drawn": [{"problem_id": "p01", "module": module}],
    }), encoding="utf-8")
    return head, problems, sample


def test_build_preserves_declaration_file_order(tmp_path: Path) -> None:
    head, problems, sample = _fixture(tmp_path)
    result = build_kits(head, problems, sample, tmp_path / "out")[0]
    assert [item["declaration"] for item in result["declarations"]] == ["First", "Second"]
    assert [item["step_count"] for item in result["declarations"]] == [2, 1]
    rendered = (tmp_path / "out/p01/version-P.paperprinter.md").read_text()
    assert rendered.index("Declaration 1: First") < rendered.index("Declaration 2: Second")
    assert r"\leanref" not in rendered


def test_refuses_drawn_module_without_rows(tmp_path: Path) -> None:
    head, problems, sample = _fixture(tmp_path, module="some.other.module")
    sample.write_text(json.dumps({
        "schema": "reader-clear.sample/v0",
        "drawn": [{"problem_id": "p01", "module": "missing.module"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no render rows for drawn module"):
        build_kits(head, problems, sample, tmp_path / "out")


def test_split_steps_gives_the_counted_sentences():
    from futon6.reader_clear_kit import split_steps, step_count

    proof = "Let \\(x = 1.5 \\). Hence \\(x > 1\\).\nDone"
    steps = split_steps(proof)
    assert steps == ["Let \\(x = 1.5 \\).", "Hence \\(x > 1\\).", "Done"]
    assert step_count(proof) == len(steps)


def test_informal_lean_block_is_omitted():
    from futon6.reader_clear_kit import prepare_informal
    text = "Intro.\n\n```lean4\ntheorem t : Icc a b = s := by\n  simp\n```\n\nProof prose.\n\n```\nplain block\n```\n"
    out, cut = prepare_informal(text)
    assert "theorem t" not in out and "Icc" not in out
    assert "[Lean code block omitted from the reading kit]" in out
    assert "Proof prose." in out and "plain block" in out
    assert cut is False
