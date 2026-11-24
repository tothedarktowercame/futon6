from __future__ import annotations

import gzip
import json
from pathlib import Path

from futon6.reader_clear_draw import build_draw


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _fixture(tmp_path: Path, *, overlay_agrees: bool = True) -> tuple[Path, Path]:
    head = tmp_path / "head-deadbeef"
    problems = tmp_path / "problems"
    head.mkdir()
    rows = []
    modules = [f"problems.{letter}{index:02d}J01.lean.Main"
               for letter in "abc" for index in range(1, 7)]
    for index, module in enumerate(modules, 1):
        rows.append({"module": module, "declaration": f"d{index}",
                     "proof": ("word " * index).strip(),
                     "proof_error": "printer failed" if index == 2 else None})
        pid = module.split(".")[1]
        solution = problems / pid / "informal-solution.md"
        solution.parent.mkdir(parents=True)
        solution.write_text("solution\n")
    _write_jsonl(head / "render.jsonl", rows)
    ink_module = modules[0]
    _write_gzip_jsonl(head / "residual-ink-annotations.jsonl.gz", [
        {"module": ink_module, "spans": [{"start": 0, "end": 1}]}
    ])
    overlay_module = ink_module if overlay_agrees else modules[3]
    _write_gzip_jsonl(head / "residual-ink-overlay.jsonl.gz", [
        {"record": {"module": overlay_module}, "coordinate": {"start": 0, "end": 1}}
    ])
    return head, problems


def test_ink_and_proof_error_are_dropped_and_counted(tmp_path: Path):
    head, problems = _fixture(tmp_path)
    draw = build_draw(head, problems, 7, n=8, controls=2)
    drawn = {row["module"] for row in draw["drawn"]}
    assert "problems.a01J01.lean.Main" not in drawn
    assert "problems.a02J01.lean.Main" not in drawn
    assert draw["counts"]["inked"] == 1
    assert draw["counts"]["dropped_proof_error"] == 1
    assert draw["counts"]["overlay_modules_agree"] is True


def test_seed_is_reproducible_and_changes_draw(tmp_path: Path):
    head, problems = _fixture(tmp_path)
    first = build_draw(head, problems, 11, n=6, controls=2)
    again = build_draw(head, problems, 11, n=6, controls=2)
    other = build_draw(head, problems, 12, n=6, controls=2)
    assert first == again
    assert first["drawn"] != other["drawn"]


def test_allocation_and_controls_obey_bounds(tmp_path: Path):
    head, problems = _fixture(tmp_path)
    draw = build_draw(head, problems, 13, n=9, controls=3)
    assert sum(row["allocated"] for row in draw["strata"]) == 9
    assert all(row["allocated"] <= row["population"] for row in draw["strata"])
    sample = {row["module"] for row in draw["drawn"] if row["role"] == "sample"}
    controls = {row["module"] for row in draw["drawn"]
                if row["role"] == "planted-control"}
    assert sample.isdisjoint(controls)
    assert len(sample) == 9
    assert len(controls) == 3


def test_overlay_disagreement_is_reported_not_rejected(tmp_path: Path):
    head, problems = _fixture(tmp_path, overlay_agrees=False)
    draw = build_draw(head, problems, 17, n=5, controls=1)
    assert draw["counts"]["overlay_modules_agree"] is False
