from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from futon6.reader_clear_loci import check_readings


SCRIPT = Path(__file__).parents[1] / "scripts" / "reader-clear-check-loci.py"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def run_cli(args: list[str]) -> int:
    spec = importlib.util.spec_from_file_location("reader_clear_check_loci_cli", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(args)


def test_verbatim_across_line_break_passes_and_paraphrase_fails(tmp_path: Path) -> None:
    printed = tmp_path / "print.md"
    printed.write_text("The exact passage spans\na line break in the proof.", encoding="utf-8")
    readings = write_jsonl(
        tmp_path / "reading.jsonl",
        [
            {"id": "verbatim", "locus_P": "exact passage spans a line break"},
            {"id": "paraphrase", "locus_P": "the passage crosses a line boundary"},
        ],
    )
    result = check_readings(printed, None, [readings])
    assert result.rows == 2
    assert [failure.id for failure in result.failures] == ["paraphrase"]


def test_case_difference_and_short_locus_fail(tmp_path: Path) -> None:
    printed = tmp_path / "print.md"
    printed.write_text("This Exact Phrase is long enough to locate.", encoding="utf-8")
    readings = write_jsonl(
        tmp_path / "reading.jsonl",
        [
            {"id": "case", "locus_P": "this Exact Phrase is long enough"},
            {"id": "short", "locus_P": "Exact"},
        ],
    )
    result = check_readings(printed, None, [readings])
    assert [failure.reason for failure in result.failures] == [
        "locus not found verbatim",
        "locus is shorter than 12 characters",
    ]


def test_ellipsis_parts_must_occur_in_order(tmp_path: Path) -> None:
    printed = tmp_path / "print.md"
    printed.write_text("alpha begins here, with middle words, and omega finishes there", encoding="utf-8")
    readings = write_jsonl(
        tmp_path / "reading.jsonl",
        [
            {"id": "ordered", "locus_P": "alpha begins…omega finishes"},
            {"id": "reversed", "locus_P": "omega finishes...alpha begins"},
        ],
    )
    result = check_readings(printed, None, [readings])
    assert [failure.id for failure in result.failures] == ["reversed"]


def test_informal_required_and_cli_exit_codes(tmp_path: Path, capsys) -> None:
    printed = tmp_path / "print.md"
    printed.write_text("A sufficiently long printed passage.", encoding="utf-8")
    informal = tmp_path / "informal.md"
    informal.write_text("A sufficiently long informal passage.", encoding="utf-8")
    passing = write_jsonl(
        tmp_path / "passing.jsonl",
        [{"id": "ok", "locus_I": "long informal passage"}],
    )
    failing = write_jsonl(
        tmp_path / "failing.jsonl",
        [{"id": "no-informal", "locus_I": "long informal passage"}],
    )
    assert run_cli(["--print", str(printed), "--informal", str(informal), str(passing)]) == 0
    assert "1 rows, 0 failures" in capsys.readouterr().out
    assert run_cli(["--print", str(printed), str(failing)]) == 1
    output = capsys.readouterr().out
    assert "no informal text was supplied" in output
    assert output.rstrip().endswith("1 rows, 1 failures")


def test_json_output_has_same_counts(tmp_path: Path, capsys) -> None:
    printed = tmp_path / "print.md"
    printed.write_text("A sufficiently long printed passage.", encoding="utf-8")
    readings = write_jsonl(
        tmp_path / "reading.jsonl",
        [{"id": "bad", "locus_P": "A sufficiently altered passage"}],
    )
    assert run_cli(["--json", "--print", str(printed), str(readings)]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["rows"] == 1
    assert output["failures"] == 1
    assert output["failure_rows"][0]["id"] == "bad"
