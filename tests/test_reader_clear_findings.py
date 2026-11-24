from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "reader-clear-findings.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_marks_table_and_existing_findings(tmp_path: Path) -> None:
    kits = tmp_path / "kits"
    out = tmp_path / "out"
    classes = tmp_path / "classes.tsv"
    classes.write_text("id\tdescription\nP-example\ttest class\n", encoding="utf-8")

    first = kits / "p01"
    first.mkdir(parents=True)
    write_jsonl(first / "steps.jsonl", [
        {"step": "D1.S1", "declaration": "decl_one", "text": "Printed first.\nNext."},
        {"step": "D1.S2", "declaration": "decl_one", "text": "Printed second."},
    ])
    (first / "kit.json").write_text(json.dumps({"declarations": [
        {"declaration": "decl_one"},
    ]}), encoding="utf-8")
    write_jsonl(first / "steps.alice.jsonl", [
        {"id": "D1.S1", "verdict": "wrong-as-printed", "note": "Wrong reason."},
        {"id": "D1.S2", "verdict": "not-justified", "note": "Missing proof."},
    ])
    write_jsonl(first / "steps.bob.jsonl", [
        {"id": "D1.S1", "verdict": "justified"},
        {"id": "D1.S2", "verdict": "setup"},
    ])
    write_jsonl(first / "questions.alice.jsonl", [
        {"id": "Q1", "steps": json.dumps(["D1.S2"]), "locus_P": "Printed second."},
    ])
    write_jsonl(first / "questions.bob.jsonl", [])
    write_jsonl(first / "answers.jsonl", [
        {"reader": "alice", "id": "Q1", "diagnosis": "printer-drops-content",
         "proposed_class": "P-example", "answer": "It needs a proof."},
    ])
    write_jsonl(first / "round2.alice.jsonl", [
        {"id": "Q1", "status": "settled", "would_the_print_have_prevented_it": "yes"},
    ])
    # bob asked nothing, so has no round2 file; that must not make p01 incomplete.

    second = kits / "p02"
    second.mkdir()
    write_jsonl(second / "steps.jsonl", [
        {"step": "D2.S1", "declaration": "decl_two", "text": "Another print."},
    ])
    write_jsonl(second / "steps.alice.jsonl", [
        {"id": "D2.S1", "verdict": "wrong-as-printed", "note": "Bad formula."},
    ])
    (second / "steps.bob.jsonl").write_text(
        '{"id":"D2.S1","verdict":"not-justified"}\n{"id":', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--head", "demo", "--kits", str(kits),
         "--classes", str(classes), "--out", str(out)],
        text=True, capture_output=True, timeout=60, check=True,
    )

    assert "warning: skipping incomplete JSONL" in result.stderr
    with (out / "marks.head-demo.tsv").open(newline="") as handle:
        assert list(csv.DictReader(handle, delimiter="\t")) == [
            {"head": "demo", "module": "p01", "declaration": "decl_one",
             "step": "D1.S1", "printed_line": "Printed first. Next.", "reader": "alice",
             "verdict": "wrong-as-printed", "note": "Wrong reason.", "question": "-",
             "answerer_diagnosis": "-", "proposed_class": "-",
             "other_readers": "bob=justified"},
            {"head": "demo", "module": "p01", "declaration": "decl_one",
             "step": "D1.S2", "printed_line": "Printed second.", "reader": "alice",
             "verdict": "not-justified", "note": "Missing proof.", "question": "Q1",
             "answerer_diagnosis": "printer-drops-content", "proposed_class": "P-example",
             "other_readers": "bob=setup"},
            {"head": "demo", "module": "p02", "declaration": "decl_two",
             "step": "D2.S1", "printed_line": "Another print.", "reader": "alice",
             "verdict": "wrong-as-printed", "note": "Bad formula.", "question": "-",
             "answerer_diagnosis": "-", "proposed_class": "-", "other_readers": "-"},
        ]
    with (out / "findings.head-demo.tsv").open(newline="") as handle:
        findings = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["module"], row["reader"], row["question"], row["settlement"])
            for row in findings] == [("p01", "alice", "Q1", "settled")]
    assert "not complete: 1 ['p02']" in result.stdout

    # A reader who asked but has no restate blocks the problem, unless the owner
    # records that the restate was not taken.
    (first / "round2.alice.jsonl").unlink()
    blocked = subprocess.run(
        [sys.executable, str(SCRIPT), "--head", "demo", "--kits", str(kits),
         "--classes", str(classes), "--out", str(out)],
        text=True, capture_output=True, timeout=60, check=True,
    )
    assert "0 findings" in blocked.stdout and "'p01'" in blocked.stdout
    (first / "no-restate.alice").write_text("reason\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--head", "demo", "--kits", str(kits),
         "--classes", str(classes), "--out", str(out)],
        text=True, capture_output=True, timeout=60, check=True,
    )
    with (out / "findings.head-demo.tsv").open(newline="") as handle:
        assert [row["settlement"] for row in csv.DictReader(handle, delimiter="\t")] == ["not-taken"]
    assert "3 marks from 2 modules; wrong-as-printed=2, not-justified=1" in result.stdout
