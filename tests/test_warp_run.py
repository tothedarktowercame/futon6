from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import warp_run


def test_confirmed_spine_orders_graph_before_embed() -> None:
    stage_ids = [stage.stage_id for stage in warp_run.SPINE_STAGES]

    assert stage_ids.index("S5") < stage_ids.index("S4c")
    defined = next(stage for stage in warp_run.SPINE_STAGES if stage.stage_id == "S2")
    assert all("concordance.json" not in str(path) for path in defined.inputs)


def test_guard_rejects_runnable_concept_index_writer(tmp_path: Path) -> None:
    guarded = warp_run.Stage(
        "bad",
        "sfc_concept_index.py",
        (tmp_path / "input.json",),
        (warp_run.WARP / "concept-index.json",),
        ("scripts/sfc_concept_index.py",),
    )

    with pytest.raises(SystemExit):
        warp_run.validate_guards([guarded])


def test_audit_only_concept_index_is_allowed() -> None:
    stage = next(stage for stage in warp_run.AUDIT_ONLY_STAGES if stage.script == "sfc_concept_index.py")

    warp_run.validate_guards([stage])


def test_freshness_uses_newest_input_and_oldest_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    old_output = tmp_path / "old-output.txt"
    new_output = tmp_path / "new-output.txt"
    input_path.write_text("input")
    old_output.write_text("old")
    new_output.write_text("new")

    os.utime(input_path, (200, 200))
    os.utime(old_output, (100, 100))
    os.utime(new_output, (300, 300))
    stage = warp_run.Stage("T", "noop.py", (input_path,), (old_output, new_output), ("noop.py",))

    assert not warp_run.is_fresh(stage)

    os.utime(old_output, (250, 250))
    assert warp_run.is_fresh(stage)


def test_dry_run_manifest_reports_would_run_without_writing(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_text("input")
    output_path.write_text("output")
    os.utime(input_path, (200, 200))
    os.utime(output_path, (100, 100))
    stage = warp_run.Stage("T", "noop.py", (input_path,), (output_path,), ("noop.py",))

    records = warp_run.run([stage], dry_run=True, manifest_path=manifest_path)

    assert records["T"]["status"] == "would-run"
    assert records["T"]["freshness"] == "stale"
    assert not manifest_path.exists()


def test_run_writes_skip_manifest_for_fresh_stage(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.json"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_text("input")
    output_path.write_text(json.dumps({"rows": [1]}))
    os.utime(input_path, (100, 100))
    os.utime(output_path, (200, 200))
    stage = warp_run.Stage("T", "noop.py", (input_path,), (output_path,), ("noop.py",))

    records = warp_run.run([stage], dry_run=False, manifest_path=manifest_path)

    assert records["T"]["status"] == "skipped"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["T"]["status"] == "skipped"
    assert manifest["T"]["input-hash"]
