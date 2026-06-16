from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mark3_eval_harness", ROOT / "scripts" / "mark3_eval_harness.py"
)
HARNESS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HARNESS)


def test_mark3_harness_builds_full_metric_set(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "0705.0102.edn").write_text(
        '{:paper/id "0705.0102" :passage/id "p" '
        ':source {:lines [10 12] :kind :proof} '
        ':nodes [{:id :x :kind :claim :text "a claim" :source {:lines [10 10]}}] '
        ':edges [{:id :e :kind :infer :premise :x :conclusion :x :source {:lines [11 11]}}] '
        ':holes []}',
        encoding="utf-8",
    )
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "fable-0705.0102-dp-emacs.json").write_text(
        json.dumps(
            {
                "paper": "0705.0102",
                "text": "body",
                "marks": [
                    {"layer": "dp", "kind": "math"},
                    {"layer": "dp", "kind": "definiendum"},
                ],
            }
        ),
        encoding="utf-8",
    )
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"df": {"a claim": 7}}), encoding="utf-8")

    def fake_run(cmd, cwd=None, text=None, capture_output=None):
        joined = " ".join(map(str, cmd))
        if "iatc_argcheck" in joined:
            return subprocess.CompletedProcess(cmd, 0, "PASS graph.edn\n", "")
        if "substance_gate" in joined:
            return subprocess.CompletedProcess(
                cmd, 0, "PASS graph.edn\n\nsubstance-gate: 1 file(s), 0 failure line(s) — PASS\n", ""
            )
        if "expository_region_extract" in joined:
            return subprocess.CompletedProcess(
                cmd,
                0,
                json.dumps(
                    {
                        "regions": [
                            {
                                "region_id": "r1",
                                "line_start": 10,
                                "line_end": 12,
                                "text": "Some exposition.",
                            }
                        ]
                    }
                ),
                "",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(HARNESS.subprocess, "run", fake_run)
    report = HARNESS.build_report(run_dir, golden, prior)

    assert report["run_kind"] == "iatc"
    assert report["metrics"]["grounding"]["grounding_percent"] == 0.5
    assert report["metrics"]["expository_coverage"]["expository_coverage_percent"] == 1.0
    assert report["metrics"]["checkers"]["checker_PASS_percent"] == 1.0
    assert report["metrics"]["checkers"]["substance_PASS_percent"] == 1.0
    assert report["metrics"]["prior_vs_posterior"]["posterior_terms_with_prior_df"] == 1
