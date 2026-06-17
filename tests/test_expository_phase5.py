from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EXTRACT = load_module(
    "mark3_extract_expository_candidates", ROOT / "scripts" / "mark3_extract_expository_candidates.py"
)
LOOP = load_module("mark3_expository_loop", ROOT / "scripts" / "mark3_expository_loop.py")


def test_expository_candidate_extractor_emits_schema_and_enrichment():
    candidates = EXTRACT.extract("0710.2254")
    assert candidates
    candidate = candidates[0]
    assert candidate["schema"] == "expo-candidate/v1"
    assert candidate["vocab-path"] == "holes/excursions/expository-superpod-vocab.edn"
    assert candidate["window-lines"][0] <= candidate["window-lines"][1]
    assert candidate["source-window"].strip()
    assert "enrichment" in candidate


def test_expository_loop_stub_gates_and_emits(tmp_path):
    candidate = EXTRACT.extract("0710.2254")[0]
    candidates_dir = tmp_path / "candidates"
    out_dir = tmp_path / "out"
    candidates_dir.mkdir()
    (candidates_dir / "one.candidate.json").write_text(json.dumps(candidate), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "mark3_expository_loop.py"),
            "--candidates",
            str(candidates_dir),
            "--out",
            str(out_dir),
            "--backend",
            "stub",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    emitted = list(out_dir.glob("*.edn"))
    assert len(emitted) == 1
    assert "expository-loop: 1/1 graphs gated PASS" in result.stdout


def test_expository_loop_refuses_pre_schema_candidate(tmp_path):
    candidates_dir = tmp_path / "stale"
    candidates_dir.mkdir()
    (candidates_dir / "stale.candidate.json").write_text(
        json.dumps(
            {
                "schema": "expo-candidate/v0",
                "paper-id": "bad",
                "passage-id": "bad:L1-1",
                "window-lines": [1, 1],
                "source-window": "text",
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "mark3_expository_loop.py"),
            "--candidates",
            str(candidates_dir),
            "--out",
            str(tmp_path / "out"),
            "--backend",
            "stub",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fail the expository precondition" in result.stderr


def test_expository_argcheck_fixtures():
    golden = subprocess.run(
        ["bb", str(ROOT / "scripts" / "expository_argcheck.bb"), str(ROOT / "holes" / "expository-argcheck" / "fixtures" / "golden")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert golden.returncode == 0, golden.stdout + golden.stderr
    assert golden.stdout.count("PASS ") >= 3

    negative = subprocess.run(
        ["bb", str(ROOT / "scripts" / "expository_argcheck.bb"), str(ROOT / "holes" / "expository-argcheck" / "fixtures" / "negative")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert negative.returncode == 1
    for gate in [
        "edn-parse",
        "unknown-kind",
        "missing-slot-fill",
        "missing-source",
        "out-of-scope-kind",
        "empty-held-reason",
    ]:
        assert f"[{gate}]" in negative.stdout
