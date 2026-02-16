"""Smoke tests for scripts/superpod-job.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_superpod(root: Path, outdir: Path, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    posts = root / "tests/fixtures/se-mini/Posts.xml"
    comments = root / "tests/fixtures/se-mini/Comments.xml"
    cmd = [
        sys.executable,
        "scripts/superpod-job.py",
        str(posts),
        "--comments-xml",
        str(comments),
        "--site",
        "math.stackexchange",
        "--output-dir",
        str(outdir),
        "--min-score",
        "0",
        "--skip-embeddings",
        "--skip-llm",
        "--skip-clustering",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_superpod_job_ct_pipeline_smoke(tmp_path: Path):
    root = Path(__file__).parent.parent
    outdir = tmp_path / "superpod-out"
    run = _run_superpod(root, outdir, ["--thread-limit", "4"])
    assert run.returncode == 0, (
        "superpod-job failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages_completed"] == ["parse", "ner_scopes", "thread_wiring"]
    assert manifest["stage7_stats"]["ct_backed"] is True
    assert manifest["stage7_stats"]["threads_processed"] == 4

    wiring = json.loads((outdir / "thread-wiring-ct.json").read_text(encoding="utf-8"))
    assert isinstance(wiring, list)
    assert len(wiring) == 4


def test_superpod_job_limit_defaults_thread_limit(tmp_path: Path):
    root = Path(__file__).parent.parent
    outdir = tmp_path / "superpod-out-limit"

    run = _run_superpod(root, outdir, ["--limit", "2"])
    assert run.returncode == 0, (
        "superpod-job failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert "Pilot mode: --thread-limit defaulted to --limit (2)" in run.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entity_count"] == 2
    assert manifest["stage7_stats"]["threads_processed"] == 2
