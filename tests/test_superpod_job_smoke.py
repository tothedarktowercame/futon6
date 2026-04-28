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


def _run_arxiv_superpod(
    root: Path,
    outdir: Path,
    input_dir: Path,
    extra_args: list[str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "scripts/superpod-job.py",
        "--input-dir",
        str(input_dir),
        "--arxiv-jsonl",
        "batch-001.jsonl",
        "--site",
        "arxiv.math",
        "--output-dir",
        str(outdir),
        "--skip-embeddings",
        "--skip-llm",
        "--skip-clustering",
        "--skip-graph-embed",
        "--skip-faiss",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_arxiv_fixture(input_dir: Path) -> Path:
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    (input_dir / "batch-001.jsonl").write_text(
        json.dumps({
            "id": "math/0102067v1",
            "title": "A toy theorem",
            "abstract": "We prove that $x=x$ by a short argument.",
            "categories": ["math.CT"],
            "date": "2001-02-07",
        }) + "\n",
        encoding="utf-8",
    )
    (eprints / "math__0102067v1.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\begin{theorem}For every object $X$, $X=X$.\\end{theorem}\n"
        "\\begin{proof}Use the identity morphism $1_X:X\\to X$.\\end{proof}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return eprints


def _write_legacy_alias_arxiv_fixture(input_dir: Path) -> Path:
    input_dir.mkdir()
    eprints = input_dir / "eprints"
    eprints.mkdir()
    (input_dir / "batch-001.jsonl").write_text(
        json.dumps({
            "id": "math/0102067v1",
            "title": "A toy theorem",
            "abstract": "We prove that $x=x$ by a short argument.",
            "categories": ["math.CT"],
            "date": "2001-02-07",
        }) + "\n",
        encoding="utf-8",
    )
    (eprints / "math__0102067v1.tex").write_text(
        "\\documentclass{article}\n"
        "\\newtheorem{thm}{Theorem}\n"
        "\\begin{document}\n"
        "\\begin{thm}For every object $X$, $X=X$.\\end{thm}\n"
        "\\begin{proof}Use the identity morphism $1_X:X\\to X$.\\end{proof}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return eprints


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
    # CPU-only mode runs parse, ner, wiring, expressions, hypergraphs (+ graph_embed/faiss if available)
    completed = manifest["stages_completed"]
    assert "parse" in completed
    assert "ner_scopes" in completed
    assert "thread_wiring" in completed
    assert manifest["stage7_stats"]["ct_backed"] is True
    assert manifest["stage7_stats"]["threads_processed"] == 4
    assert "stage_status" in manifest
    assert manifest["stage_status"]["parse"]["status"] == "completed"
    assert manifest["stage_status"]["embeddings"]["status"] == "skipped"
    assert "skip_reason" in manifest["stage_status"]["embeddings"]
    assert manifest["stage_status"]["reverse_morphogenesis"]["status"] == "skipped"
    assert manifest["readiness"]["status"] in {"pass", "warn"}
    assert isinstance(manifest.get("health_gate_thresholds"), dict)

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
    assert manifest["stage_status"]["thread_wiring"]["status"] == "completed"


def test_arxiv_paper_eprint_dir_feeds_all_paper_stages(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    eprints = _write_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        ["--paper-eprint-dir", "eprints"],
    )
    assert run.returncode == 0, (
        "superpod-job arxiv eprint run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert "Paper eprint preflight: 1/1 arXiv entities" in run.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_eprint_dir"] == str(eprints.resolve())
    assert manifest["paper_hg_eprint_dir"] == str(eprints.resolve())
    assert manifest["paper_eprint"]["source"] == "paper-eprint-dir"
    assert manifest["paper_eprint"]["preflight"]["candidate_matches"] == 1
    assert manifest["llm_batch_sizes"]["stage5d"] == 4
    assert manifest["stage_status"]["technique_ner"]["text_source_counts"]["eprint"] == 1
    assert manifest["stage_status"]["paper_hypergraph"]["text_source_counts"]["eprint"] == 1
    assert manifest["stage9a_stats"]["paper_text_source"] == "eprints"
    assert manifest["stage9a_stats"]["eprint_text_used"] == 1
    geometry = json.loads((outdir / "geometry.json").read_text(encoding="utf-8"))
    assert isinstance(geometry, list)
    assert geometry[0]["paper_id"].startswith("arxiv-")
    assert "laplacian_summary" in geometry[0]
    assert manifest["stage9a_stats"]["geometry_stats"]["papers"] == 1


def test_arxiv_batch_local_eprints_auto_default(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    eprints = _write_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(root, outdir, input_dir, [])
    assert run.returncode == 0, (
        "superpod-job arxiv auto-default eprint run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert "defaulting --discover-terms-eprint-dir to batch-local eprints" in run.stdout

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["discover_terms_eprint_dir"] == str(eprints.resolve())
    assert manifest["paper_eprint_dir"] == str(eprints.resolve())
    assert manifest["paper_eprint"]["source"] == "discover-terms-eprint-dir-default"
    assert manifest["stage9a_stats"]["eprint_text_used"] == 1
    assert (outdir / "geometry.json").exists()


def test_arxiv_legacy_theorem_aliases_are_normalized(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_legacy_alias_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(root, outdir, input_dir, [])
    assert run.returncode == 0, (
        "superpod-job arxiv legacy alias run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_status"]["paper_hypergraph"]["normalized_papers"] == 1
    assert manifest["stage_status"]["paper_hypergraph"]["normalization_rewrites"] >= 2

    hypergraphs = json.loads((outdir / "paper-hypergraphs.json").read_text(encoding="utf-8"))
    claim_nodes = [n for n in hypergraphs[0]["nodes"] if n["type"] == "claim"]
    assert len(claim_nodes) == 1
    assert claim_nodes[0]["attrs"]["block_origin"] == "alias_expanded"
    assert "newtheorem alias thm->theorem" in claim_nodes[0]["attrs"]["source_cue"]


def test_arxiv_paper_hg_eprint_dir_is_legacy_alias(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    eprints = _write_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        [
            "--paper-hg-eprint-dir",
            "eprints",
            "--llm-stage5d-batch-size",
            "2",
            "--dry-run",
        ],
    )
    assert run.returncode == 0, (
        "superpod-job arxiv dry-run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )
    assert f"--paper-eprint-dir {eprints.resolve()}" in run.stdout
    assert "--llm-stage5d-batch-size 2" in run.stdout
    assert "--paper-hg-eprint-dir" not in run.stdout


def test_arxiv_paper_eprint_dir_fails_when_no_sources_match(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    eprints = _write_arxiv_fixture(input_dir)
    (eprints / "math__0102067v1.tex").unlink()
    (eprints / "unrelated.tex").write_text("$y=y$", encoding="utf-8")
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(
        root,
        outdir,
        input_dir,
        ["--paper-eprint-dir", "eprints"],
    )
    assert run.returncode != 0
    assert "no eprint filenames matched the arXiv entities" in run.stderr


def test_arxiv_moist_run_uses_paper_shape_prompt(tmp_path: Path):
    root = Path(__file__).parent.parent
    input_dir = tmp_path / "arxiv-input"
    _write_arxiv_fixture(input_dir)
    outdir = tmp_path / "arxiv-out"

    run = _run_arxiv_superpod(root, outdir, input_dir, ["--moist-run"])
    assert run.returncode == 0, (
        "superpod-job arxiv moist-run failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    prompt_path = outdir / "moist-prompts" / "stage3-pattern-tagging.jsonl"
    first = json.loads(prompt_path.read_text(encoding="utf-8").splitlines()[0])
    prompt = first["prompt"]
    assert "mathematics paper-shape classifier" in prompt
    assert "math-strategy/" in prompt
    assert "math.stackexchange" not in prompt
