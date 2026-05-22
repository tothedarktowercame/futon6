"""Smoke test for scripts/eval-grounding.py.

Runs the eval harness on a synthetic 2-paper fixture and verifies the
report has the expected shape — strategy_meta_learning aggregate,
spot-check samples per strategy, and per-paper summaries.
"""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _write_ner_kernel(path: Path) -> None:
    path.write_text(
        "term_lower\tterm_orig\tunused\tcanon\n"
        "category\tCategory\t_\tCategory\n"
        "abelian group\tabelian group\t_\tAbelianGroup\n"
        "monad\tMonad\t_\tMonad\n",
        encoding="utf-8",
    )


def _write_paper(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_eval_grounding_smoke(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    _write_paper(
        papers_dir / "alpha.tex",
        r"\documentclass{article}\begin{document}"
        "Let $X$ be an abelian group. Then $X$ is well-defined."
        r"\end{document}",
    )
    _write_paper(
        papers_dir / "beta.tex",
        r"\documentclass{article}\begin{document}"
        "Consider the category $\\mathcal{C}$. "
        "Here $T$ denotes a monad on $\\mathcal{C}$."
        r"\end{document}",
    )

    kernel_path = tmp_path / "terms.tsv"
    _write_ner_kernel(kernel_path)
    out_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            PY, "scripts/eval-grounding.py",
            "--input-dir", str(papers_dir),
            "--ner-kernel", str(kernel_path),
            "--out", str(out_path),
            "--sample-per-strategy", "3",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"eval-grounding.py failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["paper_count"] == 2
    assert "strategy_meta_learning" in report
    # At least one of the prose strategies fired
    strategies_seen = set(report["strategy_meta_learning"].keys())
    assert strategies_seen, "expected at least one strategy to fire"
    assert "spot_check_samples" in report
    assert "per_paper_summary" in report
    assert len(report["per_paper_summary"]) == 2
