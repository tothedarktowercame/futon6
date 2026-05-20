"""Tests for arXiv CT open-term evidence building."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build-arxiv-ct-term-evidence.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_arxiv_ct_term_evidence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_seed(path: Path, entries: list[dict]) -> None:
    rows = ", ".join(
        f'{{:term/id "{entry["term/id"]}" :term/headword "{entry["term/headword"]}" :term/lower "{entry["term/lower"]}"}}'
        for entry in entries
    )
    path.write_text(
        "{:dictionary/entries [" + rows + "]}\n",
        encoding="utf-8",
    )


def test_builds_rhs_evidence_and_seed_membership(tmp_path: Path):
    module = load_script_module()
    data_root = tmp_path / "storage" / "futon6" / "data"
    out_dir = tmp_path / "out"
    data_root.mkdir(parents=True)

    tex_payload = r"""
    \documentclass{article}
    \begin{document}
    We call an \emph{functor category} a pleasant setting.
    An \emph{adjunction square} is defined as a commuting square with chosen unit data.
    A \emph{free} object is defined as one with a universal property.
    The smallest of these numbers is called an \emph{smallest of these numbers}.

    \begin{definition}
    A functor category is a category of functors.
    \end{definition}

    \begin{theorem}
    Every adjunction square induces a functor category under mild hypotheses.
    \end{theorem}
    \end{document}
    """.strip()

    eprint_path = data_root / "arxiv-math-ct-eprints" / "math__1234.5678.tar.gz"
    eprint_path.parent.mkdir(parents=True)
    with gzip.open(eprint_path, "wt", encoding="utf-8") as handle:
        handle.write(tex_payload)

    write_jsonl(
        data_root / "arxiv-math-ct-file-index.jsonl",
        [
            {
                "id": "math/1234.5678",
                "safe_id": "math__1234.5678",
                "title": "Functor categories and adjunction squares",
                "local_file": "data/arxiv-math-ct-eprints/math__1234.5678.tar.gz",
                "has_local_file": True,
            }
        ],
    )
    write_jsonl(
        data_root / "arxiv-math-ct-metadata.jsonl",
        [
            {
                "id": "math/1234.5678",
                "title": "Functor categories and adjunction squares",
                "authors": ["A. Author"],
                "categories": ["math.CT"],
            }
        ],
    )

    pm_seed = tmp_path / "entries-pm-seed.edn"
    nlab_seed = tmp_path / "entries-nlab-seed.edn"
    nnexus_snapshot = tmp_path / "snapshot-6-2014.sqlite"
    write_seed(
        pm_seed,
        [{"term/id": "functor-category", "term/headword": "Functor category", "term/lower": "functor category"}],
    )
    write_seed(nlab_seed, [])
    nnexus_snapshot.write_text(
        "\n".join(
            [
                "INSERT INTO \"concepts\" VALUES(1,'adjunction','square','XX-XX','msc','NNexus','example',1);",
                "INSERT INTO \"concepts\" VALUES(2,'functor','category','XX-XX','msc','NNexus','example',1);",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    module.main(
        [
            "--data-root", str(data_root),
            "--pm-seed", str(pm_seed),
            "--nlab-seed", str(nlab_seed),
            "--nnexus-snapshot", str(nnexus_snapshot),
            "--out-dir", str(out_dir),
            "--timestamp", "2026-05-20T12:00:00Z",
        ]
    )

    rows = [
        json.loads(line)
        for line in (out_dir / "arxiv-ct-open-term-evidence.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_term = {row["term_lower"]: row for row in rows}

    functor_category = by_term["functor category"]
    assert functor_category["known_in_pm_seed"] is True
    assert functor_category["novel_vs_seed"] == "known"
    assert functor_category["rhs_support_counts"]["definition-env"] >= 1
    assert functor_category["rhs_support_counts"]["theorem-statement"] >= 1

    adjunction_square = by_term["adjunction square"]
    assert adjunction_square["known_in_pm_seed"] is False
    assert adjunction_square["known_in_nnexus_snapshot"] is True
    assert adjunction_square["novel_vs_seed"] == "known"
    assert adjunction_square["rhs_support_counts"]["local-definitional-context"] >= 1
    assert adjunction_square["rhs_support_counts"]["theorem-statement"] >= 1

    assert "free" not in by_term
    assert "smallest of these numbers" not in by_term

    summary = json.loads((out_dir / "arxiv-ct-open-term-evidence-summary.json").read_text(encoding="utf-8"))
    assert summary["papers_processed"] == 1
    assert summary["unique_candidate_terms"] >= 2
    assert summary["prefilter_unique_candidate_terms"] >= 4
    assert summary["filtered_out_terms"] >= 2
    assert summary["known_in_pm_seed"] >= 1
    assert summary["known_in_nnexus_snapshot"] >= 1
