"""Tests for the arXiv math.CT loader."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "load-arxiv-ct.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("load_arxiv_ct", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_resolve_local_path_handles_storage_backed_data_dir(tmp_path: Path):
    module = load_script_module()
    data_dir = tmp_path / "storage" / "futon6" / "data"
    eprint_dir = data_dir / "arxiv-math-ct-eprints"
    eprint_dir.mkdir(parents=True)
    target = eprint_dir / "paper.tar.gz"
    target.write_bytes(b"placeholder")

    resolved = module._resolve_local_path("data/arxiv-math-ct-eprints/paper.tar.gz", data_dir)
    assert resolved == target


def test_iter_arxiv_entries_reads_gz_payload_from_storage_layout(tmp_path: Path):
    module = load_script_module()
    data_dir = tmp_path / "storage" / "futon6" / "data"
    data_dir.mkdir(parents=True)

    tex_payload = r"""
    \documentclass{article}
    \begin{document}
    \begin{definition}
    A functor category is a category of functors.
    \end{definition}
    \end{document}
    """.strip()
    eprint_path = data_dir / "arxiv-math-ct-eprints" / "math__1234.5678.tar.gz"
    eprint_path.parent.mkdir(parents=True)
    with gzip.open(eprint_path, "wt", encoding="utf-8") as handle:
        handle.write(tex_payload)

    write_jsonl(
        data_dir / "arxiv-math-ct-file-index.jsonl",
        [
            {
                "id": "math/1234.5678",
                "safe_id": "math__1234.5678",
                "title": "Functor categories",
                "local_file": "data/arxiv-math-ct-eprints/math__1234.5678.tar.gz",
                "has_local_file": True,
            }
        ],
    )
    write_jsonl(
        data_dir / "arxiv-math-ct-metadata.jsonl",
        [
            {
                "id": "math/1234.5678",
                "title": "Functor categories",
                "authors": ["A. Author"],
                "categories": ["math.CT"],
            }
        ],
    )

    entries = list(module.iter_arxiv_entries(data_dir, limit=1))
    assert len(entries) == 1
    assert entries[0].arxiv_id == "math/1234.5678"
    assert entries[0].body_length > 0
    assert "functor category" in entries[0].body_text.lower()
