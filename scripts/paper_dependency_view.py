#!/usr/bin/env python3
"""Build a D1/D2 dependency-review JSON for one local arXiv eprint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import anatomy_v0_sweep as sweep
import futon6_config as config
from futon6.latex_terms import extract_terms
from futon6.paper_dependency import build_dependency_graph, write_dependency_review
from futon6.paper_hypergraph import extract_paper_hypergraph_classical


def _eprint(paper: str) -> Path:
    """Resolve a paper id using the shared eprint directory conventions."""

    root = config.eprints()
    variants = list(dict.fromkeys((paper, paper.replace("/", "__"), paper.replace("__", "/"))))
    for suffix in (".tar.gz", ".tex.gz", ".gz", ".tar", ".tex", ""):
        for variant in variants:
            path = root / f"{variant}{suffix}"
            if path.exists():
                return path
    raise SystemExit(f"no eprint for {paper} under {root}")


def build(paper: str, output: Path | None = None) -> Path:
    """Read one eprint, run the classical arm, and write its review JSON."""

    files, _meta = sweep.read_eprint_files(_eprint(paper))
    parts: list[str] = []
    source_files: list[dict[str, int | str]] = []
    cursor = 0
    for file in files:
        chunk = file["text"]
        parts.append(chunk)
        source_files.append({"file": file["file"], "start": cursor, "end": cursor + len(chunk)})
        cursor += len(chunk) + 1
        parts.append("\n")
    text = "".join(parts)
    hg = extract_paper_hypergraph_classical(text, paper, concepts=extract_terms(text), techniques=[])
    graph = build_dependency_graph(hg, text, source_files=source_files)
    destination = output or (config.marks() / f"{paper}-dependency-review.json")
    return write_dependency_review(destination, paper, graph)


def main(argv: list[str] | None = None) -> int:
    """Run the one-paper dependency review command."""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: paper_dependency_view.py <arxiv-id>", file=sys.stderr)
        return 2
    path = build(args[0])
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
