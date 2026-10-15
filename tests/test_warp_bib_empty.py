"""warp_bib.py declares bib/ as an output, so it must exist even when empty.

A corpus that cites nothing -- the APM informal proofs -- produced no
per-paper bibliography file, so bib/ was never created and WARP S1c refused
to run on a missing input.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "warp_bib.py"


def test_corpus_without_references_still_gets_bib_dir(tmp_path):
    eprints = tmp_path / "eprints"
    eprints.mkdir()
    (eprints / "x.tex").write_text(
        "\\documentclass{article}\\begin{document}No references.\\end{document}\n")
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(SCRIPT), "--eprints-dir", str(eprints),
                    "--out-dir", str(out)], check=True, capture_output=True, timeout=120)
    assert (out / "bib").is_dir()
    assert (out / "bib-index.json").is_file()
