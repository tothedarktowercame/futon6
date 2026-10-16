"""S11's structure transducer reads LaTeXML 0.8.6 output, and S11 refuses when
nothing parses instead of recording zero shapes as a measurement."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bb") is None, reason="needs babashka")

# latexmlmath --cmml=- for "g(z) = f(z) - h(z)" from LaTeXML 0.8.6 (Ubuntu 22.04):
# it opens with an XML declaration, which 0.8.8 does not write.
CMML_086 = ROOT / "tests/fixtures/latexmlmath-0.8.6-cmml.xml"


def fake_latexmlmath(tmp_path, payload: Path) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "latexmlmath"
    script.write_text(f"#!/bin/sh\ncat >/dev/null\ncat '{payload}'\n")
    script.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def test_xml_declaration_does_not_empty_the_structure(tmp_path):
    env = fake_latexmlmath(tmp_path, CMML_086)
    out = subprocess.run(["bb", str(ROOT / "scripts/sfc_def_structure.bb"), "-"],
                         input="g(z) = f(z) - h(z)", capture_output=True, text=True,
                         env=env, check=True).stdout
    assert ":structure (= (* g z)" in out


def test_stage_refuses_when_no_formula_parses(tmp_path):
    empty = tmp_path / "empty.xml"
    empty.write_text("<math></math>\n")
    env = fake_latexmlmath(tmp_path, empty)
    formulae = tmp_path / "f.txt"
    formulae.write_text("g(z) = f(z) - h(z)\nx \\in A\n")
    done = subprocess.run([sys.executable, str(ROOT / "scripts/sfc_struct_canon.py"),
                           "--formulae", str(formulae)], capture_output=True, text=True, env=env)
    assert done.returncode != 0
    assert "0 of 2 formulae gave a structure" in done.stdout + done.stderr
