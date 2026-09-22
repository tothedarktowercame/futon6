#!/usr/bin/env python3
"""Typeset one paper of a run and attach the run's reading to it.

Three steps, all from run-owned data:

  1. write the paper's exact S1 marks text as <out>/<paper>.tex -- bytes, not text,
     since a paper may hold CRLF line endings (0708.2185 holds 28) and rewriting
     them would move every offset the annotations use;
  2. convert it with latexml_oxide --source-map, run from that directory so the
     source path the converter records is the file that stays there, and apply the
     Tufte transform, which preserves every data-sourcepos;
  3. build the margin page (render_scope_margin) over the result.

Each paper gets its own directory, because the renderer reads one conversion.log.

Tools: latexml_oxide on PATH or FUTON6_LATEXML_OXIDE; the Tufte transform at
FUTON6_TUFTIFY, else futon5/holes/tech-notes/paper/html-build/wysiwyg/tuftify.py
beside this checkout. TeX Live must be on PATH for the converter's kpathsea.

Usage: scripts/typeset_preview.py RUN PAPER [--out DIR] [--no-render]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import futon6_config as config

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import render_scope_margin


def tool(name: str, env: str, default: Path | None = None) -> Path:
    override = os.environ.get(env)
    if override:
        return Path(override)
    found = shutil.which(name)
    if found:
        return Path(found)
    if default and default.is_file():
        return default
    raise SystemExit(f"{name} not found: put it on PATH or set {env}")


def build(run: Path, paper: str, out: Path, render: bool = True) -> dict:
    marks = json.loads((run / "artifacts/marks" / f"fable-{paper}-dp-emacs.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    tex = out / f"{paper}.tex"
    tex.write_bytes(marks["text"].encode())
    oxide = tool("latexml_oxide", "FUTON6_LATEXML_OXIDE")
    tuftify = tool("tuftify.py", "FUTON6_TUFTIFY",
                   config.sibling("futon5") / "holes/tech-notes/paper/html-build/wysiwyg/tuftify.py")
    log = out / "conversion.log"
    with log.open("w") as fh:
        subprocess.run([str(oxide), tex.name, "--dest", f"{paper}.html", "--source-map"],
                       cwd=out, stdout=fh, stderr=subprocess.STDOUT, check=True)
    page = (out / f"{paper}.html").read_text()
    if "ltx_ERROR" in page:
        raise SystemExit(f"{paper}: the converter reported errors in the page; see {log}")
    subprocess.run([sys.executable, str(tuftify), f"{paper}.html", "-o", f"{paper}-tufte.html"],
                   cwd=out, check=True, stdout=subprocess.DEVNULL)
    result = {"paper": paper, "dir": str(out), "source-positions": page.count("data-sourcepos")}
    if render:
        html, summary = render_scope_margin.build(run, paper, out)
        (out / f"{paper}-margin.html").write_text(html)
        result["summary"] = summary
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("paper")
    ap.add_argument("--out", type=Path, help="default: RUN/typeset-preview/PAPER")
    ap.add_argument("--no-render", action="store_true", help="typeset only, do not attach the reading")
    a = ap.parse_args()
    out = a.out or (a.run / "typeset-preview" / a.paper)
    print(json.dumps(build(a.run, a.paper, out, render=not a.no_render)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
