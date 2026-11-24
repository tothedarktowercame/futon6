#!/usr/bin/env python3
"""Render one InkUp exemplar (a pilot directory of tellings) as a LaTeXML page.

Assembles a LaTeX article from the problem statement and the four tellings,
then runs latexmlc. Node tags such as **[B4]** become margin-style labels.

  python3 scripts/inkup-exemplar-page.py --pilot holes/inkup/pilots/a97A02 \
      --problem-tex <apm-lean>/problems/a97A02/problem.tex --out <dir>/a97A02.html
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TELLINGS = [
    ("reader-telling.teaching.md", "Teaching", "For a student preparing for the exam."),
    ("reader-telling.expert-reference.md", "Expert reference", "For a reader who knows the area and wants the route."),
    ("reader-telling.formal-audit.md", "Formal audit", "Says what each step relies on in the Lean proof."),
    ("publication-telling.md", "Publication", "As the proof might appear in a paper."),
]

PREAMBLE = r"""\documentclass{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{enumitem}
\usepackage{hyperref}
\newenvironment{alist}{\begin{enumerate}[label=(\alph*)]}{\end{enumerate}}
\newenvironment{rlist}{\begin{enumerate}[label=(\roman*)]}{\end{enumerate}}
\providecommand{\C}{\mathbb{C}}
\providecommand{\R}{\mathbb{R}}
\providecommand{\Z}{\mathbb{Z}}
\providecommand{\Q}{\mathbb{Q}}
\providecommand{\N}{\mathbb{N}}
\providecommand{\Rn}{\mathbb{R}^{n}}
\providecommand{\tightlist}{}
"""


def tex_escape(text: str) -> str:
    return (text.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("#", r"\#").replace("_", r"\_").replace("^", r"\^{}"))


def md_body(path: Path) -> str:
    """Telling body (after its ## heading) as LaTeX, via pandoc."""
    text = path.read_text()
    match = re.search(r"(?m)^## .+$", text)
    body = text[match.end():] if match else text
    out = subprocess.run(
        ["pandoc", "-f", "markdown+tex_math_single_backslash-smart", "-t", "latex", "--wrap=preserve"],
        input=body, capture_output=True, text=True, check=True).stdout
    return re.sub(r"\\textbf\{\{?\[\}?(B\d+)\{?\]\}?\}", r"\\textsf{\\small[\1]}", out)


def questions_table(pilot: Path) -> str:
    plan = json.loads((pilot / "content-plan.json").read_text())
    rows = []
    for q in plan.get("questions", []):
        node = q.get("blueprint_node", "").split(":")[-1]
        decisions = q.get("decisions", {})
        who = ", ".join(k for k, v in decisions.items() if v.get("action") == "anticipate") or "none"
        rows.append(f"\\item \\textsf{{\\small[{node}]}} {tex_escape(q.get('analogue', ''))} "
                    f"\\emph{{Answered for: {tex_escape(who)}.}}")
    return "\\begin{itemize}\n" + "\n".join(rows) + "\n\\end{itemize}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--problem-tex", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--note", default="", help="one paragraph of plain text placed under the title")
    args = ap.parse_args()
    pid = args.pilot.name
    statement = re.sub(r"(?m)^%.*\n", "", args.problem_tex.read_text())
    parts = [PREAMBLE, f"\\title{{{pid}: one proof told four ways}}\n\\date{{}}\n\\begin{{document}}\n\\maketitle\n"]
    if args.note:
        parts.append(tex_escape(args.note) + "\n\n")
    parts.append("\\section*{Problem}\n" + statement + "\n")
    parts.append("\\section*{Questions the tellings may answer}\n"
                 "Each question is attached to a step of the proof and is answered only for the readers named.\n"
                 + questions_table(args.pilot))
    for name, title, blurb in TELLINGS:
        path = args.pilot / name
        if not path.exists():
            sys.exit(f"missing telling: {path}")
        parts.append(f"\\section*{{{title}}}\n\\emph{{{blurb}}}\n\n" + md_body(path) + "\n")
    parts.append("\\end{document}\n")
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / f"{pid}.tex"
        tex.write_text("".join(parts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["latexmlc", "--dest", str(args.out), "--format=html5", "--pmml", "--mathtex",
             "--nodefaultresources", "--css=LaTeXML.css", "--css=ltx-article.css", str(tex)],
            capture_output=True, text=True, timeout=300)
        tail = [line for line in result.stderr.splitlines() if re.search(r"(?i)error|warning|undefined", line)]
        print("\n".join(tail[-15:]))
        if result.returncode != 0:
            sys.exit(f"latexmlc failed ({result.returncode})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
