#!/usr/bin/env python3
"""The literate weave: synthesis prose at LaTeX level 0, the mission speaking
from inside scope-boxes.

Takes a synthesis document (e.g. holes/anatomy-of-a-futonic-mission.md) whose
headings render as PLAIN article sections (level-0 prose), and which may embed
excerpt pragmas that pull sections of a mission source and render them through
render_mission_pdf's boxed machinery:

    <!-- excerpt: holes/missions/E-mission-head.md :: ## HEAD -->
    <!-- vitals: holes/missions/E-mission-head.md -->

Paths are relative to the synthesis file's repo root (futon6). An excerpt spans
from its heading to the next heading of the same-or-shallower level.

Usage: render_anatomy_pdf.py <synthesis.md> [--keep-tex]
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from render_mission_pdf import (PREAMBLE, esc, inline, render_body,
                                vitals_card, table_to_tex, STRIPPED)

ROOT = Path("/home/joe/code/futon6")

EXCERPT_RE = re.compile(r"<!--\s*excerpt:\s*(\S+)\s*::\s*(.+?)\s*-->")
VITALS_RE = re.compile(r"<!--\s*vitals:\s*(\S+)\s*-->")


def extract_section(src: Path, heading: str):
    """Lines of SRC from HEADING to the next same-or-shallower heading."""
    lines = src.read_text(encoding="utf-8").split("\n")
    level = len(heading) - len(heading.lstrip("#"))
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(heading.strip()):
            start = i
            break
    if start is None:
        return [f"> [excerpt not found: {heading}]"]
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"^(#{1,6})\s", lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    return lines[start:end]


def weave(md_path: Path) -> str:
    lines = md_path.read_text(encoding="utf-8").split("\n")
    out = []
    in_code = False
    in_quote = False
    table_buf = []
    i = 0

    def flush_table():
        nonlocal table_buf
        if table_buf:
            rows = [[c for c in r.strip().strip("|").split("|")] for r in table_buf
                    if not re.match(r"^\s*\|?[-: |]+\|?\s*$", r)]
            if rows:
                out.append(table_to_tex(rows))
            table_buf = []

    title = md_path.stem
    while i < len(lines):
        ln = lines[i]
        me = EXCERPT_RE.search(ln)
        mv = VITALS_RE.search(ln)
        if me:
            flush_table()
            src, heading = ROOT / me.group(1), me.group(2)
            out.append(render_body(extract_section(src, heading),
                                   md_path=src, with_title=False))
            i += 1
            continue
        if mv:
            flush_table()
            out.append(vitals_card(ROOT / mv.group(1)))
            i += 1
            continue
        if ln.strip().startswith("```"):
            flush_table()
            if not in_code and "mermaid" in ln:
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    i += 1
                out.append(r"{\small\sffamily [diagram elided --- see source]}")
            else:
                in_code = not in_code
                out.append(r"\begin{quote}\small\ttfamily" if in_code else r"\end{quote}")
            i += 1
            continue
        if in_code:
            out.append(esc(ln) + r"\\")
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            table_buf.append(ln)
            i += 1
            continue
        flush_table()
        m = re.match(r"^(#{1,3})\s+(.*)$", ln)
        if m:
            level, htext = len(m.group(1)), m.group(2)
            if level == 1:
                title = htext
                out.append(r"\begin{center}{\LARGE\bfseries " + inline(htext)
                           + r"}\end{center}\medskip")
            elif level == 2:
                out.append(r"\section*{" + inline(htext) + "}")
            else:
                out.append(r"\subsection*{" + inline(htext) + "}")
            i += 1
            continue
        if ln.startswith(">"):
            if not in_quote:
                out.append(r"\begin{quote}\itshape")
                in_quote = True
            out.append(inline(ln.lstrip("> ")))
            i += 1
            continue
        if in_quote and not ln.startswith(">"):
            out.append(r"\end{quote}")
            in_quote = False
        if re.match(r"^\s*[-*]\s+", ln) or re.match(r"^\s*\d+\.\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.\s+", ln))
            pat = r"^\s*\d+\.\s+" if ordered else r"^\s*[-*]\s+"
            items = []
            while i < len(lines) and (re.match(pat, lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip())):
                if re.match(pat, lines[i]):
                    items.append(re.sub(pat, "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            env = "enumerate" if ordered else "itemize"
            out.append(r"\begin{" + env + r"}[leftmargin=1.3em,itemsep=1pt]")
            out += [r"\item " + inline(it) for it in items]
            out.append(r"\end{" + env + "}")
            continue
        if ln.strip() in ("---", "***"):
            out.append(r"\medskip\hrule\medskip")
        elif ln.strip():
            out.append(inline(ln))
        else:
            out.append("")
        i += 1

    flush_table()
    if in_quote:
        out.append(r"\end{quote}")
    head = PREAMBLE.replace("MISSIONTITLEHEAD", esc(title[:60]))
    return head + "\n".join(out) + "\n\\end{document}\n"


def main():
    md = Path(sys.argv[1]).resolve()
    tex = md.with_suffix(".tex")
    tex.write_text(weave(md), encoding="utf-8")
    if STRIPPED:
        print(f"stripped glyphs: {' '.join(sorted(STRIPPED))}")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={md.parent}", str(tex)],
        capture_output=True, text=True, cwd=md.parent)
    pdf = md.with_suffix(".pdf")
    print(f"OK {pdf}" if pdf.exists() else "FAILED — see log")
    if "--keep-tex" not in sys.argv:
        for ext in (".aux", ".log", ".out", ".fls", ".fdb_latexmk"):
            p = md.with_suffix(ext)
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
