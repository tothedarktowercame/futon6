#!/usr/bin/env python3
"""Pretty-print a futonic mission doc as a PDF — the scope structure as typography.

Conventions (per Joe, 2026-06-10, after the First Proof monograph):
- paragraph-length scopes -> nested tcolorboxes (the mission-mode depth palette:
  phases teal-framed, sections amber, subsections rose/violet);
- word-length spans (`code spans` = pattern idents, file paths, concepts) ->
  highlighted;
- the sibling <mission>.health.json (if present) -> a vitals card up front;
- math markup out of scope (those are scopes at the tight symbolic layer).

Usage: render_mission_pdf.py <mission.md> [--keep-tex]
Output: <mission>.pdf next to the input. Mermaid blocks are elided with a
pointer (figures are a later pass). Unknown exotic glyphs are mapped or
stripped, and logged.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

CANON = {"head", "identify", "map", "derive", "argue", "verify",
         "instantiate", "document", "close"}

UNI = {
    "→": r"$\rightarrow$", "←": r"$\leftarrow$", "↔": r"$\leftrightarrow$",
    "⇒": r"$\Rightarrow$", "≈": r"$\approx$", "≥": r"$\geq$", "≤": r"$\leq$",
    "×": r"$\times$", "·": r"$\cdot$", "Δ": r"$\Delta$", "ψ": r"$\psi$",
    "ε": r"$\varepsilon$", "α": r"$\alpha$", "β": r"$\beta$", "θ": r"$\theta$",
    "γ": r"$\gamma$", "Σ": r"$\Sigma$", "∇": r"$\nabla$", "∈": r"$\in$",
    "♥": r"$\heartsuit$", "⊗": r"$\otimes$", "⊙": r"$\odot$",
    "∅": r"$\varnothing$", "✓": r"\checkmark{}", "✗": r"$\times$",
    "✅": r"\checkmark{}", "★": r"$\star$", "—": "---", "–": "--",
    "‘": "`", "’": "'", "“": "``", "”": "''", "…": r"\ldots{}",
    "§": r"\S{}", "↓": r"$\downarrow$", "↑": r"$\uparrow$",
    "′": r"$'$", "≠": r"$\\neq$", "∧": r"$\\wedge$", "∨": r"$\\vee$", "¬": r"$\\neg$", "⟹": r"$\\Rightarrow$", "⊆": r"$\\subseteq$", "∀": r"$\\forall$", "∃": r"$\\exists$", "🔮": r"[orb]", "①": "(1)", "②": "(2)", "③": "(3)",
}

PREAMBLE = r"""\documentclass[11pt,oneside]{article}
\usepackage[margin=2.6cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{longtable,booktabs}
\usepackage{tabularx}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\usepackage{enumitem}
\usepackage[normalem]{ulem}
\usepackage{xcolor}
\usepackage[most]{tcolorbox}
\usepackage{soul}
\usepackage{fancyhdr}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=teal!60!black, urlcolor=teal!60!black}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\sffamily MISSIONTITLEHEAD}
\fancyhead[R]{\small\sffamily\thepage}
% --- the mission-mode depth palette, in print (per First Proof house style) ---
\definecolor{phaseframe}{HTML}{0F766E}
\definecolor{sectamber}{HTML}{B45309}
\definecolor{subrose}{HTML}{D6608A}
\definecolor{subviolet}{HTML}{7C3AED}
\definecolor{hlamber}{HTML}{FDF3DF}
\definecolor{chipviolet}{HTML}{7C3AED}
\newtcolorbox{phasebox}[1][]{%
  breakable, colback=teal!3, colframe=phaseframe,
  fonttitle=\bfseries\sffamily\small, #1,
  left=6pt, right=6pt, top=4pt, bottom=4pt,
  boxrule=0.7pt, arc=2pt, before skip=10pt, after skip=6pt}
\newtcolorbox{sectionbox}[1][]{%
  breakable, colback=orange!4, colframe=sectamber,
  fonttitle=\bfseries\sffamily\small, #1,
  left=6pt, right=6pt, top=4pt, bottom=4pt,
  boxrule=0.5pt, arc=2pt, before skip=8pt, after skip=4pt}
\newtcolorbox{subbox}[1][]{%
  breakable, colback=violet!3, colframe=subrose,
  fonttitle=\bfseries\sffamily\footnotesize, #1,
  left=5pt, right=5pt, top=3pt, bottom=3pt,
  boxrule=0.4pt, arc=2pt, before skip=6pt, after skip=3pt}
\newtcolorbox{vitalsbox}[1][]{%
  colback=teal!4, colframe=phaseframe, fonttitle=\bfseries\sffamily,
  title={$\heartsuit$ vitals (computed from this document)}, #1,
  left=6pt, right=6pt, top=4pt, bottom=4pt, boxrule=0.7pt, arc=2pt}
\sethlcolor{hlamber}
\newcommand{\scopehl}[1]{\texttt{\hl{#1}}}
\begin{document}
"""


def esc(s: str) -> str:
    """Escape LaTeX specials, map unicode, leave already-mapped macros alone."""
    out = []
    for ch in s:
        if ch in UNI:
            out.append(UNI[ch])
        elif ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ord(ch) > 0x2500 or (0x4E00 <= ord(ch) <= 0x9FFF):
            STRIPPED.add(ch)
        else:
            out.append(ch)
    return "".join(out)


STRIPPED = set()


def inline(s: str) -> str:
    """Inline markdown: code spans -> scope highlights; bold; italics; links."""
    parts = re.split(r"(`[^`]+`)", s)
    rendered = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 2:
            rendered.append(r"\scopehl{" + esc(p[1:-1]) + "}")
        else:
            t = esc(p)
            t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
            t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\\emph{\1}", t)
            t = re.sub(r"\[\[([^\]]+)\]\]", r"\\textsf{[\1]}", t)
            t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
            rendered.append(t)
    return "".join(rendered)


def table_to_tex(rows):
    cols = max(len(r) for r in rows)
    # Full-width wrapping columns (Y = raggedright X) so wide tables don't overflow.
    spec = "Y" * cols
    out = [r"\begin{center}\small\begin{tabularx}{\textwidth}{" + spec + "}", r"\toprule"]
    for i, r in enumerate(rows):
        cells = [inline(c.strip()) for c in r] + [""] * (cols - len(r))
        out.append(" & ".join(cells) + r" \\")
        if i == 0:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabularx}\end{center}"]
    return "\n".join(out)


def vitals_card(md_path: Path):
    hp = md_path.with_suffix(".health.json")
    if not hp.exists():
        return ""
    h = json.loads(hp.read_text())
    health, sigil = h.get("health", {}), h.get("sigil", {}) or {}
    prox = health.get("anchor-proximity") or {}
    marks = ""
    if prox:
        strong, weak = set(prox.get("strong", [])), set(prox.get("weak", []))
        marks = " ".join(
            (r"\checkmark{}" if n in strong else r"$\otimes$") + r"\textsf{\small " + n + "}"
            for n in ["IF", "HOWEVER", "THEN", "BECAUSE"])
    body = (
        f"confidence {health.get('bit-confidence', '?')} "
        rf"$\cdot$ xenotype {int(100*health.get('xenotype-completeness', 0))}\% "
        rf"$\cdot$ exotype \texttt{{{sigil.get('exotype', '?')}}}\\[2pt]" + marks +
        rf"\\[3pt]\emph{{{esc(health.get('reading', ''))}}}\\[2pt]"
        rf"{{\scriptsize generated {esc(h.get('generated-at', '')[:19])} by "
        rf"\texttt{{{esc(h.get('generator', ''))}}}}}")
    return "\\begin{vitalsbox}\n" + body + "\n\\end{vitalsbox}\n"


def render_body(lines, md_path=None, with_title=True):
    """Render markdown LINES to boxed LaTeX body (the scope typography).
    Reusable by the literate weaver (anatomy zip): with_title=False renders
    excerpts without title page/vitals hooks."""
    out = []
    stack = []  # open box envs: ("phasebox"|"sectionbox"|"subbox")
    title = md_path.stem
    in_code = False
    in_quote = False
    table_buf = []
    i = 0

    def close_to(level):
        # level: 1=close all, 2=close sub+section, 3=close sub only
        order = {"phasebox": 1, "sectionbox": 2, "subbox": 3}
        while stack and order[stack[-1]] >= level:
            out.append("\\end{" + stack.pop() + "}")

    def flush_table():
        nonlocal table_buf
        if table_buf:
            rows = [[c for c in r.strip().strip("|").split("|")] for r in table_buf
                    if not re.match(r"^\s*\|?[-: |]+\|?\s*$", r)]
            if rows:
                out.append(table_to_tex(rows))
            table_buf = []

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            flush_table()
            fence = ln.strip()
            if not in_code and "mermaid" in fence:
                # skip mermaid block entirely
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    i += 1
                out.append(r"{\small\sffamily [diagram elided --- see source / panel]}")
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
                close_to(1)
                if with_title:
                    out.append(r"\begin{center}{\LARGE\bfseries " + inline(htext)
                               + r"}\end{center}")
                    out.append("VITALSCARD")
            elif level == 2:
                close_to(1)
                word = re.sub(r"^[0-9.]+\s*", "", htext).split()[0].strip(".:—-").lower()
                env = "phasebox" if word in CANON else "sectionbox"
                out.append("\\begin{" + env + "}[title={" + inline(htext) + "}]")
                stack.append(env)
            else:
                close_to(3)
                out.append("\\begin{subbox}[title={" + inline(htext) + "}]")
                stack.append("subbox")
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
        if re.match(r"^\s*[-*]\s+", ln):
            items = []
            while i < len(lines) and (re.match(r"^\s*[-*]\s+", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip())):
                if re.match(r"^\s*[-*]\s+", lines[i]):
                    items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(r"\begin{itemize}[leftmargin=1.2em,itemsep=1pt]")
            out += [r"\item " + inline(it) for it in items]
            out.append(r"\end{itemize}")
            continue
        if re.match(r"^\s*\d+\.\s+", ln):
            items = []
            while i < len(lines) and (re.match(r"^\s*\d+\.\s+", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip())):
                if re.match(r"^\s*\d+\.\s+", lines[i]):
                    items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(r"\begin{enumerate}[leftmargin=1.4em,itemsep=1pt]")
            out += [r"\item " + inline(it) for it in items]
            out.append(r"\end{enumerate}")
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
    close_to(1)
    return "\n".join(out)


def render(md_path: Path) -> str:
    lines = md_path.read_text(encoding="utf-8").split("\n")
    body = render_body(lines, md_path=md_path, with_title=True)
    body = body.replace("VITALSCARD", vitals_card(md_path))
    head = PREAMBLE.replace("MISSIONTITLEHEAD", esc(md_path.stem))
    return head + body + "\n\\end{document}\n"


def main():
    md = Path(sys.argv[1]).resolve()
    tex = md.with_suffix(".tex")
    tex.write_text(render(md), encoding="utf-8")
    if STRIPPED:
        print(f"stripped glyphs: {' '.join(sorted(STRIPPED))}")
    r = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={md.parent}", str(tex)],
        capture_output=True, text=True, cwd=md.parent)
    pdf = md.with_suffix(".pdf")
    if pdf.exists():
        print(f"OK {pdf}")
    else:
        print("FAILED; tail of log:")
        print(r.stdout[-1800:])
    if "--keep-tex" not in sys.argv:
        for ext in (".aux", ".log", ".out", ".fls", ".fdb_latexmk"):
            p = md.with_suffix(ext)
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
