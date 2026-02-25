#!/usr/bin/env python3
"""Build the superpod handoff PDF for Rob using fpdf2."""

from fpdf import FPDF
from pathlib import Path

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _line_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None

class HandoffPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", f"{FONT_DIR}/DejaVuSerif.ttf")
        self.add_font("DejaVu", "B", f"{FONT_DIR}/DejaVuSerif-Bold.ttf")
        self.add_font("DejaVu", "I", f"{FONT_DIR}/DejaVuSerif-Italic.ttf")
        self.add_font("DejaVu", "BI", f"{FONT_DIR}/DejaVuSerif-BoldItalic.ttf")
        self.add_font("DejaVuSans", "", f"{FONT_DIR}/DejaVuSans.ttf")
        self.add_font("DejaVuSans", "B", f"{FONT_DIR}/DejaVuSans-Bold.ttf")
        self.add_font("DejaVuMono", "", f"{FONT_DIR}/DejaVuSansMono.ttf")
        self.add_font("DejaVuMono", "B", f"{FONT_DIR}/DejaVuSansMono-Bold.ttf")

    def header(self):
        if self.page_no() > 1:
            self.set_font("DejaVu", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Superpod Run: Handoff Note", align="C")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("DejaVuSans", "B", 13)
        self.set_text_color(30, 30, 30)
        self.ln(3)
        self.cell(0, 8, title)
        self.ln(8)

    def body_text(self, text):
        self.set_font("DejaVu", "", 10.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, bold_prefix=None):
        self.set_font("DejaVu", "", 10.5)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, "\u2022")
        if bold_prefix:
            self.set_font("DejaVu", "B", 10.5)
            self.write(5.5, bold_prefix + " ")
            self.set_font("DejaVu", "", 10.5)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def numbered(self, num, text, bold_prefix=None):
        self.set_font("DejaVu", "", 10.5)
        self.set_text_color(30, 30, 30)
        self.cell(6, 5.5, f"{num}.")
        if bold_prefix:
            self.set_font("DejaVu", "B", 10.5)
            self.write(5.5, bold_prefix + " ")
            self.set_font("DejaVu", "", 10.5)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def mono(self, text):
        self.set_font("DejaVuMono", "", 8.5)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, text)
        self.set_font("DejaVu", "", 10.5)
        self.set_text_color(30, 30, 30)
        self.ln(1)

    def table_row(self, cells, widths, bold=False):
        style = "B" if bold else ""
        self.set_font("DejaVu", style, 9.5)
        h = 6
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.cell(w, h, cell, border=0)
        self.ln(h)


def build():
    pdf = HandoffPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("DejaVuSans", "B", 18)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, "Superpod Run", align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "", 13)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "What It Is and What It Produces", align="C")
    pdf.ln(7)
    pdf.set_font("DejaVu", "I", 10)
    pdf.cell(0, 6, "A Handoff Note", align="C")
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Joe Corneli  |  Hyperreal Enterprises  |  February 16, 2026", align="C")
    pdf.ln(12)

    # Horizontal rule
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    # Context
    pdf.section("Context")
    pdf.body_text(
        "Rob \u2014 this note explains the \"superpod run,\" a batch processing job "
        "that takes the full StackExchange mathematics data dumps and transforms "
        "them into structured knowledge artifacts. The run is the next concrete "
        "step in the futon6 project, and its output feeds directly into "
        "preparations for First Proof Batch 2 (expected ~1 month from now)."
    )
    pdf.body_text(
        "The short version: we're taking 567K math.stackexchange threads and "
        "100K MathOverflow threads, running them through an 11-stage pipeline "
        "(Stages 1\u20137 plus LWGM Stages 8\u201310), and producing typed artifacts for "
        "every thread \u2014 wiring diagrams, expression surfaces, hypergraphs, and "
        "a structural similarity index for nearest-neighbor retrieval by argument shape."
    )

    # What goes in
    pdf.section("What Goes In")
    pdf.body_text(
        "StackExchange publishes complete data dumps of every Q&A site. "
        "For mathematics we use two:"
    )
    pdf.bullet(
        "567K question threads. Undergraduate through graduate level. "
        "High volume, broad coverage.",
        bold_prefix="math.stackexchange.com \u2014"
    )
    pdf.bullet(
        "100K question threads. Research level. Narrower but deeper.",
        bold_prefix="mathoverflow.net \u2014"
    )
    pdf.body_text(
        "Each thread has a question, zero or more answers, and comments. "
        "The raw data is XML. Total: roughly 20 GB, compressing to about 5 GB."
    )

    # What the pipeline does
    pdf.section("What the Pipeline Does")
    pdf.body_text(
        "The superpod job has eleven stages. Four are GPU-bound (2, 3, 6, 9b); "
        "the rest are streaming CPU passes. In the required superpod run we "
        "typically skip the LLM stages (3 and 6) and still produce the core "
        "deliverables."
    )
    pdf.ln(2)

    # Pipeline table
    w = [12, 38, 105, 16]
    pdf.set_font("DejaVuSans", "B", 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(w[0], 7, "#", border="B", fill=True)
    pdf.cell(w[1], 7, "Stage", border="B", fill=True)
    pdf.cell(w[2], 7, "What it does", border="B", fill=True)
    pdf.cell(w[3], 7, "HW", border="B", fill=True)
    pdf.ln(7)

    stages = [
        ("1", "Parse", "Stream XML into structured QA pairs", "CPU"),
        ("2", "Embed", "Dense vector embeddings of posts (similarity + clustering)", "GPU"),
        ("3", "Tag", "LLM tagging: 25 argument patterns per answer", "GPU"),
        ("4", "Cluster", "Topic clustering based on embeddings", "CPU"),
        ("5", "Terms + scopes", "Term dictionary + scope openers (Let/Assume/Define)", "CPU"),
        ("6", "Rev. morph.", "LLM situation reconstruction per QA pair", "GPU"),
        ("7", "Thread wiring", "Typed wiring: IATC + CT refs + ports", "CPU"),
        ("8", "Expr. surfaces", "Parse $...$ LaTeX into typed s-expressions", "CPU"),
        ("9a", "Hypergraphs", "Assemble typed hypergraph per thread", "CPU"),
        ("9b", "Graph embed", "R-GCN contrastive embedding (LWGM)", "GPU"),
        ("10", "Similarity idx", "FAISS k-NN index over structural embeddings", "CPU"),
    ]
    pdf.set_font("DejaVu", "", 9)
    for num, name, desc, hw in stages:
        pdf.cell(w[0], 6, num)
        pdf.set_font("DejaVu", "B", 9)
        pdf.cell(w[1], 6, name)
        pdf.set_font("DejaVu", "", 9)
        pdf.cell(w[2], 6, desc)
        color = (180, 60, 60) if hw == "GPU" else (60, 120, 60)
        pdf.set_text_color(*color)
        pdf.cell(w[3], 6, hw)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(6)
    pdf.ln(4)

    pdf.body_text(
        "Run modes: core superpod run (skip LLM, run GPU where needed for embeddings "
        "and LWGM), moist-run (generate prompt files for cloud LLM), and full GPU "
        "LLM runs (Stages 3 and 6). The critical output is Stage 7; Stages 8\u201310 "
        "extend it with expression-level structure and a global structural retrieval index."
    )

    # What comes out
    pdf.section("What Comes Out")
    pdf.body_text(
        "For every thread, Stage 7 produces a JSON wiring diagram with three levels:"
    )
    pdf.numbered(1,
        "Each post becomes a node. Edges are typed by illocutionary act \u2014 "
        "assert, challenge, clarify, reference, retract, reform, etc. "
        "These capture the argumentative moves in the thread.",
        bold_prefix="Thread level (discourse)."
    )
    pdf.numbered(2,
        "Each node is checked against a reference dictionary of 8 category-theory "
        "pattern types (adjunction, equivalence, fibration, etc.), extracted from "
        "20K nLab wiki pages. IDF weighting avoids false positives.",
        bold_prefix="Categorical level."
    )
    pdf.numbered(3,
        "Mathematical content is decomposed into input ports (assumptions, "
        "let-bindings) and output ports (conclusions, constraints). Edges carry "
        "port matches showing which outputs connect to which inputs, with scores "
        "boosted by CT reference weights.",
        bold_prefix="Port level."
    )
    pdf.ln(1)
    pdf.body_text(
        "So where a raw SE thread looks like \"question \u2192 answer \u2192 comment,\" "
        "the wiring diagram looks more like: \"question introduces terms X, Y via "
        "let-bindings; answer asserts conclusion Z via Cauchy-Schwarz referencing X; "
        "comment challenges the bound on Y with an adversative discourse marker.\""
    )

    # Scale
    pdf.section("Scale")

    w2 = [60, 45, 45]
    pdf.set_font("DejaVuSans", "B", 9.5)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(w2[0], 7, "", border="B", fill=True)
    pdf.cell(w2[1], 7, "math.SE", border="B", fill=True, align="R")
    pdf.cell(w2[2], 7, "MathOverflow", border="B", fill=True, align="R")
    pdf.ln(7)
    rows = [
        ("Threads", "~567K", "~100K"),
        ("Nodes (posts)", "~3M", "~500K"),
        ("Edges (typed)", "~2.5M", "~400K"),
        ("Output size (est.)", "~45 GB", "~8 GB"),
    ]
    pdf.set_font("DejaVu", "", 9.5)
    for label, a, b in rows:
        pdf.cell(w2[0], 6, label)
        pdf.cell(w2[1], 6, a, align="R")
        pdf.cell(w2[2], 6, b, align="R")
        pdf.ln(6)
    pdf.ln(4)

    pdf.body_text(
        "The 200-thread pilot we ran this week produced 1,641 nodes, 1,441 "
        "edges, and 319 categorical detections with sharp differentiation: "
        "category-theory threads averaged 3 categorical patterns per thread "
        "vs. 0.2 for mathematical-physics threads. The signal is real."
    )

    # First Proof 2
    pdf.section("How This Connects to First Proof 2")
    pdf.body_text(
        "We did a proof sprint on the First Proof benchmark two weeks ago "
        "(10 research-level math problems, 55 hours, 4/10 correct). A detailed "
        "retrospective identified five things we'd do differently. The superpod "
        "run addresses three of them directly:"
    )
    pdf.numbered(1,
        "During Sprint 1, keyword-based searches missed critical techniques "
        "(e.g., hyperbolic Hessian contraction bounds for Problem 4). The superpod "
        "output provides a structured index of how 667K threads discuss mathematical "
        "concepts \u2014 not just keywords, but typed connections between assumptions "
        "and conclusions. When we need a bridge between proof steps, we can query: "
        "\"what threads have an output port matching this input?\"",
        bold_prefix="Better literature mining."
    )
    pdf.numbered(2,
        "We built a verifier that checks each edge in a proof wiring diagram "
        "for structural consistency (port types, discourse markers, categorical "
        "patterns). During Sprint 1 these checks were manual. Now they're automated, "
        "and the superpod gives the verifier a large corpus to calibrate against.",
        bold_prefix="Automated proof verification."
    )
    pdf.numbered(3,
        "Sprint 1's two wrong answers were both on problems requiring deep "
        "specialized knowledge. The superpod output lets us pre-identify which "
        "subfields have rich coverage and which are sparse, so we can front-load "
        "targeted mining for the harder problems.",
        bold_prefix="Domain-depth pre-loading."
    )
    pdf.ln(1)
    pdf.body_text(
        "The other two lessons (mandatory falsification protocol, better calibration "
        "tracking) are process improvements that don't depend on the corpus but "
        "benefit from it \u2014 a verification score that plateaus is a concrete trigger "
        "for switching from constructive to falsification mode."
    )

    # Timeline
    pdf.section("Timeline")
    pdf.bullet(
        "Run the required superpod block (sharded, skip LLM). "
        "This produces CT-backed wiring, expression surfaces, hypergraphs, LWGM "
        "embeddings, and a FAISS similarity index.",
        bold_prefix="This week:"
    )
    pdf.bullet(
        "Optional: backfill LLM stages (pattern tags + reverse morphogenesis) "
        "on a bounded thread sample for quality assessment. Integrate the verifier "
        "+ structural search into the proof workflow.",
        bold_prefix="Next 2 weeks:"
    )
    pdf.bullet(
        "First Proof Batch 2, if announced. Otherwise, dry-run the protocol on "
        "a practice problem (e.g., re-attempt Problem 4 targeting the all-n bridge "
        "we missed the first time).",
        bold_prefix="Week 4:"
    )

    # What I'd like from you
    pdf.section("What I'd Like From You")
    pdf.body_text(
        "Mainly: a sanity check. Does this pipeline make sense as a way to build "
        "structured mathematical knowledge at scale? The bet is that typed wiring "
        "diagrams (not just embeddings, not just keyword search) are the right "
        "intermediate representation \u2014 rich enough to support automated "
        "verification, structured enough to enable port-level queries, but not so "
        "formal that we need full theorem-proving infrastructure."
    )
    pdf.body_text(
        "If you want to poke at any of the artifacts, the repo is futon6 on GitHub "
        "(private, happy to add you). Key files:"
    )
    pipeline_lines = _line_count(REPO_ROOT / "scripts" / "superpod-job.py")
    assemble_lines = _line_count(REPO_ROOT / "scripts" / "assemble-wiring.py")
    verifier_lines = _line_count(REPO_ROOT / "scripts" / "ct-verifier.py")
    pdf.mono(
        "  scripts/superpod-job.py          # the pipeline"
        + (f" ({pipeline_lines:,} lines)" if pipeline_lines is not None else "")
    )
    pdf.mono(
        "  scripts/assemble-wiring.py       # wiring assembly"
        + (f" ({assemble_lines:,} lines)" if assemble_lines is not None else "")
    )
    pdf.mono(
        "  scripts/ct-verifier.py           # proof verifier"
        + (f" ({verifier_lines:,} lines)" if verifier_lines is not None else "")
    )
    pdf.mono("  data/nlab-ct-reference.json      # CT reference (20K pages)")
    pdf.mono("  data/first-proof/superpod-handoff-rob.lit.md  # executable ledger")

    # Output
    out = REPO_ROOT / "data" / "first-proof" / "superpod-handoff-rob.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    print(f"Written: {out}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build()
