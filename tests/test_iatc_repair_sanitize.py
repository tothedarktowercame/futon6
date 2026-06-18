"""Regression: iatc_repair.bb sanitizes invalid EDN string escapes.

Surfaced live 2026-06-18 on 0712.0724 (a category-theory paper) during the first
real-GPU Stage-A run: the 70B embedded raw LaTeX in description strings. Two
distinct hazards, both fixed by doubling the backslash of any non-EDN escape
inside strings (faithfully preserving the literal LaTeX while making the graph
parseable):

  - LOUD  : "\\circ" -> reader throws "Unsupported escape character: \\c", so the
            whole graph fails to parse and is dropped at the gate.
  - SILENT: "\\times" -> "\\t" (TAB), "\\nabla" -> "\\n" (newline) parse WITHOUT
            error and silently corrupt the anchor text (an L4 faithfulness defect
            the gate never catches).

See holes/excursions/E-sanitize-invalid-EDN.md for limitations + the deeper scan.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPAIR = REPO / "scripts" / "iatc_repair.bb"

# Mix of both hazard classes in one description string.
GRAPH = (
    '{:paper/id "x" :passage/id "x:p" '
    ':nodes [{:id :n :kind :object '
    ':text "u \\circ phi, A \\times B, \\nabla f, \\alpha" '
    ':source {:lines [1 1]}}] '
    ':edges [] :holes []}'
)


def _bb(*args):
    return subprocess.run(["bb", *map(str, args)], capture_output=True, text=True)


def _parses(path):
    """True iff `path` reads as EDN without error."""
    r = _bb("-e", f'(clojure.edn/read-string (slurp "{path}"))')
    return r.returncode == 0, r.stderr


def _text_field(path):
    """Read back the node :text via EDN so we see the decoded string."""
    r = _bb("-e", f'(print (-> (clojure.edn/read-string (slurp "{path}")) :nodes first :text))')
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_loud_class_unparseable_before_repair(tmp_path):
    f = tmp_path / "case.edn"
    f.write_text(GRAPH)
    ok, err = _parses(f)
    assert not ok, "expected the raw \\circ graph to fail EDN parse"
    assert "scape" in err  # "Unsupported escape character"


def test_repair_makes_latex_graph_parseable(tmp_path):
    f = tmp_path / "case.edn"
    f.write_text(GRAPH)
    rep = _bb(REPAIR, f)
    assert rep.returncode == 0, rep.stderr
    ok, err = _parses(f)
    assert ok, f"graph still unparseable after repair: {err}"


def test_faithfulness_no_silent_corruption(tmp_path):
    """\\times / \\nabla must stay literal, NOT become TAB / newline."""
    f = tmp_path / "case.edn"
    f.write_text(GRAPH)
    _bb(REPAIR, f)
    text = _text_field(f)
    # literal LaTeX preserved
    assert "\\circ" in text and "\\times" in text and "\\nabla" in text and "\\alpha" in text
    # the silent class did NOT decode into control characters
    assert "\t" not in text
    assert "\n" not in text
