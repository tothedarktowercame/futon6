from pathlib import Path

from scripts import build_proofcheck_demo as demo


def test_graph_path_uses_attempt_for_substance_fail():
    path = demo.graph_path_for("0708.2185")

    assert path == demo.ATTEMPT_GRAPH
    assert ".attempts" in path.parts


def test_snippet_formulas_filters_incomplete_fragments():
    snippet = (
        "Bad $\\mbf{\\mrm{E}}=\\{$ and $. % comment \\to B$ "
        "but good $F:\\mathcal{A}\\to\\mathcal{B}$."
    )

    assert demo.snippet_formulas(snippet) == [r"F:\mathcal{A}\to\mathcal{B}"]


def test_reason_text_mentions_missing_anchor_terms():
    reason = {
        "id": "extensional-category",
        "source": {"lines": [1510, 1510]},
        "reason": "matched 0/5 key terms",
        "missing": ["extensional"],
    }

    text = demo.reason_text(reason)

    assert "extensional-category" in text
    assert "extensional" in text
