import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "xdoc", Path(__file__).resolve().parent.parent / "scripts" / "mark3_xdoc_graph.py")
xdoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xdoc)


def _mk(tmp_path):
    g, c = tmp_path / "g", tmp_path / "c"
    g.mkdir(); c.mkdir()
    (g / "A.edn").write_text(
        '{:paper/id "A" :nodes [{:id :n1 :kind :claim :text "trivial step"}] '
        ':holes [{:kind :missing-warrant :edge :e1 :wanted :exactness-of-the-snake-sequence}]}')
    (g / "B.edn").write_text(
        '{:paper/id "B" :nodes [{:id :m1 :kind :claim :text "the snake sequence is exact"}] :holes []}')
    (c / "A.cite-resolution.json").write_text(json.dumps(
        {"paper-id": "A", "records": [{"cite/marker": "[1]", "resolved-corpus-id": "B",
                                       "title": "Snake", "confidence": 0.9}]}))
    return g, c


def test_citation_edge_and_cross_doc_discharge(tmp_path):
    g, c = _mk(tmp_path)
    out = xdoc.build(g, c)
    assert out["stats"]["citation-edges"] == 1
    d = out["discharge-candidates"]
    assert len(d) == 1 and d[0]["hole-paper"] == "A" and d[0]["discharged-by-paper"] == "B"
    # hole token "exactness" vs claim token "exact" don't match (no stemming); the
    # discharge fires on the shared "snake"/"sequence" tokens (>= MIN_OVERLAP).
    assert {"snake", "sequence"} <= set(d[0]["shared-tokens"])


def test_no_discharge_without_citation(tmp_path):
    """A hole matching B's claim must NOT discharge if A doesn't cite B."""
    g, c = _mk(tmp_path)
    (c / "A.cite-resolution.json").write_text(json.dumps({"paper-id": "A", "records": []}))
    out = xdoc.build(g, c)
    assert out["stats"]["citation-edges"] == 0
    assert out["stats"]["discharge-candidates"] == 0


def test_tokens_drop_stopwords():
    assert "the" not in xdoc.toks("the exact sequence")
    assert {"exact", "sequence"} <= xdoc.toks("the exact sequence")
