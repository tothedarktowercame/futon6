import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rung3_residue_llm as r33  # noqa: E402
import cas_cert  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "rung3-residue" / "gapmap.json"


def gapmap():
    return json.loads(FIXTURE.read_text())


def test_residue_only_and_bounded():
    """LLM runs only on the thin/ungrounded residue, never the grounded move."""
    doc = r33.questions_for_gapmap(gapmap(), backend="stub")
    steps = [q["step"] for q in doc["questions"]]
    assert steps == ["s2", "s3"]            # the two gaps, in order
    assert "s1" not in steps                # grounded move never touched
    assert doc["summary"]["residue"] == 2
    assert doc["summary"]["asked"] == 2


def test_question_not_verdict():
    """Each residue gap yields a phrased question + a gap-type classification — never
    a truth/correctness verdict."""
    doc = r33.questions_for_gapmap(gapmap(), backend="stub")
    for q in doc["questions"]:
        assert q["classification"] in r33.CLASSIFICATIONS   # gap-type, not true/false
        assert q["question"].strip().endswith("?")          # it is a question
        assert q["rm_pattern"]                               # menu-grounded
        assert q["ref"].startswith("arse:")                 # ArSE :ref shape (not opened)


def test_menu_grounded_phrasing():
    doc = r33.questions_for_gapmap(gapmap(), backend="stub")
    by_step = {q["step"]: q for q in doc["questions"]}
    assert by_step["s2"]["rm_pattern"] == "STRUCTURAL PROBE"
    assert "find-the-right-abstraction" in by_step["s2"]["question"]
    assert by_step["s3"]["rm_pattern"].startswith("THEOREM APPLICABILITY")


def test_stub_is_deterministic():
    a = r33.questions_for_gapmap(gapmap(), backend="stub")
    b = r33.questions_for_gapmap(gapmap(), backend="stub")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_max_questions_budget():
    doc = r33.questions_for_gapmap(gapmap(), backend="stub", max_questions=1)
    assert doc["summary"]["asked"] == 1
    assert doc["summary"]["dropped_by_budget"] == 1


def _minimal_graph(paper_id="fixture"):
    # one inference edge with a missing warrant -> a non-trivial cert with an empty port
    return {
        "paper-id": paper_id,
        "nodes": [
            {"id": ":a", "kind": ":claim", "text": "a", "source": {"lines": [1, 1]}},
            {"id": ":b", "kind": ":claim", "text": "b", "source": {"lines": [2, 2]}},
        ],
        "edges": [
            {"id": ":e", "kind": ":infer", "relation": ":implies", "premise": ":a",
             "warrant": {"kind": ":missing-warrant"}, "conclusion": ":b", "source": {"lines": [1, 2]}},
        ],
        "holes": [],
    }


def test_cert_enrichment_is_report_only():
    """cas_cert --questions adds open_questions WITHOUT changing any grain rate or gate."""
    doc = r33.questions_for_gapmap(gapmap(), backend="stub")
    qbp = {"fixture": doc}
    graph = _minimal_graph("fixture")

    without = cas_cert.certificate_for_graph(graph)
    with_q = cas_cert.certificate_for_graph(graph, questions_by_paper=qbp)

    # open_questions populated only when --questions supplied
    assert without.get("open_questions", []) == []
    assert len(with_q["open_questions"]) == 2

    # report-only: grain vectors + gate verdict byte-identical
    assert json.dumps(with_q["conformance"]["by_grain"], sort_keys=True) == \
           json.dumps(without["conformance"]["by_grain"], sort_keys=True)
    assert with_q["verdict"] == without["verdict"]
