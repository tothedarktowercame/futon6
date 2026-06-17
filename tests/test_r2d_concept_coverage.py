from pathlib import Path

import scripts.r2d_concept_coverage as r2d


def substrate() -> r2d.Substrate:
    args = r2d.parse_args([])
    return r2d.load_substrate(args)


def test_worked_extensional_category_case_reproduces_buckets():
    graph = r2d.load_edn(Path("data/iatc-argument-graphs/loop-run-70b/0709.0248.edn"))
    result = r2d.check_graph(
        graph,
        Path("data/iatc-argument-graphs/loop-run-70b/0709.0248.edn"),
        substrate(),
    )

    by_concept = {row["concept"]: row for row in result["per-item"]}
    assert result["check"] == ":concept-coverage"
    assert result["pass"] is True
    assert result["buckets"] == {"defined": 8, "known": 0, "imported": 0, "undefined": 2}
    assert by_concept["extensional category"]["bucket"] == "defined"
    assert by_concept["parameterized rules"]["bucket"] == "undefined"
    assert by_concept["standard rules"]["bucket"] == "undefined"


def test_known_recurring_core_threshold_is_report_only():
    sub = substrate()
    row = r2d.classify_concept("ab category", sub)

    assert row["bucket"] == "known"
    assert "recurring core" in row["reason"]


def test_no_extractable_concepts_is_na_not_fail():
    result = r2d.check_graph(
        {"paper/id": "9999.0001", "nodes": [], "edges": []},
        Path("9999.0001.edn"),
        substrate(),
    )

    assert result["status"] == ":na"
    assert result["pass"] is True
    assert result["rate"] is None
