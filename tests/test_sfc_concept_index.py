from scripts import sfc_concept_coverage as coverage
from scripts import sfc_concept_index as indexer


def test_build_index_collects_paper_lists_and_sfc_flags():
    usage = {
        "paper_concepts": {
            "p2": ["Natural Transformation", "natural transformation", "there exists"],
            "p1": ["natural transformation", "left adjoint"],
            "p3": ["left adjoint"],
        }
    }

    index, ranked = indexer.build_index(
        usage=usage,
        def_snippets={"snippets": {"natural transformation": [{"paper": "p1"}]}},
        defined_index={"concept_to_papers": {"natural transformation": ["p1"]}},
        encyclopedia={"entries": [{"concept": "left adjoint", "gloss": {"text": "defined"}}]},
        min_papers=1,
    )

    df = coverage.invert_usage(usage["paper_concepts"])

    assert index["natural transformation"]["df"] == df["natural transformation"] == 2
    assert index["natural transformation"]["papers"] == ["p1", "p2"]
    assert index["natural transformation"]["genuine"] is True
    assert index["natural transformation"]["defined"] is True
    assert index["natural transformation"]["sources"] == ["def-snippets", "defined-index"]

    assert index["left adjoint"]["df"] == df["left adjoint"] == 2
    assert index["left adjoint"]["papers"] == ["p1", "p3"]
    assert index["left adjoint"]["defined"] is True
    assert index["left adjoint"]["sources"] == ["concept-encyclopedia"]

    assert index["there exists"]["genuine"] is False
    assert index["there exists"]["defined"] is False

    indexer.validate_index(index, usage)
    assert [row.concept for row in ranked[:2]] == ["left adjoint", "natural transformation"]
