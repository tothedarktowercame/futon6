from collections import Counter

from scripts import sfc_concept_coverage as sfc


def test_invert_usage_counts_documents_once_per_concept():
    paper_concepts = {
        "p1": ["Natural Transformation", "natural transformation", "there exists"],
        "p2": ["natural transformations", "left adjoint"],
        "p3": ["left adjoint"],
    }

    df = sfc.invert_usage(paper_concepts)

    assert df["natural transformation"] == 2
    assert df["left adjoint"] == 2
    assert df["there exists"] == 1


def test_normalize_concept_merges_named_fragments_and_variants():
    examples = {
        "non commutative": "non-commutative",
        "unit counit": "unit-counit",
        "algebra topology": "algebraic topology",
        "n categories": "n-categories",
        "quasi inverse": "quasi-inverse",
        "quasi isomorphisms": "quasi-isomorphism",
        "hom spaces": "hom-spaces",
        "natural transformations": "natural transformation",
        "monoidal categories": "monoidal category",
        "vector spaces": "vector space",
    }

    for raw, canonical in examples.items():
        assert sfc.normalize_concept(raw) == canonical


def test_resolved_genuine_concept_filters_boilerplate_and_keeps_terms():
    df = Counter(
        {
            "there exists": 10,
            "more generally": 9,
            "natural transformation": 8,
            "monoidal natural transformation": 3,
            "left adjoint": 7,
        }
    )

    assert sfc.resolved_genuine_concept("there exists", df, min_papers=3)[0] is None
    assert sfc.resolved_genuine_concept("more generally", df, min_papers=3)[0] is None
    assert (
        sfc.resolved_genuine_concept("natural transformation", df, min_papers=3)[0]
        == "natural transformation"
    )
    assert sfc.resolved_genuine_concept("left adjoint", df, min_papers=3)[0] == "left adjoint"


def test_coverage_sources_and_summary():
    ranked = [
        {"concept": "natural transformation", "df": 8, "score": 8.0, "pagerank": 0.0,
         "resolution_action": "KEPT:1", "input_examples": ()},
        {"concept": "left adjoint", "df": 7, "score": 7.0, "pagerank": 0.0,
         "resolution_action": "KEPT:1", "input_examples": ()},
    ]
    sources = sfc.definition_sets(
        {"snippets": {"natural transformation": [{"paper": "p1"}]}},
        {"concept_to_papers": {"natural transformation": ["p1"]}},
        {"entries": [{"concept": "left adjoint", "gloss": {"text": "defined"}}]},
    )

    covered = sfc.attach_coverage(ranked, sources)
    summary = sfc.coverage_summary(covered, 2)

    assert covered[0].defined
    assert covered[0].sources == ("def-snippets", "defined-index")
    assert covered[1].defined
    assert covered[1].sources == ("concept-encyclopedia",)
    assert summary["coverage"] == 1.0
