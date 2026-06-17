import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


golden = load_script("build_golden_paper")
prior = load_script("build_term_prior")
ency = load_script("build_concept_encyclopedia")


def test_h2_c_term_coverage_audit_reports_precision_and_recall():
    text = (
        r"\begin{definition} A \emph{test category} is a category with tests.\end{definition} "
        "A test category has a homological vector field and mirror symmetry."
    )
    audit = golden.audit_concept_term_coverage(
        text,
        ["test category", "homological vector field", "mirror symmetry"],
    )
    assert audit["sample_rows"] == 3
    assert audit["recall"] == 1.0
    assert audit["precision"] == 1.0
    assert audit["missed_terms"] == []


def test_h2_c_term_coverage_audit_quantifies_dc1_misses():
    text = "A model category has maps."
    audit = golden.audit_concept_term_coverage(
        text,
        ["model category", "triangulated category"],
    )
    assert audit["sample_rows"] == 2
    assert audit["recall"] == 0.5
    assert audit["missed_terms"] == ["triangulated category"]


def test_h3_term_prior_resolves_overfed_hungry_and_hapax_terms():
    texts = [
        "Every abelian category has kernels. The category of modules over a ring is abelian.",
        "An abelian category admits exact sequences. The category of modules over a ring is complete.",
        "Interesting abelian category examples recur. The category of modules over a ring is stable.",
        "A singleton phrase appears once.",
    ]
    df = prior.document_frequencies(texts)

    overfed = prior.resolve_phrase("interesting abelian category", df, min_papers=2)
    hungry = prior.resolve_phrase("category of modules", df, min_papers=2)
    hapax = prior.resolve_phrase("singleton phrase", df, min_papers=2)

    assert overfed["action"] == "OVERFED"
    assert overfed["resolution"] == "abelian category"
    assert overfed["resolved_df"] == 3

    assert hungry["action"] == "HUNGRY"
    assert hungry["resolution"] == "category of modules over"
    assert hungry["resolved_df"] == 3

    assert hapax["action"] == "HAPAX"
    assert hapax["resolution"] is None
    assert hapax["df"] == 1


def test_h4_concept_encyclopedia_audit_requires_typed_hole():
    complete = {
        "concept": "abelian category",
        "pagerank": 0.1,
        "depends_on": ["category"],
        "gloss": {"paper": "p1", "text": "An abelian category is a category ..."},
        "provenance": {"target": "nlab:abelian_category"},
        "holes": [ency.formalisation_hole()],
    }
    missing = {
        "concept": "thin entry",
        "pagerank": None,
        "depends_on": [],
        "gloss": {"paper": None, "text": ""},
        "provenance": {},
        "holes": [{"kind": "formalise"}],
    }
    audit = ency.audit_entries([complete, missing], sample_size=2)
    assert audit["sample_size"] == 2
    assert audit["counts"]["def_passage"] == 1
    assert audit["counts"]["provenance"] == 1
    assert audit["counts"]["dep_edge"] == 1
    assert audit["counts"]["centrality"] == 1
    assert audit["counts"]["typed_hole"] == 1


def test_h4_edn_hole_is_typed_as_hole_not_absent():
    edn = ency._edn({"holes": [{"kind": ency._Kw("hole"),
                                "type": ency._Kw("formalise-structure"),
                                "wanted": "fill"}]})
    assert ":kind :hole" in edn
    assert ":type :formalise-structure" in edn
