import importlib.util
import json
import re
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

BASELINE_PRECISION = 0.0724
BASELINE_RECALL = 0.14


def strict_h2_sample_rows(limit=100):
    endings = "|".join(golden.CONCEPT_ENDINGS)
    word = r"[A-Za-z][A-Za-z-]*"
    pat = re.compile(rf"\b(?P<phrase>{word}(?:\s+{word}){{0,4}}\s+(?:{endings}))\b")
    rows = []
    for path in sorted((ROOT / "data/showcases/ct-anatomy/golden").glob("fable-*-dp-emacs.json")):
        text = json.loads(path.read_text()).get("text", "")
        body = text[text.find(r"\begin{document}"):] if r"\begin{document}" in text else text
        for match in pat.finditer(body):
            phrase = re.sub(r"\s+", " ", match.group("phrase")).strip()
            if len(phrase) < 8:
                continue
            context = body[max(0, match.start() - 120):match.end() + 120]
            rows.append((phrase, context))
            if len(rows) >= limit:
                return rows
    return rows


def aggregate_h2_audit(rows):
    agg = {
        "sample_rows": 0,
        "expected_terms": 0,
        "concept_marks": 0,
        "true_positive_marks": 0,
        "false_positive_marks": 0,
        "missed_terms": [],
    }
    for phrase, context in rows:
        audit = golden.audit_concept_term_coverage(context, [phrase])
        for key in ["sample_rows", "expected_terms", "concept_marks",
                    "true_positive_marks", "false_positive_marks"]:
            agg[key] += audit[key]
        agg["missed_terms"].extend(audit["missed_terms"])
    agg["precision"] = (
        agg["true_positive_marks"] / agg["concept_marks"]
        if agg["concept_marks"] else 1.0
    )
    agg["recall"] = (
        (agg["expected_terms"] - len(agg["missed_terms"])) / agg["expected_terms"]
        if agg["expected_terms"] else 1.0
    )
    return agg


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


def test_h2_strict_sample_regression_floor_beats_baseline():
    audit = aggregate_h2_audit(strict_h2_sample_rows())
    assert audit["sample_rows"] == 100
    assert audit["precision"] > BASELINE_PRECISION
    assert audit["recall"] > BASELINE_RECALL


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
