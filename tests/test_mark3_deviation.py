import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mark3_deviation", ROOT / "scripts" / "mark3_deviation.py")
mark3_deviation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mark3_deviation
SPEC.loader.exec_module(mark3_deviation)


def monad_entry():
    return mark3_deviation.ConceptEntry(
        concept_id="monad",
        name="monad",
        kind="object",
        depends_on=("category", "endofunctor", "natural-transformation", "identity-functor", "composition"),
        axiom_text="endofunctor multiplication unit associativity natural transformation",
        raw={},
    )


def test_structural_score_flags_synthetic_redefinition_above_clean_usage():
    entry = monad_entry()
    clean = "A monad on a category is an endofunctor with multiplication and unit natural transformations."
    bad = "We redefine monad as not an endofunctor but a left adjoint object without unit."

    clean_score, clean_evidence = mark3_deviation.structural_score(entry, clean, clean)
    bad_score, bad_evidence = mark3_deviation.structural_score(entry, bad, bad, synthetic=True)

    assert clean_score < 0.55
    assert bad_score >= 0.55
    assert bad_score > clean_score
    assert any(e.startswith("cue:") for e in bad_evidence)
    assert not any(e.startswith("cue:") for e in clean_evidence)


def test_find_usages_uses_layer_marks_and_text_fallback():
    entry = mark3_deviation.ConceptEntry(
        concept_id="monoidal-category",
        name="monoidal category",
        kind="object",
        depends_on=("category", "tensor-product"),
        axiom_text="category tensor product unit object",
        raw={},
    )
    entries = {"monoidal category": entry}
    text = "A monoidal category has tensor product. Another monoidal category appears in prose."
    first = text.index("monoidal category")
    paper = {
        "paper": "synthetic",
        "text": text,
        "marks": [
            {
                "start": first,
                "end": first + len("monoidal category"),
                "kind": "concept",
                "tip": "concept: monoidal category [synthetic]",
            }
        ],
    }

    usages = mark3_deviation.find_usages(paper, entries, context_chars=40)

    assert len(usages) == 2
    assert usages[0].anchor["mark-kind"] == "concept"
    assert {u.paper for u in usages} == {"synthetic"}


def test_write_synthetic_paper_has_concept_anchor(tmp_path):
    path = tmp_path / "bad.json"
    mark3_deviation.write_synthetic_paper(path, clean=False)
    data = json.loads(path.read_text())

    assert data["paper"] == "synthetic-redefinition"
    assert data["marks"][0]["tip"].startswith("concept: monad")
    start = data["marks"][0]["start"]
    end = data["marks"][0]["end"]
    assert data["text"][start:end].lower() == "monad"
