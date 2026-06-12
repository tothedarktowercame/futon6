import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_golden_paper", ROOT / "scripts" / "build_golden_paper.py"
)
golden = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = golden
SPEC.loader.exec_module(golden)


def test_repair_truncated_subscripts_uses_in_paper_attestation():
    text = (
        r"$\mathcal{A}_{\infty}$-algebra "
        r"$\mathcal{A}_{\infty}$-category "
        r"$\mathcal{A}_$-category"
    )
    repaired, log = golden.repair_truncated_subscripts(text)
    assert r"$\mathcal{A}_{\infty}$-category" in repaired
    assert log[0].damaged == r"\mathcal{A}_"
    assert log[0].replacement == r"\mathcal{A}_{\infty}"
    assert log[0].attestations == 2


def test_mine_definitions_and_occurrence_variants():
    text = (
        r"\newtheorem{defn}{Definition}"
        r"\begin{defn} A \emph{test category} is a category with tests.\end{defn}"
        r"Every test category has maps. Test categories recur."
    )
    definitions = golden.mine_definitions(text)
    terms = {d.term for d in definitions}
    assert "test category" in terms
    marks = golden.definition_marks(text, definitions)
    marked = [text[m.start:m.end] for m in marks]
    assert "test category" in marked
    assert "Test categories" in marked


def test_appositive_bind_marks_symbol_type_phrase():
    text = r"Namely, the Fukaya category $\mathcal{F}(X)$ is associated to a symplectic manifold $X$."
    marks = golden.appositive_bind_marks(text)
    labels = {m.label for m in marks}
    assert "Fukaya category" in labels
    assert "symplectic manifold" in labels


def test_hole_marks_skip_defined_terms():
    text = "A model category has a homological vector field and mirror symmetry."
    definitions = [golden.Definition("model category", 0, "test")]
    holes = golden.hole_marks(text, definitions)
    labels = {m.label for m in holes}
    assert "model category" not in labels
    assert "homological vector field" in labels
    assert "mirror symmetry" in labels
