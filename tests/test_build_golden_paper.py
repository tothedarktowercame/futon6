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


def load_dp_enrich():
    spec = importlib.util.spec_from_file_location(
        "dp_enrich", ROOT / "scripts" / "dp_enrich.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_mine_definitions_reads_a_definition_in_prose():
    """A paper with no definition environment still defines terms.

    0806.1324 (Krause) has none at all: reading environments only found one
    definiendum in the whole paper, so S1 marked 1 occurrence in 3,277 as a
    term the paper defines.
    """
    text = r"A category $\C$ is called \emph{small} if its objects form a set."
    terms = {d.term for d in golden.mine_definitions(text)}
    assert "small" in terms


def test_mine_definitions_takes_the_emphasis_not_the_clause_around_it():
    """The definiendum is what is emphasised, not the sentence that frames it.

    This produced terms like `a {\\it trivial fibration} if it is both a
    fibration and a weak equivalence`, markup and trailing clause included.
    """
    text = r"We call a map a {\it trivial fibration} if it is both a fibration and a weak equivalence."
    terms = {d.term for d in golden.mine_definitions(text)}
    assert "trivial fibration" in terms
    assert not any("{" in t or " if it is" in t for t in terms), terms


def test_mine_definitions_ignores_italicised_references():
    """A bibliography sets journal and publisher names in italics too."""
    text = (r"See \emph{J. Math. Phys.} and \emph{Springer-Verlag} and "
            r"\emph{preprint math.QA/9802029} for the history.")
    assert golden.mine_definitions(text) == []


def test_mine_definitions_reads_the_older_emphasis_spelling():
    text = r"An object is called {\em rigid} if it has no deformations."
    assert "rigid" in {d.term for d in golden.mine_definitions(text)}


def test_definition_marks_carry_the_reading_that_found_them():
    """A consumer must be able to tell a framed definition from a bare italic."""
    framed = r"A functor is called \emph{regular} if it preserves limits."
    marks = golden.definition_marks(framed, golden.mine_definitions(framed))
    assert {m.source for m in marks} == {"called-by-name"}


def test_select_non_overlapping_agrees_with_the_quadratic_reading():
    """The binary search must select exactly what scanning every span did."""
    import random
    random.seed(11)
    marks = [golden.Mark(start=s, end=s + ln, kind=k, title="", label=str(i))
             for i, (s, ln, k) in enumerate(
                 (random.randrange(0, 4000), random.randrange(1, 40),
                  random.choice(["bind", "defined", "hole"])) for _ in range(3000))]

    priority = {"bind": 0, "defined": 1, "hole": 2}
    ordered = sorted(marks, key=lambda m: (priority[m.kind], -(m.end - m.start), m.start))
    accepted, occupied = [], []
    for mark in ordered:
        if mark.end <= mark.start:
            continue
        if any(not (mark.end <= s or mark.start >= e) for s, e in occupied):
            continue
        accepted.append(mark)
        occupied.append((mark.start, mark.end))
    expected = sorted(accepted, key=lambda m: m.start)

    got = golden.select_non_overlapping(marks)
    assert [(m.start, m.end, m.label) for m in got] == \
           [(m.start, m.end, m.label) for m in expected]
    # and it really is a non-overlapping selection
    for a, b in zip(got, got[1:]):
        assert a.end <= b.start


def test_framed_and_bare_single_word_definienda_are_told_apart():
    """dp_enrich lets a one-word definiendum through only when the paper framed it.

    "is called \\emph{regular}" is a definition; a lone italicised "except" is
    stress. Both are single words, so only the reading that found them can
    separate them, and the gate in dp_enrich.concept_marks reads exactly the
    `source` set asserted here.
    """
    dp = load_dp_enrich()

    framed = r"A functor is called \emph{regular} if it preserves limits."
    framed_sources = {d.source for d in golden.mine_definitions(framed)}
    assert framed_sources <= dp.DEFINITION_FRAMED, framed_sources

    bare = r"Every map is continuous, \emph{except} on the boundary."
    bare_defs = golden.mine_definitions(bare)
    assert bare_defs, "the miner should still see it; the gate is what rejects it"
    assert {d.source for d in bare_defs}.isdisjoint(dp.DEFINITION_FRAMED)
