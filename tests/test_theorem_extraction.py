"""Tests for theorem extraction from LaTeX source."""

import json
import tempfile
from pathlib import Path

from futon6.theorem_extraction import (
    TheoremRecord,
    extract_theorems,
    extract_from_file,
    extract_from_tarball,
    to_stepper_missions,
)


# -- Fixture: realistic CT paper fragment --

SAMPLE_LATEX = r"""
\documentclass{article}
\begin{document}

\section{Preliminary Definitions}

\begin{definition}\label{def:enriched-cat}
An \emph{enriched category} over a monoidal category $(\mathcal{V}, \otimes, I)$
consists of a class of objects $\mathrm{Ob}(\mathcal{C})$, for each pair of
objects $A, B$ a hom-object $\mathcal{C}(A,B) \in \mathcal{V}$, composition
morphisms $\circ: \mathcal{C}(B,C) \otimes \mathcal{C}(A,B) \to \mathcal{C}(A,C)$,
and identity morphisms $j_A: I \to \mathcal{C}(A,A)$, satisfying associativity
and unit axioms.
\end{definition}

\section{Main Results}

\begin{theorem}[Enriched Yoneda Lemma]\label{thm:yoneda}
Let $\mathcal{C}$ be a $\mathcal{V}$-enriched category where $\mathcal{V}$
is complete. For any $\mathcal{V}$-functor $F: \mathcal{C}^{op} \to \mathcal{V}$
and object $A \in \mathcal{C}$, there is a natural isomorphism
$$[\mathcal{C}^{op}, \mathcal{V}](\mathcal{C}(-,A), F) \cong F(A)$$
in $\mathcal{V}$.
\end{theorem}
\begin{proof}
We construct the isomorphism explicitly. Define $\phi: [\mathcal{C}^{op},
\mathcal{V}](\mathcal{C}(-,A), F) \to F(A)$ by evaluating at $A$ and
composing with $j_A: I \to \mathcal{C}(A,A)$. The inverse $\psi: F(A) \to
[\mathcal{C}^{op}, \mathcal{V}](\mathcal{C}(-,A), F)$ sends $x \in F(A)$
to the natural transformation with components
$F(f)(x): \mathcal{C}(B,A) \to F(B)$ for each $B$.
Naturality follows from the enriched functoriality of $F$, and
$\phi \circ \psi = \mathrm{id}$ and $\psi \circ \phi = \mathrm{id}$
are verified by direct calculation using the unit axioms of $\mathcal{C}$.
See also \ref{def:enriched-cat} for the definition of enriched category.
\end{proof}

\begin{lemma}\label{lem:representable}
Every representable $\mathcal{V}$-functor $\mathcal{C}(-,A)$ preserves
all $\mathcal{V}$-enriched limits that exist in $\mathcal{C}$.
\end{lemma}

\begin{proposition}
If $\mathcal{V}$ is a cosmos (complete and cocomplete symmetric monoidal
closed category), then the category $[\mathcal{C}^{op}, \mathcal{V}]$ of
$\mathcal{V}$-presheaves on $\mathcal{C}$ is again a cosmos.
\end{proposition}
\begin{proof}
Limits and colimits in $[\mathcal{C}^{op}, \mathcal{V}]$ are computed
pointwise, inheriting completeness and cocompleteness from $\mathcal{V}$.
The internal hom is given by the end formula
$$[F,G](A) = \int_{B} [F(B), G(B)]_{\mathcal{V}}$$
which exists by completeness of $\mathcal{V}$.
\end{proof}

\section{Conjectures}

\begin{conjecture}\label{conj:stabilization}
The $k$-fold suspension of a weak $n$-category stabilizes for $k \geq n+2$.
\end{conjecture}

\begin{corollary}
Under the hypotheses of Theorem \ref{thm:yoneda}, the Yoneda embedding
$\mathcal{C} \hookrightarrow [\mathcal{C}^{op}, \mathcal{V}]$ is
fully faithful.
\end{corollary}

\end{document}
"""


def test_extract_theorems_basic():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    assert result.paper_id == "test/0001"
    assert len(result.theorems) == 5  # theorem, lemma, proposition, conjecture, corollary


def test_theorem_types():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    types = [t.env_type for t in result.theorems]
    assert "theorem" in types
    assert "lemma" in types
    assert "proposition" in types
    assert "conjecture" in types
    assert "corollary" in types


def test_theorem_labels():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    labeled = {t.label: t for t in result.theorems if t.label}
    assert "thm:yoneda" in labeled
    assert "lem:representable" in labeled
    assert "conj:stabilization" in labeled


def test_theorem_names():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    yoneda = next(t for t in result.theorems if t.label == "thm:yoneda")
    assert yoneda.name == "Enriched Yoneda Lemma"


def test_proof_pairing():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    yoneda = next(t for t in result.theorems if t.label == "thm:yoneda")
    assert yoneda.has_proof
    assert "evaluating at $A$" in yoneda.proof

    # The proposition also has a proof
    prop = next(t for t in result.theorems if t.env_type == "proposition")
    assert prop.has_proof
    assert "pointwise" in prop.proof

    # Lemma has no immediately-following proof
    lemma = next(t for t in result.theorems if t.label == "lem:representable")
    assert not lemma.has_proof


def test_cross_references():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    # The corollary references thm:yoneda
    corollary = next(t for t in result.theorems if t.env_type == "corollary")
    assert "thm:yoneda" in corollary.refs

    # The proof of yoneda references def:enriched-cat
    yoneda = next(t for t in result.theorems if t.label == "thm:yoneda")
    assert "def:enriched-cat" in yoneda.proof_refs


def test_section_detection():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    yoneda = next(t for t in result.theorems if t.label == "thm:yoneda")
    assert yoneda.section == "Main Results"

    conj = next(t for t in result.theorems if t.env_type == "conjecture")
    assert conj.section == "Conjectures"


def test_definitions_extracted():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    assert len(result.definitions) == 1
    assert result.definitions[0]["env_type"] == "definition"
    assert result.definitions[0]["label"] == "def:enriched-cat"


def test_theorem_id_stable():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    yoneda = next(t for t in result.theorems if t.label == "thm:yoneda")
    assert yoneda.theorem_id == "test/0001::thm:yoneda"

    # Unlabeled theorem gets hash-based ID
    prop = next(t for t in result.theorems if t.env_type == "proposition")
    assert prop.theorem_id.startswith("test/0001::proposition-")


def test_statement_hash_dedup():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    hashes = [t.statement_hash for t in result.theorems]
    assert len(hashes) == len(set(hashes)), "All hashes should be unique"


def test_stats():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    stats = result.stats
    assert stats["theorems"] == 5
    assert stats["with_proof"] == 2
    assert stats["with_name"] == 1
    assert stats["with_label"] == 3
    assert stats["definitions"] == 1


def test_number_hints():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    yoneda = next(t for t in result.theorems if t.label == "thm:yoneda")
    assert yoneda.number_hint == "Theorem 1"

    lemma = next(t for t in result.theorems if t.label == "lem:representable")
    assert lemma.number_hint == "Lemma 1"


def test_to_dict_roundtrip():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    d = result.to_dict()
    assert d["paper_id"] == "test/0001"
    assert len(d["theorems"]) == 5
    # Should be JSON-serializable
    json.dumps(d)


def test_extract_from_file():
    with tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False) as f:
        f.write(SAMPLE_LATEX)
        f.flush()
        result = extract_from_file(f.name, "test/file")
    assert len(result.theorems) == 5
    assert result.source_path == f.name
    Path(f.name).unlink()


def test_to_stepper_missions():
    result = extract_theorems(SAMPLE_LATEX, "q-alg/9503002")
    missions = to_stepper_missions(result.theorems)

    assert len(missions) >= 3  # at least theorem, lemma, proposition

    # Check mission structure
    m = missions[0]
    assert "canonical" in m
    assert "statement" in m["canonical"]
    assert "closure_criterion" in m["canonical"]
    assert "statement_hash" in m["canonical"]
    assert "ledger_template" in m
    assert len(m["ledger_template"]) == 5  # L-claim-type through L-conclusion
    assert "corpus_queries" in m
    assert "generic" in m["corpus_queries"]
    assert "domain" in m["corpus_queries"]

    # Conjecture gets different closure criterion
    conj_mission = next(m for m in missions if m["env_type"] == "conjecture")
    assert "disprove" in conj_mission["canonical"]["closure_criterion"].lower()


def test_to_stepper_missions_min_length():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    # With very high min_statement_len, nothing qualifies
    missions = to_stepper_missions(result.theorems, min_statement_len=10000)
    assert len(missions) == 0


def test_to_stepper_missions_require_proof():
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    missions = to_stepper_missions(result.theorems, require_proof=True)
    # Only theorem (yoneda) and proposition have proofs
    assert len(missions) == 2
    assert all(m["source"]["has_proof"] for m in missions)


def test_ledger_dag_structure():
    """Verify the ledger template enforces framing-first discipline."""
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    missions = to_stepper_missions(result.theorems)
    ledger = {item["id"]: item for item in missions[0]["ledger_template"]}

    # L-claim-type has no dependencies (entry point)
    assert ledger["L-claim-type"]["depends_on"] == []

    # L-preconditions and L-obstruction-scan depend on L-claim-type
    assert "L-claim-type" in ledger["L-preconditions"]["depends_on"]
    assert "L-claim-type" in ledger["L-obstruction-scan"]["depends_on"]

    # L-bridge depends on BOTH preconditions and obstruction-scan
    assert "L-preconditions" in ledger["L-bridge"]["depends_on"]
    assert "L-obstruction-scan" in ledger["L-bridge"]["depends_on"]

    # L-conclusion depends on L-bridge
    assert "L-bridge" in ledger["L-conclusion"]["depends_on"]


def test_empty_source():
    result = extract_theorems("", "test/empty")
    assert len(result.theorems) == 0
    assert result.stats["theorems"] == 0


def test_nested_math_in_theorem():
    """Theorem containing display math should be extracted fully."""
    result = extract_theorems(SAMPLE_LATEX, "test/0001")
    yoneda = next(t for t in result.theorems if t.label == "thm:yoneda")
    assert r"\cong" in yoneda.statement
    assert "F(A)" in yoneda.statement
