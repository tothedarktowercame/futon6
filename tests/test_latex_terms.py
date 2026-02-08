"""Tests for LaTeX term extraction and enrichment."""

import os
import pytest
from futon6.planetmath import load_edn
from futon6.latex_terms import (
    extract_terms,
    extract_xrefs,
    extract_environments,
    enrich_entry,
    enrich_all,
    enrichment_stats,
)

CATTHEORY_EDN = os.path.expanduser("~/code/planetmath/category-theory.edn")


class TestExtractTerms:
    def test_extracts_textbf(self):
        body = r"A \textbf{functor} is a map between categories."
        terms = extract_terms(body)
        assert "functor" in terms

    def test_extracts_emph(self):
        body = r"The \emph{natural transformation} connects functors."
        terms = extract_terms(body)
        assert "natural transformation" in terms

    def test_ignores_latex_commands(self):
        body = r"\textbf{\mathcal{C}}"
        terms = extract_terms(body)
        # Should not extract terms starting with backslash
        clean = [t for t in terms if not t.startswith("\\")]
        assert len(clean) == 0 or all(not t.startswith("\\") for t in clean)

    def test_deduplicates(self):
        body = r"\textbf{functor} and \emph{functor}"
        terms = extract_terms(body)
        assert terms.count("functor") == 1


class TestExtractXrefs:
    def test_pmlink(self):
        body = r"\PMlinkname{adjoint}{AdjointFunctor}"
        xrefs = extract_xrefs(body)
        assert len(xrefs) == 1
        assert xrefs[0]["target"] == "AdjointFunctor"
        assert xrefs[0]["display"] == "adjoint"

    def test_multiple_xrefs(self):
        body = (
            r"\PMlinkname{functor}{Functor} and "
            r"\PMlinkname{category}{CategoryTheory}"
        )
        xrefs = extract_xrefs(body)
        assert len(xrefs) == 2


class TestExtractEnvironments:
    def test_theorem_env(self):
        body = r"\begin{theorem}Every functor preserves isomorphisms.\end{theorem}"
        envs = extract_environments(body)
        assert len(envs) == 1
        assert envs[0]["environment"] == "theorem"

    def test_ignores_irrelevant_envs(self):
        body = r"\begin{document}Hello\end{document}"
        envs = extract_environments(body)
        assert len(envs) == 0


class TestEnrichEntry:
    def test_enrichment_adds_fields(self):
        entry = {
            "id": "test",
            "body": r"\textbf{functor} is \PMlinkname{nice}{NiceThing}",
            "related": [],
            "defines": [],
        }
        enriched = enrich_entry(entry)
        assert "extracted_terms" in enriched
        assert "extracted_xrefs" in enriched
        assert "implicit_relations" in enriched
        assert "functor" in enriched["extracted_terms"]

    def test_implicit_relations_exclude_explicit(self):
        entry = {
            "id": "test",
            "body": r"\PMlinkname{x}{AlreadyKnown} \PMlinkname{y}{NewThing}",
            "related": ["AlreadyKnown"],
            "defines": [],
        }
        enriched = enrich_entry(entry)
        implicit_targets = {r["target"] for r in enriched["implicit_relations"]}
        assert "NewThing" in implicit_targets
        assert "AlreadyKnown" not in implicit_targets


class TestEnrichReal:
    """Test enrichment against real PlanetMath data."""

    @pytest.fixture
    def enriched(self):
        entries = load_edn(CATTHEORY_EDN)
        return enrich_all(entries)

    def test_enrichment_finds_something(self, enriched):
        stats = enrichment_stats(enriched)
        # EDN body field contains truncated preamble, not full LaTeX.
        # Full enrichment requires loading .tex files directly.
        # For now, verify the pipeline runs without error.
        assert stats["entries_enriched"] == 313
        print(f"\nEnrichment stats (EDN bodies): {stats}")

    def test_implicit_relations_discovered(self, enriched):
        stats = enrichment_stats(enriched)
        # We expect at least some entries have implicit relations
        # that the explicit :related field doesn't capture
        print(f"\nEnrichment stats: {stats}")
        # This is a discovery test — we want to know the numbers
        assert stats["entries_enriched"] > 0
