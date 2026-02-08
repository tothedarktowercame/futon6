"""Tests for PlanetMath loading and graph construction."""

import os
import pytest
from futon6.planetmath import (
    load_edn,
    entries_to_entities,
    entries_to_relations,
    extract_term_entities,
    extract_msc_entities,
    load_tex_dir,
    merge_tex_bodies,
    build_graph,
)

CATTHEORY_EDN = os.path.expanduser("~/code/planetmath/category-theory.edn")
CATTHEORY_TEX = os.path.expanduser(
    "~/code/planetmath/18_Category_theory_homological_algebra"
)


@pytest.fixture
def cat_entries():
    return load_edn(CATTHEORY_EDN)


@pytest.fixture
def cat_graph():
    return build_graph(CATTHEORY_EDN)


class TestLoadEdn:
    def test_loads_entries(self, cat_entries):
        assert len(cat_entries) == 313

    def test_entry_has_required_keys(self, cat_entries):
        for e in cat_entries:
            assert "id" in e, f"Entry missing id: {e}"
            assert "title" in e, f"Entry missing title: {e}"

    def test_entry_types_present(self, cat_entries):
        types = {e.get("type") for e in cat_entries}
        assert "Definition" in types
        assert "Theorem" in types
        assert "Example" in types


class TestEntitiesToGraph:
    def test_entities_have_futon_shape(self, cat_entries):
        entities = entries_to_entities(cat_entries)
        for ent in entities:
            assert "entity/id" in ent
            assert "entity/type" in ent
            assert "entity/source" in ent

    def test_relations_have_futon_shape(self, cat_entries):
        relations = entries_to_relations(cat_entries)
        for rel in relations:
            assert "relation/id" in rel
            assert "relation/from" in rel
            assert "relation/to" in rel
            assert "relation/type" in rel

    def test_relation_types(self, cat_entries):
        relations = entries_to_relations(cat_entries)
        types = {r["relation/type"] for r in relations}
        assert "related-to" in types
        assert "defines" in types
        assert "classified-by" in types

    def test_term_entities(self, cat_entries):
        terms = extract_term_entities(cat_entries)
        assert len(terms) > 0
        for t in terms:
            assert t["entity/type"] == "DefinedTerm"
            assert t["entity/id"].startswith("term:")

    def test_msc_entities(self, cat_entries):
        mscs = extract_msc_entities(cat_entries)
        assert len(mscs) > 0
        for m in mscs:
            assert m["entity/type"] == "MSCCode"
            assert m["entity/id"].startswith("msc:")

    def test_msc_parent_codes(self, cat_entries):
        mscs = extract_msc_entities(cat_entries)
        parents = [m for m in mscs if m.get("parent")]
        assert len(parents) > 0  # most codes have a 2-char parent


class TestBuildGraph:
    def test_graph_stats(self, cat_graph):
        stats = cat_graph["stats"]
        assert stats["entries"] >= 312  # 313 minus deduped source duplicates
        assert stats["terms"] > 100
        assert stats["msc_codes"] > 50
        assert stats["relations"] > 500

    def test_entity_ids_unique(self, cat_graph):
        ids = [e["entity/id"] for e in cat_graph["entities"]]
        assert len(ids) == len(set(ids)), "Duplicate entity IDs found"

    def test_relation_ids_unique(self, cat_graph):
        ids = [r["relation/id"] for r in cat_graph["relations"]]
        assert len(ids) == len(set(ids)), "Duplicate relation IDs found"

    def test_graph_has_mixed_entity_types(self, cat_graph):
        types = {e["entity/type"] for e in cat_graph["entities"]}
        assert "Definition" in types
        assert "DefinedTerm" in types
        assert "MSCCode" in types


class TestTexLoading:
    @pytest.fixture
    def tex_data(self):
        return load_tex_dir(CATTHEORY_TEX)

    def test_loads_tex_files(self, tex_data):
        assert len(tex_data) > 100  # category theory has hundreds of .tex files

    def test_tex_entry_has_fields(self, tex_data):
        for canonical, data in tex_data.items():
            assert "canonical_name" in data
            assert "body_full" in data
            assert "source_tex" in data

    def test_tex_bodies_nonempty(self, tex_data):
        nonempty = [d for d in tex_data.values() if d["body_full"]]
        assert len(nonempty) > 50

    def test_synonyms_extracted(self, tex_data):
        all_synonyms = [s for d in tex_data.values() for s in d.get("synonyms", [])]
        assert len(all_synonyms) > 50


class TestMergeTexBodies:
    def test_merge_enriches_entries(self):
        entries = load_edn(CATTHEORY_EDN)
        tex_data = load_tex_dir(CATTHEORY_TEX)
        merged = merge_tex_bodies(entries, tex_data)
        assert len(merged) == len(entries)
        # Some entries should now have full body text
        bodies_changed = sum(
            1 for orig, m in zip(entries, merged)
            if m.get("body", "") != orig.get("body", "")
        )
        assert bodies_changed > 50

    def test_synonyms_added_to_defines(self):
        entries = load_edn(CATTHEORY_EDN)
        tex_data = load_tex_dir(CATTHEORY_TEX)
        merged = merge_tex_bodies(entries, tex_data)
        total_defines_before = sum(len(e.get("defines", [])) for e in entries)
        total_defines_after = sum(len(e.get("defines", [])) for e in merged)
        assert total_defines_after > total_defines_before


class TestBuildGraphWithTex:
    @pytest.fixture
    def graph_with_tex(self):
        return build_graph(CATTHEORY_EDN, tex_dir=CATTHEORY_TEX)

    def test_more_terms_with_tex(self, graph_with_tex):
        """With .tex bodies, synonym extraction adds many more defined terms."""
        graph_no_tex = build_graph(CATTHEORY_EDN)
        assert graph_with_tex["stats"]["terms"] > graph_no_tex["stats"]["terms"]

    def test_more_relations_with_tex(self, graph_with_tex):
        """With .tex bodies, related links from preamble add more relations."""
        graph_no_tex = build_graph(CATTHEORY_EDN)
        assert graph_with_tex["stats"]["relations"] > graph_no_tex["stats"]["relations"]

    def test_graph_entity_ids_still_unique(self, graph_with_tex):
        ids = [e["entity/id"] for e in graph_with_tex["entities"]]
        assert len(ids) == len(set(ids))
