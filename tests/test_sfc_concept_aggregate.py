import json
from pathlib import Path

from scripts import sfc_concept_aggregate as agg


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "warp" / "sfc-adjunction-fixture.json"


def test_surface_to_core_examples_are_gc_retained():
    # substrate-coupled: reads the LIVE regenerated fixture. The WARP-ORCH-4 rebuild
    # re-normalized surface forms ("all functors" -> "all functor") and GC'd some
    # singletons ("any two"); these three folds are present in the current fixture.
    data = json.loads(FIXTURE.read_text())
    surface_to_core = data["surface_to_core"]

    assert surface_to_core["all functor"]["action"] == "fold"
    assert surface_to_core["all functor"]["core"] == "functor"
    assert surface_to_core["all functor"]["retained_papers"] > 0

    assert surface_to_core["all morphism"]["action"] == "fold"
    assert surface_to_core["all morphism"]["core"] == "morphism"
    assert surface_to_core["all morphism"]["retained_papers"] > 0

    assert surface_to_core["each other"]["action"] == "fold"
    assert surface_to_core["each other"]["core"] == "relation"
    assert surface_to_core["each other"]["retained_papers"] > 0


def test_adjunction_fixture_contains_required_source_families():
    data = json.loads(FIXTURE.read_text())
    sources = {row["source"] for row in data["fixture"]["instances"]}

    assert {"PlanetMath", "nLab", "arxiv-def-snippets"} <= sources


def test_adjunction_fixture_reduces_to_genus_and_three_framings():
    data = json.loads(FIXTURE.read_text())
    reduced = agg.reduce_concept(
        data["fixture"],
        json.loads((ROOT / "data" / "concept-encyclopedia-ct.json").read_text()),
    )
    variants = {
        variant["label"]
        for variant in reduced["variant_axes"][0]["variants"]
        if variant["label"] != "contextual-use"
    }

    assert reduced["genus"] == "adjunction F⊣G"
    assert {
        "hom-set-natural-bijection",
        "unit-counit-triangle",
        "universal-arrow",
    } <= variants
    assert reduced["schema"]["name"] == "lean-family-v0"
    assert reduced["variant_axes"][0]["bridges"]
