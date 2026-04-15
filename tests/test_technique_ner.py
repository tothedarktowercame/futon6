"""Tests for Stage 5c — technique-level NER.

Spec: futon6/holes/missions/M-paper-reverse-morphogenesis.md §5c.
"""

from futon6.technique_ner import (
    TechniqueHit,
    extract_techniques_classical,
    merge_technique_arms,
    techniques_to_records,
    _parse_llm_response,
)


SAMPLE_TEXT = """We use a Borel completion adjunction to lift the measurable
structure from the base space to the completion.

By Wantzel's theorem, any number constructible by ruler and compass lies in
the quadratic closure of Q.

The argument proceeds by a standard spectral sequence computation followed
by a degeneration argument. Applying the Lefschetz fixed-point theorem
completes the proof.

\\section{Main construction}

The Borel-Moore homology decomposition is standard. The Cartan-Eilenberg
resolution gives a projective resolution. By the Hurewicz theorem, the
space is connected."""


def test_classical_catches_proper_noun_anchored_techniques():
    hits = extract_techniques_classical(SAMPLE_TEXT)
    terms = {h.canonical for h in hits}
    assert "borel completion adjunction" in terms
    assert "lefschetz fixed-point theorem" in terms
    assert "borel-moore homology decomposition" in terms
    assert "hurewicz theorem" in terms
    assert "cartan-eilenberg resolution" in terms


def test_classical_catches_possessive_technique():
    hits = extract_techniques_classical(SAMPLE_TEXT)
    assert any(h.canonical == "wantzel theorem" for h in hits)


def test_classical_rejects_non_proper_anchored_phrases():
    """'base space to the completion' must not be picked up as a technique.

    Regression for the re.IGNORECASE bug that let lowercase 'base' match
    [A-Z] in the modifier anchor."""
    hits = extract_techniques_classical(SAMPLE_TEXT)
    terms = {h.canonical for h in hits}
    assert "base space to the completion" not in terms
    assert not any("base space" in t for t in terms)


def test_classical_records_loci_with_section_and_paragraph():
    hits = extract_techniques_classical(SAMPLE_TEXT)
    borel_moore = next(
        (h for h in hits if h.canonical == "borel-moore homology decomposition"),
        None,
    )
    assert borel_moore is not None
    assert borel_moore.first_defined_at["section"] == "1"
    locus = borel_moore.loci[0]
    assert locus["section"] == "1"
    assert "char_span" in locus and len(locus["char_span"]) == 2


def test_classical_counts_repeated_occurrences():
    """'Cartan-Eilenberg resolution' appears once in the sample but the
    text also says 'gives a projective resolution' — two distinct patterns
    can match, and the dedup via canonical key should still record at least
    one locus."""
    hits = extract_techniques_classical(SAMPLE_TEXT)
    cartan = next(
        (h for h in hits if h.canonical == "cartan-eilenberg resolution"),
        None,
    )
    assert cartan is not None
    assert len(cartan.loci) >= 1


def test_merge_marks_intersection_as_both():
    classical = [
        TechniqueHit(
            canonical="borel completion adjunction",
            term="Borel completion adjunction",
            loci=[{"section": "0", "paragraph": 0, "char_span": [0, 30],
                   "pattern": "the_x_head"}],
            first_defined_at={"section": "0", "paragraph": 0},
            extraction_source="classical",
        ),
    ]
    llm = [
        TechniqueHit(
            canonical="borel completion adjunction",
            term="Borel completion adjunction",
            loci=[{"section": "0", "paragraph": 0, "char_span": [0, 30],
                   "pattern": "llm"}],
            first_defined_at={"section": "0", "paragraph": 0},
            extraction_source="llm",
        ),
        TechniqueHit(
            canonical="spectral sequence computation",
            term="spectral sequence computation",
            loci=[{"section": "0", "paragraph": 1, "char_span": [100, 130],
                   "pattern": "llm"}],
            first_defined_at={"section": "0", "paragraph": 1},
            extraction_source="llm",
        ),
    ]
    merged = {h.canonical: h for h in merge_technique_arms(classical, llm)}
    assert merged["borel completion adjunction"].extraction_source == "both"
    assert merged["spectral sequence computation"].extraction_source == "llm"


def test_merge_deduplicates_loci_by_span():
    classical = [
        TechniqueHit(
            canonical="x", term="X", extraction_source="classical",
            first_defined_at={"section": "0", "paragraph": 0},
            loci=[{"section": "0", "paragraph": 0, "char_span": [0, 5],
                   "pattern": "the_x_head"}],
        ),
    ]
    llm = [
        TechniqueHit(
            canonical="x", term="X", extraction_source="llm",
            first_defined_at={"section": "0", "paragraph": 0},
            loci=[
                {"section": "0", "paragraph": 0, "char_span": [0, 5],
                 "pattern": "llm"},  # duplicate span
                {"section": "0", "paragraph": 2, "char_span": [200, 205],
                 "pattern": "llm"},  # new span
            ],
        ),
    ]
    merged = merge_technique_arms(classical, llm)
    assert len(merged) == 1
    assert len(merged[0].loci) == 2  # one from classical + one new from llm
    spans = {tuple(l["char_span"]) for l in merged[0].loci}
    assert (0, 5) in spans and (200, 205) in spans


def test_records_schema_matches_spec():
    hits = extract_techniques_classical(SAMPLE_TEXT)
    rec = techniques_to_records("arxiv-test-001", hits)
    assert rec["paper_id"] == "arxiv-test-001"
    assert isinstance(rec["techniques"], list)
    assert rec["techniques"], "expected at least one technique"
    first = rec["techniques"][0]
    for key in ("term", "canonical", "loci", "first_defined_at", "extraction_source"):
        assert key in first, f"missing required key: {key}"
    assert first["extraction_source"] in ("classical", "llm", "both")


def test_llm_response_parser_tolerates_surrounding_prose():
    raw = (
        "Sure, here are the techniques:\n"
        "[{\"term\": \"Borel completion adjunction\", \"role\": \"primary\", "
        "\"rationale\": \"key\"}]\n"
        "Let me know if you want more."
    )
    parsed = _parse_llm_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["term"] == "Borel completion adjunction"


def test_llm_response_parser_rejects_malformed():
    assert _parse_llm_response("no JSON here") == []
    assert _parse_llm_response("[{not valid json}]") == []
    assert _parse_llm_response('{"not": "an array"}') == []
