"""Direct tests for the shared futon6.structure_seed module.

The script-level tests already exercise the module through the audit, viewer,
and superpod call sites. These tests pin module-level semantics so any future
behavior change has to be deliberate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6 import structure_seed as ss


def test_normalize_replaces_terms_math_and_lowercases():
    out = ss.normalize_structure_seed_text(
        "We study the GROUP action of $G$ on $X$.",
        [{"term_lower": "group action", "term": "group action"}],
    )
    assert "<term>" in out
    assert "<math>" in out
    assert out == out.lower()


def test_skeleton_drops_closed_class_prepositions_and_collapses_placeholders():
    full = "we show that the <term> only depends on the <term> of <math>."
    sk = ss.structure_seed_skeleton(full)
    # "on" and "of" are NOT in the cue set; adjacent placeholders collapse.
    assert sk == "we show that <term> only depend <term> <math>"


def test_coarse_signature_drops_placeholders_and_content_words():
    coarse = ss.coarse_discourse_signature("we study <term> and <term>")
    assert coarse == "we study and"


def test_signature_has_discourse_verb():
    assert ss.signature_has_discourse_verb("we prove <term>")
    assert not ss.signature_has_discourse_verb("<term> and <term>")
    assert not ss.signature_has_discourse_verb("")


def test_predict_kind_scope_beats_label_beats_wire():
    # both scope and label cues → scope wins
    assert ss.predict_kind_from_signature("let <term> we prove <term>") == "scope"
    # only label
    assert ss.predict_kind_from_signature("we prove <term>") == "label"
    # only wire
    assert ss.predict_kind_from_signature("then <term> notice") == "wire"
    # nothing
    assert ss.predict_kind_from_signature("<term> and <term>") is None


def test_summarize_clusters_by_coarse_signature():
    rows = [
        {
            "paper_id": "p1",
            "structure_seed_signature": "<term> be introduce <cite>",
            "known_term_hit_count": 2,
            "known_term_hits": [{"term_lower": "ring"}, {"term_lower": "module"}],
            "text": "Quantum groupoids were introduced in [DS].",
            "index": 1,
        },
        {
            "paper_id": "p2",
            "structure_seed_signature": "<math> be introduce <term> <num>",
            "known_term_hit_count": 2,
            "known_term_hits": [{"term_lower": "polynomial"}],
            "text": "Pi^X are introduced in section 2.",
            "index": 5,
        },
        # No discourse verb; gets filtered out
        {
            "paper_id": "p3",
            "structure_seed_signature": "<term> and <term>",
            "known_term_hit_count": 3,
            "known_term_hits": [{"term_lower": "ring"}],
            "text": "X and Y",
            "index": 2,
        },
    ]
    cands = ss.summarize_structure_seed_candidates(rows)
    # Both p1 and p2 cluster under coarse "be introduce".
    assert len(cands) == 1
    assert cands[0]["signature"] == "be introduce"
    assert cands[0]["paper_count"] == 2
    assert cands[0]["predicted_kind"] == "label"
    assert sorted(cands[0]["full_signatures"]) == sorted([
        "<term> be introduce <cite>",
        "<math> be introduce <term> <num>",
    ])


def test_subsequence_matcher_fires_on_in_order_subset():
    prior = "we study <term> and <term>"
    new = "we study <term> and <term> when there exists <term>"
    prior_tokens = ss.signature_tokens(prior)
    new_tokens = ss.signature_tokens(new)
    assert ss.is_subsequence(prior_tokens, new_tokens)
    priors = [(prior, prior_tokens)]
    assert ss.match_structure_seed_signature(new, priors) == prior


def test_subsequence_matcher_respects_min_tokens():
    short_prior = "we <term>"  # 2 tokens, below min=3
    new = "we study <term> and <term>"
    priors = [(short_prior, ss.signature_tokens(short_prior))]
    assert ss.match_structure_seed_signature(new, priors) is None


def test_load_signatures_prefers_full_signatures_over_top_level_signature():
    payload = {
        "structure_seed_candidates": [
            {
                "signature": "we study and",  # coarse
                "full_signatures": [
                    "we study <term> and <term>",
                    "we study <term> and <math>",
                ],
            },
            {"signature": "<term>"},  # legacy: top-level only
        ],
    }
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = Path(f.name)
    sigs = ss.load_structure_seed_signatures(path)
    sig_strs = [s for s, _ in sigs]
    assert "we study <term> and <term>" in sig_strs
    assert "we study <term> and <math>" in sig_strs
    assert "<term>" in sig_strs


def test_build_scope_tree_nests_and_places_terms_at_deepest():
    spans = [
        {"start": 0, "end": 100, "label": "env/proof"},
        {"start": 20, "end": 60, "label": "bind/typed"},
    ]
    tree = ss.build_scope_tree(spans, [(25, 30)])  # inside both
    assert tree["children"][0]["label"] == "env/proof"
    inner = tree["children"][0]["children"][0]
    assert inner["label"] == "bind/typed"
    assert (25, 30) in inner["terms"]
    assert tree["children"][0]["terms"] == []  # outer scope didn't claim it


def test_classify_kernel_terms_inhabited_vs_outer_vs_straddled():
    # Three terms: one fully inside, one fully outside, one straddling.
    positions = [(10, 15), (200, 210), (45, 60)]  # term3 straddles scope edge at 50
    spans = [{"start": 0, "end": 50, "label": "env/proof"}]
    stats = ss.classify_kernel_terms_from_positions(positions, spans)
    assert stats["inhabited"] == 1
    assert stats["outer"] == 1
    assert stats["straddled"] == 1
    assert stats["total"] == 3
    assert stats["depth_distribution"] == {1: 1}


def test_scope_records_to_spans_skips_unparseable():
    records = [
        {"hx/type": "env/proof", "hx/content": {"position": 0, "end": 100, "match": "..."}},
        {"hx/type": "bad", "hx/content": {"match": "no position"}},
        {"hx/type": "from_match", "hx/content": {"position": 200, "match": "hello"}},
    ]
    spans = ss.scope_records_to_spans(records)
    assert len(spans) == 2
    assert spans[0]["label"] == "env/proof"
    # Second kept span uses len(match) for end:
    assert spans[1]["start"] == 200 and spans[1]["end"] == 205
