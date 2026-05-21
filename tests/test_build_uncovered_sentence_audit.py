from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).parent.parent
    path = root / "scripts" / "build-uncovered-sentence-audit.py"
    spec = importlib.util.spec_from_file_location("build_uncovered_sentence_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()


def test_extract_uncovered_sentences_filters_out_covered_spans():
    text = (
        "Let $X$ be a set. "
        "This sentence remains uncovered and should be surfaced. "
        "There exists $f$ such that $f:X\\to X$."
    )
    records = [
        {
            "hx/id": "s1",
            "hx/role": "component",
            "hx/type": "bind/let",
            "hx/content": {
                "position": 0,
                "end": text.index(".") + 1,
                "match": "Let $X$ be a set.",
            },
        },
        {
            "hx/id": "s2",
            "hx/role": "component",
            "hx/type": "quant/existential",
            "hx/content": {"position": text.index("There exists"), "end": len(text)},
        },
    ]
    singles = {
        "sentence": ("sentence", "Sentence"),
        "surface": ("surface", "Surface"),
    }
    multi_index = {}
    rows = AUDIT.extract_uncovered_sentences(
        text,
        records,
        singles,
        multi_index,
        min_sentence_chars=20,
        max_uncovered=10,
    )
    assert len(rows) == 1
    assert "remains uncovered" in rows[0]["text"]
    assert rows[0]["known_term_hit_count"] >= 1
    assert any(item["term_lower"] == "sentence" for item in rows[0]["known_term_hits"])


def test_extract_sentence_term_features_dedupes_hits():
    singles = {
        "group": ("group", "Group"),
    }
    multi_index = {
        "group": [("group action", "group action", "GroupAction")],
    }
    features = AUDIT.extract_sentence_term_features(
        "We study group actions and a group action on curves.",
        singles,
        multi_index,
    )
    assert features["known_term_hit_count"] == 2
    lowers = [row["term_lower"] for row in features["known_term_hits"]]
    assert "group" in lowers
    assert "group action" in lowers


def test_normalize_structure_seed_text_replaces_terms_and_math():
    normalized = AUDIT.normalize_structure_seed_text(
        "We study group actions on $X$ and write group action for the induced map.",
        [
            {"term": "group action", "term_lower": "group action", "canon": "GroupAction"},
            {"term": "map", "term_lower": "map", "canon": "Map"},
        ],
    )
    assert "<term>" in normalized
    assert "<math>" in normalized
    assert "group action" not in normalized


def test_structure_seed_skeleton_keeps_cues_and_placeholders():
    skeleton = AUDIT.structure_seed_skeleton(
        "we show that the <term> only depends on the <term> of <math>."
    )
    # Closed-class prepositions ("on", "of") are intentionally NOT in the cue set:
    # they splinter signatures across papers without adding structural information.
    # The skeleton keeps verb cues, placeholders, and logical connectives only.
    assert skeleton == "we show that <term> only depend <term> <math>"


def test_summarize_structure_seed_candidates_aggregates_cross_paper_templates():
    papers = [
        {
            "paper_id": "p1",
            "uncovered_sentences": [
                {
                    "index": 1,
                    "text": "We study group actions on curves.",
                    "known_term_hit_count": 2,
                    "known_term_hits": [{"term_lower": "group action"}, {"term_lower": "curve"}],
                    "structure_seed_template": "we study <term> on <term>.",
                    "structure_seed_signature": "we study <term> on <term>",
                },
            ],
        },
        {
            "paper_id": "p2",
            "uncovered_sentences": [
                {
                    "index": 7,
                    "text": "We study group actions on schemes.",
                    "known_term_hit_count": 2,
                    "known_term_hits": [{"term_lower": "group action"}, {"term_lower": "scheme"}],
                    "structure_seed_template": "we study <term> on <term>.",
                    "structure_seed_signature": "we study <term> on <term>",
                },
            ],
        },
    ]
    rows = AUDIT.summarize_structure_seed_candidates(papers)
    assert len(rows) == 1
    # Aggregation now buckets by the COARSE signature so analogous residuals
    # across papers cluster together. The full per-residual signature lives in
    # full_signatures.
    assert rows[0]["signature"] == "we study"
    assert rows[0]["full_signatures"] == ["we study <term> on <term>"]
    assert rows[0]["paper_count"] == 2
    assert rows[0]["count"] == 2
    assert rows[0]["predicted_kind"] == "label"


def test_signature_has_discourse_verb_filters_correctly():
    assert AUDIT._signature_has_discourse_verb("we study <term> and <term>")
    assert AUDIT._signature_has_discourse_verb("let <term> be <term>")
    # Pure backbone with no discourse verb gets rejected.
    assert not AUDIT._signature_has_discourse_verb("<term> and <term>")
    assert not AUDIT._signature_has_discourse_verb("<math> <term> <math>")
    assert not AUDIT._signature_has_discourse_verb("")


def test_predict_kind_scope_beats_label_when_both_present():
    # If a sentence has both a binding cue and a rhetorical cue, scope wins.
    kind = AUDIT._predict_kind_from_signature("let <term> be <term> and we prove <term>")
    assert kind == "scope"


def test_predict_kind_label_when_only_rhetorical_cue():
    assert AUDIT._predict_kind_from_signature("we prove <term> and <term>") == "label"
    assert AUDIT._predict_kind_from_signature("we study <term> and <term>") == "label"


def test_predict_kind_none_when_no_discourse_verb():
    assert AUDIT._predict_kind_from_signature("<term> and <term>") is None


def test_signature_to_regex_anchors_cue_backbone():
    rx = AUDIT._signature_to_regex("we study <term> and <term>")
    import re as _re
    # Should match a real sentence with intervening noun phrases.
    assert _re.search(rx, "We study group actions and natural transformations", _re.IGNORECASE)
    # Should NOT match a sentence missing the cue backbone.
    assert not _re.search(rx, "He doesn't really care about any of this", _re.IGNORECASE)


def test_build_learned_discourse_patterns_gates_by_paper_count():
    candidates = [
        # Below gate: only one paper.
        {"signature": "we study <term>", "paper_count": 1, "count": 5,
         "predicted_kind": "label", "max_known_term_hit_count": 3},
        # At gate: two papers + classified.
        {"signature": "let <term> be <term>", "paper_count": 2, "count": 2,
         "predicted_kind": "scope", "max_known_term_hit_count": 4},
        # Gate met but no predicted_kind → reject.
        {"signature": "<term> and <term>", "paper_count": 3, "count": 5,
         "predicted_kind": None, "max_known_term_hit_count": 4},
    ]
    out = AUDIT.build_learned_discourse_patterns(candidates, min_paper_count=2)
    sigs = [p["signature"] for p in out]
    assert sigs == ["let <term> be <term>"]
    assert out[0]["predicted_kind"] == "scope"
    assert "regex" in out[0] and out[0]["regex"]


def test_anticlobber_drops_learned_records_overlapping_base_scope():
    # A base scope covering chars 100-200; a learned record at 150-160 (inside)
    # is clobbered. A learned record at 300-310 (outside) is kept.
    base = [{
        "hx/id": "base-0",
        "hx/role": "scope",
        "hx/type": "env/theorem",
        "hx/content": {"position": 100, "end": 200, "match": "..."},
    }]
    learned = [
        {
            "hx/id": "L-clobber",
            "hx/role": "label",
            "hx/type": "learned/label",
            "hx/content": {"position": 150, "end": 160, "match": "..."},
        },
        {
            "hx/id": "L-kept",
            "hx/role": "label",
            "hx/type": "learned/label",
            "hx/content": {"position": 300, "end": 310, "match": "..."},
        },
    ]
    kept, clobbered = AUDIT.apply_anticlobber(base, learned)
    assert [r["hx/id"] for r in kept] == ["L-kept"]
    assert [r["hx/id"] for r in clobbered] == ["L-clobber"]


def test_anticlobber_keeps_records_with_no_span():
    # A learned record without a usable position survives — defensive default.
    base = [{
        "hx/id": "base-0",
        "hx/role": "scope",
        "hx/type": "env/theorem",
        "hx/content": {"position": 100, "end": 200, "match": "..."},
    }]
    learned_no_span = {
        "hx/id": "L-nospan",
        "hx/role": "label",
        "hx/type": "learned/label",
        "hx/content": {"match": "..."},  # no position
    }
    kept, clobbered = AUDIT.apply_anticlobber(base, [learned_no_span])
    assert kept == [learned_no_span]
    assert clobbered == []


def test_anticlobber_with_no_base_keeps_everything():
    # If base detectors fired nothing, learned records can't be clobbered.
    learned = [
        {
            "hx/id": "L-1",
            "hx/role": "label",
            "hx/type": "learned/label",
            "hx/content": {"position": 0, "end": 10, "match": "..."},
        },
    ]
    kept, clobbered = AUDIT.apply_anticlobber([], learned)
    assert kept == learned
    assert clobbered == []


def test_anticlobber_partial_overlap_counts_as_clobber():
    # Boundary case: a learned record straddles the edge of a base span.
    base = [{
        "hx/id": "base-0",
        "hx/role": "scope",
        "hx/type": "env/theorem",
        "hx/content": {"position": 100, "end": 200, "match": "..."},
    }]
    straddle = {
        "hx/id": "L-straddle",
        "hx/role": "label",
        "hx/type": "learned/label",
        "hx/content": {"position": 190, "end": 250, "match": "..."},
    }
    kept, clobbered = AUDIT.apply_anticlobber(base, [straddle])
    assert kept == []
    assert clobbered == [straddle]


def test_seed_signatures_flag_loads_and_matches_via_audit_module(tmp_path):
    # Write a synthetic prior-run audit JSON with one signature long enough
    # to clear the matcher's min_tokens=3 floor.
    prior = {
        "structure_seed_candidates": [
            {"signature": "we study <term> and <term>", "count": 2, "paper_count": 2},
            {"signature": "<term>", "count": 5, "paper_count": 1},  # too short, should be ignored
        ]
    }
    prior_path = tmp_path / "prior.json"
    import json
    prior_path.write_text(json.dumps(prior), encoding="utf-8")

    sigs = AUDIT.SUPERPOD_JOB._load_structure_seed_signatures(prior_path)
    # Only the long-enough signature survives _signature_tokens filtering.
    assert len(sigs) == 2  # loader includes both; matcher filters by min_tokens
    assert any(sig == "we study <term> and <term>" for sig, _toks in sigs)

    # Subsequence matcher should fire on a residual whose signature contains
    # the prior as a strict in-order subsequence.
    new_sig = "we study <term> and <term> when there exists <term>"
    matched = AUDIT.SUPERPOD_JOB._match_structure_seed_signature(new_sig, sigs)
    assert matched == "we study <term> and <term>", (
        f"expected subsequence match, got {matched!r}"
    )


def test_select_daisychain_papers_excludes_prior_ids_and_prefers_ct():
    batch_meta = {
        "a": {"categories": ["math.CT"]},
        "b": {"categories": ["math.AG"]},
        "c": {"categories": ["math.CT"]},
        "d": {"categories": ["math.NT"]},
    }
    available = {"a", "b", "c", "d"}
    ledger = {"runs": [{"paper_ids": ["a"]}]}
    chosen = AUDIT.select_daisychain_papers(
        batch_meta,
        available,
        ledger,
        paper_count=2,
        ct_count=1,
        seed=17,
    )
    assert "a" not in chosen
    assert len(chosen) == 2
    assert any("math.CT" in batch_meta[pid]["categories"] for pid in chosen)
