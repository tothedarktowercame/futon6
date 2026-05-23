"""Tests for futon6.canon_store (F1+F2 of canon-fingerprint-store)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6.canon_store import (
    SAMPLE_PAPER_LIMIT,
    CanonAggregate,
    CanonFingerprint,
    aggregate_canon_store,
    canon_distribution,
    canon_prior,
    iter_batch_fingerprints,
    load_aggregate,
    save_aggregate,
    write_batch_fingerprints,
)


def _fp(symbol, canon, paper, strategy="let-binding", **kw):
    return CanonFingerprint(
        symbol=symbol, canon=canon, paper_id=paper, strategy=strategy,
        timestamp=kw.get("timestamp", "2026-05-23T12:00:00+00:00"),
    )


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "batch-001.jsonl"
    records = [
        _fp("G", "Group", "arxiv-001"),
        _fp(r"\\mathcal{C}", "Category", "arxiv-001", strategy="denotation"),
    ]
    n = write_batch_fingerprints(records, path)
    assert n == 2
    read = list(iter_batch_fingerprints(path))
    assert len(read) == 2
    assert read[0].symbol == "G" and read[0].canon == "Group"
    assert read[1].canon == "Category"
    assert read[1].strategy == "denotation"


def test_write_is_append_only(tmp_path):
    path = tmp_path / "batch-001.jsonl"
    write_batch_fingerprints([_fp("X", "Group", "p1")], path)
    write_batch_fingerprints([_fp("Y", "Ring", "p2")], path)
    read = list(iter_batch_fingerprints(path))
    assert len(read) == 2
    assert {r.symbol for r in read} == {"X", "Y"}


def test_aggregate_merges_repeated_pairs(tmp_path):
    path = tmp_path / "batch.jsonl"
    write_batch_fingerprints([
        _fp("G", "Group", "p1"),
        _fp("G", "Group", "p2"),
        _fp("G", "Group", "p3"),
        _fp("G", "AbelianGroup", "p4"),
    ], path)
    agg = aggregate_canon_store([path])
    assert agg[("G", "Group")].n_occurrences == 3
    assert sorted(agg[("G", "Group")].source_paper_ids) == ["p1", "p2", "p3"]
    assert agg[("G", "AbelianGroup")].n_occurrences == 1


def test_aggregate_skips_none_canon(tmp_path):
    path = tmp_path / "batch.jsonl"
    write_batch_fingerprints([
        _fp("X", None, "p1"),
        _fp("X", "Group", "p1"),
    ], path)
    agg = aggregate_canon_store([path])
    assert ("X", "Group") in agg
    assert not any(canon is None for _, canon in agg)


def test_aggregate_idempotent_over_multiple_runs(tmp_path):
    path = tmp_path / "batch.jsonl"
    write_batch_fingerprints([
        _fp("G", "Group", "p1"),
        _fp("G", "Group", "p2"),
    ], path)
    agg1 = aggregate_canon_store([path])
    # Re-running on the SAME prior should add the SAME counts again
    # (this is what "state-merge" means when the source is unchanged
    # but consumed twice — caller is responsible for not re-consuming
    # batches; we verify the merge math, not the dedup).
    agg2 = aggregate_canon_store([path], prior_aggregate=agg1)
    # The pair now has 4 occurrences (2 from each pass)
    assert agg2[("G", "Group")].n_occurrences == 4


def test_aggregate_incremental_state_merge(tmp_path):
    """Combining batch A then batch B should equal aggregating both at once
    in n_occurrences (the order doesn't matter)."""
    pa = tmp_path / "a.jsonl"
    pb = tmp_path / "b.jsonl"
    write_batch_fingerprints([_fp("G", "Group", "p1"), _fp("G", "Group", "p2")], pa)
    write_batch_fingerprints([_fp("G", "Group", "p3"), _fp("H", "Field", "p4")], pb)

    full = aggregate_canon_store([pa, pb])
    incremental = aggregate_canon_store([pb], prior_aggregate=aggregate_canon_store([pa]))
    assert full[("G", "Group")].n_occurrences == incremental[("G", "Group")].n_occurrences
    assert full[("H", "Field")].n_occurrences == incremental[("H", "Field")].n_occurrences


def test_aggregate_caps_source_paper_ids(tmp_path):
    path = tmp_path / "batch.jsonl"
    # Emit way more than the cap on the same (symbol, canon) pair
    fps = [_fp("G", "Group", f"p{i}") for i in range(SAMPLE_PAPER_LIMIT * 2 + 17)]
    write_batch_fingerprints(fps, path)
    agg = aggregate_canon_store([path])
    record = agg[("G", "Group")]
    # n_occurrences is EXACT
    assert record.n_occurrences == SAMPLE_PAPER_LIMIT * 2 + 17
    # source_paper_ids is BOUNDED
    assert len(record.source_paper_ids) == SAMPLE_PAPER_LIMIT


def test_aggregate_per_strategy_breakdown_counts(tmp_path):
    path = tmp_path / "batch.jsonl"
    write_batch_fingerprints([
        _fp("G", "Group", "p1", strategy="let-binding"),
        _fp("G", "Group", "p2", strategy="let-binding"),
        _fp("G", "Group", "p3", strategy="denotation"),
    ], path)
    agg = aggregate_canon_store([path])
    breakdown = agg[("G", "Group")].strategy_breakdown
    assert breakdown == {"let-binding": 2, "denotation": 1}


def test_save_then_load_aggregate_round_trip(tmp_path):
    path = tmp_path / "batch.jsonl"
    out = tmp_path / "aggregate.json"
    write_batch_fingerprints([
        _fp("G", "Group", "p1"),
        _fp("G", "Group", "p2"),
        _fp("H", "Field", "p3"),
    ], path)
    agg = aggregate_canon_store([path])
    save_aggregate(agg, out)
    reloaded = load_aggregate(out)
    assert set(reloaded.keys()) == set(agg.keys())
    assert reloaded[("G", "Group")].n_occurrences == agg[("G", "Group")].n_occurrences


def test_load_aggregate_missing_file_returns_empty(tmp_path):
    assert load_aggregate(tmp_path / "no-such-file.json") == {}


def test_canon_distribution_returns_per_canon_aggregates():
    agg = {
        ("G", "Group"): CanonAggregate(symbol="G", canon="Group", n_occurrences=10),
        ("G", "AbelianGroup"): CanonAggregate(symbol="G", canon="AbelianGroup", n_occurrences=4),
        ("H", "Field"): CanonAggregate(symbol="H", canon="Field", n_occurrences=3),
    }
    g_dist = canon_distribution(agg, "G")
    assert set(g_dist.keys()) == {"Group", "AbelianGroup"}
    assert g_dist["Group"].n_occurrences == 10
    z_dist = canon_distribution(agg, "Z")
    assert z_dist == {}


def test_canon_prior_normalises_distribution():
    agg = {
        ("G", "Group"): CanonAggregate(symbol="G", canon="Group", n_occurrences=10),
        ("G", "AbelianGroup"): CanonAggregate(symbol="G", canon="AbelianGroup", n_occurrences=4),
    }
    prior = canon_prior(agg, "G")
    # With smoothing = 0.1, the prior is (n + 0.1) / (total + n_options * 0.1)
    # Total = 14, n_options = 2, denom = 14.2
    # Group → 10.1 / 14.2 ≈ 0.711
    # AbelianGroup → 4.1 / 14.2 ≈ 0.289
    assert abs(prior["Group"] - 10.1 / 14.2) < 1e-9
    assert abs(prior["AbelianGroup"] - 4.1 / 14.2) < 1e-9
    assert abs(sum(prior.values()) - 1.0) < 1e-9


def test_canon_prior_empty_when_symbol_unseen():
    agg = {
        ("G", "Group"): CanonAggregate(symbol="G", canon="Group", n_occurrences=10),
    }
    assert canon_prior(agg, "Z") == {}


def test_fingerprint_to_jsonable_fills_timestamp_if_missing():
    fp = CanonFingerprint(symbol="X", canon="Y", paper_id="p", strategy="s")
    out = fp.to_jsonable()
    assert out["timestamp"]  # auto-filled


def test_fingerprint_from_dict_round_trip():
    fp = CanonFingerprint(
        symbol="X", canon="Y", paper_id="p",
        strategy="let-binding", confidence="high",
        constructor="single", timestamp="2026-05-23T12:00:00+00:00",
    )
    assert CanonFingerprint.from_dict(fp.to_jsonable()) == fp
