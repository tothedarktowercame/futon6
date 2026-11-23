from __future__ import annotations

import json

from futon6 import inkup_scope_index as scope_index


def _real_records():
    return scope_index.load_index()


def test_real_fixture_matches_inventory_and_text_is_verbatim():
    records = _real_records()
    fixture = json.loads(scope_index.DEFAULT_FIXTURE.read_text())
    inventory = json.loads(scope_index.DEFAULT_INVENTORY.read_text())
    paper = fixture["segments"][0]["text"]
    assert len(records) == inventory["layer_counts"]["s4"] == 163
    assert all(record["text"] in paper for record in records)


def test_real_proof_membership_is_pinned():
    records = {record["span_id"]: record for record in _real_records()}
    assert records["0708.2185-inflight-0021-s1"]["inside_proof"] is True
    assert records["0708.2185-leaf-0001-s1"]["inside_proof"] is False


def test_outer_neighbours_are_null():
    records = _real_records()
    assert records[0]["neighbours"]["previous"] is None
    assert records[-1]["neighbours"]["next"] is None


def test_s3_relation_requires_both_endpoints_to_overlap():
    text = "0123456789" * 3
    fixture = {
        "turn": "mark7-synthetic",
        "segments": [{"id": "paper", "role": "paper", "label": "x", "text": text}],
        "spans": [
            {"id": "s4", "seg": "paper", "start": 5, "end": 20,
             "layer": "s4", "kind": "connection"},
            {"id": "inside-a", "seg": "paper", "start": 6, "end": 8,
             "layer": "s3", "kind": "claim"},
            {"id": "inside-b", "seg": "paper", "start": 18, "end": 22,
             "layer": "s3", "kind": "claim"},
            {"id": "outside", "seg": "paper", "start": 24, "end": 28,
             "layer": "s3", "kind": "claim"},
        ],
        "edges": [
            {"from": "inside-a", "to": "inside-b", "relation": "implies"},
            {"from": "inside-a", "to": "outside", "relation": "iff"},
        ],
    }
    records = scope_index.build_index(fixture, "synthetic", expected_s4_count=1)
    assert records[0]["argument_relations"] == [
        {"relation": "implies", "from": "inside-a", "to": "inside-b"}
    ]
