"""Tests for ct-verifier.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import importlib

ct_verifier = importlib.import_module("ct-verifier")


REFERENCE = {
    "patterns": {
        "cat/adjunction": {
            "instances": ["i1", "i2"],
            "required_links": ["functor", "left adjoint"],
            "typical_links": ["right adjoint", "natural transformation"],
            "discourse_signature": {
                "components": {
                    "bind/let": 10,
                    "constrain/such-that": 8,
                    "wire/consequential": 5,
                    "wire/clarifying": 3,
                },
                "wires": {},
            },
        },
        "cat/limit": {
            "instances": ["i2"],
            "required_links": ["limit", "diagram"],
            "typical_links": ["cone"],
            "discourse_signature": {
                "components": {
                    "bind/let": 4,
                    "constrain/such-that": 2,
                    "wire/consequential": 2,
                },
                "wires": {},
            },
        },
        "cat/monad": {
            "instances": ["i3"],
            "required_links": ["monad"],
            "typical_links": ["unit", "multiplication"],
            "discourse_signature": {
                "components": {
                    "quant/existential": 6,
                    "wire/adversative": 1,
                },
                "wires": {},
            },
        },
    },
    "link_weights": {"functor": {"definition-ref": 1}},
}


def _edge_lookup(report):
    return {
        (row["edge"]["source"], row["edge"]["target"]): row
        for row in report["edge_reports"]
    }


class TestCategoricalConsistency:
    def test_consistency_and_inconsistency(self):
        wiring = {
            "thread_id": "synthetic-cat",
            "nodes": [
                {"id": "a", "body_text": "functor left adjoint", "categorical": [{"hx/type": "cat/adjunction"}]},
                {"id": "b", "body_text": "functor left adjoint", "categorical": [{"hx/type": "cat/adjunction"}]},
                {"id": "c", "body_text": "limit diagram cone", "categorical": [{"hx/type": "cat/limit"}]},
                {"id": "d", "body_text": "monad unit multiplication", "categorical": [{"hx/type": "cat/monad"}]},
            ],
            "edges": [
                {"from": "a", "to": "b", "type": "assert"},
                {"from": "a", "to": "c", "type": "assert"},
                {"from": "a", "to": "d", "type": "assert"},
            ],
        }
        report = ct_verifier.verify_wiring_dict(wiring, REFERENCE)
        edges = _edge_lookup(report)
        assert edges[("a", "b")]["checks"]["categorical"]["pass"] is True
        assert edges[("a", "c")]["checks"]["categorical"]["pass"] is True
        assert edges[("a", "d")]["checks"]["categorical"]["pass"] is False


class TestPortCompatibility:
    def test_port_type_compatibility(self):
        wiring = {
            "thread_id": "synthetic-port",
            "nodes": [
                {
                    "id": "s",
                    "body_text": "source",
                    "categorical": [{"hx/type": "cat/adjunction"}],
                    "output_ports": [
                        {"id": "s:p1", "type": "bind/let"},
                        {"id": "s:p2", "type": "bind/let"},
                    ],
                },
                {
                    "id": "t1",
                    "body_text": "target1",
                    "categorical": [{"hx/type": "cat/adjunction"}],
                    "input_ports": [{"id": "t1:p1", "type": "bind/let"}],
                },
                {
                    "id": "t2",
                    "body_text": "target2",
                    "categorical": [{"hx/type": "cat/adjunction"}],
                    "input_ports": [{"id": "t2:p1", "type": "quant/existential"}],
                },
            ],
            "edges": [
                {"from": "s", "to": "t1", "type": "assert", "port_matches": [["s:p1", "t1:p1", 2]]},
                {"from": "s", "to": "t2", "type": "assert", "port_matches": [["s:p2", "t2:p1", 2]]},
            ],
        }
        report = ct_verifier.verify_wiring_dict(wiring, REFERENCE)
        edges = _edge_lookup(report)
        assert edges[("s", "t1")]["checks"]["ports"]["pass"] is True
        assert edges[("s", "t2")]["checks"]["ports"]["pass"] is False


class TestIATCAlignment:
    def test_iatc_alignment_rules(self):
        wiring = {
            "thread_id": "synthetic-iatc",
            "nodes": [
                {
                    "id": "assert-node",
                    "body_text": "Therefore the claim follows.",
                    "categorical": [{"hx/type": "cat/adjunction"}],
                    "discourse": [{"hx/type": "wire/consequential"}],
                },
                {
                    "id": "challenge-node",
                    "body_text": "I do not see how this follows.",
                    "categorical": [{"hx/type": "cat/adjunction"}],
                    "discourse": [{"hx/type": "wire/adversative"}],
                },
                {
                    "id": "neutral",
                    "body_text": "neutral text",
                    "categorical": [{"hx/type": "cat/adjunction"}],
                },
            ],
            "edges": [
                {"from": "assert-node", "to": "neutral", "type": "assert"},
                {"from": "neutral", "to": "assert-node", "type": "challenge"},
                {"from": "challenge-node", "to": "neutral", "type": "challenge"},
            ],
        }
        report = ct_verifier.verify_wiring_dict(wiring, REFERENCE)
        edges = _edge_lookup(report)
        assert edges[("assert-node", "neutral")]["checks"]["iatc"]["pass"] is True
        assert edges[("neutral", "assert-node")]["checks"]["iatc"]["pass"] is False
        assert edges[("challenge-node", "neutral")]["checks"]["iatc"]["pass"] is True


class TestCompleteness:
    def test_completeness_scoring(self):
        wiring = {
            "thread_id": "synthetic-complete",
            "nodes": [
                {
                    "id": "n1",
                    "body_text": "A functor has a right adjoint here.",
                    "categorical": [{"hx/type": "cat/adjunction"}],
                }
            ],
            "edges": [],
        }
        report = ct_verifier.verify_wiring_dict(wiring, REFERENCE)
        nr = report["node_reports"][0]
        assert nr["node"] == "n1"
        assert nr["completeness"] == pytest.approx(0.5, abs=1e-6)
        assert "left adjoint" in nr["missing_required"]
        assert "natural transformation" in nr["missing_typical"]


class TestIntegrationProblem4:
    def test_problem4_runs_and_scores_bounded(self, tmp_path: Path):
        root = Path(__file__).parent.parent
        wiring_path = root / "data/first-proof/problem4-wiring.json"
        reference_path = root / "data/nlab-ct-reference.json"
        output_path = tmp_path / "problem4-verification.json"

        ct_verifier.verify_wiring_file(wiring_path, reference_path, output_path)
        report = json.loads(output_path.read_text(encoding="utf-8"))

        assert isinstance(report, dict)
        assert report["summary"]["edges_checked"] > 0
        assert 0.0 <= report["summary"]["completeness_mean"] <= 1.0
        assert 0.0 <= report["summary"]["overall_score"] <= 1.0
