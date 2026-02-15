"""Tests for enrich-proof-wiring.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import importlib

ct_verifier = importlib.import_module("ct-verifier")
enrich_proof = importlib.import_module("enrich-proof-wiring")
nlab_wiring = importlib.import_module("nlab-wiring")


ROOT = Path(__file__).parent.parent
PROBLEM4 = ROOT / "data/first-proof/problem4-wiring.json"
REFERENCE = ROOT / "data/nlab-ct-reference.json"
NER_TSV = ROOT / "data/ner-kernel/terms.tsv"


def _load_fixture():
    wiring = json.loads(PROBLEM4.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    singles, multi_index, _ = nlab_wiring.load_ner_kernel(NER_TSV)
    return wiring, reference, singles, multi_index


class TestNodeEnrichment:
    def test_error_node_gets_adversative_discourse(self):
        wiring, reference, singles, multi_index = _load_fixture()
        node = next(n for n in wiring["nodes"] if n["id"] == "p4-s4a")
        enriched = enrich_proof.enrich_node(node, reference, singles, multi_index)
        discourse_types = {d["hx/type"] for d in enriched["discourse"]}
        assert "wire/adversative" in discourse_types


class TestConsequentialDetection:
    def test_complete_qed_proved_trigger_consequential(self):
        _, reference, singles, multi_index = _load_fixture()
        node = {
            "id": "synthetic-conclusion",
            "node_type": "answer",
            "body_text": "[PROVED] n=4 COMPLETE by decomposition. QED.",
        }
        enriched = enrich_proof.enrich_node(node, reference, singles, multi_index)
        discourse_types = {d["hx/type"] for d in enriched["discourse"]}
        output_types = {p["type"] for p in enriched["output_ports"]}
        assert "wire/consequential" in discourse_types
        assert "wire/consequential" in output_types


class TestPortExtraction:
    def test_equations_become_bind_and_constraint_ports(self):
        _, reference, singles, multi_index = _load_fixture()
        node = {
            "id": "synthetic-equations",
            "node_type": "answer",
            "body_text": (
                "1/Phi_4 = T1+T2+R. "
                "K_red = A(P,Q)+sqrt(PQ)*B(P,Q). "
                "T1_surplus >= 0."
            ),
        }
        enriched = enrich_proof.enrich_node(node, reference, singles, multi_index)
        input_types = {p["type"] for p in enriched["input_ports"]}
        assert "bind/let" in input_types
        assert "constrain/such-that" in input_types


class TestEdgePortMatching:
    def test_related_edges_get_port_matches(self):
        wiring, reference, singles, multi_index = _load_fixture()
        enriched = enrich_proof.enrich_wiring(wiring, reference, singles, multi_index)
        edges_with_matches = [e for e in enriched["edges"] if e.get("port_matches")]
        assert len(edges_with_matches) >= 8


class TestEndToEnd:
    def test_problem4_enrichment_improves_verification_score(self):
        wiring, reference, singles, multi_index = _load_fixture()

        bare_report = ct_verifier.verify_wiring_dict(wiring, reference)
        enriched = enrich_proof.enrich_wiring(wiring, reference, singles, multi_index)
        enriched_report = ct_verifier.verify_wiring_dict(enriched, reference)

        assert bare_report["summary"]["overall_score"] < 0.05
        assert enriched_report["summary"]["overall_score"] > 0.2
