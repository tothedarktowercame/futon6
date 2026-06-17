import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cas_cert", ROOT / "scripts" / "cas_cert.py")
cas_cert = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cas_cert
SPEC.loader.exec_module(cas_cert)


def graph(
    *,
    paper_id="p1",
    anchor_status="pass",
    closure_orphans=None,
    cycle=None,
    warrant_status="resolved",
    concept_bucket="defined",
):
    closure_orphans = closure_orphans or []
    return {
        "paper-id": paper_id,
        "profile": {
            "paper-id": paper_id,
            "reasoning": [
                {
                    "id": ":e1",
                    "warrant": {"status": f":{warrant_status}", "kind": ":missing-warrant" if warrant_status != "resolved" else ":claim"},
                }
            ],
        },
        "checks": [
            {
                "check": ":anchor-faithfulness",
                "status": ":pass" if anchor_status == "pass" else ":fail",
                "per-item": [
                    {
                        "id": ":n1",
                        "status": f":{anchor_status}",
                        "source": {"lines": [1, 1]},
                        "missing": [] if anchor_status == "pass" else ["term"],
                    }
                ],
            },
            {
                "check": ":closure",
                "status": ":fail" if closure_orphans or cycle else ":pass",
                "per-item": [
                    {
                        "orphan-nodes": closure_orphans,
                        "cycle": cycle,
                        "file": "p1.edn",
                    }
                ],
            },
            {
                "check": ":warrant-resolution",
                "status": ":pass",
                "per-item": [{"resolved-warrant-edges": 1 if warrant_status == "resolved" else 0, "total-edges": 1}],
            },
            {
                "check": ":concept-coverage",
                "status": ":pass",
                "per-item": [
                    {
                        "concept": "calmod-like bicategory",
                        "bucket": concept_bucket,
                        "reason": "fixture",
                        "sources": [],
                    }
                ],
            },
        ],
    }


def test_port_partition_and_residual_sorries():
    cert = cas_cert.certificate_for_graph(
        graph(anchor_status="fail", closure_orphans=[":orphan"], warrant_status="missing", concept_bucket="undefined")
    )
    by_state = {(p["grain"], p["item"]): p["state"] for p in cert["ports"]}

    assert by_state[("symbol", "symbol-grounding")] == "na"
    assert by_state[("concept", "calmod-like bicategory")] == "empty"
    assert by_state[("proof", "anchor::n1")] == "miswired"
    assert by_state[("proof", "orphan::orphan")] == "empty"
    assert by_state[("proof", "warrant::e1")] == "empty"

    residual_kinds = {row["kind"] for row in cert["residual_sorries"]}
    assert {"undefined", "orphan", "missing-warrant"} <= residual_kinds


def test_conformance_is_vector_by_grain():
    cert = cas_cert.certificate_for_graph(graph())
    by_grain = cert["conformance"]["by_grain"]

    assert set(by_grain) == {"symbol", "concept", "technique", "proof"}
    assert by_grain["symbol"]["na"] is True
    assert by_grain["concept"]["rate"] == 1.0
    assert by_grain["proof"]["filled"] == 3
    assert "rate" in by_grain["technique"]


def test_miswire_fails_gate_but_empty_does_not():
    fail_payload = cas_cert.build_certificates({"graphs": [graph(anchor_status="fail")]})
    empty_payload = cas_cert.build_certificates(
        {"graphs": [graph(closure_orphans=[":orphan"], warrant_status="missing", concept_bucket="undefined")]}
    )

    assert fail_payload["gate"] == "FAIL"
    assert fail_payload["certificates"][0]["verdict"]["miswires"] == ["anchor::n1"]

    assert empty_payload["gate"] == "PASS"
    assert empty_payload["certificates"][0]["verdict"]["miswires"] == []
    assert empty_payload["certificates"][0]["residual_sorries"]


def test_na_grains_do_not_fail_gate():
    cert = cas_cert.certificate_for_graph(graph())

    assert cert["conformance"]["by_grain"]["symbol"]["na"] is True
    assert cert["conformance"]["by_grain"]["technique"]["na"] is True
    assert cert["verdict"]["gate"] == "PASS"


def test_confidence_is_medium_when_foundation_grains_are_na():
    cert = cas_cert.certificate_for_graph(graph())

    assert cert["confidence"] == {
        "level": "medium",
        "limiting_factors": [
            "symbol grain N/A — SFC2b not built",
            "technique grain N/A — rung-3 not built",
        ],
    }
    assert cert["verdict"] == {"well_wired": True, "miswires": [], "gate": "PASS"}


def test_confidence_high_and_low_are_structural():
    solid = {
        "symbol": {"filled": 2, "empty": 0, "miswired": 0, "na": False, "rung": "SFC2b"},
        "concept": {"filled": 1, "empty": 1, "miswired": 0, "na": False, "rung": "R2d"},
        "technique": {"filled": 0, "empty": 0, "miswired": 1, "na": False, "rung": "rung-3"},
        "proof": {"filled": 0, "empty": 10, "miswired": 0, "na": False, "rung": "R2a/R2b/R2c"},
    }
    weak = {
        **solid,
        "concept": {"filled": 1, "empty": 2, "miswired": 0, "na": False, "rung": "R2d"},
    }

    assert cas_cert.confidence(solid) == {"level": "high", "limiting_factors": []}
    assert cas_cert.confidence(weak) == {
        "level": "low",
        "limiting_factors": ["concept grain low solidity 0.333"],
    }
