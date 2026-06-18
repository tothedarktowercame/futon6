import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rung3 = load_script("rung3_technique")
cas_select = load_script("cas_select")
cas_cert = load_script("cas_cert")


FIXTURES = ROOT / "tests" / "fixtures" / "cas-select"


def test_reproduces_rung3_1_cas0_hand_classification():
    patterns = cas_select.load_patterns()
    totals = {bucket: 0 for bucket in rung3.BUCKETS}
    total_moves = 0
    for path in sorted(FIXTURES.glob("*.steps.json")):
        steps_doc = cas_select.load_steps(path)
        oracle = cas_select.load_oracle(FIXTURES / f"{steps_doc['paper_id']}.oracle.json")
        gapmap = rung3.gapmap_for_steps(steps_doc, patterns, oracle=oracle)
        total_moves += len(gapmap["moves"])
        for bucket, count in gapmap["buckets"].items():
            totals[bucket] += count

    assert total_moves == 22
    assert totals == {
        "grounded-by-pattern": 14,
        "grounded-by-citation": 0,
        "thin": 1,
        "ungrounded": 7,
        "conjecture": 0,
    }


def test_conjecture_is_credited_not_flagged():
    patterns = cas_select.load_patterns()
    steps_doc = {
        "paper_id": "toy",
        "steps": [{"id": "s1", "text": "It remains an open problem whether this extension exists."}],
    }

    gapmap = rung3.gapmap_for_steps(steps_doc, patterns)

    assert gapmap["buckets"]["conjecture"] == 1
    assert gapmap["moves"][0]["bucket"] == "conjecture"
    assert gapmap["moves"][0]["credited"] is True
    assert gapmap["gaps"][0]["credited"] is True


def test_cas_cert_rung3_ports_supersede_cas_select_and_do_not_fail_gate():
    graph = {
        "paper-id": "toy",
        "profile": {"paper-id": "toy", "reasoning": []},
        "checks": [
            {
                "check": ":concept-coverage",
                "status": ":pass",
                "per-item": [{"concept": "known object", "bucket": "defined"}],
            },
            {"check": ":closure", "status": ":pass", "per-item": [{"orphan-nodes": [], "cycle": None}]},
            {"check": ":anchor-faithfulness", "status": ":pass", "per-item": []},
        ],
    }
    rung3_doc = {
        "paper_id": "toy",
        "moves": [
            {"step": "s1", "bucket": "grounded-by-pattern", "pattern": "reduce-to-known-result", "type": "verifiable"},
            {"step": "s2", "bucket": "thin", "pattern": "local-to-global", "type": "heuristic"},
            {"step": "s3", "bucket": "ungrounded", "pattern": None, "type": "none"},
            {"step": "s4", "bucket": "conjecture", "pattern": None, "type": "none", "credited": True},
        ],
        "buckets": {},
        "gaps": [],
    }
    cas_select_payload = {"results": {"toy": {"sorry": [{"step": "old", "kind": "thin"}]}}}

    cert = cas_cert.certificate_for_graph(graph, cas_select_payload, rung3_by_paper={"toy": rung3_doc})
    technique = cert["conformance"]["by_grain"]["technique"]
    technique_ports = [p for p in cert["ports"] if p["grain"] == "technique"]

    assert technique == {
        "filled": 1,
        "empty": 3,
        "miswired": 0,
        "na": False,
        "rate": 0.25,
        "rung": "CAS-SEL/rung-3",
    }
    assert not any(p["state"] == "miswired" for p in technique_ports)
    assert not any(p["item"] == "thin:old" for p in technique_ports)
    conjecture = next(p for p in technique_ports if p.get("kind") == "conjecture")
    assert conjecture["evidence"]["credited"] is True
    assert cert["verdict"]["gate"] == "PASS"


def test_absent_rung3_preserves_existing_cas_select_fallback():
    cert = cas_cert.certificate_for_graph(
        {
            "paper-id": "toy",
            "profile": {"paper-id": "toy", "reasoning": []},
            "checks": [{"check": ":concept-coverage", "status": ":pass", "per-item": []}],
        },
        {"results": {"toy": {"sorry": [{"step": "old", "kind": "thin"}]}}},
    )

    assert any(p["item"] == "thin:old" for p in cert["ports"])
