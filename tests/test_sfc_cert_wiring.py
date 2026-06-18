import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "sfc-cert-wiring"
CANDIDATE_0709 = ROOT / "data" / "iatc-candidates" / "0709.0248.candidate.json"
GRAPH_0709 = ROOT / "data" / "iatc-argument-graphs" / "loop-run-70b" / "0709.0248.edn"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cas_cert = load_module("cas_cert_test", ROOT / "scripts" / "cas_cert.py")


def minimal_graph(paper_id="fixture"):
    return {
        "paper-id": paper_id,
        "checks": [],
        "profile": {"paper-id": paper_id, "reasoning": []},
    }


def test_symbol_status_mapping_is_report_only_and_populates_grain():
    symbols = json.loads((FIXTURES / "symbols.json").read_text())
    cert = cas_cert.certificate_for_graph(
        minimal_graph("fixture"),
        symbols_by_paper=cas_cert.symbols_by_paper(symbols),
    )

    symbol_ports = [p for p in cert["ports"] if p["grain"] == "symbol"]
    assert [(p["item"], p["state"], p.get("kind")) for p in symbol_ports] == [
        ("symbol:x", "filled", None),
        ("symbol:y", "empty", "undefined-in-context"),
        ("symbol:z", "empty", "unsupported"),
    ]
    assert all(p["state"] != "miswired" for p in symbol_ports)
    assert symbol_ports[2]["evidence"]["binding"] == "hallucinated rejected binding"
    assert cert["conformance"]["by_grain"]["symbol"] == {
        "filled": 1,
        "empty": 2,
        "miswired": 0,
        "na": False,
        "rate": 1 / 3,
        "rung": "SFC2b",
    }
    assert cert["verdict"]["gate"] == "PASS"


def test_symbol_grain_stays_na_when_symbols_absent():
    cert = cas_cert.certificate_for_graph(minimal_graph("fixture"))

    assert cert["conformance"]["by_grain"]["symbol"]["na"] is True
    assert [p for p in cert["ports"] if p["grain"] == "symbol"] == [
        {
            "grain": "symbol",
            "item": "symbol-grounding",
            "state": "na",
            "rung": "SFC2b",
            "scoped_query": "per-paper symbol/domain grounding",
            "evidence": "SFC2b not wired into CAS-CERT yet",
        }
    ]


def test_sfc_ground_paper_stub_is_deterministic_and_evidence_is_verbatim(tmp_path):
    out1 = tmp_path / "first.symbols.json"
    out2 = tmp_path / "second.symbols.json"

    for out in (out1, out2):
        subprocess.run(
            [
                "python3",
                "scripts/sfc_ground_paper.py",
                str(CANDIDATE_0709),
                "--backend",
                "stub",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

    assert out1.read_bytes() == out2.read_bytes()
    doc = json.loads(out1.read_text())
    candidate = json.loads(CANDIDATE_0709.read_text())
    context = candidate["source-window"]

    assert doc["schema"] == "sfc-symbol-grounding/v0"
    assert doc["paper_id"] == "0709.0248"
    assert doc["summary"]["symbols"] == len(doc["groundings"])
    assert any(row["status"] == "grounded" for row in doc["groundings"])
    for row in doc["groundings"]:
        if row["status"] == "grounded":
            assert row["evidence"]
            assert row["evidence"] in context


def run_json(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def test_0709_cert_symbol_grain_goes_from_na_to_populated(tmp_path):
    semcheck = tmp_path / "0709.semcheck.edn"
    symbols = tmp_path / "0709.symbols.json"

    subprocess.run(
        ["bb", "scripts/iatc_semcheck.bb", "--out", str(semcheck), str(GRAPH_0709)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "python3",
            "scripts/sfc_ground_paper.py",
            str(CANDIDATE_0709),
            "--backend",
            "stub",
            "--out",
            str(symbols),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    before = run_json(["python3", "scripts/cas_cert.py", "--semcheck", str(semcheck)])
    after = run_json(
        ["python3", "scripts/cas_cert.py", "--semcheck", str(semcheck), "--symbols", str(symbols)]
    )

    before_cert = before["certificates"][0]
    after_cert = after["certificates"][0]
    assert before_cert["conformance"]["by_grain"]["symbol"]["na"] is True
    assert after_cert["conformance"]["by_grain"]["symbol"] == {
        "filled": 3,
        "empty": 5,
        "miswired": 0,
        "na": False,
        "rate": 3 / 8,
        "rung": "SFC2b",
    }
    assert before_cert["verdict"]["gate"] == after_cert["verdict"]["gate"]
    assert "symbol grain N/A — SFC2b not built" in before_cert["confidence"]["limiting_factors"]
    assert "symbol grain N/A — SFC2b not built" not in after_cert["confidence"]["limiting_factors"]
