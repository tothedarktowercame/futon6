import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "holes/inkup/reasoning/validate_reasoning.py"
STRATEGY = ROOT / "holes/inkup/reasoning/a96J06-why-rescale.json"
WARRANT = ROOT / "holes/inkup/reasoning/a96J06-coefficient-bridge.json"
RIGIDITY = ROOT / "holes/inkup/reasoning/a96J06-schwarz-rigidity.json"
spec = importlib.util.spec_from_file_location("validate_reasoning", VALIDATOR)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_graph(tmp_path, doc):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(doc))
    return path


def test_both_reasoning_graph_forms_validate():
    assert module.validate(STRATEGY)["subject"] == "a96J06: why rescale?"
    assert module.validate(WARRANT)["form"] == "warrant-chain"
    assert module.validate(RIGIDITY)["form"] == "warrant-chain"


def test_rigidity_graph_cannot_invent_open_codomain_binding(tmp_path):
    doc = json.loads(RIGIDITY.read_text())
    node = next(node for node in doc["nodes"] if node["id"] == "maps-to-closed-ball")
    node["evidence"][0] = "lean:Main.lean#hgopenmap"
    with pytest.raises(SystemExit, match="names nothing bound in the Lean source"):
        module.validate(write_graph(tmp_path, doc))


def test_warrant_without_requires_fails_for_that_reason(tmp_path):
    doc = json.loads(WARRANT.read_text())
    doc["edges"] = [
        edge for edge in doc["edges"]
        if not (edge["from"] == "contradiction-warrant" and edge["relation"] == "requires")
    ]
    doc["edges"].append({
        "from": "top-zero-warrant", "relation": "requires", "to": "terminal-nonzero"
    })
    with pytest.raises(SystemExit, match="warrant contradiction-warrant has no requires edge"):
        module.validate(write_graph(tmp_path, doc))


def test_warrants_from_premise_has_bad_signature(tmp_path):
    doc = json.loads(WARRANT.read_text())
    edge = next(edge for edge in doc["edges"]
                if edge["from"] == "polynomial-identity-warrant" and edge["relation"] == "warrants")
    edge["from"] = "disk-agreement"
    with pytest.raises(SystemExit, match=r"bad signature for warrants: \('premise', 'claim'\)"):
        module.validate(write_graph(tmp_path, doc))


def test_warrant_relation_rejected_in_strategy_form(tmp_path):
    doc = json.loads(STRATEGY.read_text())
    doc["sources"] = {}
    doc["edges"][0]["relation"] = "requires"
    with pytest.raises(SystemExit, match="relation requires is not allowed in strategy-answer"):
        module.validate(write_graph(tmp_path, doc))


def test_lean_hash_drift_is_rejected(tmp_path):
    doc = json.loads(WARRANT.read_text())
    doc["sources"]["lean"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="lean: source hash drifted"):
        module.validate(write_graph(tmp_path, doc))


def test_lean_evidence_must_name_a_bound_identifier(tmp_path):
    doc = json.loads(WARRANT.read_text())
    node = next(node for node in doc["nodes"] if node["id"] == "polynomial-identity")
    node["evidence"] = ["lean:Main.lean#hpolynomial"]
    with pytest.raises(SystemExit, match="names nothing bound in the Lean source"):
        module.validate(write_graph(tmp_path, doc))


def test_warrant_with_two_conclusions_fails(tmp_path):
    doc = json.loads(WARRANT.read_text())
    doc["edges"].append({"from": "top-zero-warrant", "relation": "warrants", "to": "contradiction"})
    with pytest.raises(SystemExit, match="must have exactly one warrants edge"):
        module.validate(write_graph(tmp_path, doc))


def test_unreachable_premise_fails(tmp_path):
    doc = json.loads(WARRANT.read_text())
    doc["edges"] = [edge for edge in doc["edges"] if edge["to"] != "complex-domain"]
    with pytest.raises(SystemExit, match="unreachable from question"):
        module.validate(write_graph(tmp_path, doc))
