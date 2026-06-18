import importlib.util
import json
import sys
from pathlib import Path

from scripts.r2d_concept_coverage import load_edn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cas_segment", ROOT / "scripts" / "cas_segment.py")
cas_segment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cas_segment
SPEC.loader.exec_module(cas_segment)

SELECT_SPEC = importlib.util.spec_from_file_location("cas_select", ROOT / "scripts" / "cas_select.py")
cas_select = importlib.util.module_from_spec(SELECT_SPEC)
assert SELECT_SPEC.loader is not None
sys.modules[SELECT_SPEC.name] = cas_select
SELECT_SPEC.loader.exec_module(cas_select)

GRAPH = ROOT / "data" / "iatc-argument-graphs" / "loop-run-70b" / "0709.0248.edn"


def test_segment_schema_determinism_and_cas_select_load(tmp_path):
    out1 = cas_segment.write_steps(GRAPH, out_path=tmp_path / "one.steps.json")
    out2 = cas_segment.write_steps(GRAPH, out_path=tmp_path / "two.steps.json")

    assert out1.read_bytes() == out2.read_bytes()

    doc = cas_select.load_steps(out1)
    assert doc["paper_id"] == "0709.0248"
    assert [row["id"] for row in doc["steps"]] == [f"s{i}" for i in range(1, len(doc["steps"]) + 1)]
    assert all(row["text"] for row in doc["steps"])


def test_segment_order_and_edge_plus_setup_coverage():
    graph = load_edn(GRAPH)
    entries = cas_segment.build_step_entries(graph)
    edges = [edge for edge in graph["edges"] if edge.get("kind") == ":infer"]
    conclusions = {edge["conclusion"] for edge in edges}
    setup_nodes = [node for node in graph["nodes"] if node["id"] not in conclusions]

    assert len(entries) == len(edges) + len(setup_nodes)
    assert sum(1 for row in entries if row["kind"] == "edge") == len(edges)
    assert sum(1 for row in entries if row["kind"] == "setup") == len(setup_nodes)
    assert [(row["line"], row["end_line"], row["id"]) for row in entries] == sorted(
        (row["line"], row["end_line"], row["id"]) for row in entries
    )


def test_segment_text_is_resolved_math_prose_not_debug():
    doc = cas_segment.segment_graph(GRAPH)
    text = "\n".join(row["text"] for row in doc["steps"])

    assert "relation :" not in text
    assert "premise :" not in text
    assert ":e-" not in text
    assert ":parameterized-rules" not in text
    assert "parameterized versions of the rules governing identity types" in text
    assert "every locally cartesian closed category is extensional" in text


def test_segment_cli_stdout_is_steps_json(capsys):
    assert cas_segment.main([str(GRAPH), "--stdout"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["paper_id"] == "0709.0248"
    assert payload["steps"][0]["id"] == "s1"
