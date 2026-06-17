import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "cas-select"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cas = load_module("cas_select_test", ROOT / "scripts" / "cas_select.py")
cas_checks = load_module("cas_checks_test", ROOT / "scripts" / "cas_checks.py")


def load_fixture(paper_id):
    steps = json.loads((FIXTURES / f"{paper_id}.steps.json").read_text())
    oracle_rows = json.loads((FIXTURES / f"{paper_id}.oracle.json").read_text())["matches"]
    oracle = {row["step"]: row for row in oracle_rows}
    return steps, oracle


def select_with_patterns(paper_id, patterns):
    steps, oracle = load_fixture(paper_id)
    return cas.select_proof(steps, patterns, backend="stub", oracle=oracle, k=4)


def emitted_labels(select_output):
    return [
        (row["step"], row["pattern"], label)
        for row in select_output["checks"]
        for label in row["fires"]
    ]


def test_registry_reproduces_cas_select_static_checks_for_worked_proofs():
    patterns = cas.load_patterns()

    for paper_id in ["a93J05", "a96J01", "b97J01", "a96J04"]:
        result = select_with_patterns(paper_id, patterns)
        selected = cas_checks.select_registry_entries(result)

        assert [
            (row["step"], row["pattern"], row["label"])
            for row in selected
        ] == emitted_labels(result)
        assert all(row["registered"] for row in selected)


def test_registry_stubs_unbuilt_proof_shape_checks_as_na():
    select_output = {
        "paper_id": "toy",
        "checks": [
            {
                "step": "s1",
                "pattern": "count-over-a-decomposition",
                "fires": ["decomposition-exhaustive"],
            },
            {
                "step": "s2",
                "pattern": "epsilon-of-room",
                "fires": ["forall-eps-structure"],
            },
            {
                "step": "s3",
                "pattern": "separate-into-independent-pieces",
                "fires": ["R2b-disjointness"],
            },
            {
                "step": "s4",
                "pattern": "split-into-cases",
                "fires": ["cases-exhaustive"],
            },
        ],
    }

    result = cas_checks.run_selected_checks(select_output)

    assert result["status"] == "pass"
    assert [(row["dispatch-label"], row["status"], row["pass"]) for row in result["per-item"]] == [
        ("decomposition-exhaustive", "na", True),
        ("forall-eps-structure", "na", True),
        ("R2b-disjointness", "na", True),
        ("cases-exhaustive", "na", True),
    ]


def test_registry_executes_built_rung2_checks_on_loop_run_graph():
    graph = ROOT / "data" / "iatc-argument-graphs" / "loop-run-70b" / "0706.1286.edn"
    select_output = {
        "paper_id": "0706.1286",
        "checks": [
            {"step": "s1", "pattern": "reduce-to-known-result", "fires": ["R2c-warrant"]},
            {"step": "s2", "pattern": "local-to-global", "fires": ["R2b-closure"]},
            {"step": "s3", "pattern": "concept-profile", "fires": ["R2d-concept-coverage"]},
        ],
    }

    result = cas_checks.run_selected_checks(select_output, graph_path=graph)
    by_label = {row["dispatch-label"]: row for row in result["per-item"]}

    assert result["pass"] is True
    assert by_label["R2c-warrant"]["check"] == "warrant-resolution"
    assert by_label["R2c-warrant"]["status"] == "pass"
    assert by_label["R2c-warrant"]["rate"] == 0.2
    assert by_label["R2b-closure"]["check"] == "closure"
    assert by_label["R2b-closure"]["status"] == "pass"
    assert by_label["R2b-closure"]["rate"] == 1.0
    assert by_label["R2d-concept-coverage"]["check"] == "concept-coverage"
    assert by_label["R2d-concept-coverage"]["status"] == "pass"
    assert by_label["R2d-concept-coverage"]["rate"] == 0.5
    assert "calmod like bicategory" in by_label["R2d-concept-coverage"]["undefined"]


def test_unknown_registry_label_is_na_not_fail():
    result = cas_checks.run_selected_checks(
        {"checks": [{"step": "s1", "pattern": "toy", "fires": ["future-check"]}]}
    )

    assert result["pass"] is True
    assert result["per-item"][0]["dispatch-label"] == "future-check"
    assert result["per-item"][0]["status"] == "na"
