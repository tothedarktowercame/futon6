import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mark3_coverage_model", ROOT / "scripts" / "mark3_coverage_model.py")
mark3_coverage_model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mark3_coverage_model
SPEC.loader.exec_module(mark3_coverage_model)


def test_load_hierarchy_gold_and_weak_records(tmp_path):
    hierarchy_path, close_dir, proposals_dir = mark3_coverage_model.create_self_test_fixture(tmp_path)

    hierarchy = mark3_coverage_model.load_hierarchy(hierarchy_path)
    gold = mark3_coverage_model.load_gold_records(close_dir, hierarchy)
    weak = mark3_coverage_model.load_weak_records(proposals_dir, hierarchy)

    assert sorted(hierarchy) == ["connection", "connection/transfer", "rationale/telos"]
    assert [record.label for record in gold] == [
        "rationale/telos",
        "none",
        "connection/transfer",
        "connection",
    ]
    assert [record.label for record in weak] == ["rationale/telos", "connection/transfer"]


def test_eval_self_fixture_returns_metrics(tmp_path):
    hierarchy_path, close_dir, proposals_dir = mark3_coverage_model.create_self_test_fixture(tmp_path)
    args = mark3_coverage_model.parse_args(
        [
            "--hierarchy",
            str(hierarchy_path),
            "--close-reading-dir",
            str(close_dir),
            "--proposals-dir",
            str(proposals_dir),
            "--gh200-dir",
            str(ROOT / "data" / "showcases" / "ct-anatomy" / "gh200"),
            "--heldout-pct",
            "0.5",
        ]
    )

    report = mark3_coverage_model.evaluate(args)

    assert report["data"]["gold_records"] == 4
    assert report["data"]["weak_records"] == 2
    assert report["model_eval"]["n"] > 0
    assert "coverage_pct" in report["delta_vs_34_72_baseline"]
