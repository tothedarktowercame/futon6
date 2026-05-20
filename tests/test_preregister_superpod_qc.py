from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(root: Path):
    spec = importlib.util.spec_from_file_location(
        "preregister_superpod_qc_for_test",
        root / "src" / "futon6" / "preregister_superpod_qc.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, *, entity_count: int, scope_cov: float, avg_nodes: float,
                    with_claims: int, papers: int, discover_terms: bool = False,
                    learned: int = 0, rhs_supported: int = 0, new_terms: int = 0,
                    paper_eprint_dir: str | None = "/tmp/eprints",
                    text_source_counts: dict | None = None,
                    health_issues: list[dict] | None = None):
    payload = {
        "entity_count": entity_count,
        "paper_eprint_dir": paper_eprint_dir,
        "discover_terms": discover_terms,
        "readiness": {"status": "pass", "issues": len(health_issues or []), "preflight": False},
        "health_issues": health_issues or [],
        "stage5_stats": {
            "scope_coverage": scope_cov,
            "text_source_counts": text_source_counts or {"eprint": entity_count, "abstract": 0},
            "open_ner": {
                "learned_dictionary_written": learned,
                "rhs_supported_terms": rhs_supported,
                "new_terms_learned": new_terms,
            },
        },
        "stage9a_stats": {
            "avg_nodes": avg_nodes,
            "avg_edges": max(0.0, avg_nodes - 1.0),
            "geometry_stats": {
                "papers": papers,
                "with_claims": with_claims,
            },
        },
        "stage_status": {
            "ner_scopes": {
                "status": "completed",
                "entities_processed": entity_count,
                "text_source_counts": text_source_counts or {"eprint": entity_count, "abstract": 0},
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_preregister_superpod_qc_passes_expected_run(tmp_path: Path):
    root = Path(__file__).parent.parent
    module = _load_module(root)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()

    _write_manifest(baseline_dir / "001.json", entity_count=5000, scope_cov=0.18, avg_nodes=60.0, with_claims=1300, papers=5000)
    _write_manifest(baseline_dir / "002.json", entity_count=5000, scope_cov=0.21, avg_nodes=63.0, with_claims=1350, papers=5000,
                    health_issues=[{"stage": "Stage 9b", "message": "val Acc@1 0.978 < 0.980 threshold"}])
    _write_manifest(baseline_dir / "003.json", entity_count=5000, scope_cov=0.23, avg_nodes=66.0, with_claims=1400, papers=5000)

    current = tmp_path / "manifest.json"
    _write_manifest(current, entity_count=5000, scope_cov=0.22, avg_nodes=64.0, with_claims=1375, papers=5000,
                    discover_terms=True, learned=12, rhs_supported=10, new_terms=5)

    report = module.build_report(current, baseline_dir, "broad-arxiv")
    assert report["evaluation"]["overall"] == "pass"
    gate_names = {gate["name"]: gate for gate in report["evaluation"]["gates"]}
    assert gate_names["paper_text_provenance"]["status"] == "pass"
    assert gate_names["term_learning_prediction"]["status"] == "pass"
    assert gate_names["scope_coverage_prediction"]["status"] == "pass"


def test_preregister_superpod_qc_flags_bad_provenance_and_learning(tmp_path: Path):
    root = Path(__file__).parent.parent
    module = _load_module(root)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()

    _write_manifest(baseline_dir / "001.json", entity_count=5000, scope_cov=0.18, avg_nodes=60.0, with_claims=1300, papers=5000)
    _write_manifest(baseline_dir / "002.json", entity_count=5000, scope_cov=0.21, avg_nodes=63.0, with_claims=1350, papers=5000)
    _write_manifest(baseline_dir / "003.json", entity_count=5000, scope_cov=0.23, avg_nodes=66.0, with_claims=1400, papers=5000)

    current = tmp_path / "manifest.json"
    _write_manifest(current, entity_count=5000, scope_cov=0.10, avg_nodes=40.0, with_claims=400, papers=5000,
                    discover_terms=True, learned=0, rhs_supported=0, new_terms=0,
                    text_source_counts={"eprint": 0, "abstract": 5000},
                    health_issues=[{"stage": "Stage 6", "message": "unexpected parse collapse"}])

    report = module.build_report(current, baseline_dir, "broad-arxiv")
    assert report["evaluation"]["overall"] == "fail"
    gate_names = {gate["name"]: gate for gate in report["evaluation"]["gates"]}
    assert gate_names["paper_text_provenance"]["status"] == "fail"
    assert gate_names["scope_coverage_prediction"]["status"] == "fail"
    assert gate_names["hypergraph_density_prediction"]["status"] == "fail"
    assert gate_names["term_learning_prediction"]["status"] == "warn"
    assert gate_names["health_issue_profile"]["status"] == "warn"
