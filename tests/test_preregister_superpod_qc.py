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


# ============================================================
# Structure-learning + comment gates and headline summary
# ============================================================

def _write_manifest_with_structure_learning(
    path: Path,
    *,
    candidates: list[dict],
    seed_signatures_loaded: int = 0,
    seed_matches_applied: int = 0,
    entities_with_seed_matches: int = 0,
    total_comments: int = 0,
    entities_with_comments: int = 0,
    free_floating_term_ratio: float | None = 0.42,
):
    payload = {
        "entity_count": 5000,
        "paper_eprint_dir": "/tmp/eprints",
        "discover_terms": False,
        "readiness": {"status": "pass", "issues": 0, "preflight": False},
        "health_issues": [],
        "stage5_stats": {
            "scope_coverage": 0.22,
            "text_source_counts": {"eprint": 5000, "abstract": 0},
            "open_ner": {},
            "total_comments": total_comments,
            "entities_with_comments": entities_with_comments,
            "structure_learning": {
                "enabled": True,
                "candidates_written": len(candidates),
                "seed_signatures_loaded": seed_signatures_loaded,
                "seed_matches_applied": seed_matches_applied,
                "entities_with_seed_matches": entities_with_seed_matches,
                "structure_seed_candidates": candidates,
                "loss": {
                    "free_floating_term_ratio": free_floating_term_ratio,
                    "uncovered_sentences_with_known_terms": 220,
                },
            },
        },
        "stage9a_stats": {
            "avg_nodes": 64.0,
            "avg_edges": 63.0,
            "geometry_stats": {"papers": 5000, "with_claims": 1380},
        },
        "stage_status": {"ner_scopes": {"status": "completed",
                                         "text_source_counts": {"eprint": 5000, "abstract": 0}}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_qc_passes_structure_learning_gates_when_capture_and_gated_yield_present(tmp_path):
    root = Path(__file__).parent.parent
    module = _load_module(root)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    _write_manifest(baseline_dir / "001.json", entity_count=5000, scope_cov=0.21, avg_nodes=64.0, with_claims=1350, papers=5000)

    candidates = [
        {"signature": "be obtain", "paper_count": 3, "count": 4, "predicted_kind": "label"},
        {"signature": "let define", "paper_count": 2, "count": 2, "predicted_kind": "scope"},
        # Below gate (paper_count=1)
        {"signature": "show that", "paper_count": 1, "count": 1, "predicted_kind": "label"},
        # No predicted_kind
        {"signature": "<term>", "paper_count": 5, "count": 9, "predicted_kind": None},
    ]
    current = tmp_path / "manifest.json"
    _write_manifest_with_structure_learning(
        current, candidates=candidates, total_comments=412, entities_with_comments=22
    )

    report = module.build_report(current, baseline_dir, "broad-arxiv")
    gate_names = {gate["name"]: gate for gate in report["evaluation"]["gates"]}
    assert gate_names["structure_learning_capture"]["status"] == "pass"
    assert "4" in gate_names["structure_learning_capture"]["message"]  # candidates_written=4
    assert gate_names["gated_pattern_yield"]["status"] == "pass"
    assert "2" in gate_names["gated_pattern_yield"]["message"]  # 2 candidates cleared the gate

    head = report["headline_summary"]
    assert head["structure_learning_enabled"] is True
    assert head["candidates_discovered"] == 4
    assert head["candidates_classified"] == 3
    assert head["gated_for_promotion"] == 2
    assert head["candidates_kind_breakdown"] == {"label": 2, "scope": 1}
    assert head["comment_scopes_total"] == 412
    assert head["entities_with_comments"] == 22


def test_qc_warns_when_seed_loaded_but_no_matches(tmp_path):
    root = Path(__file__).parent.parent
    module = _load_module(root)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    _write_manifest(baseline_dir / "001.json", entity_count=5000, scope_cov=0.21, avg_nodes=64.0, with_claims=1350, papers=5000)

    current = tmp_path / "manifest.json"
    _write_manifest_with_structure_learning(
        current,
        candidates=[{"signature": "be obtain", "paper_count": 2, "count": 2, "predicted_kind": "label"}],
        seed_signatures_loaded=7,
        seed_matches_applied=0,
        entities_with_seed_matches=0,
    )
    report = module.build_report(current, baseline_dir, "broad-arxiv")
    gate_names = {gate["name"]: gate for gate in report["evaluation"]["gates"]}
    assert gate_names["structure_seed_replay"]["status"] == "warn"
    assert "7" in gate_names["structure_seed_replay"]["message"]


def test_qc_warns_when_zero_candidates_despite_enabled(tmp_path):
    root = Path(__file__).parent.parent
    module = _load_module(root)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    _write_manifest(baseline_dir / "001.json", entity_count=5000, scope_cov=0.21, avg_nodes=64.0, with_claims=1350, papers=5000)

    current = tmp_path / "manifest.json"
    _write_manifest_with_structure_learning(current, candidates=[])
    report = module.build_report(current, baseline_dir, "broad-arxiv")
    gate_names = {gate["name"]: gate for gate in report["evaluation"]["gates"]}
    assert gate_names["structure_learning_capture"]["status"] == "warn"


def test_headline_summary_includes_audit_aggregate_when_present(tmp_path):
    root = Path(__file__).parent.parent
    module = _load_module(root)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    _write_manifest(baseline_dir / "001.json", entity_count=5000, scope_cov=0.21, avg_nodes=64.0, with_claims=1350, papers=5000)

    # Manifest with an audit_summary block alongside structure_learning.
    current = tmp_path / "manifest.json"
    payload = {
        "entity_count": 5000,
        "paper_eprint_dir": "/tmp/eprints",
        "discover_terms": False,
        "readiness": {"status": "pass", "issues": 0, "preflight": False},
        "health_issues": [],
        "stage5_stats": {
            "scope_coverage": 0.22,
            "text_source_counts": {"eprint": 5000, "abstract": 0},
            "open_ner": {},
            "total_comments": 0,
            "entities_with_comments": 0,
            "structure_learning": {
                "enabled": True,
                "candidates_written": 1,
                "structure_seed_candidates": [
                    {"signature": "be obtain", "paper_count": 2, "predicted_kind": "label"},
                ],
                "loss": {"free_floating_term_ratio": 0.27, "uncovered_sentences_with_known_terms": 12},
            },
            "audit_summary": {
                "sample_size": 30,
                "aggregate": {
                    "inhabited": 7500,
                    "outer": 2500,
                    "straddled": 50,
                    "total": 10050,
                    "frontier_ratio": 0.2488,
                },
            },
        },
        "stage9a_stats": {
            "avg_nodes": 64.0,
            "avg_edges": 63.0,
            "geometry_stats": {"papers": 5000, "with_claims": 1380},
        },
        "stage_status": {"ner_scopes": {"status": "completed",
                                         "text_source_counts": {"eprint": 5000, "abstract": 0}}},
    }
    current.write_text(json.dumps(payload), encoding="utf-8")

    report = module.build_report(current, baseline_dir, "broad-arxiv")
    head = report["headline_summary"]
    assert head["audit_sample_size"] == 30
    assert head["audit_inhabited_terms"] == 7500
    assert head["audit_outer_terms"] == 2500
    assert head["audit_straddled_terms"] == 50
    assert abs(head["audit_frontier_ratio"] - 0.2488) < 1e-6


def test_headline_summary_handles_missing_structure_learning_block(tmp_path):
    root = Path(__file__).parent.parent
    module = _load_module(root)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    _write_manifest(baseline_dir / "001.json", entity_count=5000, scope_cov=0.21, avg_nodes=64.0, with_claims=1350, papers=5000)

    current = tmp_path / "manifest.json"
    # Run without structure-learning enabled (no structure_learning key).
    _write_manifest(current, entity_count=5000, scope_cov=0.22, avg_nodes=64.0, with_claims=1375, papers=5000)
    report = module.build_report(current, baseline_dir, "broad-arxiv")
    head = report["headline_summary"]
    assert head["structure_learning_enabled"] is False
    assert head["candidates_discovered"] == 0
    # No structure_learning gates should fire in this case.
    gate_names = {gate["name"] for gate in report["evaluation"]["gates"]}
    assert "structure_learning_capture" not in gate_names
