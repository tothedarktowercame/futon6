"""Tests for scripts/compute-paper-T.py (mark3 geometry artifact).

Hand-checked fixtures for the tension scalar T_total, the unpaired-claim
ids, and the Δ-Laplacian incidence summary, plus the geometry.json
artifact emitted per paper and per batch (mission M-superpod-mark3).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "compute_paper_T", ROOT / "scripts" / "compute-paper-T.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cpt():
    return _load_module()


def _paper_a():
    """Three claims; c1 and c2 grounded by derivations, c3 unpaired.

    Edge e3 links the unpaired claim c3 to concept s2 only, so the
    Δ-Laplacian incidence map is {s2: 1} and top support is s2.
    Hand-check: n_claims=3, n_unpaired=1, T_total=1/3.
    """
    return {
        "paper_id": "arxiv-2501.00001",
        "nodes": [
            {"id": "c1", "type": "claim"},
            {"id": "c2", "type": "claim"},
            {"id": "c3", "type": "claim"},
            {"id": "s1", "type": "concept"},
            {"id": "s2", "type": "concept"},
        ],
        "edges": [
            {"id": "e1", "type": "derivation", "roles": {"s1": "source", "c1": "target"}},
            {"id": "e2", "type": "derivation", "roles": {"s1": "source", "c2": "target"}},
            {"id": "e3", "type": "relates", "roles": {"c3": "end", "s2": "end"}},
        ],
    }


def _paper_b():
    """No claims: sentinel T_total = -1, no incidence."""
    return {
        "paper_id": "arxiv-2501.00002",
        "nodes": [{"id": "s3", "type": "concept"}],
        "edges": [{"id": "e4", "type": "relates", "roles": {"s3": "end"}}],
    }


def test_compute_one_paper_hand_checked(cpt):
    r = cpt.compute_one_paper(_paper_a())
    assert r["n_claims"] == 3
    assert r["n_unpaired"] == 1
    assert r["T_total"] == pytest.approx(1 / 3)
    assert r["unpaired_claim_ids"] == ["c3"]
    assert r["laplacian_summary"]["incidence"] == {"s2": 1}
    assert r["laplacian_summary"]["top_support_id"] == "s2"
    assert r["laplacian_summary"]["top_support_count"] == 1


def test_compute_one_paper_empty_claims_sentinel(cpt):
    r = cpt.compute_one_paper(_paper_b())
    assert r["n_claims"] == 0
    assert r["n_unpaired"] == 0
    assert r["T_total"] == -1.0
    assert r["unpaired_claim_ids"] == []
    assert r["laplacian_summary"]["incidence"] == {}
    assert r["laplacian_summary"]["top_support_id"] == ""


def test_batch_geometry_hand_checked(cpt):
    rows = [cpt.compute_one_paper(_paper_a()), cpt.compute_one_paper(_paper_b())]
    batch = cpt.compute_batch_geometry(rows)
    assert batch["n_papers"] == 2
    assert batch["n_papers_with_claims"] == 1
    assert batch["total_claims"] == 3
    assert batch["total_unpaired"] == 1
    assert batch["batch_T_total"] == pytest.approx(1 / 3)
    assert batch["mean_paper_T_total"] == pytest.approx(1 / 3)
    assert batch["laplacian_summary"]["incidence"] == {"s2": 1}
    assert batch["laplacian_summary"]["top_support_id"] == "s2"


def test_geometry_document_shape(cpt):
    rows = [cpt.compute_one_paper(_paper_a()), cpt.compute_one_paper(_paper_b())]
    doc = cpt.build_geometry_document([_paper_a(), _paper_b()], rows)
    assert doc["schema_version"] == "mark3-geometry-v1"
    assert doc["n_papers"] == 2
    assert [p["paper_id"] for p in doc["papers"]] == [
        "arxiv-2501.00001",
        "arxiv-2501.00002",
    ]
    assert doc["papers"][0]["T_total"] == pytest.approx(1 / 3)
    assert doc["batch"]["total_unpaired"] == 1
    # No timestamps anywhere: the artifact is a fixed point of its input.
    assert "generated_at" not in json.dumps(doc)


def _write_batch(tmp_path: Path) -> Path:
    batch = tmp_path / "batch"
    (batch / "output").mkdir(parents=True)
    (batch / "output" / "paper-hypergraphs.json").write_text(
        json.dumps([_paper_a(), _paper_b()])
    )
    return batch


def test_cli_emits_tsv_and_geometry_json(tmp_path: Path, cpt):
    batch = _write_batch(tmp_path)
    proc = subprocess.run(
        [sys.executable, "scripts/compute-paper-T.py", str(batch)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    tsv = (batch / "output" / "paper-T.tsv").read_text()
    assert tsv.splitlines()[0] == (
        "paper_id\tn_claims\tn_unpaired\tT_total\ttop_support_id\ttop_support_count"
    )
    assert "arxiv-2501.00001\t3\t1\t0.3333\ts2\t1" in tsv
    geometry = json.loads((batch / "output" / "geometry.json").read_text())
    assert geometry["schema_version"] == "mark3-geometry-v1"
    assert geometry["batch"]["total_unpaired"] == 1


def test_cli_is_deterministic_fixed_point(tmp_path: Path):
    """Run the generator twice; the second run must leave no diff."""
    batch = _write_batch(tmp_path)
    run = lambda: subprocess.run(
        [sys.executable, "scripts/compute-paper-T.py", str(batch)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert run().returncode == 0
    first = (batch / "output" / "geometry.json").read_bytes()
    tsv_first = (batch / "output" / "paper-T.tsv").read_bytes()
    assert run().returncode == 0
    assert (batch / "output" / "geometry.json").read_bytes() == first
    assert (batch / "output" / "paper-T.tsv").read_bytes() == tsv_first
