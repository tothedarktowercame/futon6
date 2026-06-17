from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_apm_crossdisc_pool", ROOT / "scripts" / "build_apm_crossdisc_pool.py"
)
POOL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = POOL
SPEC.loader.exec_module(POOL)


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def test_keyword_profile_filters_texish_and_keeps_math_terms(tmp_path):
    apm = tmp_path / "apm"
    apm.mkdir()
    (apm / "p1.md").write_text(
        r"""
        \begin{proof}
        Let f be a measurable function on a compact metric space.
        The integral sequence converges by dominated convergence.
        \end{proof}
        """,
        encoding="utf-8",
    )
    profile = POOL.build_keyword_profile(apm, top_k=50)
    terms = {row["term"] for row in profile["terms"]}
    assert "measurable function" in terms
    assert "compact metric space" in terms
    assert "begin" not in terms
    assert "proof" not in terms


def test_build_pool_excludes_ct_and_copies_eprints(tmp_path):
    apm = tmp_path / "apm"
    apm.mkdir()
    (apm / "p1.md").write_text(
        "A measurable function has an integral and a compact metric space "
        "supports convergent sequences.",
        encoding="utf-8",
    )
    rows = [
        {
            "id": "0704.0001v1",
            "base_id": "0704.0001",
            "title": "Measurable functions on compact metric spaces",
            "abstract": "We study integrals and convergence of sequences.",
            "categories": ["math.FA", "math.CA"],
            "primary_category": "math.FA",
        },
        {
            "id": "0704.0002v1",
            "base_id": "0704.0002",
            "title": "Category theoretic compactness",
            "abstract": "A category theory paper using compact objects.",
            "categories": ["math.CT"],
            "primary_category": "math.CT",
        },
        {
            "id": "0704.0003v1",
            "base_id": "0704.0003",
            "title": "Unrelated combinatorics",
            "abstract": "Graphs and colorings.",
            "categories": ["math.CO"],
            "primary_category": "math.CO",
        },
    ]
    batch = tmp_path / "batch-999.tar.gz"
    with tarfile.open(batch, "w:gz") as tf:
        jsonl = "\n".join(json.dumps(r) for r in rows).encode()
        _add_bytes(tf, "batch-999/batch-999.jsonl", jsonl)
        for row in rows:
            _add_bytes(
                tf,
                f"batch-999/eprints/{row['id']}.tar.gz",
                f"payload {row['id']}".encode(),
            )

    out = tmp_path / "out"
    summary = POOL.build_pool(apm, [batch], out, target_size=2)

    assert summary["pool_size"] == 1
    assert summary["math_ct_count"] == 0
    assert summary["eprints_complete"] is True
    assert summary["selected_ids"] == ["0704.0001v1"]
    assert summary["selected_sample"][0]["id"] == "0704.0001v1"
    assert (out / "pool.jsonl").exists()
    assert (out / "eprints" / "0704.0001v1.tar.gz").exists()
