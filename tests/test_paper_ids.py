"""Paper ids are read from every stage's file names, arXiv or not."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paper_ids import paper_id_from_name  # noqa: E402


@pytest.mark.parametrize("name, pid", [
    ("math__0608040__p0.edn", "math__0608040"),
    ("0705.0102__p12.edn", "0705.0102"),
    ("math__0310337_math__0310337-inflight-0001_L1-2.edn", "math__0310337"),
    ("0708.1921_0708.1921-leaf-0001_L231-245.edn", "0708.1921"),
    # the APM corpus: ids that are not arXiv ids
    ("a00J01__p0.edn", "a00J01"),
    ("a00J01__p0.rung2.edn", "a00J01"),
    ("bpm-1-4-1__p0.edn", "bpm-1-4-1"),
    ("a00J01_a00J01-section-tail-0005_L121-121.edn", "a00J01"),
    ("bpm-1-4-1_bpm-1-4-1-leaf-0003_L112-121.edn", "bpm-1-4-1"),
    ("a01A04.a01A04-in-proof-0005.candidate.json", "a01A04"),
])
def test_paper_id_from_stage_file_names(name, pid):
    assert paper_id_from_name(name) == pid


@pytest.mark.parametrize("name", ["notes.txt", "README.md", "_hidden__p0.edn", "a00J01.B.json"])
def test_names_that_carry_no_id_in_a_stage_position(name):
    assert paper_id_from_name(name) is None
