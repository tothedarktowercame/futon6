"""S11's whole-paper signature covers one-proof papers and small corpora."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_paper_signature as cps  # noqa: E402


def breakdown(macro, boxes):
    return {"macro": macro, "n_boxes": boxes, "n_wires": boxes - 1, "n_holes": 0,
            "n_discharges_known": 0, "max_fanout": 1, "depth": boxes, "n_sources": 1, "n_sinks": 1}


def test_one_proof_papers_get_signatures():
    # A problem set: one proof per paper (the APM corpus had 496 papers, 0 signatures).
    ids = ["a00J01__p0", "a00J02__p0", "b93J03__p0"]
    papers, sig, _bypaper, _macros = cps.signatures(
        ids, [breakdown("chain", 3), breakdown("chain", 5), breakdown("split", 4)])
    assert sorted(papers) == ["a00J01", "a00J02", "b93J03"]


def run(tmp_path, ids, rows):
    embed = tmp_path / "embed.json"
    embed.write_text(json.dumps({"ids": ids, "breakdowns": rows}))
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, str(ROOT / "scripts/clean_paper_signature.py"),
                           "--embed", str(embed), "--run-dir", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads((out / "structural-canon.json").read_text())


def test_a_single_paper_has_no_twin_and_does_not_fail(tmp_path):
    canon = run(tmp_path, ["a00J01__p0"], [breakdown("chain", 3)])
    assert canon["n_papers"] == 1 and canon["paper_twin_sim"] is None
    assert canon["signatures"][0]["nearest"] is None


def test_twins_are_named_when_there_are_several(tmp_path):
    canon = run(tmp_path, ["a__p0", "b__p0", "c__p0"],
                [breakdown("chain", 3), breakdown("chain", 3), breakdown("split", 9)])
    nearest = {s["paper"]: s["nearest"][0] for s in canon["signatures"]}
    assert nearest["a"] == "b" and nearest["b"] == "a"
