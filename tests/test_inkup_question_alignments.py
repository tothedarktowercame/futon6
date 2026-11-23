import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "holes/inkup/reasoning/validate_question_alignments.py"
FIXTURES = ROOT / "holes/inkup/reasoning/fixtures"


def run(name, candidates=False):
    command = [sys.executable, str(SCRIPT), str(FIXTURES / name)]
    if candidates:
        command += ["--candidates", str(FIXTURES / "candidates.jsonl")]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def codes(result):
    return set(re.findall(r"QA\d{3}", result.stdout + result.stderr))


def test_good_fixture_passes_and_reports_all_states():
    result = run("good.jsonl")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "valid question alignments: 3" in result.stdout
    assert "mapping_state: accepted=1, contested=1, rejected=1" in result.stdout
    assert "support_state: contested=1, supported=1, unsupported=1" in result.stdout


def test_bad_rule1_has_only_its_code():
    result = run("bad-rule1.jsonl")
    assert result.returncode != 0 and codes(result) == {"QA001"}


def test_bad_rule2_has_only_its_code():
    result = run("bad-rule2.jsonl")
    assert result.returncode != 0 and codes(result) == {"QA002"}


def test_bad_rule3_node_only_has_only_its_code():
    result = run("bad-rule3-node-only.jsonl")
    assert result.returncode != 0 and codes(result) == {"QA003"}


def test_bad_rule3_wrong_edge_has_only_its_code():
    result = run("bad-rule3-wrong-edge.jsonl")
    assert result.returncode != 0 and codes(result) == {"QA003"}


def test_bad_rule4_has_only_its_code():
    result = run("bad-rule4.jsonl")
    assert result.returncode != 0 and codes(result) == {"QA004"}


def test_bad_rule5_has_only_its_code():
    result = run("bad-rule5.jsonl")
    assert result.returncode != 0 and codes(result) == {"QA005"}


def test_bad_rule6_has_only_its_code():
    result = run("bad-rule6.jsonl", candidates=True)
    assert result.returncode != 0 and codes(result) == {"QA006"}


def test_bad_rule7_has_only_its_code():
    result = run("bad-rule7.jsonl")
    assert result.returncode != 0 and codes(result) == {"QA007"}
