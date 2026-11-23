import copy
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "holes/inkup/pilots/a96J06"
SIDECAR = PILOT / "realizations.sidecar.json"
VALIDATOR = PILOT / "validate_sidecar.py"
spec = importlib.util.spec_from_file_location("validate_sidecar", VALIDATOR)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def validate_doc(tmp_path, doc):
    path = tmp_path / "realizations.sidecar.json"
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return module.validate(path)[1]


def codes(errors):
    return set(re.findall(r"SC\d{3}", "\n".join(errors)))


def entry(doc, profile, node):
    return next(e for e in doc["profiles"][profile]["nodes"] if e["node"] == node)


def question(doc, profile, qid):
    return next(q for q in doc["profiles"][profile]["questions"] if q["question"] == qid)


def test_real_sidecar_passes():
    report, errors = module.validate(SIDECAR)
    assert errors == []
    assert set(report) == {"expert-reference", "teaching", "formal-audit"}


def test_every_mandatory_node_scored_in_every_profile():
    doc = json.loads(SIDECAR.read_text())
    for profile, body in doc["profiles"].items():
        assert [e["node"] for e in body["nodes"]] == [f"a96J06:B{k}" for k in range(7)]


def test_altered_quote_fails(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    target = question(doc, "teaching", "scaling-strategy")
    target["quote"] = target["quote"].replace("divides", "multiplies")
    target["ink_words"] = module.words(target["quote"])
    assert codes(validate_doc(tmp_path, doc)) == {"SC009"}


def test_wrong_ink_words_fails(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    entry(doc, "teaching", "a96J06:B5")["ink_words"] += 1
    assert codes(validate_doc(tmp_path, doc)) == {"SC006"}


def test_mandatory_node_entry_removed_fails(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    body = doc["profiles"]["formal-audit"]
    body["nodes"] = [e for e in body["nodes"] if e["node"] != "a96J06:B3"]
    assert codes(validate_doc(tmp_path, doc)) == {"SC003"}


def test_question_entry_removed_for_anticipate_fails(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    body = doc["profiles"]["teaching"]
    body["questions"] = [q for q in body["questions"] if q["question"] != "scaling-strategy"]
    assert codes(validate_doc(tmp_path, doc)) == {"SC013"}


def test_realized_claimed_for_node_without_marker_fails(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    target = entry(doc, "expert-reference", "a96J06:B0")
    target.update({"status": "realized", "ink_words": 0, "ink_percent": 0.0})
    target.pop("merged_into")
    target.pop("quote")
    assert codes(validate_doc(tmp_path, doc)) == {"SC005"}


def test_merged_quote_must_carry_the_obligation(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    target = entry(doc, "expert-reference", "a96J06:B3")
    target["quote"] = "Comparing their degree-(N) coefficients gives"
    assert codes(validate_doc(tmp_path, doc)) == {"SC008"}


def test_deferred_question_cannot_sit_under_questions(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    body = doc["profiles"]["teaching"]
    moved = body["unplanned_answers"].pop(0)
    body["questions"].append(moved)
    assert codes(validate_doc(tmp_path, doc)) == {"SC011"}


def test_partial_answer_needs_missing_statement(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    question(doc, "expert-reference", "scaling-strategy").pop("missing")
    assert codes(validate_doc(tmp_path, doc)) == {"SC012"}


def test_profile_must_match_plan(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    del doc["profiles"]["formal-audit"]
    assert codes(validate_doc(tmp_path, doc)) == {"SC001"}
