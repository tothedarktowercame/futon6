import copy
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "holes/inkup/pilots/a96J06/content-plan.json"
VALIDATOR = ROOT / "holes/inkup/pilots/a96J06/validate_content_plan.py"
spec = importlib.util.spec_from_file_location("validate_content_plan", VALIDATOR)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def validate_doc(tmp_path, doc):
    path = tmp_path / "content-plan.json"
    path.write_text(json.dumps(doc))
    _plan, errors = module.validate(path)
    return errors


def codes(errors):
    return set(re.findall(r"CP\d{3}", "\n".join(errors)))


def test_real_content_plan_passes():
    _plan, errors = module.validate(PLAN)
    assert errors == []


def test_contested_question_cannot_be_anticipated(tmp_path):
    doc = json.loads(PLAN.read_text())
    target = next(q for q in doc["questions"] if q["support_state"] == "contested")
    target["decisions"]["teaching"] = {"action": "anticipate", "reason": "bad fixture"}
    errors = validate_doc(tmp_path, doc)
    assert codes(errors) == {"CP005"}, errors


def test_question_node_must_exist(tmp_path):
    doc = json.loads(PLAN.read_text())
    doc["questions"][0]["blueprint_node"] = "a96J06:B999"
    errors = validate_doc(tmp_path, doc)
    assert codes(errors) == {"CP002"}, errors


def test_every_profile_needs_a_decision(tmp_path):
    doc = json.loads(PLAN.read_text())
    del doc["questions"][0]["decisions"]["formal-audit"]
    errors = validate_doc(tmp_path, doc)
    assert codes(errors) == {"CP003"}, errors
