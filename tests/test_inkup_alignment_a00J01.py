import copy
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "holes/inkup/pilots/a00J01"
spec = importlib.util.spec_from_file_location("validate_alignment", PILOT.parent / "validate_alignment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def validate_doc(tmp_path, doc):
    path = tmp_path / doc.get("problem", "unknown") / "alignment.json"  # pilot directory must match the problem id
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return module.validate(path)[1]


def codes(errors):
    return set(re.findall(r"AL\d{3}", "\n".join(errors)))


def node(doc, node_id):
    return next(n for n in doc["nodes"] if n["id"] == node_id)


def test_real_alignment_passes():
    doc, errors = module.validate(PILOT / "alignment.json")
    assert errors == []
    assert len(doc["nodes"]) == 8


def test_lean_anchor_must_name_an_identifier_in_the_file(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    anchor = next(a for a in node(doc, "a00J01:B3")["anchors"] if a["source"] == "lean")
    anchor["ref"] = "lemma tendsto_rpow_inv_nat_atReal"
    assert codes(validate_doc(tmp_path, doc)) == {"AL007"}


def test_have_names_in_a_lean_anchor_are_checked_too(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    anchor = next(a for a in node(doc, "a00J01:B5")["anchors"] if a["source"] == "lean")
    anchor["ref"] = "lower_bound: hlim, hfinal_bound"
    assert codes(validate_doc(tmp_path, doc)) == {"AL007"}


def test_dependency_on_a_missing_node_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    node(doc, "a00J01:B6")["depends_on"].append("a00J01:B9")
    assert codes(validate_doc(tmp_path, doc)) == {"AL004"}


def test_dependency_cycle_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    node(doc, "a00J01:B4")["depends_on"].append("a00J01:B5")
    assert codes(validate_doc(tmp_path, doc)) == {"AL011"}


def test_changed_source_hash_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    doc["sources"]["lean"]["sha256"] = "0" * 64
    assert codes(validate_doc(tmp_path, doc)) == {"AL003"}


def test_missing_source_file_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    doc["sources"]["lean"]["path"] = "apm-lean:problems/a00J01/lean/Absent.lean"
    assert codes(validate_doc(tmp_path, doc)) == {"AL002"}


def test_host_path_in_a_source_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    doc["sources"]["lean"]["path"] = "/etc/hosts"
    assert codes(validate_doc(tmp_path, doc)) == {"AL002"}


def test_goal_must_depend_on_the_closing_node(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    node(doc, "a00J01:B0")["depends_on"] = ["a00J01:B6"]
    assert codes(validate_doc(tmp_path, doc)) == {"AL012"}


def test_informal_anchor_phrase_must_occur_verbatim(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    anchor = next(a for a in node(doc, "a00J01:B7")["anchors"] if a["source"] == "informal")
    anchor["ref"] = "**Combine the two bounds.**"
    assert codes(validate_doc(tmp_path, doc)) == {"AL008"}


def test_field_outside_the_schema_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    node(doc, "a00J01:B2")["estimate"] = True
    errors = validate_doc(tmp_path, doc)
    assert any("unexpected field" in e for e in errors), errors


def test_every_anchor_is_model_proposed_or_rejected(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    assert {a["review"] for n in doc["nodes"] for a in n["anchors"]} == {"model-proposed", "rejected"}
