import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOTS = ROOT / "holes/inkup/pilots"
PILOT = PILOTS / "a99J07"
spec = importlib.util.spec_from_file_location("validate_alignment", PILOTS / "validate_alignment.py")
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


SPLIT, INSIDE, OUTSIDE, AUX = "a99J07:B2", "a99J07:B3", "a99J07:B7", "a99J07:B4"


def test_real_alignment_passes():
    doc, errors = module.validate(PILOT / "alignment.json")
    assert errors == []
    assert len(doc["nodes"]) == 9


def test_dependency_on_a_missing_node_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    node(doc, OUTSIDE)["depends_on"].append("a99J07:B9")
    assert codes(validate_doc(tmp_path, doc)) == {"AL004"}


def test_cycle_through_the_rejoining_branch_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    node(doc, AUX)["depends_on"].append(OUTSIDE)
    assert codes(validate_doc(tmp_path, doc)) == {"AL011"}


def test_changed_source_hash_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    doc["sources"]["lean"]["sha256"] = "0" * 64
    assert codes(validate_doc(tmp_path, doc)) == {"AL003"}


def test_missing_source_file_fails(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    doc["sources"]["informal"]["path"] = "apm-lean:problems/a99J07/Absent.md"
    assert codes(validate_doc(tmp_path, doc)) == {"AL002"}


def test_lean_anchor_must_name_an_identifier_in_the_file(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    anchor = next(a for a in node(doc, "a99J07:B5")["anchors"] if a["source"] == "lean")
    anchor["ref"] = "apm_a99J07_reflect_boundary: hv0, hreverse"
    assert codes(validate_doc(tmp_path, doc)) == {"AL007"}


def test_lean_anchor_on_the_wrong_pilot_file_fails(tmp_path):
    """The problem id is read from the alignment, so a99J07 cannot borrow a00J01's lemma names."""
    doc = json.loads((PILOT / "alignment.json").read_text())
    anchor = next(a for a in node(doc, "a99J07:B6")["anchors"] if a["source"] == "lean")
    anchor["ref"] = "apm_a99J07_poly_le_boundary: lower_bound"
    assert codes(validate_doc(tmp_path, doc)) == {"AL007"}


def test_informal_anchor_phrase_must_occur_verbatim(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    anchor = next(a for a in node(doc, "a99J07:B8")["anchors"] if a["source"] == "informal")
    anchor["ref"] = "Combine."
    assert codes(validate_doc(tmp_path, doc)) == {"AL008"}


def test_mined_graph_anchor_must_name_a_node(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    anchor = next(a for a in node(doc, "a99J07:B2")["anchors"] if a["source"] == "mark8-s3")
    anchor["ref"] = ":n42"
    assert codes(validate_doc(tmp_path, doc)) == {"AL010"}


def test_goal_must_depend_on_the_closing_node_only(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    node(doc, "a99J07:B0")["depends_on"] = ["a99J07:B3", "a99J07:B7"]
    assert codes(validate_doc(tmp_path, doc)) == {"AL012"}


def test_problem_id_must_match_the_pilot_directory(tmp_path):
    """The problem id is read from the record, so it is tied to the pilot directory the file lives in."""
    doc = json.loads((PILOT / "alignment.json").read_text())
    path = tmp_path / "a00J01" / "alignment.json"
    path.parent.mkdir()
    path.write_text(json.dumps(doc))
    assert codes(module.validate(path)[1]) == {"AL000"}


# ---- structure the validator cannot see -------------------------------------
# The shared validator checks that dependencies exist and are acyclic; it does not
# know that this proof has a split, so removing the split from a case's
# `depends_on` still validates. The tests below guard the file itself.


def test_removing_the_split_from_a_case_still_validates_so_the_file_is_guarded_directly(tmp_path):
    doc = json.loads((PILOT / "alignment.json").read_text())
    node(doc, OUTSIDE)["depends_on"].remove(SPLIT)
    assert validate_doc(tmp_path, doc) == []


def test_both_cases_depend_on_the_split_node():
    doc = json.loads((PILOT / "alignment.json").read_text())
    assert SPLIT in node(doc, INSIDE)["depends_on"]
    assert SPLIT in node(doc, OUTSIDE)["depends_on"]
    assert node(doc, SPLIT)["depends_on"] == []
    assert node(doc, SPLIT) is not node(doc, INSIDE) and node(doc, SPLIT) is not node(doc, OUTSIDE)


def test_auxiliary_object_is_its_own_node_and_the_use_of_it_is_another():
    doc = json.loads((PILOT / "alignment.json").read_text())
    assert node(doc, AUX)["depends_on"] == []
    assert AUX not in node(doc, AUX)["depends_on"]
    users = [n["id"] for n in doc["nodes"] if AUX in n["depends_on"]]
    assert users == ["a99J07:B5", "a99J07:B6", "a99J07:B7"]


def test_cases_rejoin_at_the_closing_node():
    doc = json.loads((PILOT / "alignment.json").read_text())
    assert node(doc, "a99J07:B8")["depends_on"] == [INSIDE, OUTSIDE]
    assert node(doc, "a99J07:B0")["depends_on"] == ["a99J07:B8"]


def test_every_anchor_is_model_proposed_or_rejected():
    doc = json.loads((PILOT / "alignment.json").read_text())
    assert {a["review"] for n in doc["nodes"] for a in n["anchors"]} == {"model-proposed", "rejected"}
    rejected = [(n["id"], a["ref"]) for n in doc["nodes"] for a in n["anchors"] if a["review"] == "rejected"]
    assert rejected == [("a99J07:B7", ":n10")]


def test_no_host_path_in_the_record():
    text = (PILOT / "alignment.json").read_text()
    assert "/home/" not in text
