import copy
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "holes/inkup/pilots/a96J06"
SIDECAR = PILOT / "realizations.sidecar.json"
TEACHING_SIDECAR = PILOT / "reader-telling.teaching.sidecar.json"
A00_TEACHING_SIDECAR = ROOT / "holes/inkup/pilots/a00J01/reader-telling.teaching.sidecar.json"
VALIDATOR = PILOT / "validate_sidecar.py"
spec = importlib.util.spec_from_file_location("validate_sidecar", VALIDATOR)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def validate_doc(tmp_path, doc):
    path = tmp_path / "realizations.sidecar.json"
    path.write_text(json.dumps(doc, ensure_ascii=False))
    return module.validate(path)[1]


def publication_fixture(tmp_path):
    plan = json.loads((PILOT / "publication-plan.json").read_text())
    telling = tmp_path / "publication.md"
    telling.write_text(
        "## Publication\n\n"
        "**[B1]** one\n\n**[B2]** two\n\n**[B3]** three\n\n"
        "**[B4]** four\n\n**[B5]** five\n\n**[B6]** six conclusion\n"
    )
    node_words = {"a96J06:B0": 0, "a96J06:B1": 1, "a96J06:B2": 1,
                  "a96J06:B3": 1, "a96J06:B4": 1, "a96J06:B5": 1,
                  "a96J06:B6": 2}
    total = sum(node_words.values())
    plan_nodes = {item["node"]: item for item in plan["nodes"]}
    nodes = []
    for node, count in node_words.items():
        planned = plan_nodes[node]
        row = {
            "node": node,
            "status": "merged" if node == "a96J06:B0" else "realized",
            "ink_words": count,
            "ink_percent": module.percent(count, total),
            "planned_percent": planned["budget_percent"],
            "planned_action": planned["action"],
        }
        if node == "a96J06:B0":
            row.update({"merged_into": "a96J06:B6", "quote": "six conclusion"})
        nodes.append(row)
    question_nodes = {
        "scaling-strategy": ("a96J06:B2", "two"),
        "rotation-specializes-to-identity": ("a96J06:B4", "four"),
        "coefficient-extensionality": ("a96J06:B5", "five"),
    }
    questions = [{
        "question": qid, "status": "answered", "node": node,
        "quote": quote, "ink_words": 1,
    } for qid, (node, quote) in question_nodes.items()]
    sidecar_doc = {
        "schema": module.SCHEMA,
        "content_plan": "holes/inkup/pilots/a96J06/content-plan.json",
        "realizations": "unused-by-test",
        "profiles": {"publication": {
            "nodes": nodes, "questions": questions, "unplanned_answers": [],
        }},
    }
    plan_path = tmp_path / "publication-plan.json"
    plan_path.write_text(json.dumps(plan))
    sidecar_path = tmp_path / "publication.sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar_doc))
    return plan, plan_path, telling, sidecar_doc, sidecar_path


def validate_publication(plan_path, telling, sidecar_path):
    return module.validate(sidecar_path, realizations=telling,
                           publication_plan=plan_path)


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


def test_teaching_only_sidecar_and_alternate_telling_pass():
    report, errors = module.validate(TEACHING_SIDECAR)
    assert errors == []
    assert set(report) == {"teaching"}
    assert report["teaching"]["questions"]["answered"] == 7


def test_a00j01_teaching_sidecar_passes():
    report, errors = module.validate(A00_TEACHING_SIDECAR)
    assert errors == []
    assert report["teaching"]["questions"]["answered"] == 4


def test_a00j01_question_quote_must_be_in_telling(tmp_path):
    doc = json.loads(A00_TEACHING_SIDECAR.read_text())
    target = question(doc, "teaching", "why-level-set-strategy")
    target["quote"] = "This sentence is not in the telling."
    target["ink_words"] = module.words(target["quote"])
    assert codes(validate_doc(tmp_path, doc)) == {"SC009"}


def test_publication_plan_drives_profile(tmp_path):
    _plan, plan_path, telling, _doc, sidecar_path = publication_fixture(tmp_path)
    report, errors = validate_publication(plan_path, telling, sidecar_path)
    assert errors == []
    assert report["publication"]["questions"]["answered"] == 3
    assert report["publication"]["words"] == {"actual": 7, "target": 120, "max": 168}


def test_publication_plan_cannot_anticipate_contested_question(tmp_path):
    plan, plan_path, telling, _doc, sidecar_path = publication_fixture(tmp_path)
    plan["questions"]["closed-disk-codomain"]["action"] = "anticipate"
    plan_path.write_text(json.dumps(plan))
    assert codes(validate_publication(plan_path, telling, sidecar_path)[1]) == {"SC016"}


def test_publication_plan_must_name_every_mandatory_node(tmp_path):
    plan, plan_path, telling, _doc, sidecar_path = publication_fixture(tmp_path)
    plan["nodes"] = plan["nodes"][:-1]
    plan_path.write_text(json.dumps(plan))
    assert codes(validate_publication(plan_path, telling, sidecar_path)[1]) == {"SC017"}


def test_publication_plan_budgets_sum_to_100(tmp_path):
    plan, plan_path, telling, _doc, sidecar_path = publication_fixture(tmp_path)
    plan["nodes"][1]["budget_percent"] -= 10
    plan_path.write_text(json.dumps(plan))
    assert codes(validate_publication(plan_path, telling, sidecar_path)[1]) == {"SC018"}


def test_publication_telling_obeys_max_words(tmp_path):
    plan, plan_path, telling, _doc, sidecar_path = publication_fixture(tmp_path)
    plan["max_words"] = 6
    plan_path.write_text(json.dumps(plan))
    assert codes(validate_publication(plan_path, telling, sidecar_path)[1]) == {"SC019"}


def test_publication_answer_for_omitted_question_is_unplanned(tmp_path):
    plan, plan_path, telling, doc, sidecar_path = publication_fixture(tmp_path)
    plan["questions"]["scaling-strategy"]["action"] = "omit"
    plan_path.write_text(json.dumps(plan))
    sidecar_path.write_text(json.dumps(doc))
    assert codes(validate_publication(plan_path, telling, sidecar_path)[1]) == {"SC011"}


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


def test_profile_must_be_a_nonempty_plan_subset(tmp_path):
    doc = json.loads(SIDECAR.read_text())
    doc["profiles"]["unknown-profile"] = doc["profiles"].pop("formal-audit")
    assert codes(validate_doc(tmp_path, doc)) == {"SC001"}


def test_graph_support_edge_must_resolve(tmp_path):
    doc = json.loads(TEACHING_SIDECAR.read_text())
    target = question(doc, "teaching", "scaling-strategy")
    target["graph_support"][0]["edge"]["to"] = "not-a-node"
    assert codes(validate_doc(tmp_path, doc)) == {"SC014"}


def test_graph_support_cannot_be_dropped_from_one_entry(tmp_path):
    doc = json.loads(TEACHING_SIDECAR.read_text())
    question(doc, "teaching", "scaling-strategy").pop("graph_support")
    assert codes(validate_doc(tmp_path, doc)) == {"SC014"}


def test_graph_support_must_use_a_graph_the_plan_entry_cites(tmp_path):
    doc = json.loads(TEACHING_SIDECAR.read_text())
    question(doc, "teaching", "scaling-strategy")["graph_support"] = [{
        "graph": "holes/inkup/reasoning/a96J06-coefficient-bridge.json",
        "edge": {"from": "top-zero-warrant", "relation": "warrants", "to": "coefficient-zero"},
    }]
    assert codes(validate_doc(tmp_path, doc)) == {"SC014"}
