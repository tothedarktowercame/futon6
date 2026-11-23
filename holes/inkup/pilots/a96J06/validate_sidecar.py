#!/usr/bin/env python3
"""Validate the a96J06 telling sidecar against the realizations it describes.

Everything countable is recomputed from ``realizations.md``: node passages,
their word counts and percentages, and the location of every quote.  The
sidecar's judgements (status, ``missing``) are not checked here.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCHEMA = "inkup.telling-sidecar/v0"
MARKER = re.compile(r"\*\*\[B(\d+)\]\*\*")
NODE_STATUSES = {"realized", "merged", "absent"}
ANSWER_STATUSES = {"answered", "partly-answered", "not-answered"}
sys.path.insert(0, str(ROOT / "holes/inkup/reasoning"))
from validate_question_alignments import graph_inventory  # noqa: E402


def profile_section_id(profile: str) -> str:
    """``formal-audit`` -> ``Formal audit``, the heading text in realizations.md."""
    headings = {"expert-reference": "Expert reference", "teaching": "Teaching",
                "formal-audit": "Formal audit"}
    return headings[profile]


def adapt_publication_plan(publication: dict, content: dict) -> tuple[dict, list[str]]:
    """Adapt a checked publication plan to the existing content-plan interface."""
    errors: list[str] = []
    content_questions = {question["id"]: question for question in content.get("questions", [])}
    publication_questions = publication.get("questions")
    if not isinstance(publication_questions, dict) or \
            set(publication_questions) != set(content_questions):
        errors.append("SC015 publication-plan question ids must exactly match the content plan")
        publication_questions = publication_questions if isinstance(publication_questions, dict) else {}
    for qid, decision in publication_questions.items():
        if qid in content_questions and isinstance(decision, dict) and \
                decision.get("action") == "anticipate" and \
                content_questions[qid].get("support_state") != "supported":
            errors.append(f"SC016 publication-plan question {qid}: only supported questions may be anticipated")

    mandatory = [item["node"] for item in content.get("mandatory_nodes", [])]
    publication_nodes = publication.get("nodes")
    node_ids = [item.get("node") for item in publication_nodes] \
        if isinstance(publication_nodes, list) and all(isinstance(item, dict) for item in publication_nodes) else []
    nodes_valid = len(node_ids) == len(set(node_ids)) and set(node_ids) == set(mandatory)
    if not nodes_valid:
        errors.append("SC017 publication-plan node ids must exactly match the mandatory nodes")
    budgets = [item.get("budget_percent") for item in publication_nodes] \
        if isinstance(publication_nodes, list) and all(isinstance(item, dict) for item in publication_nodes) else []
    if nodes_valid and (not budgets or
                        not all(isinstance(value, (int, float)) for value in budgets) or
                        sum(budgets) != 100):
        errors.append("SC018 publication-plan node budgets must sum to 100")

    profile = publication.get("profile")
    heading = publication.get("heading")
    if not isinstance(profile, str) or not profile or not isinstance(heading, str) or not heading:
        errors.append("SC000 publication plan needs non-empty profile and heading")
    if not isinstance(publication.get("target_words"), (int, float)) or \
            not isinstance(publication.get("max_words"), (int, float)):
        errors.append("SC000 publication plan needs numeric target_words and max_words")

    adapted = {
        "mandatory_nodes": content.get("mandatory_nodes", []),
        "questions": [
            {
                **question,
                "decisions": {
                    profile: publication_questions.get(question["id"], {})
                },
            }
            for question in content.get("questions", [])
        ],
    }
    return adapted, errors


def parse_sections(text: str) -> dict[str, str]:
    """Heading text (without ``##``) -> section body."""
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^## (.+?)\s*$", text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[match.end():end]
    return sections


def parse_nodes(section: str) -> dict[str, str]:
    """Node number -> passage (marker removed, up to the next marker)."""
    nodes: dict[str, str] = {}
    markers = list(MARKER.finditer(section))
    for index, match in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(section)
        nodes[match.group(1)] = section[match.end():end].strip()
    return nodes


def words(text: str) -> int:
    return len(text.split())


def percent(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def load_plan(path: Path) -> tuple[list[str], dict[str, dict], dict[str, str]]:
    plan = json.loads(path.read_text())
    nodes = [item["node"] for item in plan["mandatory_nodes"]]
    decisions = {question["id"]: question["decisions"] for question in plan["questions"]}
    blueprint = {question["id"]: question["blueprint_node"] for question in plan["questions"]}
    return nodes, decisions, blueprint


def load_ink(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    ink: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            ink[(row["profile"], row["node"])] = row
    return ink


def repo_path(reference: str) -> Path:
    path = (ROOT / reference).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path leaves repository: {reference}") from exc
    return path


def validate(sidecar: Path, plan_path: Path | None = None,
             realizations: Path | None = None,
             ink_path: Path = HERE / "generative-ink.tsv",
             publication_plan: Path | None = None) -> tuple[dict, list[str]]:
    errors: list[str] = []
    try:
        sidecar_doc = json.loads(sidecar.read_text())
        publication = json.loads(publication_plan.read_text()) if publication_plan else None
        if publication is not None:
            plan_path = repo_path(publication["content_plan"])
        else:
            plan_path = plan_path or repo_path(sidecar_doc["content_plan"])
        realizations = realizations or repo_path(sidecar_doc["realizations"])
        plan = json.loads(plan_path.read_text())
        ink = load_ink(ink_path)
        sections = parse_sections(realizations.read_text())
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        return {}, [f"SC000 cannot read inputs: {exc}"]
    if publication is not None:
        plan, plan_errors = adapt_publication_plan(publication, plan)
        if plan_errors:
            return {}, plan_errors
        publication_profile = publication["profile"]
        publication_heading = publication["heading"]
        publication_nodes = {item["node"]: item for item in publication["nodes"]}
    else:
        publication_profile = publication_heading = None
        publication_nodes = {}

    mandatory = [item["node"] for item in plan.get("mandatory_nodes", [])]
    questions = plan.get("questions", [])
    question_plans = {q["id"]: q for q in questions}
    decisions = {q["id"]: q["decisions"] for q in questions}
    blueprint = {q["id"]: q["blueprint_node"] for q in questions}
    profiles_planned = {profile for decision in decisions.values() for profile in decision}

    if sidecar_doc.get("schema") != SCHEMA:
        errors.append("SC000 wrong sidecar schema")
    doc_profiles = sidecar_doc.get("profiles")
    if not isinstance(doc_profiles, dict) or not all(isinstance(p, dict) for p in doc_profiles.values()):
        return {}, errors + ["SC001 profiles must be an object of profile objects"]
    if not doc_profiles or not set(doc_profiles) <= profiles_planned:
        errors.append(f"SC001 profiles {sorted(doc_profiles)} are not a non-empty subset of the plan's {sorted(profiles_planned)}")

    report: dict = {}
    for profile, doc in doc_profiles.items():
        report[profile] = {"nodes": Counter(), "questions": Counter(), "node_rows": [], "unplanned": 0}
        if profile not in profiles_planned:
            continue
        section_id = publication_heading if profile == publication_profile else profile_section_id(profile)
        section = sections.get(section_id)
        if section is None:
            errors.append(f"SC002 {profile}: no section '{section_id}' in realizations.md")
            continue
        passages = parse_nodes(section)
        total = sum(words(passage) for passage in passages.values())
        if profile == publication_profile:
            report[profile]["words"] = {
                "actual": total,
                "target": publication["target_words"],
                "max": publication["max_words"],
            }
            if total > publication["max_words"]:
                errors.append(
                    f"SC019 {profile}: section has {total} words, above max_words "
                    f"{publication['max_words']} (target {publication['target_words']})"
                )

        # ---- nodes -------------------------------------------------------
        entries = doc.get("nodes")
        if not isinstance(entries, list):
            errors.append(f"SC003 {profile}: nodes must be a list")
            entries = []
        seen: dict[str, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("node") not in mandatory:
                errors.append(f"SC003 {profile}: node entry must name a mandatory node: {entry!r}"[:200])
                continue
            if entry["node"] in seen:
                errors.append(f"SC003 {profile}: duplicate node entry {entry['node']}")
            seen[entry["node"]] = entry
        for node in mandatory:
            if node not in seen:
                errors.append(f"SC003 {profile}: no entry for mandatory node {node}")
        for node, entry in seen.items():
            where = f"{profile} {node}"
            status = entry.get("status")
            number = node.rsplit("B", 1)[1]
            has_marker = number in passages
            if status not in NODE_STATUSES:
                errors.append(f"SC004 {where}: unknown status {status!r}")
                report[profile]["nodes"]["?"] += 1
                continue
            report[profile]["nodes"][status] += 1
            expected_words = expected_percent = 0
            if status == "realized" and not has_marker:
                errors.append(f"SC005 {where}: realized but the section has no {node.split(':')[1]} marker")
            if status != "realized" and has_marker:
                errors.append(f"SC005 {where}: {status} but the section has a {node.split(':')[1]} marker")
            if status == "realized":
                expected_words = words(passages.get(number, ""))
                expected_percent = percent(expected_words, total)
            if entry.get("ink_words") != expected_words:
                errors.append(f"SC006 {where}: ink_words {entry.get('ink_words')} != {expected_words}")
            if not isinstance(entry.get("ink_percent"), (int, float)) or \
                    abs(entry["ink_percent"] - expected_percent) > 1e-9:
                errors.append(f"SC006 {where}: ink_percent {entry.get('ink_percent')} != {expected_percent}")
            planned = publication_nodes.get(node, {}) if profile == publication_profile \
                else ink.get((profile, node), {})
            if entry.get("planned_percent") != int(planned.get("budget_percent", -1)):
                errors.append(f"SC007 {where}: planned_percent {entry.get('planned_percent')} "
                              f"!= {planned.get('budget_percent')}")
            if entry.get("planned_action") != planned.get("action"):
                errors.append(f"SC007 {where}: planned_action {entry.get('planned_action')!r} "
                              f"!= {planned.get('action')!r}")
            if status == "merged":
                target = entry.get("merged_into")
                quote = entry.get("quote")
                if not isinstance(target, str) or target == node or target not in mandatory or \
                        target.rsplit("B", 1)[1] not in passages or \
                        seen.get(target, {}).get("status") != "realized":
                    errors.append(f"SC008 {where}: merged_into must be a realized mandatory node: {target!r}")
                elif not isinstance(quote, str) or not quote.strip() or \
                        quote not in passages[target.rsplit("B", 1)[1]]:
                    errors.append(f"SC008 {where}: merged quote not found verbatim in {target}")
            elif "merged_into" in entry:
                errors.append(f"SC008 {where}: merged_into is only allowed when merged")
            if status in {"realized", "merged"}:
                report[profile]["node_rows"].append(
                    (node, entry.get("planned_percent"), expected_percent))

        # ---- question entries --------------------------------------------
        def check_quote(entry: dict, where: str, in_node: bool) -> None:
            quote = entry.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                errors.append(f"SC009 {where}: quote must be a non-empty string")
                return
            if quote not in section:
                errors.append(f"SC009 {where}: quote not found verbatim in section '{section_id}'")
                return
            if in_node:
                node = entry.get("node")
                number = node.rsplit("B", 1)[1] if isinstance(node, str) else None
                if number not in passages:
                    errors.append(f"SC010 {where}: node {node!r} is not realized in {profile}")
                elif quote not in passages[number]:
                    errors.append(f"SC010 {where}: quote not found inside the passage of {node}")
            if entry.get("ink_words") != words(quote):
                errors.append(f"SC006 {where}: ink_words {entry.get('ink_words')} != {words(quote)}")

        profile_uses_graph_support = any(
            isinstance(entry, dict) and "graph_support" in entry
            for key in ("questions", "unplanned_answers")
            for entry in (doc.get(key) or []))

        def check_graph_support(entry: dict, where: str, qid: str) -> None:
            # The baseline sidecar predates graph_support.  A sidecar that
            # carries the field on any entry of a profile must carry it on
            # every answered or partly answered entry of that profile.
            if "graph_support" not in entry:
                if profile_uses_graph_support and entry.get("status") != "not-answered":
                    errors.append(f"SC014 {where}: graph_support missing")
                return
            anchors = entry["graph_support"]
            if not isinstance(anchors, list) or not anchors:
                errors.append(f"SC014 {where}: graph_support must be a non-empty list")
                return
            allowed_graphs = {anchor.get("graph") for anchor in question_plans[qid].get("support", [])
                              if isinstance(anchor, dict)}
            for anchor in anchors:
                edge = anchor.get("edge") if isinstance(anchor, dict) else None
                if (not isinstance(anchor, dict) or set(anchor) != {"graph", "edge"} or
                        not isinstance(edge, dict) or set(edge) != {"from", "relation", "to"}):
                    errors.append(f"SC014 {where}: malformed graph_support edge")
                    continue
                if anchor["graph"] not in allowed_graphs:
                    errors.append(f"SC014 {where}: graph {anchor['graph']} is not in the plan support")
                    continue
                try:
                    _nodes, graph_edges = graph_inventory(anchor["graph"])
                except (ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"SC014 {where}: cannot resolve support graph: {exc}")
                    continue
                triple = (edge["from"], edge["relation"], edge["to"])
                if triple not in graph_edges:
                    errors.append(f"SC014 {where}: graph_support edge does not resolve: {triple}")

        listed = doc.get("questions")
        unplanned = doc.get("unplanned_answers", [])
        if not isinstance(listed, list) or not isinstance(unplanned, list):
            errors.append(f"SC011 {profile}: questions and unplanned_answers must be lists")
            listed, unplanned = (listed if isinstance(listed, list) else []), []
        answered_ids: set[str] = set()
        for entry in listed:
            qid = entry.get("question") if isinstance(entry, dict) else None
            if qid not in decisions:
                errors.append(f"SC011 {profile}: unknown question {qid!r}")
                continue
            where = f"{profile} {qid}"
            answered_ids.add(qid)
            status = entry.get("status")
            if status not in ANSWER_STATUSES:
                errors.append(f"SC004 {where}: unknown status {status!r}")
            else:
                report[profile]["questions"][status] += 1
            if decisions[qid].get(profile, {}).get("action") != "anticipate":
                errors.append(f"SC011 {where}: decision is not anticipate; use unplanned_answers")
            if status == "answered" and entry.get("missing"):
                errors.append(f"SC012 {where}: answered entries must not carry 'missing'")
            if status != "answered" and not entry.get("missing"):
                errors.append(f"SC012 {where}: status {status} needs a 'missing' statement")
            check_quote(entry, where, in_node=True)
            check_graph_support(entry, where, qid)
        for qid, decision in decisions.items():
            if decision.get(profile, {}).get("action") == "anticipate" and qid not in answered_ids:
                errors.append(f"SC013 {profile} {qid}: anticipate decision has no question entry")
        for entry in unplanned:
            qid = entry.get("question") if isinstance(entry, dict) else None
            if qid not in decisions:
                errors.append(f"SC011 {profile}: unknown unplanned question {qid!r}")
                continue
            where = f"{profile} {qid} (unplanned)"
            if decisions[qid].get(profile, {}).get("action") == "anticipate":
                errors.append(f"SC011 {where}: decision is anticipate; list it under questions")
            status = entry.get("status")
            if status not in {"answered", "partly-answered"}:
                errors.append(f"SC004 {where}: unplanned status must be answered or partly-answered")
            if status != "answered" and not entry.get("missing"):
                errors.append(f"SC012 {where}: status {status} needs a 'missing' statement")
            check_quote(entry, where, in_node=True)
            check_graph_support(entry, where, qid)
            report[profile]["unplanned"] += 1
    return report, errors


def print_summary(report: dict, errors: list[str]) -> None:
    for error in errors:
        print(error)
    if not report:
        return
    for profile, row in report.items():
        nodes = row["nodes"]
        q = row["questions"]
        print(f"{profile}: nodes realized={nodes['realized']} merged={nodes['merged']} "
              f"absent={nodes['absent']}; questions answered={q['answered']} "
              f"partly={q['partly-answered']} not={q['not-answered']}; "
              f"unplanned answers={row['unplanned']}")
        print("  planned vs actual percent: " + ", ".join(
            f"{node.split(':')[1]} {planned}->{actual}" for node, planned, actual in row["node_rows"]))
        if "words" in row:
            count = row["words"]
            print(f"  words: {count['actual']} (target {count['target']}, max {count['max']})")
    if not errors:
        print("valid telling sidecar")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path, nargs="?", default=HERE / "realizations.sidecar.json")
    parser.add_argument("--plan", type=Path, help="publication plan defining the sidecar profile")
    args = parser.parse_args()
    report, errors = validate(args.sidecar, publication_plan=args.plan)
    print_summary(report, errors)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
