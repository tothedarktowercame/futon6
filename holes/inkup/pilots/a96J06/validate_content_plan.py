#!/usr/bin/env python3
"""Validate the shared a96J06 content presentation plan."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REASONING = ROOT / "holes/inkup/reasoning"
sys.path.insert(0, str(REASONING))
import validate_question_alignments as question_gate  # noqa: E402

ACTIONS = {"anticipate", "defer", "omit"}


def load_run_records(path: Path | None) -> dict | None:
    """Read a run's question-alignments.jsonl, keyed by candidate id."""
    if path is None:
        return None
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["candidate_id"]] = record
    return records


def validate(path: Path, run_alignments: Path | None = None) -> tuple[dict, list[str]]:
    errors = []
    try:
        run_records = load_run_records(run_alignments)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        return {}, [f"CP000 cannot read run alignments: {exc}"]
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"CP000 cannot read content plan: {exc}"]
    if plan.get("schema") != "inkup.content-plan/v0" or plan.get("problem") != "a96J06":
        errors.append("CP000 wrong content-plan identity")
    alignment_path = ROOT / plan.get("alignment", "")
    try:
        alignment = json.loads(alignment_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return plan, errors + [f"CP000 cannot read alignment: {exc}"]
    nodes = {node["id"]: node for node in alignment["nodes"]}
    profiles = {profile["id"] for profile in alignment["profiles"]}

    mandatory = plan.get("mandatory_nodes")
    if not isinstance(mandatory, list):
        errors.append("CP001 mandatory_nodes is not a list")
        mandatory = []
    seen = set()
    for item in mandatory:
        if not isinstance(item, dict) or set(item) != {"node", "depends_on"}:
            errors.append("CP001 malformed mandatory-node reference")
            continue
        node_id = item["node"]
        if node_id not in nodes or item["depends_on"] != nodes.get(node_id, {}).get("depends_on"):
            errors.append(f"CP001 mandatory node or dependencies do not match alignment: {node_id}")
        seen.add(node_id)
    if seen != set(nodes):
        errors.append(f"CP001 mandatory-node coverage differs from alignment")

    questions = plan.get("questions")
    if not isinstance(questions, list):
        return plan, errors + ["CP000 questions is not a list"]
    ids = [question.get("id") for question in questions if isinstance(question, dict)]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        errors.append("CP000 question ids are empty or duplicated")

    for question in questions:
        question_id = question.get("id", "<missing>")
        if question.get("blueprint_node") not in nodes:
            errors.append(f"CP002 {question_id}: unknown blueprint node {question.get('blueprint_node')}")

        evidence = question.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"CP006 {question_id}: no evidence entries")
            evidence = []
        for item in evidence:
            candidate_id = item.get("candidate_id") if isinstance(item, dict) else None
            if candidate_id is None or run_records is None:
                continue
            record = run_records.get(candidate_id)
            if record is None:
                errors.append(f"CP006 {question_id}: evidence {candidate_id} is not in the run")
            elif (record.get("mapping_state") != "accepted" or
                  record.get("question_function") != question.get("question_function")):
                errors.append(
                    f"CP006 {question_id}: evidence {candidate_id} is not an accepted "
                    f"{question.get('question_function')} mapping in the run")

        decisions = question.get("decisions")
        if not isinstance(decisions, dict) or set(decisions) != profiles:
            errors.append(f"CP003 {question_id}: decisions must cover exactly {sorted(profiles)}")
            decisions = decisions if isinstance(decisions, dict) else {}
        for profile, decision in decisions.items():
            if (profile not in profiles or not isinstance(decision, dict) or
                    set(decision) != {"action", "reason"} or
                    decision.get("action") not in ACTIONS or not decision.get("reason")):
                errors.append(f"CP003 {question_id}: malformed decision for {profile}")

        if question.get("support_state") == "contested":
            if not question.get("blocked_on"):
                errors.append(f"CP005 {question_id}: contested question lacks blocked_on")
            if any(decision.get("action") == "anticipate" for decision in decisions.values()
                   if isinstance(decision, dict)):
                errors.append(f"CP005 {question_id}: contested question cannot be anticipated")

        gate_record = {
            "candidate_id": f"seed:content-plan:{question_id}",
            "source_wording": question.get("analogue") or "",
            "question_function": question.get("question_function"),
            "requested_relation": question.get("requested_relation"),
            "mapping_state": "accepted",
            "proposers": ["content-plan"],
            "reviewer": "content-plan-validator",
            "support": question.get("support"),
            "support_state": question.get("support_state"),
            "disposition": "anticipate" if question.get("support_state") == "supported" else "defer",
            "note": "Synthetic record used to invoke the shared question-support gate."
        }
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8") as stream:
            stream.write(json.dumps(gate_record, ensure_ascii=False) + "\n")
            stream.flush()
            _records, gate_errors = question_gate.validate(Path(stream.name))
        for _line, message in gate_errors:
            errors.append(f"CP004 {question_id}: support gate rejected record: {message}")

        if any(decision.get("action") == "anticipate" for decision in decisions.values()
               if isinstance(decision, dict)) and question.get("support_state") != "supported":
            # Contested cases already receive the more specific CP005 above.
            if question.get("support_state") != "contested":
                errors.append(f"CP004 {question_id}: anticipate requires supported state")
    return plan, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path, nargs="?", default=HERE / "content-plan.json")
    parser.add_argument("--run-alignments", type=Path,
                        help="the evidence run's question-alignments.jsonl; checks each cited candidate")
    args = parser.parse_args()
    plan, errors = validate(args.plan, args.run_alignments)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    questions = plan["questions"]
    print(f"valid content plan: {len(plan['mandatory_nodes'])} mandatory nodes, {len(questions)} question entries")
    print("support_state: " + ", ".join(
        f"{key}={value}" for key, value in sorted(Counter(q["support_state"] for q in questions).items())))
    print(f"blocked reasoning graphs: {sum('blocked_on' in question for question in questions)}")


if __name__ == "__main__":
    main()
