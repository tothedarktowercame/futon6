#!/usr/bin/env python3
"""Validate the a96J06 Blueprint-lineage pilot without external packages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fail(message: str) -> None:
    raise SystemExit(message)


doc = json.loads((HERE / "alignment.json").read_text())
json.loads((HERE / "alignment.schema.json").read_text())

if doc.get("schema") != "inkup.blueprint-lineage/v0" or doc.get("problem") != "a96J06":
    fail("wrong pilot identity")

for name, source in doc["sources"].items():
    path = Path(source["path"])
    if not path.is_file():
        fail(f"{name}: missing source {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        fail(f"{name}: source hash drifted: {digest}")

nodes = {node["id"]: node for node in doc["nodes"]}
if len(nodes) != len(doc["nodes"]):
    fail("duplicate node id")

allowed_review = {"human-reviewed", "model-proposed", "rejected"}
for node in nodes.values():
    refs = list(node["depends_on"])
    if node["parent"] is not None:
        refs.append(node["parent"])
    missing = sorted(set(refs) - nodes.keys())
    if missing:
        fail(f"{node['id']}: unknown node reference(s) {missing}")
    if not node["reader_obligations"]:
        fail(f"{node['id']}: no reader obligation")
    for anchor in node["anchors"]:
        if anchor["source"] not in doc["sources"]:
            fail(f"{node['id']}: unknown source {anchor['source']}")
        if anchor["review"] not in allowed_review:
            fail(f"{node['id']}: unknown review state {anchor['review']}")

# Dependency edges must be acyclic. B0 depends on the closing node to state that
# the root obligation is discharged; traversal still treats it as an ordinary edge.
visiting: set[str] = set()
done: set[str] = set()


def visit(node_id: str) -> None:
    if node_id in done:
        return
    if node_id in visiting:
        fail(f"dependency cycle at {node_id}")
    visiting.add(node_id)
    for dependency in nodes[node_id]["depends_on"]:
        visit(dependency)
    visiting.remove(node_id)
    done.add(node_id)


for node_id in nodes:
    visit(node_id)

for profile in doc["profiles"]:
    missing = sorted(set(profile["mandatory_nodes"]) - nodes.keys())
    if missing:
        fail(f"profile {profile['id']}: unknown mandatory nodes {missing}")

realizations = (HERE / "realizations.md").read_text()
sections = {}
for match in re.finditer(r"^## (Expert reference|Teaching|Formal audit)\n(.*?)(?=^## |\Z)",
                         realizations, re.MULTILINE | re.DOTALL):
    sections[match.group(1)] = match.group(2)
section_for_profile = {
    "expert-reference": "Expert reference",
    "teaching": "Teaching",
    "formal-audit": "Formal audit",
}
for profile in doc["profiles"]:
    heading = section_for_profile[profile["id"]]
    if heading not in sections:
        fail(f"profile {profile['id']}: missing realization section")
    present = {f"a96J06:{tag}" for tag in re.findall(r"\*\*\[(B[0-9]+)\]\*\*", sections[heading])}
    missing = sorted(set(profile["mandatory_nodes"]) - present)
    if missing:
        fail(f"profile {profile['id']}: realization omits mandatory nodes {missing}")

counts = {state: 0 for state in allowed_review}
for node in nodes.values():
    for anchor in node["anchors"]:
        counts[anchor["review"]] += 1

print(f"valid a96J06 lineage: {len(nodes)} nodes, {len(doc['profiles'])} profiles")
print("anchor review states: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
