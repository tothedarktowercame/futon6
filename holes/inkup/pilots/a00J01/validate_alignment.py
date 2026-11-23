#!/usr/bin/env python3
"""Validate the a00J01 Blueprint alignment without external packages.

Checks the alignment against the schema beside the a96J06 pilot (with the
problem id and node-id pattern read for this problem), then against its
sources: files exist, hashes match, dependencies exist and are acyclic, and
every Lean anchor names identifiers that occur in Main.lean.  No realization,
ink or grammar file is required yet.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBLEM = "a00J01"
SCHEMA = HERE.parent / "a96J06" / "alignment.schema.json"
PROFILE_IDS = ["expert-reference", "teaching", "formal-audit"]

# Source paths are written "<root>:<relative path>" so that the record holds no
# host path.  apm-lean is the sibling checkout of this repo unless
# INKUP_APM_LEAN_ROOT says otherwise; storage is the storage home (run output,
# never in a repo), ~/storage unless INKUP_STORAGE_ROOT says otherwise.
SOURCE_ROOTS = {
    "apm-lean": Path(os.environ.get("INKUP_APM_LEAN_ROOT", HERE.parents[4] / "apm-lean")),
    "storage": Path(os.environ.get("INKUP_STORAGE_ROOT", Path.home() / "storage")),
}
# The schema is written for a96J06 only (``problem`` const and node-id pattern);
# it is applied here with that problem id replaced, not edited.
SCHEMA_PROBLEM = "a96J06"


def resolve_source(spec: str, roots: dict | None = None) -> Path:
    roots = roots or SOURCE_ROOTS
    root, sep, rel = spec.partition(":")
    if not sep or root not in roots or rel.startswith("/"):
        raise ValueError(f"source path must be <root>:<relative path> with root in {sorted(roots)}: {spec}")
    return roots[root] / rel


# ---- schema subset (draft 2020-12 keywords used by alignment.schema.json) ----

def check_schema(instance, schema: dict, where: str, errors: list[str]) -> None:
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        check_schema(instance, SCHEMA_DOC["$defs"][name], where, errors)
        return
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{where}: must equal {schema['const']!r}")
        return
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{where}: {instance!r} not in {schema['enum']}")
        return
    kind = schema.get("type")
    if isinstance(kind, list):
        if instance is not None:
            kinds = [k for k in kind if k != "null"]
            check_schema(instance, {**schema, "type": kinds[0] if kinds else "string"}, where, errors)
        return
    if kind == "object":
        if not isinstance(instance, dict):
            errors.append(f"{where}: must be an object")
            return
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{where}: missing required field {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(properties))
            if extra:
                errors.append(f"{where}: unexpected field(s) {extra}")
        elif isinstance(schema.get("additionalProperties"), dict):
            for key, value in instance.items():
                check_schema(value, schema["additionalProperties"], f"{where}.{key}", errors)
        for key, value in instance.items():
            if key in properties:
                check_schema(value, properties[key], f"{where}.{key}", errors)
        return
    if kind == "array":
        if not isinstance(instance, list):
            errors.append(f"{where}: must be an array")
            return
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{where}: needs at least {schema['minItems']} item(s)")
        if schema.get("uniqueItems") and len(instance) != len({json.dumps(i, sort_keys=True) for i in instance}):
            errors.append(f"{where}: items must be unique")
        for index, item in enumerate(instance):
            check_schema(item, schema.get("items", {}), f"{where}[{index}]", errors)
        return
    if kind == "string":
        if not isinstance(instance, str):
            errors.append(f"{where}: must be a string")
            return
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{where}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{where}: does not match {schema['pattern']}")
        return
    if kind and kind != "string":
        errors.append(f"{where}: unsupported schema type {kind}")


# ---- anchors -----------------------------------------------------------------

def lean_identifiers(ref: str) -> list[str]:
    """``lemma upper_bound; lower_bound: hβ, s`` -> [upper_bound, lower_bound, hβ, s]."""
    names = []
    for token in ref.split(";"):
        head, sep, tail = token.strip().partition(": ")
        words = head.split()
        names.append(words[-1])
        if sep:
            names.extend(name.strip() for name in tail.split(",") if name.strip())
    return names


def validate(alignment: Path, schema_path: Path = SCHEMA,
             roots: dict | None = None) -> tuple[dict, list[str]]:
    global SCHEMA_DOC
    errors: list[str] = []
    roots = roots or SOURCE_ROOTS
    try:
        doc = json.loads(alignment.read_text())
        SCHEMA_DOC = json.loads(schema_path.read_text())
    except (OSError, ValueError) as exc:
        return {}, [f"AL000 cannot read alignment or schema: {exc}"]
    SCHEMA_DOC = json.loads(json.dumps(SCHEMA_DOC).replace(SCHEMA_PROBLEM, PROBLEM))

    check_schema(doc, SCHEMA_DOC, "alignment", errors)
    if doc.get("schema") != "inkup.blueprint-lineage/v0" or doc.get("problem") != PROBLEM:
        errors.append(f"AL000 wrong pilot identity: expected {PROBLEM}")
    if [profile.get("id") for profile in doc.get("profiles", [])] != PROFILE_IDS:
        errors.append(f"AL001 profiles must be exactly {PROFILE_IDS} in that order")

    texts: dict[str, str] = {}
    for name, source in doc.get("sources", {}).items():
        try:
            path = resolve_source(source.get("path", ""), roots)
        except ValueError as exc:
            errors.append(f"AL002 {name}: {exc}")
            continue
        if not path.is_file():
            errors.append(f"AL002 {name}: missing source {source.get('path')}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source.get("sha256"):
            errors.append(f"AL003 {name}: source hash drifted: now {digest}")
            continue
        texts[name] = path.read_text()

    nodes = {node["id"]: node for node in doc.get("nodes", []) if isinstance(node, dict) and "id" in node}
    for node in nodes.values():
        node_id = node["id"]
        refs = list(node.get("depends_on", [])) + [node["parent"]]
        missing = sorted(set(refs) - nodes.keys() - {None})
        if missing:
            errors.append(f"AL004 {node_id}: depends on unknown node(s) {missing}")
        if node.get("parent") == node_id:
            errors.append(f"AL004 {node_id}: is its own parent")
        if not node.get("reader_obligations"):
            errors.append(f"AL005 {node_id}: no reader obligation")
        if not node.get("anchors"):
            errors.append(f"AL005 {node_id}: no anchor")
        for anchor in node.get("anchors", []):
            source, ref = anchor.get("source"), anchor.get("ref", "")
            if source not in doc.get("sources", {}):
                errors.append(f"AL006 {node_id}: unknown source {source!r}")
                continue
            if source not in texts:
                continue
            if source == "lean":
                absent = [name for name in lean_identifiers(ref) if not re.search(rf"(?<![\w.]){re.escape(name)}\b", texts[source])]
                if absent:
                    errors.append(f"AL007 {node_id}: Lean anchor names identifier(s) not in Main.lean: {absent}")
            elif source == "informal":
                if ref not in texts[source]:
                    errors.append(f"AL008 {node_id}: informal anchor phrase not found verbatim: {ref!r}")
            elif source == "proof-plan":
                for keyword in re.findall(r":[\w-]+", ref):
                    if keyword not in texts[source]:
                        errors.append(f"AL009 {node_id}: proof-plan has no {keyword}")
            elif source == "mark8-s3":
                for node_ref in re.findall(r":[ne]\d+", ref):
                    if f":id {node_ref} " not in texts[source]:
                        errors.append(f"AL010 {node_id}: mined graph has no {node_ref}")

    # Dependency edges must be acyclic. B0 depends on the closing node to state
    # that the root obligation is discharged; traversal still treats it as an ordinary edge.
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in done:
            return
        if node_id in visiting:
            errors.append(f"AL011 dependency cycle at {node_id}")
            return
        visiting.add(node_id)
        for dependency in nodes[node_id].get("depends_on", []):
            if dependency in nodes:
                visit(dependency)
        visiting.discard(node_id)
        done.add(node_id)

    for node_id in nodes:
        visit(node_id)
    root = f"{PROBLEM}:B0"
    if root not in nodes:
        errors.append(f"AL012 no goal node {root}")
    elif nodes[root].get("parent") is not None:
        errors.append(f"AL012 {root} must have parent null")
    else:
        depended = {d for node_id, n in nodes.items() if node_id != root for d in n.get("depends_on", [])}
        closers = sorted(set(nodes) - {root} - depended)
        if nodes[root].get("depends_on") != closers or len(closers) != 1:
            errors.append(f"AL012 {root} must depend on exactly the closing node {closers}, "
                          f"not {nodes[root].get('depends_on')}")
    for node_id, node in nodes.items():
        if node_id != root and node.get("parent") != root:
            errors.append(f"AL012 {node_id}: parent must be {root}")
    for profile in doc.get("profiles", []):
        if not profile.get("mandatory_nodes"):
            errors.append(f"AL013 profile {profile.get('id')}: no mandatory nodes")
        for node_id in profile.get("mandatory_nodes", []):
            if node_id not in nodes:
                errors.append(f"AL013 profile {profile.get('id')}: unknown mandatory node {node_id}")
    return doc, errors


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alignment", type=Path, nargs="?", default=HERE / "alignment.json")
    doc, errors = validate(parser.parse_args().alignment)
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)
    counts: dict[str, int] = {}
    for node in doc["nodes"]:
        for anchor in node["anchors"]:
            counts[anchor["review"]] = counts.get(anchor["review"], 0) + 1
    print(f"valid {PROBLEM} lineage: {len(doc['nodes'])} nodes, {len(doc['profiles'])} profiles")
    print("spine: " + "; ".join(f"{node['id'].split(':')[1]} <- {','.join(d.split(':')[1] for d in node['depends_on']) or '-'}"
                               for node in doc["nodes"]))
    print("anchor review states: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))


if __name__ == "__main__":
    main()
