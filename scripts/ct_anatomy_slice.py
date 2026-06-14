#!/usr/bin/env python3
"""ct_anatomy_slice.py — extract per-paper scope slices from the CT handoff.

scopes.json is a 19G JSON array, line-structured (one paper record per line).
This streams it once and writes one slice file per audited sample paper, plus
paper metadata from entities.json. Slices land under the handoff dir (derived
data, never committed); the showcase builder reads them.
"""
import json
import sys
from pathlib import Path

HANDOFF = Path("/home/joe/code/storage/mark2/ct-handoff/output")
SLICES = HANDOFF.parent / "ct-anatomy-slices"
SLICES.mkdir(exist_ok=True)

import sys
if len(sys.argv) > 1 and sys.argv[1] == "--all":
    wanted = None  # slice every paper
    print("slicing ALL papers from scopes.json", flush=True)
else:
    audit = json.load(open(HANDOFF / "audit-summary.json"))
    wanted = {p["entity_id"] for p in audit["papers"]}
    print(f"slicing {len(wanted)} audited papers from scopes.json", flush=True)

found = {}
with open(HANDOFF / "scopes.json") as f:
    for line in f:
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        # cheap pre-filter before full parse
        eid_at = line.find('"entity_id": "')
        if eid_at < 0:
            continue
        eid = line[eid_at + 14 : line.find('"', eid_at + 14)]
        if wanted is not None and eid not in wanted:
            continue
        rec = json.loads(line)
        out = SLICES / (eid.replace("/", "_") + ".json")
        out.write_text(json.dumps(rec))
        found[eid] = rec.get("count", len(rec.get("scopes", [])))
        if wanted is not None or len(found) % 500 == 0:
            print(f"  {len(found)} sliced (latest {eid}: {found[eid]} scopes)", flush=True)
        if wanted is not None and len(found) == len(wanted):
            break

# paper metadata (titles) for the index
meta = {}
try:
    ents = json.load(open(HANDOFF / "entities.json"))
    ent_list = ents if isinstance(ents, list) else ents.get("entities", [])
    for e in ent_list:
        eid = e.get("id") or e.get("entity_id")
        if wanted is None or eid in wanted:
            meta[eid] = {k: e.get(k) for k in ("title", "name", "tags", "created") if e.get(k)}
except Exception as exc:
    print(f"entities.json metadata pass failed: {exc}", flush=True)
(SLICES / "_meta.json").write_text(json.dumps(meta))
print(f"DONE: {len(found)}/{len(wanted) if wanted is not None else 'all'} sliced, {len(meta)} with metadata", flush=True)
