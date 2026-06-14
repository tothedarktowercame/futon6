#!/usr/bin/env python3
"""diffsub_scope_dump.py — scope-grain v2 data foundation (M-differentiable-substrate).

Dump all substrate-2 mission-scope hyperedges with their REAL scope-ids + verbatim
passages + open/detached state, so the gradient producer can run at scope grain (the
real epistemic atoms) instead of mission aggregates — and emit :have/:want as real
substrate-2 scope-ids that join claude-4's rollout reachability.

Reads 7071 with BOUNDED per-binder queries (&limit=; never unbounded type= — that can
wedge the JVM). Output: futon6/data/diffsub-scopes.json
"""
import json, urllib.request, urllib.parse, sys
from pathlib import Path

OUT = Path("/home/joe/code/futon6/data/diffsub-scopes.json")
BASE = "http://localhost:7071/api/alpha/hyperedges"
BINDERS = ["eightfold-phase", "loose-section", "mission-scope-in", "mission-scope-out",
           "map-item", "source-material", "relates-to", "capability-scope",
           "pattern", "psr", "pur"]
LIMIT = 8000  # bound per type — sum over types ≈ the full corpus, never unbounded


def fetch(binder):
    q = urllib.parse.urlencode({"type": f"mission-scope/{binder}", "limit": LIMIT})
    req = urllib.request.Request(f"{BASE}?{q}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("hyperedges", [])


def main():
    scopes, per_binder, det_count, with_passage = [], {}, 0, 0
    for b in BINDERS:
        try:
            edges = fetch(b)
        except Exception as e:
            print(f"  {b}: FETCH FAILED {e}", file=sys.stderr)
            continue
        per_binder[b] = len(edges)
        for e in edges:
            props = e.get("hx/props", {}) or {}
            content = e.get("hx/content", {}) or {}
            sid = props.get("scope/id")
            if not sid:
                continue
            state = props.get("anchor/state", "")
            det = (state == "detached") or (props.get("scope/parent-state") == "detached")
            passage = content.get("anchor/passage", "") or ""
            # role-tagged endpoints minus the mission entity + the scope environment itself
            ends = {d.get("role"): d.get("entity-id") for d in e.get("hx/ends", []) if d.get("role")}
            mission = sid.split("/")[0]
            scopes.append({
                "scope_id": sid,
                "mission": mission,
                "binder": b,
                "det": bool(det),
                "state": state,
                "passage": passage[:400],
                "capability": ends.get("capability"),
                "concepts": [v for r, v in ends.items() if r == "concept"]
                            or [v for d in e.get("hx/ends", []) if d.get("role") == "concept" for v in [d.get("entity-id")]],
                "mission_node": ends.get("entity"),
            })
            if det:
                det_count += 1
            if passage:
                with_passage += 1
    OUT.write_text(json.dumps(scopes, indent=0))
    print(f"wrote {OUT}")
    print(f"total scopes: {len(scopes)} | detached/open: {det_count} | with-passage: {with_passage}")
    print("per-binder:", json.dumps(per_binder))
    miss = len({s['mission'] for s in scopes})
    print(f"distinct missions: {miss}")


if __name__ == "__main__":
    main()
