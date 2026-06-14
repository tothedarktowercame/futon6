#!/usr/bin/env python3
"""mission_efe_scope_dump.py — the REPRODUCIBLE slim input for mission_efe_field.py.

Replaces the transient /tmp/scopes.json (last made 2026-06-09, by hand, 11 binder
types, pre-anatomy). Three upgrades, per the 2026-06-12 redraw (task: redraw the
EFE landscape in light of the improved anatomy-of-a-Mission work):

  1. ALL binder types — including the anatomy additions verify-gate, certificate,
     and plain-argument that the old dump never saw.
  2. Anatomy enrichment per scope, joined from the detector trees
     (data/mission-scope-trees/<M>.json): the Skolem grade where computable
     (a scope whose binder introduces names its body never uses is "vacuous" —
     quantification without binding), and the certificate verdict for
     certificate scopes (green/red).
  3. Committed + rerunnable: substrate-2 (7071) is the source of truth, bounded
     per-binder queries only (never unbounded — that can wedge the JVM).

Output: data/efe-scopes.json — [{m, binder, det, phase, skolem, verdict}, ...]
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/home/joe/code/futon6")
OUT = ROOT / "data" / "efe-scopes.json"
TREES = ROOT / "data" / "mission-scope-trees"
BASE = "http://localhost:7071/api/alpha/hyperedges"
BINDERS = [
    "eightfold-phase", "loose-section", "mission-scope-in", "mission-scope-out",
    "map-item", "source-material", "relates-to", "capability-scope",
    "pattern", "psr", "pur",
    # anatomy additions (2026-06-10/11): typed gates and their outcomes
    "verify-gate", "certificate", "plain-argument",
]
LIMIT = 8000


def fetch(binder):
    q = urllib.parse.urlencode({"type": f"mission-scope/{binder}", "limit": LIMIT})
    req = urllib.request.Request(f"{BASE}?{q}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("hyperedges", [])


def tree_index():
    """scope-id -> anatomy facts from the detector trees.

    verdict: a certificate scope's {role: verdict, state: pass|fail} end.
    vacuous: an environment scope with ZERO concept ends — the Skolem
      audit's "scope without named entities inside" suspect class.
    phase: the environment end's phase tag."""
    idx = {}
    for tf in sorted(TREES.glob("*.json")):
        try:
            data = json.loads(tf.read_text())
        except Exception:
            continue
        for hx in data.get("scope-hyperedges", []):
            sid = hx.get("scope-id")
            if not sid:
                continue
            ends = hx.get("ends", [])
            entry = {}
            concepts = [e for e in ends if e.get("role") == "concept"]
            for e in ends:
                if e.get("role") == "verdict":
                    entry["verdict"] = e.get("state")
                if e.get("role") == "environment" and e.get("phase"):
                    entry["phase"] = e["phase"]
            if "environment" in {e.get("role") for e in ends} and not concepts:
                entry["vacuous"] = True
            if entry:
                idx[sid] = entry
    return idx


def main():
    anatomy = tree_index()
    out = []
    counts = {}
    for binder in BINDERS:
        hxs = fetch(binder)
        counts[binder] = len(hxs)
        for h in hxs:
            props = h.get("hx/props") or {}
            sid = props.get("scope/id")
            if not sid:
                continue
            # mission stem from the scope-id prefix (the diffsub convention)
            mission = sid.split("/")[0]
            state = props.get("anchor/state", "")
            det = (state == "detached") or (props.get("scope/parent-state") == "detached")
            joined = anatomy.get(sid, {})
            out.append({
                "m": mission[2:] if mission.startswith("M-") else mission,
                "binder": binder,
                "det": bool(det),
                "phase": joined.get("phase"),
                "vacuous": joined.get("vacuous", False),
                "verdict": joined.get("verdict"),
            })
    OUT.write_text(json.dumps(out))
    print(f"{len(out)} scopes -> {OUT}")
    for b, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {b}: {n}")


if __name__ == "__main__":
    main()
