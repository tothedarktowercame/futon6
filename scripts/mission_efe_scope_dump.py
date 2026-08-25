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
import os
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/home/joe/code/futon6")
OUT = ROOT / "data" / "efe-scopes.json"
TREES = ROOT / "data" / "mission-scope-trees"
# Substrate URL resolution, same precedence as futon3c.watcher.multi: the
# canonical FUTON_SUBSTRATE_URL wins, FUTON1A_URL is the compatibility input,
# and the default follows the 2026-07-12 futon1a(:7071) -> futon1b(:7073)
# switchover.  The old hardcoded :7071 made this a nightly connection-refused
# that aborted daily_reembed.sh under `set -e` before the EFE field regen.
SUBSTRATE = (
    os.environ.get("FUTON_SUBSTRATE_URL")
    or os.environ.get("FUTON1A_URL")
    or "http://127.0.0.1:7073"
).rstrip("/")
BASE = f"{SUBSTRATE}/api/alpha/hyperedges"
BINDERS = [
    "eightfold-phase", "loose-section", "mission-scope-in", "mission-scope-out",
    "map-item", "source-material", "relates-to", "capability-scope",
    "pattern", "psr", "pur",
    # anatomy additions (2026-06-10/11): typed gates and their outcomes
    "verify-gate", "certificate", "plain-argument",
]
# futon1b's hyperedge window caps pages at 1000 and advances with
# `next-cursor`. A larger limit is a hard 400; a single accepted page would
# silently truncate loose-section. fetch() therefore consumes the complete
# cursor chain and checks the server's exact count.
LIMIT = 1000
FETCH_TIMEOUT_S = int(os.environ.get("EFE_SCOPE_TIMEOUT_S", "180"))

# The dump is an overwrite, and the mission-scope surface in futon1b is only
# partly populated while the watcher scope lane is dark.  Refuse to replace a
# healthy dump with a drastically smaller one, in the same spirit as
# refresh_pattern_attestation.sh's "refusing to overwrite" guard: a shrunken
# EFE landscape must be an operator decision, not a silent nightly regression.
SHRINK_FLOOR = 0.75


def fetch(binder):
    hxs = []
    after = None
    expected = None
    seen_cursors = set()
    while True:
        params = {"type": f"mission-scope/{binder}", "limit": LIMIT}
        if after:
            params["after"] = after
        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{BASE}?{q}", headers={"Accept": "application/json"})
        # futon1b routes these through with-expensive-read!; the largest binder
        # (loose-section) can take tens of seconds per cold page.
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
            page = json.load(r)
        rows = page.get("hyperedges", [])
        if expected is None and page.get("count-exact?") is True:
            expected = page.get("count")
        hxs.extend(rows)
        cursor = page.get("next-cursor")
        if not cursor:
            break
        if not rows or cursor in seen_cursors:
            raise SystemExit(f"binder {binder!r} returned a stalled cursor page")
        seen_cursors.add(cursor)
        after = cursor
    if expected is None:
        raise SystemExit(f"binder {binder!r} did not report an exact total")
    if len(hxs) != expected:
        raise SystemExit(
            f"binder {binder!r} pagination mismatch: consumed {len(hxs)} of {expected} rows"
        )
    return hxs


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
    if OUT.exists():
        try:
            prev = len(json.loads(OUT.read_text()))
        except (ValueError, OSError):
            prev = 0
        allow_shrink = os.environ.get("EFE_SCOPE_ALLOW_SHRINK", "").lower() in {
            "1", "true", "yes", "on",
        }
        if prev and len(out) < prev * SHRINK_FLOOR and not allow_shrink:
            raise SystemExit(
                f"refusing to overwrite {OUT.name}: {len(out)} scopes vs {prev} "
                f"previously ({len(out) / prev:.0%}). The substrate is under-"
                "populated for this surface — check that the mission-scope "
                "reingest lane has run (FUTON3C_WATCHER_SCOPE_LANE / "
                "futon3c/scripts/mission-scope-reingest.sh) before accepting a "
                "smaller landscape. Override with EFE_SCOPE_ALLOW_SHRINK=1."
            )
    OUT.write_text(json.dumps(out))
    print(f"{len(out)} scopes -> {OUT}")
    for b, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {b}: {n}")


if __name__ == "__main__":
    main()
