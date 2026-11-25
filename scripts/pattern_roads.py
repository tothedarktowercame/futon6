#!/usr/bin/env python3
"""pattern_roads.py — constructive road table for the Futon City archipelago.

One row per (mission pair, pattern) from futon1b mission-scope/pattern
hyperedges. Replaces the text co-mention channel of
data/mission-carpet-roads.json (which drops pattern identity and counts
incidental mentions) for the tide prototypes, per claude-17's archipelago
brief §4/§8/§9:

  * Only :pattern/state "linked" scopes make roads; other states are tallied
    in excluded_states, never silently dropped.
  * Both ends of a road must be positioned missions; pattern scopes on
    unpositioned missions land in patterns[ident].off_map_missions.
  * A mission with two scopes for the same pattern contributes the
    earliest-dated scope to that end.
  * "constructed" = later of the two ends' scope dates (when the road
    existed); constructed_source is the WEAKER end
    (undated < mission-fallback < blame). Unknown stays null, never 0.

Output: data/pattern-roads.json.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futon6_config as config

ROOT = config.ROOT
POS_EMBED = ROOT / "data" / "mission-carpet-pos-embed.json"
TEXT_ROADS = ROOT / "data" / "mission-carpet-roads.json"
DATING = ROOT / "data" / "mission-dating.json"
OUT = ROOT / "data" / "pattern-roads.json"

SUBSTRATE = (
    os.environ.get("FUTON_SUBSTRATE_URL")
    or os.environ.get("FUTON1A_URL")
    or "http://127.0.0.1:7073"
).rstrip("/")
BASE = f"{SUBSTRATE}/api/alpha/hyperedges"
LIMIT = 1000
FETCH_TIMEOUT_S = int(os.environ.get("PATTERN_ROADS_TIMEOUT_S", "180"))

# Weakest-wins ordering for constructed_source.
SOURCE_RANK = {"undated": 0, "mission-fallback": 1, "blame": 2}


def fetch_pattern_scopes():
    """All mission-scope/pattern hyperedges, paged like
    mission_dating.fetch_scopes."""
    out = []
    after = None
    seen = set()
    while True:
        params = {"type": "mission-scope/pattern", "limit": LIMIT,
                  "fields": "hx/props"}
        if after:
            params["after"] = after
        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{BASE}?{q}",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
            page = json.load(r)
        rows = page.get("hyperedges", [])
        for h in rows:
            props = h.get("hx/props") or {}
            sid = props.get("scope/id")
            ident = props.get("pattern/ident")
            if sid and ident:
                out.append({
                    "scope/id": sid,
                    "mission": props.get("mission") or sid.split("/")[0],
                    "ident": ident,
                    "ref": props.get("pattern/ref"),
                    "state": props.get("pattern/state") or "",
                })
        cursor = page.get("next-cursor")
        if not cursor:
            break
        if not rows or cursor in seen:
            raise SystemExit("mission-scope/pattern: stalled cursor page")
        seen.add(cursor)
        after = cursor
    return out


def _dated_key(entry):
    """Sort key picking the earliest-dated scope; undated sorts last."""
    d = entry.get("dated")
    return (d is None, d or "")


def build_roads(scopes, positioned, dating):
    """scopes: fetch_pattern_scopes rows; positioned: set of mission ids;
    dating: {scope/id: {"dated", "date_source"}}. Returns the output dict
    minus "generated"/"source"/text-pair fields."""
    excluded = Counter()
    # pattern -> mission -> [linked scope entries for that end]
    by_pattern = defaultdict(lambda: defaultdict(list))
    refs = {}
    for s in scopes:
        refs.setdefault(s["ident"], s["ref"])
        if s["state"] != "linked":
            excluded[s["state"] or "missing"] += 1
            continue
        d = dating.get(s["scope/id"], {})
        by_pattern[s["ident"]][s["mission"]].append({
            "scope": s["scope/id"],
            "dated": d.get("dated"),
            "date_source": d.get("date_source", "undated"),
        })

    patterns = {}
    roads = []
    for ident in sorted(by_pattern):
        ends = {}
        for mission, entries in by_pattern[ident].items():
            entries.sort(key=_dated_key)
            ends[mission] = entries[0]  # earliest-dated scope per end
        on_map = sorted(m for m in ends if m in positioned)
        off_map = sorted(m for m in ends if m not in positioned)
        patterns[ident] = {
            "ref": refs.get(ident),
            "missions": on_map,
            "n_missions": len(on_map),
            "off_map_missions": off_map,
        }
        for a, b in combinations(on_map, 2):
            ea, eb = ends[a], ends[b]
            dates = [e["dated"] for e in (ea, eb)]
            constructed = max(dates) if all(dates) else None
            src = min((ea["date_source"], eb["date_source"]),
                      key=lambda s: SOURCE_RANK.get(s, 0))
            roads.append({"a": a, "b": b, "pattern": ident,
                          "a_scope": ea["scope"], "b_scope": eb["scope"],
                          "constructed": constructed,
                          "constructed_source": src})
    roads.sort(key=lambda r: (r["a"], r["b"], r["pattern"]))
    return {"patterns": patterns, "roads": roads,
            "excluded_states": dict(sorted(excluded.items()))}


def pair_diffs(roads, text_roads):
    """text_only / scope_only pair lists vs the old text co-mention roads
    ([[a, b, w], ...])."""
    scope_pairs = {(r["a"], r["b"]) for r in roads}
    text_pairs = set()
    for row in text_roads:
        a, b = sorted(row[:2])
        text_pairs.add((a, b))
    return {
        "text_only_pairs": [list(p) for p in sorted(text_pairs - scope_pairs)],
        "scope_only_pairs": [list(p) for p in sorted(scope_pairs - text_pairs)],
    }


def main():
    positioned = set(json.loads(POS_EMBED.read_text()))
    dating = {}
    if DATING.exists():
        raw = json.loads(DATING.read_text())
        dating = raw.get("scopes", {})
    else:
        print(f"warning: {DATING} absent; all scope dates unknown",
              file=sys.stderr)
    scopes = fetch_pattern_scopes()
    table = build_roads(scopes, positioned, dating)
    text_roads = json.loads(TEXT_ROADS.read_text()) if TEXT_ROADS.exists() else []
    table.update(pair_diffs(table["roads"], text_roads))
    table.update({
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "futon1b mission-scope/pattern",
    })
    # deterministic top-level key order for readability
    ordered = {k: table[k] for k in
               ("generated", "source", "patterns", "roads",
                "excluded_states", "text_only_pairs", "scope_only_pairs")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ordered, indent=1, sort_keys=False) + "\n")
    pairs = {(r["a"], r["b"]) for r in table["roads"]}
    print(f"wrote {OUT}: {len(table['patterns'])} patterns, "
          f"{len(table['roads'])} roads over {len(pairs)} pairs, "
          f"excluded {table['excluded_states']}, "
          f"text_only {len(table['text_only_pairs'])}, "
          f"scope_only {len(table['scope_only_pairs'])}")


if __name__ == "__main__":
    main()
