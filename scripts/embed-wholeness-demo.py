#!/usr/bin/env python3
"""Embed mission-wholeness.edn into the standalone mission hypergraph demo.

The demo opens via file://, so it cannot reliably fetch the EDN at runtime.
Run this script after futon6/data/mission-wholeness.edn is re-emitted.
"""

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDN = ROOT / "data" / "mission-wholeness.edn"
DEFAULT_HTML = ROOT / "data" / "mission-hypergraph-demo.html"


def parse_mission_wholeness(text):
    """Parse the current simple mission-wholeness EDN map into JSON rows."""
    missions = {}
    for block in re.findall(r"\{[^{}]*:mission[^{}]*\}", text, flags=re.S):
        mission_match = re.search(r':mission\s+"([^"]+)"', block)
        if not mission_match:
            continue
        mission = mission_match.group(1)
        row = {}
        class_match = re.search(r":class\s+:([A-Za-z0-9_-]+)", block)
        if class_match:
            row["class"] = class_match.group(1).upper()
        for key in ("L", "T", "H", "C", "N", "n"):
            value_match = re.search(r":" + key + r"\s+([-0-9.]+)", block)
            if value_match:
                value = float(value_match.group(1))
                row[key] = int(value) if key in {"N", "n"} else value
        pathology_match = re.search(r":pathology\s+\[([^\]]*)\]", block)
        row["pathology"] = (
            re.findall(r'"([^"]+)"', pathology_match.group(1))
            if pathology_match
            else []
        )
        missions[mission] = row
    if not missions:
        raise ValueError("no :mission rows parsed from mission-wholeness EDN")
    return missions


def embed(html_text, missions):
    replacement = (
        "const MISSION_WHOLENESS = "
        + json.dumps(missions, separators=(",", ":"), sort_keys=False)
        + ";"
    )
    updated, count = re.subn(
        r"const MISSION_WHOLENESS = \{.*?\};\nconst MANDALA_EXEMPLARS =",
        replacement + "\nconst MANDALA_EXEMPLARS =",
        html_text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise ValueError("could not find MISSION_WHOLENESS assignment in demo HTML")
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edn", type=Path, default=DEFAULT_EDN)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()

    missions = parse_mission_wholeness(args.edn.read_text())
    args.html.write_text(embed(args.html.read_text(), missions))
    classes = {}
    for row in missions.values():
        classes[row.get("class", "UNCLASSED")] = classes.get(row.get("class", "UNCLASSED"), 0) + 1
    print(
        f"embedded {len(missions)} mission wholeness rows into {args.html} "
        f"classes={dict(sorted(classes.items()))}"
    )


if __name__ == "__main__":
    main()
