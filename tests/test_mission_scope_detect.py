from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mission_scope_detect.py"
SPEC = importlib.util.spec_from_file_location("mission_scope_detect", SCRIPT)
assert SPEC and SPEC.loader
scope_detect = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scope_detect
SPEC.loader.exec_module(scope_detect)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _by_binder(tree: dict, binder: str) -> list[dict]:
    return [s for s in tree["scope-hyperedges"] if s["binder-type"] == binder]


def test_loose_agency_style_sections_do_not_crash_and_bind_slots(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "M-agency-demo.md",
        """# Mission: Agency Demo

## Motivation
This mission improves futon agency evidence.

## Scope

### Scope In
- registry shape
- dispatch path

### Scope Out
- web polish

## Source Material
- futon3c/src/futon3c/agency/registry.clj
- POST /api/alpha/bell

## Dependencies
- Blocks M-war-machine-pilot
- Enables M-web-arxana-missions
""",
    )

    tree = scope_detect.detect_mission_scopes(
        path,
        kernel_terms=["futon", "agency", "evidence", "dispatch"],
        capabilities=set(),
    )

    assert tree["scope-count-by-binder-type"]["loose-section"] >= 4
    assert _by_binder(tree, "mission-scope-in")
    assert _by_binder(tree, "mission-scope-out")
    assert _by_binder(tree, "source-material")
    assert _by_binder(tree, "relates-to")
    concept_terms = {
        end["term"]
        for scope in tree["scope-hyperedges"]
        for end in scope["ends"]
        if end["role"] == "concept"
    }
    assert {"futon", "agency", "evidence"} <= concept_terms


def test_eightfold_map_items_are_nested_under_map_phase(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "M-war-demo.md",
        """# Mission: War Demo

## 1. IDENTIFY
The scope names a capability.

## 2. MAP

### Q1: Existing registry
See futon3c/src/futon3c/agency/registry.clj and M-agency-rebuild.

### Q2: Capability surface
The agency capability must remain visible.

## 3. DERIVE
Derive the frame.
""",
    )

    tree = scope_detect.detect_mission_scopes(
        path,
        kernel_terms=["capability", "agency", "derive"],
        capabilities={"agency"},
    )

    phases = _by_binder(tree, "eightfold-phase")
    assert [s["ends"][1]["phase"] for s in phases] == ["identify", "map", "derive"]
    map_scope = next(s for s in phases if s["ends"][1]["phase"] == "map")
    map_items = _by_binder(tree, "map-item")
    assert len(map_items) == 2
    assert all(item["parent"] == map_scope["scope-id"] for item in map_items)
    assert _by_binder(tree, "capability-scope")
    assert any(
        end["role"] == "mission" and end["ident"] == "M-agency-rebuild"
        for item in map_items
        for end in item["ends"]
    )
