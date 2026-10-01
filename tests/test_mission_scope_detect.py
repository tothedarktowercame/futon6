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


def test_operator_gates_are_distinct_typed_nodes_with_source_lines(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "M-gated-demo.md",
        """# Mission: Gated Demo

**Status:** INSTANTIATE
**Gate:** operator-acceptance — inspect the rendered graph
**Gate:** operator-input — choose the production threshold

## 1. IDENTIFY
The implementation is complete; the remaining actions belong to the operator.
""",
    )

    tree = scope_detect.detect_mission_scopes(
        path,
        kernel_terms=["implementation", "operator"],
        capabilities=set(),
        patterns=[],
    )

    gates = _by_binder(tree, "operator-gate")
    assert tree["scope-count-by-binder-type"]["operator-gate"] == 2
    assert [gate["gate-kind"] for gate in gates] == [
        "operator-acceptance",
        "operator-input",
    ]
    assert [gate["source-line"] for gate in gates] == [4, 5]
    assert gates[0]["gate-text"] == "inspect the rendered graph"
    assert gates[0]["ends"][2] == {
        "role": "operator-gate",
        "kind": "operator-acceptance",
        "text": "inspect the rendered graph",
        "source-line": 4,
    }
    assert all(gate["hx/type"] == "mission-scope/operator-gate" for gate in gates)


def test_real_argue_table_keeps_all_seven_library_citations(tmp_path: Path) -> None:
    root = SCRIPT.parents[2]
    mission = root / "futon2/holes/M-G-over-cascades.md"
    text = mission.read_text()
    table = text.split("**Pattern cross-reference", 1)[1].split("**Trade-offs", 1)[0]
    path = _write(tmp_path, "M-table.md",
                  "# Mission\n\n## MAP\n"
                  "`aif/expected-free-energy-scorecard`\n\n## ARGUE\n" + table)
    tree = scope_detect.detect_mission_scopes(
        path, kernel_terms=[], capabilities=set(),
        patterns=scope_detect.load_pattern_index(root),
    )
    scopes = _by_binder(tree, "pattern")
    argue = [s for s in scopes if s["ends"][1]["phase"] == "argue"]
    assert [s["ends"][2]["ident"] for s in argue] == [
        "futon-theory/structural-tension-as-observation",
        "aif/expected-free-energy-scorecard",
        "aif/candidate-pattern-action-space",
        "aif/admissibility",
        "aif/no-self-certification",
        "aif/off-continuity-null-discriminates",
        "aif/niche-construction",
    ]
    assert len(scopes) == 8
    assert all((root / s["ends"][2]["ref"]).is_file() for s in argue)


def test_short_prose_and_unknown_qualified_names_are_not_pattern_citations():
    index = {"aif/admissibility": "library/aif/admissibility.flexiarg",
             "no-self-certification": "library/aif/no-self-certification.flexiarg"}
    assert scope_detect.pattern_slots(
        "admissibility; unknown/no-self-certification", index) == []
    slots = scope_detect.pattern_slots(
        "`aif/admissibility` then `aif/admissibility`", index)
    assert len(slots) == 2
    assert slots[0]["offset"] < slots[1]["offset"]


def test_ambiguous_basenames_require_qualification_and_ignore_worktrees(tmp_path):
    for repo, category in [("futon3", "aif"), ("futon3", "other"),
                           ("futon3-old-worktree", "aif")]:
        library = tmp_path / repo / "library" / category
        library.mkdir(parents=True, exist_ok=True)
        (library / "no-self-certification.flexiarg").write_text("pattern")
    patterns = scope_detect.load_pattern_index(tmp_path)
    assert len(patterns) == 2
    path = _write(tmp_path, "M-ambiguous.md",
                  "# Mission\n\n## ARGUE\nno-self-certification\n"
                  "`aif/no-self-certification`\n")
    tree = scope_detect.detect_mission_scopes(
        path, kernel_terms=[], capabilities=set(), patterns=patterns)
    citations = _by_binder(tree, "pattern")
    assert len(citations) == 1
    assert citations[0]["ends"][2]["ident"] == "aif/no-self-certification"
