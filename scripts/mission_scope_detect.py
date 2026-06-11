#!/usr/bin/env python3
"""Recover mission scope trees as n-ary hx/ends hyperedges.

Mission scopes are frame binders: top-level lifecycle/loose sections bind
sub-scopes and concept/source/mission/capability slots.  The output shape follows
the futon6 nLab scope detector records closely enough for Arxana ingestion.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/joe/code")
KERNEL = ROOT / "futon6" / "data" / "mission-ner-kernel.json"
STAR_MAP = ROOT / "futon0" / "holes" / "missions" / "M-capability-star-map.graph.edn"
OUT_DIR = ROOT / "futon6" / "data" / "mission-scope-trees"

ENSEMBLE = [
    ROOT / "futon3" / "holes" / "missions" / "M-agency-forum.md",
    ROOT / "futon3" / "holes" / "missions" / "M-agency-rebuild.md",
    ROOT / "futon3c" / "holes" / "missions" / "M-agency-refactor.md",
    ROOT / "futon3c" / "holes" / "missions" / "M-war-machine.md",
    ROOT / "futon3c" / "holes" / "missions" / "M-war-machine-pilot.md",
    ROOT / "futon3c" / "holes" / "missions" / "M-war-machine-tuning.md",
]

PHASES = {
    "head": "head",
    "identify": "identify",
    "map": "map",
    "derive": "derive",
    "argue": "argue",
    "verify": "verify",
    "instantiate": "instantiate",
    "document": "document",
    "pending document": "document",
}

LOOSE_PHASE_MAP = [
    (re.compile(r"\b(scope|motivation|overview|conceptual foundation|time box|exit conditions?|success criteria)\b"), "identify"),
    (re.compile(r"\b(parts?|dependencies|source material|sequence|layer|work plan|blocks|enables|shares)\b"), "map"),
    (re.compile(r"\b(derivation|patterns?|decisions?|r11 enforcement)\b"), "derive"),
    (re.compile(r"\b(open questions?|notes?)\b"), "argue"),
    (re.compile(r"\b(validation|evidence|verify|test)\b"), "verify"),
]

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)

# In-passing phase closure: a phase satisfied inline rather than by a section,
# e.g. `**DOCUMENT phase:** satisfied by README...`. High-precision bold form.
INLINE_PHASE_CLOSURE_RE = re.compile(
    r"\*\*(HEAD|IDENTIFY|MAP|DERIVE|ARGUE|VERIFY|INSTANTIATE|DOCUMENT)"
    r"\s+phase:?\*\*\s*(?:is\s+)?(satisfied|closed|complete|done)",
    re.I,
)
MISSION_REF_RE = re.compile(r"\bM-[A-Za-z0-9][A-Za-z0-9-]*\b")
PATH_RE = re.compile(r"\b[\w./{}-]+\.(?:clj|cljs|cljc|py|edn|bb|el|json|md|html|css|ts|js)\b")
URL_RE = re.compile(r"https?://[^\s)>\"]+")
API_RE = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+(/[A-Za-z0-9_./:<>{}-]+)")
SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
WORD_RE = re.compile(r"[a-z][a-z0-9-]*")
PATTERN_CANDIDATE_RE = re.compile(r"(?<![a-z0-9-])([a-z][a-z0-9]*-[a-z0-9-]+(?:-[a-z0-9-]+)+)(?![a-z0-9-])")


@dataclass(frozen=True)
class Section:
    level: int
    title: str
    start: int
    content_start: int
    end: int
    text: str


def slug(s: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return out or "scope"


def clean_title(title: str) -> str:
    title = re.sub(r"`([^`]+)`", r"\1", title)
    title = re.sub(r"^\d+[.\-)]\s*", "", title.strip())
    title = re.sub(r"^\d+[–-]\d+[.\s-]*", "", title)
    title = re.sub(r"\s+\([^)]*\)$", "", title)
    title = title.replace("_", "").replace("*", "")
    return title.strip()


def phase_for_title(title: str) -> str | None:
    t = slug(clean_title(title)).replace("-", " ")
    words = t.split()
    if words:
        first = words[0]
        if first in PHASES:
            return PHASES[first]
    if t in PHASES:
        return PHASES[t]
    return None


def loose_phase_for_title(title: str) -> str:
    t = slug(clean_title(title)).replace("-", " ")
    for pat, phase in LOOSE_PHASE_MAP:
        if pat.search(t):
            return phase
    return "loose"


def split_sections(text: str) -> list[Section]:
    matches = list(HEADER_RE.finditer(text))
    sections: list[Section] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(
            Section(
                level=len(m.group(1)),
                title=clean_title(m.group(2)),
                start=m.start(),
                content_start=m.end(),
                end=end,
                text=text[m.end():end],
            )
        )
    return sections


def load_kernel_terms(path: Path = KERNEL, limit: int = 800) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["terms"] if r.get("df", 0) > 0]
    rows = sorted(rows, key=lambda r: (-r.get("score", 0), -r.get("df", 0), r["term"]))
    terms = [r["term"] for r in rows[:limit]]
    return sorted(set(terms), key=lambda t: (-len(t), t))


def load_capabilities(path: Path = STAR_MAP) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    cap_block = text.split(":capabilities", 1)[1].split(":missions", 1)[0]
    return set(re.findall(r"(?m)^\s*:([a-z][a-z0-9-]+)\s*\{", cap_block))


def load_pattern_index(root: Path = ROOT) -> list[dict]:
    """Distinctive flexiarg basenames available as literal pattern citations."""
    patterns = []
    seen: set[str] = set()
    for path in sorted(root.glob("futon*/library/**/*.flexiarg")):
        name = path.stem
        if name in seen:
            continue
        if name.count("-") >= 2 and len(name) >= 12:
            patterns.append(
                {
                    "name": name,
                    "ref": path.relative_to(root).as_posix(),
                }
            )
            seen.add(name)
    return patterns


def find_concepts(text: str, terms: Iterable[str], max_terms: int = 40) -> list[dict]:
    norm = text.lower()
    hits: dict[str, int] = {}
    for term in terms:
        if len(term) < 3:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(term).replace(r"\-", r"[-\s]") + r"(?![a-z0-9])"
        m = re.search(pattern, norm)
        if m:
            hits[term] = m.start()
    return [
        {"role": "concept", "term": term}
        for term, _ in sorted(hits.items(), key=lambda kv: (kv[1], kv[0]))[:max_terms]
    ]


def bullet_items(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$", line)
        if m:
            item = re.sub(r"\s+", " ", m.group(1)).strip()
            if item:
                items.append(item[:240])
    return items


def source_slots(text: str) -> list[dict]:
    slots = []
    seen = set()
    for kind, values in [
        ("url", URL_RE.findall(text)),
        ("api", [m.group(0) for m in API_RE.finditer(text)]),
        ("file", PATH_RE.findall(text)),
        ("commit", SHA_RE.findall(text)),
    ]:
        for value in values:
            key = (kind, value)
            if key not in seen:
                slots.append({"role": "source", "kind": kind, "ref": value})
                seen.add(key)
    return slots


def relation_for_line(line: str, fallback: str = "relates-to") -> str:
    l = line.lower()
    if re.search(r"\b(blocks?|requires?|depends?|prereq)", l):
        return "depends"
    if re.search(r"\b(enables?|unblocks?|produces?)", l):
        return "enables"
    if re.search(r"\b(shares?|sibling|related|relation)", l):
        return "relates-to"
    return fallback


def mission_ref_slots(text: str, fallback: str = "relates-to") -> list[dict]:
    slots = []
    seen = set()
    for line in text.splitlines():
        for mission in MISSION_REF_RE.findall(line):
            key = (mission, relation_for_line(line, fallback))
            if key not in seen:
                slots.append({"role": "mission", "ident": mission, "relation": key[1]})
                seen.add(key)
    return slots


def capability_slots(text: str, capabilities: set[str], concept_terms: Iterable[str]) -> list[dict]:
    slots = []
    seen = set()
    low = text.lower()
    for cap in sorted(capabilities):
        if re.search(r"(?<![a-z0-9])" + re.escape(cap).replace(r"\-", r"[-\s]") + r"(?![a-z0-9])", low):
            slots.append({"role": "capability", "ident": cap, "source": "star-map"})
            seen.add(cap)
    for term in concept_terms:
        if "capab" in term and term not in seen and re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", low):
            slots.append({"role": "capability", "ident": term, "source": "kernel"})
            seen.add(term)
    return slots


def pattern_slots(text: str, pattern_index: dict[str, str]) -> list[dict]:
    slots = []
    seen = set()
    for m in PATTERN_CANDIDATE_RE.finditer(text):
        name = m.group(1)
        if name in seen:
            continue
        if name in pattern_index:
            slots.append(
                {
                    "role": "pattern",
                    "ident": name,
                    "ref": pattern_index[name],
                }
            )
            seen.add(name)
    return slots


def _record_field(window: str, label_re: str) -> str | None:
    """Extract the value of a `Label:` field up to the next field/blank."""
    m = re.search(
        r"\*{0,2}(?:" + label_re + r")\*{0,2}:?\s*(.+?)"
        r"(?:\n\s*[-*]?\s*\*{0,2}[A-Z][A-Za-z ]{2,20}\*{0,2}:|\n\s*\n|$)",
        window,
        re.S,
    )
    return re.sub(r"\s+", " ", m.group(1)).strip().strip("*").strip()[:400] if m else None


def psr_pur_records(text: str) -> list[dict]:
    """Detect genuine PSR/PUR records via the template discriminator (use, not
    mention): a `Pattern:` line with adjacent Outcome+Prediction-error => PUR;
    a `Pattern chosen:` line with adjacent Rationale/Candidates => PSR. Carries
    the named pattern + the structured facets (the prediction-error/outcome being
    the grounding signal for the downstream priming layer)."""
    lines = text.split("\n")
    out = []
    for i, ln in enumerate(lines):
        window = "\n".join(lines[i : i + 12])
        # Pattern idents may be backtick-quoted and namespaced
        # (`structure/two-projections-of-one-quantity`) — the house style.
        mp = re.match(r"\s*[-*]?\s*\*{0,2}Pattern:?\*{0,2}\s*`?([a-z0-9][a-z0-9/-]{6,})", ln)
        if mp and "chosen" not in ln.lower() and re.search(r"Prediction error", window) and re.search(r"Outcome", window):
            out.append({
                "kind": "pur",
                "pattern": mp.group(1).rstrip("-"),
                "anchor": ln.strip()[:160],
                "facets": {
                    "actions": _record_field(window, "Actions taken|Actions"),
                    "outcome": _record_field(window, "Outcome"),
                    "prediction-error": _record_field(window, "Prediction error"),
                    "notes": _record_field(window, "Notes"),
                },
            })
            continue
        cp = re.search(r"Pattern chosen:?\s*\*{0,2}\s*`?([a-z0-9][a-z0-9/-]{6,})", ln)
        if cp and re.search(r"Rationale|Candidates", window):
            out.append({
                "kind": "psr",
                "pattern": cp.group(1).rstrip("-"),
                "anchor": ln.strip()[:160],
                "facets": {
                    "candidates": _record_field(window, "Candidates considered|Candidates"),
                    "rationale": _record_field(window, "Rationale"),
                },
            })
    return out


def make_scope(
    entity_id: str,
    idx: int,
    binder_type: str,
    parent: str | None,
    title: str,
    phase: str,
    position: int,
    end: int,
    ends: list[dict],
) -> dict:
    scope_id = f"{entity_id}:scope-{idx:03d}"
    all_ends = [{"role": "entity", "ident": entity_id}, {"role": "environment", "name": title, "phase": phase}]
    all_ends.extend(ends)
    return {
        "scope-id": scope_id,
        "binder-type": binder_type,
        "parent": parent,
        "ends": all_ends,
        "hx/id": scope_id,
        "hx/role": "component",
        "hx/type": f"mission-scope/{binder_type}",
        "hx/parent": parent,
        "hx/ends": all_ends,
        "hx/content": {"match": title[:120], "position": position, "end": end},
        "hx/labels": ["scope", "mission-scope", binder_type, phase],
    }


def is_scope_in(title: str) -> bool:
    return bool(re.search(r"\bscope\s+in\b|\bin\s*/\s*out\b", title.lower()))


def is_scope_out(title: str) -> bool:
    return bool(re.search(r"\bscope\s+out\b|\bin\s*/\s*out\b", title.lower()))


def is_source_material(title: str) -> bool:
    return "source" in title.lower() and "material" in title.lower()


def is_relation_section(title: str) -> bool:
    return bool(re.search(r"\b(relationship|relation|dependencies|blocks|enables|shares)\b", title.lower()))


def is_map_item(parent_phase: str, section: Section) -> bool:
    if parent_phase != "map":
        return False
    return section.level >= 3 or bool(re.match(r"^(?:q\d+|part\s+[ivx]+|phase\s+\d+|inventory|ready|missing)", section.title.lower()))


def detect_mission_scopes(
    path: Path,
    kernel_terms: list[str] | None = None,
    capabilities: set[str] | None = None,
    patterns: list[dict] | None = None,
) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    entity_id = path.stem
    kernel_terms = kernel_terms if kernel_terms is not None else load_kernel_terms()
    capabilities = capabilities if capabilities is not None else load_capabilities()
    patterns = patterns if patterns is not None else load_pattern_index()
    sections = split_sections(text)
    pattern_index = {pattern["name"]: pattern["ref"] for pattern in patterns}

    scopes: list[dict] = []
    phase_stack: list[tuple[int, str, str, str]] = []
    seen_patterns: set[str] = set()
    idx = 0

    def current_parent(level: int) -> tuple[str | None, str]:
        for stack_level, scope_id, phase, _title in reversed(phase_stack):
            if stack_level < level:
                return scope_id, phase
        return None, "head"

    def current_phase_context(level: int) -> str:
        """Nearest enclosing NON-loose phase. A ### subsection binds as a
        loose-section whose own phase is often 'loose', which used to mask the
        eightfold phase above it — so map-items/capabilities under `## MAP`
        stopped firing (the E-scope-audit W3 side-regression; quantified by
        mission_scope_bindings.py as MAP binding nothing). Sub-binder decisions
        should see the phase of the enclosing eightfold scope, not the loose
        wrapper."""
        for stack_level, _scope_id, phase, _title in reversed(phase_stack):
            if stack_level < level and phase not in (None, "loose"):
                return phase
        return "loose"

    for sec in sections:
        phase = phase_for_title(sec.title)
        binder = None
        mapped_phase = phase
        if phase is not None and sec.level <= 2:
            binder = "eightfold-phase"
        elif re.search(r"plain[- ](language|text|english)[- ]?(argument|statement|version)?",
                       sec.title, re.I):
            # Defined sub-scope of ARGUE (Joe, 2026-06-10): the plain-language
            # statement of the argument is a lifecycle requirement, not just
            # another loose section.
            binder = "plain-argument"
            mapped_phase = "argue"
        elif sec.level <= 3:
            # Level-3 subsections are real scope structure (INSTANTIATE
            # handoffs, VERIFY hooks, ARGUE rounds, PSR/PUR sections) — bind
            # them as nested loose-sections; phase_stack supplies the parent.
            binder = "loose-section"
            mapped_phase = loose_phase_for_title(sec.title)

        parent, parent_phase = current_parent(sec.level)
        if binder is not None:
            ends = [{"role": "heading", "level": sec.level, "title": sec.title}]
            ends.extend(find_concepts(sec.text, kernel_terms))
            scope = make_scope(entity_id, idx, binder, parent, sec.title, mapped_phase or "loose", sec.start, sec.end, ends)
            scopes.append(scope)
            phase_stack = [p for p in phase_stack if p[0] < sec.level]
            phase_stack.append((sec.level, scope["scope-id"], mapped_phase or "loose", sec.title))
            idx += 1
            parent, parent_phase = scope["scope-id"], mapped_phase or "loose"

        parent, parent_phase = current_parent(sec.level + 1)
        phase_ctx = parent_phase if parent_phase != "loose" else current_phase_context(sec.level + 1)
        title_low = sec.title.lower()

        sub_binders: list[tuple[str, list[dict], str]] = []
        if is_scope_in(sec.title):
            items = bullet_items(sec.text)
            sub_binders.append(("mission-scope-in", [{"role": "bounded-item", "text": item} for item in items], "identify"))
            if is_scope_out(sec.title):
                sub_binders.append(("mission-scope-out", [{"role": "bounded-item", "text": item} for item in items], "identify"))
        elif is_scope_out(sec.title):
            sub_binders.append(("mission-scope-out", [{"role": "bounded-item", "text": item} for item in bullet_items(sec.text)], "identify"))

        if is_source_material(sec.title):
            sub_binders.append(("source-material", source_slots(sec.text), "map"))
        if is_relation_section(sec.title):
            sub_binders.append(("relates-to", mission_ref_slots(sec.text), "map"))

        cap_slots = capability_slots(sec.text, capabilities, kernel_terms)
        if cap_slots and ("capab" in title_low or phase_ctx in {"identify", "map", "derive"}):
            sub_binders.append(("capability-scope", cap_slots, phase_ctx))

        pat_slots = [slot for slot in pattern_slots(sec.text, pattern_index) if slot["ident"] not in seen_patterns]
        if pat_slots:
            seen_patterns.update(slot["ident"] for slot in pat_slots)
            sub_binders.append(("pattern", pat_slots, phase_ctx))

        if is_map_item(phase_ctx, sec):
            map_ends = [{"role": "map-item", "title": sec.title}]
            map_ends.extend(source_slots(sec.text)[:12])
            map_ends.extend(mission_ref_slots(sec.text)[:12])
            sub_binders.append(("map-item", map_ends, "map"))

        for binder_type, ends, phase_name in sub_binders:
            if not ends and binder_type not in {"map-item"}:
                continue
            ends = ends + find_concepts(sec.text, kernel_terms)
            scope = make_scope(entity_id, idx, binder_type, parent, sec.title, phase_name, sec.start, sec.end, ends)
            scopes.append(scope)
            idx += 1

        # In-passing phase closures: `**DOCUMENT phase:** satisfied by ...`
        # anchor an eightfold-phase scope at the closure line itself.
        for m in INLINE_PHASE_CLOSURE_RE.finditer(sec.text):
            phase_name = m.group(1).lower()
            pos = sec.content_start + m.start()
            ends = [{"role": "phase-closure", "phase": phase_name,
                     "verdict": m.group(2).lower()}]
            scope = make_scope(entity_id, idx, "eightfold-phase", parent,
                               f"{phase_name.upper()} (closed in passing)",
                               phase_name, pos, pos + (m.end() - m.start()), ends)
            scope["closure-in-passing"] = True
            scopes.append(scope)
            idx += 1

        # psr/pur: attested pattern-application records (use, not mention).
        for rec in psr_pur_records(sec.text):
            p_ends = [{"role": "pattern", "ident": rec["pattern"], "ref": pattern_index.get(rec["pattern"])}]
            scope = make_scope(entity_id, idx, rec["kind"], parent, sec.title, phase_ctx, sec.start, sec.end, p_ends)
            scope["record"] = rec["kind"]
            scope["pattern-ident"] = rec["pattern"]
            scope["facets"] = rec["facets"]
            scope["anchor-line"] = rec["anchor"]
            scopes.append(scope)
            idx += 1

    if not scopes:
        ends = find_concepts(text, kernel_terms)
        scopes.append(make_scope(entity_id, 0, "loose-section", None, "whole mission", "loose", 0, len(text), ends))

    counts = Counter(scope["binder-type"] for scope in scopes)
    return {
        "mission": entity_id,
        "path": str(path),
        "scope-count-by-binder-type": dict(sorted(counts.items())),
        "scope-hyperedges": scopes,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("missions", nargs="*", type=Path, help="Mission markdown files. Defaults to the six L3 ensemble missions.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    missions = args.missions or ENSEMBLE
    kernel_terms = load_kernel_terms()
    capabilities = load_capabilities()
    patterns = load_pattern_index()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in missions:
        p = path if path.is_absolute() else ROOT / path
        tree = detect_mission_scopes(p, kernel_terms, capabilities, patterns)
        out = args.out_dir / f"{p.stem}.json"
        out.write_text(json.dumps(tree, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        counts = tree["scope-count-by-binder-type"]
        counts_s = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"{p.stem}: {counts_s} -> {out}")


if __name__ == "__main__":
    main()
