#!/usr/bin/env python3
"""Skolem audit over mission scope trees: is anything actually bound?

A mission read as a binding structure (E-mission-head / Anatomy follow-up,
Joe 2026-06-11): the HEAD states an existential (there is a construction
satisfying these conditions), MAP is the universal binder (it enumerates the
context items the construction must be a function OF), and the body phases
(DERIVE..DOCUMENT) must USE what MAP binds — otherwise the map was decoration
and the mission is "vibe coded". Skolem's suspicion, operationalized: an
empty scope is `∀x:` with no body.

Three failure classes, per mission:
  vacuous-binder  — a scope that binds no content ends (∀x: <nothing>)
  unused-binding  — an item bound in HEAD/IDENTIFY/MAP that no body-phase
                    scope ever uses (bound variable, zero occurrences)
  free-variable   — an item used in a body phase that was never introduced
                    (occurrence with no binder)

Two channels, and their disagreement is the diagnostic:
  ends channel — occurrences as detector-bound scope ends (structural)
  text channel — literal occurrences in the raw text of phase regions
A violation confirmed by BOTH channels is a real binding failure in the
mission; an ends-only violation is detector blindness (an E-scope-audit
W-class finding), not necessarily a document vice.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path("/home/joe/code")
TREES = ROOT / "futon6" / "data" / "mission-scope-trees"
SUBSTRATE_URL = "http://localhost:7071"

BINDER_PHASES = {"head", "identify", "map"}
BODY_PHASES = {"derive", "argue", "verify", "instantiate", "document"}

# Roles that are scope plumbing, not bound content. A map-item end carries
# only the section title (the binder's own name), so it does not count as
# content either.
BOILERPLATE_ROLES = {"entity", "environment", "heading", "map-item"}
# Concept ends are kernel-unigram keyword grabs (E-scope-audit W7): too noisy
# to certify a scope as bound, so they rate a scope "concept-only", not bound.
WEAK_ROLES = {"concept"}


def scope_phase(scope: dict) -> str:
    for end in scope.get("ends", []):
        if end.get("role") == "environment":
            return end.get("phase", "loose")
    return "loose"


def scope_title(scope: dict) -> str:
    for end in scope.get("ends", []):
        if end.get("role") == "environment":
            return end.get("name", "")
    return ""


def content_grade(scope: dict) -> str:
    """'bound' | 'concept-only' | 'vacuous'."""
    roles = {end.get("role") for end in scope.get("ends", [])}
    strong = roles - BOILERPLATE_ROLES - WEAK_ROLES
    if strong:
        return "bound"
    if roles & WEAK_ROLES:
        return "concept-only"
    return "vacuous"


def item_key(end: dict) -> tuple[str, str] | None:
    """High-precision bindable items only (W7 keeps concepts out)."""
    role = end.get("role")
    if role == "source" and end.get("kind") == "file":
        return ("file", end["ref"])
    if role == "mission":
        return ("mission", end["ident"])
    if role == "pattern":
        return ("pattern", end["ident"])
    if role == "capability":
        return ("capability", end["ident"])
    return None


def text_needle(kind: str, ident: str) -> re.Pattern:
    if kind == "file":
        ident = Path(ident).name  # paths vary; the basename is the stable cite
    if kind == "pattern" and "/" in ident:
        ident = ident.rsplit("/", 1)[1]
    return re.compile(r"(?<![\w/-])" + re.escape(ident) + r"(?![\w-])")


def endpoint_values(edge: dict) -> list[str]:
    return [str(e) for e in edge.get("hx/endpoints", [])]


def edge_type(edge: dict) -> str | None:
    return edge.get("hx/type") or edge.get("type")


def file_equivalent(bound_ident: str, edited_ident: str) -> bool:
    """Match exact paths, suffix paths, or stable basenames."""
    bound = Path(bound_ident)
    edited = Path(edited_ident)
    bound_s = str(bound)
    edited_s = str(edited)
    return (
        bound_s == edited_s
        or bound_s.endswith("/" + edited_s)
        or edited_s.endswith("/" + bound_s)
        or bound.name == edited.name
    )


def attributed_code_files(mission: str, code_edges: Iterable[dict] | None) -> set[str]:
    """Files edited by commits attributed to MISSION via substrate-2 code edges.

    Expected edge shapes mirror substrate-2 hyperedges:
    - code/v05/commit→mission endpoints [commit-sha, mission-id, ...]
    - code/v05/edits endpoints [commit-sha, file-or-var-id, ...]
    Direction sentinels in endpoint slot 3 are ignored.
    """
    if not code_edges:
        return set()

    commits = set()
    edits_by_commit: dict[str, set[str]] = defaultdict(set)
    for edge in code_edges:
        endpoints = endpoint_values(edge)
        if len(endpoints) < 2:
            continue
        etype = edge_type(edge)
        if etype == "code/v05/commit→mission" and endpoints[1] == mission:
            commits.add(endpoints[0])
        elif etype == "code/v05/edits":
            edits_by_commit[endpoints[0]].add(endpoints[1])

    files = set()
    for commit in commits:
        files.update(edits_by_commit.get(commit, set()))
    return files


def fetch_code_edges(base_url: str = SUBSTRATE_URL) -> list[dict]:
    """Fetch the two substrate-2 edge lanes needed by the code channel."""
    edges = []
    for hx_type in ("code/v05/commit→mission", "code/v05/edits"):
        query = urlencode({"type": hx_type, "limit": 5000})
        req = Request(
            f"{base_url.rstrip('/')}/api/alpha/hyperedges?{query}",
            headers={"Accept": "application/json"},
        )
        with urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        edges.extend(payload.get("hyperedges", []))
    return edges


def phase_region_text(scopes: list[dict], text: str, phases: set[str]) -> str:
    """Union of raw-text intervals belonging to scopes in PHASES."""
    intervals = []
    for scope in scopes:
        if scope_phase(scope) in phases:
            c = scope.get("hx/content", {})
            start, end = c.get("position"), c.get("end")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                intervals.append((start, end))
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return "\n".join(text[s:e] for s, e in merged)


def analyze_tree(tree: dict, text: str, code_edges: Iterable[dict] | None = None) -> dict:
    scopes = tree["scope-hyperedges"]
    code_files = attributed_code_files(tree["mission"], code_edges)

    vacuous = []
    concept_only = 0
    for scope in scopes:
        grade = content_grade(scope)
        if grade == "vacuous":
            vacuous.append(
                {
                    "scope-id": scope.get("scope-id"),
                    "binder-type": scope.get("binder-type"),
                    "phase": scope_phase(scope),
                    "title": scope_title(scope)[:80],
                }
            )
        elif grade == "concept-only":
            concept_only += 1

    # Ends channel: where does each item occur?
    binder_items: dict[tuple, set] = defaultdict(set)
    body_items: dict[tuple, set] = defaultdict(set)
    for scope in scopes:
        phase = scope_phase(scope)
        side = (
            binder_items
            if phase in BINDER_PHASES
            else body_items
            if phase in BODY_PHASES
            else None
        )
        if side is None:
            continue
        for end in scope.get("ends", []):
            key = item_key(end)
            if key:
                side[key].add(scope.get("scope-id"))

    binder_text = phase_region_text(scopes, text, BINDER_PHASES)
    body_text = phase_region_text(scopes, text, BODY_PHASES)

    def text_hits(key: tuple) -> tuple[bool, bool]:
        needle = text_needle(*key)
        return bool(needle.search(binder_text)), bool(needle.search(body_text))

    unused = []
    for key in sorted(binder_items):
        if key in body_items:
            continue
        _, in_body_text = text_hits(key)
        code_discharged = (
            key[0] == "file"
            and any(file_equivalent(key[1], edited) for edited in code_files)
        )
        verdict = (
            "doc-used"
            if in_body_text
            else "code-discharged"
            if code_discharged
            else "confirmed-unused"
        )
        unused.append(
            {
                "kind": key[0],
                "ident": key[1],
                "bound-in": sorted(binder_items[key]),
                "verdict": verdict,
                "confirmed": verdict == "confirmed-unused",
            }
        )

    free = []
    for key in sorted(body_items):
        if key in binder_items:
            continue
        in_binder_text, _ = text_hits(key)
        free.append(
            {
                "kind": key[0],
                "ident": key[1],
                "used-in": sorted(body_items[key]),
                "confirmed": not in_binder_text,  # both: never introduced
            }
        )

    phases_present = {scope_phase(s) for s in scopes}
    spine = bool(phases_present & BINDER_PHASES) and bool(phases_present & BODY_PHASES)

    return {
        "mission": tree["mission"],
        "scopes": len(scopes),
        "spine": spine,
        "body-chars": len(body_text),
        "vacuous": vacuous,
        "concept-only": concept_only,
        "bound-items": len(binder_items),
        "used-items": len({k for k in binder_items if k in body_items}),
        "code-files": sorted(code_files),
        "unused-bindings": unused,
        "free-variables": free,
    }


def fmt_report(r: dict, verbose: bool = False) -> str:
    lines = []
    confirmed_unused = [u for u in r["unused-bindings"] if u["confirmed"]]
    confirmed_free = [f for f in r["free-variables"] if f["confirmed"]]
    lines.append(
        f"{r['mission']}: scopes={r['scopes']} spine={'yes' if r['spine'] else 'NO'} "
        f"body={r['body-chars'] // 1000}k "
        f"vacuous={len(r['vacuous'])} concept-only={r['concept-only']} "
        f"bound={r['bound-items']} used={r['used-items']} "
        f"unused={len(r['unused-bindings'])}({len(confirmed_unused)} confirmed) "
        f"free={len(r['free-variables'])}({len(confirmed_free)} confirmed)"
    )
    if verbose:
        for v in r["vacuous"]:
            lines.append(f"  vacuous  [{v['binder-type']}/{v['phase']}] {v['title']}")
        for u in r["unused-bindings"]:
            tag = u.get("verdict") or ("CONFIRMED" if u["confirmed"] else "ends-only (detector blind?)")
            lines.append(f"  unused   {u['kind']}:{u['ident']} — {tag}")
        for f in r["free-variables"]:
            tag = "CONFIRMED" if f["confirmed"] else "ends-only (detector blind?)"
            lines.append(f"  free     {f['kind']}:{f['ident']} — {tag}")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("missions", nargs="*", help="Mission names (tree stems). Default: all trees.")
    ap.add_argument("--trees-dir", type=Path, default=TREES)
    ap.add_argument("--json", action="store_true", help="Emit full JSON instead of the table.")
    ap.add_argument(
        "--code-edges",
        type=Path,
        default=None,
        help="Optional JSON file containing substrate-2 code/v05/commit→mission and code/v05/edits hyperedges.",
    )
    ap.add_argument(
        "--code-channel",
        action="store_true",
        help="Fetch substrate-2 code edges and use them as the third audit channel.",
    )
    ap.add_argument("--substrate-url", default=SUBSTRATE_URL)
    ap.add_argument("-v", "--verbose", action="store_true", help="Per-finding detail lines.")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    stems = args.missions or sorted(p.stem for p in args.trees_dir.glob("*.json"))
    code_edges = []
    if args.code_edges:
        code_edges = json.loads(args.code_edges.read_text(encoding="utf-8"))
    elif args.code_channel:
        code_edges = fetch_code_edges(args.substrate_url)
    reports = []
    for stem in stems:
        tree = json.loads((args.trees_dir / f"{stem}.json").read_text(encoding="utf-8"))
        path = Path(tree["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            print(f"{tree['mission']}: SKIPPED (missing doc: {path})")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        reports.append(analyze_tree(tree, text, code_edges=code_edges))

    if args.json:
        print(json.dumps(reports, indent=2))
        return

    # Worst offenders first: confirmed unused bindings, then vacuity.
    reports.sort(
        key=lambda r: (
            -sum(u["confirmed"] for u in r["unused-bindings"]),
            -len(r["vacuous"]),
        )
    )
    for r in reports:
        print(fmt_report(r, verbose=args.verbose))
    total_unused = sum(sum(u["confirmed"] for u in r["unused-bindings"]) for r in reports)
    total_free = sum(sum(f["confirmed"] for f in r["free-variables"]) for r in reports)
    total_vacuous = sum(len(r["vacuous"]) for r in reports)
    spineless = sum(1 for r in reports if not r["spine"])
    print(
        f"\n{len(reports)} missions: {total_vacuous} vacuous scopes, "
        f"{total_unused} confirmed unused bindings, {total_free} confirmed free variables, "
        f"{spineless} missions with no binder/body spine"
    )


if __name__ == "__main__":
    main()
