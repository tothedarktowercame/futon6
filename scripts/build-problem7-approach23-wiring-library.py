#!/usr/bin/env python3
"""Build a focused wiring-library for Problem 7 Approaches II/III.

Inputs:
- data/first-proof/problem7-approach23-curated-kb.json

Outputs:
- data/first-proof/problem7-approach23-wiring-library.json
- data/first-proof/problem7-approach23-wiring-library.md
"""

from __future__ import annotations

import json
from pathlib import Path

INPUT = Path("data/first-proof/problem7-approach23-curated-kb.json")
JSON_OUT = Path("data/first-proof/problem7-approach23-wiring-library.json")
MD_OUT = Path("data/first-proof/problem7-approach23-wiring-library.md")

SCHEMA = {
    "node_roles": [
        "Q: target subproblem",
        "D: decomposition/data model",
        "M: main method",
        "C: certificate theorem/tool",
        "O: output capability",
        "B: bridge verdict to P7 obligations",
    ],
    "edge_pattern": [
        "D -> Q (clarify)",
        "M -> D (assert)",
        "C -> M (reference)",
        "O -> Q (assert)",
        "B -> O (reform|challenge)",
    ],
}

# Higher-signal custom wiring text for key papers.
OVERRIDES = {
    "1705.10909": {
        "q": "Construct equivariant normal data for group actions so fixed-set surgery can be controlled.",
        "d": "Model manifold action with equivariant Spivak normal bundle rather than only ordinary normal data.",
        "m": "Use equivariant surgery input to build/compare normal maps compatible with the action.",
        "c": "Equivariant Spivak bundle and equivariant surgery theorems for compact Lie group actions.",
        "o": "Concrete route to repair Approach II normal-structure and transfer-compatibility gaps.",
        "b": "High-priority bridge for Approach II and for G2/G3-style compatibility constraints.",
        "b_edge": "reform",
    },
    "1811.08794": {
        "q": "Translate global-quotient orbifold data into equivariant bordism information usable for manifold upgrades.",
        "d": "Treat orbifolds as global quotients with equivariant Pontrjagin-Thom models.",
        "m": "Compute orbifold cobordism classes via equivariant collapse maps and quotient structures.",
        "c": "Equivariant Pontrjagin-Thom construction for global quotient orbifolds.",
        "o": "Potential mechanism for Approach III resolution while tracking quotient/fundamental-group behavior.",
        "b": "Strong cross-over module between Approach II equivariant surgery and Approach III orbifold route.",
        "b_edge": "reform",
    },
    "2506.23994": {
        "q": "Control fixed-hypersurface geometry in congruence hyperbolic manifolds under reflections/involutions.",
        "d": "Start from arithmetic reflective lattice and pass to congruence manifold covers.",
        "m": "Track induced involutions and geometric properties of fixed totally geodesic hypersurfaces.",
        "c": "Reflection/nonseparating fixed-set theorems in congruence hyperbolic settings.",
        "o": "Concrete geometric substrate for E2 and for odd-dimension alternative hunts.",
        "b": "Directly informs fixed-set side constraints needed before either Approach II or III closure.",
        "b_edge": "reform",
    },
    "math/0406607": {
        "q": "Realize prescribed finite symmetries in compact hyperbolic manifolds for action-construction flexibility.",
        "d": "Encode target finite symmetry as a normalizer quotient in finite-index lattice subgroups.",
        "m": "Use subgroup growth and commensurator control to force chosen finite isometry groups.",
        "c": "Finite-group realization theorem for compact hyperbolic manifold isometry groups.",
        "o": "Symmetry-existence supply for building candidate involution/reflection action models.",
        "b": "Useful constructor for alternate E2 branches, but fixed-set Euler behavior is not automatic.",
        "b_edge": "challenge",
    },
    "0705.3249": {
        "q": "Represent orbifold quotient constructions in a cohomological language compatible with group-action invariants.",
        "d": "Use translation groupoids to encode orbifold action data and isotropy structure.",
        "m": "Compute orbifold/Bredon invariants via groupoid-level functorial machinery.",
        "c": "Translation-groupoid and orbifold Bredon cohomology comparison framework.",
        "o": "Formal infrastructure for Approach III pi_1-aware resolution bookkeeping.",
        "b": "Framework module: clarifies semantics but does not itself construct the closed manifold.",
        "b_edge": "challenge",
    },
    "math/0312378": {
        "q": "Control family-level classifying spaces needed in orbifold and assembly transitions.",
        "d": "Parameterize subgroup families with E_F models and orbit-category functors.",
        "m": "Move between family domains (Fin, VCyc, etc.) while preserving homological meaning.",
        "c": "Classifying-space-for-families toolkit and comparison results.",
        "o": "Reliable domain control for Approach III orbifold/family transitions and cross-checks.",
        "b": "Infrastructure bridge supporting both Approach III and fallback assembly verifications.",
        "b_edge": "reform",
    },
    "1112.2104": {
        "q": "Track signatures and bundle data in bordisms with proper discrete-group actions.",
        "d": "Model manifolds with proper actions via equivariant bordism and G-bundle descriptors.",
        "m": "Use bordism invariants to constrain allowable surgery transformations under group actions.",
        "c": "Signature/bundle classification statements for proper-action bordisms.",
        "o": "Potential mechanism for proving Approach II surgeries preserve required global invariants.",
        "b": "Method candidate for rigorous action-preserving surgery bookkeeping.",
        "b_edge": "reform",
    },
}


def classify_track(tags: list[str]) -> str:
    if "Approach-II" in tags:
        return "approach_ii"
    if "Approach-III" in tags:
        return "approach_iii"
    if "E2-odd-alt" in tags:
        return "e2_odd_alt"
    return "interface"


def default_text(track: str, title: str, why: str) -> dict[str, str]:
    if track == "approach_ii":
        return {
            "q": "Advance Approach II: remove or control involution fixed-set effects while preserving target quotient group.",
            "d": "Represent data in an equivariant bordism/surgery model of manifolds with group action.",
            "m": f"Apply method pattern from '{title}' to fixed-set surgery or action-preserving modification steps.",
            "c": "Use the paper's equivariant topology/cobordism/surgery theorem package as certificate.",
            "o": "Candidate subroutine for action-compatible manifold upgrades in Approach II.",
            "b": why,
            "b_edge": "reform",
        }
    if track == "approach_iii":
        return {
            "q": "Advance Approach III: resolve orbifold singular structure without losing pi_1 control.",
            "d": "Model quotient/orbifold structure via groupoid/family-level descriptors.",
            "m": f"Use '{title}' techniques to connect orbifold invariants and manifold replacement operations.",
            "c": "Invoke orbifold/topological-stack/family comparison results as certificate step.",
            "o": "Potential resolution component with explicit orbifold bookkeeping.",
            "b": why,
            "b_edge": "reform",
        }
    if track == "e2_odd_alt":
        return {
            "q": "Find odd-dimensional E2 alternatives where fixed sets can have chi=0 despite even dimension.",
            "d": "Parameterize hyperbolic involution/reflection models by fixed-set geometry and arithmetic constraints.",
            "m": f"Extract fixed-set construction/rigidity behavior from '{title}'.",
            "c": "Use geometric fixed-set theorems/constraints as certificate.",
            "o": "Constraint map for viable odd-dimensional E2 candidate families.",
            "b": why,
            "b_edge": "challenge",
        }
    return {
        "q": "Provide interface support between Approach II/III work and legacy Approach I machinery.",
        "d": "Express assumptions/results in family-level or surgery-obstruction language.",
        "m": f"Leverage '{title}' as an interface theorem linking constructions to obstruction frameworks.",
        "c": "Use foundational assembly/PD/surgery statements from the paper.",
        "o": "Compatibility checks preventing hidden hypothesis mismatches.",
        "b": why,
        "b_edge": "challenge",
    }


def build_diagram(idx: int, paper: dict) -> dict:
    pid = f"p7a23-d{idx:02d}"
    paper_id = paper["id"]
    tags = paper.get("tags", [])
    track = classify_track(tags)

    txt = OVERRIDES.get(paper_id)
    if txt is None:
        txt = default_text(track, paper.get("title", "(untitled)"), paper.get("why", ""))

    nodes = [
        {"id": f"{pid}-q", "node_type": "question", "body_text": txt["q"]},
        {"id": f"{pid}-d", "node_type": "answer", "body_text": txt["d"]},
        {"id": f"{pid}-m", "node_type": "answer", "body_text": txt["m"]},
        {"id": f"{pid}-c", "node_type": "comment", "body_text": txt["c"]},
        {"id": f"{pid}-o", "node_type": "answer", "body_text": txt["o"]},
        {"id": f"{pid}-b", "node_type": "comment", "body_text": txt["b"]},
    ]
    edges = [
        {"source": f"{pid}-d", "target": f"{pid}-q", "edge_type": "clarify"},
        {"source": f"{pid}-m", "target": f"{pid}-d", "edge_type": "assert"},
        {"source": f"{pid}-c", "target": f"{pid}-m", "edge_type": "reference"},
        {"source": f"{pid}-o", "target": f"{pid}-q", "edge_type": "assert"},
        {"source": f"{pid}-b", "target": f"{pid}-o", "edge_type": txt["b_edge"]},
    ]

    return {
        "id": pid,
        "title": paper.get("title", "(untitled)"),
        "paper": paper_id,
        "url": paper.get("url", ""),
        "date": paper.get("date", ""),
        "track": track,
        "tags": tags,
        "why": paper.get("why", ""),
        "nodes": nodes,
        "edges": edges,
    }


def main() -> int:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    papers = source.get("papers", [])
    diagrams = [build_diagram(i + 1, p) for i, p in enumerate(papers)]

    track_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for d in diagrams:
        tr = d["track"]
        track_counts[tr] = track_counts.get(tr, 0) + 1
        for t in d.get("tags", []):
            tag_counts[t] = tag_counts.get(t, 0) + 1

    out = {
        "library_id": "problem7-approach23-wiring-library",
        "date": "2026-02-12",
        "focus": "Approach II and Approach III methods (with E2-odd-alt and interface dependencies)",
        "source": str(INPUT),
        "diagram_count": len(diagrams),
        "schema": SCHEMA,
        "track_counts": dict(sorted(track_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "diagrams": diagrams,
    }

    JSON_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Problem 7 Approach II/III Wiring Library",
        "",
        "Date: 2026-02-12",
        "",
        "This library is focused on the new S-branch framing from `0fa4e82`:",
        "- Approach II: equivariant surgery on `(M, tau)`",
        "- Approach III: orbifold resolution with `pi_1` control",
        "- E2-odd-alt: cross-cutting fixed-set geometry for odd-dimensional alternatives",
        "- Interface: retained dependencies linking back to Approach I infrastructure",
        "",
        f"Total diagrams: {len(diagrams)}",
        "",
        "## Counts by Track",
    ]
    for k, v in sorted(track_counts.items()):
        lines.append(f"- `{k}`: {v}")

    lines += ["", "## Counts by Tag"]
    for k, v in sorted(tag_counts.items()):
        lines.append(f"- `{k}`: {v}")

    lines += ["", "## Diagram Index", ""]
    for d in diagrams:
        tags = ", ".join(d.get("tags", []))
        lines.append(f"- `{d['id']}` — {d['title']} (`{d['paper']}`)")
        lines.append(f"  track: `{d['track']}`; tags: {tags}")

    lines += ["", "## Diagrams", ""]
    for d in diagrams:
        lines.append(f"### {d['id']}: {d['title']}")
        lines.append(f"- Paper: `{d['paper']}`")
        lines.append(f"- Track: `{d['track']}`")
        lines.append(f"- Tags: {', '.join(d.get('tags', []))}")
        lines.append(f"- Why selected: {d.get('why', '')}")
        node_map = {n['id'].rsplit('-',1)[-1]: n['body_text'] for n in d['nodes']}
        lines.append(f"- `Q`: {node_map['q']}")
        lines.append(f"- `D`: {node_map['d']}")
        lines.append(f"- `M`: {node_map['m']}")
        lines.append(f"- `C`: {node_map['c']}")
        lines.append(f"- `O`: {node_map['o']}")
        lines.append(f"- `B`: {node_map['b']}")
        lines.append("")

    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"diagram_count={len(diagrams)}")
    for k, v in sorted(track_counts.items()):
        print(f"track[{k}]={v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
