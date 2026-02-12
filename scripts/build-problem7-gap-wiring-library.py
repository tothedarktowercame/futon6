#!/usr/bin/env python3
"""Build a paper-mined wiring-diagram library for Problem 7 gaps.

Outputs:
- data/first-proof/problem7-gap-wiring-library.json
- data/first-proof/problem7-gap-wiring-library.md
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = {
    "node_roles": [
        "Q: target subproblem",
        "D: decomposition or data model",
        "M: core mechanism",
        "C: certificate theorem/tool",
        "O: output statement",
        "B: bridge status back to P7 gaps",
    ],
    "edge_pattern": [
        "D -> Q (clarify)",
        "M -> D (assert)",
        "C -> M (reference)",
        "O -> Q (assert)",
        "B -> O (reform|challenge)",
    ],
}

# 22 diagrams mined from the curated Problem 7 paper KB.
DIAGRAM_SPECS = [
    {
        "id": "p7g-d01",
        "title": "Fowler FH(Q) Fixed-Set Criterion",
        "paper": "arXiv:1204.4667",
        "gap_tags": ["G1", "E2"],
        "bridge_status": "direct",
        "q": "Place an orbifold extension group Gamma in FH(Q) via finite-action data.",
        "d": "Model Gamma as pi_1((EG x Bpi)/G) for finite G-action on finite Bpi.",
        "m": "Reduce finiteness obstruction to Euler-characteristic data on fixed components.",
        "c": "Main theorem: vanishing chi on nontrivial fixed components kills obstruction.",
        "o": "Existence of finite CW with pi_1=Gamma and rationally acyclic universal cover.",
        "b": "Directly discharges E2 when concrete Z/2 fixed-set checks are verified.",
        "b_edge": "reform",
    },
    {
        "id": "p7g-d02",
        "title": "Avramidi Rational Manifold Model Pipeline",
        "paper": "arXiv:1506.06293",
        "gap_tags": ["G2"],
        "bridge_status": "partial",
        "q": "Upgrade rational duality/finiteness input to manifold-level realization.",
        "d": "Start from duality-group and finite-classifying-space hypotheses.",
        "m": "Construct rational surgery/normal-map pipeline through manifold-with-boundary stage.",
        "c": "Rational surgery interface theorems and reflection-group closure methods.",
        "o": "Manifold models with rationally acyclic universal covers in controlled ranges.",
        "b": "Potential route for G2, but requires careful pi_1-preservation in P7 setting.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d03",
        "title": "Bredon-Poincare Duality Group Framework",
        "paper": "arXiv:1311.7629",
        "gap_tags": ["G1"],
        "bridge_status": "support",
        "q": "Interpret Poincare duality when torsion prevents classical PD-group framing.",
        "d": "Use proper actions/family language and Bredon module categories.",
        "m": "Translate geometric duality into Bredon cohomological duality statements.",
        "c": "Bredon-PD theorems and criteria for duality over families of subgroups.",
        "o": "Cohomological PD control for torsion-containing groups acting properly.",
        "b": "Supports G1 at homology/cohomology level, not chain-level Poincare complex by itself.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d04",
        "title": "Orbifold Translation Groupoid Cohomology Scaffold",
        "paper": "arXiv:0705.3249",
        "gap_tags": ["G1"],
        "bridge_status": "support",
        "q": "Relate orbifold quotient geometry to computable equivariant invariants.",
        "d": "Encode orbifold actions by translation groupoids.",
        "m": "Compute invariants via Bredon-style/orbifold cohomological functors.",
        "c": "Groupoid-cohomology comparison theorems for orbifold invariants.",
        "o": "Formal bridge between quotient-orbifold language and equivariant homological data.",
        "b": "Useful infrastructure for G1 framing, but not a direct Poincare-complex proof.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d05",
        "title": "Classifying Spaces for Families Toolkit",
        "paper": "arXiv:math/0312378",
        "gap_tags": ["G1", "FJ"],
        "bridge_status": "support",
        "q": "Control transitions among E_Fin, E_VCyc and family-indexed assembly domains.",
        "d": "Model group actions with family-classifying spaces and orbit-category functors.",
        "m": "Use family filtration and equivariant homology functoriality.",
        "c": "General classifying-space theorems and family comparison machinery.",
        "o": "Reusable setup for obstruction-domain changes and assembly-map inputs.",
        "b": "Supports U1/FJ structure checks; no direct obstruction vanishing.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d06",
        "title": "FJ with Coefficients Inheritance Template",
        "paper": "arXiv:math/0510602",
        "gap_tags": ["FJ"],
        "bridge_status": "support",
        "q": "Prove assembly isomorphism robustness under constructions relevant to lattices.",
        "d": "Work in additive-category-with-involution coefficient framework.",
        "m": "Exploit inheritance/closure properties under group extensions/actions.",
        "c": "Coefficient-form Farrell-Jones statements and transitivity principles.",
        "o": "Assembly reduction remains valid in broader categorical settings.",
        "b": "Strengthens FJ leg of S-branch; does not identify sigma(f).",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d07",
        "title": "K/L-Theory of Group Rings Survey Pattern",
        "paper": "arXiv:1003.5002",
        "gap_tags": ["FJ"],
        "bridge_status": "support",
        "q": "Organize obstruction computations after assembly reduction.",
        "d": "Express target L-groups via equivariant homology and known decomposition pieces.",
        "m": "Use high-level computation templates and conjectural implications.",
        "c": "Survey-level statements linking FJ to explicit L-group calculations.",
        "o": "Roadmap for moving from abstract assembly to computable summands.",
        "b": "Method guidance only; needs problem-specific arithmetic/topological input.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d08",
        "title": "Hyperbolic/Virtually Abelian L-Group Computation Pattern",
        "paper": "arXiv:1007.0845",
        "gap_tags": ["FJ", "U3"],
        "bridge_status": "partial",
        "q": "Extract explicit L-group formulas in groups with VCyc structure.",
        "d": "Decompose by finite and virtually cyclic subgroup contributions.",
        "m": "Apply assembly plus nil-term analysis to obtain concrete group-ring formulas.",
        "c": "Computation theorems for K/L-theory in hyperbolic and related classes.",
        "o": "Explicit algebraic targets for obstruction classes in selected cases.",
        "b": "Informative for U3-style coefficient behavior but not directly P7 lattice-specific.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d09",
        "title": "Assembly-Map Meta-Library",
        "paper": "arXiv:1805.00226",
        "gap_tags": ["FJ"],
        "bridge_status": "support",
        "q": "Choose safe assembly formulations and decorations for obstruction calculations.",
        "d": "Track model choices, decorations, and families in a unified assembly framework.",
        "m": "Map problem data to the appropriate assembly variant and comparison diagram.",
        "c": "Comprehensive assembly map taxonomy and compatibility results.",
        "o": "Reduced risk of decoration/family mismatches in L-theory arguments.",
        "b": "Quality-control module for S-branch rigor, not an existence theorem.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d10",
        "title": "FJ for Cocompact Lattices",
        "paper": "arXiv:1101.0469",
        "gap_tags": ["FJ"],
        "bridge_status": "direct",
        "q": "Establish FJ for the cocompact lattice class used in P7.",
        "d": "Identify Gamma as cocompact lattice in virtually connected Lie group.",
        "m": "Apply geometric-flow/control machinery to prove assembly isomorphism.",
        "c": "Main theorem covering K/L-theoretic FJ with coefficients for this class.",
        "o": "Valid reduction from L_n(Z[Gamma]) to equivariant homology of VCyc family.",
        "b": "Directly secures core FJ input in p7r-s3b.",
        "b_edge": "reform",
    },
    {
        "id": "p7g-d11",
        "title": "FJ Extension to Arbitrary Lattices",
        "paper": "arXiv:1401.0876",
        "gap_tags": ["FJ"],
        "bridge_status": "support",
        "q": "Ensure FJ remains available beyond cocompact specialization choices.",
        "d": "Embed lattice cases into arbitrary-lattice class in virtually connected Lie groups.",
        "m": "Transfer cocompact arguments via broader inheritance and reduction steps.",
        "c": "Main theorem for arbitrary lattices.",
        "o": "Robust fallback if lattice family variations occur in later revisions.",
        "b": "Stability module for scope changes; not itself a gap closer.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d12",
        "title": "Closed-Manifold vs Inertia Subgroup Interface",
        "paper": "arXiv:0905.0104",
        "gap_tags": ["G3"],
        "bridge_status": "partial",
        "q": "Relate algebraic surgery obstructions to realizability by closed manifolds.",
        "d": "Fix orientation character and compare closed-manifold and inertia subgroups in L-groups.",
        "m": "Use surgery exact-sequence interfaces and subgroup-identification theorems.",
        "c": "Theorems A/B relating these subgroups under hypotheses.",
        "o": "Criteria for when assembly/image elements correspond to closed-manifold realizations.",
        "b": "Potentially sharpens final S-branch closure after obstruction class is identified.",
        "b_edge": "reform",
    },
    {
        "id": "p7g-d13",
        "title": "Infinite Dihedral Surgery Obstruction Computations",
        "paper": "arXiv:math/0306054",
        "gap_tags": ["U3"],
        "bridge_status": "direct",
        "q": "Control UNil/torsion contributions in VCyc and dihedral pieces.",
        "d": "Model type-II VCyc subgroups through dihedral group-ring computations.",
        "m": "Compute surgery obstruction groups and identify torsion behavior.",
        "c": "Explicit calculation theorems for infinite-dihedral L-groups/UNil terms.",
        "o": "Evidence that problematic terms are 2-primary and vanish after rationalization.",
        "b": "Directly supports the E_Fin to E_VCyc rational comparison step.",
        "b_edge": "reform",
    },
    {
        "id": "p7g-d14",
        "title": "Modern Wall Finiteness Obstruction View",
        "paper": "arXiv:1707.07960",
        "gap_tags": ["G1"],
        "bridge_status": "support",
        "q": "Understand finiteness obstruction semantics behind FH(Q) style statements.",
        "d": "Represent finite-domination questions through Wall obstruction classes.",
        "m": "Analyze vanishing criteria and categorical interpretations of obstruction.",
        "c": "Survey-style synthesis of Wall obstruction machinery.",
        "o": "Improved conceptual control over what FH(Q) does and does not guarantee.",
        "b": "Supports precise gap wording for G1 without closing chain-level PD.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d15",
        "title": "Classical Wall Finiteness Obstruction Survey",
        "paper": "arXiv:math/0008070",
        "gap_tags": ["G1"],
        "bridge_status": "support",
        "q": "Track legacy finiteness-obstruction tools used in PD/FH transitions.",
        "d": "Package classical examples and obstruction calculations in Wall framework.",
        "m": "Map group/space data to obstruction-group elements.",
        "c": "Foundational survey references and computation templates.",
        "o": "Historical/technical base for careful finite-complex claims.",
        "b": "Background module; indirect value for G1 language hygiene.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d16",
        "title": "Equivariant Spivak Bundle and Surgery",
        "paper": "arXiv:1705.10909",
        "gap_tags": ["G2", "G3"],
        "bridge_status": "direct",
        "q": "Construct normal-map data equivariantly so cover restriction is controlled.",
        "d": "Work with equivariant Spivak normal bundle for group actions.",
        "m": "Lift/compare equivariant normal structures through surgery setup.",
        "c": "Equivariant Spivak and equivariant surgery theorems.",
        "o": "Potential direct route to solve G2 and enforce G3 compatibility.",
        "b": "Highest-value method candidate for unresolved normal-map/transfer gaps.",
        "b_edge": "reform",
    },
    {
        "id": "p7g-d17",
        "title": "Non-Simply-Connected Rational Homotopy Models",
        "paper": "arXiv:2304.00880",
        "gap_tags": ["G1", "G3"],
        "bridge_status": "partial",
        "q": "Justify rational-homotopy comparisons for spaces with nontrivial pi_1.",
        "d": "Represent spaces by rational models adapted to non-simply-connected setting.",
        "m": "Use algebraic models to compare rational homotopy types beyond nilpotent defaults.",
        "c": "Model-construction and comparison statements in non-simply-connected rational homotopy.",
        "o": "Candidate citation path for replacing hand-wavy rational-equivalence claims.",
        "b": "Promising for G3 caveat cleanup; must verify applicability hypotheses.",
        "b_edge": "reform",
    },
    {
        "id": "p7g-d18",
        "title": "Smith-Theory Constraints on Periodic Actions",
        "paper": "arXiv:1106.1704",
        "gap_tags": ["E2"],
        "bridge_status": "support",
        "q": "Filter impossible periodic-action scenarios in aspherical/lattice contexts.",
        "d": "Separate homotopically trivial periodic actions from geometric action data.",
        "m": "Apply Smith/L2-cohomology rigidity arguments.",
        "c": "No-homotopically-trivial-periodic-diffeomorphism theorems in specified settings.",
        "o": "Constraint layer preventing invalid fixed-point constructions.",
        "b": "Useful negative filter; not a positive constructor for P7.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d19",
        "title": "Reflections in Congruence Hyperbolic Manifolds",
        "paper": "arXiv:2506.23994",
        "gap_tags": ["E2"],
        "bridge_status": "direct",
        "q": "Produce explicit reflective involutions in cocompact arithmetic families.",
        "d": "Use arithmetic lattice with reflection and congruence-cover tower.",
        "m": "Induce manifold involutions with totally geodesic fixed hypersurfaces.",
        "c": "Theorem/remark package on reflective congruence manifolds and fixed sets.",
        "o": "Concrete E2 instantiation substrate for Z/2 action and fixed-set geometry.",
        "b": "Direct constructor feeding the successful p7r-s2b route.",
        "b_edge": "reform",
    },
    {
        "id": "p7g-d20",
        "title": "Finite Groups as Isometry Groups of Hyperbolic Manifolds",
        "paper": "arXiv:math/0406607",
        "gap_tags": ["E2"],
        "bridge_status": "partial",
        "q": "Realize prescribed finite symmetry groups in compact hyperbolic settings.",
        "d": "Construct finite-index subgroups with targeted normalizer quotient.",
        "m": "Combine subgroup growth and lattice-commensurator control.",
        "c": "Main realization theorem for finite groups as full isometry groups.",
        "o": "Existence of compact hyperbolic manifolds with chosen finite symmetry group.",
        "b": "Good symmetry-existence backup, but fixed-set Euler data not automatic.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d21",
        "title": "Finite Subgroup Counting in Lattices",
        "paper": "arXiv:1209.2484",
        "gap_tags": ["E2"],
        "bridge_status": "support",
        "q": "Control isotropy complexity in lattice/orbifold models.",
        "d": "Count maximal finite subgroups and isotropy classes versus covolume.",
        "m": "Use lattice geometry and subgroup-growth estimates.",
        "c": "Linear/sublinear bounds for finite subgroup conjugacy classes.",
        "o": "Quantitative isotropy control for families of orbifold quotients.",
        "b": "Auxiliary selection/filtering tool, not a direct P7 closure theorem.",
        "b_edge": "challenge",
    },
    {
        "id": "p7g-d22",
        "title": "Homology Bounds for Hyperbolic Orbifolds",
        "paper": "arXiv:2012.15322",
        "gap_tags": ["E2", "FJ"],
        "bridge_status": "support",
        "q": "Estimate orbifold homology growth in arithmetic/nonuniform families.",
        "d": "Build efficient simplicial thick-part models for orbifolds.",
        "m": "Derive linear-volume homology/torsion bounds from geometric models.",
        "c": "Theorems bounding Betti numbers and torsion parts by volume.",
        "o": "Practical constraints on candidate families for obstruction-space dimensions.",
        "b": "Useful quantitative side-information for strategy A/B computations.",
        "b_edge": "reform",
    },
]


def build_diagram(spec: dict) -> dict:
    pid = spec["id"]
    nodes = [
        {"id": f"{pid}-q", "node_type": "question", "body_text": spec["q"]},
        {"id": f"{pid}-d", "node_type": "answer", "body_text": spec["d"]},
        {"id": f"{pid}-m", "node_type": "answer", "body_text": spec["m"]},
        {"id": f"{pid}-c", "node_type": "comment", "body_text": spec["c"]},
        {"id": f"{pid}-o", "node_type": "answer", "body_text": spec["o"]},
        {"id": f"{pid}-b", "node_type": "comment", "body_text": spec["b"]},
    ]
    edges = [
        {"source": f"{pid}-d", "target": f"{pid}-q", "edge_type": "clarify"},
        {"source": f"{pid}-m", "target": f"{pid}-d", "edge_type": "assert"},
        {"source": f"{pid}-c", "target": f"{pid}-m", "edge_type": "reference"},
        {"source": f"{pid}-o", "target": f"{pid}-q", "edge_type": "assert"},
        {"source": f"{pid}-b", "target": f"{pid}-o", "edge_type": spec["b_edge"]},
    ]
    return {
        "id": spec["id"],
        "title": spec["title"],
        "paper": spec["paper"],
        "gap_tags": spec["gap_tags"],
        "bridge_status": spec["bridge_status"],
        "nodes": nodes,
        "edges": edges,
    }


def main() -> int:
    diagrams = [build_diagram(s) for s in DIAGRAM_SPECS]

    gap_counts: dict[str, int] = {}
    for d in diagrams:
        for g in d["gap_tags"]:
            gap_counts[g] = gap_counts.get(g, 0) + 1

    out = {
        "library_id": "problem7-gap-wiring-library",
        "date": "2026-02-12",
        "source_scope": "paper-mined wiring diagrams for Problem 7 gaps",
        "target_gaps": ["G1", "G2", "G3", "U1", "U2", "U3", "FJ", "E2"],
        "diagram_count": len(diagrams),
        "similarity_schema": SCHEMA,
        "gap_tag_counts": dict(sorted(gap_counts.items())),
        "diagrams": diagrams,
    }

    json_path = Path("data/first-proof/problem7-gap-wiring-library.json")
    md_path = Path("data/first-proof/problem7-gap-wiring-library.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Problem 7 Gap Wiring Library",
        "",
        "Date: 2026-02-12",
        "",
        "This file mines proof/method patterns into wiring diagrams for the current",
        "Problem 7 gaps (`G1`, `G2`, `G3`, `U1-U3`) plus dependency branches (`FJ`, `E2`).",
        "",
        f"Total diagrams: {len(diagrams)}",
        "",
        "## Shape Schema",
        "- `Q`: target subproblem",
        "- `D`: decomposition/data model",
        "- `M`: core mechanism",
        "- `C`: certificate theorem/tool",
        "- `O`: output statement",
        "- `B`: bridge status back to P7 gaps",
        "",
        "Canonical edges:",
        "- `D -> Q` (`clarify`)",
        "- `M -> D` (`assert`)",
        "- `C -> M` (`reference`)",
        "- `O -> Q` (`assert`)",
        "- `B -> O` (`reform` or `challenge`)",
        "",
        "## Gap Coverage",
    ]
    for k, v in sorted(gap_counts.items()):
        lines.append(f"- `{k}`: {v} diagrams")

    lines.append("")
    lines.append("## Diagram Index")
    lines.append("")
    for d in diagrams:
        tags = ", ".join(d["gap_tags"])
        lines.append(f"- `{d['id']}` — {d['title']} ({d['paper']})")
        lines.append(f"  tags: {tags}; bridge: {d['bridge_status']}")

    lines.append("")
    lines.append("## Diagrams")
    lines.append("")
    for d in diagrams:
        lines.append(f"### {d['id']}: {d['title']}")
        lines.append(f"- Paper: `{d['paper']}`")
        lines.append(f"- Gap tags: {', '.join(d['gap_tags'])}")
        lines.append(f"- Bridge status: `{d['bridge_status']}`")
        node_map = {n["id"].rsplit("-", 1)[-1]: n["body_text"] for n in d["nodes"]}
        lines.append(f"- `Q`: {node_map['q']}")
        lines.append(f"- `D`: {node_map['d']}")
        lines.append(f"- `M`: {node_map['m']}")
        lines.append(f"- `C`: {node_map['c']}")
        lines.append(f"- `O`: {node_map['o']}")
        lines.append(f"- `B`: {node_map['b']}")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"diagram_count={len(diagrams)}")
    for k, v in sorted(gap_counts.items()):
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
