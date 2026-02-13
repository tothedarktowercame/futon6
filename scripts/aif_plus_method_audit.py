#!/usr/bin/env python3
"""AIF+ method audit for proof-route trajectories (P4/P6).

Two modes are provided:
- pilot: lightweight route telemetry from the curated project-flow DAG
- full: evidence-backed audit for Problems 4 and 6 against I1-I6 and G5-G0
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class RouteSpec:
    key: str
    label: str
    node_ids: List[str]


@dataclass(frozen=True)
class EvidenceQuery:
    path: str
    patterns: List[str]
    max_hits: int = 2


@dataclass(frozen=True)
class CriterionSpec:
    cid: str
    title: str
    status: str
    assessment: str
    queries: List[EvidenceQuery]


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    title: str
    overall_status: str
    overall_assessment: str
    invariants: List[CriterionSpec]
    gates: List[CriterionSpec]
    gap_queries: List[EvidenceQuery]


@dataclass(frozen=True)
class EvidenceHit:
    path: str
    line: int
    text: str
    pattern: str


@dataclass(frozen=True)
class CriterionResult:
    cid: str
    title: str
    status: str
    assessment: str
    hits: List[EvidenceHit]


ROUTES: List[RouteSpec] = [
    RouteSpec(
        key="p6_a_only",
        label="P6 A-Only",
        node_ids=[
            "p6-dispatch",
            "p6-dir-a",
            "p6-a-stress",
            "p6-transfer-handoff",
        ],
    ),
    RouteSpec(
        key="p6_a_to_ef",
        label="P6 A->E+F",
        node_ids=[
            "p6-dispatch",
            "p6-dir-a",
            "p6-a-stress",
            "p6-transfer-handoff",
            "p6-dir-d",
            "p6-arnt",
            "p6-gl-balance",
            "p6-layer-switch",
            "p6-pigeonhole-artifacts",
            "p6-proof-draft",
            "p6-gap-honesty",
            "p6-amplification",
            "p6-coupling",
            "p6-architectural",
            "p6-e-f-hybrid",
        ],
    ),
    RouteSpec(
        key="p4_main",
        label="P4 Main Route",
        node_ids=[
            "p4-id-search",
            "p4-n3-proof",
            "p4-n4-cert",
        ],
    ),
]

OBSERVE_KW = (
    "probe",
    "search",
    "hunt",
    "diagnostic",
    "isolat",
    "stress",
    "audit",
    "fails",
    "quantif",
    "gap",
)
ACT_KW = (
    "proof",
    "switch",
    "reduction",
    "committed",
    "resolve",
    "close",
    "draft",
    "introduced",
    "package",
)
ARTIFACT_KW = (
    "artifact",
    "handoff",
    "draft",
    "package",
    "hardened",
    "tracked",
)
ARCH_KW = (
    "architectural",
    "layer switch",
    "hybrid",
    "bridge",
    "reframed",
)


def _node_text(node: Dict[str, object]) -> str:
    commit = node.get("commit", {})
    if not isinstance(commit, dict):
        commit = {}
    parts = [
        str(node.get("short_label", "")),
        str(node.get("significance", "")),
        str(commit.get("subject", "")),
    ]
    return " ".join(parts).lower()


def _has_any(text: str, keys: tuple[str, ...]) -> bool:
    return any(k in text for k in keys)


def _ratio(a: int, b: int) -> float:
    hi = max(a, b)
    if hi == 0:
        return 0.0
    return min(a, b) / hi


def _route_verdict(label: str, balance: float, artifacts: int, architecture: int) -> str:
    if label == "P6 A-Only":
        return "Limited: observation-heavy, weak closure"
    if label == "P6 A->E+F":
        if balance >= 0.55 and artifacts >= 3 and architecture >= 3:
            return "Improved: richer loop; still incomplete"
        return "Mixed: improvement signals but weak closure"
    if label == "P4 Main Route":
        return "Focused partial: closes n<=3, leaves cert branch"
    return "Unclassified"


def _route_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        raise SystemExit("invalid DAG json: nodes must be a list")
    by_id = {n.get("id"): n for n in nodes if isinstance(n, dict)}
    route_rows: List[Dict[str, object]] = []

    for spec in ROUTES:
        route_nodes = [by_id[rid] for rid in spec.node_ids if rid in by_id]
        observe = 0
        act = 0
        artifact = 0
        architecture = 0
        for node in route_nodes:
            text = _node_text(node)
            if _has_any(text, OBSERVE_KW):
                observe += 1
            if _has_any(text, ACT_KW):
                act += 1
            if _has_any(text, ARTIFACT_KW):
                artifact += 1
            if _has_any(text, ARCH_KW):
                architecture += 1
        balance = _ratio(observe, act)
        route_rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "steps": len(route_nodes),
                "observe": observe,
                "act": act,
                "artifact": artifact,
                "architecture": architecture,
                "balance": balance,
                "verdict": _route_verdict(spec.label, balance, artifact, architecture),
            }
        )
    return route_rows


def _render_pilot_markdown(payload: Dict[str, object], routes: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# AIF+ Pilot Method Audit (P4/P6)")
    lines.append("")
    lines.append("This is a heuristic route audit over commit-level evidence.")
    lines.append("It is **not** a full AIF runtime instrumentation.")
    lines.append("")
    lines.append(f"Source DAG timestamp: `{payload.get('generated', 'unknown')}`")
    lines.append("")
    lines.append("## Route Metrics")
    lines.append("")
    lines.append("| Route | Steps | Observe events | Act events | Balance (I2 proxy) | Artifact events | Architecture events | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in routes:
        lines.append(
            f"| {r['label']} | {r['steps']} | {r['observe']} | {r['act']} | {r['balance']:.2f} | "
            f"{r['artifact']} | {r['architecture']} | {r['verdict']} |"
        )
    lines.append("")
    lines.append("## Readout")
    lines.append("")
    lines.append("1. P6 A-only is constrained by a low observe/action balance and low closure evidence.")
    lines.append("2. P6 A->E+F shows a stronger loop profile (more architecture shifts and artifactization),")
    lines.append("but still ends in a partial state rather than a fully closed proof.")
    lines.append("3. P4 main route is sharp and short: strong local closure at n<=3,")
    lines.append("with an explicit remaining certification branch for n>=4.")
    lines.append("")
    lines.append("## Suggested Next Instrumentation")
    lines.append("")
    lines.append("1. Add event-level tags to proof episodes (observe/propose/act/verify/handoff).")
    lines.append("2. Record explicit invariant checks (I3/I4/I6) per episode, not just per commit summary.")
    lines.append("3. Re-run this audit after each major strategy pivot (especially on Problem 6).")
    lines.append("")
    return "\n".join(lines) + "\n"


def _q(path: str, *patterns: str, max_hits: int = 2) -> EvidenceQuery:
    return EvidenceQuery(path=path, patterns=list(patterns), max_hits=max_hits)


def _find_hits(repo_root: Path, query: EvidenceQuery) -> List[EvidenceHit]:
    abs_path = repo_root / query.path
    if not abs_path.exists():
        return []
    try:
        lines = abs_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    comp = [re.compile(p, flags=re.IGNORECASE) for p in query.patterns]
    hits: List[EvidenceHit] = []
    used = set()
    for lineno, line in enumerate(lines, start=1):
        for rgx in comp:
            if rgx.search(line):
                if lineno in used:
                    continue
                text = " ".join(line.strip().split())
                hits.append(
                    EvidenceHit(
                        path=query.path,
                        line=lineno,
                        text=text[:180],
                        pattern=rgx.pattern,
                    )
                )
                used.add(lineno)
                break
        if len(hits) >= query.max_hits:
            break
    return hits


def _audit_criterion(repo_root: Path, spec: CriterionSpec) -> CriterionResult:
    hits: List[EvidenceHit] = []
    for query in spec.queries:
        hits.extend(_find_hits(repo_root, query))
    status = spec.status
    if not hits:
        status = "INSUFFICIENT"
    return CriterionResult(
        cid=spec.cid,
        title=spec.title,
        status=status,
        assessment=spec.assessment,
        hits=hits,
    )


def _evidence_refs(hits: Iterable[EvidenceHit], max_refs: int = 3) -> str:
    refs: List[str] = []
    seen = set()
    for hit in hits:
        ref = f"`{hit.path}:{hit.line}`"
        if ref in seen:
            continue
        refs.append(ref)
        seen.add(ref)
        if len(refs) >= max_refs:
            break
    if not refs:
        return "none"
    return ", ".join(refs)


def _status_points(status: str) -> int:
    if status == "PASS":
        return 2
    if status == "PARTIAL":
        return 1
    return 0


def _problem_specs() -> List[ProblemSpec]:
    p4_invariants = [
        CriterionSpec(
            cid="I1",
            title="Boundary integrity",
            status="PASS",
            assessment="The theorem boundary and exact target inequality are explicit and stable.",
            queries=[
                _q("data/first-proof/problem4-solution.md", r"## Problem Statement", r"Question \(Spielman\)"),
                _q("data/first-proof/problem4-proof-strategy-skeleton.md", r"## Goal"),
            ],
        ),
        CriterionSpec(
            cid="I2",
            title="Observe/action asymmetry",
            status="PARTIAL",
            assessment="The route separates exploration from proof actions, but there is no episode-level instrumentation.",
            queries=[
                _q("data/first-proof/problem4-proof-strategy-skeleton.md", r"Verified Inputs We Can Use", r"Recommended Route"),
                _q("data/first-proof/project-flow-dag.md", r"P4 identity hunt", r"P4 n=3 proof"),
            ],
        ),
        CriterionSpec(
            cid="I3",
            title="Timescale separation",
            status="PASS",
            assessment="Fast checks and slow closure are separated: n<4 is closed while n>=4 remains explicitly open.",
            queries=[
                _q("data/first-proof/problem4-proof-strategy-skeleton.md", r"Status: n=2 proved .* n>=4 open"),
                _q("data/first-proof/problem4-solution.md", r"n = 2: equality", r"n >= 4: .* remains open"),
            ],
        ),
        CriterionSpec(
            cid="I4",
            title="Preference exogeneity",
            status="PASS",
            assessment="The writeup preserves target integrity under failed approaches instead of rewriting success criteria.",
            queries=[
                _q("data/first-proof/problem4-conditional-stam.md", r"Approach A killed", r"need alternative"),
                _q("data/first-proof/problem4-proof-strategy-skeleton.md", r"ELIMINATED", r"NEGATIVE"),
            ],
        ),
        CriterionSpec(
            cid="I5",
            title="Model adequacy",
            status="PARTIAL",
            assessment="Empirical support is strong, but n>=4 still lacks an unconditional theorem.",
            queries=[
                _q("data/first-proof/problem4-solution.md", r"8000\+ random trials", r"n >= 4 remains open"),
                _q("data/first-proof/problem4-ngeq4-research-brief.md", r"three structural obstacles", r"0/35K\+ tests"),
            ],
        ),
        CriterionSpec(
            cid="I6",
            title="Compositional closure",
            status="PARTIAL",
            assessment="The argument closes n=2 and n=3, but does not yet compose to n>=4.",
            queries=[
                _q("data/first-proof/problem4-solution.md", r"n = 3: \*\*PROVED\*\*", r"n >= 4: numerically verified"),
                _q("data/first-proof/problem4-proof-strategy-skeleton.md", r"Proof Summary", r"\| >=4 \| Open \|"),
            ],
        ),
    ]
    p4_gates = [
        CriterionSpec(
            cid="G5",
            title="Task specification",
            status="PASS",
            assessment="Problem statement, symbols, and success condition are explicit.",
            queries=[
                _q("data/first-proof/problem4-solution.md", r"## Problem Statement", r"## Answer"),
            ],
        ),
        CriterionSpec(
            cid="G4",
            title="Capability/assignment",
            status="PARTIAL",
            assessment="Model and run metadata exist for verifier runs, but formal role typing is lightweight.",
            queries=[
                _q("data/first-proof/problem4-lt4-verification-summary.md", r"Model:", r"Run metadata"),
            ],
        ),
        CriterionSpec(
            cid="G3",
            title="Pattern reference",
            status="PASS",
            assessment="Named strategy routes and route eliminations are documented.",
            queries=[
                _q("data/first-proof/problem4-proof-strategy-skeleton.md", r"Recommended Route", r"Secondary Route"),
            ],
        ),
        CriterionSpec(
            cid="G2",
            title="Execution",
            status="PASS",
            assessment="Execution artifacts include scripts and route-specific verification commands.",
            queries=[
                _q("data/first-proof/problem4-solution.md", r"Verification script"),
                _q("data/first-proof/problem4-proof-strategy-skeleton.md", r"Verification scripts"),
            ],
        ),
        CriterionSpec(
            cid="G1",
            title="Validation",
            status="PASS",
            assessment="Verifier outcomes and flagged gaps are explicitly recorded.",
            queries=[
                _q("data/first-proof/problem4-lt4-verification-summary.md", r"Headline result", r"Key flagged issues"),
            ],
        ),
        CriterionSpec(
            cid="G0",
            title="Evidence durability",
            status="PASS",
            assessment="Durable route artifacts and summary tables exist in canonical project files.",
            queries=[
                _q("data/first-proof/project-flow-dag.md", r"P4 Main Route", r"Pivotal Moments"),
                _q("data/first-proof/problem4-solution.md", r"## 7\. Summary and status"),
            ],
        ),
    ]
    p6_invariants = [
        CriterionSpec(
            cid="I1",
            title="Boundary integrity",
            status="PASS",
            assessment="The reduced theorem target and spectral normalization are explicit.",
            queries=[
                _q("data/first-proof/problem6-method-wiring-library.md", r"Reduced Problem", r"Input:", r"Output:"),
                _q("data/first-proof/problem6-solution.md", r"## 1\. Exact reformulation"),
            ],
        ),
        CriterionSpec(
            cid="I2",
            title="Observe/action asymmetry",
            status="PASS",
            assessment="Probe directions and constructive pivots are both present and separated.",
            queries=[
                _q("data/first-proof/problem6-gpl-h-attack-dispatch.md", r"Directions to explore", r"Direction A", r"Direction B", r"Direction C"),
                _q("data/first-proof/project-flow-dag.md", r"Direction A probe", r"Elementary layer switch"),
            ],
        ),
        CriterionSpec(
            cid="I3",
            title="Timescale separation",
            status="PASS",
            assessment="The route shows staged progression from diagnostics to architecture switch to draft and gap correction.",
            queries=[
                _q("data/first-proof/project-flow-dag.md", r"P6 A->E\+F Route", r"blocked->switch", r"Gap-status correction"),
            ],
        ),
        CriterionSpec(
            cid="I4",
            title="Preference exogeneity",
            status="PASS",
            assessment="Gap honesty and counterexamples are surfaced explicitly rather than hidden.",
            queries=[
                _q("data/first-proof/problem6-solution.md", r"ONE GAP", r"SINGLE remaining gap"),
                _q("data/first-proof/problem6-direction-e-f-proof.md", r"F-Lemma is false", r"counterexample"),
            ],
        ),
        CriterionSpec(
            cid="I5",
            title="Model adequacy",
            status="PARTIAL",
            assessment="Evidence is strong (440/440 numeric support), but a theorem-level general-graph bridge is still open.",
            queries=[
                _q("data/first-proof/problem6-solution.md", r"440/440", r"formal bound .* remains open"),
                _q("data/first-proof/problem6-gpl-h-attack-paths.md", r"Remaining formal gap"),
            ],
        ),
        CriterionSpec(
            cid="I6",
            title="Compositional closure",
            status="PARTIAL",
            assessment="K_n is closed with c=1/3; composition to arbitrary graphs is not yet fully discharged.",
            queries=[
                _q("data/first-proof/problem6-solution.md", r"K_n: PROVED", r"General graphs: ONE GAP"),
                _q("data/first-proof/problem6-direction-e-f-proof.md", r"dbar_all < 1 conjecture", r"OPEN"),
            ],
        ),
    ]
    p6_gates = [
        CriterionSpec(
            cid="G5",
            title="Task specification",
            status="PASS",
            assessment="The problem, constraints, and universal constant target are explicit.",
            queries=[
                _q("data/first-proof/problem6-solution.md", r"## Problem Statement", r"Question:"),
            ],
        ),
        CriterionSpec(
            cid="G4",
            title="Capability/assignment",
            status="PASS",
            assessment="Explicit student-dispatch and direction assignment are documented.",
            queries=[
                _q("data/first-proof/problem6-gpl-h-attack-dispatch.md", r"Pattern: agent/student-dispatch", r"From:", r"To:"),
            ],
        ),
        CriterionSpec(
            cid="G3",
            title="Pattern reference",
            status="PASS",
            assessment="Method library D1..D10 and bridge statuses are concretely mapped.",
            queries=[
                _q("data/first-proof/problem6-method-wiring-library.md", r"D1", r"D10", r"Bridge status"),
            ],
        ),
        CriterionSpec(
            cid="G2",
            title="Execution",
            status="PASS",
            assessment="Direction artifacts and proof drafts show active execution across routes.",
            queries=[
                _q("data/first-proof/problem6-direction-e-f-proof.md", r"Direction E\+F Hybrid Proof Draft"),
                _q("data/first-proof/project-flow-dag.md", r"Direction D probe", r"Near-final draft"),
            ],
        ),
        CriterionSpec(
            cid="G1",
            title="Validation",
            status="PARTIAL",
            assessment="Validation is extensive, but one theorem-level condition remains open.",
            queries=[
                _q("data/first-proof/problem6-solution.md", r"verified 440/440", r"Q-polynomial"),
                _q("data/first-proof/problem6-direction-e-f-proof.md", r"dbar_all < 1 conjecture", r"OPEN"),
            ],
        ),
        CriterionSpec(
            cid="G0",
            title="Evidence durability",
            status="PASS",
            assessment="DAG, route tables, and durable writeups preserve the full trajectory.",
            queries=[
                _q("data/first-proof/project-flow-dag.md", r"Generated:", r"Pivotal Moments"),
                _q("data/first-proof/problem6-solution.md", r"## 6\. Final conclusion"),
            ],
        ),
    ]
    p4_gap_queries = [
        _q("data/first-proof/problem4-solution.md", r"open", r"proof incomplete", r"conjectur", max_hits=8),
        _q("data/first-proof/problem4-proof-strategy-skeleton.md", r"open", r"ELIMINATED", r"fails", max_hits=8),
        _q("data/first-proof/problem4-ngeq4-research-brief.md", r"obstacle", r"open", r"fails", max_hits=8),
    ]
    p6_gap_queries = [
        _q("data/first-proof/problem6-solution.md", r"open", r"gap", r"conjecture", max_hits=10),
        _q("data/first-proof/problem6-direction-e-f-proof.md", r"OPEN", r"FALSE", r"counterexample", max_hits=8),
        _q("data/first-proof/problem6-gpl-h-attack-paths.md", r"Remaining formal gap", r"open", max_hits=8),
    ]
    return [
        ProblemSpec(
            key="p4",
            title="Problem 4 (Finite Free Convolution)",
            overall_status="PARTIAL",
            overall_assessment="n=2 and n=3 are closed; n>=4 remains open with strong evidence.",
            invariants=p4_invariants,
            gates=p4_gates,
            gap_queries=p4_gap_queries,
        ),
        ProblemSpec(
            key="p6",
            title="Problem 6 (Epsilon-Light Vertex Subsets)",
            overall_status="PARTIAL",
            overall_assessment="K_n is closed with c=1/3; general-graph closure has one explicit open bridge.",
            invariants=p6_invariants,
            gates=p6_gates,
            gap_queries=p6_gap_queries,
        ),
    ]


def _render_criterion_table(rows: List[CriterionResult]) -> List[str]:
    lines = [
        "| Check | Status | Assessment | Evidence |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.cid} {row.title} | {row.status} | {row.assessment} | {_evidence_refs(row.hits)} |"
        )
    return lines


def _render_gap_ledger(gaps: List[EvidenceHit], limit: int = 10) -> List[str]:
    lines: List[str] = []
    lines.append("| Evidence | Excerpt |")
    lines.append("|---|---|")
    for hit in gaps[:limit]:
        excerpt = hit.text.replace("|", "\\|")
        lines.append(f"| `{hit.path}:{hit.line}` | {excerpt} |")
    if not gaps:
        lines.append("| none | No explicit open-gap lines matched. |")
    return lines


def _render_full_markdown(
    repo_root: Path,
    payload: Dict[str, object],
    route_rows: List[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# AIF+ Full Method Audit (P4/P6)")
    lines.append("")
    lines.append("Scope: Problems 4 and 6 only.")
    lines.append("Definitions: I1-I6 from `chapter0-aif-as-wiring-diagram.md`; gates G5-G0 from `gate-pattern-mapping.md`.")
    lines.append("")
    lines.append(f"Source DAG timestamp: `{payload.get('generated', 'unknown')}`")
    lines.append("")
    lines.append("## Route Telemetry")
    lines.append("")
    lines.append("| Route | Steps | Observe events | Act events | Balance (I2 proxy) | Artifact events | Architecture events | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in route_rows:
        lines.append(
            f"| {r['label']} | {r['steps']} | {r['observe']} | {r['act']} | {r['balance']:.2f} | "
            f"{r['artifact']} | {r['architecture']} | {r['verdict']} |"
        )
    lines.append("")

    for problem in _problem_specs():
        inv_rows = [_audit_criterion(repo_root, c) for c in problem.invariants]
        gate_rows = [_audit_criterion(repo_root, c) for c in problem.gates]
        gap_hits: List[EvidenceHit] = []
        for q in problem.gap_queries:
            gap_hits.extend(_find_hits(repo_root, q))

        inv_points = sum(_status_points(r.status) for r in inv_rows)
        gate_points = sum(_status_points(r.status) for r in gate_rows)
        inv_max = 2 * len(inv_rows)
        gate_max = 2 * len(gate_rows)

        lines.append(f"## {problem.title}")
        lines.append("")
        lines.append(
            f"Overall verdict: **{problem.overall_status}** - {problem.overall_assessment}"
        )
        lines.append("")
        lines.append(
            f"Coverage score: invariants `{inv_points}/{inv_max}`, gates `{gate_points}/{gate_max}`."
        )
        lines.append("")
        lines.append("### Invariants (I1-I6)")
        lines.extend(_render_criterion_table(inv_rows))
        lines.append("")
        lines.append("### Gates (G5-G0)")
        lines.extend(_render_criterion_table(gate_rows))
        lines.append("")
        lines.append("### Open-Gap Ledger")
        lines.extend(_render_gap_ledger(gap_hits))
        lines.append("")

    lines.append("## Audit Conclusion")
    lines.append("")
    lines.append("1. Problem 4 is structurally well-audited and partially closed (n<=3 closed, n>=4 open).")
    lines.append("2. Problem 6 is structurally well-audited and partially closed (K_n closed, one general-graph bridge open).")
    lines.append("3. The P6 A->E+F route is a real improvement over A-only, but still not a full closure.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
        help="Audit mode (pilot route telemetry or full criterion audit).",
    )
    ap.add_argument(
        "--dag-json",
        default="data/first-proof/project-flow-dag.json",
        help="Path to project DAG json",
    )
    ap.add_argument(
        "--out-md",
        default=None,
        help="Path to output markdown report",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dag_path = (repo_root / args.dag_json).resolve()
    if not dag_path.exists():
        raise SystemExit(
            f"missing DAG json: {dag_path} (run scripts/generate-project-flow-dag.py first)"
        )
    payload = json.loads(dag_path.read_text(encoding="utf-8"))
    route_rows = _route_rows(payload)

    if args.out_md is not None:
        out_path = (repo_root / args.out_md).resolve()
    elif args.mode == "full":
        out_path = (repo_root / "data/first-proof/aif-plus-method-audit-p4-p6-full.md").resolve()
    else:
        out_path = (repo_root / "data/first-proof/aif-plus-method-audit-p4-p6.md").resolve()

    if args.mode == "pilot":
        out_text = _render_pilot_markdown(payload, route_rows)
    else:
        out_text = _render_full_markdown(repo_root, payload, route_rows)

    out_path.write_text(out_text, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
