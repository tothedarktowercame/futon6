#!/usr/bin/env python3
"""Consolidate expository-scope proposal votes into a minted vocabulary report.

The input proposal files are JSONL files named by proposing agent, with rows:

  {"paper","region_id","region_type","line","quote","kind","confidence",
   "source_class","new_subkind": null|{"name","parent","definition","why"}}

Mint policy is read from the seed hierarchy EDN.  The parser is intentionally
small and deterministic: it extracts only the policy threshold and seed kind
symbols needed for this consolidation report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLOSE_READING = ROOT / "holes" / "excursions" / "close-reading"
DEFAULT_PROPOSALS = DEFAULT_CLOSE_READING / "proposals"
DEFAULT_HIERARCHY = DEFAULT_CLOSE_READING / "expository-scope-hierarchy.edn"
DEFAULT_REPORT = DEFAULT_CLOSE_READING / "consolidation-report.json"
DEFAULT_SUMMARY = DEFAULT_CLOSE_READING / "consolidation-report.md"
DEFAULT_GH_ORDER = ROOT / "data" / "warp" / "gh200.txt"
EXTRACTOR = ROOT / "scripts" / "expository_region_extract.py"


@dataclass
class Proposal:
    row: dict[str, Any]
    agent: str
    file: str

    @property
    def paper(self) -> str:
        return str(self.row.get("paper") or "")

    @property
    def kind(self) -> str:
        return normalize_kind(str(self.row.get("kind") or ""))


@dataclass
class SuggestionGroup:
    parent: str
    norm_name: str
    names: Counter[str] = field(default_factory=Counter)
    definitions: Counter[str] = field(default_factory=Counter)
    whys: Counter[str] = field(default_factory=Counter)
    papers: set[str] = field(default_factory=set)
    agents: set[str] = field(default_factory=set)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def record(self, proposal: Proposal, suggestion: dict[str, Any]) -> None:
        name = str(suggestion.get("name") or self.norm_name)
        definition = str(suggestion.get("definition") or "").strip()
        why = str(suggestion.get("why") or "").strip()
        self.names[name] += 1
        if definition:
            self.definitions[definition] += 1
        if why:
            self.whys[why] += 1
        self.papers.add(proposal.paper)
        self.agents.add(proposal.agent)
        self.evidence.append(evidence_ref(proposal))

    @property
    def display_name(self) -> str:
        if self.names:
            return self.names.most_common(1)[0][0]
        return self.norm_name

    @property
    def minted_kind(self) -> str:
        base = self.norm_name
        parent = self.parent.rstrip("/")
        if "/" in base:
            return normalize_kind(base)
        return f"{parent}/{base}" if parent else base

    @property
    def definition(self) -> str:
        return self.definitions.most_common(1)[0][0] if self.definitions else ""

    @property
    def why(self) -> str:
        return self.whys.most_common(1)[0][0] if self.whys else ""


def normalize_kind(value: str) -> str:
    value = value.strip()
    if value.startswith(":"):
        value = value[1:]
    return value


def normalize_name(value: str) -> str:
    value = normalize_kind(value).lower()
    value = re.sub(r"[_\s]+", "-", value)
    value = re.sub(r"[^a-z0-9/-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-/")
    synonyms = {
        "motivation-rationale": "rationale",
        "motivational-rationale": "rationale",
        "why-this-exists": "rationale",
        "analogy-transfer": "transfer",
        "transfer-interpretation": "transfer",
        "literature-gap": "literature-gap",
        "prior-work-gap": "literature-gap",
        "open-question": "open-problem",
        "open-status": "open-problem",
    }
    return synonyms.get(value, value)


def agent_from_path(path: Path) -> str:
    suffix = ".proposals.jsonl"
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def parse_hierarchy(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    min_papers_match = re.search(r":min-papers\s+(\d+)", text)
    min_agents_match = re.search(r":min-agents\s+(\d+)", text)
    seed_kinds = sorted({normalize_kind(match) for match in re.findall(r":kind\s+:([^\s\]}]+)", text)})
    return {
        "path": str(path),
        "policy": {
            "mint_threshold": {
                "min_papers": int(min_papers_match.group(1)) if min_papers_match else 5,
                "min_agents": int(min_agents_match.group(1)) if min_agents_match else 2,
            },
            "fringe_resolution": "resolve-to-parent",
        },
        "seed_kinds": seed_kinds,
    }


def read_proposals(proposals_dir: Path) -> list[Proposal]:
    proposals: list[Proposal] = []
    if not proposals_dir.exists():
        return proposals
    for path in sorted(proposals_dir.glob("*.proposals.jsonl")):
        agent = agent_from_path(path)
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                proposals.append(Proposal(row=row, agent=agent, file=f"{path.name}:{lineno}"))
    return proposals


def evidence_ref(proposal: Proposal) -> dict[str, Any]:
    row = proposal.row
    return {
        "paper": proposal.paper,
        "region_id": row.get("region_id"),
        "line": row.get("line"),
        "quote": row.get("quote"),
        "agent": proposal.agent,
        "file": proposal.file,
    }


def tally_kind_votes(proposals: list[Proposal], sample_size: int = 5) -> dict[str, Any]:
    tallies: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        kind = proposal.kind
        if kind not in tallies:
            tallies[kind] = {
                "kind": kind,
                "votes": 0,
                "papers": set(),
                "agents": set(),
                "sample_evidence": [],
            }
        item = tallies[kind]
        item["votes"] += 1
        item["papers"].add(proposal.paper)
        item["agents"].add(proposal.agent)
        if len(item["sample_evidence"]) < sample_size:
            item["sample_evidence"].append(evidence_ref(proposal))

    normalized: dict[str, Any] = {}
    for kind, item in sorted(tallies.items()):
        normalized[kind] = {
            "kind": kind,
            "votes": item["votes"],
            "papers": sorted(item["papers"]),
            "n_papers": len(item["papers"]),
            "agents": sorted(item["agents"]),
            "n_agents": len(item["agents"]),
            "sample_evidence": item["sample_evidence"],
        }
    return normalized


def group_suggestions(proposals: list[Proposal]) -> dict[tuple[str, str], SuggestionGroup]:
    groups: dict[tuple[str, str], SuggestionGroup] = {}
    for proposal in proposals:
        suggestion = proposal.row.get("new_subkind")
        if not suggestion:
            continue
        parent = normalize_kind(str(suggestion.get("parent") or proposal.kind))
        norm_name = normalize_name(str(suggestion.get("name") or ""))
        if not parent or not norm_name:
            continue
        key = (parent, norm_name)
        groups.setdefault(key, SuggestionGroup(parent=parent, norm_name=norm_name))
        groups[key].record(proposal, suggestion)
    return groups


def classify_suggestions(
    groups: dict[tuple[str, str], SuggestionGroup], min_papers: int, min_agents: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    minted: list[dict[str, Any]] = []
    fringe: list[dict[str, Any]] = []
    for _, group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        base = {
            "name": group.display_name,
            "normalized_name": group.norm_name,
            "parent": group.parent,
            "n_papers": len(group.papers),
            "papers": sorted(group.papers),
            "n_agents": len(group.agents),
            "agents": sorted(group.agents),
            "votes": len(group.evidence),
            "definition": group.definition,
            "why": group.why,
            "sample_evidence": group.evidence[:5],
        }
        if len(group.papers) >= min_papers and len(group.agents) >= min_agents:
            minted.append({**base, "kind": group.minted_kind, "status": "minted"})
        else:
            fringe.append(
                {
                    **base,
                    "status": "fringe-resolved",
                    "resolved_kind": group.parent,
                    "resolution": "resolve-to-parent",
                }
            )
    return minted, fringe


def load_gh_order(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def proposal_paper_order(proposals: list[Proposal], gh_order: list[str]) -> list[str]:
    proposal_papers = {proposal.paper for proposal in proposals if proposal.paper}
    if gh_order:
        extras = sorted(proposal_papers.difference(gh_order))
        return gh_order + extras
    return sorted(proposal_papers)


def discovery_curve(
    groups: dict[tuple[str, str], SuggestionGroup],
    minted: list[dict[str, Any]],
    paper_order: list[str],
    min_papers: int,
    min_agents: int,
) -> list[dict[str, Any]]:
    minted_keys = {(item["parent"], item["normalized_name"]) for item in minted}
    discovered: set[tuple[str, str]] = set()
    seen_papers: set[str] = set()
    curve: list[dict[str, Any]] = []
    for index, paper in enumerate(paper_order, 1):
        seen_papers.add(paper)
        for key, group in groups.items():
            if key not in minted_keys or key in discovered:
                continue
            visible = [ev for ev in group.evidence if ev["paper"] in seen_papers]
            visible_papers = {ev["paper"] for ev in visible}
            visible_agents = {ev["agent"] for ev in visible}
            if len(visible_papers) >= min_papers and len(visible_agents) >= min_agents:
                discovered.add(key)
        curve.append(
            {
                "paper_index": index,
                "paper": paper,
                "n_minted": len(discovered),
            }
        )
    return curve


def load_extractor_module() -> Any:
    spec = importlib.util.spec_from_file_location("expository_region_extract", EXTRACTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import extractor from {EXTRACTOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def count_sentences(text: str) -> int:
    scrubbed = re.sub(r"%.*", "", text)
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    if not scrubbed:
        return 0
    matches = re.findall(r"[.!?](?=(?:[`'\"}\])]*\s+|[`'\"}\])]*$))", scrubbed)
    return max(1, len(matches))


def expository_sentence_count(extractor: Any, paper: str) -> int | None:
    try:
        entity_id, text = extractor.load_text(paper)
        result = extractor.extract_regions(entity_id, text)
    except Exception:
        return None
    return sum(count_sentences(region.get("text", "")) for region in result.get("regions", []))


def coverage_report(proposals: list[Proposal]) -> dict[str, Any]:
    proposals_by_paper = Counter(proposal.paper for proposal in proposals if proposal.paper)
    if not proposals_by_paper:
        return {
            "overall": {"proposals": 0, "expository_sentences": 0, "pct": 0.0},
            "per_paper": {},
            "missing_denominator_papers": [],
        }
    extractor = load_extractor_module()
    per_paper: dict[str, Any] = {}
    missing: list[str] = []
    total_proposals = 0
    total_sentences = 0
    for paper in sorted(proposals_by_paper):
        proposal_count = proposals_by_paper[paper]
        sentence_count = expository_sentence_count(extractor, paper)
        total_proposals += proposal_count
        if sentence_count is None:
            missing.append(paper)
            per_paper[paper] = {
                "proposals": proposal_count,
                "expository_sentences": None,
                "pct": None,
            }
            continue
        total_sentences += sentence_count
        per_paper[paper] = {
            "proposals": proposal_count,
            "expository_sentences": sentence_count,
            "pct": round((proposal_count / sentence_count * 100.0) if sentence_count else 0.0, 2),
        }
    return {
        "overall": {
            "proposals": total_proposals,
            "expository_sentences": total_sentences,
            "pct": round((total_proposals / total_sentences * 100.0) if total_sentences else 0.0, 2),
        },
        "per_paper": per_paper,
        "missing_denominator_papers": missing,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    hierarchy = parse_hierarchy(args.hierarchy)
    threshold = hierarchy["policy"]["mint_threshold"]
    proposals = read_proposals(args.proposals_dir)
    kind_votes = tally_kind_votes(proposals)
    suggestion_groups = group_suggestions(proposals)
    minted, fringe = classify_suggestions(
        suggestion_groups, threshold["min_papers"], threshold["min_agents"]
    )
    gh_order = load_gh_order(args.gh_order)
    paper_order = proposal_paper_order(proposals, gh_order)
    curve = discovery_curve(
        suggestion_groups,
        minted,
        paper_order,
        threshold["min_papers"],
        threshold["min_agents"],
    )
    coverage = coverage_report(proposals)
    sufficient_scope_set = sorted(set(hierarchy["seed_kinds"]) | {item["kind"] for item in minted})
    return {
        "inputs": {
            "proposals_dir": str(args.proposals_dir),
            "hierarchy": str(args.hierarchy),
            "gh_order": str(args.gh_order),
        },
        "policy": hierarchy["policy"],
        "proposal_count": len(proposals),
        "paper_count": len({proposal.paper for proposal in proposals if proposal.paper}),
        "agent_count": len({proposal.agent for proposal in proposals}),
        "minted_hierarchy": {
            "seed_kinds": hierarchy["seed_kinds"],
            "newly_minted": minted,
            "sufficient_scope_set": sufficient_scope_set,
        },
        "per_kind_votes": kind_votes,
        "fringe_resolved": fringe,
        "discovery_curve": curve,
        "coverage": coverage,
    }


def write_summary(report: dict[str, Any], path: Path) -> None:
    minted = report["minted_hierarchy"]["newly_minted"]
    fringe = report["fringe_resolved"]
    lines = [
        "# Expository Scope Consolidation",
        "",
        f"Proposals: {report['proposal_count']}",
        f"Papers: {report['paper_count']}",
        f"Agents: {report['agent_count']}",
        f"Mint threshold: {report['policy']['mint_threshold']}",
        "",
        "## Minted Kinds",
    ]
    if minted:
        for item in minted:
            lines.append(
                f"- `{item['kind']}` from `{item['parent']}`: "
                f"{item['n_papers']} papers, {item['n_agents']} agents, {item['votes']} votes"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Fringe Resolutions"])
    if fringe:
        for item in fringe:
            lines.append(
                f"- `{item['normalized_name']}` -> `{item['resolved_kind']}`: "
                f"{item['n_papers']} papers, {item['n_agents']} agents, {item['votes']} votes"
            )
    else:
        lines.append("- None")
    coverage = report["coverage"]["overall"]
    lines.extend(
        [
            "",
            "## Coverage",
            f"Overall: {coverage['proposals']} proposals / "
            f"{coverage['expository_sentences']} expository sentences = {coverage['pct']}%",
            "",
            "## Discovery Curve",
        ]
    )
    if report["discovery_curve"]:
        last = report["discovery_curve"][-1]
        lines.append(
            f"Final: paper_index={last['paper_index']}, paper={last['paper']}, n_minted={last['n_minted']}"
        )
    else:
        lines.append("No proposal papers; curve is empty.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def create_synthetic_fixture(root: Path) -> Path:
    proposals_dir = root / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    papers = ["0905.0595", "0711.1761", "0807.1872", "1012.1220", "0801.2567"]
    shared_subkind = {
        "name": "Bridge Analogy",
        "parent": "connection",
        "definition": "Transfers a construction or claim across an analogy between settings.",
        "why": "Synthetic repeated suggestion for mint-threshold validation.",
    }
    fringe_subkind = {
        "name": "One-Off Aside",
        "parent": "rationale/telos",
        "definition": "A one-paper aside that should resolve to its parent.",
        "why": "Synthetic fringe validation.",
    }
    agent_a: list[dict[str, Any]] = []
    agent_b: list[dict[str, Any]] = []
    for idx, paper in enumerate(papers):
        row = {
            "paper": paper,
            "region_id": f"{paper}-synthetic-{idx}",
            "region_type": "inflight",
            "line": 100 + idx,
            "quote": f"Synthetic bridge analogy evidence {idx}.",
            "kind": "connection",
            "confidence": 0.93,
            "source_class": "PROSE",
            "new_subkind": shared_subkind,
        }
        (agent_a if idx < 3 else agent_b).append(row)
    agent_a.append(
        {
            "paper": papers[0],
            "region_id": "synthetic-fringe",
            "region_type": "inflight",
            "line": 220,
            "quote": "Synthetic one-off aside evidence.",
            "kind": "rationale/telos",
            "confidence": 0.72,
            "source_class": "PROSE",
            "new_subkind": fringe_subkind,
        }
    )
    write_jsonl(proposals_dir / "codex-3.proposals.jsonl", agent_a)
    write_jsonl(proposals_dir / "claude-3.proposals.jsonl", agent_b)
    return proposals_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals-dir", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--gh-order", type=Path, default=DEFAULT_GH_ORDER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--stdout", action="store_true", help="also print the JSON report")
    parser.add_argument("--self-test", action="store_true", help="run a synthetic mint/fringe fixture")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        with tempfile.TemporaryDirectory(prefix="scope-votes-") as tmp:
            args.proposals_dir = create_synthetic_fixture(Path(tmp))
            report = build_report(args)
            json.dump(report, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        return 0
    report = build_report(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(report, args.summary)
    if args.stdout:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
