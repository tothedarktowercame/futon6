#!/usr/bin/env python3
"""Prepare a hotspot-only proof stepper package for a chosen problem/commit.

Outputs:
- reconstructed snapshot files from git commit (solution, wiring, mermaid files)
- hotspot node list from accumulated codex result JSONLs
- hotspot-only wiring subgraph
- stepper manifest (baseline = legacy output, intervention = hotspot-only run)

This lets you hop to a proof at any commit and run only hotspot nodes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"verified", "plausible", "gap", "error"}


@dataclass
class HotspotNode:
    node_id: str
    observations: int
    verified: int
    plausible: int
    gap: int
    error: int
    parse: int


def run_git(repo_root: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def git_show_file(repo_root: Path, commit: str, rel_path: str) -> str:
    return run_git(repo_root, ["show", f"{commit}:{rel_path}"])


def git_file_exists(repo_root: Path, commit: str, rel_path: str) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{rel_path}"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0


def git_list_paths(repo_root: Path, commit: str, prefix: str) -> list[str]:
    out = run_git(repo_root, ["ls-tree", "-r", "--name-only", commit, prefix])
    return [line.strip() for line in out.splitlines() if line.strip()]


def classify_status(rec: dict[str, Any]) -> str:
    st = rec.get("claim_verified")
    if st in VALID_STATUSES:
        return str(st)
    if rec.get("parse_error"):
        return "parse"
    return "parse"


def collect_hotspots(
    data_dir: Path,
    problem: int,
    min_observations: int,
    unresolved_threshold: float,
) -> tuple[list[HotspotNode], list[Path]]:
    patterns = [
        f"problem{problem}-codex-results*.jsonl",
        f"problem{problem}*codex-results*.jsonl",
    ]
    files_set: set[Path] = set()
    for pat in patterns:
        files_set.update(data_dir.glob(pat))
    files = sorted(files_set)

    per_node: dict[str, Counter[str]] = defaultdict(Counter)
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                node_id = rec.get("node_id")
                if not isinstance(node_id, str) or not node_id:
                    continue
                st = classify_status(rec)
                per_node[node_id][st] += 1

    hotspots: list[HotspotNode] = []
    for node_id, c in sorted(per_node.items()):
        obs = sum(c.values())
        if obs < min_observations:
            continue
        unresolved = c["plausible"] + c["gap"] + c["error"] + c["parse"]
        unresolved_rate = unresolved / obs if obs else 0.0

        is_hot = (c["verified"] == 0) or (unresolved_rate >= unresolved_threshold)
        if not is_hot:
            continue

        hotspots.append(
            HotspotNode(
                node_id=node_id,
                observations=obs,
                verified=c["verified"],
                plausible=c["plausible"],
                gap=c["gap"],
                error=c["error"],
                parse=c["parse"],
            )
        )

    return hotspots, files


def build_hotspot_wiring(wiring: dict[str, Any], hotspot_ids: set[str]) -> dict[str, Any]:
    nodes = [n for n in wiring.get("nodes", []) if n.get("id") in hotspot_ids]
    kept = {n.get("id") for n in nodes if isinstance(n.get("id"), str)}
    edges = [
        e
        for e in wiring.get("edges", [])
        if e.get("source") in kept and e.get("target") in kept
    ]

    edge_types: Counter[str] = Counter()
    for e in edges:
        t = e.get("edge_type")
        if isinstance(t, str) and t:
            edge_types[t] += 1

    out = dict(wiring)
    out["nodes"] = nodes
    out["edges"] = edges
    out["stats"] = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "edge_types": dict(edge_types),
    }
    out["hotspot_subgraph"] = True
    out["generated_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def infer_runner(repo_root: Path, problem: int) -> Path:
    candidates = []
    if problem in {6, 10}:
        candidates.append(repo_root / "scripts" / "run-proof-polish-codex.py")
    candidates.append(repo_root / "scripts" / f"run-proof-polish-codex-p{problem}.py")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No runner script found for problem {problem}")


def choose_legacy_baseline(data_dir: Path, problem: int) -> Path:
    preferred = data_dir / f"problem{problem}-codex-results.jsonl"
    if preferred.exists():
        return preferred
    files = sorted(set(data_dir.glob(f"problem{problem}-codex-results*.jsonl")) | set(data_dir.glob(f"problem{problem}*codex-results*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No existing codex result files found for problem {problem}")
    return files[0]


def rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", type=int, required=True, help="Problem number (e.g., 3, 7)")
    ap.add_argument("--commit", default="HEAD", help="Git commit-ish to reconstruct from")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/first-proof"),
        help="First-proof data directory",
    )
    ap.add_argument(
        "--stepper-dir",
        type=Path,
        default=Path("data/first-proof/stepper"),
        help="Stepper metadata directory",
    )
    ap.add_argument("--runner", type=Path, default=None, help="Optional explicit runner script")
    ap.add_argument("--min-observations", type=int, default=3)
    ap.add_argument(
        "--unresolved-threshold",
        type=float,
        default=0.8,
        help="Mark node hotspot if unresolved_rate >= threshold",
    )
    ap.add_argument(
        "--extra-runner-arg",
        action="append",
        default=[],
        help="Extra arg(s) appended to intervention command (repeatable)",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Set manifest run.execute=true (default false for safe prep)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir if args.data_dir.is_absolute() else (repo_root / args.data_dir)
    stepper_dir = args.stepper_dir if args.stepper_dir.is_absolute() else (repo_root / args.stepper_dir)
    stepper_dir.mkdir(parents=True, exist_ok=True)

    commit_full = run_git(repo_root, ["rev-parse", args.commit]).strip()
    commit_short = run_git(repo_root, ["rev-parse", "--short", commit_full]).strip()

    problem = args.problem
    solution_rel = f"data/first-proof/problem{problem}-solution.md"
    wiring_rel = f"data/first-proof/problem{problem}-wiring.json"

    if not git_file_exists(repo_root, commit_full, solution_rel):
        raise FileNotFoundError(f"{solution_rel} not found at commit {commit_full}")
    if not git_file_exists(repo_root, commit_full, wiring_rel):
        raise FileNotFoundError(f"{wiring_rel} not found at commit {commit_full}")

    snap_dir = stepper_dir / "snapshots" / f"problem{problem}" / commit_short
    snap_dir.mkdir(parents=True, exist_ok=True)

    solution_path = snap_dir / f"problem{problem}-solution.md"
    wiring_path = snap_dir / f"problem{problem}-wiring.json"
    solution_path.write_text(git_show_file(repo_root, commit_full, solution_rel), encoding="utf-8")
    wiring_path.write_text(git_show_file(repo_root, commit_full, wiring_rel), encoding="utf-8")

    # Reconstruct any mermaid versions present at this commit.
    all_paths = git_list_paths(repo_root, commit_full, "data/first-proof")
    mmd_paths = [p for p in all_paths if p.startswith(f"data/first-proof/problem{problem}") and p.endswith(".mmd")]
    mmd_written: list[str] = []
    for rel_mmd in mmd_paths:
        outp = snap_dir / Path(rel_mmd).name
        outp.write_text(git_show_file(repo_root, commit_full, rel_mmd), encoding="utf-8")
        mmd_written.append(rel_mmd)

    wiring = json.loads(wiring_path.read_text(encoding="utf-8"))
    hotspots, result_files = collect_hotspots(
        data_dir=data_dir,
        problem=problem,
        min_observations=args.min_observations,
        unresolved_threshold=args.unresolved_threshold,
    )

    hotspot_ids = {h.node_id for h in hotspots}
    hotspot_wiring = build_hotspot_wiring(wiring, hotspot_ids)
    hotspot_wiring_path = snap_dir / f"problem{problem}-hotspot-wiring.json"
    hotspot_wiring_path.write_text(json.dumps(hotspot_wiring, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    hotspots_json_path = stepper_dir / f"problem{problem}-hotspots-{commit_short}.json"
    hotspots_md_path = stepper_dir / f"problem{problem}-hotspots-{commit_short}.md"

    hotspots_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "problem": problem,
        "commit": commit_full,
        "snapshot_dir": rel(repo_root, snap_dir),
        "inputs": {
            "min_observations": args.min_observations,
            "unresolved_threshold": args.unresolved_threshold,
            "result_files": [rel(repo_root, p) for p in result_files],
        },
        "hotspots": [h.__dict__ for h in hotspots],
        "hotspot_node_ids": sorted(hotspot_ids),
        "hotspot_wiring": rel(repo_root, hotspot_wiring_path),
        "mermaid_files": mmd_written,
    }
    hotspots_json_path.write_text(json.dumps(hotspots_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# Problem {problem} Hotspots @ {commit_short}",
        "",
        f"- commit: `{commit_full}`",
        f"- snapshot dir: `{rel(repo_root, snap_dir)}`",
        f"- hotspot wiring: `{rel(repo_root, hotspot_wiring_path)}`",
        f"- source result files: `{len(result_files)}`",
        "",
        "| Node | Obs | Verified | Plausible | Gap | Error | Parse | Unresolved % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in hotspots:
        unresolved = h.plausible + h.gap + h.error + h.parse
        pct = (100.0 * unresolved / h.observations) if h.observations else 0.0
        md_lines.append(
            f"| {h.node_id} | {h.observations} | {h.verified} | {h.plausible} | {h.gap} | {h.error} | {h.parse} | {pct:.1f}% |"
        )
    if not hotspots:
        md_lines.append("| (none) | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% |")
    md_lines.append("")
    hotspots_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    runner = args.runner if args.runner is not None else infer_runner(repo_root, problem)
    if not runner.is_absolute():
        runner = (repo_root / runner)

    baseline_out = choose_legacy_baseline(data_dir, problem)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    intervention_out = data_dir / f"problem{problem}-codex-results-stepper-hotspots-{commit_short}-{run_stamp}.jsonl"
    intervention_prompts = data_dir / f"problem{problem}-codex-prompts-stepper-hotspots-{commit_short}-{run_stamp}.jsonl"

    intervention_cmd: list[str] = [
        "python3",
        rel(repo_root, runner),
        "--wiring",
        rel(repo_root, hotspot_wiring_path),
        "--solution",
        rel(repo_root, solution_path),
        "--output",
        rel(repo_root, intervention_out),
        "--prompts-out",
        rel(repo_root, intervention_prompts),
    ]
    for extra in args.extra_runner_arg:
        intervention_cmd.append(extra)

    interventions: list[dict[str, Any]] = []
    if hotspots:
        interventions.append(
            {
                "label": "hotspot-only",
                "command": intervention_cmd,
                "output_jsonl": rel(repo_root, intervention_out),
                "notes": "Run only hotspot nodes via reconstructed commit snapshot + hotspot subgraph wiring.",
                "hypothesis": "Hotspot-only run should yield faster learning on unresolved nodes than broad reruns.",
            }
        )

    manifest = {
        "experiment_id": f"p{problem}-hotspots-{commit_short}",
        "wiring": rel(repo_root, hotspot_wiring_path),
        "compare_script": "scripts/compare-proof-polish-arms.py",
        "run": {
            "execute": bool(args.execute),
            "stop_on_error": True,
        },
        "baseline": {
            "label": "legacy-baseline",
            "command": None,
            "output_jsonl": rel(repo_root, baseline_out),
            "notes": "Unchanged legacy run artifact; used as control.",
            "hypothesis": "Baseline preserves historical behavior without new interventions.",
        },
        "interventions": interventions,
        "learning_log": {
            "decision_rule": "Adopt hotspot-only step if unresolved hotspot nodes improve without regressions on node-level quality.",
            "questions": [
                "Which hotspot node statuses changed (gap/plausible -> verified)?",
                "Did hotspot-only mode reduce latency/retry overhead versus broad runs?",
                "What new dependency (theorem/citation/assumption) was identified per remaining hotspot?",
            ],
        },
    }

    manifest_path = stepper_dir / f"problem{problem}-hotspot-stepper-{commit_short}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"prepared problem {problem} hotspot stepper package")
    print(f"- commit: {commit_full}")
    print(f"- snapshot: {rel(repo_root, snap_dir)}")
    print(f"- hotspots json: {rel(repo_root, hotspots_json_path)}")
    print(f"- hotspots md: {rel(repo_root, hotspots_md_path)}")
    print(f"- manifest: {rel(repo_root, manifest_path)}")
    print("next:")
    print(f"  python3 scripts/run-proof-stepper.py --manifest {rel(repo_root, manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
