#!/usr/bin/env python3
"""Run a proof stepper experiment (baseline + interventions) and record learning.

Manifest-driven workflow:
- run baseline and intervention commands (optional)
- summarize each arm from output JSONL (latest row per node)
- compare each intervention against baseline via compare-proof-polish-arms.py
- emit run artifacts + structured learning record
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"verified", "plausible", "gap", "error"}
SCORE_MAP = {"verified": 2, "plausible": 1, "gap": 0, "error": 0, "parse": 0}


@dataclass
class ArmRun:
    label: str
    output_jsonl: Path
    command: list[str] | None
    enabled: bool
    notes: str
    hypothesis: str


@dataclass
class ArmSummary:
    label: str
    rows: int
    verified: int
    plausible: int
    gap: int
    error: int
    parse: int
    timed_out: int
    retries_gt1: int
    score_sum: int
    score_avg: float


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_label(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", label).strip("-") or "arm"


def abs_path(repo_root: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    return repo_root / p


def parse_command(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return shlex.split(raw)
    raise ValueError("command must be string or list")


def load_latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            node_id = rec.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError(f"{path}:{lineno}: missing/invalid node_id")
            rows[node_id] = rec
    return rows


def rec_status(rec: dict[str, Any]) -> str:
    st = rec.get("claim_verified")
    if st in VALID_STATUSES:
        return str(st)
    if rec.get("parse_error"):
        return "parse"
    return "parse"


def summarize_output(label: str, output_jsonl: Path) -> ArmSummary:
    latest = load_latest_rows(output_jsonl)
    counts = {k: 0 for k in ("verified", "plausible", "gap", "error", "parse")}
    retries_gt1 = 0
    timed_out = 0
    score_sum = 0

    for rec in latest.values():
        st = rec_status(rec)
        counts[st] += 1
        score_sum += SCORE_MAP[st]
        attempts = rec.get("attempts", 1)
        if isinstance(attempts, int) and attempts > 1:
            retries_gt1 += 1
        if bool(rec.get("timed_out", False)):
            timed_out += 1

    rows = len(latest)
    score_avg = (score_sum / rows) if rows else 0.0
    return ArmSummary(
        label=label,
        rows=rows,
        verified=counts["verified"],
        plausible=counts["plausible"],
        gap=counts["gap"],
        error=counts["error"],
        parse=counts["parse"],
        timed_out=timed_out,
        retries_gt1=retries_gt1,
        score_sum=score_sum,
        score_avg=score_avg,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "baseline" not in payload:
        raise ValueError("manifest missing 'baseline'")
    if "interventions" not in payload or not isinstance(payload["interventions"], list):
        raise ValueError("manifest missing 'interventions' list")
    return payload


def parse_arm(repo_root: Path, arm: dict[str, Any]) -> ArmRun:
    label = str(arm.get("label", "")).strip()
    if not label:
        raise ValueError("arm missing label")
    output_raw = arm.get("output_jsonl")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise ValueError(f"arm '{label}' missing output_jsonl")

    command = parse_command(arm.get("command"))
    enabled = bool(arm.get("enabled", True))
    notes = str(arm.get("notes", "")).strip()
    hypothesis = str(arm.get("hypothesis", "")).strip()

    output_jsonl = abs_path(repo_root, output_raw)
    assert output_jsonl is not None

    return ArmRun(
        label=label,
        output_jsonl=output_jsonl,
        command=command,
        enabled=enabled,
        notes=notes,
        hypothesis=hypothesis,
    )


def run_arm(repo_root: Path, arm: ArmRun, run_dir: Path, execute: bool) -> dict[str, Any]:
    stamp = sanitize_label(arm.label)
    log_path = run_dir / f"{stamp}.log"
    start = now_utc()
    exit_code = None
    ran = False
    skipped_reason = None

    if not arm.enabled:
        skipped_reason = "disabled"
    elif not execute:
        skipped_reason = "execution-disabled"
    elif arm.command is None:
        skipped_reason = "no-command"
    else:
        ran = True
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"# start {start}\n")
            logf.write("# cmd " + json.dumps(arm.command, ensure_ascii=False) + "\n\n")
            proc = subprocess.run(
                arm.command,
                cwd=str(repo_root),
                text=True,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            exit_code = proc.returncode
            logf.write(f"\n# exit_code {exit_code}\n")

    end = now_utc()

    summary = None
    summary_error = None
    if arm.output_jsonl.exists():
        try:
            summary = summarize_output(arm.label, arm.output_jsonl)
        except Exception as exc:
            summary_error = str(exc)
    else:
        summary_error = f"output file not found: {arm.output_jsonl}"

    return {
        "label": arm.label,
        "output_jsonl": str(arm.output_jsonl),
        "command": arm.command,
        "ran": ran,
        "enabled": arm.enabled,
        "skipped_reason": skipped_reason,
        "start_utc": start,
        "end_utc": end,
        "exit_code": exit_code,
        "log": str(log_path) if ran else None,
        "notes": arm.notes,
        "hypothesis": arm.hypothesis,
        "summary": summary.__dict__ if summary else None,
        "summary_error": summary_error,
    }


def compare_arms(
    repo_root: Path,
    run_dir: Path,
    compare_script: Path,
    wiring: Path | None,
    baseline: ArmRun,
    candidate: ArmRun,
) -> dict[str, Any]:
    ctag = sanitize_label(candidate.label)
    md_out = run_dir / f"compare-{ctag}.md"
    json_out = run_dir / f"compare-{ctag}.json"

    cmd = [
        "python3",
        str(compare_script),
        "--baseline",
        str(baseline.output_jsonl),
        "--candidate",
        str(candidate.output_jsonl),
        "--baseline-label",
        baseline.label,
        "--candidate-label",
        candidate.label,
        "--output-md",
        str(md_out),
        "--output-json",
        str(json_out),
    ]
    if wiring is not None:
        cmd.extend(["--wiring", str(wiring)])

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    payload = None
    err = None
    if proc.returncode == 0 and json_out.exists():
        payload = json.loads(json_out.read_text(encoding="utf-8"))
    else:
        err = (proc.stderr or proc.stdout or "comparison failed").strip()

    return {
        "candidate_label": candidate.label,
        "command": cmd,
        "exit_code": proc.returncode,
        "output_md": str(md_out),
        "output_json": str(json_out),
        "payload": payload,
        "error": err,
    }


def render_summary_md(
    experiment_id: str,
    run_dir: Path,
    baseline_result: dict[str, Any],
    intervention_results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    learning_prompts: list[str],
    decision_rule: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Proof Stepper Run: {experiment_id}")
    lines.append("")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")

    lines.append("## Arm Outcomes")
    lines.append("")
    lines.append("| Arm | Rows | Verified | Plausible | Gap | Error | Parse | Timed Out | Retries>1 | Score Avg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for rec in [baseline_result] + intervention_results:
        s = rec.get("summary")
        if not s:
            lines.append(f"| {rec['label']} | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {rec['label']} | {s['rows']} | {s['verified']} | {s['plausible']} | {s['gap']} | "
            f"{s['error']} | {s['parse']} | {s['timed_out']} | {s['retries_gt1']} | {s['score_avg']:.3f} |"
        )
    lines.append("")

    lines.append("## Baseline Comparisons")
    lines.append("")
    lines.append("| Candidate | Better | Worse | Tie | Common Nodes | Score Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for comp in comparisons:
        payload = comp.get("payload")
        if not payload:
            lines.append(f"| {comp['candidate_label']} | - | - | - | - | error |")
            continue
        pair = payload["pairwise"]
        base = payload["baseline"]
        cand = payload["candidate"]
        delta = float(cand["score_avg"]) - float(base["score_avg"])
        lines.append(
            f"| {comp['candidate_label']} | {pair['candidate_better']} | {pair['candidate_worse']} | "
            f"{pair['tie']} | {pair['common_nodes']} | {delta:+.3f} |"
        )
    lines.append("")

    lines.append("## Learning Record")
    lines.append("")
    if decision_rule:
        lines.append(f"Decision rule: {decision_rule}")
        lines.append("")
    if learning_prompts:
        lines.append("Questions to answer after reviewing comparisons:")
        lines.append("")
        for i, q in enumerate(learning_prompts, start=1):
            lines.append(f"{i}. {q}")
        lines.append("")

    lines.append("Per intervention notes:")
    lines.append("")
    for rec in intervention_results:
        lines.append(f"### {rec['label']}")
        lines.append("")
        if rec.get("hypothesis"):
            lines.append(f"- Hypothesis: {rec['hypothesis']}")
        if rec.get("notes"):
            lines.append(f"- Change notes: {rec['notes']}")
        lines.append("- What changed in understanding:")
        lines.append("- Decision: keep / discard / needs-followup")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--no-execute", action="store_true",
                    help="Do not execute commands; only summarize/compare existing outputs")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Optional explicit run directory")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = args.manifest if args.manifest.is_absolute() else (repo_root / args.manifest)
    manifest = load_manifest(manifest_path)

    experiment_id = str(manifest.get("experiment_id", "proof-stepper")).strip() or "proof-stepper"

    run_cfg = manifest.get("run", {}) if isinstance(manifest.get("run", {}), dict) else {}
    execute = bool(run_cfg.get("execute", True)) and (not args.no_execute)
    stop_on_error = bool(run_cfg.get("stop_on_error", True))

    base_out = repo_root / "data" / "first-proof" / "stepper-runs"
    if args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else (repo_root / args.run_dir)
    else:
        run_dir = base_out / sanitize_label(experiment_id) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_copy = run_dir / "manifest.json"
    manifest_copy.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    compare_script_raw = manifest.get("compare_script", "scripts/compare-proof-polish-arms.py")
    compare_script = abs_path(repo_root, str(compare_script_raw))
    assert compare_script is not None
    wiring = abs_path(repo_root, manifest.get("wiring")) if isinstance(manifest.get("wiring"), str) else None

    baseline = parse_arm(repo_root, manifest["baseline"])
    interventions = [parse_arm(repo_root, x) for x in manifest["interventions"]]

    baseline_result = run_arm(repo_root, baseline, run_dir, execute=execute)
    if stop_on_error and baseline_result.get("ran") and baseline_result.get("exit_code") not in (0, None):
        raise SystemExit(f"baseline arm failed: {baseline.label}")

    intervention_results: list[dict[str, Any]] = []
    for arm in interventions:
        rec = run_arm(repo_root, arm, run_dir, execute=execute)
        intervention_results.append(rec)
        if stop_on_error and rec.get("ran") and rec.get("exit_code") not in (0, None):
            raise SystemExit(f"intervention arm failed: {arm.label}")

    comparisons: list[dict[str, Any]] = []
    for arm in interventions:
        if not baseline.output_jsonl.exists() or not arm.output_jsonl.exists():
            comparisons.append({
                "candidate_label": arm.label,
                "payload": None,
                "error": "missing baseline or candidate output file",
            })
            continue
        comp = compare_arms(
            repo_root=repo_root,
            run_dir=run_dir,
            compare_script=compare_script,
            wiring=wiring,
            baseline=baseline,
            candidate=arm,
        )
        comparisons.append(comp)

    learning_cfg = manifest.get("learning_log", {}) if isinstance(manifest.get("learning_log"), dict) else {}
    prompts = learning_cfg.get("questions", []) if isinstance(learning_cfg.get("questions"), list) else []
    prompts = [str(x) for x in prompts]
    decision_rule = str(learning_cfg.get("decision_rule", "")).strip()

    summary_md = render_summary_md(
        experiment_id=experiment_id,
        run_dir=run_dir,
        baseline_result=baseline_result,
        intervention_results=intervention_results,
        comparisons=comparisons,
        learning_prompts=prompts,
        decision_rule=decision_rule,
    )
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    record = {
        "generated_utc": now_utc(),
        "experiment_id": experiment_id,
        "manifest": str(manifest_copy),
        "run_dir": str(run_dir),
        "execute": execute,
        "baseline": baseline_result,
        "interventions": intervention_results,
        "comparisons": comparisons,
    }
    (run_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"stepper run completed: {run_dir}")
    print(f"summary: {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
