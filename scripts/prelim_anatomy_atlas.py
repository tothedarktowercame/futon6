#!/usr/bin/env python3
"""Build a Prelim Anatomy Atlas from the UT-Austin APM problem corpus.

This is a thin corpus adapter around dp_paper_view.  Each worker points the
existing detector at storage/apm, where every manifest problem already has a
plain ``<id>.tex`` file, and emits the standard fable-*-dp-emacs JSON under
data/showcases/prelim-atlas/golden.  The parent uses a fresh Python subprocess
per problem so ConceptAuthority/nLab/Mathlib state is released promptly.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/joe/code/futon6")
APM_DIR = Path("/home/joe/code/storage/apm")
MANIFEST = APM_DIR / "manifest.edn"
ATLAS_DIR = ROOT / "data" / "showcases" / "prelim-atlas"
GOLDEN_DIR = ATLAS_DIR / "golden"
LOSS_DIR = ATLAS_DIR / "loss"
SUMMARY_JSON = ATLAS_DIR / "atlas.json"
SUMMARY_MD = ATLAS_DIR / "README.md"
LOG_JSONL = ATLAS_DIR / "run-log.jsonl"


def load_manifest_ids(manifest: Path = MANIFEST) -> list[str]:
    """Read the EDN manifest just far enough to recover the 489 problem IDs."""
    text = manifest.read_text()
    return re.findall(r':id "([^"]+)"', text)


def _worker(problem_id: str, out_dir: Path, flags: list[str]) -> int:
    import dp_paper_view as dpv

    dpv.EPRINTS = APM_DIR
    data = dpv.build(
        problem_id,
        with_ca="--with-concept-authority" in flags,
        with_binders="--with-binders" in flags,
        with_scopes="--with-scopes" in flags,
        with_xref="--with-xref" in flags,
    )
    data["paper"] = f"apm-{problem_id}-dp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fable-apm-{problem_id}-dp-emacs.json"
    out.write_text(json.dumps({k: v for k, v in data.items() if k != "_counts"}))
    counts = data.get("_counts", {})
    print(json.dumps({
        "id": problem_id,
        "out": str(out),
        "marks": len(data.get("marks", [])),
        "counts": counts,
    }))
    return 0


def run_worker(problem_id: str, flags: list[str]) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        problem_id,
        "--out-dir",
        str(GOLDEN_DIR),
        *flags,
    ]
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = round(time.time() - started, 3)
    record = {
        "id": problem_id,
        "status": "done" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
    }
    if proc.stdout.strip():
        last = proc.stdout.strip().splitlines()[-1]
        try:
            record["worker"] = json.loads(last)
        except json.JSONDecodeError:
            record["stdout_tail"] = last[-500:]
    if proc.stderr.strip():
        record["stderr_tail"] = proc.stderr.strip()[-1000:]
    return record


def run_invariants() -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_invariants.py"),
        "--corpus",
        "--golden-dir",
        str(GOLDEN_DIR),
        "--loss-dir",
        str(LOSS_DIR),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "check_invariants failed\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return json.loads((LOSS_DIR / "dashboard.json").read_text())


def collect_exemplars(limit: int = 20) -> list[dict]:
    rows = []
    for f in sorted(LOSS_DIR.glob("apm-*.json")):
        rep = json.loads(f.read_text())
        cov = rep["coverage"]
        rows.append({
            "problem": rep["paper"],
            "best_guess": cov["best_guess"],
            "symbol_grounded": cov["symbol_grounded"],
            "symbol_tagged": cov["symbol_tagged"],
            "symbols": cov["symbols"],
            "math_spans": cov["math_spans"],
            "wellformed_errors": cov["wellformed_errors"],
            "debt": sum(1 for v in rep["violations"] if v["severity"] == "debt"),
        })
    rows.sort(key=lambda r: (
        r["wellformed_errors"],
        -r["best_guess"],
        -r["symbols"],
        r["problem"],
    ))
    return rows[:limit]


def write_summary(run_records: list[dict], dashboard: dict) -> dict:
    done = [r for r in run_records if r["status"] == "done"]
    failed = [r for r in run_records if r["status"] != "done"]
    exemplars = collect_exemplars()
    summary = {
        "source": str(APM_DIR),
        "manifest": str(MANIFEST),
        "problems_manifest": len(load_manifest_ids()),
        "problems_marked": len(done),
        "failures": failed,
        "coverage": dashboard,
        "top_exemplars": exemplars,
    }
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=1))
    lines = [
        "# Prelim Anatomy Atlas",
        "",
        f"- source: `{APM_DIR}`",
        f"- manifest problems: {summary['problems_manifest']}",
        f"- marked problems: {summary['problems_marked']}",
        f"- failures: {len(failed)}",
        f"- corpus grounded: {dashboard.get('corpus_best_guess', 0):.1%}",
        f"- wf errors: {dashboard.get('totals', {}).get('errors', 0)}",
        "",
        "## Top Exemplars",
        "",
    ]
    for row in exemplars[:10]:
        lines.append(
            f"- `{row['problem']}`: grounded {row['best_guess']:.1%}, "
            f"{row['symbols']} symbols, wf {row['wellformed_errors']}, "
            f"debt {row['debt']}"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", metavar="PROBLEM_ID")
    parser.add_argument("--out-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--with-binders", action="store_true", default=True)
    parser.add_argument("--with-scopes", action="store_true", default=True)
    parser.add_argument("--with-concept-authority", action="store_true", default=True)
    parser.add_argument("--with-xref", action="store_true", default=True)
    parser.add_argument("--throttle-sec", type=float, default=0.1)
    args = parser.parse_args(argv)

    flags = []
    for flag in ("--with-binders", "--with-scopes",
                 "--with-concept-authority", "--with-xref"):
        if getattr(args, flag[2:].replace("-", "_")):
            flags.append(flag)

    if args.worker:
        return _worker(args.worker, args.out_dir, flags)

    ids = load_manifest_ids()
    if args.limit:
        ids = ids[:args.limit]
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    with LOG_JSONL.open("a") as log:
        for idx, problem_id in enumerate(ids, 1):
            out = GOLDEN_DIR / f"fable-apm-{problem_id}-dp-emacs.json"
            if out.exists() and not args.force:
                record = {"id": problem_id, "status": "skipped", "reason": "exists"}
            else:
                record = run_worker(problem_id, flags)
            records.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"[{idx}/{len(ids)}] {problem_id}: {record['status']}")
            if args.throttle_sec:
                time.sleep(args.throttle_sec)
    dashboard = run_invariants()
    summary = write_summary(records, dashboard)
    print(json.dumps({
        "problems_marked": summary["problems_marked"],
        "failures": len(summary["failures"]),
        "coverage": summary["coverage"].get("corpus_best_guess"),
        "top_exemplars": summary["top_exemplars"][:5],
    }, indent=1))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
