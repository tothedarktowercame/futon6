#!/usr/bin/env python3
"""mark3 held-step evaluation harness.

Aggregates the measurement set used to grade mark3 runs:

* grounding-% against layer-(a) golden-mark baseline when artifact evidence can
  be tied to golden papers,
* expository-coverage-% when artifacts carry source line intervals,
* checker-PASS-% from the structural checker applicable to the run kind,
* substance-PASS-% from ``substance_gate.py``, and
* prior-vs-posterior against the CT term-prior.

The harness deliberately reports non-computable metrics explicitly instead of
inventing denominators.  H1 concept entries and H2 IATC graphs are different
artifact schemas; the script detects the kind and runs the checker that applies.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
DEFAULT_PRIOR = ROOT / "data" / "term-prior-ct.json"
EXPOSITORY_EXTRACT = ROOT / "scripts" / "expository_region_extract.py"
CHECKERS = {
    "concept": ROOT / "scripts" / "concept_argcheck.bb",
    "iatc": ROOT / "scripts" / "iatc_argcheck.bb",
}
SUBSTANCE_GATE = ROOT / "scripts" / "substance_gate.py"

PAPER_ID_RE = re.compile(
    r':(?:paper/id|paper)\s+"([^"]+)"|'
    r':sample\s+\[\s+"([^"]+)"|'
    r'\b([0-9]{4}\.[0-9]{4,5}|math__[0-9]{7}|quant-ph__[0-9]{7})\b'
)
SOURCE_LINES_RE = re.compile(r":source\s+\{[^{}]*:lines\s+\[([0-9]+)\s+([0-9]+)\]", re.S)
CONCEPT_ID_RE = re.compile(r":concept/id\s+:([\w./+-]+)")
NAME_RE = re.compile(r':name\s+"([^"]+)"')


def pct(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def collect_edn(run_dir: Path) -> list[Path]:
    if run_dir.is_file():
        return [run_dir] if run_dir.suffix == ".edn" else []
    return sorted(p for p in run_dir.rglob("*.edn") if p.is_file())


def detect_kind_for_text(text: str) -> str:
    if ":concept/id" in text:
        return "concept"
    if ":nodes" in text and ":edges" in text:
        return "iatc"
    return "unknown"


def detect_run_kind(files: list[Path]) -> tuple[str, dict[str, int]]:
    counts = Counter(detect_kind_for_text(read_text(p)) for p in files)
    non_unknown = [k for k in counts if k != "unknown"]
    if len(non_unknown) == 1:
        return non_unknown[0], dict(counts)
    if not non_unknown:
        return "unknown", dict(counts)
    return "mixed", dict(counts)


def extract_paper_ids(text: str, fallback_stem: str | None = None) -> set[str]:
    ids: set[str] = set()
    for match in PAPER_ID_RE.finditer(text):
        ids.update(g for g in match.groups() if g)
    if fallback_stem and re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}|math__[0-9]{7}|quant-ph__[0-9]{7}", fallback_stem):
        ids.add(fallback_stem)
    return ids


def extract_source_intervals(text: str) -> list[tuple[int, int]]:
    out = []
    for a, b in SOURCE_LINES_RE.findall(text):
        lo, hi = int(a), int(b)
        if lo <= hi:
            out.append((lo, hi))
    return out


def golden_file(golden_dir: Path, paper_id: str) -> Path:
    return golden_dir / f"fable-{paper_id}-dp-emacs.json"


def golden_mark_count(path: Path) -> int:
    try:
        data = json.loads(read_text(path))
    except Exception:
        return 0
    marks = data.get("marks") or []
    return sum(1 for m in marks if m.get("layer") in {None, "dp"})


def artifact_index(files: list[Path]) -> dict[str, Any]:
    paper_ids: set[str] = set()
    intervals_by_paper: dict[str, list[tuple[int, int]]] = {}
    terms: list[str] = []
    for path in files:
        text = read_text(path)
        ids = extract_paper_ids(text, path.stem)
        paper_ids.update(ids)
        intervals = extract_source_intervals(text)
        if intervals:
            for pid in ids or {path.stem}:
                intervals_by_paper.setdefault(pid, []).extend(intervals)
        concept = CONCEPT_ID_RE.search(text)
        if concept:
            terms.append(concept.group(1).replace("-", " "))
        elif ":nodes" in text:
            terms.extend(
                t.lower()
                for t in re.findall(r':text\s+"([^"]{4,80})"', text)
                if len(t.split()) <= 6
            )
        else:
            name = NAME_RE.search(text)
            if name:
                terms.append(name.group(1).lower())
    return {
        "paper_ids": sorted(paper_ids),
        "intervals_by_paper": intervals_by_paper,
        "terms": terms,
    }


def grounding_metric(files: list[Path], idx: dict[str, Any], golden_dir: Path) -> dict[str, Any]:
    paper_ids = idx["paper_ids"]
    if not paper_ids:
        return {
            "computable": False,
            "reason": "artifacts do not name source papers",
            "grounded_artifacts": len(files),
            "baseline_layer_a_marks": None,
            "grounding_percent": None,
        }
    baseline = sum(golden_mark_count(golden_file(golden_dir, pid)) for pid in paper_ids)
    if not baseline:
        return {
            "computable": False,
            "reason": "no matching golden mark files for artifact paper ids",
            "paper_count": len(paper_ids),
            "grounded_artifacts": len(files),
            "baseline_layer_a_marks": 0,
            "grounding_percent": None,
        }
    return {
        "computable": True,
        "method": "artifact-count divided by layer-(a) dp mark count over referenced golden papers",
        "paper_count": len(paper_ids),
        "grounded_artifacts": len(files),
        "baseline_layer_a_marks": baseline,
        "grounding_percent": pct(len(files), baseline),
    }


def expository_regions(paper_id: str) -> list[dict[str, Any]]:
    cmd = [sys.executable, str(EXPOSITORY_EXTRACT), paper_id]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        return []
    try:
        return json.loads(proc.stdout).get("regions") or []
    except Exception:
        return []


def line_set(intervals: list[tuple[int, int]]) -> set[int]:
    out: set[int] = set()
    for lo, hi in intervals:
        out.update(range(lo, hi + 1))
    return out


def expository_coverage_metric(idx: dict[str, Any]) -> dict[str, Any]:
    intervals_by_paper = idx["intervals_by_paper"]
    if not intervals_by_paper:
        return {
            "computable": False,
            "reason": "artifacts do not carry source line intervals",
            "covered_expository_lines": None,
            "expository_lines": None,
            "expository_coverage_percent": None,
        }
    covered = 0
    denominator = 0
    paper_rows = []
    for pid, intervals in sorted(intervals_by_paper.items()):
        regions = expository_regions(pid)
        expo = set()
        for r in regions:
            line_start = r.get("line_start")
            line_end = r.get("line_end")
            if isinstance(line_start, int) and isinstance(line_end, int):
                expo.update(range(line_start, line_end + 1))
        artifact_lines = line_set(intervals)
        c = len(artifact_lines & expo)
        d = len(expo)
        covered += c
        denominator += d
        paper_rows.append({
            "paper": pid,
            "covered_expository_lines": c,
            "expository_lines": d,
            "coverage": pct(c, d),
        })
    return {
        "computable": bool(denominator),
        "method": "artifact source lines intersected with expository_region_extract.py regions",
        "covered_expository_lines": covered,
        "expository_lines": denominator,
        "expository_coverage_percent": pct(covered, denominator),
        "per_paper": paper_rows,
    }


def parse_pass_fail(stdout: str) -> tuple[int, int]:
    passed = failed = 0
    for line in stdout.splitlines():
        if line.startswith("PASS "):
            passed += 1
        elif line.startswith("FAIL "):
            failed += 1
    return passed, failed


def run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    passed, failed = parse_pass_fail(proc.stdout)
    total = passed + failed
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "passed": passed,
        "failed": failed,
        "total": total,
        "pass_percent": pct(passed, total),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def checker_metrics(run_dir: Path, run_kind: str) -> dict[str, Any]:
    structural: dict[str, Any] = {}
    applicable = []
    for kind, checker in CHECKERS.items():
        if run_kind in {kind, "mixed"}:
            cmd = ["bb", str(checker), str(run_dir)]
            structural[kind] = run_command(cmd)
            applicable.append(structural[kind])
        else:
            structural[kind] = {
                "skipped": True,
                "reason": f"run kind is {run_kind}, not {kind}",
                "pass_percent": None,
            }
    total = sum(x.get("total", 0) for x in applicable)
    passed = sum(x.get("passed", 0) for x in applicable)
    substance_kind = run_kind if run_kind in {"concept", "iatc"} else "auto"
    substance = run_command(
        [sys.executable, str(SUBSTANCE_GATE), str(run_dir), "--kind", substance_kind]
    )
    return {
        "structural": structural,
        "checker_PASS_percent": pct(passed, total),
        "checker_passed": passed,
        "checker_total": total,
        "substance": substance,
        "substance_PASS_percent": substance.get("pass_percent"),
    }


def prior_vs_posterior_metric(terms: list[str], prior_path: Path) -> dict[str, Any]:
    unique_terms = sorted({t.strip().lower() for t in terms if t and t.strip()})
    if not prior_path.exists():
        return {
            "computable": False,
            "reason": f"prior file not found: {prior_path}",
            "posterior_terms": len(unique_terms),
        }
    try:
        prior = json.loads(read_text(prior_path)).get("df") or {}
    except Exception as exc:
        return {
            "computable": False,
            "reason": f"could not read prior: {exc}",
            "posterior_terms": len(unique_terms),
        }
    hits = {t: prior[t] for t in unique_terms if t in prior}
    dfs = list(hits.values())
    return {
        "computable": True,
        "method": "posterior artifact terms looked up in data/term-prior-ct.json document-frequency prior",
        "posterior_terms": len(unique_terms),
        "prior_terms": len(prior),
        "posterior_terms_with_prior_df": len(hits),
        "posterior_prior_hit_percent": pct(len(hits), len(unique_terms)),
        "median_prior_df_for_hits": median(dfs) if dfs else None,
        "novel_posterior_terms": [t for t in unique_terms if t not in prior][:50],
        "top_prior_supported_posterior_terms": [
            {"term": t, "df": df}
            for t, df in sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
        ],
    }


def build_report(run_dir: Path, golden_dir: Path, prior_path: Path) -> dict[str, Any]:
    files = collect_edn(run_dir)
    run_kind, kind_counts = detect_run_kind(files)
    idx = artifact_index(files)
    checkers = checker_metrics(run_dir, run_kind) if files else {
        "structural": {},
        "checker_PASS_percent": None,
        "checker_passed": 0,
        "checker_total": 0,
        "substance": {"skipped": True, "reason": "no .edn artifacts"},
        "substance_PASS_percent": None,
    }
    return {
        "schema": "futon6.mark3-eval-harness.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "golden_dir": str(golden_dir),
        "prior_path": str(prior_path),
        "artifact_count": len(files),
        "run_kind": run_kind,
        "kind_counts": kind_counts,
        "paper_ids": idx["paper_ids"],
        "metrics": {
            "grounding": grounding_metric(files, idx, golden_dir),
            "expository_coverage": expository_coverage_metric(idx),
            "checkers": checkers,
            "prior_vs_posterior": prior_vs_posterior_metric(idx["terms"], prior_path),
        },
    }


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def human_summary(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    g = metrics["grounding"]
    e = metrics["expository_coverage"]
    c = metrics["checkers"]
    p = metrics["prior_vs_posterior"]
    lines = [
        "# mark3 eval harness summary",
        "",
        f"Run: `{report['run_dir']}`",
        f"Kind: `{report['run_kind']}` ({report['artifact_count']} EDN artifact(s))",
        f"Papers referenced: {len(report['paper_ids'])}",
        "",
        "## Metrics",
        "",
        f"- grounding-%: {fmt_pct(g.get('grounding_percent'))} "
        f"({g.get('grounded_artifacts')} artifacts / {g.get('baseline_layer_a_marks')} layer-(a) marks; "
        f"computable={g.get('computable')})",
        f"- expository-coverage-%: {fmt_pct(e.get('expository_coverage_percent'))} "
        f"({e.get('covered_expository_lines')} / {e.get('expository_lines')} lines; "
        f"computable={e.get('computable')})",
        f"- checker-PASS-%: {fmt_pct(c.get('checker_PASS_percent'))} "
        f"({c.get('checker_passed')} / {c.get('checker_total')} structural items)",
        f"- substance-PASS-%: {fmt_pct(c.get('substance_PASS_percent'))} "
        f"({c.get('substance', {}).get('passed')} / {c.get('substance', {}).get('total')} items)",
        f"- prior-vs-posterior: {fmt_pct(p.get('posterior_prior_hit_percent'))} "
        f"({p.get('posterior_terms_with_prior_df')} / {p.get('posterior_terms')} posterior terms in prior)",
        "",
        "## Checker commands",
        "",
    ]
    for kind, rec in c.get("structural", {}).items():
        if rec.get("skipped"):
            lines.append(f"- {kind}: skipped ({rec.get('reason')})")
        else:
            lines.append(
                f"- {kind}: exit={rec.get('exit_code')} pass={rec.get('passed')}/"
                f"{rec.get('total')} command=`{' '.join(rec.get('command', []))}`"
            )
    s = c.get("substance", {})
    if not s.get("skipped"):
        lines.append(
            f"- substance: exit={s.get('exit_code')} pass={s.get('passed')}/"
            f"{s.get('total')} command=`{' '.join(s.get('command', []))}`"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory or .edn file to grade")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--out", type=Path, default=Path("mark3-eval-report.json"))
    parser.add_argument("--summary-out", type=Path, default=Path("mark3-eval-summary.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.run_dir.resolve(), args.golden.resolve(), args.prior.resolve())
    summary = human_summary(report)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary_out.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote JSON: {args.out}")
    print(f"Wrote summary: {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
