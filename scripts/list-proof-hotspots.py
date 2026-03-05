#!/usr/bin/env python3
"""List hotspot nodes across all proof-polish result files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID = {"verified", "plausible", "gap", "error"}
PAT = re.compile(r"problem(\d+)-codex-results")


def status(rec: dict[str, Any]) -> str:
    st = rec.get("claim_verified")
    if st in VALID:
        return str(st)
    if rec.get("parse_error"):
        return "parse"
    return "parse"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/first-proof"))
    ap.add_argument("--min-observations", type=int, default=3)
    ap.add_argument("--unresolved-threshold", type=float, default=0.8)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--output-md", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.data_dir.glob("problem*-codex-results*.jsonl"))
    per_node: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for p in files:
        m = PAT.search(p.name)
        if not m:
            continue
        pid = f"P{int(m.group(1))}"
        with p.open("r", encoding="utf-8", errors="ignore") as f:
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
                per_node[(pid, node_id)][status(rec)] += 1

    hotspots = []
    for (pid, nid), c in sorted(per_node.items(), key=lambda x: (int(x[0][0][1:]), x[0][1])):
        obs = sum(c.values())
        if obs < args.min_observations:
            continue
        unresolved = c["plausible"] + c["gap"] + c["error"] + c["parse"]
        unr = unresolved / obs if obs else 0.0
        is_hot = (c["verified"] == 0) or (unr >= args.unresolved_threshold)
        if not is_hot:
            continue
        hotspots.append(
            {
                "problem": pid,
                "node_id": nid,
                "observations": obs,
                "verified": c["verified"],
                "plausible": c["plausible"],
                "gap": c["gap"],
                "error": c["error"],
                "parse": c["parse"],
                "unresolved_rate": round(unr, 4),
            }
        )

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for h in hotspots:
        by_problem[h["problem"]].append(h)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "data_dir": str(args.data_dir),
            "min_observations": args.min_observations,
            "unresolved_threshold": args.unresolved_threshold,
            "files_scanned": len(files),
        },
        "hotspot_count": len(hotspots),
        "by_problem": dict(by_problem),
        "hotspots": hotspots,
    }

    md_lines = [
        "# Proof Hotspots",
        "",
        f"Generated: `{payload['generated_utc']}`",
        "",
        f"- files scanned: `{len(files)}`",
        f"- hotspot nodes: `{len(hotspots)}`",
        "",
    ]
    for pid in sorted(by_problem.keys(), key=lambda s: int(s[1:])):
        md_lines.append(f"## {pid}")
        md_lines.append("")
        md_lines.append("| Node | Obs | Verified | Plausible | Gap | Error | Parse | Unresolved % |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for h in by_problem[pid]:
            md_lines.append(
                f"| {h['node_id']} | {h['observations']} | {h['verified']} | {h['plausible']} | "
                f"{h['gap']} | {h['error']} | {h['parse']} | {100*h['unresolved_rate']:.1f}% |"
            )
        md_lines.append("")

    md = "\n".join(md_lines) + "\n"

    if args.output_json:
        outj = args.output_json if args.output_json.is_absolute() else Path.cwd() / args.output_json
        outj.parent.mkdir(parents=True, exist_ok=True)
        outj.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        outm = args.output_md if args.output_md.is_absolute() else Path.cwd() / args.output_md
        outm.parent.mkdir(parents=True, exist_ok=True)
        outm.write_text(md, encoding="utf-8")

    print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
