#!/usr/bin/env python3
"""Audit hash citations in pivotal-moments narrative files against git reality."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


TARGETS = [
    "data/first-proof/latex/part4-proof-patterns.tex",
    "data/first-proof/making-of.md",
]

HASH_RE = re.compile(r"\\texttt\{([0-9a-f]{7,40})\}|\b([0-9a-f]{7,40})\b")


def git_ok_commit(h: str) -> bool:
    p = subprocess.run(["git", "cat-file", "-e", f"{h}^{{commit}}"], capture_output=True)
    return p.returncode == 0


def git_subject(h: str) -> str:
    p = subprocess.run(["git", "show", "-s", "--format=%h|%ad|%s", "--date=iso-strict", h], capture_output=True, text=True)
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def extract_hashes(text: str):
    out = []
    for m in HASH_RE.finditer(text):
        h = m.group(1) or m.group(2)
        if h is None:
            continue
        # avoid years/ids by requiring at least one a-f
        if not re.search(r"[a-f]", h):
            continue
        out.append((h, m.start()))
    return out


def line_of_pos(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data/first-proof/pivotal-moments-audit.md"

    lines = []
    lines.append("# Pivotal Moments Reality Audit")
    lines.append("")
    lines.append(f"Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("Check: cited commit hashes in narrative files resolve to real commits.")
    lines.append("")

    total = 0
    missing = 0

    for rel in TARGETS:
        p = root / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        hashes = extract_hashes(txt)
        # de-dup by hash preserving first location
        first = {}
        for h, pos in hashes:
            first.setdefault(h, pos)

        lines.append(f"## `{rel}`")
        lines.append("")
        lines.append("| Hash | Status | First line | Commit subject |")
        lines.append("|---|---|---:|---|")

        for h, pos in sorted(first.items(), key=lambda kv: kv[1]):
            total += 1
            ln = line_of_pos(txt, pos)
            ok = git_ok_commit(h)
            if not ok:
                missing += 1
                lines.append(f"| `{h}` | MISSING | {ln} | - |")
            else:
                subj = git_subject(h).replace("|", "\\|")
                lines.append(f"| `{h}` | OK | {ln} | {subj} |")

        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total distinct hash citations checked: `{total}`")
    lines.append(f"- Missing hashes: `{missing}`")
    if missing == 0:
        lines.append("- Result: all cited hashes resolve to real commits.")
    else:
        lines.append("- Result: some cited hashes are stale and should be corrected.")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
