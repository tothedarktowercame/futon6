#!/usr/bin/env python3
"""Mine dry basins: open mission holes plus failed closure-fold seeds.

The positive mission triple miner intentionally skips not-completed missions.
This script reuses its discovery and extraction helpers, then emits the negative
class and a forward hitlist for likely closable work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_ROOT = Path("/home/joe/code")
FUTON6 = CODE_ROOT / "futon6"
SCRIPT_DIR = FUTON6 / "scripts"
OUT_DIR = FUTON6 / "data" / "dry-basins"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mission_triple_miner as triple  # noqa: E402


PHASES = ["head", "identify", "map", "derive", "argue", "verify", "instantiate", "document"]
TYPE_RANK = {
    ":checkpoint-only": 4,
    ":ratified-car": 3,
    ":design": 2,
    ":operator-ruling": 1,
    ":stale-unknown": 0,
}

ADVANCE_PATTERNS = [
    (
        ":checkpoint-only",
        re.compile(
            r"(?i)(substantively done|ready to close|all exits? met|exit criteria (?:met|pass)|"
            r"bundle-closure|documentation-only|needs? (?:only )?(?:a )?closing checkpoint|"
            r"checkpoint-only|close(?:d)? by checkpoint)"
        ),
    ),
    (
        ":ratified-car",
        re.compile(
            r"(?i)(next car|ratified|next step|next up|one codex fix away|handoff|"
            r"delegated|bell codex|car #[0-9]+|codex[- ]ready|documented next)"
        ),
    ),
    (
        ":operator-ruling",
        re.compile(
            r"(?i)(operator ruling|joe.*(?:rule|decid|consent|close)|consent gate|"
            r"mission-close|mission close|golden ruling|awaits? ruling|needs? operator)"
        ),
    ),
    (
        ":design",
        re.compile(
            r"(?i)(needs? design|design needed|no construction named|unknown shape|"
            r"needs? shape|derive pending|map pending|hole needs|open question|"
            r"not yet designed|shape-first)"
        ),
    ),
]


@dataclass(frozen=True)
class ClosureSeed:
    scope: str
    record: dict[str, Any]


def edn_key(k: str) -> str:
    return ":" + k.replace("_", "-").replace("?", "?")


def to_edn(obj: Any, indent: int = 0) -> str:
    return triple.to_edn(obj, indent)


def strip_comments(text: str) -> str:
    return "\n".join(line.split(";;", 1)[0] for line in text.splitlines())


def parse_closure_folds(path: Path) -> list[dict[str, Any]]:
    """Parse the flat EDN subset used by closure-folds.edn."""
    text = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
    rows: list[dict[str, Any]] = []
    for m in re.findall(r"\{[^{}]*\}", text, flags=re.S):
        rec: dict[str, Any] = {"source-file": str(path)}
        for key, value in re.findall(
            r":([\w\-/?.]+)\s+(\"(?:\\.|[^\"])*\"|true|false|\[[^\]]*\])",
            m,
            flags=re.S,
        ):
            if value == "true":
                rec[key] = True
            elif value == "false":
                rec[key] = False
            elif value.startswith("["):
                rec[key] = [
                    bytes(x, "utf-8").decode("unicode_escape")
                    for x in re.findall(r"\"((?:\\.|[^\"])*)\"", value)
                ]
            else:
                rec[key] = bytes(value[1:-1], "utf-8").decode("unicode_escape")
        if "scope" in rec:
            rows.append(rec)
    return rows


def discover_closure_seeds() -> list[ClosureSeed]:
    paths = []
    for repo in sorted(CODE_ROOT.glob("futon*")):
        if not repo.is_dir() or ".state" in repo.parts:
            continue
        for path in repo.glob("**/closure-folds.edn"):
            parts = set(path.parts)
            if ".state" not in parts and "target" not in parts and path.is_file():
                paths.append(path)
    seeds = []
    seen = set()
    for path in sorted(set(paths)):
        for rec in parse_closure_folds(path):
            if rec.get("success") is False:
                key = (rec.get("scope"), str(path), tuple(rec.get("used", [])))
                if key not in seen:
                    seen.add(key)
                    seeds.append(ClosureSeed(str(rec.get("scope")), rec))
    return seeds


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")
    return s or "dry-basin"


def mission_for_seed(scope: str, mission_by_stem: dict[str, Path]) -> str:
    head = scope.split("/", 1)[0]
    if head in mission_by_stem:
        return head
    if head.startswith("E-"):
        m = "M-" + head[2:]
        if m in mission_by_stem:
            return m
    return "closure-fold-" + slug(scope)


def git_last_activity(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    repo = path
    while repo != repo.parent and not (repo / ".git").exists():
        repo = repo.parent
    if not (repo / ".git").exists():
        return None
    rel = os.path.relpath(path, repo)
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "log", "-1", "--format=%aI%x09%h%x09%s", "--", rel],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        out = ""
    if not out:
        return {"repo": str(repo), "path": rel, "last": None}
    at, sha, subject = (out.split("\t", 2) + ["", ""])[:3]
    return {"repo": str(repo), "path": rel, "last": {"at": at, "sha": sha, "subject": subject}}


def phase_ghosts(sections: list[triple.Section]) -> dict[str, Any]:
    rows = {}
    for phase in PHASES:
        sec = triple.section_by_phase(sections, phase)
        rows[phase] = {"present?": bool(sec), "section": sec.title if sec else None}
    return {
        "present": [p for p, row in rows.items() if row["present?"]],
        "absent": [p for p, row in rows.items() if not row["present?"]],
        "phases": rows,
    }


def status_text(status: dict[str, Any], text: str) -> str:
    if status.get("evidence"):
        return str(status["evidence"])
    first = triple.first_substantial_paragraph(text)
    return first or text[:300].replace("\n", " ").strip()


def first_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    return pattern.search(text)


def advance_type(text: str, status: dict[str, Any], hole: dict[str, Any], closure: dict[str, Any] | None) -> dict[str, Any]:
    haystack = "\n".join(
        str(x)
        for x in [
            status.get("evidence"),
            hole.get("via"),
            closure.get("note") if closure else None,
            text,
        ]
        if x
    )
    for typ, pat in ADVANCE_PATTERNS:
        m = first_match(pat, haystack)
        if m:
            return {
                "type": typ,
                "via": triple.sentence_window(haystack, m.start(), m.end(), 260),
                "rule": "ruling-surface/advance-typing",
            }
    fallback = status.get("evidence") or hole.get("via") or (closure or {}).get("note") or haystack[:400]
    return {
        "type": ":stale-unknown",
        "via": re.sub(r"\s+", " ", str(fallback)[:600]).strip(),
        "rule": "ruling-surface/advance-typing-no-signal",
    }


def hunger_for(advance: dict[str, Any], hole: dict[str, Any], cascade: dict[str, Any], closure: dict[str, Any] | None) -> dict[str, Any]:
    typ = advance["type"]
    if closure and closure.get("missing"):
        return {"type": ":canon", "via": str(closure.get("missing")), "rule": "dry-basin/closure-fold-missing-pattern"}
    if typ in {":checkpoint-only", ":operator-ruling"}:
        return {"type": ":ruling", "via": advance["via"], "rule": "dry-basin/advance-type-to-hunger"}
    if typ == ":ratified-car":
        return {"type": ":payoff", "via": advance["via"], "rule": "dry-basin/advance-type-to-hunger"}
    if hole.get("confidence") == ":unminable" or not cascade.get("pattern-cites"):
        return {"type": ":design", "via": hole.get("via", ""), "rule": "dry-basin/hole-or-cascade-gap"}
    return {"type": ":design", "via": advance["via"], "rule": "dry-basin/default-design-hunger"}


def closure_pattern_cites(seed: ClosureSeed) -> list[dict[str, Any]]:
    out = []
    rec = seed.record
    note = rec.get("note", "")
    for ident in rec.get("used", []):
        pos = note.find(ident)
        out.append(
            {
                "ident": ident,
                "offset": pos if pos >= 0 else None,
                "via": note if note else f"closure-fold {seed.scope} used {ident}",
                "rule": "dry-basin/failed-closure-fold-used-pattern",
            }
        )
    missing = rec.get("missing")
    if missing:
        pos = note.find(missing)
        out.append(
            {
                "ident": missing,
                "offset": pos if pos >= 0 else None,
                "role": ":missing",
                "via": note if note else f"closure-fold {seed.scope} missing {missing}",
                "rule": "dry-basin/failed-closure-fold-missing-pattern",
            }
        )
    return out


def mission_basins(force: bool, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patterns = triple.SCOPE.load_pattern_index()
    pattern_index = {p["name"]: p["ref"] for p in patterns}
    missions = triple.discover_missions()
    if limit:
        missions = missions[:limit]
    rows = []
    skipped = []
    for path in missions:
        text = path.read_text(encoding="utf-8", errors="ignore")
        status = triple.status_classifier(text)
        if status["completed?"]:
            continue
        skipped.append(path.stem)
        sections = triple.split_sections(text)
        hole = triple.mine_hole(text, sections)
        cascade = triple.mine_patterns(path, text, sections, pattern_index, hole)
        advance = advance_type(text, status, hole, None)
        rows.append(
            {
                "mission": path.stem,
                "source": str(path),
                "status": status,
                "hole-candidate": hole,
                "cited-patterns": cascade["pattern-cites"],
                "vitals": {
                    "last-git-activity": git_last_activity(path),
                    "status-text": status_text(status, text),
                    "phase-ghost-census": phase_ghosts(sections),
                },
                "advance-type-candidate": advance,
                "hunger": hunger_for(advance, hole, cascade, None),
                "seed": ":mission-skip",
            }
        )
    return rows, {"missions-seen": len(missions), "skipped-not-completed": len(skipped), "skipped-missions": skipped}


def closure_basins(mission_by_stem: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    pattern_index = {p["name"]: p["ref"] for p in triple.SCOPE.load_pattern_index()}
    for seed in discover_closure_seeds():
        mission = mission_for_seed(seed.scope, mission_by_stem)
        path = mission_by_stem.get(mission)
        if path and path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            sections = triple.split_sections(text)
            status = triple.status_classifier(text)
            hole = triple.mine_hole(text, sections)
            cascade = triple.mine_patterns(path, text, sections, pattern_index, hole)
            vitals = {
                "last-git-activity": git_last_activity(path),
                "status-text": status_text(status, text),
                "phase-ghost-census": phase_ghosts(sections),
            }
        else:
            note = str(seed.record.get("note", ""))
            text = note
            status = {"completed?": False, "evidence": note, "classifier": "closure-fold success false"}
            hole = {
                "confidence": ":reconstructed-thin" if seed.record.get("missing") else ":unminable",
                "id": None,
                "have": seed.scope,
                "want": seed.record.get("missing"),
                "source-section": "closure-folds.edn",
                "via": note,
                "rule": "dry-basin/failed-closure-fold-seed",
            }
            cascade = {"pattern-cites": closure_pattern_cites(seed)}
            vitals = {
                "last-git-activity": git_last_activity(Path(seed.record.get("source-file", ""))),
                "status-text": note,
                "phase-ghost-census": {"present": [], "absent": PHASES, "phases": {p: {"present?": False, "section": None} for p in PHASES}},
            }
        advance = advance_type(text, status, hole, seed.record)
        rows.append(
            {
                "mission": mission,
                "source": str(path) if path else seed.record.get("source-file"),
                "status": status,
                "hole-candidate": hole,
                "cited-patterns": cascade["pattern-cites"],
                "vitals": vitals,
                "advance-type-candidate": advance,
                "hunger": hunger_for(advance, hole, cascade, seed.record),
                "seed": ":closure-fold-failure",
                "closure-fold": seed.record,
            }
        )
    return rows


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def hit_score(row: dict[str, Any]) -> tuple[float, float, float, str]:
    typ = row["advance-type-candidate"]["type"]
    base = TYPE_RANK.get(typ, 0)
    at = (((row.get("vitals") or {}).get("last-git-activity") or {}).get("last") or {}).get("at")
    dt = parse_date(at)
    if dt:
        age_days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
        recency = 1.0 / (1.0 + age_days / 30.0)
    else:
        recency = 0.0
    conf = row["hole-candidate"].get("confidence")
    clarity = {":authored": 2.0, ":reconstructed-thin": 1.0, ":unminable": 0.0}.get(conf, 0.5)
    return (base, recency, clarity, row["mission"])


def one_line_why(row: dict[str, Any]) -> str:
    typ = row["advance-type-candidate"]["type"].lstrip(":")
    hunger = row["hunger"]["type"].lstrip(":")
    hole = row["hole-candidate"]
    if hole.get("have") and hole.get("want"):
        return f"{typ}; hungers for {hunger}; {hole.get('have')} -> {hole.get('want')}"
    return f"{typ}; hungers for {hunger}; {hole.get('confidence')} hole"


def write_outputs(rows: list[dict[str, Any]], out_dir: Path, force: bool) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    used_names = set()
    for row in rows:
        base = slug(row["mission"])
        name = base
        i = 2
        while name in used_names:
            name = f"{base}-{i}"
            i += 1
        used_names.add(name)
        path = out_dir / f"{name}.edn"
        if path.exists() and not force:
            row["output"] = str(path)
            row["skipped-existing"] = True
        else:
            row["output"] = str(path)
            path.write_text(to_edn(row) + "\n", encoding="utf-8")
        written.append(row)
    ranked = sorted(rows, key=hit_score, reverse=True)
    hitlist = []
    for idx, row in enumerate(ranked, 1):
        hitlist.append(
            {
                "rank": idx,
                "mission": row["mission"],
                "advance-type": row["advance-type-candidate"]["type"],
                "one-line-why": one_line_why(row),
                "evidence-ref": {
                    "output": row.get("output"),
                    "via": row["advance-type-candidate"].get("via"),
                },
                "score-parts": {
                    "advance-ease": TYPE_RANK.get(row["advance-type-candidate"]["type"], 0),
                    "recency": hit_score(row)[1],
                    "hole-clarity": hit_score(row)[2],
                },
            }
        )
    (out_dir / "_hitlist.json").write_text(json.dumps(hitlist, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hitlist


def summarize(rows: list[dict[str, Any]], mission_summary: dict[str, Any], hitlist: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_seed: dict[str, int] = {}
    for row in rows:
        by_type[row["advance-type-candidate"]["type"]] = by_type.get(row["advance-type-candidate"]["type"], 0) + 1
        by_seed[row["seed"]] = by_seed.get(row["seed"], 0) + 1
    return {
        "dry-basins": len(rows),
        "mission-skip-count": by_seed.get(":mission-skip", 0),
        "closure-failure-seeds": by_seed.get(":closure-fold-failure", 0),
        "triple-miner-reconcile": {
            "missions-seen": mission_summary["missions-seen"],
            "skipped-not-completed": mission_summary["skipped-not-completed"],
        },
        "advance-types": dict(sorted(by_type.items())),
        "hitlist-count": len(hitlist),
        "m-cold-chain": next((row for row in rows if row["mission"] == "M-cold-chain"), None) is not None,
        "checkpoint-only-count": by_type.get(":checkpoint-only", 0),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, help="Limit mission discovery before filtering; closure seeds still included.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missions = triple.discover_missions()
    mission_by_stem = {p.stem: p for p in missions}
    mission_rows, mission_summary = mission_basins(args.force, args.limit)
    closure_rows = closure_basins(mission_by_stem)
    seen = {(row["mission"], row["seed"]) for row in mission_rows}
    rows = list(mission_rows)
    for row in closure_rows:
        key = (row["mission"], row["seed"])
        if key not in seen:
            rows.append(row)
            seen.add(key)
    hitlist = write_outputs(rows, args.out_dir, args.force)
    summary = summarize(rows, mission_summary, hitlist)
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
