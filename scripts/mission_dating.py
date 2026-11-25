#!/usr/bin/env python3
"""mission_dating.py — dated table of positioned missions and their scopes.

Input for the Futon City "tide" (temporal inundation) control, per
claude-17's archipelago brief §3/§8/§9:

  * Mission dates come from git history of the mission doc
    (futon*/holes/**/M-*.md under the code root). A commit touching more
    than ADMIN_THRESHOLD mission docs is ADMINISTRATIVE (e.g. the
    mission-accept-cNNN disposition sweeps): it feeds last_admin /
    n_admin_commits only, never last_work.
  * A doc with only administrative commits is "git-admin-only" (dated by
    last_admin), not "undated".
  * Scope dates come from git blame of the scope's :anchor/passage in the
    mission doc, with admin commits ignored (--ignore-rev). A passage that
    cannot be located falls back to the mission's last_work
    ("mission-fallback"); an undated mission gives null ("undated").
    Unknown is NEVER 0 / epoch / "old".

Output: data/mission-dating.json (or $FUTON6_STORAGE_ROOT/futon6/... if it
would exceed 5 MB).
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futon6_config as config

ROOT = config.ROOT
POS_EMBED = ROOT / "data" / "mission-carpet-pos-embed.json"
OUT = ROOT / "data" / "mission-dating.json"
OUT_LIMIT_BYTES = 5 * 1024 * 1024

ADMIN_THRESHOLD = 5

SUBSTRATE = (
    os.environ.get("FUTON_SUBSTRATE_URL")
    or os.environ.get("FUTON1A_URL")
    or "http://127.0.0.1:7073"
).rstrip("/")
BASE = f"{SUBSTRATE}/api/alpha/hyperedges"
BINDERS = [
    "eightfold-phase", "loose-section", "mission-scope-in", "mission-scope-out",
    "map-item", "source-material", "relates-to", "capability-scope",
    "pattern", "psr", "pur",
    "verify-gate", "certificate", "plain-argument",
]
LIMIT = 1000
FETCH_TIMEOUT_S = int(os.environ.get("MISSION_DATING_TIMEOUT_S", "180"))
GIT_TIMEOUT_S = int(os.environ.get("MISSION_DATING_GIT_TIMEOUT_S", "60"))


# ---------------------------------------------------------------- git pieces

def repo_root(path: Path) -> str:
    """The git worktree root owning a file."""
    out = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    if out.returncode != 0:
        raise RuntimeError(f"{path} is not inside a git worktree")
    return out.stdout.strip()


def _iter_log(repo: str):
    """Yield (sha, author_iso, subject, [touched paths]) per commit touching
    any mission doc (holes/**/M-*.md) in repo."""
    out = subprocess.run(
        ["git", "-C", repo, "log",
         "--pretty=format:%x01%H%x02%aI%x02%s", "--name-only",
         "--", ":(glob)holes/**/M-*.md"],
        capture_output=True, text=True, timeout=GIT_TIMEOUT_S * 4)
    if out.returncode != 0:
        return
    sha = date = subject = None
    files = []
    for ln in out.stdout.splitlines():
        if ln.startswith("\x01"):
            if sha:
                yield sha, date, subject, files
            parts = (ln[1:].split("\x02", 2) + [""] * 3)
            sha, date, subject = parts[0], parts[1], parts[2]
            files = []
        elif ln.strip():
            files.append(ln.strip())
    if sha:
        yield sha, date, subject, files


def collect_commits(doc_paths: dict, repo_of: dict | None = None):
    """Scan every repo holding a mission doc, dedupe commits by sha.

    doc_paths: {mission_id: Path}. Returns (missions, admin_commits):
      missions: {mid: {"born": iso|None, "work_dates": [iso],
                        "admin_dates": [iso], "n_work": int, "n_admin": int}}
      admin_commits: {sha: {"repo", "sha", "date", "subject", "n_docs"}}
    """
    if repo_of is None:
        repo_of = {mid: repo_root(p) for mid, p in doc_paths.items()}
    missions = {mid: {"born": None, "work_dates": [], "admin_dates": [],
                      "n_work": 0, "n_admin": 0}
                for mid in doc_paths}
    commits = {}  # sha -> {date, subject, repo, docs: set(basename)}
    rel = {mid: str(p.relative_to(repo_of[mid])) if p.is_absolute()
           else str(p) for mid, p in doc_paths.items()}
    for repo in sorted(set(repo_of.values())):
        for sha, date, subject, files in _iter_log(repo):
            entry = commits.setdefault(
                sha, {"date": date, "subject": subject, "repo": repo, "docs": set()})
            for f in files:
                if os.path.basename(f).startswith("M-") and f.endswith(".md"):
                    entry["docs"].add(f)
    admin = {}
    for sha, e in commits.items():
        touched = e["docs"]
        is_admin = len(touched) > ADMIN_THRESHOLD
        if is_admin:
            admin[sha] = {"repo": e["repo"], "sha": sha, "date": e["date"],
                          "subject": e["subject"], "n_docs": len(touched)}
        for mid, rp in rel.items():
            if rp in touched:
                m = missions[mid]
                d = e["date"]
                if m["born"] is None or d < m["born"]:
                    m["born"] = d
                if is_admin:
                    m["admin_dates"].append(d)
                    m["n_admin"] += 1
                else:
                    m["work_dates"].append(d)
                    m["n_work"] += 1
    return missions, admin


def blame_line_dates(repo: str, relpath: str, ignore_shas=()) -> dict:
    """{final_lineno: author_iso} from git blame, ignoring admin commits."""
    cmd = ["git", "-C", repo, "blame", "--line-porcelain"]
    for sha in ignore_shas:
        cmd += ["--ignore-rev", sha]
    cmd += ["--", relpath]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=GIT_TIMEOUT_S)
    if out.returncode != 0:
        return {}
    dates = {}
    sha = lineno = None
    for ln in out.stdout.splitlines():
        if not ln.startswith("\t"):
            parts = ln.split(" ")
            if len(parts) >= 3 and len(parts[0]) == 40:
                sha, lineno = parts[0], int(parts[2])
            elif ln.startswith("author-time "):
                ts = int(ln.split(" ", 1)[1])
                if lineno is not None:
                    dates[lineno] = datetime.fromtimestamp(
                        ts, tz=timezone.utc).isoformat()
    return dates


def passage_lines(text: str, passage: str):
    """Line numbers (1-based) covered by the first verbatim occurrence of
    passage in text, or None if absent."""
    idx = text.find(passage)
    if idx < 0:
        return None
    start = text.count("\n", 0, idx) + 1
    span = passage.count("\n")
    return list(range(start, start + span + 1))


def date_scope(text, passage, line_dates, mission_last_work, mission_dated):
    """One scope's (dated, date_source)."""
    lines = passage_lines(text, passage) if passage else None
    if lines:
        ds = [line_dates[n] for n in lines if n in line_dates]
        if ds:
            return max(ds), "blame"
    if mission_dated:
        return mission_last_work, "mission-fallback"
    return None, "undated"


# ------------------------------------------------------------- substrate

def fetch_scopes(binders=BINDERS):
    """All mission-scope hyperedges from futon1b, paged like
    mission_efe_scope_dump.fetch."""
    scopes = []
    for binder in binders:
        after = None
        seen = set()
        while True:
            params = {"type": f"mission-scope/{binder}", "limit": LIMIT,
                      "fields": "hx/props"}
            if after:
                params["after"] = after
            q = urllib.parse.urlencode(params)
            req = urllib.request.Request(f"{BASE}?{q}",
                                         headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
                page = json.load(r)
            rows = page.get("hyperedges", [])
            for h in rows:
                props = h.get("hx/props") or {}
                sid = props.get("scope/id")
                if sid:
                    scopes.append({
                        "scope/id": sid,
                        "mission": props.get("mission") or sid.split("/")[0],
                        "binder": props.get("scope/binder-type") or binder,
                        "passage": props.get("anchor/passage") or "",
                    })
            cursor = page.get("next-cursor")
            if not cursor:
                break
            if not rows or cursor in seen:
                raise SystemExit(f"binder {binder!r}: stalled cursor page")
            seen.add(cursor)
            after = cursor
    return scopes


# -------------------------------------------------------------------- main

_COPY_DIRS = ("labs", "runs", "draft-missions", "authority")


def canonical_doc(paths):
    """Pick the mission's own doc among same-named files.

    Run directories copy mission docs (futon3c/holes/labs/wm-contract/runs/
    .../authority/holes/missions/M-u88-*.md has 14 copies), so the glob's
    last hit can be a frozen copy. Prefer a path with no copy directory in
    it, then one under holes/missions/, then the shortest path."""
    def rank(p):
        parts = p.parts
        return (any(d in parts for d in _COPY_DIRS),
                "missions" not in parts,
                len(parts), str(p))
    return min(paths, key=rank)


def build_table(doc_paths, scopes, repo_of=None):
    """doc_paths: {mid: Path}; scopes: fetch_scopes() rows. Returns the
    output dict (missions/scopes/admin_commits)."""
    missions_git, admin = collect_commits(doc_paths, repo_of=repo_of)
    if repo_of is None:
        repo_of = {mid: repo_root(p) for mid, p in doc_paths.items()}

    # admin shas to ignore in blame, per repo
    ignore_by_repo = defaultdict(list)
    for a in admin.values():
        ignore_by_repo[a["repo"]].append(a["sha"])

    out_missions = {}
    mission_rows = {}
    for mid, path in doc_paths.items():
        g = missions_git[mid]
        # A doc whose every commit is administrative (bulk import, repo sync,
        # disposition sweep) is still datable: it gets "git-admin-only", so
        # the tide can use last_admin with that flag rather than calling it
        # undated (never submerged) or old.
        if g["work_dates"]:
            source = "git"
        elif g["admin_dates"]:
            source = "git-admin-only"
        else:
            source = "undated"
        row = {
            "doc": str(path),
            "born": g["born"],
            "last_work": max(g["work_dates"]) if g["work_dates"] else None,
            "last_admin": max(g["admin_dates"]) if g["admin_dates"] else None,
            "n_work_commits": g["n_work"],
            "n_admin_commits": g["n_admin"],
            "date_source": source,
        }
        out_missions[mid] = row
        mission_rows[mid] = row

    # scopes grouped by mission; blame once per mission doc
    by_mission = defaultdict(list)
    for s in scopes:
        if s["mission"] in doc_paths:
            by_mission[s["mission"]].append(s)

    out_scopes = {}
    for mid, srows in sorted(by_mission.items()):
        repo = repo_of[mid]
        rel = str(doc_paths[mid].relative_to(repo)) \
            if doc_paths[mid].is_absolute() else str(doc_paths[mid])
        line_dates = blame_line_dates(repo, rel,
                                      ignore_by_repo.get(repo, ()))
        try:
            text = doc_paths[mid].read_text(encoding="utf-8",
                                            errors="replace")
        except OSError:
            text = ""
        m = mission_rows[mid]
        fallback = m["last_work"] or m["last_admin"]
        for s in srows:
            dated, source = date_scope(text, s["passage"], line_dates,
                                       fallback, bool(fallback))
            out_scopes[s["scope/id"]] = {
                "mission": mid, "binder": s["binder"],
                "dated": dated, "date_source": source,
            }

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "admin_commit_threshold": ADMIN_THRESHOLD,
        "missions": out_missions,
        "scopes": out_scopes,
        "admin_commits": sorted(admin.values(),
                                key=lambda a: (a["date"], a["sha"])),
    }


def main():
    positioned = json.loads(POS_EMBED.read_text())
    ids = sorted(positioned.keys())
    code_root = config.code_root()
    candidates = defaultdict(list)
    for p in code_root.glob("futon*/holes/**/M-*.md"):
        candidates[p.stem].append(p)
    doc_paths = {mid: canonical_doc(candidates[mid]) for mid in ids if mid in candidates}
    missing = sorted(set(ids) - set(doc_paths))
    if missing:
        print(f"warning: {len(missing)} positioned missions have no doc: "
              f"{missing[:10]}", file=sys.stderr)

    scopes = fetch_scopes()
    table = build_table(doc_paths, scopes)

    blob = json.dumps(table, indent=1, sort_keys=True)
    out = OUT
    if len(blob.encode()) > OUT_LIMIT_BYTES:
        out = config.storage() / "futon6" / "mission-dating.json"
        print(f"output exceeds 5 MB; writing {out} instead of {OUT}",
                  file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob + "\n")
    print(f"wrote {out} ({len(blob)} bytes): "
          f"{len(table['missions'])} missions, "
          f"{len(table['scopes'])} scopes, "
          f"{len(table['admin_commits'])} admin commits")


if __name__ == "__main__":
    main()
