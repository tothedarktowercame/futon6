#!/usr/bin/env python3
"""Build futon6/data/mission-activity.json: per-mission activity + code churn.

Stdlib only. See packet from claude-12 (M-the-perfect-crime) for schema.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import futon6_config as config

CODE = str(config.code_root())
FUTON6 = str(config.ROOT)
DATA = os.path.join(FUTON6, "data")
OUT = os.path.join(DATA, "mission-activity.json")

WHOLENESS = os.path.join(DATA, "mission-wholeness.edn")
EDGES = os.path.join(DATA, "fold-embed/edges.jsonl")
CARPET = os.path.join(DATA, "mission-carpet-pos-embed.json")

PHASES = ["IDENTIFY", "MAP", "DERIVE", "ARGUE", "VERIFY", "INSTANTIATE", "DOCUMENT"]
WEEKS = 26
DAY = 86400
EXTS = (".clj", ".cljc", ".cljs", ".bb")
DEFAULT_ROOTS = ["src", "scripts", "dev", "test"]


def canonical_repos():
    """futon* dirs with no dash after 'futon' (skip worktree copies)."""
    repos = []
    for name in sorted(os.listdir(CODE)):
        if not name.startswith("futon"):
            continue
        if not os.path.isdir(os.path.join(CODE, name)):
            continue
        if "-" in name[len("futon"):]:
            continue
        repos.append(name)
    return repos


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_docs(repos):
    """stem -> (repo, relpath) for M-*.md at depth <=2 under holes/."""
    docs = {}
    for repo in repos:
        holes = os.path.join(CODE, repo, "holes")
        if not os.path.isdir(holes):
            continue
        for root, dirs, files in os.walk(holes):
            depth = os.path.relpath(root, holes).count(os.sep)
            if depth >= 2:
                dirs[:] = []
                continue
            for fn in files:
                if fn.startswith("M-") and fn.endswith(".md"):
                    stem = fn[:-3]
                    rel = os.path.relpath(os.path.join(root, fn), CODE)
                    if stem not in docs:
                        docs[stem] = (repo, rel)
    return docs


def parse_wholeness():
    txt = open(WHOLENESS, encoding="utf-8").read()
    out = {}
    pat = re.compile(
        r'\{:mission\s+"([^"]+)"\s+:class\s+:(\S+)\s+:L\s+([\d.]+)\s+:T\s+(\d+)\s+:H\s+([\d.]+)')
    for m in pat.finditer(txt):
        out[m.group(1)] = {"class": m.group(2), "L": float(m.group(3)),
                           "T": int(m.group(4)), "H": float(m.group(5))}
    return out


def parse_edges():
    """mission stem -> set of var ids (touches edges only)."""
    touches = defaultdict(set)
    with open(EDGES, encoding="utf-8") as f:
        for line in f:
            if '"touches"' not in line:
                continue
            try:
                a, b, kind = json.loads(line)
            except ValueError:
                continue
            if kind == "touches" and a.startswith("mission-doc:"):
                touches[a[len("mission-doc:"):]].add(b)
    return touches


def repo_source_roots(repo):
    roots = []
    deps = os.path.join(CODE, repo, "deps.edn")
    if os.path.isfile(deps):
        txt = open(deps, encoding="utf-8", errors="replace").read()
        m = re.search(r":paths\s*\[([^\]]*)\]", txt)
        if m:
            roots.extend(re.findall(r'"([^"]+)"', m.group(1)))
    roots.extend(DEFAULT_ROOTS)
    seen, out = set(), []
    for r in roots:
        r = r.rstrip("/") or "."
        ap = os.path.join(CODE, repo, r)
        if r not in seen and os.path.isdir(ap):
            seen.add(r)
            out.append(r)
    return out


def build_ns_index(repos):
    """ns name -> (repo, relpath-from-repo). First repo wins."""
    idx = {}
    for repo in repos:
        for root in repo_source_roots(repo):
            aroot = os.path.join(CODE, repo, root)
            for dirpath, dirs, files in os.walk(aroot):
                dirs[:] = [d for d in dirs if not d.startswith(".")
                           and d not in ("node_modules", "target", ".git")]
                for fn in files:
                    if not fn.endswith(EXTS):
                        continue
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, aroot)
                    noext = os.path.splitext(rel)[0]
                    ns = noext.replace(os.sep, ".").replace("_", "-")
                    if ns not in idx:
                        idx[ns] = (repo, os.path.relpath(full, os.path.join(CODE, repo)))
    return idx


def resolve_var(var_id, ns_index):
    """var id '<ns>/<name>' -> (repo, relpath) or None. Odd prefixes unresolved."""
    parts = var_id.split("/")
    if len(parts) != 2:
        return None
    ns = parts[0]
    if not ns or not re.match(r"^[a-zA-Z][\w.*+!?-]*$", ns):
        return None
    return ns_index.get(ns)


def git_file_commits(repo, paths):
    """relpath -> [(sha, ct)] via per-file git log (same semantics as the
    spec formula: git -C <repo> log --format=%H%x09%ct -- <path>)."""
    out = {}
    for rel in sorted(paths):
        p = subprocess.run(
            ["git", "-C", os.path.join(CODE, repo), "log",
             "--format=%H%x09%ct", "--", rel],
            capture_output=True, text=True)
        if p.returncode != 0:
            sys.stderr.write(f"git log failed {repo}/{rel}: {p.stderr[:200]}\n")
            continue
        rows = []
        for line in p.stdout.splitlines():
            if "\t" in line:
                sha, ct = line.split("\t")
                rows.append((sha, int(ct)))
        out[rel] = rows
    return out


def doc_git(repo, rel_from_code):
    """(weekly[26], last_commit_iso) for the doc file, --follow."""
    rel_repo = os.path.relpath(os.path.join(CODE, rel_from_code), os.path.join(CODE, repo))
    p = subprocess.run(
        ["git", "-C", os.path.join(CODE, repo), "log", "--follow",
         "--format=%ct", "--", rel_repo],
        capture_output=True, text=True)
    cts = [int(x) for x in p.stdout.split() if x.strip().isdigit()] if p.returncode == 0 else []
    return cts


def weekly_buckets(cts, now):
    buckets = [0] * WEEKS
    start = now - WEEKS * 7 * DAY
    for ct in cts:
        i = int((ct - start) // (7 * DAY))
        if 0 <= i < WEEKS:
            buckets[i] += 1
    return buckets


def indent_complexity(repo, rel):
    """Total leading-whitespace indent units (tab=4 cols, unit=2 cols)."""
    total = 0.0
    try:
        with open(os.path.join(CODE, repo, rel), encoding="utf-8",
                  errors="replace") as f:
            for line in f:
                cols = 0
                for ch in line:
                    if ch == " ":
                        cols += 1
                    elif ch == "\t":
                        cols += 4
                    else:
                        break
                total += cols / 2.0
    except OSError:
        pass
    return total


def doc_fields(path):
    """(status_line, lifecycle_phases) from a mission doc."""
    status = None
    phases = set()
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if status is None and "Status:" in line:
                    s = line.split("Status:", 1)[1]
                    s = s.strip().lstrip("*").strip()
                    status = s[:120]
                ls = line.lstrip()
                if ls.startswith("#"):
                    up = line.upper()
                    for ph in PHASES:
                        if ph in up:
                            phases.add(ph)
    except OSError:
        pass
    return status, len(phases)


def main():
    t0 = time.time()
    now = int(time.time())
    cutoff90 = now - 90 * DAY

    repos = canonical_repos()
    sys.stderr.write(f"canonical repos: {repos}\n")
    docs = find_docs(repos)
    wholeness = parse_wholeness()
    touches = parse_edges()
    carpet = json.load(open(CARPET))
    ns_index = build_ns_index(repos)
    sys.stderr.write(f"docs={len(docs)} wholeness={len(wholeness)} "
                     f"touched={len(touches)} ns_index={len(ns_index)}\n")

    # Resolve all touched vars
    mission_files = {}      # stem -> set of (repo, relpath)
    mission_vars = {}       # stem -> (n_vars, n_unresolved)
    all_files = defaultdict(set)  # repo -> set of relpaths
    n_vars_total = n_res_total = 0
    for stem, varids in touches.items():
        files = set()
        unres = 0
        for v in varids:
            r = resolve_var(v, ns_index)
            n_vars_total += 1
            if r is None:
                unres += 1
            else:
                n_res_total += 1
                files.add(r)
                all_files[r[0]].add(r[1])
        mission_files[stem] = files
        mission_vars[stem] = (len(varids), unres)

    # Git history per repo over the union of resolved files
    file_commits = {}  # (repo, rel) -> [(sha, ct)]
    for repo, paths in all_files.items():
        t = time.time()
        fc = git_file_commits(repo, paths)
        file_commits.update({(repo, k): v for k, v in fc.items()})
        sys.stderr.write(f"git log {repo}: {len(paths)} files -> "
                         f"{len(fc)} with history ({time.time()-t:.1f}s)\n")

    # Complexity cache per file
    cplx_cache = {}

    def cplx(rf):
        if rf not in cplx_cache:
            cplx_cache[rf] = indent_complexity(*rf)
        return cplx_cache[rf]

    # file -> missions sharing it (for coupling)
    file_to_missions = defaultdict(set)
    for stem, files in mission_files.items():
        for rf in files:
            file_to_missions[rf].add(stem)

    stems = set(docs) | set(wholeness) | set(touches) | set(carpet)
    missions = []
    for stem in sorted(stems):
        row = {"mission": stem}
        d = docs.get(stem)
        if d:
            repo, rel = d
            row["doc"] = rel
            status, nphases = doc_fields(os.path.join(CODE, rel))
            row["status_line"] = status
            row["lifecycle_phases"] = nphases
            cts = doc_git(repo, rel)
            row["doc_commits_weekly"] = weekly_buckets(cts, now)
            row["doc_last_commit"] = (datetime.fromtimestamp(max(cts), timezone.utc)
                                      .date().isoformat() if cts else None)
        else:
            row["doc"] = None
            row["status_line"] = None
            row["lifecycle_phases"] = None
            row["doc_commits_weekly"] = None
            row["doc_last_commit"] = None

        w = wholeness.get(stem)
        row["wholeness"] = w if w else None

        if stem in touches:
            nvars, unres = mission_vars[stem]
            files = mission_files[stem]
            commits = {}  # sha -> ct
            weekly_cts = []
            for rf in files:
                for sha, ct in file_commits.get(rf, []):
                    commits[sha] = ct
            weekly_cts = list(commits.values())
            row["code"] = {
                "vars_touched": nvars,
                "files_resolved": len(files),
                "vars_unresolved": unres,
                "commits_90d": sum(1 for ct in commits.values() if ct >= cutoff90),
                "commits_all": len(commits),
                "weekly": weekly_buckets(weekly_cts, now),
                "complexity": round(sum(cplx(rf) for rf in files) / len(files), 2) if files else 0.0,
            }
        else:
            row["code"] = None

        # coupling: top 5 other missions by shared resolved files
        shared = defaultdict(int)
        for rf in mission_files.get(stem, ()):  # noqa
            for other in file_to_missions[rf]:
                if other != stem:
                    shared[other] += 1
        row["coupling"] = [[o, c] for o, c in
                           sorted(shared.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
        row["carpet"] = stem in carpet
        missions.append(row)

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {os.path.relpath(p, FUTON6): sha256(p)
                    for p in (WHOLENESS, EDGES, CARPET)},
        "resolution": {"vars": n_vars_total, "resolved": n_res_total},
        "missions": missions,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    sys.stderr.write(f"wrote {OUT} in {time.time()-t0:.1f}s; "
                     f"resolution {n_res_total}/{n_vars_total}\n")


if __name__ == "__main__":
    main()
