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


def mission_code_files(var_ids, ns_index):
    """Resolve one mission's touched vars to files.
    Returns (sorted_files, n_unresolved) where sorted_files is a deduplicated,
    sorted list of [repo, relpath] pairs."""
    files = set()
    unres = 0
    for v in var_ids:
        r = resolve_var(v, ns_index)
        if r is None:
            unres += 1
        else:
            files.add(r)
    return [list(rf) for rf in sorted(files)], unres


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
    """(cts, shas) for the doc file, --follow."""
    rel_repo = os.path.relpath(os.path.join(CODE, rel_from_code), os.path.join(CODE, repo))
    p = subprocess.run(
        ["git", "-C", os.path.join(CODE, repo), "log", "--follow",
         "--format=%H %ct", "--", rel_repo],
        capture_output=True, text=True)
    cts, shas = [], {}
    if p.returncode == 0:
        for line in p.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                shas[parts[0]] = int(parts[1])
                cts.append(int(parts[1]))
    return cts, shas


# ---------------------------------------------------------------- v05 (futon1b)

V05_BASE = "http://127.0.0.1:7073/api/alpha/hyperedges"
V05_CACHE = "/tmp/v05-cache"
V05_LIMIT = 1000  # server rejects >1000 (:invalid-limit)


def _v05_cursor(txt):
    m = re.search(r':next-cursor\s+"((?:[^"\\]|\\.)*)"', txt)
    return m.group(1) if m else None


def v05_pull(kind):
    """Page futon1b hyperedges of the given type, caching raw pages in
    /tmp/v05-cache. Resume-capable: reuses cached pages and continues from
    the last page's cursor. Returns list of raw page texts (EDN)."""
    import urllib.request
    import urllib.parse
    cdir = os.path.join(V05_CACHE, kind)
    os.makedirs(cdir, exist_ok=True)
    pages = []
    i = 0
    while os.path.isfile(os.path.join(cdir, f"page-{i:04d}.edn")):
        pages.append(open(os.path.join(cdir, f"page-{i:04d}.edn"),
                          encoding="utf-8").read())
        i += 1
    if os.path.isfile(os.path.join(cdir, "COMPLETE")):
        sys.stderr.write(f"v05 {kind}: {len(pages)} cached pages (complete)\n")
        return pages
    cursor = _v05_cursor(pages[-1]) if pages else None
    sys.stderr.write(f"v05 {kind}: resuming at page {i}\n")
    failures = 0
    while True:
        q = {"type": f"code/v05/{kind}", "limit": str(V05_LIMIT)}
        if cursor:
            q["after"] = cursor
        url = V05_BASE + "?" + urllib.parse.urlencode(q)
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                txt = r.read().decode("utf-8")
        except Exception as e:
            failures += 1
            sys.stderr.write(f"v05 {kind} page {i}: {e} (failure {failures})\n")
            if failures > 60:
                raise
            time.sleep(min(60, 5 * failures))
            continue
        if '"{:error' in txt[:20] or txt.startswith("{:error"):
            failures += 1
            sys.stderr.write(f"v05 {kind} page {i}: server error {txt[:120]}\n")
            if failures > 60:
                raise RuntimeError(txt[:200])
            time.sleep(min(60, 5 * failures))
            continue
        failures = 0
        open(os.path.join(cdir, f"page-{i:04d}.edn"), "w", encoding="utf-8").write(txt)
        pages.append(txt)
        n = txt.count(':hx/id "hx:code/v05/')
        new = _v05_cursor(txt)
        sys.stderr.write(f"v05 {kind} page {i}: {n} edges\n")
        if not new or n == 0 or new == cursor:
            break
        cursor = new
        i += 1
    open(os.path.join(cdir, "COMPLETE"), "w").write(str(len(pages)))
    return pages


def _unescape_edn(s):
    return s.replace('\\"', '"').replace("\\\\", "\\")


def v05_parse_commits(pages):
    """sha -> {"repo":..,"ts":..,"subject":..}"""
    out = {}
    for txt in pages:
        chunks = txt.split('"hx:code/v05/commit:')
        for ch in chunks[1:]:
            m = re.search(r':hx/endpoints\s+\["([0-9a-f]+)"', ch)
            if not m:
                continue
            sha = m.group(1)
            ts = re.search(r':timestamp\s+(\d+)', ch)
            repo = re.search(r':repo\s+"([^"]+)"', ch)
            subj = re.search(r':subject\s+"((?:[^"\\]|\\.)*)"', ch)
            out[sha] = {
                "repo": repo.group(1) if repo else None,
                "ts": int(ts.group(1)) if ts else None,
                "subject": _unescape_edn(subj.group(1)) if subj else "",
            }
    return out


def v05_parse_edits(pages):
    """list of (sha, var_id, repo)"""
    out = []
    for txt in pages:
        chunks = txt.split('"hx:code/v05/edits:')
        for ch in chunks[1:]:
            m = re.search(r':hx/endpoints\s+\["([0-9a-f]+)"\s+"((?:[^"\\]|\\.)*)"', ch)
            if not m:
                continue
            repo = re.search(r':repo\s+"([^"]+)"', ch)
            out.append((m.group(1), _unescape_edn(m.group(2)),
                        repo.group(1) if repo else None))
    return out


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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="per-mission activity and code churn")
    ap.add_argument("--no-v05", action="store_true",
                    help="skip the futon1b code/v05 pull; missions then carry no code_v05 "
                         "and the output's v05 block says it was not pulled")
    args = ap.parse_args(argv)
    use_v05 = not args.no_v05
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
        files_list, unres = mission_code_files(varids, ns_index)
        files = {tuple(rf) for rf in files_list}
        n_vars_total += len(varids)
        n_res_total += len(varids) - unres
        for repo, rel in files:
            all_files[repo].add(rel)
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

    # ---------------- v05: commit->mission links and per-var churn from futon1b
    commit_pages = v05_pull("commit") if use_v05 else []
    edit_pages = v05_pull("edits") if use_v05 else []
    v05_commits = v05_parse_commits(commit_pages)      # sha -> {repo,ts,subject}
    v05_edits = v05_parse_edits(edit_pages)            # [(sha, var, repo)]
    sys.stderr.write(f"v05: {len(v05_commits)} commits, {len(v05_edits)} edits\n")

    newest_by_repo = {}
    for c in v05_commits.values():
        if c["repo"] and c["ts"]:
            newest_by_repo[c["repo"]] = max(newest_by_repo.get(c["repo"], 0), c["ts"])

    sha_vars = defaultdict(list)   # sha -> [var_id]
    for sha, var, _repo in v05_edits:
        sha_vars[sha].append(var)

    # rule (a): subject contains the mission stem as a token (case-sensitive)
    stem_re = {s: re.compile(r"(?<![A-Za-z0-9-])" + re.escape(s) + r"(?![A-Za-z0-9-])")
               for s in stems}
    subj_hits = defaultdict(set)   # stem -> shas
    for sha, c in v05_commits.items():
        subj = c["subject"]
        if not subj:
            continue
        for s, rx in stem_re.items():
            if rx.search(subj):
                subj_hits[s].add(sha)
    sys.stderr.write(f"v05 subject-rule hits for {len(subj_hits)} stems\n")

    # rule (b): sha in git log --follow -- <mission doc>; collect while iterating
    doc_sha_map = {}               # stem -> {sha: ct}
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
            cts, dshas = doc_git(repo, rel)
            doc_sha_map[stem] = dshas
            row["doc_commits_weekly"] = weekly_buckets(cts, now)
            row["doc_last_commit"] = (datetime.fromtimestamp(max(cts), timezone.utc)
                                      .date().isoformat() if cts else None)
        else:
            row["doc"] = None
            row["status_line"] = None
            row["lifecycle_phases"] = None
            row["doc_commits_weekly"] = None
            row["doc_last_commit"] = None
            doc_sha_map[stem] = set()

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
                "files": sorted([list(rf) for rf in files]),
                # No resolved file means no git history was read: null, not zero churn.
                "commits_90d": sum(1 for ct in commits.values() if ct >= cutoff90) if files else None,
                "commits_all": len(commits) if files else None,
                "weekly": weekly_buckets(weekly_cts, now) if files else None,
                "complexity": round(sum(cplx(rf) for rf in files) / len(files), 2) if files else None,
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

    if use_v05:
        # ---------------- code_v05 per mission
        var_to_missions = defaultdict(set)   # edited var -> missions (v05)
        v05_per_mission = {}
        for row in missions:
            stem = row["mission"]
            subj_shas = subj_hits.get(stem, set())
            doc_shas = set(doc_sha_map.get(stem, {}))
            linked = subj_shas | doc_shas
            if not linked:
                v05_per_mission[stem] = None
                continue
            ts_of = {}
            for sha in linked:
                c = v05_commits.get(sha)
                if c and c["ts"]:
                    ts_of[sha] = c["ts"]
                elif sha in doc_sha_map.get(stem, {}):
                    ts_of[sha] = doc_sha_map[stem][sha]
            edits = []          # (sha, var)
            var_counts = defaultdict(int)
            for sha in linked:
                for var in sha_vars.get(sha, ()):  # edits only exist for ingested commits
                    edits.append((sha, var))
                    var_counts[var] += 1
                    var_to_missions[var].add(stem)
            edit_cts = [ts_of[sha] for sha, _ in edits if sha in ts_of]
            last_ts = max(ts_of.values()) if ts_of else None
            v05_per_mission[stem] = {
                "commits": len(linked),
                "by_rule": {"subject": len(subj_shas), "doc": len(doc_shas)},
                "commits_after_ingest": sum(1 for sha in linked if sha not in v05_commits),
                "vars_edited": len(var_counts),
                "edits_90d": sum(1 for ct in edit_cts if ct >= cutoff90),
                "weekly": weekly_buckets(edit_cts, now),
                "last_commit": (datetime.fromtimestamp(last_ts, timezone.utc)
                                .date().isoformat() if last_ts else None),
                "top_vars": [[v, n] for v, n in
                             sorted(var_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]],
            }
        # coupling: shared edited vars
        for row in missions:
            stem = row["mission"]
            blk = v05_per_mission[stem]
            if blk is None:
                row["code_v05"] = None
                continue
            shared = defaultdict(int)
            my_vars = {var for sha in (subj_hits.get(stem, set()) | set(doc_sha_map.get(stem, {})))
                       for var in sha_vars.get(sha, ())}
            for var in my_vars:
                for other in var_to_missions[var]:
                    if other != stem:
                        shared[other] += 1
            blk["coupling"] = [[o, c] for o, c in
                               sorted(shared.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
            row["code_v05"] = blk

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {os.path.relpath(p, FUTON6): sha256(p)
                    for p in (WHOLENESS, EDGES, CARPET)},
        "resolution": {"vars": n_vars_total, "resolved": n_res_total},
        # Not pulled (--no-v05): say so; missions carry no code_v05 key, so the
        # absence cannot read as "no linked commits".
        "v05": ({"pulled": True, "commits": len(v05_commits), "edits": len(v05_edits),
                 "newest_commit_by_repo": {
                     r: datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
                     for r, ts in sorted(newest_by_repo.items())}}
                if use_v05 else {"pulled": False}),
        "missions": missions,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    sys.stderr.write(f"wrote {OUT} in {time.time()-t0:.1f}s; "
                     f"resolution {n_res_total}/{n_vars_total}\n")


if __name__ == "__main__":
    main()
