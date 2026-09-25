#!/usr/bin/env python3
"""Build session -> commit index and operator-turn -> commit join.

Reads codex/claude transcripts, Kimi/Zai turn-round evidence in futon1b,
operator turn records, and local git repos;
writes data/session-commit-index.json. Stdlib only.
"""
import json, os, re, subprocess, sys, glob, datetime
from collections import defaultdict

HOME = os.path.expanduser("~")
CODE = os.path.join(HOME, "code")
OUT = os.path.join(CODE, "futon6", "data", "session-commit-index.json")
SINCE = "2026-09-01"
SINCE_DT = datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc)
TURN_DIR = os.path.join(HOME, ".emacs-graph", "session-turn-analysis")

COMMIT_RE = re.compile(r"\[[\w./\-]+ ([0-9a-f]{7,40})\] (.+)")


def find_repos():
    repos = {}
    for name in sorted(os.listdir(CODE)):
        if re.fullmatch(r"futon[\w]*", name) or name in ("voxterm", "marimo-zone"):
            p = os.path.join(CODE, name)
            if os.path.isdir(os.path.join(p, ".git")):
                repos[name] = p
    return repos


def repo_logs(repos):
    """short->[(full,repo,at,subject)], full->entry, and per-repo list since SINCE."""
    short_idx = defaultdict(list)
    full_idx = {}
    per_repo = defaultdict(list)
    for name, path in repos.items():
        try:
            out = subprocess.run(
                ["git", "-C", path, "log", "--since=" + SINCE,
                 "--format=%H%x09%h%x09%aI%x09%s"],
                capture_output=True, text=True, timeout=300).stdout
        except Exception:
            continue
        for line in out.splitlines():
            parts = line.split("\t", 3)
            if len(parts) < 4:
                continue
            full, short, at, subj = parts
            e = {"sha": full, "repo": name, "at": at, "subject": subj}
            short_idx[short].append(e)
            full_idx[full] = e
            per_repo[name].append(e)
    return short_idx, full_idx, per_repo


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def resolve_short(short, short_idx, full_idx):
    if short in full_idx:
        return full_idx[short], None
    cands = short_idx.get(short, [])
    repos_seen = {c["repo"] for c in cands}
    if len(repos_seen) == 1 and cands:
        return cands[0], None
    if len(repos_seen) > 1:
        return None, "ambiguous"
    return None, "absent"


def extract_m_message(cmd):
    """First line of commit message: heredoc -F -, or first -m argument."""
    h = re.search(r"-F\s*-\s*<<\s*\\?'?([A-Za-z_]+)'?[^\n]*\n(.*?)\n\\?\1\b", cmd, re.S)
    if h:
        body = h.group(2)
        return body.splitlines()[0].strip() or None
    m = re.search(r"git commit\b.*?-m\s*\"((?:[^\"\\]|\\.)*)\"", cmd, re.S)
    if not m:
        m = re.search(r"git commit\b.*?-m\s*'([^']*)'", cmd, re.S)
    if not m:
        m = re.search(r"git commit\b.*?-m\s*(\S+)", cmd, re.S)
    if m:
        msg = m.group(1).replace("\\\"", "\"").replace("\\n", "\n")
        return msg.splitlines()[0].strip() if msg.strip() else None
    return None


def extract_repo_path(cmd):
    m = re.search(r"git\s+-C\s+(\S+)", cmd)
    if m:
        return m.group(1)
    m = re.search(r"\bcd\s+([^\s;&|]+)", cmd)
    if m:
        return m.group(1)
    return None


def scan_codex(short_idx, full_idx, commits, counters):
    files = sorted(glob.glob(os.path.join(
        HOME, ".codex", "sessions", "2026", "*", "*", "*.jsonl")))
    n = 0
    for f in files:
        # date dir MM/DD
        parts = f.split(os.sep)
        try:
            mm, dd = int(parts[-3]), int(parts[-2])
            if (mm, dd) < (9, 1):
                continue
        except Exception:
            pass
        n += 1
        sid = None
        try:
            fh = open(f, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if sid is None and '"session_meta"' in line:
                    try:
                        d = json.loads(line)
                        sid = d["payload"].get("session_id") or d["payload"].get("id")
                    except Exception:
                        pass
                m = COMMIT_RE.search(line)
                if not m:
                    continue
                short, subj = m.group(1), m.group(2).strip()
                ts = None
                try:
                    ts = json.loads(line).get("timestamp")
                except Exception:
                    pass
                e, why = resolve_short(short, short_idx, full_idx)
                if e:
                    commits.append({"sha": e["sha"], "repo": e["repo"],
                                    "session": sid, "agent_kind": "codex",
                                    "at": e["at"], "subject": e["subject"],
                                    "match": "sha-printed"})
                    counters["sha-printed"] += 1
                else:
                    commits.append({"sha": short, "repo": None,
                                    "session": sid, "agent_kind": "codex",
                                    "at": ts, "subject": subj,
                                    "match": "sha-printed-unresolved:" + (why or "absent")})
    return n


def scan_claude(short_idx, full_idx, per_repo, commits, counters):
    files = sorted(glob.glob(os.path.join(
        HOME, ".claude", "projects", "*", "*.jsonl")))
    # pre-compact rotations: <uuid>.jsonl.pre-compact-<ts> hold the rotated-out
    # portion of the same session; scan them under the same session id.
    files += sorted(glob.glob(os.path.join(
        HOME, ".claude", "projects", "*", "*.jsonl.pre-compact-*")))
    n = 0
    path_to_repo = {}
    for rname, elist in per_repo.items():
        pass
    repo_by_path = {p: n2 for n2, p in REPOS.items()}
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        if sid.endswith(".jsonl"):  # pre-compact rotation
            sid = sid[:-len(".jsonl")]
        n += 1
        try:
            fh = open(f, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if "git commit" not in line:
                    # still catch printed shas cheaply
                    if "] " not in line:
                        continue
                printed = COMMIT_RE.search(line)
                has_commit_cmd = "git commit" in line
                if not printed and not has_commit_cmd:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts = d.get("timestamp")
                # walk content blocks for Bash tool_use
                content = d.get("message", {}).get("content")
                blocks = content if isinstance(content, list) else []
                handled_cmd = False
                for b in blocks:
                    if not (isinstance(b, dict) and b.get("type") == "tool_use"
                            and b.get("name") == "Bash"):
                        continue
                    cmd = b.get("input", {}).get("command", "")
                    if "git commit" not in cmd:
                        continue
                    handled_cmd = True
                    msg = extract_m_message(cmd)
                    rpath = extract_repo_path(cmd)
                    rname = repo_by_path.get(rpath) if rpath else None
                    if rname is None and rpath:
                        # basename fallback
                        base = os.path.basename(rpath.rstrip("/"))
                        if base in REPOS:
                            rname = base
                    match = None
                    t0 = parse_iso(ts)
                    if msg and t0:
                        repos_to_try = [rname] if rname else list(per_repo.keys())
                        hits = []
                        for rn in repos_to_try:
                            for e in per_repo.get(rn, []):
                                te = parse_iso(e["at"])
                                if te and abs((te - t0).total_seconds()) <= 600 \
                                        and e["subject"] == msg:
                                    hits.append(e)
                        uniq = {h["sha"]: h for h in hits}
                        if len(uniq) == 1:
                            match = next(iter(uniq.values()))
                        elif rname and hits:
                            match = hits[0]
                    if match:
                        commits.append({"sha": match["sha"], "repo": match["repo"],
                                        "session": sid, "agent_kind": "claude",
                                        "at": match["at"], "subject": match["subject"],
                                        "match": "subject+time"})
                        counters["subject+time"] += 1
                # printed sha lines (tool results etc.)
                if printed:
                    short, subj = printed.group(1), printed.group(2).strip()
                    e, why = resolve_short(short, short_idx, full_idx)
                    if e:
                        commits.append({"sha": e["sha"], "repo": e["repo"],
                                        "session": sid, "agent_kind": "claude",
                                        "at": e["at"], "subject": e["subject"],
                                        "match": "sha-printed"})
                        counters["sha-printed"] += 1
                    elif not handled_cmd:
                        commits.append({"sha": short, "repo": None,
                                        "session": sid, "agent_kind": "claude",
                                        "at": ts, "subject": subj,
                                        "match": "sha-printed-unresolved:" + (why or "absent")})
    return n


EVIDENCE = os.environ.get("FUTON1B_URL", "http://127.0.0.1:7073") + "/api/alpha/evidence"
# Kimi and Zai keep no transcript on disk; their turns live only in futon1b as
# :transcript :turn-round evidence, one entry per round with each tool call's
# full args and a preview of its result. Retired seats are not on the roster,
# so probe a fixed range and skip authors with no entries.
EVIDENCE_AUTHORS = [f"kimi-{i}" for i in range(1, 31)] + [f"zai-{i}" for i in range(1, 11)]
EDN_ENTRY = re.compile(r"\{:evidence/body ")
EDN_STR = r'"((?:[^"\\]|\\.)*)"'


def edn_unescape(s):
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s)


def evidence_pages(author):
    """Yield raw EDN entry chunks for AUTHOR's turn rounds since SINCE, newest first."""
    import urllib.parse, urllib.request
    before = None
    while True:
        q = {"author": author, "tags": "turn-round", "limit": "1000",
             "since": SINCE + "T00:00:00Z"}
        if before:
            q["before"] = before
        try:
            with urllib.request.urlopen(EVIDENCE + "?" + urllib.parse.urlencode(q), timeout=300) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception as e:
            print(f"evidence {author}: {e}", file=sys.stderr)
            return
        starts = [m.start() for m in EDN_ENTRY.finditer(text)]
        chunks = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
        if not chunks:
            return
        yield from chunks
        ats = re.findall(r':evidence/at "([^"]+)"', chunks[-1])
        if len(chunks) < 1000 or not ats or ats[-1] == before:
            return
        before = ats[-1]


def scan_evidence(short_idx, full_idx, per_repo, commits, counters):
    """Commits made by Kimi/Zai seats, read from their futon1b turn-round evidence."""
    n_sessions = defaultdict(set)
    seen = set()
    for author in EVIDENCE_AUTHORS:
        kind = author.split("-")[0]
        for chunk in evidence_pages(author):
            if "git commit" not in chunk and "] " not in chunk:
                continue
            eid = re.search(r':evidence/id "([^"]+)"', chunk)
            if not eid or eid.group(1) in seen:
                continue
            seen.add(eid.group(1))
            sid = re.search(r':evidence/session-id "([^"]+)"', chunk)
            sid = sid.group(1) if sid else None
            n_sessions[kind].add(sid)
            at = re.search(r':evidence/at "([^"]+)"', chunk)
            t0 = parse_iso(at.group(1)) if at else None
            text = edn_unescape(chunk)
            # A shell call's args are an EDN map printed into a string, so the
            # command inside is escaped twice.
            matched = set()
            for cmd in re.findall(r':command ' + EDN_STR, text):
                cmd = edn_unescape(cmd)
                if "git commit" not in cmd:
                    continue
                msg = extract_m_message(cmd)
                rpath = extract_repo_path(cmd)
                rname = os.path.basename(rpath.rstrip("/")) if rpath else None
                if not (msg and t0):
                    continue
                hits = {e["sha"]: e for rn in ([rname] if rname in per_repo else list(per_repo))
                        for e in per_repo.get(rn, [])
                        if e["subject"] == msg and parse_iso(e["at"])
                        and abs((parse_iso(e["at"]) - t0).total_seconds()) <= 600}
                if len(hits) == 1:
                    e = next(iter(hits.values()))
                    matched.add(e["sha"])
                    commits.append({"sha": e["sha"], "repo": e["repo"], "session": sid,
                                    "agent_kind": kind, "at": e["at"], "subject": e["subject"],
                                    "match": "subject+time"})
                    counters["subject+time"] += 1
            for m in COMMIT_RE.finditer(text):
                e, why = resolve_short(m.group(1), short_idx, full_idx)
                if e and e["sha"] not in matched:
                    matched.add(e["sha"])
                    commits.append({"sha": e["sha"], "repo": e["repo"], "session": sid,
                                    "agent_kind": kind, "at": e["at"], "subject": e["subject"],
                                    "match": "sha-printed"})
                    counters["sha-printed"] += 1
    return {k: len(v) for k, v in n_sessions.items()}


def load_turns(commits):
    turns = []
    for f in sorted(glob.glob(os.path.join(TURN_DIR, "turn-*.json"))):
        if f.endswith(".analysis.json") or f.endswith(".candidates.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        patterns = None
        af = f + ".analysis.json"
        if os.path.exists(af):
            try:
                a = json.load(open(af))
                prefs = set()
                # Pattern refs live on each sentence's fragments, as {"id": ...}.
                for s in a.get("sentences", []):
                    for fr in s.get("fragments", []) or []:
                        for r in fr.get("pattern_refs", []) or []:
                            rid = r.get("id") if isinstance(r, dict) else r
                            if rid:
                                prefs.add(str(rid))
                if prefs:
                    patterns = sorted(prefs)
            except Exception:
                pass
        turns.append({"turn_id": d.get("turn_id"),
                      "session": d.get("session_id"),
                      "at": d.get("created_at"),
                      "patterns": patterns,
                      "_agent": d.get("agent_id")})
    # join commits: per session, sorted by time
    by_session = defaultdict(list)
    for c in commits:
        if c.get("session"):
            by_session[c["session"]].append(c)
    out_turns = []
    for sess, tgroup in defaultdict(list, {t["session"]: [] for t in turns}).items():
        pass
    turns_by_session = defaultdict(list)
    for t in turns:
        turns_by_session[t["session"]].append(t)
    for sess, tg in turns_by_session.items():
        tg.sort(key=lambda t: t["at"] or "")
        clist = sorted(by_session.get(sess, []), key=lambda c: c["at"] or "")
        for i, t in enumerate(tg):
            t0 = parse_iso(t["at"])
            t1 = parse_iso(tg[i + 1]["at"]) if i + 1 < len(tg) else None
            cshas = [c["sha"] for c in clist
                     if (t0 is None or (parse_iso(c["at"]) or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)) >= t0)
                     and (t1 is None or (parse_iso(c["at"]) or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)) < t1)]
            out_turns.append({"turn_id": t["turn_id"], "session": sess,
                              "at": t["at"], "commits": cshas,
                              "patterns": t["patterns"]})
    return out_turns


def main():
    global REPOS
    REPOS = find_repos()
    short_idx, full_idx, per_repo = repo_logs(REPOS)
    commits = []
    counters = defaultdict(int)
    n_codex = scan_codex(short_idx, full_idx, commits, counters)
    n_claude = scan_claude(short_idx, full_idx, per_repo, commits, counters)
    n_evidence = scan_evidence(short_idx, full_idx, per_repo, commits, counters)

    # dedupe by (sha, session)
    seen = set()
    deduped = []
    for c in commits:
        key = (c["sha"], c.get("session"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    commits = deduped

    turns = load_turns(commits)

    repo_commits = {r: len(per_repo.get(r, [])) for r in REPOS}
    # Distinct commits per repo: a commit quoted in two sessions counts once.
    attributed_shas = defaultdict(set)
    for c in commits:
        if c.get("repo") and len(c["sha"]) == 40:
            attributed_shas[c["repo"]].add(c["sha"])
    attributed = {r: len(v) for r, v in attributed_shas.items()}
    coverage = {
        "sessions_scanned": {"codex": n_codex, "claude": n_claude,
                             "kimi": n_evidence.get("kimi", 0), "zai": n_evidence.get("zai", 0)},
        "commits_by_agent_kind": {k: len({c["sha"] for c in commits if c["agent_kind"] == k and c.get("repo")})
                                  for k in ("codex", "claude", "kimi", "zai")},
        "commits_found": len(commits),
        "by_match": {"sha-printed": sum(1 for c in commits if c["match"] == "sha-printed"),
                     "subject+time": sum(1 for c in commits if c["match"] == "subject+time")},
        "repo_commits_since_0901": repo_commits,
        "attributed_since_0901": dict(attributed),
    }
    out = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "coverage": coverage, "commits": commits, "turns": turns}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)
    print(json.dumps(coverage, indent=1))
    n_with = sum(1 for t in turns if t["commits"])
    print("turns:", len(turns), "with >=1 commit:", n_with)


if __name__ == "__main__":
    main()
