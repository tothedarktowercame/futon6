#!/usr/bin/env python3
"""namespace_influence.py — namespace influence discovery (M-live-efe-map).

Measures namespace influence from the futon1b substrate's code call graph
(code/v05/calls, code/v05/namespace) and probes how namespaces could attach
to missions. Discovery only: no map, no rendering.

Output: $FUTON6_STORAGE_ROOT/futon6/archipelago/influence/namespace-influence.json

Pure functions (parse_endpoint, base_repo, aggregate, pagerank) are
importable for tests; fetch and main do the I/O.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import futon6_config as config
    ROOT = config.ROOT
except Exception:
    ROOT = Path(__file__).resolve().parent.parent

STORAGE_ROOT = Path(os.environ.get(
    "FUTON6_STORAGE_ROOT", str(Path.home() / "storage")))
OUT = STORAGE_ROOT / "futon6" / "archipelago" / "influence" / "namespace-influence.json"
POS_EMBED = ROOT / "data" / "mission-carpet-pos-embed.json"

SUBSTRATE = (
    os.environ.get("FUTON_SUBSTRATE_URL")
    or os.environ.get("FUTON1A_URL")
    or "http://127.0.0.1:7073"
).rstrip("/")
BASE = f"{SUBSTRATE}/api/alpha/hyperedges"
LIMIT = 1000
FETCH_TIMEOUT_S = 120
BUDGET_S = int(os.environ.get("NAMESPACE_INFLUENCE_BUDGET_S", str(25 * 60)))

# Sibling checkouts live beside futon6 (FUTON_CODE_ROOT overrides).
CODE_ROOT = Path(os.environ.get("FUTON_CODE_ROOT", str(Path(__file__).resolve().parents[2])))
REPO_SUFFIX_RE = re.compile(r"-(py-|cljs-|clj-)?d\d*$")


# ---------------------------------------------------------------- pure pieces

def base_repo(repo: str) -> str:
    """futon6-py-d -> futon6, futon3c-d -> futon3c."""
    return REPO_SUFFIX_RE.sub("", repo)


def parse_endpoint(eid: str):
    """"<repo>/<ns>/<var>" -> (repo, ns, var). A var name containing "/"
    stays whole: only the first two slashes split."""
    parts = eid.split("/", 2)
    if len(parts) < 3:
        return parts[0], parts[1] if len(parts) > 1 else "", ""
    return parts[0], parts[1], parts[2]


def aggregate(call_rows):
    """call_rows: iterable of (caller_eid, callee_eid).

    Returns (namespaces, graph):
      namespaces: {(repo, ns): {"vars": set, "calls_out": int, "calls_in": int,
                                "calls_in_cross_ns": int,
                                "calls_in_cross_repo": int}}
      graph: {(repo, ns): {(repo, ns): weight}}  (self-loops dropped)
    """
    namespaces = defaultdict(lambda: {
        "vars": set(), "calls_out": 0, "calls_in": 0,
        "calls_in_cross_ns": 0, "calls_in_cross_repo": 0})
    graph = defaultdict(lambda: defaultdict(int))
    for caller, callee in call_rows:
        crepo, cns, cvar = parse_endpoint(caller)
        drepo, dns, dvar = parse_endpoint(callee)
        src, dst = (crepo, cns), (drepo, dns)
        namespaces[src]["vars"].add(cvar)
        namespaces[dst]["vars"].add(dvar)
        namespaces[src]["calls_out"] += 1
        namespaces[dst]["calls_in"] += 1
        if src != dst:
            namespaces[dst]["calls_in_cross_ns"] += 1
            graph[src][dst] += 1
        if crepo != drepo:
            namespaces[dst]["calls_in_cross_repo"] += 1
    return namespaces, graph


def pagerank(graph, damping=0.85, iters=200, tol=1e-12):
    """Weighted PageRank over the namespace graph. graph: {src: {dst: w}}.
    Sinks redistribute to all nodes. Returns {node: rank} summing to 1."""
    nodes = set(graph)
    for src, dsts in graph.items():
        nodes.update(dsts)
    nodes = sorted(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    rank = {u: 1.0 / n for u in nodes}
    out_w = {u: sum(graph.get(u, {}).values()) for u in nodes}
    for _ in range(iters):
        new = {u: (1.0 - damping) / n for u in nodes}
        dangling = sum(rank[u] for u in nodes if out_w[u] == 0)
        for u in nodes:
            if out_w[u] == 0:
                continue
            for v, w in graph.get(u, {}).items():
                new[v] += damping * rank[u] * w / out_w[u]
        for u in nodes:
            new[u] += damping * dangling / n
        delta = sum(abs(new[u] - rank[u]) for u in nodes)
        rank = new
        if delta < tol:
            break
    total = sum(rank.values())
    return {u: r / total for u, r in rank.items()}


# ---------------------------------------------------------------- fetch

class BudgetExceeded(Exception):
    pass


def fetch_hyperedges(hx_type, fields, deadline, gaps):
    """Sequential paged fetch; one retry per failed page, then record a gap."""
    rows = []
    after = None
    seen = set()
    while True:
        if time.monotonic() > deadline:
            gaps.append({"type": hx_type, "reason": "budget-exceeded",
                         "after": after, "rows_so_far": len(rows)})
            raise BudgetExceeded()
        params = {"type": hx_type, "limit": LIMIT, "fields": fields}
        if after:
            params["after"] = after
        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{BASE}?{q}",
                                     headers={"Accept": "application/json"})
        page = None
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
                    page = json.load(r)
                break
            except Exception as e:  # noqa: BLE001
                err = str(e)
                time.sleep(2)
        if page is None:
            gaps.append({"type": hx_type, "reason": f"page failed twice: {err}",
                         "after": after, "rows_so_far": len(rows)})
            return rows
        rows.extend(page.get("hyperedges", []))
        cursor = page.get("next-cursor")
        if not cursor:
            return rows
        if cursor in seen:
            gaps.append({"type": hx_type, "reason": "stalled cursor",
                         "after": after, "rows_so_far": len(rows)})
            return rows
        seen.add(cursor)
        after = cursor


# ---------------------------------------------------------------- scratch detection

_worktree_cache = {}


def is_scratch(source_file: str, base: str) -> bool:
    """A source-file is a scratch/worktree copy when it is not under the
    repo's canonical tree (CODE_ROOT/<base>) or sits under a nested
    git worktree dir directly below that root (e.g. zai-7-write-rate)."""
    if not source_file:
        return True
    root = CODE_ROOT / base
    try:
        rel = Path(source_file).relative_to(root)
    except ValueError:
        return True
    if len(rel.parts) < 2:
        return False
    first = rel.parts[0]
    key = (base, first)
    if key not in _worktree_cache:
        _worktree_cache[key] = (root / first / ".git").exists()
    return _worktree_cache[key]


# ---------------------------------------------------------------- main

def props_of(h):
    """hx/props is usually a dict; tolerate strings/other shapes."""
    p = h.get("hx/props")
    return p if isinstance(p, dict) else {}


def main():
    started = time.monotonic()
    deadline = started + BUDGET_S
    gaps = []

    calls_rows, ns_rows = [], []
    try:
        calls_rows = fetch_hyperedges(
            "code/v05/calls", "hx/endpoints,hx/props", deadline, gaps)
        ns_rows = fetch_hyperedges(
            "code/v05/namespace", "hx/endpoints,hx/props", deadline, gaps)
    except BudgetExceeded:
        pass

    repos_seen = sorted({
        (props_of(h)).get("repo") or ""
        for h in calls_rows + ns_rows} - {""})
    repo_map = {r: base_repo(r) for r in repos_seen}

    # --- influence aggregation
    call_pairs = []
    scratch_rows = 0
    missing_source_file_rows = 0
    scratch_examples = []
    cross_repo_endpoint_rows = 0
    for h in calls_rows:
        ep = h.get("hx/endpoints") or []
        if len(ep) < 2:
            continue
        props = props_of(h)
        sf = props.get("source-file") or ""
        repo = props.get("repo") or parse_endpoint(ep[0])[0]
        if not sf:
            missing_source_file_rows += 1
        elif is_scratch(sf, repo_map.get(repo, repo)):
            scratch_rows += 1
            if len(scratch_examples) < 5:
                scratch_examples.append(sf)
        if parse_endpoint(ep[0])[0] != parse_endpoint(ep[1])[0]:
            cross_repo_endpoint_rows += 1
        call_pairs.append((ep[0], ep[1]))

    namespaces, graph = aggregate(call_pairs)

    # every namespace, not only those with calls; source-file -> ns index
    ns_source_files = defaultdict(set)
    for h in ns_rows:
        props = props_of(h)
        repo = props.get("repo") or ""
        ns = props.get("namespace") or ""
        if not ns:
            ep = h.get("hx/endpoints") or []
            if ep:
                repo, ns, _ = parse_endpoint(ep[0] + "/x") if ep[0].count("/") < 2 else parse_endpoint(ep[0])
        if repo and ns:
            namespaces[(repo, ns)]  # ensure present
            sf = props.get("source-file")
            if sf:
                ns_source_files[(repo, ns)].add(sf)

    ranks = pagerank(graph)

    ns_out = {}
    for (repo, ns), d in sorted(namespaces.items()):
        key = f"{repo}/{ns}"
        ns_out[key] = {
            "repo": repo,
            "base_repo": repo_map.get(repo, base_repo(repo)),
            "n_vars": len(d["vars"]),
            "calls_out": d["calls_out"],
            "calls_in": d["calls_in"],
            "calls_in_cross_ns": d["calls_in_cross_ns"],
            "calls_in_cross_repo": d["calls_in_cross_repo"],
            "pagerank": ranks.get((repo, ns), 0.0),
        }

    repo_summary = defaultdict(lambda: {
        "n_namespaces": 0, "calls_in_cross_repo": 0,
        "pagerank_sum": 0.0, "_ns": []})
    for key, d in ns_out.items():
        b = d["base_repo"]
        s = repo_summary[b]
        s["n_namespaces"] += 1
        s["calls_in_cross_repo"] += d["calls_in_cross_repo"]
        s["pagerank_sum"] += d["pagerank"]
        s["_ns"].append((d["pagerank"], key))
    repo_summary = {
        b: {"n_namespaces": s["n_namespaces"],
            "calls_in_cross_repo": s["calls_in_cross_repo"],
            "pagerank_sum": s["pagerank_sum"],
            "top_ns": [k for _, k in sorted(s["_ns"], reverse=True)[:5]]}
        for b, s in sorted(repo_summary.items())}

    # --- mission attach coverage
    coverage = mission_coverage(ns_out, ns_source_files, repo_map)

    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "rows_read": {"code/v05/calls": len(calls_rows),
                      "code/v05/namespace": len(ns_rows)},
        "gaps": gaps,
        "repo_map": repo_map,
        "scratch_call_rows": scratch_rows,
        "missing_source_file_rows": missing_source_file_rows,
        "cross_repo_endpoint_rows": cross_repo_endpoint_rows,
        "scratch_examples": scratch_examples,
        "namespaces": ns_out,
        "repo_summary": repo_summary,
        "mission_coverage": coverage,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"rows: calls={len(calls_rows)} namespace={len(ns_rows)} gaps={len(gaps)}")
    print(f"namespaces={len(ns_out)} scratch_rows={scratch_rows}")
    top = sorted(ns_out.items(), key=lambda kv: -kv[1]["pagerank"])[:15]
    print("top pagerank:")
    for k, d in top:
        print(f"  {d['pagerank']:.5f} {k}")
    top2 = sorted(ns_out.items(), key=lambda kv: -kv[1]["calls_in_cross_repo"])[:15]
    print("top calls_in_cross_repo:")
    for k, d in top2:
        print(f"  {d['calls_in_cross_repo']:6d} {k}")
    for b, s in sorted(repo_summary.items(), key=lambda kv: -kv[1]["pagerank_sum"]):
        print(f"repo {b}: ns={s['n_namespaces']} pr_sum={s['pagerank_sum']:.4f} "
              f"xrepo_in={s['calls_in_cross_repo']}")


# ---------------------------------------------------------------- mission coverage

def find_mission_docs():
    """mission_id -> Path, like mission_dating: futon*/holes/**/M-*.md,
    preferring holes/missions/ over copy dirs."""
    docs = defaultdict(list)
    for p in CODE_ROOT.glob("futon*/holes/**/M-*.md"):
        docs[p.stem].append(p)
    out = {}
    for mid, paths in docs.items():
        def rank(p):
            parts = p.parts
            pref = 0 if ("missions" in parts and len(parts) > 1) else 1
            return (pref, len(str(p)))
        out[mid] = sorted(paths, key=rank)[0]
    return out


def mission_coverage(ns_out, ns_source_files, repo_map):
    positioned = set()
    if POS_EMBED.exists():
        positioned = set(json.loads(POS_EMBED.read_text()))
    docs = find_mission_docs()

    # (a) commits touching both a mission doc and a source file
    file_to_ns = {}
    for (repo, ns), files in ns_source_files.items():
        base = repo_map.get(repo, base_repo(repo))
        root = CODE_ROOT / base
        for sf in files:
            try:
                rel = str(Path(sf).relative_to(root))
            except ValueError:
                continue
            file_to_ns[(base, rel)] = f"{repo}/{ns}"

    a_map = defaultdict(set)
    repos_with_docs = {docs[m].relative_to(CODE_ROOT).parts[0]
                       for m in positioned if m in docs}
    for base in sorted(repos_with_docs):
        repo_dir = CODE_ROOT / base
        if not (repo_dir / ".git").exists():
            continue
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_dir), "log",
                 "--pretty=format:%x01%H", "--name-only"],
                capture_output=True, text=True, timeout=240)
        except Exception:  # noqa: BLE001
            continue
        if out.returncode != 0:
            continue
        cur = None
        commit_files = []
        def flush(sha, files):
            mids = [Path(f).stem for f in files
                    if re.match(r"holes/.*/M-[^/]+\.md$", f)
                    or re.match(r"holes/M-[^/]+\.md$", f)]
            if not mids:
                return
            nss = {file_to_ns[(base, f)] for f in files
                   if (base, f) in file_to_ns}
            if not nss:
                return
            for m in mids:
                a_map[m].update(nss)
        for ln in out.stdout.splitlines():
            if ln.startswith("\x01"):
                if cur:
                    flush(cur, commit_files)
                cur = ln[1:]
                commit_files = []
            elif ln.strip():
                commit_files.append(ln.strip())
        if cur:
            flush(cur, commit_files)

    # (b) verbatim namespace-name / file-path mention in the mission doc
    # Surface forms -> ns keys. Bare namespace names only qualify when they
    # look like real namespaces (contain "." or "-"): single-word script
    # names like "assemble"/"diagnostic" substring-match prose and were
    # observed to false-positive. Matching is word-boundary, not substring.
    ns_names = defaultdict(set)  # surface form -> {ns keys}
    for key in ns_out:
        repo, ns = key.split("/", 1)
        if len(ns) >= 8 and ("." in ns or "-" in ns):
            ns_names[ns].add(key)
    for (repo, ns), files in ns_source_files.items():
        base = repo_map.get(repo, base_repo(repo))
        root = CODE_ROOT / base
        for sf in files:
            try:
                rel = str(Path(sf).relative_to(root))
            except ValueError:
                continue
            ns_names[rel].add(f"{repo}/{ns}")
    surface_res = [(re.compile(r"(?<![\w./-])" + re.escape(s) + r"(?![\w/-])"), s)
                   for s in ns_names]

    b_map = defaultdict(set)
    for mid in positioned:
        p = docs.get(mid)
        if not p:
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for rx, surface in surface_res:
            if rx.search(text):
                b_map[mid].update(ns_names[surface])

    def summarize(m):
        covered = sorted(m)
        ex = [{"mission": k, "namespaces": sorted(m[k])[:4]}
              for k in covered[:5]]
        return {"positioned_missions": len(positioned),
                "missions_with_namespace": len([k for k in covered if k in positioned]),
                "examples": ex}

    return {"a_commits": summarize(a_map), "b_text": summarize(b_map)}


if __name__ == "__main__":
    main()
