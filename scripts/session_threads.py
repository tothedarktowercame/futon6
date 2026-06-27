#!/usr/bin/env python3
"""SESSION THREADS — compute a session's THREADS as HYPEREDGES (M-points-de-fuite DERIVE).

A thread = a RECURRENT pattern (the binder) + the set of turns that retrieved it (the ends).
That is exactly a hyperedge: binder + N role-typed ends — homogeneous with the mission-scope-trees.
The thread is the recurrent strand the session keeps returning to (winding), and its ends are the
support of the v·∇ trajectory we will later plot.  Read-only: reads the live `context-retrieval`
evidence (the per-turn embedding pattern retrievals) and emits a hyperedge artifact + a report.

  futon6/.venv/bin/python scripts/session_threads.py [--session-id UUID] [--min N] [--topk K]
  default: this session's threads -> futon2/holes/session-threads.json
"""
import json, os, sys, glob, re, urllib.request
from collections import defaultdict

API = "http://localhost:7070"
PIDX = "/home/joe/code/futon3/resources/sigils/patterns-index.tsv"
OUT = "/home/joe/code/futon2/holes/session-threads.json"


def resolve_session_id(arg):
    if arg:
        return arg
    # most-recently-written transcript's full uuid
    path = max(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")), key=os.path.getmtime)
    return os.path.basename(path)[:-6]  # strip .jsonl


def fetch_retrievals(sid):
    url = f"{API}/api/alpha/evidence?tag=context-retrieval&session-id={sid}&limit=500"
    d = json.load(urllib.request.urlopen(url, timeout=10))
    seen = {}
    for e in d.get("entries", []):
        b = e.get("evidence/body", {}) or {}
        t = b.get("turn")
        if t is None or t in seen:
            continue
        seen[t] = {"turn": t, "query": (b.get("query") or "")[:60],
                   "results": [{"id": r.get("id"), "rank": r.get("rank"), "score": r.get("score")}
                               for r in (b.get("results") or [])]}
    return sorted(seen.values(), key=lambda o: o["turn"] or 0)


def pattern_sigils():
    sig = {}
    if os.path.exists(PIDX):
        for line in open(PIDX):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c and c[0]:
                sig[c[0]] = (c[1] if len(c) > 1 else "", c[2] if len(c) > 2 else "")
    return sig


def main():
    args = sys.argv[1:]
    sid = resolve_session_id(args[args.index("--session-id") + 1] if "--session-id" in args else None)
    minr = int(args[args.index("--min") + 1]) if "--min" in args else 3
    topk = int(args[args.index("--topk") + 1]) if "--topk" in args else 3
    out_path = args[args.index("--out") + 1] if "--out" in args else OUT

    rets = fetch_retrievals(sid)
    sig = pattern_sigils()
    short8 = sid[:8]

    # invert: pattern -> [{turn, rank, score}] over each turn's top-K retrievals (the thread's ends)
    threads = defaultdict(list)
    for o in rets:
        for r in o["results"]:
            if r["rank"] and r["rank"] <= topk:
                threads[r["id"]].append({"turn": o["turn"], "rank": r["rank"], "score": r["score"]})

    recur = {p: ts for p, ts in threads.items() if len(set(t["turn"] for t in ts)) >= minr}

    hx = []
    for p, ts in sorted(recur.items(), key=lambda kv: -len(set(t["turn"] for t in kv[1]))):
        turns = sorted(set(t["turn"] for t in ts))
        ok, tr = sig.get(p, ("", ""))
        hx.append({
            "hx/type": "thread/pattern",
            "binder-type": "thread/pattern",
            "thread-id": f"{short8}:thread/{p.replace('/', '--')}",
            "pattern": p,
            "collection": p.split("/")[0] if "/" in p else None,
            "sigil": {"okipona": ok, "truth": tr},
            "ends": [{"role": "turn", "turn": t["turn"], "rank": t["rank"], "score": t["score"]}
                     for t in sorted(ts, key=lambda x: (x["turn"] or 0, x["rank"] or 9))],
            "recurrence": len(turns),
            "span": [turns[0], turns[-1]],
        })

    out = {"session": short8, "session_id": sid, "n_turns": len(rets),
           "min_recurrence": minr, "topk": topk, "n_threads": len(hx),
           "thread-hyperedges": hx}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)

    # report
    distinct = len(threads)
    print(f"session {short8}: {len(rets)} turns · {distinct} distinct patterns retrieved (top-{topk})")
    print(f"threads (recurrence >= {minr}): {len(hx)}   ->  {out_path}")
    print()
    for t in hx[:18]:
        spine = "".join("●" if i in set(t["span"] and range(t["span"][0], t["span"][1] + 1))
                        and i in {e['turn'] for e in t['ends']} else
                        ("·" if t["span"][0] <= i <= t["span"][1] else " ")
                        for i in range(1, (rets[-1]["turn"] or 0) + 1)) if rets else ""
        sg = (t["sigil"]["truth"] + " " + t["sigil"]["okipona"]).strip() or "—"
        print(f"  {t['recurrence']:2d}× {sg:>8} · {t['pattern']:<46} turns {t['ends'][0]['turn']}…{t['ends'][-1]['turn']}")


if __name__ == "__main__":
    main()
