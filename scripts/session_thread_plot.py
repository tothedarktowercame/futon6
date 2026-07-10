#!/usr/bin/env python3
"""SESSION THREAD PLOT — the first LOOK at a thread as a path (M-points-de-fuite).

The session's own surface: each TURN is placed at the centroid of its retrieved-pattern
embeddings (futon3a MiniLM, 384-d), projected to 2-D by PCA.  The full session is drawn as a
faint trajectory (the v·∇ integral curve); ONE thread is highlighted — the turns that recur on
its pattern, connected in time-order — so we can see whether a thread reads as a coherent
recurrent orbit (a real strange-attractor loop) or as scatter.  Read-only SVG/HTML.

  futon6/.venv/bin/python scripts/session_thread_plot.py [--session-id UUID] [--thread PATTERN]
  default: this session, the highest-recurrence thread -> futon2/holes/session-thread-plot.html
"""
import json, os, sys, glob, urllib.request
import numpy as np

API = "http://localhost:7070"
EMB = "/home/joe/code/futon3a/resources/notions/minilm_pattern_embeddings.json"
THREADS = "/home/joe/code/futon2/holes/session-threads.json"
OUT = "/home/joe/code/futon2/holes/session-thread-plot.html"


def fetch_retrievals(sid):
    url = f"{API}/api/alpha/evidence?tag=context-retrieval&session-id={sid}&limit=500"
    d = json.load(urllib.request.urlopen(url, timeout=10))
    seen = {}
    for e in d.get("entries", []):
        b = e.get("evidence/body", {}) or {}
        t = b.get("turn")
        if t is None or t in seen:
            continue
        seen[t] = {"turn": t, "query": (b.get("query") or "")[:48],
                   "patterns": [r.get("id") for r in (b.get("results") or [])]}
    return sorted(seen.values(), key=lambda o: o["turn"] or 0)


def main():
    args = sys.argv[1:]
    sid = (args[args.index("--session-id") + 1] if "--session-id" in args
           else os.path.basename(max(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")),
                                     key=os.path.getmtime))[:-6])
    th = json.load(open(THREADS))
    thread_pat = (args[args.index("--thread") + 1] if "--thread" in args
                  else th["thread-hyperedges"][0]["pattern"])
    thread = next((h for h in th["thread-hyperedges"] if h["pattern"] == thread_pat), th["thread-hyperedges"][0])
    sigil = (thread["sigil"]["truth"] + " " + thread["sigil"]["okipona"]).strip() or thread["pattern"].split("/")[-1]

    rets = fetch_retrievals(sid)
    emb = {p["id"]: np.array(p["vector"], dtype=float) for p in json.load(open(EMB))}

    # turn -> centroid of its retrieved-pattern embeddings (those we have vectors for)
    turns, vecs = [], []
    for o in rets:
        vs = [emb[p] for p in o["patterns"] if p in emb]
        if vs:
            turns.append(o["turn"])
            vecs.append(np.mean(vs, axis=0))
    X = np.array(vecs)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T                       # (n,2) PCA projection

    # scale to an SVG viewport
    W, H, pad = 1400, 1000, 70
    mn, mx = P.min(axis=0), P.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    xy = {t: ((P[i] - mn) / span * np.array([W - 2 * pad, H - 2 * pad]) + pad)
          for i, t in enumerate(turns)}

    thread_turns = [e["turn"] for e in thread["ends"] if e["turn"] in xy]

    def pt(t):
        return f"{xy[t][0]:.1f},{xy[t][1]:.1f}"

    # full session trajectory (faint), in time order
    full = " ".join(pt(t) for t in turns)
    # the thread's recurrent path (time order)
    tpath = " ".join(pt(t) for t in thread_turns)

    dots = "".join(
        f'<circle cx="{xy[t][0]:.1f}" cy="{xy[t][1]:.1f}" r="3" fill="#3a4254"/>'
        for t in turns)
    tdots = "".join(
        f'<circle cx="{xy[t][0]:.1f}" cy="{xy[t][1]:.1f}" r="7" fill="#0f766e" stroke="#7ee" stroke-width="1.5"/>'
        f'<text x="{xy[t][0]+9:.1f}" y="{xy[t][1]+4:.1f}" fill="#9fe" font-size="13">t{t}</text>'
        for t in thread_turns)

    svg = f'''<!doctype html><meta charset=utf-8><title>thread {thread_pat}</title>
<body style="margin:0;background:#05060a;color:#cdd3df;font:13px sans-serif">
<header style="padding:11px 20px">
<b style="font-size:16px">Session {sid[:8]} — thread trajectory</b> &nbsp;
〘 {sigil} 〙 <b>{thread_pat}</b> &nbsp; · {thread["recurrence"]}× recurrent, turns {thread_turns[0]}…{thread_turns[-1]}
<br><span style="color:#7a829a">Each dot = a turn placed by its retrieved-pattern embedding (PCA→2D). Faint line = the whole session's path (v·∇ trajectory); teal = this thread's recurrent orbit.</span>
</header>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<polyline points="{full}" fill="none" stroke="#2a3142" stroke-width="1.2" opacity="0.7"/>
{dots}
<polyline points="{tpath}" fill="none" stroke="#0f766e" stroke-width="2.5" opacity="0.9"/>
{tdots}
</svg></body>'''
    open(OUT, "w").write(svg)
    print(f"session {sid[:8]}: {len(turns)} turns positioned · PCA variance captured "
          f"{(S[:2]**2).sum()/(S**2).sum():.0%}")
    print(f"thread 〘{sigil}〙 {thread_pat}: {len(thread_turns)} turns (of {thread['recurrence']})")
    print(f"wrote {OUT}   (open in a browser)")


if __name__ == "__main__":
    main()
