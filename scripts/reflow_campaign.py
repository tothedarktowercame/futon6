#!/usr/bin/env python3
"""reflow_campaign <campaign> — the CAMPAIGN-scale weaving.

A mission's orbit is one session retracted onto its scopes.  A *campaign* spans MANY agents'
sessions, so its threads are sourced ACROSS sessions: a pattern recurring over claude-4's AND
claude-10's AND claude-1's turns is a cross-agent thread — the campaign's shared strand.  This
unions the per-turn pattern-retrievals across sessions, retracts onto the campaign's scope-surface
(0.42), and writes the orbit for WebArxana (focus the campaign by name).

  futon6/.venv/bin/python scripts/reflow_campaign.py C-cascade-real [--sessions S1 S2 …] [--min-turns 15]
"""
import os, sys, json, urllib.request, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_mission_comb import fetch_scopes

API = "http://localhost:7070"
PIDX = "/home/joe/code/futon3/resources/sigils/patterns-index.tsv"
WA = "/home/joe/code/futon4/data/webarxana/public/wa/thread-orbits.json"
OUTROOT = "/home/joe/code/futon2/holes/reflow"


def sigils():
    h = {}
    for line in open(PIDX):
        p = line.split("\t")
        if len(p) >= 3 and p[1].strip() and p[2].strip():
            h[p[0].strip()] = (p[1].strip(), p[2].strip())
    return h


def fetch_session(sid):
    d = json.load(urllib.request.urlopen(
        f"{API}/api/alpha/evidence?tag=context-retrieval&session-id={sid}&limit=500", timeout=20))
    out = {}
    for e in d.get("entries", []):
        b = e.get("evidence/body") or {}
        t = b.get("turn")
        if t is None or t in out:
            continue
        out[t] = {"session": sid[:8], "agent": b.get("agent-id") or sid[:8], "turn": t,
                  "text": (b.get("query") or "")[:600],
                  "patterns": [r.get("id") for r in (b.get("results") or [])[:3] if r.get("id")],
                  "at": e.get("evidence/at") or ""}
    return list(out.values())


def discover_sessions(min_turns):
    d = json.load(urllib.request.urlopen(
        f"{API}/api/alpha/evidence?tag=context-retrieval&limit=800", timeout=20))
    c = collections.Counter()
    for e in d.get("entries", []):
        sid = e.get("evidence/session-id")
        if sid:
            c[sid] += 1
    return [s for s, n in c.items() if n >= min_turns]


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return
    campaign = a[0]
    anchor = a[a.index("--scope-anchor") + 1] if "--scope-anchor" in a else campaign
    thresh = float(a[a.index("--thresh") + 1]) if "--thresh" in a else 0.42
    minr = int(a[a.index("--min") + 1]) if "--min" in a else 3
    min_turns = int(a[a.index("--min-turns") + 1]) if "--min-turns" in a else 15
    wa = a[a.index("--wa") + 1] if "--wa" in a else WA
    sessions = a[a.index("--sessions") + 1:] if "--sessions" in a else discover_sessions(min_turns)

    scopes = fetch_scopes(anchor)
    if not scopes:
        print(f"no scopes for {campaign} (ingested?)"); return
    turns = []
    for sid in sessions:
        turns += fetch_session(sid)
    turns.sort(key=lambda t: t["at"])               # one global time-ordering across sessions
    if not turns:
        print("no turns"); return

    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("all-MiniLM-L6-v2")
    SV = m.encode([s["text"] for s in scopes], normalize_embeddings=True)
    TV = m.encode([t["text"] for t in turns], normalize_embeddings=True)
    sim = TV @ SV.T
    for i, t in enumerate(turns):
        j = int(np.argmax(sim[i])); sc = float(sim[i][j])
        t["engages"] = sc >= thresh
        t["best_scope"] = scopes[j]["id"] if t["engages"] else None
        t["best_name"] = scopes[j]["name"] if t["engages"] else None
        t["score"] = round(sc, 4)

    sig = sigils()
    pat_turns = collections.defaultdict(list)
    for i, t in enumerate(turns):
        for p in t["patterns"]:
            pat_turns[p].append(i)

    orbits = []
    for p, tis in pat_turns.items():
        distinct = len({(turns[i]["session"], turns[i]["turn"]) for i in tis})
        if distinct < minr:
            continue
        pts = [{"turn": f"{turns[i]['agent']}:{turns[i]['turn']}", "scope": turns[i]["best_scope"],
                "name": turns[i]["best_name"], "score": turns[i]["score"]}
               for i in tis if turns[i]["engages"]]
        if len(pts) >= 2:
            ok, tr = sig.get(p, ("", ""))
            agents = sorted({turns[i]["agent"] for i in tis})
            orbits.append({"pattern": p, "sigil": {"okipona": ok, "truth": tr},
                           "recurrence": distinct, "n-points": len(pts),
                           "sessions": agents, "orbit": pts})
    orbits.sort(key=lambda o: o["n-points"])
    covered = {pt["scope"] for o in orbits for pt in o["orbit"]}
    allsc = {s["id"]: s["name"] for s in scopes}

    out = {"mission": campaign, "multi-session": True, "n-sessions": len(sessions),
           "n-orbits": len(orbits), "scopes-covered": len(covered), "scopes-total": len(allsc),
           "orbits": orbits}
    json.dump(out, open(wa, "w"), ensure_ascii=False)
    d = os.path.join(OUTROOT, f"campaign__{campaign}")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(f"{d}/orbit.json", "w"), ensure_ascii=False, indent=1)

    print(f"{campaign}: {len(orbits)} CROSS-SESSION orbits · {len(sessions)} sessions · "
          f"{len(turns)} turns · {len(covered)}/{len(allsc)} scopes")
    for o in orbits[::-1][:10]:
        print(f"  {o['n-points']:2d}pts  [{'+'.join(o['sessions'])[:26]:26}] {o['pattern']}")
    print(f"  wrote {wa}")


if __name__ == "__main__":
    main()
