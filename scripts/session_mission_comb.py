#!/usr/bin/env python3
"""SESSION→MISSION COMB (embeddings) — keep the session-overview comb current, for free.

The session-overview comb (session→mission incidence, *session-overview* lines 7-18) is sourced from
the PAID mining snapshot and goes stale.  This recomputes the incidence from CURRENT free signals:
embed the mission's scopes and the session's turns with the same MiniLM; a turn ENGAGES a scope when
their cosine similarity clears a threshold.  Output: turn→scope incidence (to retract a thread onto a
mission's scope-surface as an orbit) + the mission-level spine (which turns touch the mission) — all
current, no GPU mining.

  futon6/.venv/bin/python scripts/session_mission_comb.py [--mission M-...] [--session-id UUID] [--thresh 0.30]
"""
import json, os, sys, glob, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_scope_view import parse_turns, _norm

API = "http://localhost:7070"
OUTDIR = "/home/joe/code/futon2/holes"


def fetch_retrievals(sid):
    d = json.load(urllib.request.urlopen(
        f"{API}/api/alpha/evidence?tag=context-retrieval&session-id={sid}&limit=500", timeout=15))
    seen = {}
    for e in d.get("entries", []):
        b = e.get("evidence/body", {}) or {}
        t = b.get("turn")
        if t is not None and t not in seen:
            seen[t] = {"turn": t, "query": b.get("query") or ""}
    return sorted(seen.values(), key=lambda o: o["turn"])


def fetch_scopes(mission):
    # the folded ego (scope frames) is served by the WebArxana proxy on :3100
    d = json.load(urllib.request.urlopen(
        f"http://localhost:3100/api/futon/ego/{mission}?fold=1&depth=3", timeout=15))
    out = []
    for r in (d.get("ego", {}).get("outgoing") or []):
        e = r.get("entity", {})
        if e.get("type") == "scope/frame":
            p = e.get("props", {}) or {}
            concepts = [c.get("term") for c in (p.get("fold/top-concepts") or []) if c.get("term")]
            name = e.get("name") or ""
            out.append({"id": e.get("id"), "name": name, "binder": p.get("scope/binder"),
                        "text": (name + " " + " ".join(concepts)).strip()})
    return out


def main():
    a = sys.argv[1:]
    mission = a[a.index("--mission") + 1] if "--mission" in a else "M-points-de-fuite"
    # the ego anchor: missions resolve by short name; excursions/campaigns need
    # the full vertex name (<repo>-d/mission/<lower-id>). Defaults to the mission id.
    scope_anchor = a[a.index("--scope-anchor") + 1] if "--scope-anchor" in a else mission
    sid = (a[a.index("--session-id") + 1] if "--session-id" in a
           else os.path.basename(max(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")),
                                     key=os.path.getmtime))[:-6])
    thresh = float(a[a.index("--thresh") + 1]) if "--thresh" in a else 0.30
    out_path = a[a.index("--out") + 1] if "--out" in a else None

    rets = fetch_retrievals(sid)
    tpath = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))[0]
    ops, _ = parse_turns(tpath)
    opnorm = [(_norm(o["full"]), o["full"]) for o in ops]
    turns = []
    for o in rets:
        q = _norm(o["query"])[:50]
        full = next((f for n, f in opnorm if q and (q in n or n[:30] == q[:30])), o["query"])
        turns.append({"turn": o["turn"], "text": (full or "")[:600]})

    scopes = fetch_scopes(scope_anchor)
    if not scopes:
        print(f"no scopes for {mission} (is it ingested?)"); return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    SV = model.encode([s["text"] for s in scopes], normalize_embeddings=True)
    TV = model.encode([t["text"] for t in turns], normalize_embeddings=True)
    sim = TV @ SV.T   # (n_turns, n_scopes)

    comb_turns = []
    for i, t in enumerate(turns):
        order = np.argsort(-sim[i])
        j = int(order[0]); sc = float(sim[i][j])
        # the turn's full scope-NEIGHBOURHOOD (every scope it engages >= thresh, best-first) — not
        # just the argmax.  Winner-take-all starves scopes that are genuinely engaged but never any
        # turn's single best; the neighbourhood is what lets the orbit retraction cover the surface.
        engaged = [{"id": scopes[int(k)]["id"], "name": scopes[int(k)]["name"],
                    "score": round(float(sim[i][int(k)]), 3)}
                   for k in order if sim[i][int(k)] >= thresh]
        comb_turns.append({"turn": t["turn"],
                           "best_scope": scopes[j]["id"] if sc >= thresh else None,
                           "best_scope_name": scopes[j]["name"] if sc >= thresh else None,
                           "score": round(sc, 3), "engages": bool(sc >= thresh),
                           "engaged": engaged})

    scope_turns = {s["id"]: [] for s in scopes}
    for ct in comb_turns:
        if ct["best_scope"]:
            scope_turns[ct["best_scope"]].append(ct["turn"])

    engaged = [ct["turn"] for ct in comb_turns if ct["engages"]]
    out = {"mission": mission, "session": sid[:8], "thresh": thresh, "n_turns": len(turns),
           "n_engaged": len(engaged), "engaged_turns": engaged, "turns": comb_turns,
           "scopes": [{"id": s["id"], "name": s["name"], "binder": s["binder"],
                       "turns": scope_turns[s["id"]]} for s in scopes]}
    path = out_path or os.path.join(OUTDIR, f"mission-comb-{mission}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)

    scores = sorted((ct["score"] for ct in comb_turns), reverse=True)
    print(f"mission comb {mission} · session {sid[:8]} · thresh {thresh}  ->  {os.path.basename(path)}")
    print(f"  {len(engaged)}/{len(turns)} turns engage {mission} (CURRENT, embeddings)")
    print(f"  score distribution: max {scores[0]:.2f} · p75 {scores[len(scores)//4]:.2f} · "
          f"median {scores[len(scores)//2]:.2f}")
    print(f"  scopes with >=1 engaging turn: {sum(1 for s in scopes if scope_turns[s['id']])}/{len(scopes)}")
    for s in sorted(scopes, key=lambda s: -len(scope_turns[s["id"]]))[:8]:
        if scope_turns[s["id"]]:
            print(f"    [{s['binder']:<16}] {s['name'][:40]:<40} <- {len(scope_turns[s['id']])} turns {scope_turns[s['id']][:8]}")


if __name__ == "__main__":
    main()
