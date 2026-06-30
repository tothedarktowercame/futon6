#!/usr/bin/env python3
"""C-cascade-real RUN/DELIVER — VALUE DEMO (show value before ingesting).

Composes, for a mission, WITHOUT writing anything to substrate-2 (:7071):
  - the MINE (futon6/data/meme-mine/joint-memes.openai.json, read as-is): the open
    moves it found (have->want, op, maturity, evidence) + provenance sessions;
  - LIVE O3 lineage (substrate-2 :7071, read-only): who/which session is clocked
    on the mission right now.

The point (Joe): .edn will swallow anything — the question is what composing these
diverging shapes is actually GOOD FOR. The join key is the mission id. If the
dossier is useful, it both justifies an ingest AND specifies its shape (the join
this demo performs). Usage: mission_dossier.py M-capability-star-map
"""
import json, sys, urllib.request, collections

MINE = "/home/joe/code/futon6/data/meme-mine/joint-memes.openai.json"
FUTON1A = "http://localhost:7071"


def reflist(x):
    r = (x or {}).get("ref")
    return [] if r is None else (r if isinstance(r, list) else [r])


def mine_for_mission(mission):
    """Open moves + sessions the mine found referencing MISSION (key 'mission/<id>')."""
    key = "mission/" + mission
    memes = json.load(open(MINE))
    moves, sessions, asks = [], set(), 0
    for m in memes:
        cands = (m.get("candidates") or {}).get("missions", [])
        hit_cand = key in cands
        hit_move = False
        for mv in m.get("memes", []):
            if key in reflist(mv.get("have")) or key in reflist(mv.get("want")):
                hit_move = True
                h, w = mv.get("have") or {}, mv.get("want") or {}
                moves.append({
                    "op": mv.get("op"), "maturity": mv.get("maturity"),
                    "have": h.get("text"), "want": w.get("text"),
                    "evidence": h.get("evidence") or w.get("evidence"),
                    "session": (m.get("provenance") or {}).get("session"),
                })
        if hit_cand or hit_move:
            asks += 1
            s = (m.get("provenance") or {}).get("session")
            if s:
                sessions.add(s)
    return {"asks": asks, "moves": moves, "sessions": sorted(sessions)}


def live_clock(mission):
    """Agents/sessions currently clocked on MISSION (live O3, read-only)."""
    url = "%s/api/alpha/hyperedges?end=mission:%s" % (FUTON1A, mission)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/edn"})
        body = urllib.request.urlopen(req, timeout=5).read().decode()
    except Exception as e:
        return {"error": str(e), "agents": []}
    # crude EDN scrape: agent: endpoints on clock/clocked-on edges
    import re
    agents = sorted(set(re.findall(r'"agent:([^"]+)"', body)))
    return {"agents": agents}


def main():
    mission = sys.argv[1] if len(sys.argv) > 1 else "M-capability-star-map"
    mine = mine_for_mission(mission)
    live = live_clock(mission)
    print("=" * 72)
    print("MISSION DOSSIER  —  %s   (composed; nothing ingested)" % mission)
    print("=" * 72)
    live_agents = live.get("agents") or []
    print("LIVE (O3 lineage): %s" %
          (", ".join(live_agents) if live_agents else "no agent clocked here now"))
    print("MINE: %d asks referenced this mission across %d session(s); %d open move(s)."
          % (mine["asks"], len(mine["sessions"]), len(mine["moves"])))
    if mine["sessions"]:
        print("  provenance sessions: %s" % ", ".join(mine["sessions"][:6]) +
              (" …" if len(mine["sessions"]) > 6 else ""))
    print("-" * 72)
    by_mat = collections.Counter(m["maturity"] for m in mine["moves"])
    print("open moves by maturity: %s" % dict(by_mat))
    for mv in mine["moves"][:12]:
        ev = (" | ev: " + mv["evidence"][:60]) if mv["evidence"] else ""
        print("  [%s/%s] %s → %s%s"
              % (mv["op"], mv["maturity"], mv["have"], mv["want"], ev))
    if len(mine["moves"]) > 12:
        print("  … %d more" % (len(mine["moves"]) - 12))
    print("-" * 72)
    composed = bool(live_agents) and bool(mine["moves"])
    print("COMPOSED: %s" %
          ("live agent × mined open-moves — the dossier the cascade is FOR."
           if composed else
           "mine-only here (no live agent on this mission, or mine is stale for it)."))


if __name__ == "__main__":
    main()
