#!/usr/bin/env python3
"""THREAD ORBITS (plural) — the full PHASE PORTRAIT of a mission's scope-surface.

A single thread retracted onto a mission is one integral curve and covers only a sliver of the
surface (e.g. social/ARGUMENT touches 3 scopes).  Full coverage is the FAMILY: every recurrent
thread that engages the mission (via the comb), each retracted to its own orbit, drawn together.
Together they cover the scopes the session actually swept while clocked near the mission.

  futon6/.venv/bin/python scripts/thread_orbits.py [--min-turns 2]
  -> futon2/holes/thread-orbits.edn  +  futon4/data/webarxana/public/wa/thread-orbits.json
"""
import json, os, sys

THREADS = "/home/joe/code/futon2/holes/session-threads.json"
COMB = "/home/joe/code/futon2/holes/mission-comb-M-points-de-fuite.json"
OUT = "/home/joe/code/futon2/holes/thread-orbits.edn"
WA = "/home/joe/code/futon4/data/webarxana/public/wa/thread-orbits.json"


def to_edn(v, ind=0):
    pad = "  " * ind
    if isinstance(v, dict):
        items = "\n".join(f"{pad}  :{k} {to_edn(val, ind + 1)}" for k, val in v.items())
        return "{\n" + items + "\n" + pad + "}"
    if isinstance(v, list):
        if not v:
            return "[]"
        return "[\n" + "\n".join(pad + "  " + to_edn(x, ind + 1) for x in v) + "\n" + pad + "]"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def main():
    a = sys.argv[1:]
    min_turns = int(a[a.index("--min-turns") + 1]) if "--min-turns" in a else 2
    threads_path = a[a.index("--threads") + 1] if "--threads" in a else THREADS
    comb_path = a[a.index("--comb") + 1] if "--comb" in a else COMB
    out_path = a[a.index("--out") + 1] if "--out" in a else OUT
    wa_path = a[a.index("--wa") + 1] if "--wa" in a else WA
    th = json.load(open(threads_path))
    comb = json.load(open(comb_path))
    tmap = {t["turn"]: t for t in comb["turns"] if t.get("engages")}
    # turn -> {scope-id -> score} over the turn's whole engaged neighbourhood (for the backfill)
    eng = {t["turn"]: {e["id"]: e["score"] for e in (t.get("engaged") or [])}
           for t in comb["turns"] if t.get("engages")}
    all_scopes = {s["id"]: s["name"] for s in comb["scopes"]}

    orbits = []
    for h in th["thread-hyperedges"]:
        pts, turns = [], []
        for e in sorted(h["ends"], key=lambda e: e["turn"]):
            c = tmap.get(e["turn"])
            if c:
                turns.append(e["turn"])
                pts.append({"turn": e["turn"], "scope": c["best_scope"],
                            "name": c["best_scope_name"], "score": round(c["score"], 4)})
        if len(pts) >= min_turns:
            orbits.append({"pattern": h["pattern"], "sigil": h["sigil"],
                           "recurrence": h["recurrence"], "orbit": pts, "_turns": turns})

    # SCOPE-ANCHOR BACKFILL — each scope still dark after the winner-take-all best-per-turn pass is
    # attached to the thread+turn that MOST engages it (>= thresh), as one extra station.  The clean
    # trajectories are preserved; coverage rises to every scope some thread's turn actually swept.
    covered = {p["scope"] for o in orbits for p in o["orbit"]}
    backfilled = 0
    for sid_, sname in all_scopes.items():
        if sid_ in covered:
            continue
        best = None  # (score, orbit, turn)
        for o in orbits:
            for t in o["_turns"]:
                sc = eng.get(t, {}).get(sid_)
                if sc is not None and (best is None or sc > best[0]):
                    best = (sc, o, t)
        if best:
            sc, o, t = best
            o["orbit"].append({"turn": t, "scope": sid_, "name": sname,
                               "score": round(sc, 4), "backfill": True})
            covered.add(sid_); backfilled += 1

    for o in orbits:
        o["orbit"].sort(key=lambda p: p["turn"])   # keep the curve time-ordered after backfill
        o["n-points"] = len(o["orbit"])
        o.pop("_turns", None)
    # longest threads last so they render on top of the short 2-point segments
    orbits.sort(key=lambda o: o["n-points"])

    out = {"mission": comb["mission"], "n-orbits": len(orbits),
           "scopes-covered": len(covered), "scopes-total": len(all_scopes),
           "backfilled": backfilled, "orbits": orbits}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write(";; thread orbits — full phase portrait of the mission scope-surface\n"
                              + to_edn(out) + "\n")
    try:
        json.dump(out, open(wa_path, "w"), ensure_ascii=False)
    except Exception as ex:
        print("warn: could not write WA json:", ex)

    print(f"{comb['mission']}: {len(orbits)} orbits covering {len(covered)}/{len(all_scopes)} scopes "
          f"({backfilled} backfilled, min-turns={min_turns})")
    for o in orbits[::-1][:12]:
        sg = (o["sigil"].get("truth", "") + o["sigil"].get("okipona", "")).strip() or "·"
        print(f"  {o['n-points']:2d}pts  {sg:8} {o['pattern']}")
    print(f"  wrote {out_path}\n  wrote {wa_path}")


if __name__ == "__main__":
    main()
