#!/usr/bin/env python3
"""THREAD ORBIT — retract a session thread onto a mission's scope-surface, export as .edn.

A session thread (recurrent pattern) restricted to the turns that ENGAGE the mission (via the
mission comb) maps each surviving turn to its best scope; in turn-time order those scopes trace the
ORBIT — the path the recurrent strand makes across the mission's scopes.  This is the classical-
mechanics trajectory WebArxana draws on the spiral surface.  Read-only; emits .edn for WebArxana.

  futon6/.venv/bin/python scripts/thread_orbit.py [--thread PATTERN]
  default: the thread with the most mission-engaging turns -> futon2/holes/thread-orbit.edn
"""
import json, os, sys

THREADS = "/home/joe/code/futon2/holes/session-threads.json"
COMB = "/home/joe/code/futon2/holes/mission-comb-M-points-de-fuite.json"
OUT = "/home/joe/code/futon2/holes/thread-orbit.edn"


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
    want = a[a.index("--thread") + 1] if "--thread" in a else None
    th = json.load(open(THREADS))
    comb = json.load(open(COMB))
    tmap = {t["turn"]: t for t in comb["turns"] if t.get("engages")}

    def orbit(thread):
        pts = []
        for e in sorted(thread["ends"], key=lambda e: e["turn"]):
            c = tmap.get(e["turn"])
            if c:
                pts.append({"turn": e["turn"], "scope": c["best_scope"],
                            "name": c["best_scope_name"], "score": c["score"]})
        return pts

    threads = th["thread-hyperedges"]
    if want:
        thread = next((h for h in threads if h["pattern"] == want), None)
        if not thread:
            print(f"no thread '{want}'"); return
    else:
        thread = max(threads, key=lambda h: len(orbit(h)))
    pts = orbit(thread)

    out = {"thread": {"pattern": thread["pattern"], "sigil": thread["sigil"],
                      "recurrence": thread["recurrence"], "span": thread["span"]},
           "mission": comb["mission"],
           "n-orbit-points": len(pts),
           "orbit": pts}
    open(OUT, "w").write(";; thread orbit (retracted onto the mission scope-surface) — for WebArxana\n"
                         + to_edn(out) + "\n")
    # also emit JSON into WebArxana's static dir so the client can fetch /wa/thread-orbit.json
    wa = "/home/joe/code/futon4/data/webarxana/public/wa/thread-orbit.json"
    try:
        json.dump(out, open(wa, "w"), ensure_ascii=False)
    except Exception:
        pass

    sg = (thread["sigil"]["truth"] + " " + thread["sigil"]["okipona"]).strip() or "—"
    print(f"thread 〘{sg}〙 {thread['pattern']} ({thread['recurrence']}× recurrent)")
    print(f"  retracted onto {comb['mission']}: {len(pts)} orbit points (of {thread['recurrence']} turns)")
    seen = []
    for p in pts:
        tag = "  (revisit)" if p["scope"] in seen else ""
        seen.append(p["scope"])
        print(f"    t{p['turn']:<3} → {(p['name'] or '?')[:44]:<44} ({p['score']:.2f}){tag}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
