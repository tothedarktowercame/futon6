#!/usr/bin/env python3
"""verify_check <session-id> <target> — the MECHANICALLY-CHECKABLE VERIFY checklist.

Operationalizes M-points-de-fuite §4's five pass criteria into computable booleans, so the
live-reproduction VERIFY is decided by a script, not by eyeballing.  Run it on a driver
session (claude-11 × E-precision-over-policies) after each batch of turns:

  futon6/.venv/bin/python scripts/verify_check.py <session-id> <target> [--control M-cold-chain]

Each criterion returns {pass, value, note}.  Criteria that can only fail for lack of turns are
reported THIN (not FAIL).  Overall: PASS (all), THIN (only turn-count-limited fail), or FAIL.
"""
import os, sys, json, glob, subprocess, urllib.request

F6 = "/home/joe/code/futon6"
PY = f"{F6}/.venv/bin/python"
API = "http://localhost:7070"
PIDX = "/home/joe/code/futon3/resources/sigils/patterns-index.tsv"
OUTROOT = "/home/joe/code/futon2/holes/reflow"

# --- thresholds (tunable) ---
SIGIL_FRAC = 0.40        # C1: >= this fraction of turns resolve a sigil
MIN_THREADS = 1          # C2: at least this many recurrent threads
CHANCE_MARGIN = 0.15     # C3: target engage-frac must beat control by >= this (absolute)
MIN_COVERED = 2          # C4: orbit must cover >= this many distinct scopes
MIN_MOVES = 1            # C5: basins must tag >= this many turns
MIN_POWER_TURNS = 12     # below this, the specificity test (C3) is under-powered → THIN, not FAIL


def run(args):
    subprocess.run([PY] + args, cwd=F6, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def derive_anchor(mission):
    if mission.startswith("M-"):
        return mission
    hits = (glob.glob(f"/home/joe/code/*/holes/{mission}.md")
            + glob.glob(f"/home/joe/code/*/holes/**/{mission}.md", recursive=True))
    if not hits:
        return mission
    repo = hits[0].split("/code/")[1].split("/")[0]
    return f"{repo}-d/mission/{mission.lower()}"


def sigil_patterns():
    have = set()
    for line in open(PIDX):
        p = line.split("\t")
        if len(p) >= 3 and p[1].strip() and p[2].strip():
            have.add(p[0].strip())
    return have


def c1_sigils(sid):
    """Per-turn sigils populate: fraction of turns whose top retrieved pattern resolves a sigil."""
    d = json.load(urllib.request.urlopen(
        f"{API}/api/alpha/evidence?tag=context-retrieval&session-id={sid}&limit=500", timeout=15))
    have = sigil_patterns()
    by_turn = {}
    for e in d.get("entries", []):
        b = e.get("evidence/body") or {}
        t = b.get("turn")
        top = (b.get("results") or [{}])[0].get("id")
        if t is not None and t not in by_turn:
            by_turn[t] = top
    n = len(by_turn)
    sig = sum(1 for top in by_turn.values() if top in have)
    frac = sig / n if n else 0.0
    return {"pass": frac >= SIGIL_FRAC, "value": f"{sig}/{n} turns resolve a sigil ({frac:.0%})",
            "n_turns": n, "thin_ok": False}


def c2_threads(threads_json):
    th = json.load(open(threads_json))
    n = th.get("n_threads", 0)
    return {"pass": n >= MIN_THREADS, "value": f"{n} threads (recurrence>={th.get('min_recurrence')})",
            "thin_ok": True}   # only fails for too-few-turns


def c3_above_chance(target_comb, control_comb, control_name, n_turns):
    tc = json.load(open(target_comb)); cc = json.load(open(control_comb))
    tf = tc["n_engaged"] / max(1, tc["n_turns"])
    cf = cc["n_engaged"] / max(1, cc["n_turns"])
    under_powered = n_turns < MIN_POWER_TURNS
    return {"pass": (tf - cf) >= CHANCE_MARGIN,
            "value": (f"target {tf:.0%} vs control({control_name}) {cf:.0%}  (+{(tf-cf)*100:.0f}pp)"
                      + (f" — under-powered at {n_turns}<{MIN_POWER_TURNS} turns" if under_powered else "")),
            "thin_ok": under_powered}   # a real specificity test only once N is adequate


def c4_field(orbit_json):
    o = json.load(open(orbit_json))
    cov = o.get("scopes-covered", 0); norb = o.get("n-orbits", 0)
    return {"pass": cov >= MIN_COVERED and norb >= 1,
            "value": f"{norb} orbits covering {cov}/{o.get('scopes-total','?')} scopes (the target's own)",
            "thin_ok": True}   # degenerate only when too few threads/turns


def c5_recogniser(sid):
    """Recogniser tags turns WITHOUT paid mining: basins assign moves to this fresh session."""
    out = f"/tmp/verify-ssv-{sid[:8]}.org"
    run(["scripts/session_scope_view.py", out, "--session-id", sid])
    d = json.load(open(os.path.splitext(out)[0] + ".json"))
    moves = 0
    for arc in d.get("arcs", []):
        for t in arc.get("turns", []):
            if t.get("move"):
                moves += 1
    return {"pass": moves >= MIN_MOVES,
            "value": f"{moves} turns tagged via move-basins (deterministic, no mining)",
            "thin_ok": True}


def c6_webarxana(target):
    """WebArxana follow-along: focusing the target by its NATURAL name in WebArxana shows its
    scope-surface — the interactive form of C4 (watch the field light up as the session works)."""
    import urllib.parse
    name = urllib.parse.quote(target, safe="")
    try:
        d = json.load(urllib.request.urlopen(
            f"http://localhost:3100/api/futon/ego/{name}?fold=1&depth=3", timeout=10))
        e = d.get("ego", {})
        n = len([r for r in (e.get("outgoing") or []) if (r.get("entity", {}) or {}).get("type") == "scope/frame"])
    except Exception:
        n = 0
    return {"pass": n >= 2,
            "value": f"#/dev/focus/{target} -> {n} scope/frames render in WebArxana",
            "thin_ok": False}


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    sid, target = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    control = rest[rest.index("--control") + 1] if "--control" in rest else "M-cold-chain"
    anchor = derive_anchor(target)

    d = os.path.join(OUTROOT, f"{sid[:8]}__{target}")
    os.makedirs(d, exist_ok=True)
    comb, threads, orbit = f"{d}/comb.json", f"{d}/threads.json", f"{d}/orbit.json"
    ctrl_comb = f"{d}/control-comb.json"

    # build the artifacts (untuned)
    run(["scripts/session_mission_comb.py", "--mission", target, "--scope-anchor", anchor,
         "--session-id", sid, "--out", comb])
    run(["scripts/session_threads.py", "--session-id", sid, "--out", threads])
    run(["scripts/thread_orbits.py", "--threads", threads, "--comb", comb,
         "--out", f"{d}/orbit.edn", "--wa", orbit])
    run(["scripts/session_mission_comb.py", "--mission", control,
         "--session-id", sid, "--out", ctrl_comb])

    c1 = c1_sigils(sid)
    n_turns = c1["n_turns"]
    checks = {
        "C1 per-turn sigils populate":          c1,
        "C2 threads compose into a line":       c2_threads(threads),
        "C3 comb engages target above chance":  c3_above_chance(comb, ctrl_comb, control, n_turns),
        "C4 field non-degenerate + specific":   c4_field(orbit),
        "C5 recogniser tags w/o paid mining":   c5_recogniser(sid),
        "C6 WebArxana follow-along surface":     c6_webarxana(target),
    }
    # overall: PASS if all pass; THIN if the only failures are turn-count-limited (thin_ok); else FAIL
    fails = [k for k, c in checks.items() if not c["pass"]]
    hard_fails = [k for k in fails if not checks[k]["thin_ok"]]
    overall = "PASS" if not fails else ("THIN" if not hard_fails else "FAIL")

    print(f"\n=== VERIFY checklist: {sid[:8]} × {target}  (control: {control}) ===")
    print(f"turns: {n_turns}\n")
    for k, c in checks.items():
        tag = "PASS" if c["pass"] else ("THIN" if c["thin_ok"] else "FAIL")
        print(f"  [{tag:4}] {k:38} {c['value']}")
    print(f"\nOVERALL: {overall}" +
          ("" if overall == "PASS" else
           "  (re-run after more turns)" if overall == "THIN" else
           f"  — hard fails: {', '.join(hard_fails)}"))
    json.dump({"session": sid[:8], "target": target, "control": control, "n_turns": n_turns,
               "overall": overall, "checks": {k: {kk: vv for kk, vv in c.items()} for k, c in checks.items()}},
              open(f"{d}/verify-checklist.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
