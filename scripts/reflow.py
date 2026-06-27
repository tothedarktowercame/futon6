#!/usr/bin/env python3
"""reflow <session-id> <mission|excursion> — the M-points-de-fuite VERIFY apparatus, parameterized.

Runs the in-flow weaving pipeline (embeddings comb -> threads -> orbit/phase-portrait) for ANY
session against ANY mission/excursion/campaign, UNTUNED. Output namespaced per (session, mission)
so runs never clobber one another. This is the apparatus the VERIFY protocol (M-points-de-fuite
section 4) runs on a fresh session/author to test that the weaving reproduces by mechanism, not by
hand-fit.

  futon6/.venv/bin/python scripts/reflow.py <session-id> <mission-id> [--scope-anchor NAME]
                                            [--thresh 0.30] [--min-turns 2] [--wa PATH]
"""
import os, sys, json, glob, subprocess

F6 = "/home/joe/code/futon6"
PY = f"{F6}/.venv/bin/python"
OUTROOT = "/home/joe/code/futon2/holes/reflow"
WA_DIR = "/home/joe/code/futon4/data/webarxana/public/wa"


def derive_scope_anchor(mission):
    """Missions resolve in the ego by short name; excursions/campaigns anchor on the full vertex
    name <repo>-d/mission/<lower-id>. Find the doc to learn its repo."""
    if mission.startswith("M-"):
        return mission
    hits = (glob.glob(f"/home/joe/code/*/holes/{mission}.md")
            + glob.glob(f"/home/joe/code/*/holes/**/{mission}.md", recursive=True))
    if not hits:
        return mission
    repo = hits[0].split("/code/")[1].split("/")[0]
    return f"{repo}-d/mission/{mission.lower()}"


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    sid, mission = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    anchor = (rest[rest.index("--scope-anchor") + 1] if "--scope-anchor" in rest
              else derive_scope_anchor(mission))
    thresh = rest[rest.index("--thresh") + 1] if "--thresh" in rest else "0.30"
    min_turns = rest[rest.index("--min-turns") + 1] if "--min-turns" in rest else "2"
    # PER-TARGET served file (default) so mission + campaign trackers don't share one path
    wa = rest[rest.index("--wa") + 1] if "--wa" in rest else f"{WA_DIR}/thread-orbits-{mission}.json"

    d = os.path.join(OUTROOT, f"{sid[:8]}__{mission}")
    os.makedirs(d, exist_ok=True)
    comb, threads, orbit = f"{d}/comb.json", f"{d}/threads.json", f"{d}/orbit.edn"

    def run(args):
        subprocess.run([PY] + args, cwd=F6, check=True)

    print(f"\n=== reflow: session {sid[:8]} × {mission} (anchor: {anchor}) ===")
    run(["scripts/session_mission_comb.py", "--mission", mission, "--scope-anchor", anchor,
         "--session-id", sid, "--thresh", thresh, "--out", comb])
    run(["scripts/session_threads.py", "--session-id", sid, "--out", threads])
    run(["scripts/thread_orbits.py", "--threads", threads, "--comb", comb,
         "--out", orbit, "--wa", wa, "--min-turns", min_turns])

    # --- VERIFY summary (M-points-de-fuite section 4 pass criteria, where measurable here) ---
    cb = json.load(open(comb))
    n_turns, n_eng = cb["n_turns"], cb["n_engaged"]
    scopes_total = len(cb["scopes"])
    th = json.load(open(threads))
    n_threads = th["n_threads"]
    wo = json.load(open(wa))
    covered, orbits = wo["scopes-covered"], wo["n-orbits"]
    # above-chance: a turn's best scope clears thresh more often than a uniform pick would
    print("\n--- VERIFY readout ---")
    print(f"  turns:            {n_turns}")
    print(f"  engaging turns:   {n_eng}/{n_turns}  ({100*n_eng/max(1,n_turns):.0f}%)  [criterion 3: above chance?]")
    print(f"  threads formed:   {n_threads}  (recurrence >= {th['min_recurrence']})  [criterion 2: compose into a line]")
    print(f"  scopes covered:   {covered}/{scopes_total}  via {orbits} orbits  [criterion 4: the field]")
    print(f"  outputs:          {d}/  +  {wa}")
    if n_threads == 0 or n_eng == 0:
        print("  >> THIN: not enough turns yet for a weaving — re-run after the driver has worked more.")


if __name__ == "__main__":
    main()
