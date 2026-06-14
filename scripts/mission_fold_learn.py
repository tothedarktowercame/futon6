#!/usr/bin/env python3
# mission_fold_learn.py — CLOSE THE DUAL LOOP.
#
# mission_fold.py does the structure-learning half (Bayesian model reduction:
# the dense scope-net -> a folded spine, SIP PRIOR as salience). This script
# does the OTHER half the Campaign's completion criterion demands — "improves
# its OWN model" (M-bayesian-structure-learning §3.1 reliability-as-posterior):
#
#   the operator's expand/collapse behaviour IS the evidence.
#
# A frame the operator keeps EXPANDING was folded too aggressively (its content
# was wanted) -> raise its salience posterior. A frame they keep COLLAPSING was
# folded correctly -> the fold was right. Each event is one Bernoulli trial on
# "did the operator want this frame open?", updating a Beta(alpha,beta) per frame
# (prior seeded from the SIP subtree mass — posteriors-not-counters, seeded by
# the prior we already built). After a trace the spine RE-DERIVES: high-posterior
# frames auto-expand, low ones sink. The fold adapts to the operator — locally,
# in-the-loop, no superpod. That is the Friston dual loop, closed and visible.
#
# Synthetic traces here are a MECHANISM PROOF; the live render (codex-1) will
# emit the real expand/collapse events that replace them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mission_fold import load_sip, load_tree, build, top_sip  # noqa: E402


def seed_prior(mass, mass_max):
    # SIP subtree mass -> a weak Beta prior. More distinctive subtree => higher
    # prior belief the operator will want it open. Kept weak (sum ~3) so a few
    # real interactions dominate — the model is meant to LEARN, not ossify.
    frac = mass / mass_max if mass_max else 0.0
    return {"alpha": 1.0 + 2.0 * frac, "beta": 1.0 + 2.0 * (1 - frac)}


def mean(p):
    return p["alpha"] / (p["alpha"] + p["beta"])


def run(mission, trace, threshold=0.5):
    sip = load_sip()
    tree, stem = load_tree(mission)
    nodes, roots = build(tree, sip)

    # spine candidates = the root's direct children (the phase/loose frames)
    spine = []
    for r in roots:
        spine.extend(nodes[r]["children"])
    spine = [s for s in spine if nodes[s]["sub_count"]]
    mass_max = max((nodes[s]["sub_mass"] for s in spine), default=1.0)
    post = {s: seed_prior(nodes[s]["sub_mass"], mass_max) for s in spine}

    def title_of(s):
        return nodes[s]["title"]

    def find(sub):
        for s in spine:
            if sub.lower() in title_of(s).lower():
                return s
        return None

    print(f"================  DUAL LOOP: {stem}  ================\n")
    print("  PRIOR spine (SIP-only — the structure-learning half):")
    for s in sorted(spine, key=lambda s: -mean(post[s])):
        m = mean(post[s])
        print(f"    {'EXPAND' if m >= threshold else 'fold  '}  {title_of(s)[:34]:34} "
              f"p(want-open)={m:.2f}  ⊞{nodes[s]['sub_count']}")

    # ---- the operator interacts; each event updates a posterior ----
    print(f"\n  OPERATOR TRACE ({len(trace)} events) — the evidence:")
    for action, sub in trace:
        s = find(sub)
        if not s:
            print(f"    ? no frame matches '{sub}'"); continue
        if action == "expand":
            post[s]["alpha"] += 1          # wanted open -> fold was too aggressive
        elif action == "collapse":
            post[s]["beta"] += 1           # didn't need it -> fold was right
        print(f"    {action:8} {title_of(s)[:30]:30} -> Beta({post[s]['alpha']:.0f},{post[s]['beta']:.0f})")

    print("\n  POSTERIOR spine (after interaction — the model improved itself):")
    moved = []
    for s in sorted(spine, key=lambda s: -mean(post[s])):
        m_post = mean(post[s])
        m_prior = mean(seed_prior(nodes[s]["sub_mass"], mass_max))
        arrow = ""
        crossed = (m_prior >= threshold) != (m_post >= threshold)
        if crossed:
            arrow = "  <== FLIPPED"
            moved.append((title_of(s), m_prior, m_post))
        print(f"    {'EXPAND' if m_post >= threshold else 'fold  '}  {title_of(s)[:34]:34} "
              f"p={m_prior:.2f}->{m_post:.2f}{arrow}")

    print(f"\n  AHA (model reduction/promotion): {len(moved)} frame(s) changed fold-state "
          f"from interaction alone — the operator taught the fold what they care about.")
    for t, a, b in moved:
        print(f"     '{t[:40]}'  {a:.2f} -> {b:.2f}")


def _edn(obj, ind=0):
    # minimal edn serializer — the WM reads graph.edn, so emit its native format
    pad = "  " * ind
    if isinstance(obj, dict):
        items = "\n".join(f"{pad}  :{k} {_edn(v, ind + 1).lstrip()}" for k, v in obj.items())
        return "{\n" + items + "}"
    if isinstance(obj, list):
        return "[" + " ".join(_edn(v, ind) for v in obj) + "]"
    if isinstance(obj, str):
        return '"' + obj.replace('"', '\\"') + '"'
    if isinstance(obj, float):
        return f"{obj:.3f}"
    return str(obj)


GAP_THIN = 8       # a section-frame with >= this many distinct concepts counts as "filled"
GROWTH_FLOOR = 12  # missions with raw < this are nascent/abandoned stubs -> NO growth signal (0)
MASS_CAP = 120     # 8 missing-slots x 15 empty sections = "maximally hollow"; linear below it


def gap_score(nodes):
    """The WM's complement signal: GROWTH-SURFACE = the MASS of announced-but-empty
    structure = Σ over section-frames of (GAP_THIN − sub_count) for under-filled ones —
    i.e. total missing concept-slots = concrete room to grow. Linearly normalized below
    MASS_CAP; ∈[0,1]; CROSS-MISSION-COMPARABLE; corpus-independent (fixed cap).

    NB1 (STANDARD-VERIFY, 2026-06-08): de-biased vs the original mean-stub-FRACTION (which
    rewarded tiny outlines): MASS is a floored SUM, so a 5-node mission scores ~0 and
    sub-FLOOR missions are zeroed. Gap is the WITHIN-LOCAL expansion-refiner only; domain
    selection (local vs math) is the WM's ascent-gate, not gap.
    NB2 (F1 fix, 2026-06-08, claude-1's regulator harness): MASS replaces the earlier empty
    *count* with CAP→1.0, which clamped ~25 local missions to a shared 1.0 ceiling so the EFE
    gap term TIED the within-local top (weight-invariant). MASS is near-distinct (it breaks
    the 9-way empty=11 tie), spreading the top so the EFE can discriminate."""
    raw = len(nodes) + sum(len(n["fillers"]) for n in nodes.values())
    if raw < GROWTH_FLOOR:
        return 0.0
    secs = [n for n in nodes.values()
            if n["binder"] in ("eightfold-phase", "loose-section") and n["parent"]]
    mass = sum(GAP_THIN - n["sub_count"] for n in secs if n["sub_count"] < GAP_THIN)
    return round(min(mass, MASS_CAP) / MASS_CAP, 3)


EVENTS_DEFAULT = Path(__file__).parent.parent / "data/mission-fold-events.jsonl"
# EVENT CONTRACT — the live render (codex-1) appends ONE json line per operator
# interaction; this is the intake that closes the operator-evidence loop:
#   {"mission":"M-war-machine","frame":"MAP","action":"expand"}   # action: expand|collapse
# frame = the spine frame title (substring-matched). Events drive SALIENCE only
# (behavioural); :gap-score stays structural (a stub is a stub regardless of clicks).
def load_events(path=EVENTS_DEFAULT):
    import json as _json
    traces = {}
    p = Path(path)
    if not p.exists():
        return traces
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = _json.loads(line)
        except ValueError:
            continue
        m, act, fr = e.get("mission"), e.get("action"), e.get("frame")
        if m and act in ("expand", "collapse") and fr:
            traces.setdefault(m, []).append((act, fr))
    return traces


def build_view(mission, trace=None):
    """The fold→WM-selection CONTRACT: the reduced, salience-ranked minimal view
    the WM picks from instead of the raw hairball. Salience is SIP-prior until the
    live render emits real expand/collapse events (then `trace` makes it adaptive)."""
    sip = load_sip()
    tree, stem = load_tree(mission)
    nodes, roots = build(tree, sip)
    spine = [c for r in roots for c in nodes[r]["children"] if nodes[c]["sub_count"]]
    mass_max = max((nodes[s]["sub_mass"] for s in spine), default=1.0)
    post = {s: seed_prior(nodes[s]["sub_mass"], mass_max) for s in spine}
    for action, sub in (trace or []):
        for s in spine:
            if sub.lower() in nodes[s]["title"].lower():
                post[s]["alpha" if action == "expand" else "beta"] += 1
                break
    frames = [{
        "frame": nodes[s]["title"],
        "binder": nodes[s]["binder"],
        "salience": mean(post[s]),
        "concept-count": nodes[s]["sub_count"],
        "top-concepts": top_sip(nodes[s]["sub_fillers"], sip, 4),
    } for s in sorted(spine, key=lambda s: -mean(post[s]))]
    return {
        "mission": stem,
        "gap-score": gap_score(nodes),   # PRIMARY: the WM's cross-mission expansion signal
        "raw-count": len(nodes) + sum(len(n["fillers"]) for n in nodes.values()),
        "visible-count": len(frames),
        "spine": frames,                 # tail (low :salience) = within-mission gap localizer
    }


if __name__ == "__main__":
    import argparse, glob as _glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-view", action="store_true",
                    help="emit the fold→WM-selection contract (edn) over all missions")
    ap.add_argument("--from-events", nargs="?", const=str(EVENTS_DEFAULT), default=None,
                    help="fold real operator expand/collapse events into salience "
                         "(default log: data/mission-fold-events.jsonl)")
    a = ap.parse_args()
    if a.emit_view:
        missions = [Path(f).stem for f in sorted(_glob.glob(
            str(Path(__file__).parent.parent / "data/mission-scope-trees/*.json")))]
        traces = load_events(a.from_events) if a.from_events else {}
        n_ev = sum(len(t) for t in traces.values())
        view = {"source": "mission-fold",
                "note": (f"salience adaptive from {n_ev} operator events" if n_ev
                         else "salience is SIP-prior; operator-adaptive once render emits events"),
                "missions": [build_view(m, trace=traces.get(m)) for m in missions]}
        out = Path(__file__).parent.parent / "data/mission-fold-view.edn"
        out.write_text(_edn(view))
        print(f"wrote {out}  ({len(view['missions'])} missions)\n")
        print("  WM GLOBAL RANKING (by :gap-score — expansion targets first):")
        for m in sorted(view["missions"], key=lambda m: -m["gap-score"]):
            print(f"    gap={m['gap-score']:.3f}  {m['mission']:26} "
                  f"(raw {m['raw-count']}, spine {m['visible-count']})")
        print("\n  sample (war-machine):")
        print(_edn(build_view("war-machine")))
    else:
        # synthetic operator: cares about the mission-web (keeps opening MAP),
        # already knows the framing (keeps collapsing IDENTIFY).
        TRACE = [
            ("expand", "MAP"), ("collapse", "IDENTIFY"), ("expand", "MAP"),
            ("collapse", "IDENTIFY"), ("expand", "Integration"), ("expand", "MAP"),
            ("collapse", "IDENTIFY"),
        ]
        run("war-machine", TRACE)
