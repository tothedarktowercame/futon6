#!/usr/bin/env python3
# mission_fold.py — the FOLD operator: a repeatable curation of a mission's
# dense scope-neighbourhood into a laptop-legible projection.
#
# THESIS (Joe, 2026-06-08): the fold is the *ideal first example* of Bayesian
# structure learning (M-bayesian-structure-learning §3.4 "Bayesian model
# reduction"): find the SIMPLER generative model (the folded spine) that
# explains the neighbourhood equally well, where "equally well" = every concept
# remains reachable by expansion. It is BSL you can SEE — local, small, no
# superpod, no gold. The pieces:
#   * generative model      = the full scope-tree (every scope + every concept)
#   * model reduction       = collapse concept-slots + sub-scopes into their
#                             parent frame; show counts + a SIP-ranked sample
#   * posteriors-not-counters = the SIP/distinctiveness score (prior+kernel)
#                             decides WHICH concepts surface per frame
#   * epistemic value (§3.3) = which frame to EXPAND = the one whose folded
#                             subtree carries the most distinctive (SIP) mass
#   * scope-binding (§7b)   = the fold's units ARE the partly-filled scopes
#
# Repeatable: keyed only by hx/parent (nesting) + the SIP lexicon, so the SAME
# operator folds any mission (and is the default render for the mined corpus).
import json, glob, argparse
from pathlib import Path

ROOT = Path("/home/joe/code/futon6")
TREES = ROOT / "data" / "mission-scope-trees"
LEX = ROOT / "data" / "mission-self-representing-lexicon.json"

STRUCT_ROLES = {"entity", "environment", "heading"}  # frame scaffolding, not fillers


def load_sip():
    d = json.loads(LEX.read_text())
    return {r["term"]: r["score"] for r in d["terms"]}


def load_tree(name):
    f = [x for x in glob.glob(str(TREES / "*.json"))
         if Path(x).stem == name or Path(x).stem == "M-" + name][0]
    return json.loads(Path(f).read_text()), Path(f).stem


def fillers(edge):
    """The bound slots of a scope (concepts/capabilities/map-items/...),
    i.e. ends that are content, not frame-scaffolding."""
    out = []
    for e in edge.get("hx/ends") or edge.get("ends") or []:
        if e.get("role") in STRUCT_ROLES:
            continue
        term = e.get("term") or e.get("name") or e.get("ident") or e.get("title")
        if term:
            out.append((e["role"], term))
    return out


def title(edge):
    for e in edge.get("hx/ends") or edge.get("ends") or []:
        if e.get("role") == "heading" and e.get("title"):
            return e["title"]
        if e.get("role") == "environment" and e.get("name"):
            return e["name"]
    return edge.get("scope-id", "?")


def build(tree, sip):
    """Index scopes; attach own fillers + own SIP mass; build child lists;
    then compute subtree aggregates (the folded content of each frame)."""
    edges = tree.get("scope-hyperedges") or tree.get("scope_hyperedges")
    nodes = {}
    for e in edges:
        sid = e["hx/id"]
        fs = fillers(e)
        nodes[sid] = {
            "id": sid,
            "binder": e.get("binder-type") or e.get("hx/type", "").split("/")[-1],
            "title": title(e),
            "parent": e.get("hx/parent") or e.get("parent"),
            "fillers": fs,
            "own_mass": sum(sip.get(t, 0.0) for _, t in fs),
            "children": [],
        }
    roots = []
    for n in nodes.values():
        p = n["parent"]
        if p and p in nodes:
            nodes[p]["children"].append(n["id"])
        else:
            roots.append(n["id"])

    # subtree aggregates (post-order): folded concept count + SIP mass + all fillers
    def agg(nid, seen):
        if nid in seen:
            return 0, 0.0, []
        seen.add(nid)
        n = nodes[nid]
        cnt = len(n["fillers"])
        mass = n["own_mass"]
        allf = list(n["fillers"])
        for c in n["children"]:
            cc, cm, cf = agg(c, seen)
            cnt += cc; mass += cm; allf += cf
        n["sub_count"] = cnt
        n["sub_mass"] = mass
        n["sub_fillers"] = allf
        return cnt, mass, allf
    seen = set()
    for r in roots:
        agg(r, seen)
    return nodes, roots


def top_sip(fs, sip, k=4):
    ranked = sorted(set(t for _, t in fs), key=lambda t: -sip.get(t, 0.0))
    return ranked[:k]


def render(nodes, roots, sip, depth=1, expand=None):
    raw_scopes = len(nodes)
    raw_slots = sum(len(n["fillers"]) for n in nodes.values())
    raw_total = raw_scopes + raw_slots

    lines = []
    visible = [0]

    def walk(nid, d, prefix):
        n = nodes[nid]
        visible[0] += 1
        folded = n["sub_count"]
        tip = ", ".join(top_sip(n["sub_fillers"], sip)) if folded else ""
        badge = f"  ⊞ {folded} concepts" + (f"  [top: {tip}]" if tip else "") if folded else ""
        lines.append(f"{prefix}{n['binder']:16} {n['title'][:42]:42}{badge}")
        if d < depth:
            for c in sorted(n["children"], key=lambda x: -nodes[x]["sub_mass"]):
                walk(c, d + 1, prefix + "    ")
    for r in sorted(roots, key=lambda x: -nodes[x]["sub_mass"]):
        walk(r, 0, "  ")

    print(f"  RAW (the hairball the render would draw): "
          f"{raw_scopes} scopes + {raw_slots} concept-slots = {raw_total} nodes")
    print(f"  FOLDED (depth {depth}): {visible[0]} frames visible  "
          f"→ {100*(1-visible[0]/raw_total):.0f}% compression, every concept ≤{depth+1} expands away\n")
    print("\n".join(lines))

    # epistemic value: which frame, if expanded, reveals the most distinctive mass?
    frontier = [n for n in nodes.values() if n["children"] and n["sub_count"]]
    frontier.sort(key=lambda n: -n["sub_mass"])
    if frontier:
        best = frontier[0]
        print(f"\n  EXPECTED INFO-GAIN (§3.3): expand '{best['title'][:40]}' "
              f"(SIP mass {best['sub_mass']:.1f}) → reveals {best['sub_count']} concepts: "
              f"{', '.join(top_sip(best['sub_fillers'], sip, 6))}")

    if expand:
        for n in nodes.values():
            if expand.lower() in n["title"].lower():
                print(f"\n  ── expanded '{n['title']}' ──")
                for role, term in sorted(n["sub_fillers"], key=lambda x: -sip.get(x[1], 0)):
                    print(f"     {role:10} {term:24} sip={sip.get(term,0):.2f}")
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mission", nargs="?", default="war-machine")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--expand", default=None)
    ap.add_argument("--all", action="store_true", help="fold all 6, compare spines")
    a = ap.parse_args()
    sip = load_sip()
    if a.all:
        for f in sorted(glob.glob(str(TREES / "*.json"))):
            tree = json.loads(Path(f).read_text())
            nodes, roots = build(tree, sip)
            raw = len(nodes) + sum(len(n["fillers"]) for n in nodes.values())
            spine = sum(1 for r in roots for _ in [0]) + sum(
                len(nodes[r]["children"]) for r in roots)
            print(f"{Path(f).stem:26} raw {raw:4d} nodes  →  spine {spine:3d} frames "
                  f"({100*(1-spine/raw):.0f}% fold)")
        return
    tree, stem = load_tree(a.mission)
    nodes, roots = build(tree, sip)
    print(f"================  FOLD: {stem}  ================\n")
    render(nodes, roots, sip, depth=a.depth, expand=a.expand)


if __name__ == "__main__":
    main()
