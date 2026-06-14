#!/usr/bin/env python3
# mission_wholeness.py — Alexander/Salingaros LIFE (mana) over the mission scope-tree.
# Liaison claude-3 <-> claude-2 (M-pudding-peradams 11.4-11.6.1), 2026-06-08.
#
#   L = T*H  (Life=mana) ;  C = T*(10-H)  (mess) ;  (10-H) = architectural entropy.
#   T = density of differentiated/complementary centres.
#   H = harmony = how much centres reduce each other's entropy (mutual reinforcement).
#   Scaling (law iii): ideal n = 1 + ln(x_max) - ln(x_min); order ~ how close ACTUAL
#   distinct-scale count N is to n; a MISSING intermediate scale (empty log-e band) => pathological.
#
# CENTRE = a scope; SIZE = its subtree leaf-concept count (sub_count) = Salingaros "extent"
# (claude-2's call, fixes the median-per-depth artifact). SCALES = log-e bands of size.
# H = cross-PHASE thread density (concepts binding >=2 phases = centres reinforcing) — avoids
# the sub-scope binder-duplication that would inflate a scope-level coherence count.
#
# POC: the 0-10 assignment rubric (peradams 11.6.1) is still open; T/H normalisations here are
# defensible proxies. The ROBUST outputs are N-vs-n + the pathology flag + the relative L-ranking.
import sys, glob, math, json, re
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mission_fold import load_sip, load_tree, build, top_sip  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "mission-wholeness.edn"
PHASES = ["head","identify","map","derive","argue","verify","instantiate","document"]
GAP_THIN = 8  # a section-frame with >= this many concepts counts as articulated (for T1)


def canon(t):
    u = t.strip().upper()
    return next((v for v in PHASES if u == v.upper() or re.match(v.upper()+r'(\b|[^A-Z])', u)), None)


def wholeness(stem, sip):
    tree, _ = load_tree(stem)
    nodes, roots = build(tree, sip)
    centres = [n for n in nodes.values() if n["sub_count"] > 0]
    sizes = [n["sub_count"] for n in centres]
    if len(sizes) < 2:
        return None
    xmax, xmin = max(sizes), max(1, min(sizes))
    n_ideal = max(1, round(1 + math.log(xmax) - math.log(xmin)))
    bands = {int(math.log(max(1, s))) for s in sizes}      # log-e (ratio ~e) scale bands
    lo, hi = int(math.log(xmin)), int(math.log(xmax))
    missing = [b for b in range(lo, hi + 1) if b not in bands]
    N = len(bands)

    # T (temperature = DIFFERENTIATION, Salingaros Table 5.1 / peradams 11.6.2): T1..T5 each 0/1/2.
    # NOT size — so a big-but-LINEAR mission scores low (T3 branching). Mission mapping (to validate).
    types = len({n["binder"] for n in centres})
    secs = [n for n in centres if n["binder"] in ("eightfold-phase", "loose-section")]
    artic = sum(n["sub_count"] >= GAP_THIN for n in secs) / len(secs) if secs else 0
    T1 = 2 if artic > 0.6 else 1 if artic > 0.3 else 0          # articulation/concreteness
    distinct = len({n["title"].strip().lower() for n in centres})
    T2 = 2 if distinct > 15 else 1 if distinct >= 5 else 0      # count of distinct centres
    # T3 BRANCHING (mandala vs pipeline): does the tree branch at MULTIPLE depths (mandala) or
    # only at the root (a flat pipeline of phases)? Raw out-degree can't tell — every scope-tree
    # branches once; multi-SCALE branching is the discriminator (87 pipeline / 107 mandala).
    depth = {}
    stack = [(r, 0) for r in roots]
    while stack:
        nid, d = stack.pop()
        depth[nid] = d
        stack.extend((c, d + 1) for c in nodes[nid]["children"])
    bdepths = {depth[nid] for nid in nodes
               if len({nodes[c]["title"].strip().lower() for c in nodes[nid]["children"]}) >= 2}
    T3 = 2 if len(bdepths) >= 2 else 1 if bdepths else 0
    T4 = 2 if types >= 5 else 1 if types >= 3 else 0            # diversity of centre-kinds
    fam = set()
    for n in centres:
        b = n["binder"]
        fam.add("struct" if b in ("eightfold-phase", "loose-section", "capability-scope", "map-item")
                else "rel" if b == "relates-to" else "evid" if b == "source-material"
                else "bound" if b in ("mission-scope-in", "mission-scope-out") else "x")
    fam.discard("x")
    T5 = 2 if len(fam) >= 3 else 1 if len(fam) == 2 else 0      # complementarity of kind-families
    Tcomp = dict(T1=T1, T2=T2, T3=T3, T4=T4, T5=T5)
    T = T1 + T2 + T3 + T4 + T5

    # H = mutual reinforcement = fraction of distinctive concepts bound across >=2 DISTINCT
    # centres (deduped by title to collapse the binder-duplication). A recurring concept
    # threads centres = reduces their mutual entropy. Works for ALL missions (not phase-bound).
    bytitle = {}
    for c in centres:
        bytitle.setdefault(c["title"].strip().lower(), []).append(c)
    occ = Counter()
    for group in bytitle.values():
        concs = {t for c in group for _, t in c["fillers"] if sip.get(t, 0) > 0}
        for t in concs:
            occ[t] += 1
    threaded = sum(1 for c in occ.values() if c >= 2)
    H = round(10 * threaded / len(occ), 1) if occ else 0.0

    L = round(T * H, 1)            # Salingaros eq 3: L = T*H (organized complexity = mana), [0,100]
    C = round(T * (10 - H), 1)     # eq 4: C = T*(10-H) (disorganized complexity = mess), [0,100]
    # 4 classes (claude-2 / peradams 11.6.3) — each names the REMEDY the mission needs:
    #   alive    high T (incl T3 branch), high H, low C   -> minting peradams
    #   mess     high T, low H, high C                    -> CENTRING pass (organize -> raise H)
    #   pipeline adequate centres but flat (low T, T2>=1) -> RE-STRUCTURE (branch -> raise T3 -> mandala)
    #   stub     few centres (low T, T2=0)                -> DEVELOP (add centres)
    # stub-vs-pipeline discriminator = centre count (T2). NB anamnesis-generative: alive wants
    # high-but-not-max H (open holes seed new scales), so we don't reward H->10.
    if T >= 4:
        cls = "alive" if H >= 5 else "mess"
    else:
        cls = "pipeline" if Tcomp["T2"] >= 1 else "stub"
    path = [f"missing-scale{missing}"] if missing else []
    return dict(mission=stem, centres=len(centres), N=N, n=n_ideal,
                T=T, H=H, L=L, C=C, cls=cls, Tcomp=Tcomp, pathology=path)


def main():
    sip = load_sip()
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/mission-scope-trees/*.json"))):
        stem = Path(f).stem
        try:
            w = wholeness(stem, sip)
        except Exception:
            w = None
        if w:
            rows.append(w)
    rows.sort(key=lambda r: -r["L"])
    # emit edn for the render to consume L as colour/intensity
    def edn(rs):
        out = ['{:source "mission-wholeness" :model "Salingaros L=T*H over scope-tree centres"',
               ' :missions [']
        for r in rs:
            out.append(f'  {{:mission "{r["mission"]}" :class :{r["cls"]} :L {r["L"]} :T {r["T"]} '
                       f':H {r["H"]} :C {r["C"]} :N {r["N"]} :n {r["n"]} '
                       f':pathology [{" ".join(chr(34)+p+chr(34) for p in r["pathology"])}]}}')
        return "\n".join(out) + " ]}\n"
    OUT.write_text(edn(rows))

    tally = Counter(r["cls"] for r in rows)
    print(f"=== Salingaros LIFE (L=T*H, 0-100) over {len(rows)} missions  classes={dict(tally)} ===\n")
    print("ALIVE (high T, high H):")
    for r in [x for x in rows if x["cls"] == "alive"][:8]:
        print(f"  L={r['L']:5.1f}  T={r['T']:2d} H={r['H']:4.1f}  br(T3)={r['Tcomp']['T3']}  {r['mission']}")
    print("\nMESS (high T, low H, high C — ripe for a centring pass):")
    for r in sorted([x for x in rows if x["cls"] == "mess"], key=lambda r: -r["C"])[:6]:
        print(f"  C={r['C']:5.1f} L={r['L']:5.1f}  T={r['T']:2d} H={r['H']:4.1f}  {r['mission']}")
    print("\nPIPELINE (developed but FLAT — RE-STRUCTURE: branch into a mandala):")
    for r in sorted([x for x in rows if x["cls"] == "pipeline"], key=lambda r: -r["centres"])[:6]:
        print(f"  L={r['L']:5.1f}  T={r['T']:2d} centres={r['centres']:3d}  {r['mission']}")
    print("\nKNOWN missions (class + component-T breakdown — did the re-rank fire?):")
    for t in ("M-self-documenting-stack","M-war-machine","M-capability-star-map","M-aif2",
              "M-weird-modernism","M-canon-fingerprint-store","M-xor-coupling-probe"):
        r = next((x for x in rows if x["mission"] == t), None)
        if r:
            c = r["Tcomp"]
            print(f"  {r['cls']:6} L={r['L']:5.1f} T={r['T']:2d}(art{c['T1']} n{c['T2']} br{c['T3']} "
                  f"div{c['T4']} comp{c['T5']}) H={r['H']:4.1f}  {t}")
    print("\nT=10 SANITY (claude-2: are these GENUINE flagships, or thin missions reaching 10?):")
    for r in sorted([x for x in rows if x["T"] == 10], key=lambda r: -r["centres"]):
        print(f"  H={r['H']:4.1f} centres={r['centres']:3d} {r['cls']:5}  {r['mission']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
