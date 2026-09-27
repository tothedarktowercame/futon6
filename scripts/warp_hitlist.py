#!/usr/bin/env python3
"""WARP hit-list (Joe's plan, step 2): cross the corpus-wide defined-index with
the concordance into a ranked, groundable concept hit-list.

Each concept is CANONICALIZED (dash/case/whitespace unified so
'Frobenius-Perron', 'Frobenius–Perron', 'frobenius perron' collapse to one) but
its observed SURFACE VARIANTS are KEPT — that multiplicity is paraphrase signal
for the GPU stage, not noise to discard.

A concept is a definition-SCOPE: grounding it (concept -> a defining paper) once
propagates to every paper that USES it. Ranked by used-breadth so the
highest-traffic groundable concepts come first. The 'frontier' = used widely but
defined nowhere/rarely = the residual definition debt (GPU / formalization
targets).

    warp_hitlist.py  ->  data/warp/hitlist.json
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import futon6_config as config

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

W = config.warp()
DASH = re.compile(r"[‐-―−-]")  # hyphen/en/em/minus variants


def canon(t):
    t = DASH.sub(" ", t.lower())
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# The concordance indexes LaTeX control-sequences as "terms" (\times->times,
# \delta->delta, \forall->forall). Those are NOT concepts. Real CT concepts are
# multi-word ("homotopy colimit") or specific nouns ("operad"). So: keep
# multi-word, OR a curated single-word concept; drop everything else.
STOP = set("proof lemma theorem definition remark example corollary proposition "
           "keywords abstract introduction references acknowledgements notation "
           "set not all the strict where then thus hence let strictly every some "
           "thanks refs subsection section thm cor prop eq fig case step "
           "times circ cong subset supset subseteq cap cup oplus otimes wedge vee "
           "cdot ldots dots cdots quad qquad colon mapsto rightarrow leftarrow "
           "forall exists prod sum partial nabla infty left right langle rangle "
           "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu "
           "xi pi rho sigma tau upsilon phi chi psi omega text mathrm mathbf "
           "mathcal mathbb hbox mbox coev acute restr rhom multimap hom".split())
# curated single-word real concepts (the few that aren't multi-word):
CONCEPT_SINGLE = set("operad comodule coend topos sheaf scheme groupoid monad "
                     "comonad bialgebra coalgebra bimodule cofibration fibration "
                     "presheaf prestack stack gerbe quiver bicategory dendroidal "
                     "polytope matroid quasicategory simplicial".split())
JOURNAL = re.compile(r"\b(math|algebra|soc|journal|adv|ann|proc|trans|amer|appl|"
                     r"pure|geom|topol|preprint|arxiv|izv|nauk|mat|sb)\b")


def is_noise(c):
    words = c.split()
    if len(JOURNAL.findall(c)) >= 2:           # 'j pure appl algebra', 'adv math'
        return True
    if all(w in STOP for w in words):          # all-stopword phrases
        return True
    if len(words) == 1:                        # single token: only curated concepts
        return c not in CONCEPT_SINGLE
    if len(c) < 5:
        return True
    return False


def parse_args(argv):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=4000,
                    help="How many groundable concepts to keep, by used-breadth (0 = all). "
                         "The downstream stages build on exactly this list.")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="A file of canonical concepts (one per line, # comments) that are not "
                         "groundable for this corpus -- e.g. phrasing a caller has judged generic "
                         "to all mathematical writing rather than particular to the subject.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write (default: the WARP directory's hitlist.json).")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    import concept_filter

    active_filter = concept_filter.configured()
    excluded: set[str] = set()
    if args.exclude is not None:
        excluded = {line.split("#", 1)[0].strip()
                    for line in args.exclude.read_text(encoding="utf-8").splitlines()}
        excluded.discard("")
    defidx = json.load(open(W / "defined-index.json"))["concept_to_papers"]
    defc = defaultdict(lambda: {"variants": set(), "papers": set()})
    for term, papers in defidx.items():
        c = canon(term)
        if c:
            defc[c]["variants"].add(term)
            defc[c]["papers"].update(papers)

    # Read one term at a time: a corpus-scale concordance costs ~4x its size to json.load,
    # and only these per-concept aggregates are kept. Terms whose canonical form nothing
    # defines are dropped as they stream past, exactly as the loop below would drop them.
    import warp_concordance

    usedc = defaultdict(lambda: {"variants": set(), "used": set(), "defined": set()})
    meta: dict = {}
    for term, rows in warp_concordance.iter_concordance_terms(W / "concordance.json", meta=meta):
        c = canon(term)
        if not c or c not in defc:
            continue
        u = usedc[c]
        u["variants"].add(term)
        for r in rows:
            paper = r.get("paper")
            # interned: the same paper id recurs across tens of thousands of terms
            (u["defined"] if r.get("role") == "defined" else u["used"]).add(
                sys.intern(paper) if isinstance(paper, str) else paper)

    # A concordance mined from raw e-prints carries usage as per-concept paper COUNTS
    # rather than rows (the rows ran to tens of millions). They count exactly what
    # len(used-set) counts from rows, so the ranking is the same computation.
    concept_used = meta.get(warp_concordance.CONCEPT_USED_PAPERS)
    if concept_used is not None:
        for c in concept_used:
            if c in defc:
                usedc[c]    # a concept only ever USED in prose still enters the ranking

    hit = []
    for c, u in usedc.items():
        d = defc.get(c)
        if not d or is_noise(c) or c in excluded:
            continue
        if active_filter is not None and active_filter.match(c) is not None:
            continue
        defpapers = d["papers"] | u["defined"]
        hit.append({
            "concept": c,
            "variants": sorted(u["variants"] | d["variants"])[:12],
            "n_variants": len(u["variants"] | d["variants"]),
            "used_papers": concept_used.get(c, 0) if concept_used is not None else len(u["used"]),
            "defining_papers": len(defpapers),
            "defining_sample": sorted(defpapers)[:8],
        })
    hit.sort(key=lambda r: -r["used_papers"])
    frontier = sorted([h for h in hit if h["defining_papers"] <= 2 and h["used_papers"] >= 10],
                      key=lambda r: -r["used_papers"])
    kept = hit[:args.cap] if args.cap > 0 else hit
    (args.out or W / "hitlist.json").write_text(json.dumps({
        "schema": "hitlist-v1", "n_groundable": len(hit),
        "hitlist": kept, "frontier": frontier[:200]}))
    print(f"groundable concepts (used AND defined, noise-filtered): {len(hit)}")
    print("=== top 18 groundable (by used-breadth) ===")
    for h in hit[:18]:
        v = f"  [{h['n_variants']} surface variants]" if h["n_variants"] > 1 else ""
        print(f"  {h['used_papers']:5}u {h['defining_papers']:4}d  {h['concept']}{v}")
    print("=== frontier: used>=10 but definers<=2 (residual debt) top 10 ===")
    for h in frontier[:10]:
        print(f"  {h['used_papers']:5}u {h['defining_papers']:4}d  {h['concept']}")
    # canonicalization sanity: did frobenius-perron variants collapse?
    fp = next((h for h in hit if "frobenius perron" in h["concept"]), None)
    print("=== canon check: frobenius-perron ===")
    print("  ", {k: fp[k] for k in ("concept", "variants", "used_papers", "defining_papers")} if fp else "(not in hitlist)")


if __name__ == "__main__":
    raise SystemExit(main())
