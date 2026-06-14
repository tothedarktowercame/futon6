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
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

W = Path("/home/joe/code/futon6/data/warp")
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


def main():
    defidx = json.load(open(W / "defined-index.json"))["concept_to_papers"]
    defc = defaultdict(lambda: {"variants": set(), "papers": set()})
    for term, papers in defidx.items():
        c = canon(term)
        if c:
            defc[c]["variants"].add(term)
            defc[c]["papers"].update(papers)

    conc = json.load(open(W / "concordance.json"))["terms"]
    usedc = defaultdict(lambda: {"variants": set(), "used": set(), "defined": set()})
    for term, rows in conc.items():
        c = canon(term)
        if not c:
            continue
        u = usedc[c]
        u["variants"].add(term)
        for r in rows:
            (u["defined"] if r.get("role") == "defined" else u["used"]).add(r.get("paper"))

    hit = []
    for c, u in usedc.items():
        d = defc.get(c)
        if not d or is_noise(c):
            continue
        defpapers = d["papers"] | u["defined"]
        hit.append({
            "concept": c,
            "variants": sorted(u["variants"] | d["variants"])[:12],
            "n_variants": len(u["variants"] | d["variants"]),
            "used_papers": len(u["used"]),
            "defining_papers": len(defpapers),
            "defining_sample": sorted(defpapers)[:8],
        })
    hit.sort(key=lambda r: -r["used_papers"])
    frontier = sorted([h for h in hit if h["defining_papers"] <= 2 and h["used_papers"] >= 10],
                      key=lambda r: -r["used_papers"])
    (W / "hitlist.json").write_text(json.dumps({
        "schema": "hitlist-v1", "n_groundable": len(hit),
        "hitlist": hit[:4000], "frontier": frontier[:200]}))
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
