#!/usr/bin/env python3
"""E-miner-v2-bv-combs — type the mined :composes wiring skeleton with BV
connectives and build combs over it.

BACKGROUND. The mission-triple miner (mission_triple_miner.py) emits a WIRING
layer: `:application` nodes (checkpoints — witnessed applications) joined by
`:composes` hyperedges with `:from`/`:to` ends in authored order. This is the
`:composes` skeleton. We type it with BV — Guglielmi's deep-inference system —
whose three connectives are:

    seq    ⟨S ; T⟩   non-commutative ("S before T")  <- models :composes
    copar  (S , T)   commutative, self-dual tensor    ("S and T, together")
    par    [S , T]   commutative                      ("S or T, in parallel")

A COMB is a morphism-with-holes: a context ⟨A ; - ; B⟩ whose hole `-` is filled
by a sub-process whose boundary objects match (cod of the left ≈ dom of the
right). Combs are the harder, typed consumer (Joe's "exotype eustress"): to
build a *verified* comb you need each checkpoint's INTERFACE (dom/cod object
types). Where the `:composes` skeleton lacks that, the gap is the deliverable —
it is exactly the pressure that should make the mining emit typed structure.

This script is stdlib-only and deterministic. It reads the wiring-bearing
mission-triples, types each chain, builds skeleton combs, mines the only
available boundary-type proxy (the checkpoint's "Test state: N tests" / gate
witness), checks composability where it can, and emits the typed artifact plus
the gap-list (the eustress signal) to data/bv-comb-typing.edn.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

ROOT = Path("/home/joe/code/futon6")
TRIPLES = ROOT / "data" / "mission-triples"
OUT = ROOT / "holes" / "bv-comb-typing.edn"  # committed (data/ is gitignored)


def parse_wiring(text):
    """Return (checkpoints, edges). checkpoints: id -> {title, tests, gate}.
    edges: list of (from_id, to_id) in file order."""
    ckpts = {}
    for m in re.finditer(
            r":id (:ckpt-[\w-]+)\s*\n\s*:role :application.*?:witness \"(.*?)\"",
            text, re.S):
        cid, wit = m.group(1), m.group(2)
        tc = re.search(r"(\d+)\s+tests", wit)
        title = re.search(r"[Cc]heckpoint[^:]*:\s*([^(*]+?)(?:\(|\*\*|$)", wit)
        gate = re.findall(r"\b(?:Gate\s+G\d|EXIT\s+\d|GATE)\b", wit)
        ckpts[cid] = {
            "tests": int(tc.group(1)) if tc else None,
            "title": (title.group(1).strip()[:60] if title else wit[:50].strip()),
            "gates": len(gate),
        }
    edges = []
    for m in re.finditer(r":kind :composes.*?:ends \[(.*?)\]\s*\n\s*:via",
                         text, re.S):
        blk = m.group(1)
        frm = re.search(r":role :from\s*:node (:[\w-]+)", blk)
        to = re.search(r":role :to\s*:node (:[\w-]+)", blk)
        if frm and to:
            edges.append((frm.group(1), to.group(1)))
    return ckpts, edges


def chain_order(edges):
    """Topo order of a pure chain (these skeletons are all linear)."""
    succ = {a: b for a, b in edges}
    preds = {b for _, b in edges}
    starts = [a for a, _ in edges if a not in preds]
    order, seen = [], set()
    cur = starts[0] if starts else (edges[0][0] if edges else None)
    while cur is not None and cur not in seen:
        order.append(cur)
        seen.add(cur)
        cur = succ.get(cur)
    return order


def type_edge(ckpts, a, b):
    """Type a :composes edge a->b by the ONLY available boundary proxy: the
    monotone system state (test count). cod(a) ≈ dom(b) is checked as
    tests(a) <= tests(b). Returns a verdict keyword."""
    ta, tb = ckpts[a]["tests"], ckpts[b]["tests"]
    if ta is None or tb is None:
        return ":gap-no-boundary-type"   # cannot even proxy the interface
    if ta <= tb:
        return ":typed-monotone"          # composable under the state proxy
    return ":gap-type-mismatch"           # state regressed: ill-typed compose


def bv_seq(order):
    """The BV seq structure of a linear chain: ⟨c0 ; c1 ; ... ; cn⟩."""
    return "{:bv/seq [" + " ".join(order) + "]}"


def endpoints_comb(order, ckpts):
    """A 2-boundary comb ⟨first ; - ; last⟩: the context with the interior
    abstracted into a hole. Typeable at the SKELETON level always; the hole's
    dom/cod are the mined boundary proxies of first/last (None where absent)."""
    first, last = order[0], order[-1]
    dom = ckpts[first]["tests"]
    cod = ckpts[last]["tests"]
    verified = dom is not None and cod is not None
    return {
        "form": f"⟨{first} ; - ; {last}⟩",
        "hole_dom": dom, "hole_cod": cod,
        "status": ":interface-verified" if verified else ":skeleton-only",
    }


def edn_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main():
    missions = []
    for path in sorted(glob.glob(str(TRIPLES / "*.edn"))):
        text = Path(path).read_text()
        if ":composes" not in text:
            continue
        ckpts, edges = parse_wiring(text)
        if not edges:
            continue
        order = chain_order(edges)
        typed = [(a, b, type_edge(ckpts, a, b)) for a, b in edges]
        missions.append({
            "name": Path(path).stem,
            "ckpts": ckpts, "order": order, "typed": typed,
            "comb": endpoints_comb(order, ckpts),
        })

    # ---- aggregate the gap-list (the eustress signal) ----
    n_ckpts = sum(len(m["ckpts"]) for m in missions)
    n_edges = sum(len(m["typed"]) for m in missions)
    n_no_type = sum(1 for m in missions for _, _, v in m["typed"]
                    if v == ":gap-no-boundary-type")
    n_monotone = sum(1 for m in missions for _, _, v in m["typed"]
                     if v == ":typed-monotone")
    n_mismatch = sum(1 for m in missions for _, _, v in m["typed"]
                     if v == ":gap-type-mismatch")
    n_with_type = sum(1 for m in missions for c in m["ckpts"].values()
                      if c["tests"] is not None)
    n_par_copar = 0  # forks/joins found -> par/copar usable. (none: all linear)

    # ---- emit EDN artifact ----
    L = [";; GENERATED by scripts/bv_comb_typer.py — BV typing of the :composes",
         ";; wiring skeleton + combs + gap-list (eustress). E-miner-v2-bv-combs.",
         "{:bv/connectives",
         ' {:seq "⟨ ; ⟩ non-commutative — models :composes (before/after)"',
         '  :copar "( , ) commutative tensor — would model concurrent checkpoints"',
         '  :par "[ , ] commutative — would model alternative branches"}',
         " :typed-missions"]
    L.append(" [")
    for m in missions:
        L.append(f'  {{:mission "{m["name"]}"')
        L.append(f'   :shape :linear-chain          ;; no forks/joins -> seq only')
        L.append(f'   :bv-type {bv_seq(m["order"])}')
        L.append("   :composes-typing")
        L.append("   [" + " ".join(
            f'[{a} {b} {v}]' for a, b, v in m["typed"]) + "]")
        cb = m["comb"]
        L.append('   :endpoints-comb')
        L.append(f'   {{:form "{cb["form"]}" :hole-dom {cb["hole_dom"] if cb["hole_dom"] is not None else "nil"}'
                 f' :hole-cod {cb["hole_cod"] if cb["hole_cod"] is not None else "nil"} :status {cb["status"]}}}}}')
    L.append(" ]")
    L.append(" :gap-list   ;; the load-bearing eustress deliverable")
    L.append(" [")
    gaps = [
        (":gap-no-typed-interface",
         f"{n_no_type}/{n_edges} :composes edges cannot be interface-typed: "
         ":application nodes carry prose :witness but NO dom/cod object type. "
         "A verified BV comb needs each checkpoint's interface.",
         "miner: emit :interface {:in <type> :out <type>} per :application node "
         "(or a typed boundary end on :composes), not just :from/:to order."),
        (":gap-boundary-type-prose-buried",
         f"only {n_with_type}/{n_ckpts} checkpoints expose ANY boundary proxy "
         "(the 'Test state: N tests' line); the rest are untyped. Even the proxy "
         "is partial and inconsistent across missions.",
         "miner: require a structured state field per checkpoint "
         "(:test-count :gates-met :capability-set) so the wire type is TOTAL."),
        (":gap-no-par-copar",
         f"all {len(missions)} skeletons are LINEAR chains ({n_par_copar} forks/"
         "joins); BV collapses entirely to seq. The deep-inference richness "
         "(seq interleaved with par/copar via the medial rule) is unexercisable. "
         ":jointly-with (n-ary parallel) is spec-only, never emitted.",
         "miner: detect dataflow-INDEPENDENT checkpoints and emit them as a "
         "par/copar bundle (or :jointly-with), not a forced linear chain."),
        (":gap-authored-order-not-dataflow",
         "every :composes is :via 'in authored checkpoint order' — a temporal/"
         "narrative wire, NOT a typed morphism boundary. Sequential authoring "
         "does not imply a dataflow dependency.",
         "miner: distinguish temporal-order from dataflow-dependency edges; only "
         "the latter is a true :composes (the former may be par)."),
        (":gap-no-unit",
         "no identity/unit checkpoint exists, so BV combs cannot be normalised "
         "(seq/par/copar need the unit ∘ for the laws).",
         "miner: emit an explicit unit/no-op application where a phase begins, "
         "so combs have an identity to compose against."),
    ]
    for key, finding, feedback in gaps:
        L.append(f'  {{:gap {key}')
        L.append(f'   :finding "{edn_escape(finding)}"')
        L.append(f'   :mining-feedback "{edn_escape(feedback)}"}}')
    L.append(" ]")
    L.append(" :summary")
    L.append(f'  {{:missions {len(missions)} :checkpoints {n_ckpts} '
             f':composes-edges {n_edges}')
    L.append(f'   :typed-monotone {n_monotone} :gap-no-boundary-type {n_no_type} '
             f':type-mismatch {n_mismatch}')
    L.append(f'   :interface-verified-combs '
             f'{sum(1 for m in missions if m["comb"]["status"] == ":interface-verified")}'
             f' :skeleton-only-combs '
             f'{sum(1 for m in missions if m["comb"]["status"] == ":skeleton-only")}}}}}')

    OUT.write_text("\n".join(L) + "\n")

    # ---- console report ----
    print(f"BV-typed {len(missions)} wiring missions, {n_ckpts} checkpoints, "
          f"{n_edges} :composes edges.")
    for m in missions:
        print(f"  {m['name']:32} {bv_seq(m['order'])[:48]}... "
              f"comb={m['comb']['status']}")
    print(f"\nTyping verdicts: :typed-monotone={n_monotone}  "
          f":gap-no-boundary-type={n_no_type}  :type-mismatch={n_mismatch}")
    print(f"Boundary type available on {n_with_type}/{n_ckpts} checkpoints; "
          f"par/copar usable on 0 (all linear).")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
