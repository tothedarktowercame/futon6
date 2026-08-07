#!/usr/bin/env python3
"""Per-proof comprehension score — the FLOOR of E-comprehension-foundation.

Comprehension(proof | corpus) = R2d (nouns grounded) ⊕ rung-3 (strategies grounded),
both keyed on the normalized substrate. It is corpus-relative: re-running against a
richer corpus raises it. It GATES the verdict — you may only call a proof "weak"
once you have understood it; otherwise the signal is "weak extraction / study more",
never "flawed".

This COMPOSES existing machinery, inventing nothing:
  - noun comprehension N  = r2d_concept_coverage.check_graph(...)  (defined/known/imported vs undefined)
  - strategy comprehension S = data/rung3-technique/<pid>.technique.json (grounded vs thin/ungrounded;
                               author-declared conjectures are CREDITED, not penalized)

Verdict gate (thresholds LO/HI, tunable):
  comp = min(N, S)                       # you understand a proof as well as its weaker axis
  comp < LO                  -> weak-extraction      (didn't understand it; need richer corpus / help)
  LO <= comp < HI            -> partial-comprehension
  comp >= HI and ungrounded>0 -> weak-proof          (genuine unjustified gap, comprehension established)
  comp >= HI and conjectures>0-> open-problem-bearing (credited author-declared gaps — the valuable kind)
  comp >= HI otherwise       -> well-formed

Usage:
  futon6/.venv/bin/python scripts/clean_comprehension.py \
      [--graphs data/iatc-argument-graphs/loop-run-70b] \
      [--rung3 data/rung3-technique/loop-run-70b] [--lo 0.5 --hi 0.8] \
      [--out data/showcases/clean-demo/comprehension.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r2d_concept_coverage as r2d  # noqa: E402
import cas_select  # noqa: E402
import rung3_technique as r3  # noqa: E402
import strategy_recognizer as sr  # noqa: E402


def strategy_buckets(pid, steps_dir, rung3_dir, patterns):
    """Live rung-3 from cas-select-steps if present (so the floor reflects the
    CURRENT pattern pool — re-runs lift as the pool grows); else fall back to a
    precomputed technique.json."""
    sp = steps_dir / f"{pid}.steps.json"
    if sp.exists():
        return r3.gapmap_for_steps(json.load(open(sp)), patterns)["buckets"]
    tp = rung3_dir / f"{pid}.technique.json"
    if tp.exists():
        return json.load(open(tp)).get("buckets", {})
    return None


def strategy_score(b, thin_credit):
    if not b:
        return None, {}
    moves = sum(b.values())
    conj = b.get("conjecture", 0)
    grounded = b.get("grounded-by-pattern", 0) + b.get("grounded-by-citation", 0)
    thin = b.get("thin", 0)
    assessable = moves - conj
    # resolution-graded credit: a verified pattern is full comprehension; a thin
    # (recognized-but-unverified) match is partial — we see the move's shape, not
    # its detail (the hologram's lower-resolution layer). conjectures excluded.
    S = (grounded + thin_credit * thin) / assessable if assessable > 0 else None
    return S, {"moves": moves, "grounded": grounded, "thin": thin,
               "ungrounded": b.get("ungrounded", 0), "conjecture": conj}


def verdict(N, S, ungrounded, conj, lo, hi):
    axes = [x for x in (N, S) if x is not None]
    if not axes:
        return "no-structure", None
    comp = min(axes)
    if comp < lo:
        return "weak-extraction", comp
    if comp < hi:
        return "partial-comprehension", comp
    if ungrounded and ungrounded > 0:
        return "weak-proof", comp
    if conj and conj > 0:
        return "open-problem-bearing", comp
    return "well-formed", comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/iatc-argument-graphs/loop-run-70b")
    ap.add_argument("--steps", default="data/cas-select-steps/loop-run-70b")
    ap.add_argument("--rung3", default="data/rung3-technique/loop-run-70b")
    ap.add_argument("--candidates", default="data/iatc-candidates",
                    help="dir of *.candidate.json for prose-sourced strategy recognition")
    ap.add_argument("--thin-credit", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--out", default="data/showcases/clean-demo/comprehension.json")
    ap.add_argument("--substrate-papers", help="file of paper-ids: scope grounding to this "
                    "RUN-CORPUS (so comprehension rises with the run — finding #1 / accretion sweep)")
    ap.add_argument("--run-dir", help="if set, emit S5 MetricRecords here (INSTANTIATE-GPU)")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    args = ap.parse_args()

    substrate = r2d.load_substrate(r2d.parse_args([]))
    if args.substrate_papers:   # scope grounding to the run-corpus (finding #1)
        raw = [l.strip() for l in open(args.substrate_papers) if l.strip()]
        ids = set(raw) | {i.replace("__", "/") for i in raw}   # match concept-index id form
        substrate = r2d.restrict_substrate(substrate, ids)
        print(f"  (substrate scoped to run-corpus: {len(raw)} papers, {len(substrate.concept_index)} concepts in scope)")
    patterns = cas_select.load_patterns()
    vocab = sr.load_vocab(str(ROOT / "holes/clean/tactic-gesture-vocab.edn"))
    graph_dir = ROOT / args.graphs
    steps_dir = ROOT / args.steps
    rung3_dir = ROOT / args.rung3

    rows = []
    skipped = []
    # Sidecar rung-2 reports are not argument graphs; scoring them produced 98
    # spurious `no-structure` verdict rows that any reader of the distribution
    # then had to discount by hand (run_artifacts, 2026-08-07).
    try:
        import sys as _sys
        _h = os.path.dirname(os.path.abspath(__file__))
        if _h not in _sys.path:
            _sys.path.insert(0, _h)
        from run_artifacts import proof_graphs
        _graph_paths = [Path(x) for x in proof_graphs(str(graph_dir))]
    except Exception:
        _graph_paths = sorted(x for x in graph_dir.glob("*.edn")
                              if not x.name.endswith(".rung2.edn"))
    for path in _graph_paths:
        pid = path.stem
        try:  # per-graph isolation: a single malformed graph (e.g. illegal EDN from the
            # model) must not sink the whole comprehension batch
            res = r2d.check_graph(r2d.load_edn(path), path, substrate)
        except Exception as exc:
            skipped.append((pid, str(exc)[:80]))
            print(f"  SKIP {pid}: {type(exc).__name__} {str(exc)[:80]}")
            continue
        N = res["rate"]
        nb = res["buckets"]
        S_r3, sb = strategy_score(strategy_buckets(pid, steps_dir, rung3_dir, patterns),
                                  args.thin_credit)
        # prose-sourced recognition on the candidate source-window — strategy
        # gestures live in the prose, not the distilled IATC claims (E-strategy-recognizer)
        S_prose = None
        cf = ROOT / args.candidates / f"{pid}.candidate.json"
        if cf.exists():
            window = json.load(open(cf)).get("source-window", "")
            pb, _ = sr.recognize_text(window, vocab)
            S_prose = sr.strat_score(pb, args.thin_credit)
        cands = [x for x in (S_r3, S_prose) if x is not None]
        S = max(cands) if cands else None   # comprehended if EITHER method grounds the strategy
        ung, conj = sb.get("ungrounded", 0), sb.get("conjecture", 0)
        v, comp = verdict(N, S, ung, conj, args.lo, args.hi)
        rows.append({"pid": pid, "noun": N, "strategy": S, "strategy_rung3": S_r3,
                     "strategy_prose": S_prose, "comprehension": comp,
                     "verdict": v, "undefined_nouns": res["undefined"][:6],
                     "ungrounded_moves": ung, "conjectures": conj,
                     "noun_buckets": nb, "strategy_buckets": sb,
                     "gap_to_ceiling": (None if comp is None else round(1 - comp, 3))})

    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"lo": args.lo, "hi": args.hi, "proofs": rows},
              open(ROOT / args.out, "w"), indent=2)

    if args.run_dir:  # S5 inline metrics: weak-point (flag) + confidence + symbol-grounding by kind
        try:
            import sys as _sys
            _sys.path.insert(0, str(ROOT / "scripts"))
            import metric_harness as mh
            for r in rows:
                pid = r["pid"]
                def em(metric, val, axis="completeness"):
                    if val is not None:
                        mh.emit_record(args.run_dir, run_id=args.run_id, corpus_id=args.corpus_id,
                                       paper_id=pid, stage="S5", metric=metric, axis=axis,
                                       value=round(float(val), 4), computable=True)
                # weak-POINT iff genuine gap with comprehension established (never weak-extraction)
                em("weak-point", 1.0 if r["verdict"] in ("weak-proof", "open-problem-bearing") else 0.0)
                em("comprehension-confidence", r["comprehension"], axis="accretion")
                em("symbol-grounding/named-concept", r["noun"])     # noun axis = named concepts
                em("symbol-grounding/proof-move", r["strategy"])    # verb axis = proof-moves
        except Exception as ee:
            print(f"  (S5 metric emit skipped: {ee})")

    def fmt(x):
        return "  -- " if x is None else f"{x:.2f}"
    print(f"{'paper':14s} {'noun':>5s} {'s:r3':>5s} {'s:pr':>5s} {'strat':>5s} {'comp':>5s}  verdict")
    print("-" * 76)
    from collections import Counter
    vc = Counter()
    for r in sorted(rows, key=lambda r: (r["comprehension"] is None, r["comprehension"] or 0)):
        vc[r["verdict"]] += 1
        extra = ""
        if r["conjectures"]:
            extra += f"  +{r['conjectures']} conj"
        print(f"{r['pid']:14s} {fmt(r['noun']):>5s} {fmt(r.get('strategy_rung3')):>5s} "
              f"{fmt(r.get('strategy_prose')):>5s} {fmt(r['strategy']):>5s} "
              f"{fmt(r['comprehension']):>5s}  {r['verdict']}{extra}")
    print("-" * 64)
    print("verdict distribution:", dict(vc))
    tot_g = sum(r["strategy_buckets"].get("grounded", 0) for r in rows)
    tot_thin = sum(r["strategy_buckets"].get("thin", 0) for r in rows)
    tot_ung = sum(r["strategy_buckets"].get("ungrounded", 0) for r in rows)
    tot_conj = sum(r["strategy_buckets"].get("conjecture", 0) for r in rows)
    print(f"\nstrategy moves: grounded={tot_g}  thin={tot_thin}  ungrounded={tot_ung}  "
          f"conjecture(credited)={tot_conj}")
    print(f"  CPU levers to raise strategy comprehension before Linode:")
    print(f"   - VERIFY lever:    thin={tot_thin} moves matched a pattern but it's not "
          f"verifiable -> promote/verify -> grounded")
    print(f"   - RETRIEVAL lever: ungrounded={tot_ung} moves hit no pattern at all -> "
          f"widen hotwords / embedding retrieval -> at least thin")
    print("\nThe gate in action: 'weak-proof' is only ever assigned to proofs we"
          "\ncomprehend (comp>=hi); everything below that is 'we didn't understand it',"
          "\nnot 'the proof is flawed' — exactly the comprehension-as-self-knowledge floor.")


if __name__ == "__main__":
    main()
