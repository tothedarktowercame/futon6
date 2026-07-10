#!/usr/bin/env python3
"""Gates-as-code for E-fold-embed-pipeline (library/data-mining/gates-as-code).
Torch-free, stdlib-only — runs on the LAPTOP before the box is rented (smoke-before-the-paid-run),
and again on the box output. Author != producer: this checker is independent of train_fold_embed.py.

  python scripts/fold_embed/check_fold_embed_gates.py --data data/fold-embed            # DATASET gate (pre-spend)
  python scripts/fold_embed/check_fold_embed_gates.py --data data/fold-embed --scorecards  # OUTPUT gate (post-run)

Exits nonzero on any HARD-gate fail — so the runner can bail before/after the paid run.
"""
import json, argparse, os, statistics, sys, glob

def load(d):
    nodes=[json.loads(l) for l in open(f"{d}/nodes.jsonl")]
    edges=[json.loads(l) for l in open(f"{d}/edges.jsonl")]
    pairs=[json.loads(l) for l in open(f"{d}/pairs.jsonl")]
    return nodes,edges,pairs

def dataset_gate(d):
    nodes,edges,pairs=load(d)
    ids={n["id"] for n in nodes}
    varids={n["id"] for n in nodes if n["type"]=="var"}
    fails=[]; ok=lambda b: "PASS" if b else "FAIL"
    print(f"== DATASET gate: {d} ==  nodes={len(nodes)} edges={len(edges)} pairs={len(pairs)}")

    # G-typing: every node typed in the known vocabulary
    types={}
    for n in nodes: types[n.get("type","?")]=types.get(n.get("type","?"),0)+1
    untyped=sum(v for k,v in types.items() if k not in ("var","pattern","mission","namespace"))
    hard=untyped==0; fails+=[] if hard else ["G-typing"]
    print(f"  [{ok(hard)}] G-typing        types={types} untyped/unknown={untyped}")

    # G-edge-integrity: edge endpoints resolve to nodes
    dangling=sum(1 for s,t,_ in edges if s not in ids or t not in ids)
    hard=dangling==0; fails+=[] if hard else ["G-edge-integrity"]
    print(f"  [{ok(hard)}] G-edge-integrity dangling-endpoint edges={dangling} ({dangling/max(len(edges),1):.3%})")

    # G-split-leakage: a mission lives in exactly one split (no train/test leakage)
    by_mission={}
    for p in pairs: by_mission.setdefault(p["mission"],set()).add(p["split"])
    leaked=[m for m,s in by_mission.items() if len(s)>1]
    counts={}
    for p in pairs: counts[p["split"]]=counts.get(p["split"],0)+1
    hard=not leaked; fails+=[] if hard else ["G-split-leakage"]
    print(f"  [{ok(hard)}] G-split-leakage  splits={counts} missions-in-2-splits={leaked or 'none'}")

    # G-eval-nonvacuous: every val/test pair has >=1 pos that is a REAL var node.
    #   A pair whose pos vars are absent scores recall=0 vacuously -> degenerate, un-diagnostic eval.
    for split in ("val","test"):
        sp=[p for p in pairs if p["split"]==split]
        empty=[p["mission"] for p in sp if not [v for v in p["pos"] if v in varids]]
        hard=not empty; fails+=[] if hard else [f"G-eval-nonvacuous/{split}"]
        print(f"  [{ok(hard)}] G-eval-nonvacuous/{split:4} pairs={len(sp)} with-no-resolvable-pos={empty or 'none'}")

    # G-cascade-signal: pairs with empty/unresolved cascade fall back to the GLOBAL MEAN in the
    #   scorer (casc_vec -> H.mean(0)) => no query signal, identical for all such pairs. SOFT gate,
    #   but a high share means the embedding arms are being judged largely on a constant query.
    def casc_ok(p): return len([c for c in p["cascade"] if c in ids])>0
    for split in ("train","val","test"):
        sp=[p for p in pairs if p["split"]==split]
        if not sp: continue
        withsig=sum(1 for p in sp if casc_ok(p))
        share=withsig/len(sp)
        soft=share>=0.5
        print(f"  [{ok(soft)}] G-cascade-signal/{split:5} {withsig}/{len(sp)} pairs have a resolvable cascade ({share:.0%})"
              + ("" if soft else "  <- SOFT: many queries collapse to the global mean"))

    # G-hard-negatives: every pair carries hard negatives (else the loss/eval sees only random contrast)
    nohn=sum(1 for p in pairs if not p.get("hard_neg"))
    soft=nohn==0
    print(f"  [{ok(soft)}] G-hard-negatives pairs-without-hard_neg={nohn}  (SOFT)")

    # Degree sanity (informational): zero-degree nodes can't get structural signal
    deg={}
    for s,t,_ in edges:
        deg[s]=deg.get(s,0)+1; deg[t]=deg.get(t,0)+1
    degs=[deg.get(n["id"],0) for n in nodes]
    zero=sum(1 for x in degs if x==0)
    print(f"  [info] degree           min={min(degs)} median={int(statistics.median(degs))} "
          f"max={max(degs)} zero-degree-nodes={zero} ({zero/len(nodes):.1%})")

    print(f"== DATASET gate: {'PASS' if not fails else 'FAIL '+','.join(fails)} ==")
    return not fails

def output_gate(d):
    scores={}
    for m in ("text","struct","hybrid"):
        p=f"{d}/scorecard-{m}.json"
        if os.path.exists(p): scores[m]=json.load(open(p))
    if not scores:
        print(f"== OUTPUT gate: no scorecards in {d} (run train_fold_embed.py first) =="); return False
    anys=next(iter(scores.values()))
    pop=anys.get("popularity",{}).get("recall@20",0.0)
    fails=[]; ok=lambda b:"PASS" if b else "FAIL"
    print(f"== OUTPUT gate: {d} ==  popularity-recall@20={pop}")
    best_m,best_r=None,-1.0
    for m in ("text","struct","hybrid"):
        if m not in scores: continue
        r=scores[m].get(m,{}).get("recall@20",0.0)
        finite = r==r and r not in (float("inf"),float("-inf"))  # NaN/inf guard
        if not finite: fails.append(f"G-finite/{m}")
        print(f"  [{ok(finite)}] {m:7} recall@20={r} finite={finite}")
        if finite and r>best_r: best_m,best_r=m,r
    text_r=scores.get("text",{}).get("text",{}).get("recall@20",0.0)
    beats_pop=best_r>pop; beats_text=(best_m!="text" and best_r>text_r)
    # HARD: the winner must at least clear popularity (else the ansatz shows nothing on this task)
    fails+=[] if beats_pop else ["G-beats-popularity"]
    print(f"  [{ok(beats_pop)}] G-beats-popularity winner={best_m}({best_r}) vs popularity({pop})")
    # SOFT/reporting: structure adds over text-only (the excursion's actual question)
    print(f"  [{ok(beats_text)}] G-structure-adds  winner={best_m} vs BGE-text({text_r})  (report either way)")
    print(f"== OUTPUT gate: {'PASS' if not fails else 'FAIL '+','.join(fails)} ==")
    return not fails

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/fold-embed")
    ap.add_argument("--scorecards",action="store_true",help="run the OUTPUT gate on scorecard-*.json instead of the dataset gate")
    a=ap.parse_args()
    ok = output_gate(a.data) if a.scorecards else dataset_gate(a.data)
    sys.exit(0 if ok else 1)

if __name__=="__main__": main()
