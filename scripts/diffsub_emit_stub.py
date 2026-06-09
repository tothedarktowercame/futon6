#!/usr/bin/env python3
# diffsub_emit_stub.py — M-differentiable-substrate (claude-3): emit a STUB move-set in the
# LOCKED §3.1 shape so claude-4's rollout consumer builds against the real contract NOW.
#
# The scorer here is a HAND heuristic over the materialized substrate-2 metric (det-holes +
# Salingaros class + capability frontier/ascent) — a stand-in for the real grad(loss)(A)
# producer. The DERIVE step replaces THIS scorer with the gradient; the OUTPUT SHAPE is the
# locked contract and does not change. Reachability is the rollout's gate: :conjectural moves
# are not-yet-reachable (islands); the consumer renormalizes :prior over its reachable subset.
#
# Output: futon6/data/diffsub-moves-stub.edn
import json, re, math, time
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/joe/code")
SCOPES = json.load(open("/tmp/scopes.json"))
CAPS = json.load(open(ROOT / "futon6/data/capability-graph.json"))
CLS = dict(re.findall(r':mission "M-([^"]+)" :class :(\w+)',
                      (ROOT / "futon6/data/mission-wholeness.edn").read_text()))
FRONTIER = {"capability-scope", "pattern", "psr", "pur"}
CLASSW = {"mess": 1.00, "pipeline": 0.65, "alive": 0.40, "stub": 0.55}

moves = []
claimed = lambda cap: bool(CAPS.get(cap, {}).get("claimed"))

# 1. :advance-capability — REAL cap ids. Reachable summit (claimed ascent-parent) vs
#    conjectural island (no claimed parent — no terrain yet).
for cap, info in CAPS.items():
    if info["claimed"]:
        continue
    claimed_parents = [p for p in info.get("scope", []) if claimed(p)]
    if claimed_parents:                                   # SUMMIT — reachable
        have, conf = f"scope/capability/{claimed_parents[0]}", "claimed-substrate"
        sc = 2.7 if info.get("frontier") else 1.8
    else:                                                 # ISLAND — not-yet-reachable
        have, conf = f"scope/conjectural/{cap}-foothold", "conjectural"
        sc = 1.9 if info.get("frontier") else 1.2
    moves.append(dict(cls="advance-capability", have=have, want=f"scope/capability/{cap}",
                      adv=cap, score=sc, conf=conf, note=info.get("title", "")[:60]))

# 2. :close-hole — representative open (:detached) holes in the highest-open missions.
#    No per-scope id in this dump → stub-flagged ids (the real producer emits substrate-2 ids).
det_by_m = defaultdict(lambda: defaultdict(int))
for s in SCOPES:
    if s["det"]:
        det_by_m[s["m"]][s["binder"]] += 1
for m, binders in sorted(det_by_m.items(), key=lambda kv: -sum(kv[1].values()))[:6]:
    binder = max(binders, key=binders.get)
    cw = CLASSW.get(CLS.get(m, ""), 0.40)
    sc = (1.0 + (0.30 if binder in FRONTIER else 0.0)) * (0.70 + cw)
    moves.append(dict(cls="close-hole", have=f"scope/{m}/{binder}#open",
                      want=f"scope/{m}/{binder}#closed", adv=None, score=round(sc, 3),
                      conf="claimed-substrate",
                      note=f"{sum(binders.values())} open holes in {m} ({CLS.get(m,'?')})"))

# 3. :graft-pattern — proposed (conjectural) pattern grafts.
for m, pat in [("invariant-queue-unstuck", "logic-model-before-code"),
               ("self-documenting-stack", "combining-methods-as-diagnostic")]:
    moves.append(dict(cls="graft-pattern", have=f"scope/{m}/map-item#0",
                      want=f"scope/pattern/{pat}", adv=None, score=1.3, conf="conjectural",
                      note=f"propose grafting {pat} onto {m}"))

# 4. :centre-mess — restructure the highest-open mess district (the Salingaros remedy).
mess = [m for m, c in CLS.items() if c == "mess" and m in det_by_m]
if mess:
    mm = max(mess, key=lambda m: sum(det_by_m[m].values()))
    moves.append(dict(cls="centre-mess", have=f"scope/{mm}/cluster", want=f"scope/{mm}/centred",
                      adv=None, score=1.7, conf="claimed-substrate", note=f"centre the {mm} mess district"))

# prior = softmax(score); rank desc; delta-g = first-order descent on the cost-metric (negative=good)
moves.sort(key=lambda d: -d["score"])
mx = max(d["score"] for d in moves)
exps = [math.exp(d["score"] - mx) for d in moves]
Z = sum(exps)
for i, (d, e) in enumerate(zip(moves, exps)):
    d["prior"] = round(e / Z, 4)
    d["rank"] = i + 1
    d["delta_g"] = -round(d["score"] * 0.08, 4)


def edn_move(d):
    adv = f'"{d["adv"]}"' if d["adv"] else "nil"
    # :move/terminal? — true ⇒ the rollout carries its g-cost but does NOT expand through it.
    # Only :centre-mess in v1: its transition T is a compound cluster graph-rewrite whose
    # mechanism (pattern→wiring→structure) isn't built yet (M-memes). Don't fabricate its T.
    term = "true" if d["cls"] == "centre-mess" else "false"
    return (f'  {{:move/id "{d["have"]}->{d["want"]}" :move/class :{d["cls"]}'
            f' :have "{d["have"]}" :want "{d["want"]}" :advances-cap {adv}'
            f' :score {d["score"]} :prior {d["prior"]} :delta-g {d["delta_g"]}'
            f' :confidence :{d["conf"]} :rank {d["rank"]} :move/terminal? {term}'
            f' :note "{d["note"].replace(chr(34), chr(39))}"}}')


edn = (";; diffsub-moves-stub.edn — M-differentiable-substrate STUB move-set (claude-3).\n"
       ";; LOCKED §3.1 shape. Scores = a HAND heuristic over the materialized substrate-2 metric\n"
       ";; (det-holes + Salingaros class + capability frontier/ascent), standing in for the real\n"
       ";; grad(loss)(A) producer — DERIVE replaces the scorer, the SHAPE is the contract.\n"
       ";; CONSUMER (claude-4 rollout): reachability is YOUR gate. :conjectural = not-yet-reachable\n"
       ";; (islands, no terrain); intersect with your :constructed-reachable set per node and\n"
       ";; RENORMALIZE :prior over the survivors. Sim on your COPY (no 7071 writes during search).\n"
       f"{{:emit/at {int(time.time())}\n"
       " :emit/metric {:compose :additive :epistemic :C-holes :pragmatic :cap-ascent :C-variant :salingaros}\n"
       f" :emit/k {len(moves)}\n"
       " :emit/stub? true\n"
       " :moves [\n" + "\n".join(edn_move(d) for d in moves) + "\n ]}\n")
OUT = ROOT / "futon6/data/diffsub-moves-stub.edn"
OUT.write_text(edn)
print(f"wrote {OUT} — {len(moves)} moves")
for d in moves:
    print(f"  {d['rank']:2d}. score={d['score']:.2f} p={d['prior']:.3f} :{d['cls']:18s} "
          f"{d['conf']:16s} {d['have']} -> {d['want']}")
