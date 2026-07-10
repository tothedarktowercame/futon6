#!/usr/bin/env python3
"""CONCEPT-TAG (noun axis): NNexus-style concept auto-link over turns + missions.

Tags load-bearing CONCEPTS (patterns, missions, capabilities, futonic-logic vocabulary, components,
R-criteria) in turn + mission text, building the inverted index concept→{turns, missions}. Per
futonic-logic, a hotword firing is 香 (embodied salience), NOT identification — so the CPU spotter only
GROUNDS high-precision surfaces (canonical id literals + distinctive multi-word phrases); single common
tokens are flagged weak-salience (the 間 false-salience risk), deferred to the LLM layer.

Payoff: concept→turns ⊕ concept→missions composes into turn→mission via shared grounded concepts — a
model-free routing bridge that strengthens the weak autoclock link the rest of the pipeline routes around.

  futon6/.venv/bin/python scripts/mission_concept_tag.py [--turns N]
"""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meme_mine_runner import read_asks

ROOT = "/home/joe/code/futon6"; OUT = f"{ROOT}/data/meme-mine"
F3A = "/home/joe/code/futon3a/resources/notions"
FUTONIC = {"composition": "ft/composition", "articulation": "ft/articulation", "salience": "ft/salience",
           "recognition loop": "ft/recognition-loop", "free energy": "ft/free-energy", "precision": "ft/precision",
           "expected free energy": "ft/efe", "cascade": "ft/cascade", "rollout": "ft/rollout",
           "structure learning": "ft/structure-learning", "niche construction": "ft/niche-construction",
           "active inference": "ft/active-inference", "differential operator": "ft/differential-operator",
           "evidence landscape": "ft/evidence-landscape", "sorry factory": "ft/sorry-factory"}
COMPONENTS = {"war machine": "component/war-machine", "agency": "component/agency", "drawbridge": "component/drawbridge",
              "substrate-2": "component/substrate-2", "street sweeper": "component/street-sweeper",
              "neo4j": "tech/neo4j", "pgvector": "tech/pgvector", "vllm": "tech/vllm", "xtdb": "tech/xtdb"}


def gazetteer():
    """surface-phrase -> (canonical-id, tier). Grounded surfaces only: id-literals + multi-word phrases."""
    gaz = {}
    def add(surface, cid, tier):
        s = surface.lower().strip()
        if s and (s not in gaz or tier == "grounded"):
            gaz[s] = (cid, tier)
    # missions (198): the hyphenated stem as a phrase
    for s in json.load(open(f"{ROOT}/data/diffsub-scopes.json")):
        m = s.get("mission")
        if m:
            phr = m.replace("-", " ")
            if len(phr.split()) >= 2:
                add(phr, f"mission/M-{m}", "grounded")
    # patterns (490): last path-seg as a phrase
    try:
        for e in json.load(open(f"{F3A}/minilm_pattern_embeddings.json")):
            seg = e["id"].rsplit("/", 1)[-1].replace("-", " ")
            if len(seg.split()) >= 2:
                add(seg, f"pattern/{e['id']}", "grounded")
    except Exception:
        pass
    # capabilities
    try:
        for cap in json.load(open(f"{ROOT}/data/capability-graph.json")):
            phr = cap.replace("-", " ")
            if len(phr.split()) >= 2:
                add(phr, f"scope/capability/{cap}", "grounded")
    except Exception:
        pass
    for term, cid in {**FUTONIC, **COMPONENTS}.items():
        add(term, cid, "grounded" if len(term.split()) >= 2 else "weak")
    return gaz


# id-literal patterns (always grounded): M-*, R\d+, agent ids
LIT = re.compile(r"\b(M-[a-z0-9][a-z0-9-]{3,}|R\d+[a-z]?|(?:claude|codex|fable)-\d+)\b", re.I)


def spot(text, gaz):
    low = text.lower()
    hits = {}  # cid -> tier
    for lit in LIT.findall(text):
        kind = "hole" if re.match(r"R\d", lit) else ("agent" if "-" in lit and lit[0].lower() in "cf" else "mission")
        hits[f"{kind}/{lit if kind!='hole' else lit.lower()}"] = "grounded"
    for surf, (cid, tier) in gaz.items():
        if re.search(rf"\b{re.escape(surf)}\b", low):
            hits[cid] = "grounded" if tier == "grounded" else hits.get(cid, "weak")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=300)
    a = ap.parse_args()
    gaz = gazetteer()
    print(f"gazetteer: {len(gaz)} grounded-capable concept surfaces + id-literals (M-* / R\\d / agent)")

    # mission text = stem + its scope concepts (diffsub-scopes 'concepts')
    mis_concepts = defaultdict(list)
    for s in json.load(open(f"{ROOT}/data/diffsub-scopes.json")):
        if s.get("mission"):
            mis_concepts[s["mission"]] += (s.get("concepts") or [])
    mission_tags = {m: spot(m.replace("-", " ") + " " + " ".join(c), gaz) for m, c in mis_concepts.items()}

    asks = read_asks(a.turns)
    turn_tags = {s["id"]: spot(s["ask"], gaz) for s in asks}

    # inverted index concept -> {missions, turns} (grounded only)
    idx = defaultdict(lambda: {"missions": [], "turns": []})
    for m, hits in mission_tags.items():
        for cid, t in hits.items():
            if t == "grounded":
                idx[cid]["missions"].append(m)
    for tid, hits in turn_tags.items():
        for cid, t in hits.items():
            if t == "grounded":
                idx[cid]["turns"].append(tid)
    json.dump({c: v for c, v in idx.items()}, open(f"{OUT}/concept-index.json", "w"), indent=2)

    tagged_turns = sum(1 for h in turn_tags.values() if any(t == "grounded" for t in h.values()))
    print(f"tagged: {len(mission_tags)} missions · {len(asks)} turns ({tagged_turns} with >=1 grounded concept)")
    co = Counter(c for c, v in idx.items() if v["turns"] and v["missions"])
    print(f"concepts bridging BOTH turns and missions (the turn→mission routing substrate): {len(co)}")
    # turn→mission via shared grounded concept — examples
    bridged = []
    for tid, hits in turn_tags.items():
        ms = set()
        for cid, t in hits.items():
            if t == "grounded":
                ms.update(m for m in idx[cid]["missions"])
        if ms:
            bridged.append((tid, sorted(ms)[:3]))
    print(f"turns routed to >=1 mission via shared concept: {len(bridged)}/{len(asks)} "
          f"(model-free turn→mission — complements the weak autoclock)")
    for tid, ms in bridged[:4]:
        print(f"   {tid} → {ms}")
    print(f"wrote {OUT}/concept-index.json")


if __name__ == "__main__":
    main()
