#!/usr/bin/env python3
# mission_pattern_scopes.py — the PATTERN-SCOPE LAYER (Joe, 2026-06-08).
# A pattern application IS a scope. Per mission, two signals kept SEPARATE (combining-methods
# discipline — their disagreement is the diagnostic):
#   APPLIED  = distinctive flexiargs literally cited in the mission text (deliberate pattern-scopes);
#   NEAR-MISS = patterns geometrically closest (MiniLM cosine, Patterns+Missions embedding) but NOT
#              applied = the WM's TRY-A-PATTERN candidates (the "what gene would fit here that it lacks").
# Emits mission-pattern-scopes.edn: the HGT genes + the try-a-pattern menu + claude-1 pre-witness input.
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/home/joe/code")
OUT = ROOT / "futon6" / "data" / "mission-pattern-scopes.edn"
pe = json.load(open(ROOT / "futon3a/resources/notions/minilm_pattern_embeddings.json"))
me = json.load(open(ROOT / "futon3a/resources/notions/minilm_mission_embeddings.json"))

# distinctive pattern basenames (>=3 words) -> mean MiniLM vector
flex = defaultdict(list)
for r in pe:
    b = Path(r["source"]).stem if r.get("source") else r["id"].split("/")[0]
    if b.count("-") >= 2 and len(b) >= 12:
        flex[b].append(r["vector"])
pat_names = sorted(flex)
pat_vec = np.array([np.mean(flex[b], axis=0) for b in pat_names])
pat_vec /= np.linalg.norm(pat_vec, axis=1, keepdims=True) + 1e-9
patset = set(pat_names)

mvec = {r["basename"]: np.array(r["vector"], dtype=float) for r in me if r.get("basename")}
paths = {p.stem: p for p in ROOT.glob("futon*/holes/**/M-*.md")}

# literal applied pattern-scopes (deliberate citation in text)
applied = {}
for s, p in paths.items():
    t = p.read_text(errors="ignore")
    applied[s] = {b for b in patset if b in t}

# near-misses: cosine to every pattern, top-8 NOT applied
rows = {}
covered = 0
for s in paths:
    v = mvec.get(s)
    if v is None:
        rows[s] = {"applied": sorted(applied[s]), "near": None}
        continue
    covered += 1
    vv = v / (np.linalg.norm(v) + 1e-9)
    cos = pat_vec @ vv
    ap = applied[s]
    near = [(pat_names[i], round(float(cos[i]), 3))
            for i in np.argsort(-cos) if pat_names[i] not in ap][:8]
    rows[s] = {"applied": sorted(ap), "near": near}

# HGT edges (literal shared genes)
pm = defaultdict(set)
for s in paths:
    for b in applied[s]:
        pm[b].add(s)
hgt = len({(a, b) for ms in pm.values() if len(ms) >= 2 for a in ms for b in ms if a < b})

def edn():
    out = [f'{{:source "mission-pattern-scopes" :n-patterns {len(pat_names)} :hgt-edges {hgt}',
           " :missions ["]
    for s in sorted(rows):
        r = rows[s]
        ap = " ".join(chr(34) + a + chr(34) for a in r["applied"])
        nm = (" ".join(f'{{:pattern "{p}" :cos {c}}}' for p, c in r["near"])
              if r["near"] is not None else "")
        out.append(f'  {{:mission "{s}" :applied [{ap}] :try-candidates [{nm}]}}')
    return "\n".join(out) + " ]}\n"
OUT.write_text(edn())

print(f"{len(pat_names)} distinctive pattern-scopes; {covered}/{len(paths)} missions have embeddings")
print(f"HGT edges (literal shared genes): {hgt}")
print(f"\nexamples (APPLIED = cited genes | NEAR-MISS = try-a-pattern candidates):")
for s in ("M-war-machine", "M-capability-star-map", "M-canon-fingerprint-store", "M-emacs-cursor-peripheral"):
    if s in rows and rows[s]["near"] is not None:
        r = rows[s]
        print(f"\n  {s}")
        print(f"    applied: {', '.join(r['applied'][:6]) or '(none)'}")
        print(f"    try    : {', '.join(f'{p}({c})' for p, c in r['near'][:5])}")
print(f"\nwrote {OUT}")
