#!/usr/bin/env python3
"""Concept embedding from MULTIPLICITY + graph — the superpod-skip (Joe).

A concept's LOCAL STRUCTURE is its multiplicity: the surface-variant labels it
appears under + the scope-shapes (definition-snippet tokens) it's defined in +
its dependency-graph neighbours. That is the concept-level analog of the
'mission->scope' feature that gives the Futon-City EFE landscape its local
geometry — so we get neighbourhoods WITHOUT a neural/superpod run. Classical:
TF-IDF feature bag -> truncated SVD -> k-d embedding + 2D layout.

Inputs:  data/warp/hitlist.json (variants), def-snippets.json (scope-shapes),
         concept-graph.json (edges, authority)
Outputs: data/warp/concept-embed.npy  (N x k)
         data/warp/concept-carpet-pos.json  (concept -> [x,y], the EFE layout)
"""
import json
import re
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np

W = Path("/home/joe/code/futon6/data/warp")
DASH = re.compile(r"[‐-―−-]")
STOPTOK = set("the a an of to in on for and or is are be by with we have there "
              "exists every some any all that this it its as at from".split())


def toks(s):
    return [w for w in DASH.sub(" ", s.lower()).replace("-", " ").split() if w not in STOPTOK]


def main():
    hl = json.load(open(W / "hitlist.json"))["hitlist"]
    snips = json.load(open(W / "def-snippets.json"))["snippets"]
    graph = {n["concept"]: n for n in json.load(open(W / "concept-graph.json"))["authority"]}

    concepts = [h["concept"] for h in hl]
    cidx = {c: i for i, c in enumerate(concepts)}
    N = len(concepts)

    # feature bag per concept: variant tokens (v:) + scope-shape tokens (s:) +
    # neighbour concepts (g:) — the three multiplicity channels.
    feats = [defaultdict(float) for _ in range(N)]
    for i, h in enumerate(hl):
        for v in h.get("variants", []):
            for t in toks(v):
                feats[i]["v:" + t] += 1.0
        for s in snips.get(h["concept"], []):
            for t in set(toks(s.get("snippet", ""))):       # scope-shape vocab
                feats[i]["s:" + t] += 1.0
    # graph neighbours (dependency edges) as features
    edges = {}
    cg = json.load(open(W / "concept-graph.json"))
    # concept-graph stored authority only; rebuild adjacency from snippets cheaply
    cset = set(concepts)
    for i, c in enumerate(concepts):
        for s in snips.get(c, []):
            ws = toks(s.get("snippet", ""))
            grams = {" ".join(ws[j:j + n]) for n in (1, 2, 3) for j in range(len(ws) - n + 1)}
            for d in (grams & cset):
                if d != c:
                    feats[i]["g:" + d] += 1.0

    # vocab + IDF
    df = defaultdict(int)
    for f in feats:
        for k in f:
            df[k] += 1
    vocab = {k: j for j, k in enumerate(k for k in df if df[k] >= 2)}  # drop hapax
    V = len(vocab)
    idf = {k: log(N / df[k]) for k in vocab}
    M = np.zeros((N, V), dtype=np.float32)
    for i, f in enumerate(feats):
        for k, w in f.items():
            j = vocab.get(k)
            if j is not None:
                M[i, j] = w * idf[k]
    # row-normalize
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M /= np.clip(norms, 1e-9, None)

    # truncated SVD -> k-d embedding + 2D layout
    k = min(48, V - 1, N - 1)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    emb = U[:, :k] * S[:k]
    np.save(W / "concept-embed.npy", emb.astype(np.float32))
    pos = {concepts[i]: [float(emb[i, 0]), float(emb[i, 1])] for i in range(N)}
    (W / "concept-carpet-pos.json").write_text(json.dumps(pos))

    print(f"N={N} concepts, V={V} multiplicity features, k={k}-d embedding")
    # local-structure check: nearest neighbours of a few concepts (cosine in emb)
    En = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    for probe in ["monoidal category", "natural transformation", "model category",
                  "operad", "topos"]:
        if probe in cidx:
            i = cidx[probe]
            sims = En @ En[i]
            nn = sorted(range(N), key=lambda j: -sims[j])[1:7]
            print(f"  {probe:24} ~ {', '.join(concepts[j] for j in nn)}")


if __name__ == "__main__":
    raise SystemExit(main())
