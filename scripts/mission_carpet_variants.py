#!/usr/bin/env python3
# mission_carpet_variants.py — the FAMILY of mission layouts (Joe, 2026-06-12).
# The carpet (springs), the BGE embedding, and the EFE terrain are competing PROJECTIONS of a
# latent mission-metric we don't yet know. So don't pick one — emit them ALL and compare; their
# convergence/disagreement IS the diagnostic (see memory feedback_projections_converge_on_metric).
#
# Produces mission-carpet-pos-{force,embed,springs,seed}.json (same coord frame, drop-in for
# mission_efe_field.py) + a cross-method pairwise-distance agreement matrix.
#   force   = existing force sim over citation + pattern-road springs (no embedding) — baseline
#   embed   = MDS of BGE cosine-distance — the embedding IS the metric, made 2D
#   springs = force sim + BGE-cosine springs (semantics pulls alongside citations/roads)
#   seed    = init from embed (MDS), then relax with the force springs (semantic backbone + graph)
import re, math, json, glob
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.manifold import MDS

ROOT = Path("/home/joe/code")
ATT = json.load(open(ROOT / "futon6/data/pattern-attestation.json")).get("by_name", {})

# ---- graph (citations + attestation-weighted pattern roads), mirrors mission_carpet.py ----
paths = {p.stem: p for p in ROOT.glob("futon*/holes/**/M-*.md")}
stems = sorted(paths)
idx = {s: i for i, s in enumerate(stems)}
N = len(stems)
texts = {s: paths[s].read_text(errors="ignore") for s in stems}

refs = defaultdict(set); indeg = Counter()
for s in stems:
    for r in set(re.findall(r'\bM-[a-z0-9][a-z0-9-]+', texts[s])):
        if r in paths and r != s:
            refs[s].add(r); indeg[r] += 1
parent = {s: (max(refs[s], key=lambda r: (indeg[r], r)) if refs[s] else None) for s in stems}

flex = {Path(f).stem for f in glob.glob(str(ROOT / "futon*/library/**/*.flexiarg"), recursive=True)}
flex = {b for b in flex if b.count("-") >= 2 and len(b) >= 12}
applied = {s: {b for b in flex if b in texts[s]} for s in stems}
pm = defaultdict(set)
for s in stems:
    for b in applied[s]:
        pm[b].add(s)
pair_att = defaultdict(int)
for g, ms in pm.items():
    a = ATT.get(g, 0); ms = sorted(ms)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            pair_att[(ms[i], ms[j])] = max(pair_att[(ms[i], ms[j])], a)

E = []
for s in stems:
    if parent[s]:
        E.append((idx[s], idx[parent[s]], 0.9))
for (a, b), w in pair_att.items():
    E.append((idx[a], idx[b], 0.02 + 0.006 * min(w, 200)))

def force_sim(P0, extra_edges=()):
    """The mission_carpet.py force integrator, from seed P0, optionally + extra springs."""
    edges = list(E) + list(extra_edges)
    ei = np.array([e[0] for e in edges]); ej = np.array([e[1] for e in edges])
    ek = np.array([e[2] for e in edges])
    deg = np.bincount(np.concatenate([ei, ej]), minlength=N).astype(float)
    P = P0.copy()
    for it in range(460):
        diff = P[:, None, :] - P[None, :, :]
        d2 = (diff ** 2).sum(-1) + 1.0
        rep = (diff / d2[..., None]).sum(1) * 1300.0
        att = np.zeros((N, 2))
        f = (P[ej] - P[ei]) * ek[:, None] * 0.036
        np.add.at(att, ei, f); np.add.at(att, ej, -f)
        grav = (-P) * (0.022 / (1.0 + deg))[:, None]
        P += (rep + att + grav) * (0.85 ** (it / 60))
        P -= P.mean(0)
    return P

def normalize(P):
    P = P - P.min(0)
    return P * (3200 / max(P.max(0))) + 200

# ---- BGE embedding, aligned to stems by basename ----
bge = json.load(open(ROOT / "futon3a/resources/notions/bge_mission_embeddings.json"))
recs = bge if isinstance(bge, list) else bge.get("records", bge)
vec = {}
for r in recs:
    b = r.get("basename") or ""
    if b in idx and "vector" in r and b not in vec:
        vec[b] = np.asarray(r["vector"], float)
emb_stems = [s for s in stems if s in vec]                       # missions WITH a BGE vector
V = np.array([vec[s] for s in emb_stems])                        # already unit-normalized
cos = V @ V.T                                                    # cosine similarity
cosd = np.clip(1.0 - cos, 0.0, 2.0)                              # cosine distance metric

# embed positions via metric MDS on the cosine-distance matrix (the metric, made 2D)
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=7,
          normalized_stress="auto", n_init=4, max_iter=400)
emb_xy = mds.fit_transform(cosd)
emb_pos = {s: emb_xy[i] for i, s in enumerate(emb_stems)}

def embed_full():
    """Positions for ALL stems: BGE-MDS where available, else centroid of cited-neighbours
    that DO have a position (so citation-only missions still land sensibly), else global centroid."""
    P = np.zeros((N, 2)); gc = emb_xy.mean(0)
    placed = dict(emb_pos)
    for s in stems:
        if s in placed:
            P[idx[s]] = placed[s]
        else:
            nbrs = [placed[r] for r in refs[s] if r in placed] + \
                   [placed[r] for r in stems if s in refs[r] and r in placed]
            P[idx[s]] = np.mean(nbrs, 0) if nbrs else gc
    return P

# ---- BGE-cosine springs (top-k semantic neighbours per mission) for the 'springs' variant ----
K = 6
bge_edges = []
order = np.argsort(-cos, axis=1)
for i, s in enumerate(emb_stems):
    for j in order[i, 1:K + 1]:
        c = cos[i, j]
        if c > 0.45:                                            # only real semantic affinity
            bge_edges.append((idx[s], idx[emb_stems[j]], 0.25 * (c - 0.45) / 0.55))

# ---- build the four variants (same coordinate frame) ----
rng = np.random.default_rng(7)
P_force = force_sim(rng.standard_normal((N, 2)) * 280)
P_embed = embed_full()
P_springs = force_sim(rng.standard_normal((N, 2)) * 280, extra_edges=bge_edges)
P_seed = force_sim(normalize(P_embed) - 1700)                   # seed from embed (centred), then relax
variants = {"force": P_force, "embed": P_embed, "springs": P_springs, "seed": P_seed}

for name, P in variants.items():
    Pn = normalize(P)
    json.dump({s: [round(float(Pn[idx[s]][0]), 1), round(float(Pn[idx[s]][1]), 1)] for s in stems},
              open(ROOT / f"futon6/data/mission-carpet-pos-{name}.json", "w"))
    print(f"wrote mission-carpet-pos-{name}.json")

# ---- cross-method agreement: correlate pairwise mission-distance matrices over emb_stems ----
def pdist_vec(P):
    ii = [idx[s] for s in emb_stems]
    Q = normalize(P)[ii]
    d = np.sqrt(((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    return d[np.triu_indices(len(ii), 1)]
names = list(variants)
D = {n: pdist_vec(variants[n]) for n in names}
print("\ncross-method pairwise-distance agreement (Spearman ρ over "
      f"{len(emb_stems)} embedded missions):")
from scipy.stats import spearmanr
print("        " + "  ".join(f"{n:>8}" for n in names))
for a in names:
    row = []
    for b in names:
        rho = 1.0 if a == b else spearmanr(D[a], D[b]).statistic
        row.append(f"{rho:+.3f}")
    print(f"{a:>8} " + "  ".join(f"{v:>8}" for v in row))
print("\n(low off-diagonal ρ = the projections DISAGREE there = metric still undetermined;\n"
      " high ρ = robust structure all methods recover. MDS stress = "
      f"{mds.stress_:.1f})")
print(f"\n{N} missions · {len(emb_stems)} with BGE vectors · {N-len(emb_stems)} citation-placed · "
      f"{len(bge_edges)} BGE springs")
