#!/usr/bin/env python3
# mission_embed_diagnostics.py — M-efe-bge-followon-actions step 1 (fable-2, 2026-06-12).
#
# Settles whether the embed layout's "3x richer terrain" (1820 vs ~580 contour segs)
# is real semantic signal or an MDS-on-cosine artifact, BEFORE anything leans on it.
#
# MEASURED here (everything below is computed, nothing is prior):
#   1. Kruskal stress-1 of the real BGE-MDS layout (the reported 464 was RAW sklearn stress).
#   2. k-NN trustworthiness + continuity (k=10) between 1024-d cosine space and the 2D map.
#   3. Classical-MDS eigenspectrum of the cosine-distance matrix -> intrinsic dimensionality.
#   4. Restart stability: MDS under different seeds, Procrustes disparity vs the seed-7 layout.
#   5. Controls through the REAL terrain pipeline (mission_efe_field.py <variant>):
#        ctrl-shuf{i} — same BGE vectors, permuted mission assignment (label control)
#        ctrl-rand{i} — random gaussian unit vectors, same dimensionality (geometry control)
#      Verdict rule: if meaningless cosine matrices reproduce the embed contour count,
#      the richness is a property of MDS-of-cosine, not of the missions.
#
# Writes control position files as mission-carpet-pos-ctrl-*.json and renders
# mission-efe-field-ctrl-*.html. NEVER touches the canonical mission-efe-field.html
# (a variant arg is always passed).
import re, json, glob, subprocess, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.manifold import MDS

ROOT = Path("/home/joe/code")
HERE = Path(__file__).resolve().parent

# ---- mission stems + citation graph, mirrors mission_carpet_variants.py ----
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

bge = json.load(open(ROOT / "futon3a/resources/notions/bge_mission_embeddings.json"))
recs = bge if isinstance(bge, list) else bge.get("records", bge)
vec = {}
for r in recs:
    b = r.get("basename") or ""
    if b in idx and "vector" in r and b not in vec:
        vec[b] = np.asarray(r["vector"], float)
emb_stems = [s for s in stems if s in vec]
V = np.array([vec[s] for s in emb_stems])
M = len(emb_stems)


def cosine_dist(W):
    c = np.clip(W @ W.T, -1.0, 1.0)
    return np.clip(1.0 - c, 0.0, 2.0)


def run_mds(D, seed=7):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
              normalized_stress="auto", n_init=4, max_iter=400)
    xy = mds.fit_transform(D)
    return xy, mds.stress_


def kruskal_stress1(D, xy):
    iu = np.triu_indices(len(D), 1)
    d2 = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))[iu]
    delta = D[iu]
    return float(np.sqrt(((d2 - delta) ** 2).sum() / (d2 ** 2).sum()))


def _trust_precomp(D_orig, D_emb, k):
    # trustworthiness(original=D_orig, embedded distances=D_emb), both precomputed
    n = len(D_orig)
    rank_orig = np.argsort(np.argsort(D_orig + np.eye(n) * 1e9, axis=1), axis=1)
    nn_emb = np.argsort(D_emb + np.eye(n) * 1e9, axis=1)[:, :k]
    t = 0.0
    for i in range(n):
        for j in nn_emb[i]:
            r = rank_orig[i, j]
            if r >= k:
                t += r - k + 1
    return float(1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * t)


def trust(D, xy, k=10):
    d2 = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    return _trust_precomp(D, d2, k)


def cont(D, xy, k=10):
    d2 = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    return _trust_precomp(d2, D, k)


def classical_eigsharing(D):
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    ev = np.sort(np.linalg.eigvalsh(B))[::-1]
    pos = ev[ev > 0]
    cum = np.cumsum(pos) / pos.sum()
    dims = {p: int(np.searchsorted(cum, p) + 1) for p in (0.5, 0.8, 0.9)}
    return dims, float(cum[1])  # dims for 50/80/90% mass, and the 2-D share


def normalize(P):
    P = P - P.min(0)
    return P * (3200 / max(P.max(0))) + 200


def full_positions(xy_by_stem):
    """All-stem positions: given where embedded missions sit, place the rest at
    cited-neighbour centroids (mirrors mission_carpet_variants.embed_full)."""
    P = np.zeros((N, 2))
    gc = np.mean(list(xy_by_stem.values()), 0)
    for s in stems:
        if s in xy_by_stem:
            P[idx[s]] = xy_by_stem[s]
        else:
            nbrs = [xy_by_stem[r] for r in refs[s] if r in xy_by_stem] + \
                   [xy_by_stem[r] for r in stems if s in refs[r] and r in xy_by_stem]
            P[idx[s]] = np.mean(nbrs, 0) if nbrs else gc
    return P


def write_variant(name, xy, stems_order):
    by_stem = {s: xy[i] for i, s in enumerate(stems_order)}
    Pn = normalize(full_positions(by_stem))
    out = ROOT / f"futon6/data/mission-carpet-pos-{name}.json"
    json.dump({s: [round(float(Pn[idx[s]][0]), 1), round(float(Pn[idx[s]][1]), 1)] for s in stems},
              open(out, "w"))
    return out


def render_contours(variant):
    r = subprocess.run([sys.executable, str(HERE / "mission_efe_field.py"), variant],
                       capture_output=True, text=True, timeout=900)
    m = re.search(r"(\d+) contour segs", r.stdout + r.stderr)
    return int(m.group(1)) if m else None


def main():
    rng = np.random.default_rng(2026)
    D_real = cosine_dist(V)
    iu = np.triu_indices(M, 1)
    print(f"missions={N} embedded={M}")
    print(f"BGE cosine-distance over embedded pairs: mean={D_real[iu].mean():.3f} "
          f"std={D_real[iu].std():.3f} cv={D_real[iu].std()/D_real[iu].mean():.3f}")

    # 1-2. real layout: stress-1, trustworthiness, continuity
    xy_real, raw = run_mds(D_real)
    s1 = kruskal_stress1(D_real, xy_real)
    tw, cn = trust(D_real, xy_real), cont(D_real, xy_real)
    print(f"\nREAL embed MDS: raw-stress={raw:.1f}  KRUSKAL-STRESS-1={s1:.3f}  "
          f"trustworthiness(k=10)={tw:.3f}  continuity(k=10)={cn:.3f}")

    # 3. intrinsic dimensionality
    dims, share2 = classical_eigsharing(D_real)
    print(f"classical-MDS eigenspectrum: dims for 50/80/90% mass = "
          f"{dims[0.5]}/{dims[0.8]}/{dims[0.9]}  ·  2-D share = {share2:.3f}")

    # 4. restart stability (Procrustes disparity vs seed-7)
    disps = []
    for seed in (1, 2, 3, 4, 5):
        xy_s, _ = run_mds(D_real, seed=seed)
        _, _, d = procrustes(xy_real, xy_s)
        disps.append(d)
    print(f"restart stability: Procrustes disparity vs seed-7 over seeds 1-5 = "
          f"{', '.join(f'{d:.4f}' for d in disps)} (0=identical, 1=unrelated)")

    # 5. controls through the real terrain pipeline
    results = {}
    for i in range(1, 4):
        # shuffle control: same point cloud, mission labels permuted — mission
        # emb_stems[j] takes the position of emb_stems[perm[j]].
        perm = rng.permutation(M)
        by = {emb_stems[j]: xy_real[perm[j]] for j in range(M)}
        Pn = normalize(full_positions(by))
        json.dump({s: [round(float(Pn[idx[s]][0]), 1), round(float(Pn[idx[s]][1]), 1)] for s in stems},
                  open(ROOT / f"futon6/data/mission-carpet-pos-ctrl-shuf{i}.json", "w"))

        W = rng.standard_normal((M, V.shape[1]))
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        D_r = cosine_dist(W)
        xy_r, _ = run_mds(D_r)
        s1_r = kruskal_stress1(D_r, xy_r)
        write_variant(f"ctrl-rand{i}", xy_r, emb_stems)
        results[f"rand{i}-stress1"] = s1_r
        print(f"ctrl-rand{i}: cosd mean={D_r[iu].mean():.3f} std={D_r[iu].std():.3f} "
              f"stress-1={s1_r:.3f}")

    print("\nrendering controls + measured baselines through mission_efe_field.py ...")
    counts = {}
    for v in ("force", "embed",
              "ctrl-shuf1", "ctrl-shuf2", "ctrl-shuf3",
              "ctrl-rand1", "ctrl-rand2", "ctrl-rand3"):
        counts[v] = render_contours(v)
        print(f"  {v}: {counts[v]} contour segs")

    shufs = [counts[f"ctrl-shuf{i}"] for i in (1, 2, 3) if counts.get(f"ctrl-shuf{i}")]
    rands = [counts[f"ctrl-rand{i}"] for i in (1, 2, 3) if counts.get(f"ctrl-rand{i}")]
    print("\n==== VERDICT INPUTS ====")
    print(f"force baseline: {counts['force']}   embed: {counts['embed']}")
    print(f"shuffle controls (label): {shufs}")
    print(f"random-vector controls (geometry): {rands}")
    print("rule: random ~ embed  => richness is an MDS-on-cosine artifact;")
    print("      random << embed => the extra terrain needs real BGE structure.")


if __name__ == "__main__":
    main()
