#!/usr/bin/env python3
"""Structure embeddings for CLean proofs (+ a text baseline for contrast).

The demo's central claim (E-clean / EXP-3): the load-bearing signal for proof
similarity is the *compositional shape* (the comb of typed holes + the method
spine), not the prose. So we build TWO embeddings per proof and compare:

  - STRUCTURE  : a deterministic ~33-d feature vector over the CLean comb
                 (method bag, macro shape, hole/discharge type histograms,
                 comb-shape scalars). No model, fully explainable.
  - TEXT       : MiniLM-L6-v2 (384-d) over the proof prose — the naive baseline
                 a text-retrieval pipeline would use. Falls back to a hashed
                 bag-of-words if the model is unavailable offline.

Outputs (default to data/showcases/clean-demo/):
  structure-embeddings.npy   (N x 33, L2-normalized)
  text-embeddings.npy        (N x D,  L2-normalized)
  clean-embed.json           ids, titles, macros, feature breakdowns, and
                             cosine matrices + nearest-neighbor rankings for both.

Usage:
  futon6/.venv/bin/python scripts/clean_structure_embed.py \
      [--clean-dir holes/clean] [--out data/showcases/clean-demo] \
      [--apm-dir /home/joe/code/futon3c/data/apm-informal-proofs]
"""
import argparse
import glob
import json
import os
import re
import numpy as np
import edn_format as edn

# ---- the controlled vocabularies (must match clean-method-vocab.edn) ----
METHOD_VOCAB = [
    "construct-auxiliary-object", "reduce-to-known-result", "quotient-by-irrelevance",
    "local-to-global", "transport-along-symmetry", "argue-by-contradiction",
    "count-by-decomposition", "compute-invariant", "divisibility-or-parity",
    "induct-up-a-tower", "cover-and-estimate", "epsilon-of-room",
]
MACRO_VOCAB = [
    "construct-exploit-discharge", "count-invariant-obstruct", "cover-estimate",
    "contradiction-reduce", "induct-tower",
]
SATIETY = ["parse", "payoff", "canon", "bundling", "role"]
DISCHARGE_KIND = ["sorryProof", "queryAnswer", "ungroundedBinder"]


def kw(x):
    """edn Keyword/Symbol -> plain name string (drop ns and leading colon)."""
    s = str(x)
    if s.startswith(":"):
        s = s[1:]
    return s.split("/")[-1]


def load_clean(path):
    m = edn.loads(open(path).read())
    # edn_format returns an ImmutableDict keyed by Keyword
    d = {kw(k): v for k, v in dict(m).items()}
    return d


def boxes_of(d):
    out = []
    for b in d["boxes"]:
        bd = {kw(k): v for k, v in dict(b).items()}
        box = {
            "id": kw(bd["id"]),
            "method": kw(bd["method"]),
            "text": str(bd.get("text", "")),
            "consumes": [kw(c) for c in bd.get("consumes", [])],
            "produces": kw(bd["produces"]) if "produces" in bd else None,
        }
        if "hole" in bd:
            h = {kw(k): v for k, v in dict(bd["hole"]).items()}
            box["hole"] = {"satiety": kw(h.get("satiety")),
                           "discharge": kw(h.get("discharge")),
                           "wanted": str(h.get("wanted", ""))}
        if "discharges" in bd:
            dd = {kw(k): v for k, v in dict(bd["discharges"]).items()}
            box["discharges"] = {"to": kw(dd.get("to"))}
        out.append(box)
    return out


def wires_of(d):
    out = []
    for w in d["wires"]:
        wd = {kw(k): v for k, v in dict(w).items()}
        out.append({"from": kw(wd["from"]), "to": kw(wd["to"]), "carries": kw(wd["carries"])})
    return out


def longest_path(ids, wires):
    succ = {i: [] for i in ids}
    indeg = {i: 0 for i in ids}
    for w in wires:
        succ[w["from"]].append(w["to"])
        indeg[w["to"]] += 1
    # DAG longest path via topo DP
    depth = {i: 0 for i in ids}
    order, q = [], [i for i in ids if indeg[i] == 0]
    indeg2 = dict(indeg)
    while q:
        n = q.pop()
        order.append(n)
        for m in succ[n]:
            depth[m] = max(depth[m], depth[n] + 1)
            indeg2[m] -= 1
            if indeg2[m] == 0:
                q.append(m)
    return max(depth.values()) + 1 if depth else 0


def structure_vector(d):
    boxes = boxes_of(d)
    wires = wires_of(d)
    ids = [b["id"] for b in boxes]
    seq = [kw(x) for x in d["seq"]]
    shape = {kw(k): v for k, v in dict(d["shape"]).items()}
    macro = kw(shape["macro"])

    n = len(boxes)
    feats = {}

    # 1. method bag (normalized by spine length)
    bag = np.zeros(len(METHOD_VOCAB))
    for t in seq:
        if t in METHOD_VOCAB:
            bag[METHOD_VOCAB.index(t)] += 1
    if seq:
        bag /= len(seq)
    feats["method_bag"] = bag

    # 2. macro one-hot
    macro_v = np.zeros(len(MACRO_VOCAB))
    if macro in MACRO_VOCAB:
        macro_v[MACRO_VOCAB.index(macro)] = 1.0
    feats["macro"] = macro_v

    # 3. satiety histogram (over holes)
    sat = np.zeros(len(SATIETY))
    holes = [b["hole"] for b in boxes if "hole" in b]
    for h in holes:
        if h["satiety"] in SATIETY:
            sat[SATIETY.index(h["satiety"])] += 1
    if holes:
        sat /= len(holes)
    feats["satiety"] = sat

    # 4. discharge-kind histogram (over holes)
    dis = np.zeros(len(DISCHARGE_KIND))
    for h in holes:
        if h["discharge"] in DISCHARGE_KIND:
            dis[DISCHARGE_KIND.index(h["discharge"])] += 1
    if holes:
        dis /= len(holes)
    feats["discharge_kind"] = dis

    # 5. comb-shape scalars
    n_holes = len(holes)
    n_disch_known = sum(1 for b in boxes if "discharges" in b)
    fanout = {}
    indeg = {i: 0 for i in ids}
    for w in wires:
        fanout[w["from"]] = fanout.get(w["from"], 0) + 1
        indeg[w["to"]] += 1
    max_fanout = max(fanout.values()) if fanout else 0
    n_sources = sum(1 for i in ids if indeg[i] == 0)
    n_sinks = sum(1 for i in ids if fanout.get(i, 0) == 0)
    depth = longest_path(ids, wires)
    scal = np.array([
        n / 6.0,
        len(wires) / 6.0,
        n_holes / max(n, 1),
        n_disch_known / max(n, 1),
        max_fanout / 3.0,
        depth / max(n, 1),
        n_sources / max(n, 1),
        n_sinks / max(n, 1),
    ])
    feats["comb_scalars"] = scal

    # WIDEN-DYNAMIC-RANGE (todo #16, validated on mark6 breakdowns; apply on the next run):
    # the structure-sim was tight (mean 0.80) for two reasons, both fixable here —
    #   (a) comb_scalars use ad-hoc /6.0,/3.0 constants, not corpus z-scores, so they don't
    #       have unit variance and dominate the L2-cosine. FIX: collect RAW vectors in main(),
    #       z-normalize the comb_scalar columns ACROSS the corpus, THEN L2. (-> sim mean 0.35)
    #   (b) method_bag loses ORDER, but method-SEQUENCES are 56/58 distinct. FIX: add a
    #       method-BIGRAM block (consecutive (seq[i],seq[i+1]) pairs over a corpus bigram
    #       vocab built in main()). (-> sim mean 0.10, range -0.70..1.00)
    # Both together take structure-sim 0.80 -> 0.10 (far finer structural twins). Kept as a
    # documented recipe (not applied) because mark6's CLeans were lost with the box, so it
    # can't be re-run/validated end-to-end until the next live run.
    vec = np.concatenate([feats["method_bag"], feats["macro"], feats["satiety"],
                          feats["discharge_kind"], feats["comb_scalars"]])
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    breakdown = {
        "macro": macro,
        "methods": seq,
        "n_boxes": n, "n_wires": len(wires), "n_holes": n_holes,
        "n_discharges_known": n_disch_known, "max_fanout": max_fanout,
        "depth": depth, "n_sources": n_sources, "n_sinks": n_sinks,
        "satiety": {SATIETY[i]: float(sat[i]) for i in range(len(SATIETY)) if sat[i] > 0},
    }
    return vec, breakdown, boxes


def proof_text(pid, boxes, apm_dir):
    """Text-baseline input: prefer the real informal proof .md; else box texts."""
    cand = os.path.join(apm_dir, f"apm-{pid}.md")
    if os.path.exists(cand):
        raw = open(cand).read()
        # strip the provenance comment + code fences to keep it prose-ish
        raw = re.sub(r"<!--.*?-->", " ", raw, flags=re.S)
        raw = re.sub(r"```.*?```", " ", raw, flags=re.S)
        return raw, "apm-md"
    return " ".join(b["text"] for b in boxes), "box-text"


def text_embeddings(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs), "minilm-l6-v2"
    except Exception as e:  # offline / no cache — deterministic hashed BoW fallback
        print(f"[text] MiniLM unavailable ({e}); using hashed bag-of-words fallback")
        D = 512
        mat = np.zeros((len(texts), D))
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-zA-Z]+", t.lower()):
                if len(tok) < 3:
                    continue
                mat[i, hash(tok) % D] += 1.0
            nrm = np.linalg.norm(mat[i])
            if nrm > 0:
                mat[i] /= nrm
        return mat, "hashed-bow-512"


def cosine_matrix(M):
    return (M @ M.T)


def nn_ranking(ids, sim):
    out = {}
    for i, pid in enumerate(ids):
        order = sorted(range(len(ids)), key=lambda j: -sim[i, j])
        out[pid] = [{"id": ids[j], "sim": round(float(sim[i, j]), 4)}
                    for j in order if j != i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="holes/clean")
    ap.add_argument("--out", default="data/showcases/clean-demo")
    ap.add_argument("--apm-dir", default="/home/joe/code/futon3c/data/apm-informal-proofs")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.clean_dir, "*.clean.edn")))
    os.makedirs(args.out, exist_ok=True)

    ids, titles, macros, struct_rows, breakdowns, texts, text_srcs = [], [], [], [], [], [], []
    for f in files:
        d = load_clean(f)
        pid = kw(d["proof"])
        vec, bd, boxes = structure_vector(d)
        txt, src = proof_text(pid, boxes, args.apm_dir)
        ids.append(pid)
        titles.append(str(d.get("title", "")))
        macros.append(bd["macro"])
        struct_rows.append(vec)
        breakdowns.append(bd)
        texts.append(txt)
        text_srcs.append(src)

    S = np.vstack(struct_rows)
    T, text_model = text_embeddings(texts)

    np.save(os.path.join(args.out, "structure-embeddings.npy"), S)
    np.save(os.path.join(args.out, "text-embeddings.npy"), T)

    s_sim = cosine_matrix(S)
    t_sim = cosine_matrix(T)

    payload = {
        "ids": ids,
        "titles": titles,
        "macros": macros,
        "text_sources": text_srcs,
        "structure_dim": int(S.shape[1]),
        "text_model": text_model,
        "text_dim": int(T.shape[1]),
        "breakdowns": breakdowns,
        "structure_sim": [[round(float(x), 4) for x in row] for row in s_sim],
        "text_sim": [[round(float(x), 4) for x in row] for row in t_sim],
        "structure_nn": nn_ranking(ids, s_sim),
        "text_nn": nn_ranking(ids, t_sim),
    }
    with open(os.path.join(args.out, "clean-embed.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"embedded {len(ids)} proofs  |  structure {S.shape}  text {T.shape} ({text_model})")
    print("\nstructure nearest-neighbor (top-1 each):")
    for pid in ids:
        nn = payload["structure_nn"][pid][0]
        nn_macro = macros[ids.index(nn["id"])]
        print(f"  {pid:8s} ({macros[ids.index(pid)]:26s}) -> {nn['id']:8s} "
              f"({nn_macro:26s}) sim={nn['sim']}")
    print("\ntext nearest-neighbor (top-1 each):")
    for pid in ids:
        nn = payload["text_nn"][pid][0]
        print(f"  {pid:8s} -> {nn['id']:8s} sim={nn['sim']}")


if __name__ == "__main__":
    main()
