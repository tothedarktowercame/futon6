#!/usr/bin/env python3
"""Structure embeddings for MISSIONS (+ a text baseline for contrast).

The port of clean_structure_embed.py (proofs) to missions, for criterion (2) of
M-aif-wiring R16: give the rollout's move-mining a *structural* notion of "a
mission like this one," so a mined move borrows its provenance/prior from a
structurally-similar mission instead of the weak nearest-TEXT neighbour
(cos 0.31, below E-vwm's 0.5 stall line).

Two embeddings per mission, compared:
  - STRUCTURE : a deterministic feature vector over the mission's scope-tree
                (phase presence/detached, class, binder histogram, shape
                scalars, phase-transition bigrams, applied-pattern presence).
                No model, fully explainable. The analog of the proof comb.
  - TEXT      : the live MiniLM mission vectors (futon3a) over the prose — the
                naive baseline the current co-embedding uses.

Discrimination metric (the analog of the proof side's same-macro match): does a
mission's top-1 nearest neighbour share its CLASS (mess/pipeline/alive/stub)?

Inputs (all live):
  futon6/data/diffsub-scopes.json                       per-scope {mission,binder,det,scope_id,...}
  futon6/data/mission-wholeness.edn                     :mission -> :class
  futon6/data/mission-pattern-scopes.edn                per-mission :applied [patterns]
  futon3a/resources/notions/minilm_mission_embeddings.json   384-d text vectors
  futon6/data/diffsub-moves.edn                         which missions already have rollout moves

Usage: futon6/.venv/bin/python scripts/mission_structure_embed.py [--out DIR]
"""
import argparse, glob, json, os, re
from collections import Counter, defaultdict
import numpy as np

ROOT = "/home/joe/code"
F6 = f"{ROOT}/futon6"
SCOPES = f"{F6}/data/diffsub-scopes.json"
WHOLE = f"{F6}/data/mission-wholeness.edn"
PSCOPES = f"{F6}/data/mission-pattern-scopes.edn"
MINILM = f"{ROOT}/futon3a/resources/notions/minilm_mission_embeddings.json"
MOVES = f"{F6}/data/diffsub-moves.edn"

CANON_PHASES = ["head", "identify", "map", "derive", "argue", "verify", "instantiate", "document"]
CLASSES = ["mess", "pipeline", "alive", "stub", "neutral"]
BINDERS = ["eightfold-phase", "loose-section", "map-item", "capability-scope",
           "pattern", "psr", "pur", "mission-scope-in", "mission-scope-out",
           "relates-to", "source-material"]
FRONTIER = {"capability-scope", "pattern", "psr", "pur"}


def canon_phase(scope_id):
    return scope_id.rsplit("/", 1)[-1].split("--")[0]


def mission_classes():
    text = open(WHOLE).read()
    return {m: c for m, c in re.findall(r':mission "M-([^"]+)" :class :(\w+)', text)}


def applied_patterns():
    """Per-mission :applied list, parsed from the EDN (best-effort, regex-scoped per entry)."""
    text = open(PSCOPES).read()
    out = {}
    # each entry: :mission "M-NAME" ... :applied [ ... ]  (applied may follow mission within the map)
    for m in re.finditer(r':mission "M-([^"]+)"(.*?)(?=:mission "M-|$)', text, flags=re.S):
        stem, body = m.group(1), m.group(2)
        am = re.search(r':applied \[(.*?)\]', body, flags=re.S)
        pats = re.findall(r'"([^"]+)"', am.group(1)) if am else []
        out[stem] = pats
    return out


def build():
    scopes = json.load(open(SCOPES))
    cls = mission_classes()
    applied = applied_patterns()
    by_m = defaultdict(list)
    for s in scopes:
        if s.get("mission"):
            by_m[s["mission"]].append(s)

    # pattern vocab: applied in >=2 missions (df>=2), like the proof bigram block
    pat_df = Counter(p for stem in by_m for p in applied.get(stem, []))
    pat_vocab = sorted(p for p, d in pat_df.items() if d >= 2)
    pidx = {p: i for i, p in enumerate(pat_vocab)}

    # phase-transition bigram vocab over detached phases in canonical order
    def det_phase_seq(scs):
        present = {canon_phase(s["scope_id"]) for s in scs
                   if s.get("binder") == "eightfold-phase" and s.get("det")}
        return [p for p in CANON_PHASES if p in present]
    bigrams = sorted({(a, b) for scs in by_m.values()
                      for a, b in zip(det_phase_seq(scs), det_phase_seq(scs)[1:])})
    bidx = {g: i for i, g in enumerate(bigrams)}

    stems, rows, meta = [], [], []
    for stem in sorted(by_m):
        scs = by_m[stem]
        n = len(scs)
        present = {canon_phase(s["scope_id"]) for s in scs if s.get("binder") == "eightfold-phase"}
        det_present = {canon_phase(s["scope_id"]) for s in scs
                       if s.get("binder") == "eightfold-phase" and s.get("det")}
        binder_ct = Counter(s.get("binder") for s in scs)
        n_det = sum(1 for s in scs if s.get("det"))
        n_frontier = sum(v for b, v in binder_ct.items() if b in FRONTIER)
        phase_idx = [CANON_PHASES.index(p) for p in present if p in CANON_PHASES]
        span = (max(phase_idx) - min(phase_idx)) / 7.0 if phase_idx else 0.0

        f_phase = np.array([1.0 if p in present else 0.0 for p in CANON_PHASES])
        f_detph = np.array([1.0 if p in det_present else 0.0 for p in CANON_PHASES])
        c = cls.get(stem, "neutral")
        f_class = np.array([1.0 if c == k else 0.0 for k in CLASSES])
        f_binder = np.array([binder_ct.get(b, 0) / max(n, 1) for b in BINDERS])
        f_scalar = np.array([
            n / 12.0, n_det / max(n, 1), n_frontier / max(n, 1),
            binder_ct.get("capability-scope", 0) / max(n, 1),
            (binder_ct.get("psr", 0) + binder_ct.get("pur", 0)) / max(n, 1),
            binder_ct.get("map-item", 0) / max(n, 1), span,
        ])
        f_bi = np.zeros(len(bigrams))
        for a, b in zip(det_phase_seq(scs), det_phase_seq(scs)[1:]):
            f_bi[bidx[(a, b)]] = 1.0
        f_pat = np.zeros(len(pat_vocab))
        for p in applied.get(stem, []):
            if p in pidx:
                f_pat[pidx[p]] = 1.0

        vec = np.concatenate([f_phase, f_detph, f_class, f_binder, f_scalar, f_bi, f_pat])
        stems.append(stem)
        rows.append(vec)
        meta.append({"class": c, "n_scopes": n, "n_detached": n_det,
                     "phases": sorted(present), "detached_phases": sorted(det_present)})

    S = np.vstack(rows)
    # WIDEN-DYNAMIC-RANGE (mirror clean_structure_embed): z-normalize columns across the
    # corpus so no feature dominates, then L2 per row.
    sd = S.std(axis=0); sd[sd == 0] = 1.0
    S = (S - S.mean(axis=0)) / sd
    rn = np.linalg.norm(S, axis=1, keepdims=True); rn[rn == 0] = 1.0
    S = S / rn
    return stems, S, meta, {"pat_vocab": len(pat_vocab), "bigrams": len(bigrams),
                            "struct_dim": int(S.shape[1])}


def text_matrix(stems):
    emb = {(e["basename"][2:] if e["basename"].startswith("M-") else e["basename"]): e["vector"]
           for e in json.load(open(MINILM))}
    T, have = [], []
    for s in stems:
        if s in emb:
            v = np.asarray(emb[s], dtype=np.float64)
            T.append(v / (np.linalg.norm(v) or 1.0)); have.append(True)
        else:
            T.append(None); have.append(False)
    return T, have


def move_missions():
    text = open(MOVES).read()
    return {m for m in re.findall(r'-d/mission/([^"/]+)', text)}


def nn(stems, vecs, i, mask=None):
    best, bi = -2.0, None
    for j in range(len(stems)):
        if j == i or vecs[j] is None or (mask is not None and not mask[j]):
            continue
        sim = float(np.dot(vecs[i], vecs[j]))
        if sim > best:
            best, bi = sim, j
    return bi, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{F6}/data/mission-structure-embed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stems, S, meta, dims = build()
    Slist = [S[i] for i in range(len(stems))]
    T, have_text = text_matrix(stems)
    have_moves = move_missions()
    cls = {stems[i]: meta[i]["class"] for i in range(len(stems))}

    # discrimination: top-1 NN shares CLASS? (structure vs text)
    def class_match(vecs):
        ok = tot = 0
        for i in range(len(stems)):
            if vecs[i] is None:
                continue
            j, _ = nn(stems, vecs, i)
            if j is None:
                continue
            tot += 1
            ok += (cls[stems[i]] == cls[stems[j]])
        return ok / tot if tot else 0.0, tot

    sm, sn = class_match(Slist)
    tm, tn = class_match(T)
    s_sims = [nn(stems, Slist, i)[1] for i in range(len(stems))]
    t_sims = [nn(stems, T, i)[1] for i in range(len(stems)) if T[i] is not None]
    print(f"missions={len(stems)}  struct_dim={dims['struct_dim']} "
          f"(pat-block={dims['pat_vocab']}, bigrams={dims['bigrams']})  text=384")
    print(f"top-1 NN sim — STRUCTURE median {np.median(s_sims):.3f}  TEXT median {np.median(t_sims):.3f}")
    print(f"top-1 NN shares CLASS — STRUCTURE {sm:.2f} (n={sn})  vs  TEXT {tm:.2f} (n={tn})")

    # witness payoff: provenance for an abstaining target, restricted to missions WITH moves
    payload_nn = {}
    for tgt in ["canon-fingerprint-store", "bayesian-structure-learning"]:
        if tgt not in stems:
            continue
        i = stems.index(tgt)
        mask = [stems[j] in have_moves for j in range(len(stems))]
        sj, ss = nn(stems, Slist, i, mask)
        tj, ts = nn(stems, T, i, mask) if T[i] is not None else (None, 0.0)
        rec = {"target_class": cls[tgt],
               "structural_provenance": {"mission": stems[sj], "sim": round(ss, 3),
                                         "class": cls[stems[sj]], "class_match": cls[tgt] == cls[stems[sj]]} if sj else None,
               "text_provenance": {"mission": stems[tj], "sim": round(ts, 3),
                                   "class": cls[stems[tj]], "class_match": cls[tgt] == cls[stems[tj]]} if tj else None}
        payload_nn[tgt] = rec
        print(f"\n[{tgt}] class={cls[tgt]} — provenance among missions-with-moves:")
        print(f"   STRUCTURAL -> {rec['structural_provenance']}")
        print(f"   TEXT       -> {rec['text_provenance']}")

    np.save(os.path.join(args.out, "structure-embeddings.npy"), S)
    json.dump({"stems": stems, "meta": meta, "dims": dims,
               "class_match": {"structure": sm, "text": tm},
               "witness_provenance": payload_nn},
              open(os.path.join(args.out, "mission-embed.json"), "w"), indent=2)
    print(f"\nwrote {args.out}/structure-embeddings.npy + mission-embed.json")


if __name__ == "__main__":
    main()
