#!/usr/bin/env python3
"""MINING RECOGNISER — distil the paid mining into a reproducible NNexus-style move recogniser.

The ▶build / ◀reach / ✎steer chips came from a paid GPU mining run (non-reproducible per session).
This learns a DETERMINISTIC recogniser FROM the mining output: each move-type gets a BASIN — an
embedding centroid + its top hotwords — over the turns the mining tagged.  A new turn is recognised
by its nearest basin.  It reproduces the previous tagging by construction and generalises for free
(no LLM at runtime).  M-points-de-fuite: the State apparatus distilled into a light reusable artifact.

  futon6/.venv/bin/python scripts/mining_recogniser.py   ->  data/c-vector/move-basins.json + validation
"""
import json, os, re
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FWD = os.path.join(HERE, "../data/meme-mine/joint-memes.openai.json")
BWD = os.path.join(HERE, "../data/c-vector/c-entries.openai.json")
OUT = os.path.join(HERE, "../data/c-vector/move-basins.json")
STOP = set(("the a an and or of to in is it that this for on with as we i you do so be at by are not "
            "can if then will would could let lets our your my me but just like into out up now what "
            "which when have has had was were they them their about from also more most some such "
            "one two yes no ok").split())


def examples():
    ex = []
    for r in json.load(open(FWD)):
        t = (r.get("ask") or "").strip()
        if len(t) > 15 and r.get("memes"):
            op = Counter(m.get("op") for m in r["memes"]).most_common(1)[0][0]
            ex.append((t, "build", op))
    for r in json.load(open(BWD)):
        p = r.get("provenance") or {}
        if r["flavour"] == "reach":
            t = (p.get("assistant_span") or "").strip()
            if len(t) > 15:
                ex.append((t, "reach", "satisfied"))
        elif r["flavour"] == "correction":
            t = (p.get("reply_span") or "").strip()
            if len(t) > 15:
                ex.append((t, "steer", "align"))
    return ex


def hotwords(texts, n=12):
    c = Counter()
    for t in texts:
        for w in set(re.findall(r"[a-z][a-z'-]{3,}", t.lower())):
            if w not in STOP:
                c[w] += 1
    return [w for w, _ in c.most_common(n)]


def main():
    ex = examples()
    texts = [e[0][:400] for e in ex]
    moves = [e[1] for e in ex]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    V = model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

    classes = sorted(set(moves))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ex))
    split = int(0.8 * len(ex))
    tr, te = idx[:split], idx[split:]

    def centroids(rows):
        c = {cl: V[[i for i in rows if moves[i] == cl]].mean(0) for cl in classes}
        return {cl: v / (np.linalg.norm(v) or 1) for cl, v in c.items()}

    cent = centroids(tr)
    C = np.array([cent[cl] for cl in classes])
    pred = [classes[int(np.argmax(V[i] @ C.T))] for i in te]
    acc = float(np.mean([pred[k] == moves[te[k]] for k in range(len(te))]))
    conf = defaultdict(Counter)
    for k in range(len(te)):
        conf[moves[te[k]]][pred[k]] += 1

    # final basins on ALL examples
    fcent = centroids(range(len(ex)))
    hw = {cl: hotwords([texts[i] for i in range(len(ex)) if moves[i] == cl]) for cl in classes}
    counts = {cl: int(sum(1 for m in moves if m == cl)) for cl in classes}
    json.dump({"classes": classes, "counts": counts,
               "centroids": {cl: v.tolist() for cl, v in fcent.items()},
               "hotwords": hw, "model": "all-MiniLM-L6-v2",
               "val_accuracy": round(acc, 3)},
              open(OUT, "w"))

    print(f"move recogniser: {len(ex)} mined examples -> basins per {classes}")
    print(f"  counts: {counts}")
    print(f"  held-out (20%) nearest-basin accuracy: {acc:.0%}")
    print("  confusion (rows = mined truth, cols = recognised):")
    hdr = "          " + "".join(f"{c:>8}" for c in classes)
    print(hdr)
    for c in classes:
        print(f"    {c:<6}" + "".join(f"{conf[c][p]:>8}" for p in classes))
    print("  basin hotwords (the NNexus terminology):")
    for c in classes:
        print(f"    {c:<6} {', '.join(hw[c][:8])}")
    print(f"  wrote {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
