#!/usr/bin/env python3
"""Mine rollout moves for ABSTAINING missions via structural provenance.

Generalizes the one-off witness (E-mine-mission-transitions.md): for every mission
that has NO move in the canonical move-set (so its act-gate ΔG is nil → abstain),
mint ONE excursion-shaped phase-advance move ("fill the next hole") whose
score/prior is BORROWED from the mission's nearest STRUCTURAL neighbour that
already has moves (mission_structure_embed.py). Provenance is recorded; nothing
is fabricated beyond "advance your own derivation like your structural kin did."

Sim-only / overlay: writes data/diffsub-moves-mined.edn (canonical + mined). The
mined moves are for ΔG computation; they are NOT auto-promoted.

Metric (progress, not throughput): how many move-less missions now get a move, and
the provenance QUALITY (same-class rate, sim distribution). A low-quality provenance
is a weak guess (→ richer corpus / turns), not an island.

Usage: futon6/.venv/bin/python scripts/mission_mine_moves.py
"""
import json, re
import numpy as np

ROOT = "/home/joe/code"; F6 = f"{ROOT}/futon6"
EMB_DIR = f"{F6}/data/mission-structure-embed"
SCOPES = f"{F6}/data/diffsub-scopes.json"
MOVES = f"{F6}/data/diffsub-moves.edn"
OUT = f"{F6}/data/diffsub-moves-mined.edn"
STEMS_OUT = "/tmp/minted-stems.edn"
CANON_PHASES = ["head", "identify", "map", "derive", "argue", "verify", "instantiate", "document"]


def f(b, k):
    m = re.search(rf':{k} (".*?"|[^ }}]+)', b)
    return m.group(1).strip('"') if m else None


def main():
    payload = json.load(open(f"{EMB_DIR}/mission-embed.json"))
    stems = payload["stems"]
    meta = {stems[i]: payload["meta"][i] for i in range(len(stems))}
    S = np.load(f"{EMB_DIR}/structure-embeddings.npy")
    sidx = {s: i for i, s in enumerate(stems)}

    scopes = json.load(open(SCOPES))
    node = {}
    for s in scopes:
        m = s.get("mission")
        if m and m not in node and s.get("mission_node"):
            node[m] = s["mission_node"]

    raw = open(MOVES).read()
    blocks = re.findall(r'\{:move/id.*?\}', raw)
    have_moves, template = set(), {}
    for b in blocks:
        have = f(b, "have") or ""
        mm = re.match(r'.*-d/mission/([^"/]+)', have)
        if mm:
            stem = mm.group(1)
            have_moves.add(stem)
            template.setdefault(stem, b)

    have_idx = [sidx[s] for s in have_moves if s in sidx]
    moveless = [s for s in stems if s not in have_moves and s in node]

    mined, minted_stems, sims, class_hits = [], [], [], 0
    skipped = []
    for stem in moveless:
        i = sidx[stem]
        # nearest structural neighbour that already has moves
        best, bj = -2.0, None
        for j in have_idx:
            sim = float(np.dot(S[i], S[j]))
            if sim > best:
                best, bj = sim, j
        if bj is None:
            skipped.append((stem, "no-neighbour-with-moves")); continue
        nb = stems[bj]
        # the hole to fill = first canonical phase not present; else first detached phase
        present = set(meta[stem]["phases"])
        target = next((p for p in CANON_PHASES if p not in present), None) \
            or (sorted(meta[stem]["detached_phases"])[0] if meta[stem]["detached_phases"] else None)
        if target is None:
            skipped.append((stem, "no-open-derivation-phase")); continue
        tb = template[nb]
        score, prior, dg = f(tb, "score"), f(tb, "prior"), f(tb, "delta-g")
        cmatch = meta[stem]["class"] == meta[nb]["class"]
        class_hits += cmatch; sims.append(best)
        note = f"MINED structural: fill {target}; prior from {('same' if cmatch else 'diff')}-class neighbour M-{nb} (sim {best:.3f})"
        mined.append(f'  {{:move/id "{node[stem]}->{stem}/{target}" :move/class :close-hole'
                     f' :have "{node[stem]}" :want "{stem}/{target}" :advances-cap nil'
                     f' :score {score} :prior {prior} :delta-g {dg} :confidence :mined-structural'
                     f' :rank 999 :move/terminal? false :note "{note}"}}')
        minted_stems.append(stem)

    open(OUT, "w").write(raw.replace('\n ]}', '\n' + "\n".join(mined) + '\n ]}'))
    open(STEMS_OUT, "w").write("[" + " ".join(f'"{s}"' for s in minted_stems) + "]")

    print(f"move-less missions (with a mission node): {len(moveless)}")
    print(f"minted moves: {len(minted_stems)}   skipped: {len(skipped)} {dict(__import__('collections').Counter(r for _,r in skipped))}")
    if sims:
        print(f"provenance sim — median {np.median(sims):.3f}  min {min(sims):.3f}  max {max(sims):.3f}")
        print(f"same-class provenance: {class_hits}/{len(sims)} = {class_hits/len(sims):.2f}")
    print(f"wrote {OUT} ({len(blocks)} canonical + {len(mined)} mined) and {STEMS_OUT}")


if __name__ == "__main__":
    main()
