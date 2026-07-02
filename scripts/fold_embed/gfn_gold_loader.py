"""Gold-corpus loader for the GFN seed (E-fold-embed-pipeline, open decision (a)).

Loads the 10 A-next gold triples (futon2/holes/labs/A-next-gold-corpus.md — the
canonical pin), builds the endpoint POOL (union of deduped :ref strings across
all 10 missions) and each mission's GOLD index set. Each distinct :ref string is
one discrete item — NO substrate-2 resolution in v0 (that contextualisation is
A-next's later job). For mission m, the other 9 missions' endpoints are the
hard negatives (sibling-mission near-misses, the B.2 idea).

EDN is parsed by shelling out to bb (cheshire), cached as JSON.
Torch-free; `--check` runs the deterministic self-check (loader + reward).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

LABS = Path("/home/joe/code/futon2/holes/labs")
DATA = Path("/home/joe/code/futon6/data/fold-embed-gfn")
CACHE = DATA / "gold-corpus.json"

# The canonical 10 (A-next-gold-corpus.md numbering).
MISSIONS = [
    "autoclock-in", "invariant-queue-unstuck", "a-sorry-enterprise",
    "agency-rebuild", "f6-ingest", "pattern-ingest", "patterns-done-right",
    "single-entry-point", "state-snapshot-witness", "stepper-calibration",
]

BETA = 6.0        # reward steepness: max/min ratio = e^BETA ~ 403x (>= the ~100x bar;
                  # the flat-C lesson — a 2.6x range failed to concentrate TB)
EPS = 1e-6


def reward_for_coverage(c: float) -> float:
    """The 0-sorry adapter: coverage 1.0 (all gold endpoints selected = all
    sorries discharged) is max reward; steep in between."""
    return EPS + math.exp(BETA * c)


def _edn_path(mission: str) -> Path:
    d = LABS / f"A-next-{mission}"
    hits = sorted(d.glob("*EMPIRICAL*.edn"))
    if not hits:
        raise FileNotFoundError(f"no EMPIRICAL edn under {d}")
    return hits[0]


def _edn_to_json(path: Path) -> dict:
    out = subprocess.run(
        ["bb", "-e",
         "(require '[cheshire.core :as json])"
         f'(println (json/generate-string (clojure.edn/read-string (slurp "{path}"))))'],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def load_corpus(refresh: bool = False) -> dict:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    corpus = {}
    for m in MISSIONS:
        edn = _edn_to_json(_edn_path(m))
        refs = []
        for ep in edn["endpoints"]:
            r = ep["ref"]
            if r not in refs:          # dedupe within mission
                refs.append(r)
        corpus[m] = {"id": edn["id"], "refs": refs,
                     "n-typed-holes": len(edn.get("typed-holes", []))}
    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(corpus, indent=1))
    return corpus


def build_pool(corpus: dict):
    """-> (pool: list[str], gold: dict[mission, sorted list of pool indices])."""
    pool = sorted({r for m in corpus.values() for r in m["refs"]})
    index = {r: i for i, r in enumerate(pool)}
    gold = {m: sorted(index[r] for r in v["refs"]) for m, v in corpus.items()}
    return pool, gold


def shared_refs(corpus: dict) -> dict:
    counts = {}
    for v in corpus.values():
        for r in v["refs"]:
            counts[r] = counts.get(r, 0) + 1
    return {r: n for r, n in counts.items() if n > 1}


def self_check() -> dict:
    corpus = load_corpus(refresh=True)
    pool, gold = build_pool(corpus)
    ks = {m: len(g) for m, g in gold.items()}
    ratio = reward_for_coverage(1.0) / reward_for_coverage(0.0)
    assert len(corpus) == 10, f"expected 10 missions, got {len(corpus)}"
    assert all(6 <= k <= 9 for k in ks.values()), f"gold sizes out of 6..9: {ks}"
    assert ratio >= 100.0, f"reward range too flat: {ratio:.1f}x"
    assert all(0 <= i < len(pool) for g in gold.values() for i in g)
    report = {"pool-size": len(pool), "gold-sizes": ks,
              "shared-refs": shared_refs(corpus),
              "reward-beta": BETA, "reward-range-x": round(ratio, 1)}
    return report


if __name__ == "__main__":
    if "--check" in sys.argv:
        rep = self_check()
        print(json.dumps(rep, indent=1))
        print("self-check PASS")
    else:
        c = load_corpus(refresh="--refresh" in sys.argv)
        pool, gold = build_pool(c)
        print(f"{len(c)} missions · pool {len(pool)} · "
              f"gold sizes {[len(g) for g in gold.values()]}")
