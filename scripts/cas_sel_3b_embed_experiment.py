#!/usr/bin/env python3
"""CAS-SEL-3b experiment — does an embedding modality lift the Tier-0 hotword recall
ceiling and recover the 3 zero-overlap steps?

This is a DISCRIMINATING test (see E-informal-proof-checking-final-checklist §6):
- if a strong text model (bge-large) recovers the 3 zero-overlap steps → the ceiling
  was *model size*; ship the embedding modality for CAS-SEL-3b.
- if even bge-large does NOT recover them → the ceiling is *text-vs-structure*, i.e.
  those matches are structural not lexical/semantic-text (e.g. "z=z₀+mω₁+nω₂" →
  quotient-by-irrelevance) → evidence for the R-GCN / structure-first direction (§6).

Self-contained + deterministic given a fixed model. Runs on the dev box with a light
model (bge-small/MiniLM) or on the Linode/bigger box with bge-large (the spec's model,
which OOM'd on the dev box). Reuses cas_select.retrieve (hotword) for the union, and
audits embedding collapse via the cosine-to-mean-std metric (audit-graph-embeddings.py).

  # dev box (light model, also (re)builds the portable pattern snapshot):
  futon6/.venv/bin/python scripts/cas_sel_3b_embed_experiment.py --model BAAI/bge-small-en-v1.5
  # Linode/bigger box (the real test), CPU to avoid contending with vLLM:
  python scripts/cas_sel_3b_embed_experiment.py --model BAAI/bge-large-en-v1.5 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FIX = ROOT / "tests" / "fixtures" / "cas-select"
PIDS = ["a93J05", "a96J01", "b97J01", "a96J04"]
QINSTR = "Represent this sentence for searching relevant passages: "
# portable snapshot of the 39-pattern corpus, so the experiment runs where futon3
# (cas_select's pattern source) is absent — e.g. a fresh Linode box. In resources/
# (tracked) not data/ (gitignored), so it ships via git.
SNAPSHOT = ROOT / "resources" / "cas-select" / "pattern-texts.json"


def load_fixture(pid):
    steps = json.loads((FIX / f"{pid}.steps.json").read_text())["steps"]
    oracle = {r["step"]: r for r in json.loads((FIX / f"{pid}.oracle.json").read_text())["matches"]}
    return steps, oracle


def build_corpus():
    """Return ({name: {title,conclusion,hotwords,however}}, hotword_retrieve_fn).
    Prefer the live cas_select patterns (dev box); fall back to the committed snapshot
    (Linode). Writes/refreshes the snapshot when live patterns are available."""
    try:
        import cas_select as cs
        patterns = cs.load_patterns()
        corpus = {
            n: {
                "title": p.title,
                "conclusion": p.conclusion,
                "hotwords": list(p.hotwords),
                "however": getattr(p, "however", "") or "",
            }
            for n, p in patterns.items()
        }
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n")

        def hot_retrieve(text, k):
            return {r["pattern"] for r in cs.retrieve(text, patterns, k=k)}

        return corpus, hot_retrieve
    except Exception as e:  # futon3 absent → snapshot + a local hotword scorer
        if not SNAPSHOT.exists():
            raise SystemExit(f"no live patterns ({e}) and no snapshot at {SNAPSHOT}")
        corpus = json.loads(SNAPSHOT.read_text())

        def tok(s):
            return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if len(w) >= 3}

        def hot_retrieve(text, k):
            q = tok(text)
            scored = sorted(
                corpus,
                key=lambda n: -len(q & (set(w.lower() for w in corpus[n]["hotwords"]) | tok(corpus[n]["title"]))),
            )
            return set(scored[:k])

        return corpus, hot_retrieve


def pattern_text(c, mode):
    if mode == "title+conclusion":
        return f"{c['title']}. {c['conclusion']}"
    if mode == "full":
        return f"{c['title']}. {c['conclusion']} {c['however']} {' '.join(c['hotwords'])}"
    return f"{c['title']}. {c['conclusion']} {' '.join(c['hotwords'])}"  # title+conclusion+hotwords


def collapse_audit(P):
    """cosine-to-mean std (audit-graph-embeddings.py): <0.01 collapse, <0.05 mild."""
    m = P.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    std = float((P @ m).std())
    return {"cosine_to_mean_std": round(std, 4),
            "verdict": "COLLAPSE" if std < 0.01 else ("mild" if std < 0.05 else "ok")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--repr", dest="repr_mode",
                    choices=["title+conclusion+hotwords", "title+conclusion", "full"],
                    default="title+conclusion+hotwords")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--context", type=int, default=0,
                    help="EXP-3b: step-context window radius for the EMBEDDING query. "
                         "0 = isolated step (EXP-3 behaviour); 1 = step + its ±1 proof "
                         "neighbours; etc. Tests whether proof-flow context (not model size) "
                         "recovers the zero-overlap steps. Hotword baseline stays isolated.")
    ap.add_argument("--device", default=None, help="cpu | cuda | (default auto)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    corpus, hot_retrieve = build_corpus()
    names = sorted(corpus)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(a.model, device=a.device)
    use_prefix = "bge" in a.model.lower()
    P = np.asarray(model.encode([pattern_text(corpus[n], a.repr_mode) for n in names],
                                normalize_embeddings=True))

    def embed_top(text, k):
        q = np.asarray(model.encode([(QINSTR + text) if use_prefix else text],
                                    normalize_embeddings=True))[0]
        return {names[i] for i in np.argsort(-(P @ q))[:k]}

    hot = emb = union = total = 0
    zero, recovered, per = [], [], []
    for pid in PIDS:
        steps, oracle = load_fixture(pid)
        texts = [s["text"] for s in steps]
        for i, s in enumerate(steps):
            want = oracle[s["id"]]["pattern"]
            total += 1
            # EXP-3b: the embedding query carries the step's proof-flow context
            # (its ±context neighbours, in proof order). The hotword baseline and the
            # zero-overlap definition below stay on the isolated step text — so any
            # recovery is attributable to context, not a changed baseline.
            lo, hi = max(0, i - a.context), min(len(texts), i + a.context + 1)
            qtext = " ".join(texts[lo:hi]) if a.context else s["text"]
            h = hot_retrieve(s["text"], a.k)
            e = embed_top(qtext, a.k)
            ih, ie, iu = want in h, want in e, want in (h | e)
            hot += ih; emb += ie; union += iu
            full = hot_retrieve(s["text"], len(names))
            if want not in full:
                zero.append(f"{pid}/{s['id']}")
                if ie:
                    recovered.append(f"{pid}/{s['id']}")
            per.append({"step": f"{pid}/{s['id']}", "want": want, "hot": ih, "emb": ie, "union": iu})

    ceiling = total - len(zero)
    result = {
        "model": a.model, "repr": a.repr_mode, "k": a.k, "context": a.context, "device": a.device or "auto",
        "recall": {"hotword": f"{hot}/{total}", "embed": f"{emb}/{total}", "union": f"{union}/{total}"},
        "hotword_full_pool_ceiling": f"{ceiling}/{total}",
        "zero_overlap_steps": zero,
        "recovered_by_embed": recovered,
        "collapse_audit": collapse_audit(P),
        "acceptance": {
            "union_above_ceiling": union > ceiling,
            "all_zero_overlap_recovered": bool(zero) and len(recovered) == len(zero),
        },
        "per_step": per,
    }
    if a.out:
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_step"}, indent=2))
    print(f"\n== {a.model} [{a.repr_mode}] ==")
    print(f"  hotword {hot}/{total} | embed {emb}/{total} | UNION {union}/{total}  (full-pool ceiling {ceiling}/{total})")
    print(f"  zero-overlap recovered by embed: {recovered or 'NONE'}  of {zero}")
    print(f"  collapse audit: {result['collapse_audit']}")
    if result["acceptance"]["all_zero_overlap_recovered"]:
        print("  VERDICT: embedding recovers the structural steps -> model-size was the issue; ship embedding CAS-SEL-3b.")
    else:
        print("  VERDICT: embedding does NOT recover them -> ceiling is text-vs-structure -> evidence for R-GCN / structure-first (§6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
