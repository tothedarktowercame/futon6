#!/usr/bin/env python3
"""Plot the 'improve as we run' accretion curves: a metric rising with corpus size.

Tier-2 move-grounding curve: harvest move-cues from the first-n PROOF GRAPHS (not papers;
one paper contributes ~15 of them on the citation head), feed
them to the strategy recognizer, measure proof-move grounding over a FIXED set of proof
windows. As n grows, more cues → more recognition → the curve rises (until the move
vocabulary saturates = convergence). Writes an SVG line chart.

  futon6/.venv/bin/python scripts/accretion_curves.py
"""
import glob
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import strategy_recognizer as sr  # noqa: E402
from iatc_lexicon_harvest import harvest  # noqa: E402
from iatc_move_reground import cluster_cues, score  # noqa: E402

OUT = os.path.join(ROOT, "data/showcases/mark6-accretion-curve.html")


def move_curve(graph_dir="data/iatc-argument-graphs/loop-run-70b",
               cand_dir="data/cand-neighborhood", ns=None):
    gd = graph_dir if os.path.isabs(graph_dir) else os.path.join(ROOT, graph_dir)
    cd = cand_dir if os.path.isabs(cand_dir) else os.path.join(ROOT, cand_dir)
    golden = sorted(g for g in glob.glob(os.path.join(gd, "*.edn")) if "rung2" not in g)
    windows = [json.load(open(f)).get("source-window", "")
               for f in glob.glob(os.path.join(cd, "*.candidate.json"))]
    if ns is None:   # log-spaced checkpoints up to the corpus size (the accretion sweep)
        N = len(golden)
        ns = sorted({0, 1, 2, 3, 5, 7} | {n for n in (10, 30, 100, 300, 1000) if n < N} | {N})
    vocab = sr.load_vocab(os.path.join(ROOT, "holes/clean/tactic-gesture-vocab.edn"))
    pts = []
    for k in ns:
        k = min(k, len(golden))
        if k == 0:
            pts.append((0, 0, score(vocab, windows)["proof-move-grounding"]))
            continue
        td = tempfile.mkdtemp()
        for g in golden[:k]:
            shutil.copy(g, td)
        lex, _, _ = harvest(td)
        cues = cluster_cues(lex)
        aug = {**vocab, "heuristic": {**vocab["heuristic"], "corpus-move": cues}}
        pts.append((k, len(cues), score(aug, windows)["proof-move-grounding"]))
        shutil.rmtree(td)
    return pts, len(windows)


def svg(pts, n_windows):
    xs = [p[0] for p in pts]
    ys = [p[2] for p in pts]
    W, H, pad = 520, 300, 44
    xmax, ymax = max(xs) or 1, max(ys) * 1.15 or 1
    def X(x): return pad + x / xmax * (W - 2 * pad)
    def Y(y): return H - pad - y / ymax * (H - 2 * pad)
    poly = " ".join(f"{X(x):.0f},{Y(y):.0f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{X(x):.0f}" cy="{Y(y):.0f}" r="4" fill="#0f766e"/>'
                   f'<text x="{X(x):.0f}" y="{Y(y) - 9:.0f}" font-size="10" text-anchor="middle">{y:.3f}</text>'
                   for x, y in zip(xs, ys))
    labs = "".join(f'<text x="{X(x):.0f}" y="{H - pad + 16:.0f}" font-size="10" text-anchor="middle">{x}</text>'
                   for x in xs)
    return (f'<!doctype html><meta charset=utf-8><title>accretion curve</title>'
            f'<body style="font:14px Georgia,serif;background:#f7f5ef;margin:24px">'
            f'<h2>"Improve as we run" — proof-move grounding vs #proof-graphs harvested</h2>'
            f'<p>fixed {n_windows} measure windows; cues accrete from more proof-graphs → grounding rises '
            f'(honest: lift is recognition/"thin", not faked verification)</p>'
            f'<svg width="{W}" height="{H}" style="background:#fff;border:1px solid #d9cdbd">'
            f'<polyline points="{poly}" fill="none" stroke="#0f766e" stroke-width="2"/>{dots}{labs}'
            f'<text x="{W//2}" y="{H-8}" font-size="11" text-anchor="middle">proof-graphs harvested (n)</text>'
            f'</svg></body>')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/iatc-argument-graphs/loop-run-70b")
    ap.add_argument("--candidates", default="data/cand-neighborhood")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    a = ap.parse_args()
    pts, nw = move_curve(a.graphs, a.candidates)
    print("proof-move grounding accretion curve (n PROOF-GRAPHS, #cues, grounding):")
    for k, c, g in pts:
        print(f"  n={k:2d}  cues={c:2d}  grounding={g:.3f}")
    rise = pts[-1][2] - pts[0][2]
    rising = all(pts[i][2] <= pts[i + 1][2] + 1e-9 for i in range(len(pts) - 1))
    print(f"  rise {pts[0][2]:.3f} → {pts[-1][2]:.3f}  (+{rise:.3f}); rising={rising}")

    # The curve IS the product of the sweep ("coverage is a bonus; the curve is
    # the product" — playbook §1), so it must land where RETRIEVE looks and must
    # not clobber a previous run's artifact. --run-dir was previously accepted
    # and ignored, and OUT is a mark6-hardcoded showcase path outside the
    # RETRIEVE manifest (E-superpod-hardening H16, 2026-08-06).
    targets = [OUT]
    if a.run_dir:
        out_dir = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(ROOT, a.run_dir)
        os.makedirs(out_dir, exist_ok=True)
        targets.append(os.path.join(out_dir, "accretion-curve.html"))
        import json as _json
        with open(os.path.join(out_dir, "accretion-curve.json"), "w") as fh:
            _json.dump({"run_id": a.run_id, "corpus_id": a.corpus_id,
                        "metric": "proof-move-grounding",
                        "points": [{"n_proof_graphs": k, "cues": c, "grounding": g} for k, c, g in pts],
                        "rise": round(rise, 6), "rising": rising}, fh, indent=1)
        print(f"wrote {os.path.join(a.run_dir, 'accretion-curve.json')}")
        try:
            sys.path.insert(0, os.path.join(ROOT, "scripts"))
            import metric_harness as mh
            for k, _c, g in pts:
                mh.emit_record(out_dir, run_id=a.run_id, corpus_id=a.corpus_id,
                               paper_id=f"(corpus n={k})", stage="S12",
                               metric="proof-move-grounding", axis="accretion",
                               value=round(g, 4), computable=True)
        except Exception as ee:
            print(f"  (metric emit skipped: {ee})")
    for t in targets:
        open(t, "w").write(svg(pts, nw))
        print(f"wrote {os.path.relpath(t, ROOT)}")


if __name__ == "__main__":
    main()
