#!/usr/bin/env python3
"""Learn cascade posteriors and phylogeny overlays from recorded closure folds.

Recompute-from-scratch by design: running this repeatedly over the same
closure-folds.edn produces byte-identical outputs, so there is no double-count.
"""
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

ROOT = Path("/home/joe/code")
FUTON6 = ROOT / "futon6"
DEFAULT_CLOSURES = FUTON6 / "holes/closure-folds.edn"
DEFAULT_COMPUTED = FUTON6 / "data/pattern-phylogeny-edges.json"
DEFAULT_POSTERIORS = FUTON6 / "data/pattern_posteriors.grounded.json"
DEFAULT_LEARNED = FUTON6 / "data/pattern-phylogeny-learned.json"
DEFAULT_GAPS = FUTON6 / "data/cascade-coverage-gaps.edn"


def strip_comments(text):
    return "\n".join(line.split(";;", 1)[0] for line in text.splitlines())


def parse_closure_folds(path):
    """Parse the small EDN subset used by closure-folds.edn."""
    text = strip_comments(Path(path).read_text())
    maps = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    records = []
    for m in maps:
        rec = {}
        for key, value in re.findall(r":([\w\-/?.]+)\s+(\"(?:\\.|[^\"])*\"|true|false|\[[^\]]*\])", m, flags=re.S):
            if value == "true":
                rec[key] = True
            elif value == "false":
                rec[key] = False
            elif value.startswith("["):
                rec[key] = re.findall(r"\"((?:\\.|[^\"])*)\"", value)
            else:
                rec[key] = bytes(value[1:-1], "utf-8").decode("unicode_escape")
        records.append(rec)
    return records


def stem(pattern_id):
    return str(pattern_id).split("/")[-1]


def ordered_pair(a, b):
    return tuple(sorted((stem(a), stem(b))))


def load_computed(path):
    data = json.loads(Path(path).read_text())
    co = {}
    for a, b, w in data.get("co_app", []):
        co[ordered_pair(a, b)] = int(w)
    descent = {tuple(edge[:2]) for edge in data.get("descent", [])}
    descent_undirected = {tuple(sorted(edge[:2])) for edge in data.get("descent", [])}
    return {"co_app": co, "descent": descent, "descent_undirected": descent_undirected}


def learned_from(closures, computed):
    alpha = {}
    beta = {}
    co_overlay = {}
    descent_overlay = []
    gaps = []

    for rec in closures:
        if not rec.get("success"):
            continue
        scope = rec.get("scope")
        used = [stem(p) for p in rec.get("used", [])]

        for p in used:
            alpha[p] = alpha.get(p, 0) + 1
            beta.setdefault(p, 0)

        for a, b in combinations(sorted(set(used)), 2):
            pair = ordered_pair(a, b)
            if pair in computed["co_app"]:
                co_overlay[pair] = {
                    "weight": computed["co_app"][pair] + 1,
                    "origin": f"upvote:{scope}",
                }
            elif pair in computed["descent_undirected"]:
                descent_overlay.append([a, b, f"upvote:{scope}"])
            else:
                co_overlay[pair] = {"weight": 1, "origin": f"seed:{scope}"}

        missing = rec.get("missing") or rec.get("missing-pattern") or rec.get("missing_pattern")
        if missing:
            gaps.append({"scope": scope, "missing": missing, "at": rec.get("at") or "closure-folds.edn"})

    posteriors = {
        p: {
            "alpha": alpha[p],
            "beta": beta.get(p, 0),
            "mean": round(alpha[p] / (alpha[p] + beta.get(p, 0)), 6),
        }
        for p in sorted(alpha)
    }
    learned = {
        "co_app": [[a, b, row["weight"], row["origin"]] for (a, b), row in sorted(co_overlay.items())],
        "descent": sorted(descent_overlay),
        "source": str(DEFAULT_CLOSURES),
    }
    return posteriors, learned, gaps


def write_outputs(posteriors, learned, gaps, posterior_path, learned_path, gaps_path):
    Path(posterior_path).write_text(json.dumps(posteriors, indent=2, sort_keys=True) + "\n")
    Path(learned_path).write_text(json.dumps(learned, indent=2, sort_keys=True) + "\n")
    gap_rows = "\n".join(
        f'{{:scope "{g["scope"]}" :missing "{g["missing"]}" :at "{g["at"]}"}}'
        for g in gaps
    )
    Path(gaps_path).write_text("[" + (("\n " + gap_rows + "\n") if gap_rows else "") + "]\n")


def run(args):
    closures = parse_closure_folds(args.closures)
    computed = load_computed(args.computed)
    posteriors, learned, gaps = learned_from(closures, computed)
    write_outputs(posteriors, learned, gaps, args.posteriors, args.learned, args.gaps)
    print(f"closures={len(closures)} successful={sum(1 for r in closures if r.get('success'))}")
    print(f"grounded-posteriors={len(posteriors)} -> {args.posteriors}")
    print(f"learned-co-app={len(learned['co_app'])} learned-descent={len(learned['descent'])} -> {args.learned}")
    print(f"coverage-gaps={len(gaps)} -> {args.gaps}")
    if args.demo:
        print(json.dumps({"posteriors": posteriors, "learned": learned, "gaps": gaps}, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closures", default=str(DEFAULT_CLOSURES))
    ap.add_argument("--computed", default=str(DEFAULT_COMPUTED))
    ap.add_argument("--posteriors", default=str(DEFAULT_POSTERIORS))
    ap.add_argument("--learned", default=str(DEFAULT_LEARNED))
    ap.add_argument("--gaps", default=str(DEFAULT_GAPS))
    ap.add_argument("--demo", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
