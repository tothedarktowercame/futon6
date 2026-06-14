#!/usr/bin/env python3
"""Read-only EXPLORE proposer for grounded-learning Build 3.

Ranks detached holes by expected information gain over the pattern and edge
units their current cascade would exercise. This does not modify the star-map
scheduler; it only emits a curriculum proposal artifact.
"""
import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

ROOT = Path("/home/joe/code")
FUTON6 = ROOT / "futon6"
FUTON3A = ROOT / "futon3a"
DEFAULT_SCOPES = FUTON6 / "data/diffsub-scopes.json"
DEFAULT_POSTERIORS = FUTON6 / "data/pattern_posteriors.grounded.json"
DEFAULT_EDGES = FUTON6 / "data/pattern-phylogeny-edges.json"
DEFAULT_LEARNED = FUTON6 / "data/pattern-phylogeny-learned.json"
DEFAULT_GAPS = FUTON6 / "data/cascade-coverage-gaps.edn"
DEFAULT_OUT = FUTON6 / "data/curriculum-proposals.edn"
CASCADE_PATH = FUTON3A / "holes/labs/M-memes-arrows/cascade_construct.py"


def load_cascade_module():
    spec = importlib.util.spec_from_file_location("cascade_construct", CASCADE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stem(pattern_id):
    return str(pattern_id).split("/")[-1]


def ordered_pair(a, b):
    return tuple(sorted((stem(a), stem(b))))


def digamma(x):
    """Dependency-free digamma approximation, accurate enough for small EFE ranking."""
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return result + math.log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))


def bernoulli_entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def beta_bernoulli_info(alpha_count, beta_count):
    """Expected KL from one Bernoulli observation under Beta(1+alpha, 1+beta)."""
    a = 1.0 + float(alpha_count)
    b = 1.0 + float(beta_count)
    total = a + b
    p = a / total
    expected_conditional_entropy = -(
        (a / total) * (digamma(a + 1.0) - digamma(total + 1.0))
        + (b / total) * (digamma(b + 1.0) - digamma(total + 1.0))
    )
    return max(0.0, bernoulli_entropy(p) - expected_conditional_entropy), p, total


def load_posteriors(path):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def load_edges(computed_path, learned_path):
    computed = json.loads(Path(computed_path).read_text()) if Path(computed_path).exists() else {}
    co = {}
    descent = set()
    for a, b, w in computed.get("co_app", []):
        co[ordered_pair(a, b)] = int(w)
    for edge in computed.get("descent", []):
        if len(edge) >= 2:
            descent.add(tuple(edge[:2]))
    if Path(learned_path).exists():
        learned = json.loads(Path(learned_path).read_text())
        for edge in learned.get("descent", []):
            if len(edge) >= 2:
                descent.add(tuple(edge[:2]))
        for edge in learned.get("co_app", []):
            if len(edge) >= 3:
                a, b, w = edge[:3]
                co[ordered_pair(a, b)] = int(w)
    return co, descent


def load_gap_scopes(path):
    if not Path(path).exists():
        return set()
    text = Path(path).read_text()
    return set(re.findall(r':scope\s+"([^"]+)"', text))


def scope_query(scope):
    parts = [
        scope.get("scope_id", ""),
        scope.get("mission", ""),
        scope.get("passage", ""),
        scope.get("capability") or "",
        " ".join(scope.get("concepts") or []),
    ]
    return " ".join(str(p) for p in parts if p)


def unit_stats_for_pattern(pattern_stem, posteriors):
    row = posteriors.get(pattern_stem, {})
    return int(row.get("alpha", 0)), int(row.get("beta", 0))


def score_scope(scope, cascade, posteriors, co_edges, descent_edges, gap_scopes):
    selected = [stem(pid) for _, pid, _ in cascade["trajectory"]]
    non_phylogeny = [stem(p) for p in cascade.get("non-phylogeny", [])]
    pattern_units = sorted(set(selected))
    gap_units = sorted(set(non_phylogeny))
    pair_units = [ordered_pair(a, b) for i, a in enumerate(selected) for b in selected[i + 1:]]

    info = 0.0
    ps = []
    counts = []
    why = set()

    for p in pattern_units:
        alpha, beta = unit_stats_for_pattern(p, posteriors)
        gain, pred, total = beta_bernoulli_info(alpha, beta)
        info += gain
        ps.append(pred)
        counts.append(total)
        if alpha + beta == 0:
            why.add("thin-pattern")

    for p in gap_units:
        gain, pred, total = beta_bernoulli_info(0, 0)
        info += gain
        ps.append(pred)
        counts.append(total)
        why.add("gap-pattern")

    edge_descriptions = []
    for a, b in sorted(set(pair_units)):
        has_descent = (a, b) in descent_edges or (b, a) in descent_edges
        weight = co_edges.get((a, b), 0)
        alpha = weight if weight else 0
        beta = 0
        gain, pred, total = beta_bernoulli_info(alpha, beta)
        if not weight and not has_descent:
            gain += math.log(2.0)
            why.add("untested-edge")
        else:
            why.add("edge-contraction")
        info += gain
        ps.append(pred)
        counts.append(total)
        edge_descriptions.append(f"{a}<->{b}")

    if scope.get("scope_id") in gap_scopes:
        gain, pred, total = beta_bernoulli_info(0, 0)
        info += gain
        ps.append(pred)
        counts.append(total)
        why.add("known-coverage-gap")

    mean_p = sum(ps) / len(ps) if ps else 0.5
    mean_count = sum(counts) / len(counts) if counts else 0.0
    s_size = len(pattern_units) + len(gap_units) + len(set(pair_units))
    return {
        "hole": scope.get("scope_id"),
        "mission": scope.get("mission"),
        "info_gain": round(info, 6),
        # info-per-unit exposes the breadth-vs-density confound: while every unit sits at the
        # Beta(1,1) floor (no recorded outcomes yet), info_gain ~= 0.5*S_size and this ratio is
        # ~constant — i.e. the ranking is cascade-breadth, not learning-density. As real outcomes
        # spread the posteriors, this ratio differentiates and becomes the honest EOC signal.
        "info_per_unit": round(info / s_size, 6) if s_size else 0.0,
        "why": sorted(why),
        "S_size": s_size,
        "patterns": pattern_units,
        "coverage_candidates": gap_units,
        "edges": edge_descriptions,
        "predictive_mean": round(mean_p, 3),
        "mean_pseudocount": round(mean_count, 3),
    }


def detached_scopes(path):
    rows = json.loads(Path(path).read_text())
    return [row for row in rows if row.get("state") == "detached"]


def propose(args):
    cascade_mod = load_cascade_module()
    posteriors = load_posteriors(args.posteriors)
    co_edges, descent_edges = load_edges(args.edges, args.learned)
    gap_scopes = load_gap_scopes(args.gaps)
    scopes = detached_scopes(args.scopes)
    rows = []
    for scope in scopes[: args.max_candidates if args.max_candidates else None]:
        query = scope_query(scope)
        cascade = cascade_mod.construct_cascade(query, epsilon=args.epsilon, pool=args.pool)
        rows.append(score_scope(scope, cascade, posteriors, co_edges, descent_edges, gap_scopes))
    rows.sort(key=lambda r: (-r["info_gain"], r["hole"]))
    return rows


def edn_string(s):
    return json.dumps(str(s))


def edn_value(v):
    if isinstance(v, str):
        return edn_string(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + " ".join(edn_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + " ".join(f":{k.replace('_', '-')} {edn_value(val)}" for k, val in v.items()) + "}"
    if v is None:
        return "nil"
    return edn_string(v)


def write_edn(path, rows):
    Path(path).write_text("[\n" + "\n".join(" " + edn_value(row) for row in rows) + "\n]\n")


def eoc_self_test(rows, top_n=10):
    top = rows[:top_n]
    if not top:
        return {"pass": False, "reason": "no proposals"}
    means = [r["predictive_mean"] for r in top]
    counts = [r["mean_pseudocount"] for r in top]
    midband = [p for p in means if 0.35 <= p <= 0.75]
    all_certain = all(p <= 0.1 or p >= 0.9 for p in means)
    all_unsupported = all(c <= 2.0 and abs(p - 0.5) < 0.01 for p, c in zip(means, counts))
    return {
        "pass": bool(midband) and not all_certain and not all_unsupported,
        "top_n": len(top),
        "midband": len(midband),
        "all_certain": all_certain,
        "all_unsupported": all_unsupported,
        "predictive_mean_range": [round(min(means), 3), round(max(means), 3)],
        "mean_pseudocount_range": [round(min(counts), 3), round(max(counts), 3)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", default=str(DEFAULT_SCOPES))
    ap.add_argument("--posteriors", default=str(DEFAULT_POSTERIORS))
    ap.add_argument("--edges", default=str(DEFAULT_EDGES))
    ap.add_argument("--learned", default=str(DEFAULT_LEARNED))
    ap.add_argument("--gaps", default=str(DEFAULT_GAPS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    rows = propose(args)
    write_edn(args.out, rows)
    test = eoc_self_test(rows)
    print(f"wrote {args.out} proposals={len(rows)}")
    print(f"EOC self-test: {json.dumps(test, sort_keys=True)}")
    if args.demo:
        print("top-10:")
        for i, row in enumerate(rows[:10], 1):
            print(f"{i}. {row['hole']} info={row['info_gain']} info/u={row['info_per_unit']} S={row['S_size']} why={','.join(row['why'])} p={row['predictive_mean']} n={row['mean_pseudocount']}")


if __name__ == "__main__":
    main()
