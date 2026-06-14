#!/usr/bin/env python3
"""Content-free probe over proof SHAPES (Joe, spoken 2026-06-11).

The marked-up full proofs give each proof a shape: the ordered sequence of
scope types (bind/let, constrain/relation, wire/consequential, ...) with no
content at all. Two questions, both askable tonight without a GPU:

1. PREDICTABILITY PER TYPE — masked-token prediction, leave-one-proof-out:
   mask each position, predict its type from (prev, next) using pooled
   bigram statistics from the other nine proofs. A type that is easy to
   predict from context is grammatical filler; a type that is hard to
   predict carries information. Joe's visual claim: section-body-grade
   scopes are not predictive — here is the number.

2. SHAPE vs CORRECTNESS — the comparison-with-official-solutions scorecard
   labels the ten proofs (correct / incomplete / wrong). n=10 is anecdote,
   not statistics, but the per-class shape profiles are the first entry in
   the proofs-with-outcomes corpus that E-anatomy-of-a-proof's F5 condition
   demands before scope metrics may claim validity.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proof_tex_audit as pta  # noqa: E402
import apm_proof_audit as apa  # noqa: E402

LABELS = {
    1: "wrong", 7: "wrong",
    4: "incomplete", 5: "incomplete", 6: "incomplete",
    2: "correct", 3: "correct", 8: "correct", 9: "correct", 10: "correct",
}

OUT = Path(__file__).resolve().parent.parent / "data" / "proof-shape-probe.json"


def shape_sequence(result: dict) -> list[str]:
    """Ordered scope-type sequence = the proof's content-free shape."""
    scopes = result.get("scopes", [])
    typed = []
    for s in scopes:
        c = s.get("hx/content", {})
        pos = c.get("position")
        if isinstance(pos, int):
            typed.append((pos, str(s.get("hx/type", "?"))))
    return [t for _, t in sorted(typed)]


def masked_accuracy(sequences: dict[int, list[str]]) -> dict:
    """Leave-one-proof-out masked prediction from (prev, next) bigrams."""
    per_type_hit: Counter[str] = Counter()
    per_type_n: Counter[str] = Counter()
    for held in sequences:
        # pooled context counts from the other proofs
        ctx: dict[tuple, Counter] = defaultdict(Counter)
        for other, seq in sequences.items():
            if other == held:
                continue
            for i, t in enumerate(seq):
                prev = seq[i - 1] if i > 0 else "<s>"
                nxt = seq[i + 1] if i + 1 < len(seq) else "</s>"
                ctx[(prev, nxt)][t] += 1
                ctx[("prev", prev)][t] += 1
                ctx[("next", nxt)][t] += 1
        seq = sequences[held]
        for i, t in enumerate(seq):
            prev = seq[i - 1] if i > 0 else "<s>"
            nxt = seq[i + 1] if i + 1 < len(seq) else "</s>"
            dist = (ctx.get((prev, nxt))
                    or (ctx.get(("prev", prev), Counter())
                        + ctx.get(("next", nxt), Counter())))
            guess = dist.most_common(1)[0][0] if dist else None
            per_type_n[t] += 1
            if guess == t:
                per_type_hit[t] += 1
    rows = []
    for t, n in per_type_n.most_common():
        rows.append({"type": t, "n": n,
                     "masked-accuracy": round(100 * per_type_hit[t] / n, 1)})
    overall = round(100 * sum(per_type_hit.values()) / max(1, sum(per_type_n.values())), 1)
    return {"overall": overall, "per-type": rows}


def shape_profile(seq: list[str]) -> dict:
    n = max(1, len(seq))
    c = Counter(seq)
    fams = Counter(t.split("/")[0] for t in seq)
    # transition entropy: how stereotyped is the local grammar
    trans: dict[str, Counter] = defaultdict(Counter)
    for a, b in zip(seq, seq[1:]):
        trans[a][b] += 1
    import math
    ent = []
    for a, dist in trans.items():
        tot = sum(dist.values())
        ent.append(-sum((k / tot) * math.log2(k / tot) for k in dist.values()))
    return {
        "len": len(seq),
        "distinct-types": len(c),
        "mean-transition-entropy": round(sum(ent) / max(1, len(ent)), 3),
        "frac-bind": round(fams.get("bind", 0) / n, 3),
        "frac-constrain": round(fams.get("constrain", 0) / n, 3),
        "frac-wire": round(fams.get("wire", 0) / n, 3),
        "frac-quant": round(fams.get("quant", 0) / n, 3),
        "frac-assume": round(fams.get("assume", 0) / n, 3),
        "frac-env": round(fams.get("env", 0) / n, 3),
    }


def shape_vs_label(profiles: dict[str, dict], labels: dict[str, str]) -> dict[str, dict]:
    keys = ["len", "distinct-types", "mean-transition-entropy",
            "frac-bind", "frac-constrain", "frac-wire", "frac-quant", "frac-assume"]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for pid, prof in profiles.items():
        by_label[labels[pid]].append(prof)
    out = {}
    for label, rows in sorted(by_label.items()):
        out[label] = {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}
        out[label]["n"] = len(rows)
    return out


def main() -> None:
    sequences: dict[str, list[str]] = {}
    profiles: dict[str, dict] = {}
    labels: dict[str, str] = {}
    for path in pta.tex_files(pta.FULL_TEX_DIR):
        n = int("".join(ch for ch in path.stem.split("-")[0] if ch.isdigit()))
        result = pta.audit_tex(path)
        seq = shape_sequence(result)
        pid = f"first-proof-{n}"
        sequences[pid] = seq
        profiles[pid] = shape_profile(seq)
        labels[pid] = LABELS[n]
    for result in apa.run_audit():
        pid = f"apm-{result['problem']}"
        seq = shape_sequence(result)
        sequences[pid] = seq
        profiles[pid] = shape_profile(seq)
        labels[pid] = result["lean"]["status"]

    pred = masked_accuracy(sequences)
    print(f"masked prediction, overall: {pred['overall']}%  (leave-one-proof-out)")
    print("\ntype,n,masked-accuracy%   — high = grammatical filler, low = informative")
    for row in pred["per-type"]:
        print(f"{row['type']},{row['n']},{row['masked-accuracy']}")

    print("\nshape vs label (first-proof outcomes + APM Lean status):")
    keys = ["len", "distinct-types", "mean-transition-entropy",
            "frac-bind", "frac-constrain", "frac-wire", "frac-quant", "frac-assume"]
    print("class," + ",".join(keys))
    combined = shape_vs_label(profiles, labels)
    for cls, prof in combined.items():
        print(f"{cls}," + ",".join(str(prof[k]) for k in keys))
    print("\nper-proof:")
    for pid in sorted(profiles):
        print(f"{pid} ({labels[pid]}): {profiles[pid]}")

    OUT.write_text(json.dumps({
        "masked-prediction": pred,
        "profiles": profiles,
        "labels": labels,
        "shape-vs-label": combined,
    }, indent=1), encoding="utf-8")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
