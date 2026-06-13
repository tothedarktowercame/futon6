#!/usr/bin/env python3
"""Standalone invariant checker over DP paper-anatomy markup — the convergence
engine for non-interactive structure mining (Joe, 2026-06-13).

Detectors PROPOSE structure (dp_paper_view); this checker DISPOSES — it emits a
typed violation list against the markup invariants. The two are adversaries:
COVERAGE invariants push tagging UP (every symbol, every $-span, every entity);
WELL-FORMEDNESS invariants punish sloppy tagging (atomic math, proper nesting,
no straddle). You can't satisfy coverage by spraying tags (it trips
well-formedness) or well-formedness by tagging nothing (it trips coverage); the
fixpoint where BOTH hold is the correct markup. Author != reviewer even in
code: the checker NEVER imports the detector — it reads only the emitted JSON.

Violation schema — the unit the dispatch pool consumes:
    {inv, severity, start, end, msg, fix}
    severity: "error" (well-formedness; must fix) | "debt" (coverage gap)
    fix:      "tighten-detector" | "extend-coverage" | "irreducible-debt"

A violation with fix=tighten-detector/extend-coverage is dispatchable work
(route to a Codex agent). fix=irreducible-debt is honest residue, NOT a bug —
a symbol used but undefined anywhere is a real definition hole (the shuttle's
Skuld/DEBT cell, a prose `sorry`); it is recorded, not dispatched.

    check_invariants.py <paper-id>     # one paper  -> data/loss/<paper>.json
    check_invariants.py --corpus       # aggregate   -> data/loss/dashboard.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import anatomy_v0_sweep as sweep  # SHARED math-span tokenizer (delimiter parity,
# \$ / $$ handling) — the same ground-truth the detector uses for "where math
# is". Agreeing on the span tokenizer is not an author≠reviewer breach (it's
# agreeing what a "line" is); the invariant LOGIC below stays independent.

ROOT = Path("/home/joe/code/futon6")
GOLDEN_DIR = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
LOSS_DIR = ROOT / "data" / "loss"

# marks whose extent is a structural scope (must not straddle math, must nest):
STRUCTURAL_SCOPE = {"let-binder"}          # dp layer
STRUCTURAL_ENV_PREFIXES = ("env/",)        # legitimately multi-sentence
SYMBOL_KINDS = {"symbol", "symbol-grounded", "classified", "concept-typed",
                "role-gap", "unknown"}      # any tag that "covers" a letter-run
MATH_KINDS = {"math"}
LETTER_RUN = re.compile(r"(?<!\\)(?<![A-Za-z])[A-Za-z][A-Za-z0-9]*")


def _math_spans(text):
    """(start, end) of every math span — the atomic math regions — via the
    SHARED tokenizer (proper $ parity / \\$ / $$), NOT a naive $[^$]+$ regex
    (which mis-pairs a closing $ with the next opening $ and swallows the prose
    between two spans). math_spans yields (start, end, delim, body)."""
    return [(s, e) for s, e, _d, _b in sweep.math_spans(text)]


def _is_structural(m):
    k = m.get("kind", "")
    if m.get("layer") == "scope":
        return not k.startswith(STRUCTURAL_ENV_PREFIXES)
    return m.get("layer") == "dp" and k in STRUCTURAL_SCOPE


def check_paper(paper, data=None):
    """Return {paper, coverage{...}, violations[...], counts{...}}."""
    if data is None:
        f = GOLDEN_DIR / f"fable-{paper}-dp-emacs.json"
        data = json.loads(f.read_text())
    text, marks = data["text"], data["marks"]
    spans = _math_spans(text)
    V = []  # violations

    def add(inv, sev, s, e, msg, fix):
        V.append({"inv": inv, "severity": sev, "start": s, "end": e,
                  "msg": msg, "fix": fix})

    struct = [m for m in marks if _is_structural(m)]
    math_marks = [m for m in marks if m.get("kind") in MATH_KINDS]

    # --- WELL-FORMEDNESS ------------------------------------------------
    # W-ATOMIC: no structural-scope boundary strictly inside a $...$ span.
    # (A scope may CONTAIN a span or sit outside it; it may not split one.)
    for m in struct:
        for s, e in spans:
            inside_start = s < m["start"] < e
            inside_end = s < m["end"] < e
            if inside_start or inside_end:
                add("W-ATOMIC", "error", m["start"], m["end"],
                    f'{m.get("kind")} boundary inside math span '
                    f'{text[s:e][:30]!r} — math is atomic', "tighten-detector")
                break
    # W-NEST: structural scopes nest or are disjoint, never partially overlap.
    ss = sorted(struct, key=lambda m: (m["start"], -m["end"]))
    for i, a in enumerate(ss):
        for b in ss[i + 1:]:
            if b["start"] >= a["end"]:
                break  # sorted: no later scope can overlap a
            if b["end"] > a["end"]:  # a.start <= b.start < a.end < b.end
                add("W-NEST", "error", b["start"], b["end"],
                    f'{b.get("kind")} partially overlaps {a.get("kind")} '
                    f'(crossing, not nesting)', "tighten-detector")
    # W-SENTENCE: a non-env structural scope must not cross ". ".
    for m in struct:
        seg = text[m["start"]:m["end"]]
        if ". " in seg:
            add("W-SENTENCE", "error", m["start"], m["end"],
                f'{m.get("kind")} crosses a sentence boundary (". ")',
                "tighten-detector")

    # --- COVERAGE -------------------------------------------------------
    # C-MATH-NONNULL: every $...$ span carries a math mark (R1, hungry-$).
    math_extents = [(m["start"], m["end"]) for m in math_marks]
    null_spans = 0
    for s, e in spans:
        if not any(ms <= s and me >= e for ms, me in math_extents):
            null_spans += 1
            add("C-MATH-NONNULL", "debt", s, e,
                f'$-span {text[s:e][:30]!r} has no math mark', "extend-coverage")
    # C-SYM-TAGGED / C-SYM-GROUND: every letter-run inside math is tagged;
    # ungrounded tagged symbols are explicit (countable) debt.
    sym_marks = [m for m in marks if m.get("kind") in SYMBOL_KINDS]
    sym_extents = [(m["start"], m["end"], m["kind"]) for m in sym_marks]
    total_syms = tagged = grounded = 0
    for s, e in spans:
        for lm in LETTER_RUN.finditer(text[s:e]):
            ls, le = s + lm.start(), s + lm.end()
            total_syms += 1
            cover = [k for ms, me, k in sym_extents if ms <= ls and me >= le]
            if cover:
                tagged += 1
                if "symbol-grounded" in cover or "concept-typed" in cover \
                        or "classified" in cover:
                    grounded += 1
                else:
                    add("C-SYM-GROUND", "debt", ls, le,
                        f'symbol {text[ls:le]!r} tagged but ungrounded',
                        "extend-coverage")
            else:
                add("C-SYM-TAGGED", "debt", ls, le,
                    f'letter-run {text[ls:le]!r} in math is untagged',
                    "extend-coverage")
    # C-DEFINIENS-DEBT: a definiens that resolves nowhere = irreducible debt.
    for m in marks:
        if m.get("kind") == "definiens" and m.get("fields"):
            cov = next((v for k, v in m["fields"] if k == "coverage"), "")
            if "DEBT" in cov:
                add("C-DEFINIENS-DEBT", "debt", m["start"], m["end"],
                    f'definiens {text[m["start"]:m["end"]][:40]!r} undefined '
                    f'(Lean/PM/nLab all miss)', "irreducible-debt")

    # --- COVERAGE BEST-GUESS (per-paper scalars) ------------------------
    def rate(n, d):
        return round(n / d, 4) if d else 1.0
    errors = [v for v in V if v["severity"] == "error"]
    coverage = {
        "math_spans": len(spans),
        "math_null": null_spans,
        "math_coverage": rate(len(spans) - null_spans, len(spans)),
        "symbols": total_syms,
        "symbol_tagged": rate(tagged, total_syms),
        "symbol_grounded": rate(grounded, total_syms),
        "wellformed_errors": len(errors),
        # one headline number: grounded-symbol rate is the live convergence dial
        "best_guess": rate(grounded, total_syms),
    }
    counts = {}
    for v in V:
        counts[v["inv"]] = counts.get(v["inv"], 0) + 1
    return {"paper": paper, "coverage": coverage, "counts": counts,
            "violations": V}


def _print_paper(rep):
    c, counts = rep["coverage"], rep["counts"]
    print(f"\n{rep['paper']}  —  best-guess coverage {c['best_guess']:.0%} "
          f"(grounded {c['symbol_grounded']:.0%}, tagged {c['symbol_tagged']:.0%} "
          f"of {c['symbols']} symbols; math {c['math_coverage']:.0%} of "
          f"{c['math_spans']} spans)")
    err = c["wellformed_errors"]
    print(f"  well-formedness: {'CLEAN' if err == 0 else str(err)+' ERRORS'}"
          f"   debt: {sum(1 for v in rep['violations'] if v['severity']=='debt')}")
    for inv in sorted(counts):
        sev = next(v["severity"] for v in rep["violations"] if v["inv"] == inv)
        fix = next(v["fix"] for v in rep["violations"] if v["inv"] == inv)
        flag = "✗" if sev == "error" else "·"
        print(f"    {flag} {inv:18} {counts[inv]:5}   [{fix}]")


def corpus():
    LOSS_DIR.mkdir(parents=True, exist_ok=True)
    reps = []
    for f in sorted(GOLDEN_DIR.glob("fable-*-dp-emacs.json")):
        paper = f.name[len("fable-"):-len("-dp-emacs.json")]
        rep = check_paper(paper, json.loads(f.read_text()))
        (LOSS_DIR / f"{paper}.json").write_text(json.dumps(rep))
        reps.append(rep)
        _print_paper(rep)
    # aggregate
    agg = {"papers": len(reps), "by_invariant": {}, "totals": {
        "errors": 0, "debt": 0, "symbols": 0, "grounded": 0}}
    for rep in reps:
        for inv, n in rep["counts"].items():
            agg["by_invariant"][inv] = agg["by_invariant"].get(inv, 0) + n
        agg["totals"]["errors"] += rep["coverage"]["wellformed_errors"]
        agg["totals"]["debt"] += sum(1 for v in rep["violations"]
                                     if v["severity"] == "debt")
        agg["totals"]["symbols"] += rep["coverage"]["symbols"]
        agg["totals"]["grounded"] += round(rep["coverage"]["symbol_grounded"]
                                           * rep["coverage"]["symbols"])
    t = agg["totals"]
    agg["corpus_best_guess"] = round(t["grounded"] / t["symbols"], 4) \
        if t["symbols"] else 1.0
    (LOSS_DIR / "dashboard.json").write_text(json.dumps(agg, indent=1))
    print(f"\n{'='*60}\nCORPUS  ({agg['papers']} papers)  "
          f"best-guess {agg['corpus_best_guess']:.0%}   "
          f"errors {t['errors']}   debt {t['debt']}")
    for inv in sorted(agg["by_invariant"]):
        print(f"    {inv:18} {agg['by_invariant'][inv]:6}")
    print(f"wrote {LOSS_DIR}/dashboard.json")
    return agg


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: check_invariants.py <paper-id> | --corpus")
        return 2
    if argv[0] == "--corpus":
        corpus()
        return 0
    rep = check_paper(argv[0])
    LOSS_DIR.mkdir(parents=True, exist_ok=True)
    (LOSS_DIR / f"{argv[0]}.json").write_text(json.dumps(rep))
    _print_paper(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
