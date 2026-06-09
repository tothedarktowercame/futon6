#!/usr/bin/env python3
# mission_domain_classify.py — CANDIDATE :domain tags for the R1 gap-gate (Campaign BSL).
#
# claude-1's (b): growth-surface fixed SIZE but gap is still DOMAIN-BLIND (canon-math 0.8
# ~ war-machine-tuning-local 0.9). The WM-reader must gate gap to LOCAL-CAPABILITY missions
# so math gaps don't pull the WM off Joe's realignment priorities. The ascent-subgraph is
# too narrow (it'd zero war-machine-tuning, a non-graph-producer). So the domain signal must
# come from the CORPUS — here.
#
# This is a FIRST-PASS CANDIDATE ONLY. The local-vs-math partition is JOE'S to ratify (it's
# his realignment priorities; edge cases like M-web-arxana-missions are genuinely both-ish).
# So every tag carries its EVIDENCE (which seeds fired) + a confidence, for inspection. Joe
# ratifies/edits; claude-1's reader then gates gap on the RATIFIED domain.
import re, glob
from pathlib import Path

ROOT = Path("/home/joe/code")
OUT = ROOT / "futon6" / "data" / "mission-domain-candidates.edn"

# high-precision seeds (word-boundary matched). Deliberately NOT generic stack words
# (mission/hole/futon/arxana) — those fire everywhere and don't discriminate.
# NB: dropped proof/embedding/lean/gpu — in THIS corpus they're proof-eval, vector
# embeddings, the verb "lean", and generic compute, not math. (Precision hygiene; the
# local-vs-math LINE stays Joe's to ratify.)
MATH_SEEDS = {
    "theorem", "lemma", "symbol-grounding", "canon", "msc", "arxiv", "planetmath",
    "nlab", "morphism", "functor", "categorical", "category theory", "mathoverflow",
    "stackexchange", "stack exchange", "pythagorean", "denotation", "fingerprint", "ner",
    "mathematics", "mathematical", "superpod", "math.ct", "math.qa", "msc-code",
}
LOCAL_SEEDS = {
    "war-machine", "war machine", "agency", "codex", "sorry", "drawbridge", "efe", "blend",
    "ascent", "guardrail", "overnight", "whistle", "vsatarcs", "capability", "pilot", "wm",
    "peripheral", "nag", "consent-gate", "drawbridge", "star-map", "advanceability",
}


def hits(text, seeds):
    out = {}
    for s in seeds:
        n = len(re.findall(r"(?<![\w-])" + re.escape(s) + r"(?![\w-])", text))
        if n:
            out[s] = n
    return out


def classify(path):
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    mh, lh = hits(text, MATH_SEEDS), hits(text, LOCAL_SEEDS)
    m, l = len(mh), len(lh)               # distinct seeds present = the robust signal
    if m == 0 and l == 0:
        dom, conf = "other", 0.0
    elif l >= m and l > 0:
        dom = "local-capability"
        conf = round((l - m) / (l + m), 2)
    else:
        dom = "math"
        conf = round((m - l) / (l + m), 2)
    return dom, conf, sorted(mh, key=lambda k: -mh[k]), sorted(lh, key=lambda k: -lh[k])


def edn(rows):
    out = ["{:source \"mission-domain-classify\"",
           " :note \"CANDIDATE — Joe ratifies the local-vs-math partition; evidence per mission\"",
           " :missions",
           " ["]
    for stem, repo, dom, conf, mh, lh in rows:
        out.append(f"  {{:mission \"{stem}\" :repo \"{repo}\" :domain :{dom} :confidence {conf:.2f}"
                   f" :math-evidence [{' '.join(chr(34)+t+chr(34) for t in mh[:6])}]"
                   f" :local-evidence [{' '.join(chr(34)+t+chr(34) for t in lh[:6])}]}}")
    out.append(" ]}")
    return "\n".join(out)


WORKSHEET = ROOT / "futon6" / "data" / "mission-domain-ratification-worksheet.md"
EDGE_CASES = {"M-differentiable-math", "M-categorical-code", "M-web-arxana-missions"}


def emit_worksheet(rows):
    """A bounded, one-line-each decision surface for Joe — only the calls that need HIM
    (the ambiguous + the named both-ish edge cases). High-confidence tags are pre-accepted."""
    from collections import Counter
    accepted = [r for r in rows if r[3] >= 0.34 and r[0] not in EDGE_CASES]
    tally = Counter(r[2] for r in accepted)
    edges = [r for r in rows if r[0] in EDGE_CASES]
    amb = [r for r in rows if r[3] < 0.34 and r[2] != "other" and r[0] not in EDGE_CASES]

    def line(r):
        stem, repo, dom, conf, mh, lh = r
        return (f"- [ ] **{stem}**  (candidate `:{dom}` conf {conf:.2f}, {repo})  "
                f"math{{{', '.join(mh[:4])}}} local{{{', '.join(lh[:4])}}}  →  RULING: ______")

    out = [
        "# Mission domain — ratification worksheet (for Joe)",
        "",
        "**Why:** this partition gates the R1 gap-term (Campaign-BSL). The WM's expansion-credit",
        "(`:gap-score`) applies **only to `:local-capability` missions**; `:math`/`:other` → gap 0.",
        "This is literally your realignment priority (local-capability over math), made explicit.",
        "",
        "**Files:** candidate (all 194, auto) = `futon6/data/mission-domain-candidates.edn`.",
        "Ratified (what the WM gap-reader consumes) = `futon6/data/mission-domain-ratified.edn` —",
        "**does NOT exist yet, so the gap-gate is OFF** (safe default: absent ratified file = gap fully off).",
        "",
        "**How to rule:** the high-confidence tags below are pre-accepted (spot-check in the candidate",
        "file). For the ones that need your call, mark `local` / `math` / `other` in RULING. Tell me your",
        "rulings (or tick the boxes) and I'll compile `mission-domain-ratified.edn` — that act is your",
        "ratification; nothing activates until it exists and you flip the gap-reader on.",
        "",
        f"**Pre-accepted (conf ≥ 0.34):** {tally.get('local-capability',0)} local-capability · "
        f"{tally.get('math',0)} math · {sum(1 for r in rows if r[2]=='other')} other.",
        "",
        "## Decisions needed",
        "",
        "### Named edge cases — genuinely both-ish, your call matters most",
        *[line(r) for r in edges],
        "",
        f"### Ambiguous (confidence < 0.34) — {len(amb)} missions",
        *[line(r) for r in sorted(amb, key=lambda r: r[0])],
    ]
    WORKSHEET.write_text("\n".join(out) + "\n")
    print(f"\nwrote worksheet {WORKSHEET}  ({len(edges)} edge cases + {len(amb)} ambiguous = "
          f"{len(edges)+len(amb)} decisions for Joe)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worksheet", action="store_true", help="also emit Joe's ratification worksheet")
    args = ap.parse_args()
    paths = sorted(Path(".").glob("futon*/holes/**/M-*.md"),
                   key=lambda p: p.stem) if Path(".").samefile(ROOT) else \
        sorted(ROOT.glob("futon*/holes/**/M-*.md"), key=lambda p: p.stem)
    seen, rows = set(), []
    for p in paths:
        if p.stem in seen:
            continue
        seen.add(p.stem)
        dom, conf, mh, lh = classify(p)
        rows.append((p.stem, p.parts[-4] if len(p.parts) >= 4 else "?", dom, conf, mh, lh))
    OUT.write_text(edn(rows) + "\n")
    from collections import Counter
    tally = Counter(r[2] for r in rows)
    print(f"wrote {OUT}  ({len(rows)} missions)  tally={dict(tally)}\n")
    print("=== key missions (the realignment story) ===")
    key = {"M-canon-fingerprint-store", "M-bayesian-structure-learning", "M-war-machine-tuning",
           "M-war-machine-pilot", "M-capability-star-map", "M-web-arxana-missions",
           "M-symbol-grounding-scaling-plan", "M-prior-mathematics"}
    for stem, repo, dom, conf, mh, lh in rows:
        if stem in key:
            print(f"  :{dom:16} conf{conf:.2f}  {stem:34} math{mh[:3]} local{lh[:3]}")
    print("\n=== low-confidence (conf<0.34) — the ones for Joe to scrutinize ===")
    amb = [r for r in rows if r[3] < 0.34 and r[2] != "other"]
    for stem, repo, dom, conf, mh, lh in amb[:14]:
        print(f"  :{dom:16} conf{conf:.2f}  {stem:34} m{mh[:3]} l{lh[:3]}")
    print(f"  ...({len(amb)} ambiguous total)")

    if args.worksheet:
        emit_worksheet(rows)


if __name__ == "__main__":
    main()
