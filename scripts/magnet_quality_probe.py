#!/usr/bin/env python3
"""magnet_quality_probe.py (claude-6, E-have-want-pairs Q-B) — the magnet-quality scorecard.

Q-B: does a MECHANICALLY-DERIVABLE (have->want) magnet produce qualitatively-good cascades on
REAL data, or is good cascade quality fundamentally hand-curation-dependent?

Method: a mechanical-magnet LADDER, scored by F (= accuracy - lambda*complexity, Bayesian Occam;
F>0 = the expansion is accepted). For each real (have,want) source:
  M_idstem   baseline = the live lane's weak feed (mission->psi: strip M-, hyphens->spaces)
  M_want     tokenized GOAL text (capability id / sorry title / mission name) -- want only
  M_havewant reconstructed have->want (current mission state + goal) -- the real impl-#3 magnet
Every magnet is produced by a DETERMINISTIC function of stored fields -- NO hand curation.

Anchors (from E-have-want-pairs.md 1b, hand-curated meme phrasing, the ceiling we test against):
  M-value-creation-loop id-stem F=-0.19  ->  hand-meme F=+0.90
Reads: /tmp/magnet_probe_inputs.json (from magnet_probe_extract.bb). Sim-only; no writes to :7071.

Run:  cd ~/code/futon3a && .venv/bin/python3 \
        /home/joe/code/futon6/scripts/magnet_quality_probe.py /tmp/magnet_probe_inputs.json
"""
import json, re, sys, statistics
sys.path.insert(0, "/home/joe/code/futon3a/holes/labs/M-memes-arrows")
from cascade_construct import construct_cascade, pattern_stem

STOP = {
    "m", "the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "at", "by",
    "sorry", "devmap", "run", "complete", "held", "open",  # scaffolding tokens
}

def toks(s):
    """Deterministic tokenizer: lowercase, split on non-alphanumeric, drop stopwords + 1-char."""
    out = []
    for t in re.split(r"[^a-z0-9]+", (s or "").lower()):
        if len(t) > 1 and t not in STOP:
            out.append(t)
    return out

def baseline_psi(idstem):
    """Replicate cascade_lane/mission->psi: strip leading M-, hyphens->spaces (the weak feed)."""
    s = re.sub(r"^M-", "", str(idstem))
    s = s.replace("-", " ").replace("/", " ").replace("|", " ")
    return s.strip().lower()

def magnets(item):
    idstem = baseline_psi(item["idstem"])
    want = " ".join(toks(item.get("want_raw", "")))
    have = " ".join(toks(item.get("have_raw", "")))
    havewant = (have + " " + want).strip()
    return {"M_idstem": idstem or want, "M_want": want or idstem,
            "M_havewant": havewant or want or idstem}

def on_topic_frac(cascade_stems, want_tokens):
    """Proxy: fraction of cascade patterns sharing >=1 content token with the want text.
    Weak automation of 'on-topic'; the verdict still relies on eyeballed cascades (SAMPLE dump)."""
    if not cascade_stems:
        return 0.0
    wt = set(want_tokens)
    hits = 0
    for stem in cascade_stems:
        st = set(re.split(r"[^a-z0-9]+", stem.lower()))
        if st & wt:
            hits += 1
    return hits / len(cascade_stems)

def score(psi):
    r = construct_cascade(psi, epsilon=0.15)
    stems = [pattern_stem(pid) for (pid, _r, _mc) in r["cascade"]]
    return {"psi": psi, "F": r["F-free-energy"], "size": r["size"],
            "wholeness": r["wholeness"], "accuracy": r["accuracy"],
            "complexity": r["complexity"], "stems": stems}

def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return (num / den) if den else None

def summarize(rows, key):
    fs = [r[key]["F"] for r in rows]
    if not fs:
        return {}
    return {"n": len(fs), "F_mean": round(statistics.mean(fs), 3),
            "F_median": round(statistics.median(fs), 3),
            "frac_F_pos": round(sum(1 for f in fs if f > 0) / len(fs), 3),
            "size_mean": round(statistics.mean([r[key]["size"] for r in rows]), 2)}

def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/magnet_probe_inputs.json"
    items = json.load(open(inp))
    rows = []
    for n, item in enumerate(items):
        mg = magnets(item)
        scored = {k: score(v) for k, v in mg.items()}
        wt = toks(item.get("want_raw", ""))
        for k in scored:
            scored[k]["on_topic"] = round(on_topic_frac(scored[k]["stems"], wt), 3)
        rows.append({"channel": item["channel"], "key": item["key"],
                     "delta_g": item.get("delta_g"), **scored})
        if (n + 1) % 40 == 0:
            print(f"  ...{n + 1}/{len(items)} scored", file=sys.stderr)

    channels = sorted({r["channel"] for r in rows})
    scorecard = {"n_total": len(rows), "magnets": ["M_idstem", "M_want", "M_havewant"],
                 "by_channel": {}, "overall": {}, "lift_havewant_vs_idstem": {}, "diffsub_delta_g": {}}

    for mag in ["M_idstem", "M_want", "M_havewant"]:
        scorecard["overall"][mag] = summarize(rows, mag)
    for ch in channels:
        chrows = [r for r in rows if r["channel"] == ch]
        scorecard["by_channel"][ch] = {mag: summarize(chrows, mag)
                                       for mag in ["M_idstem", "M_want", "M_havewant"]}

    # Lift: M_havewant vs M_idstem, paired.
    for scope, rr in [("overall", rows)] + [(ch, [r for r in rows if r["channel"] == ch]) for ch in channels]:
        d = [r["M_havewant"]["F"] - r["M_idstem"]["F"] for r in rr]
        improved = sum(1 for x in d if x > 1e-6)
        crossed = sum(1 for r in rr if r["M_idstem"]["F"] <= 0 < r["M_havewant"]["F"])
        scorecard["lift_havewant_vs_idstem"][scope] = {
            "n": len(rr), "mean_dF": round(statistics.mean(d), 3) if d else None,
            "frac_improved": round(improved / len(rr), 3) if rr else None,
            "n_crossed_zero": crossed}

    # diffsub: does F track delta_g?
    ds = [r for r in rows if r["channel"] == "diffsub" and r["delta_g"] is not None]
    if ds:
        for mag in ["M_idstem", "M_want", "M_havewant"]:
            fs = [r[mag]["F"] for r in ds]
            dg = [r["delta_g"] for r in ds]
            scorecard["diffsub_delta_g"][mag] = {
                "n": len(ds), "spearman_F_vs_deltaG": round(spearman(fs, dg), 3) if spearman(fs, dg) is not None else None,
                "frac_F_pos": round(sum(1 for f in fs if f > 0) / len(fs), 3)}

    # SAMPLE dump for eyeballed on-topic judgment (spread across channels).
    sample = []
    for ch in channels:
        chrows = [r for r in rows if r["channel"] == ch]
        for r in chrows[:3]:
            sample.append({"channel": ch, "key": r["key"], "delta_g": r["delta_g"],
                           "M_idstem": {"psi": r["M_idstem"]["psi"], "F": r["M_idstem"]["F"], "size": r["M_idstem"]["size"], "stems": r["M_idstem"]["stems"]},
                           "M_havewant": {"psi": r["M_havewant"]["psi"], "F": r["M_havewant"]["F"], "size": r["M_havewant"]["size"], "stems": r["M_havewant"]["stems"], "on_topic": r["M_havewant"]["on_topic"]}})
    scorecard["sample"] = sample

    out = "/home/joe/code/futon6/data/c-vector/magnet-quality-scorecard.json"
    json.dump({"scorecard": scorecard, "rows": rows}, open(out, "w"), indent=1)
    print(json.dumps(scorecard, indent=1))
    print(f"\nwrote {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
