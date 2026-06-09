#!/usr/bin/env python3
# Build the MISSION term-frequency PRIOR (Zipf/Pareto base-rate) from M-*.md.
# L3 Pass 1 for M-web-arxana-missions (futon4), mirroring futon6 build_ct_prior.py.
# PRIOR = P(term | mission corpus) = document-frequency over every M-*.md under
# */holes/ across the futon repos. Two derived lexicons (Joe 2026-06-07), one base-rate:
#   SELF-REPRESENTING    : high-df terms NOT in common English = in-stack jargon (our NEs)
#   PROJECTION/KNOCK-OUT : high-df terms that ARE common English = generic-drop / English target
# Output: futon6/data/mission-term-prior.json  { n_docs, unigram_df, bigram_df }
import os, re, json
from collections import Counter

ROOTS = [f"/home/joe/code/{r}" for r in
         ["futon0","futon1a","futon2","futon3","futon3c","futon4","futon5","futon5a","futon6","futon7"]]
OUT = "/home/joe/code/futon6/data/mission-term-prior.json"
WORDS = "/usr/share/dict/words"

RE_FENCE = re.compile(r"```.*?```", re.S)
RE_INLINE = re.compile(r"`[^`]*`")
RE_NONALPHA = re.compile(r"[^a-z\s]")

STOP = set((
    "a an and are as at be by for from has have had in is it its of on or our "
    "that the their then there these this to with if let any every each all such we can which "
    "where when so not but also both may will would could should into per via "
    "no yes do does done how what who why was were been being he she they them his her you your "
    "i me my mine ours us about above after again against between out up down over under more most "
    "other some only own same than too very just now even still much many one two three"
).split())

def doc_tokens(md):
    s = md.lower()
    s = RE_FENCE.sub(" ", s)
    s = RE_INLINE.sub(" ", s)
    s = RE_NONALPHA.sub(" ", s)
    return [t for t in s.split() if len(t) > 2 and t not in STOP]

def main():
    files = []
    for root in ROOTS:
        for dp, _, fns in os.walk(root):
            if "/.git" in dp or "/holes" not in dp:
                continue
            for fn in fns:
                if fn.startswith("M-") and fn.endswith(".md"):
                    files.append(os.path.join(dp, fn))

    n_docs = 0
    uni = Counter(); bi = Counter()
    for f in files:
        try:
            md = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        toks = doc_tokens(md)
        if not toks:
            continue
        n_docs += 1
        uni.update(set(toks))
        bi.update(set(" ".join(b) for b in zip(toks, toks[1:])))

    out = {"n_docs": n_docs, "unigram_df": dict(uni),
           "bigram_df": {k: v for k, v in bi.items() if v >= 3}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False)

    common = set()
    if os.path.exists(WORDS):
        for w in open(WORDS, encoding="utf-8", errors="ignore"):
            w = w.strip().lower()
            if w.isalpha():
                common.add(w)

    ranked = uni.most_common(4000)
    self_rep = [(t, c) for t, c in ranked if t not in common][:40]
    knockout = [(t, c) for t, c in ranked if t in common][:20]
    top_bi = [(k, v) for k, v in bi.most_common(30) if v >= 3]

    print(f"n_docs={n_docs}  unigram_vocab={len(uni)}  bigram_vocab(df>=3)={len(out['bigram_df'])}  -> {OUT}")
    print("\n--- SELF-REPRESENTING lexicon (in-stack jargon: high-df, NOT common English) ---")
    for t, c in self_rep:
        print(f"  P={c/n_docs:4.2f}  df={c:4d}  {t}")
    print("\n--- COMMON-ENGLISH knock-out (high-df dictionary words = generic / projection target) ---")
    for t, c in knockout:
        print(f"  P={c/n_docs:4.2f}  df={c:4d}  {t}")
    print("\n--- top BIGRAMs (multi-word concepts) ---")
    for k, v in top_bi:
        print(f"  P={v/n_docs:4.2f}  df={v:4d}  {k}")

if __name__ == "__main__":
    main()
