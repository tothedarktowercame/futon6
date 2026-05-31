#!/usr/bin/env python3
"""Build the CT term-frequency PRIOR from raw eprint text.

M-prior-mathematics step 1. The PRIOR is P(term | CT corpus) computed from the
RAW .tex of every math.CT eprint -- independent of our extractor. (The extractor's
output in mark2/ner-terms is the POSTERIOR; the over-detection signal is posterior
>> prior. This script builds only the prior.)

Input:  storage/futon6/data/arxiv-math-ct-eprints/*.tar.gz
        NOTE (verified 2026-05-31): despite the .tar.gz extension these are NOT
        tarballs -- each is a single gzip-compressed .tex file. The directory
        itself is the math.CT filter. Read via gzip.open, not tarfile.
Output: futon6/data/ct-term-prior.json
        { "n_docs": N, "unigram_df": {term: doc_count}, "bigram_df": {term: doc_count} }
        prior P(term|CT) ~= df[term] / n_docs  (document frequency = #papers containing term)

Document-frequency (not raw count) is the unit: it matches "how many papers is
this term in", which is the right denominator for the over-detection comparison.
"""
import sys, os, re, json, gzip, glob
from collections import Counter

EPRINT_DIR = "/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints"
OUT = "/home/joe/code/futon6/data/ct-term-prior.json"

# Light LaTeX stripping: drop commands, math, braces; keep word tokens.
RE_COMMENT   = re.compile(r"(?<!\\)%.*")
RE_MATH      = re.compile(r"\$[^$]*\$")
RE_COMMAND   = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?")
RE_NONALPHA  = re.compile(r"[^a-z\s]")
STOP = {
    "a","an","and","are","as","at","be","by","for","from","has","in","is","it",
    "its","of","on","or","our","that","the","their","then","there","these",
    "this","to","with","if","let","any","every","each","all","such","we","can",
    "be","which","where","when","so","not","but","also","both","may","one","two",
}

def doc_terms(tex):
    s = tex.lower()
    s = RE_COMMENT.sub(" ", s)
    s = RE_MATH.sub(" ", s)
    s = RE_COMMAND.sub(" ", s)
    s = s.replace("{"," ").replace("}"," ").replace("\\"," ")
    s = RE_NONALPHA.sub(" ", s)
    toks = [t for t in s.split() if len(t) >= 2]
    unis = {t for t in toks if t not in STOP and len(t) >= 3}
    bis = set()
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i+1]
        if a in STOP or b in STOP: continue
        if len(a) < 3 or len(b) < 3: continue
        bis.add(a + " " + b)
    return unis, bis

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all
    files = sorted(glob.glob(os.path.join(EPRINT_DIR, "*.tar.gz")))
    if limit: files = files[:limit]
    uni_df, bi_df = Counter(), Counter()
    n_docs = 0
    n_err = 0
    for i, fp in enumerate(files):
        try:
            with gzip.open(fp, "rb") as gf:
                tex = gf.read().decode("utf-8", "ignore")
        except Exception:
            n_err += 1; continue
        if not tex:
            n_err += 1; continue
        unis, bis = doc_terms(tex)
        uni_df.update(unis); bi_df.update(bis)
        n_docs += 1
        if (i+1) % 1000 == 0:
            print(f"  ... {i+1}/{len(files)} docs, uni_vocab={len(uni_df)}", flush=True)
    # prune hapax bigrams to keep file size sane
    bi_df = Counter({k:v for k,v in bi_df.items() if v >= 3})
    out = {"n_docs": n_docs, "n_err": n_err,
           "unigram_df": dict(uni_df), "bigram_df": dict(bi_df)}
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"DONE n_docs={n_docs} n_err={n_err} uni_vocab={len(uni_df)} bi_vocab(>=3)={len(bi_df)}")
    print(f"wrote {OUT}")

    # Probe: the terms from our debate. Generic words should have HIGH df;
    # genuine novel CT terms should have LOW/MODERATE df.
    probe = ["objects","left","cartesian","functor","morphism","category",
             "pretopos","lextensive","bicategory","operad","sheaf",
             "stable marriage","gale shapley"]
    print("\nPROBE (term : doc-freq : prior P(term|CT)):")
    for t in probe:
        df = bi_df.get(t, 0) if " " in t else uni_df.get(t, 0)
        print(f"  {t:18s} {df:6d}  {df/n_docs:.4f}" if n_docs else f"  {t}: n_docs=0")

if __name__ == "__main__":
    main()
