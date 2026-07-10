#!/usr/bin/env python3
"""Post-filter the existing forward artifact for the provenance leak (F1, data side).

The code is fixed (read_asks now gates on transcript_provenance.is_operator), but data/meme-mine/
*.openai.json was produced BEFORE the fix and still carries the ~7% inter-agent/harness asks. Each
record's id is "ask-"+sha1(unwrapped-ask)[:8], so we re-derive provenance two ways:
  (1) AUTHORITATIVE — index the live corpus by that same id, classifying the FULL record (promptSource);
  (2) FALLBACK — for ids the (since-grown) corpus no longer matches, classify the record's own .ask text
      (the leaked asks embed their 'Caller:'/surface preamble, so the body heuristic still catches them).
Keep operator-authored records only. Originals backed up to *.pre-f1.json (reversible).
"""
import glob, hashlib, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcript_provenance import classify, raw_text, WRAP
from meme_mine_runner import ASK, DROP

HOME = os.path.expanduser("~"); MM = "/home/joe/code/futon6/data/meme-mine"

def ask_id(one):
    return "ask-" + hashlib.sha1(one.encode()).hexdigest()[:8]

# (1) authoritative index: ask-id -> provenance, from the live corpus with full-record metadata
index = {}
for f in sorted(glob.glob(f"{HOME}/.claude/projects/*/*.jsonl")):
    for line in open(f, errors="replace"):
        if '"type"' not in line: continue
        try: o = json.loads(line)
        except Exception: continue
        if o.get("type") != "user": continue
        txt = raw_text(o)
        if not txt or not txt.strip(): continue
        wm = WRAP.search(txt); body = wm.group(1).strip() if wm else txt
        index[ask_id(body.replace("\n", " ").strip())] = classify(o)

def prov_of(rec):
    p = index.get(rec["id"])
    if p: return p, "corpus"
    # fallback: classify the record's own ask text (leaked asks carry their Caller:/preamble)
    return classify({"message": {"content": rec["ask"]}, "promptSource": None}), "body"

J = json.load(open(f"{MM}/joint-memes.openai.json"))
kept, dropped = [], []
src = Counter()
for r in J:
    p, how = prov_of(r); src[how] += 1
    (kept if p == "operator" else dropped).append((r, p))

print(f"joint records: {len(J)}  →  kept {len(kept)} operator · dropped {len(dropped)} non-operator")
print("  dropped by provenance:", dict(Counter(p for _, p in dropped)))
print("  resolution source:", dict(src))
print("  sample dropped asks:")
for r, p in dropped[:5]:
    print(f"    [{p}] {r['ask'][:80]!r}")

# write cleaned artifacts (back up originals first)
keptJ = [r for r, _ in kept]
keep_ids = {r["id"] for r in keptJ}
for name in ("joint-memes.openai.json", "resolved-memes.openai.json"):
    path = f"{MM}/{name}"
    if os.path.exists(path) and not os.path.exists(f"{path}.pre-f1"):
        os.rename(path, f"{path}.pre-f1")
json.dump(keptJ, open(f"{MM}/joint-memes.openai.json", "w"), indent=2)
# resolved-memes = flat memes of the kept joint records (the consume tail's input shape)
flat = [{"id": r["id"], "ask": r["ask"], "provenance": r.get("provenance", {}), "meme": m}
        for r in keptJ for m in (r.get("memes") or [])]
json.dump(flat, open(f"{MM}/resolved-memes.openai.json", "w"), indent=2)
print(f"\nwrote cleaned joint-memes.openai.json ({len(keptJ)}) + resolved-memes.openai.json ({len(flat)} memes)")
print("  originals backed up to *.pre-f1")

# re-measure F2 (new_patterns over-firing) + F5 (retriever recall) on the CLEAN set
newp = [n for r in keptJ for n in (r.get("new_patterns") or [])]
asks_with_new = sum(1 for r in keptJ if (r.get("new_patterns") or []))
recall = sum(1 for r in keptJ if (r.get("candidates") or {}).get("missions"))
print(f"\n=== re-measure on clean set ===")
print(f"  F2 new_patterns: {len(newp)} from {len(keptJ)} asks; {asks_with_new}/{len(keptJ)} = {100*asks_with_new/max(1,len(keptJ)):.0f}% of asks propose >=1 (was 95%)")
print(f"  F5 retriever recall: {recall}/{len(keptJ)} = {100*recall/max(1,len(keptJ)):.0f}% (was 27%)")
