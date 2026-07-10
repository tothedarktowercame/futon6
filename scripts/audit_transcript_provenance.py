#!/usr/bin/env python3
"""Quantify the forward-pass leak (E-patch-agent-evidence-leaks, VERIFY item a).

Over the whole transcript corpus: (1) the 4-way provenance split of all `user` turns, and (2) among
the turns the OLD read_asks would ACCEPT as asks (ASK-cue + length + not-DROP, minus its AUTO_CALLERS
check), how many are NON-operator by the shared classifier — i.e. the inter-agent/harness contamination
that landed in data/meme-mine/. The new read_asks (is_operator gate) removes exactly these.
"""
import glob, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcript_provenance import classify, raw_text, WRAP, CALLER, AUTO_CALLERS
from meme_mine_runner import ASK, DROP

HOME = os.path.expanduser("~")
OLD_BELL = re.compile(r"finished job|bell sent|🔔|\(state: ")   # the old read_asks' partial regex

def old_accepts(o, body, wm):
    """Replicate the OLD read_asks acceptance test (pre-F1)."""
    cm = CALLER.search(raw_text(o)); caller = cm.group(1).lower() if cm else None
    if (caller and caller in AUTO_CALLERS) or (not wm and OLD_BELL.search(body)):
        return False
    one = body.replace("\n", " ").strip()
    return bool(40 < len(one) < 600 and not DROP.match(one) and ASK.search(one))

split = Counter(); old_ask_split = Counter(); old_n = 0
for f in sorted(glob.glob(f"{HOME}/.claude/projects/*/*.jsonl")):
    for line in open(f, errors="replace"):
        if '"type"' not in line: continue
        try: o = json.loads(line)
        except Exception: continue
        if o.get("type") != "user": continue
        txt = raw_text(o)
        if not txt or not txt.strip(): continue
        prov = classify(o)
        split[prov] += 1
        wm = WRAP.search(txt); body = wm.group(1).strip() if wm else txt
        if old_accepts(o, body, wm):
            old_n += 1; old_ask_split[prov] += 1

leak = old_n - old_ask_split["operator"]
print("=== all user turns — 4-way provenance split ===")
for k in ("operator", "agent", "harness", "unknown"): print(f"  {k:9s} {split[k]}")
print(f"\n=== OLD read_asks ask-candidates: {old_n} ===")
for k in ("operator", "agent", "harness", "unknown"): print(f"  {k:9s} {old_ask_split[k]}")
print(f"\nFORWARD LEAK (non-operator asks the old filter accepted): {leak}/{old_n} = {100*leak/max(1,old_n):.1f}%")
print(f"NEW read_asks (is_operator gate) keeps {old_ask_split['operator']} operator asks, drops the {leak} leaked.")
