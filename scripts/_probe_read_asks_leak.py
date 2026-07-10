#!/usr/bin/env python3
"""Read-only audit: quantify the forward-pass provenance leak in meme_mine_runner.read_asks.

For every user turn that read_asks WOULD KEEP as a human->agent ask, apply the validated
c_mine_joint provenance classifier (promptSource-first, agent-caller exclusion, legacy body
fallback) and count how many are actually agent/harness, not operator. Mirrors how read_pairs
was sized (572/4137 ~14%). Writes nothing; prints counts + a sample of leaked asks.
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meme_mine_runner import text_of, WRAP, CALLER, AUTO_CALLERS, DROP, ASK
from c_mine_joint import AGENT_CALLER, AGENT_AUTHORED, PLUMBING

HOME = os.path.expanduser("~")


def classify(o, body, caller):
    """c_mine_joint's is_human test -> 'operator' | 'agent' | 'harness' | 'agent?(legacy)'."""
    psrc = o.get("promptSource")
    if caller and AGENT_CALLER.match(caller):
        return "agent"            # explicit Caller: claude-N/codex-N
    if caller == "joe":
        return "operator"
    if psrc == "typed":
        return "operator"
    if psrc == "sdk":
        return "agent"            # programmatic bell
    if psrc == "system":
        return "harness"
    # legacy (<none>/queued): body heuristic
    if AGENT_AUTHORED.search(body) or PLUMBING.search(body) or (caller and caller in AUTO_CALLERS):
        return "agent?(legacy)"
    return "operator"


kept = 0
leak = {"agent": 0, "harness": 0, "agent?(legacy)": 0}
samples = []
for f in sorted(glob.glob(f"{HOME}/.claude/projects/*/*.jsonl")):
    for line in open(f, errors="replace"):
        if '"type"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "user":
            continue
        txt = text_of(o.get("message", {}).get("content"))
        if not txt or not txt.strip():
            continue
        txt = txt.strip()
        body = txt
        wm = WRAP.search(txt)
        if wm:
            body = wm.group(1).strip()
        cm = CALLER.search(txt)
        caller = cm.group(1).lower() if cm else None
        one = body.replace("\n", " ").strip()
        # read_asks keep test (verbatim from meme_mine_runner.read_asks)
        if (caller and caller in AUTO_CALLERS) or (not wm and re.search(r"finished job|bell sent|🔔|\(state: ", body)):
            continue
        if not (40 < len(one) < 600 and not DROP.match(one) and ASK.search(one)):
            continue
        kept += 1
        verdict = classify(o, body, caller)
        if verdict != "operator":
            leak[verdict] += 1
            if len(samples) < 12:
                samples.append((verdict, caller, o.get("promptSource"), one[:140]))

total_leak = sum(leak.values())
print(f"read_asks KEEPS:        {kept} asks")
print(f"of those, NON-operator: {total_leak}  ({100*total_leak/max(kept,1):.1f}%)")
print(f"  by class: {leak}")
print("\nsample leaked asks (verdict | caller | promptSource | text):")
for v, c, p, s in samples:
    print(f"  [{v}] caller={c} psrc={p}  {s}")
