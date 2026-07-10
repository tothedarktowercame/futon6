#!/usr/bin/env python3
"""MEME-MINE (v0, small-sample): human→agent turns → memes (have→want arrows).

A human→agent turn is an ASK = a (have, want) arrow = a meme (M-operational-vocabulary).
This v0 harness does the DETERMINISTIC half (sample real asks + provenance + 香 salience
pre-tag); the EXTRACTION half (香 → identification: ask → {have, want, op}) is an LLM turn
done in-the-loop on the small sample (the same step a served model runs at scale on a Linode GPU).

  --sample K : pull K real asks (reproducible strided sample) → data/meme-mine/sample.json (+print)
  --emit     : read data/meme-mine/memes.json (the extracted memes) → memes.edn + report

A meme: {have, want, op, maturity, salience_terms} + provenance {project, session, ask}.
"""
import argparse, glob, json, os, re, hashlib
from collections import Counter

HOME = os.path.expanduser("~")
OUT = f"{HOME}/code/futon6/data/meme-mine"
ASK = re.compile(r"\b(can you|could you|please|let'?s|i want|we want|i need|we need|let me|build|fix|add|implement|write|update|create|check|run|show me|wire|port|mine|how do|what about|why (is|are|does)|make (it|sure|a)|should we|do we|let'?s work)\b", re.I)
# drop pure pleasantries / meta
DROP = re.compile(r"^(ok|okay|sure|thanks|thank you|yes|no|yep|nice|cool|great|sounds good)\b[\s.!,]*$", re.I)


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(isinstance(x, dict) and x.get("type") == "tool_result" for x in content):
            return None
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text")
    return None


def collect_asks():
    asks = []
    seen = set()
    for f in sorted(glob.glob(f"{HOME}/.claude/projects/*/*.jsonl")):
        proj = os.path.basename(os.path.dirname(f))
        sess = os.path.basename(f)[:8]
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
            if not txt:
                continue
            txt = txt.strip().replace("\n", " ")
            if not (40 < len(txt) < 600) or DROP.match(txt) or not ASK.search(txt):
                continue
            key = re.sub(r"\W+", "", txt.lower())[:80]
            if key in seen:
                continue
            seen.add(key)
            asks.append({"project": proj, "session": sess, "ask": txt})
    return asks


def do_sample(k):
    asks = collect_asks()
    n = len(asks)
    stride = max(1, n // k)
    sample = [asks[i] for i in range(0, n, stride)][:k]
    for s in sample:
        s["id"] = "ask-" + hashlib.sha1(s["ask"].encode()).hexdigest()[:8]
    os.makedirs(OUT, exist_ok=True)
    json.dump(sample, open(f"{OUT}/sample.json", "w"), indent=2)
    print(f"qualifying asks in corpus: {n}; sampled {len(sample)} (stride {stride})\n")
    for i, s in enumerate(sample):
        print(f"[{i}] {s['id']} ({s['project'][:24]})\n    {s['ask'][:220]}\n")
    print(f"wrote {OUT}/sample.json — fill memes.json then --emit")


def edn_str(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', "'") + '"'


def do_emit():
    memes = json.load(open(f"{OUT}/memes.json"))
    lines = []
    for m in memes:
        mm = m["meme"]
        sal = " ".join(edn_str(t) for t in mm.get("salience_terms", []))
        lines.append(
            f'  {{:meme/id {edn_str(m["id"])} :have {edn_str(mm["have"])} :want {edn_str(mm["want"])}'
            f' :op :{mm["op"]} :maturity :{mm.get("maturity","open")}'
            f' :salience [{sal}] :provenance {{:project {edn_str(m["project"])} :session {edn_str(m["session"])}}}'
            f' :ask {edn_str(m["ask"])}}}')
    edn = (";; memes.edn — MEME-MINE v0: human→agent asks → (have,want) arrows.\n"
           ";; Extracted by an LLM-in-the-loop (claude-1) on the small sample; scale on GPU later.\n"
           f"{{:meme/count {len(memes)}\n :memes [\n" + "\n".join(lines) + "\n ]}\n")
    open(f"{OUT}/memes.edn", "w").write(edn)
    ops = Counter(m["meme"]["op"] for m in memes)
    print(f"emitted {len(memes)} memes → {OUT}/memes.edn")
    print(f"operation-class vocabulary (EMPIRICAL, from the turns — vs the 3 hand-coded move-classes):")
    for op, c in ops.most_common():
        print(f"   :{op}  ×{c}")


def do_dedup():
    """Layer 3 (CPU, SFC-NORM-like): once endpoints carry canonical ids, exact-merge
    (have.ref, want.ref, op) duplicates. op is in the key so self-edge memes (build-R2d
    vs dispatch-R2d, same have==want) don't falsely merge."""
    rm = json.load(open(f"{OUT}/resolved-memes.json"))
    from collections import defaultdict
    tier = Counter()
    keyed = defaultdict(list)
    anchors = Counter()
    for m in rm:
        mm = m["meme"]
        for end in ("have", "want"):
            e = mm[end]
            tier[e["tier"]] += 1
            if e["tier"] == "named" and e["ref"]:
                anchors[e["ref"]] += 1
        h, w = mm["have"]["ref"], mm["want"]["ref"]
        if h and w:
            keyed[(h, w, mm["op"])].append(m["id"])
    ne = sum(tier.values())
    print(f"endpoint resolution tiers (over {ne} endpoints): " +
          " · ".join(f"{k} {v} ({v/ne:.0%})" for k, v in tier.most_common()))
    dedupable = sum(len(v) for v in keyed.values())
    coll = {k: v for k, v in keyed.items() if len(v) > 1}
    print(f"dedupable memes (both endpoints resolved): {dedupable}; unique (have,want,op) keys: {len(keyed)}")
    print(f"collisions to unify-not-mint: {len(coll)} " + (str(coll) if coll else "(none at this N; named anchors collide at scale)"))
    print(f"named anchors (CPU-catchable unification points): {dict(anchors)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    a = ap.parse_args()
    if a.sample:
        do_sample(a.sample)
    elif a.emit:
        do_emit()
    elif a.dedup:
        do_dedup()
    else:
        ap.error("use --sample K | --emit | --dedup")


if __name__ == "__main__":
    main()
