#!/usr/bin/env python3
"""MEME-MINE (joint) — the RIGHT mining: turn × missions × patterns, on the GPU box.

Joe's correction (2026-06-25): sending only turns wastes the model and leaves the turn↔mission/pattern
grounding to a weak CPU reconstruction. We HAVE all three objects (turns=inference-steps, missions=preprints,
patterns=heuristics) — so the CPU is the RETRIEVER (candidate missions+patterns per turn) and the GPU 70B
does the joint reasoning:
  (a) GROUND endpoints to real mission/pattern/cap ids (high recall),
  (b) CHARACTERIZE which candidate patterns the turn instantiates (PSR/PUR from the record),
  (c) COMPOSE a cascade (the pattern-semilattice for the circumstance — the ARGUE move),
  (d) WRITE a new pattern when no candidate fits (R17 structure-learning / niche-construction).

Reuses the simple runner's turn reader + backend pattern, and concept-tag's gazetteer as the retriever.
Stub-runnable here (plumbing + retrieval); openai/GPU for the real joint reasoning.

  futon6/.venv/bin/python scripts/meme_mine_joint.py --backend stub --limit 6
  # on box (vLLM): --backend openai
"""
import argparse, json, os, re, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meme_mine_runner import read_asks, _sanitize_json_escapes
from mission_concept_tag import gazetteer, spot

ROOT = "/home/joe/code/futon6"; OUT = f"{ROOT}/data/meme-mine"
F3A = "/home/joe/code/futon3a/resources/notions"


def registry():
    """id → short description, for enriching retrieved candidates (the 'send missions+patterns' payload)."""
    desc = {}
    try:
        for e in json.load(open(f"{F3A}/minilm_pattern_embeddings.json")):
            desc[f"pattern/{e['id']}"] = e.get("title", e["id"].rsplit("/", 1)[-1])
    except Exception:
        pass
    try:
        for e in json.load(open(f"{F3A}/minilm_mission_embeddings.json")):
            b = e["basename"]
            desc[f"mission/{b}"] = (e.get("summary") or e.get("title") or b)[:160]
    except Exception:
        pass
    return desc


def retrieve(ask, thread, gaz, desc, k=6):
    """CPU retriever: candidate missions + patterns for this turn (concept-tag gazetteer over turn+thread)."""
    hits = spot(ask + " " + " ".join(thread), gaz)
    miss = [(cid, desc.get(cid, cid)) for cid, t in hits.items() if cid.startswith("mission/")][:k]
    pat = [(cid, desc.get(cid, cid)) for cid, t in hits.items() if cid.startswith("pattern/")][:k]
    return miss, pat


INSTR = """You mine ONE human→agent chat turn (an INFERENCE STEP) jointly against CANDIDATE MISSIONS
(preprints) and CANDIDATE PATTERNS (heuristics) retrieved for it. Use the THREAD CONTEXT to resolve
references. Emit STRICT JSON only:
{"memes":[{"have":{"text","ref","tier":"named|contextual|unsupported","evidence"},"want":{...},"op","maturity":"open|correlated|constructed"}],
 "pattern_apps":[{"pattern":<candidate-pattern-id>,"role":"used|relevant","evidence":<verbatim span>}],
 "cascade":{"patterns":[<candidate-pattern-ids, >=2, ordered>],"rationale":<one line>} | null,
 "new_patterns":[{"name":<kebab-id>,"if":...,"however":...,"then":...,"because":...,"evidence":<verbatim span>}]}
Rules:
- OP is the OPERATION the turn invokes on an artifact/mission — a WM move-class, NOT a discourse act.
  Choose ONE from: build · create · add · update · fix · wire · port · mine · implement · write · extend ·
  refine · execute · run · deploy · dispatch · find · investigate · reuse · relate · assign · preregister ·
  reconstruct · review · verify. If the turn is purely conversational (agreeing, elaborating, comparing,
  asking-about) with NO operation on an artifact, set op="none" — do NOT invent a discourse verb
  (elaborate/contrast/compare/request/clarify/describe/discuss are NOT valid ops). op="none" memes are kept
  for vocabulary but are non-actionable; never emit a meme whose have==want.
- ENDPOINTS: prefer the TRUE referent. Use a CANDIDATE id only when the turn genuinely refers to it
  (ref=that id, tier="named"); otherwise a contextual referent WITH a verbatim "evidence" span
  (tier="contextual"); else ref=null, tier="unsupported". Do NOT force a candidate id the turn doesn't mean.
- pattern_apps: ONLY candidate patterns the turn actually instantiates, each citing verbatim evidence.
- cascade: the smallest candidate-pattern set that makes the case for THIS circumstance (null if <2).
- new_patterns: ALMOST ALWAYS []. Across a batch expect FEWER THAN 1 in 5 turns to propose one — most turns
  are routine asks (build/create/fix/find/commit) that are NOT new patterns. Propose ONE only when ALL hold:
  pattern_apps is empty AND op is operational AND the turn enacts a GENERALIZABLE method reusable ACROSS
  missions that NO candidate pattern names AND it is NOT a paraphrase of this meme's have→want. A concrete
  one-off task is never a pattern. If you cannot state the reusable IF/THEN crisply, emit []. Default []."""


_FEWSHOT_CACHE = None
def _fewshot_messages():
    """Golden few-shot exemplars prepended to every call (gitignored data/golden/forward-fewshot.json) —
    teaches the new_patterns/op discipline. Validated 2026-06-25 on a held-out batch: new_patterns 27%->8%,
    contextual tiers 0%->25%, gate PASS. Loaded+cached once. Disable: FEWSHOT_OFF=1. Override: MEME_FEWSHOT."""
    global _FEWSHOT_CACHE
    if _FEWSHOT_CACHE is not None:
        return _FEWSHOT_CACHE
    msgs = []
    if os.environ.get("FEWSHOT_OFF") != "1":
        path = os.environ.get("MEME_FEWSHOT", f"{ROOT}/data/golden/forward-fewshot.json")
        try:
            for ex in json.load(open(path)):
                u = ("CANDIDATE MISSIONS:\n  (none)\nCANDIDATE PATTERNS:\n  (none)\n"
                     f"THREAD CONTEXT (recent last):\n(none)\nTURN:\n{ex['turn']}")
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(ex["ideal"])}]
        except FileNotFoundError:
            pass
    _FEWSHOT_CACHE = msgs
    return msgs


def call_openai(ask, thread, miss, pat, model):
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    cm = "\n".join(f"  {cid} — {d}" for cid, d in miss) or "  (none)"
    cp = "\n".join(f"  {cid} — {d}" for cid, d in pat) or "  (none)"
    user = (f"CANDIDATE MISSIONS:\n{cm}\nCANDIDATE PATTERNS:\n{cp}\n"
            f"THREAD CONTEXT (recent last):\n" + ("\n".join(thread) or "(none)") + f"\nTURN:\n{ask}")
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "system", "content": INSTR}] + _fewshot_messages()
                                   + [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(_sanitize_json_escapes(m.group(0))) if m else {}


def call_stub(ask, thread, miss, pat):
    """Plumbing/retrieval stub: ground the meme to the first candidate mission; characterize the first
    candidate pattern. NOT real joint reasoning — validates retrieval + schema only."""
    have = ({"text": miss[0][1][:40], "ref": miss[0][0], "tier": "named", "evidence": "(retrieved)"}
            if miss else {"text": "(context)", "ref": None, "tier": "unsupported", "evidence": None})
    return {"memes": [{"have": have, "want": {"text": "(stub)", "ref": None, "tier": "unsupported", "evidence": None},
                       "op": "act", "maturity": "open"}],
            "pattern_apps": ([{"pattern": pat[0][0], "role": "relevant", "evidence": "(retrieved)"}] if pat else []),
            "cascade": None, "new_patterns": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="mark4-70b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    gaz, desc = gazetteer(), registry()
    asks = read_asks(a.limit or None)
    print(f"asks: {len(asks)}  backend={a.backend}  registry: {len(desc)} mission+pattern descriptions\n")
    out, n_pat, n_casc, n_new, n_miss_cand = [], 0, 0, 0, 0
    last_ckpt = 0
    for s in asks:
        miss, pat = retrieve(s["ask"], s["thread"], gaz, desc, a.k)
        n_miss_cand += 1 if miss else 0
        try:
            r = call_openai(s["ask"], s["thread"], miss, pat, a.model) if a.backend == "openai" \
                else call_stub(s["ask"], s["thread"], miss, pat)
            # The model may emit JSON null for any list field; normalize so a single bad record
            # cannot crash the whole run (learned 2026-06-25: new_patterns=null aborted a 1448-ask
            # run AFTER all GPU work, before the end-of-loop write — total loss). Building the record
            # inside the try means any malformed record is caught by the per-ask skip below, not main().
            r["memes"] = r.get("memes") or []
            r["pattern_apps"] = r.get("pattern_apps") or []
            r["new_patterns"] = r.get("new_patterns") or []
            rec = {"id": s["id"], "ask": s["ask"], "provenance": {"project": s["project"], "session": s["session"]},
                   "candidates": {"missions": [m[0] for m in miss], "patterns": [p[0] for p in pat]}, **r}
        except Exception as e:
            print(f"  ! {s['id']} failed: {e}"); continue
        n_pat += len(r["pattern_apps"])
        n_casc += 1 if r.get("cascade") else 0
        n_new += len(r["new_patterns"])
        out.append(rec)
        # Checkpoint every >=200 NEW records so a late crash/kill never discards a long expensive run.
        # Since-last counter, not `len(out) % 200 == 0`: robust even if a future change appends more than
        # one record per ask (the "Y2C" rollover that can skip an exact-multiple test).
        if len(out) - last_ckpt >= 200:
            json.dump(out, open(f"{OUT}/joint-memes.{a.backend}.json", "w"), indent=2)
            print(f"  .. checkpoint: {len(out)} records written")
            last_ckpt = len(out)
    o = f"{OUT}/joint-memes.{a.backend}.json"
    json.dump(out, open(o, "w"), indent=2)
    print(f"turns with >=1 candidate mission (retriever recall): {n_miss_cand}/{len(asks)}")
    print(f"pattern-applications: {n_pat} · cascades: {n_casc} · NEW patterns proposed: {n_new}")
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
