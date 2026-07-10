#!/usr/bin/env python3
"""MEME-MINE runner — the GPU mining runner, to holes/meme-mine-runner-spec.md.

Three layers (deterministic brackets, LLM core), for M-operational-vocabulary:
  Layer 1  CPU 香 pre-tag   — high-precision exact-match of named ids (M-*, R\\d+, agent-ids, lexicon)
  Layer 2  LLM extract+resolve+cite (stub|openai) — read the ask IN ITS THREAD WINDOW → memes
           {(have,want):{text,ref,tier,evidence}, op, maturity}; resolve each endpoint, citing a
           verbatim span or marking :unsupported (the SFC2b defeasible check → avoids 間 false-salience)
  Layer 3  CPU dedup        — exact-merge on (have.ref, want.ref, op) (op in key: self-edges don't merge)

Backend convention mirrors sfc_symbol_grounding.py (the SFC2b template): vLLM OpenAI endpoint
(OPENAI_BASE_URL, default http://localhost:8000/v1; model mark4-70b). Runs here with --backend stub
(no GPU; exercises Layers 1+3 + the plumbing) and on the 4-GPU Linode (README-linode.md) with --backend openai.

  futon6/.venv/bin/python scripts/meme_mine_runner.py --backend stub --limit 12
  # on box, after linode-4gpu-setup.sh: --backend openai   (all ~6.4k human→agent asks)
"""
import argparse, glob, json, os, re, hashlib, urllib.request, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcript_provenance import is_operator   # the ONE operator/agent/harness test (E-patch leak fix)

HOME = os.path.expanduser("~")
OUT = f"{HOME}/code/futon6/data/meme-mine"
ASK = re.compile(r"\b(can you|could you|please|let'?s|i want|we want|i need|we need|let me|build|fix|add|implement|write|update|create|check|run|show me|wire|port|mine|how do|what about|why (is|are|does)|make (it|sure|a)|should we|do we)\b", re.I)
DROP = re.compile(r"^(ok|okay|sure|thanks|thank you|yes|no|yep|nice|cool|great|sounds good)\b[\s.!,]*$", re.I)
WRAP = re.compile(r"User message:\s*(.*)$", re.S)
CALLER = re.compile(r"Caller:\s*(\S+)")
AUTO_CALLERS = {"auto-bellback", "auto", "system", "cron", "heartbeat"}

# --- Layer 1: high-precision named-id registry (香 salience, exact match only) ---
ID_PATTS = [("hole", re.compile(r"\bR\d+[a-z]?\b")),
            ("agent", re.compile(r"\b(?:claude|codex)-\d+\b", re.I)),
            ("mission", re.compile(r"\bM-[a-z0-9][a-z0-9-]{3,}\b"))]
LEXICON = {"agency": "component/agency", "neo4j": "tech/neo4j", "pgvector": "tech/pgvector",
           "lean": "tech/lean", "vllm": "tech/vllm", "xtdb": "tech/xtdb", "drawbridge": "component/drawbridge",
           "substrate-2": "component/substrate-2", "war machine": "component/war-machine",
           "linode": "infra/linode", "superpod": "infra/superpod", "codex": "agent/codex", "git blame": "op/git-blame"}


def layer1_tags(text):
    hits = {}
    for kind, patt in ID_PATTS:
        for m in patt.findall(text):
            hits[f"{kind}/{m if kind == 'mission' else m.lower()}"] = m
    low = text.lower()
    for term, ref in LEXICON.items():
        if term in low:
            hits[ref] = term
    return hits


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(isinstance(x, dict) and x.get("type") == "tool_result" for x in content):
            return None
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text")
    return None


def read_asks(limit, window=4):
    """Human→agent asks WITH a thread window (finding C); auto/system callers excluded (finding /noise)."""
    asks = []
    for f in sorted(glob.glob(f"{HOME}/.claude/projects/*/*.jsonl")):
        proj = os.path.basename(os.path.dirname(f))
        sess = os.path.basename(f)[:8]
        recent = []
        for line in open(f, errors="replace"):
            if '"type"' not in line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = o.get("type")
            if t not in ("user", "assistant"):
                continue
            txt = text_of(o.get("message", {}).get("content"))
            if not txt or not txt.strip():
                continue
            txt = txt.strip()
            if t == "assistant":
                recent = (recent + [f"assistant: {txt[:400]}"])[-window:]
                continue
            # user turn: unwrap the surface-contract preamble; provenance via the shared classifier.
            body = txt
            wm = WRAP.search(txt)
            if wm:
                body = wm.group(1).strip()
            one = body.replace("\n", " ").strip()
            if not is_operator(o):   # operator/agent/harness — the ONE shared test (was AUTO_CALLERS + a partial regex; missed promptSource sdk/system bells)
                recent = (recent + [f"user(non-op): {one[:200]}"])[-window:]
                continue
            if 40 < len(one) < 600 and not DROP.match(one) and ASK.search(one):
                asks.append({"id": "ask-" + hashlib.sha1(one.encode()).hexdigest()[:8],
                             "project": proj, "session": sess, "ask": one, "thread": list(recent)})
                if limit and len(asks) >= limit:
                    return asks
            recent = (recent + [f"user: {one[:200]}"])[-window:]
    return asks


# --- Layer 2: the extract+resolve+cite prompt, and the two backends ---
INSTR = """You mine ONE human→agent chat turn into MEMES. A meme reframes an ASK as a (have, want) arrow:
HAVE = the precondition/context in hand; WANT = the desired outcome; OP = the operation verb (lowercase, e.g.
build/fix/dispatch/find/deploy/investigate/reuse/relate/assign/preregister). Use THREAD CONTEXT to resolve
references like "it". Extract 1-2 memes. For EACH endpoint (have, want), resolve it:
 - if it names a REGISTRY HINT id -> set "ref" to that id, "tier":"named";
 - else if a referent is identifiable in context -> set "ref" to a short normalized id, "tier":"contextual",
   and "evidence" MUST be a verbatim span copied from the ASK or CONTEXT;
 - else (pure operation, or unresolvable) -> "ref":null, "tier":"unsupported".
Output STRICT JSON only: {"memes":[{"have":{"text":..,"ref":..,"tier":..,"evidence":..},
"want":{...},"op":..,"maturity":"open|correlated|constructed","salience_terms":[..]}]}"""


def _sanitize_json_escapes(s):  # 70B sometimes emits raw LaTeX/path backslashes that break json.loads
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)


def call_openai(ask, thread, hints, model):
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    hint_str = "; ".join(f"{span!r}->{ref}" for ref, span in hints.items()) or "(none)"
    user = f"REGISTRY HINTS: {hint_str}\nTHREAD CONTEXT (recent last):\n" + ("\n".join(thread) or "(none)") + f"\nASK:\n{ask}"
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "system", "content": INSTR}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(_sanitize_json_escapes(m.group(0)))["memes"] if m else []


def call_stub(ask, thread, hints):
    """No-GPU plumbing stub: exercises Layers 1+3. have.ref = first named hint (else :unsupported);
    want :unsupported; op = first ASK verb. NOT a real extraction — validates the pipeline only."""
    named = next(iter(hints.items()), None)
    VERBS = {"build", "fix", "add", "implement", "write", "update", "create", "check", "run", "wire",
             "port", "mine", "dispatch", "deploy", "find", "redeploy", "reload", "commit", "review"}
    verb = next((t for t in re.findall(r"[a-z]+", ask.lower()) if t in VERBS), "act")
    have = ({"text": named[1], "ref": named[0], "tier": "named", "evidence": named[1]}
            if named else {"text": "(context)", "ref": None, "tier": "unsupported", "evidence": None})
    return [{"have": have, "want": {"text": "(stub)", "ref": None, "tier": "unsupported", "evidence": None},
             "op": verb, "maturity": "open", "salience_terms": list(hints.values())[:4]}]


def verbatim(span, hay):
    if not span:
        return False
    n = lambda s: re.sub(r"\s+", " ", s.lower())
    return n(span)[:60] in n(hay)


def evidence_check(memes, ask, thread, hints):
    """Defeasible: a 'contextual' endpoint must cite a verbatim span; a 'named' endpoint must be a real
    hint. Else downgrade to :unsupported (SFC2b discipline — no fabricated referents / 間 false-salience)."""
    hay = ask + " " + " ".join(thread)
    for mm in memes:
        for end in ("have", "want"):
            e = mm.get(end) or {"text": "", "ref": None, "tier": "unsupported", "evidence": None}
            if e.get("tier") == "named" and e.get("ref") not in hints:
                e.update(tier="unsupported", ref=None)
            elif e.get("tier") == "contextual" and not verbatim(e.get("evidence"), hay):
                e.update(tier="unsupported", ref=None)
            mm[end] = e
    return memes


def dedup_report(rm):
    tier = Counter(); keyed = defaultdict(list); anchors = Counter(); ops = Counter()
    for m in rm:
        mm = m["meme"]; ops[mm["op"]] += 1
        for end in ("have", "want"):
            e = mm[end]; tier[e["tier"]] += 1
            if e["tier"] == "named" and e["ref"]:
                anchors[e["ref"]] += 1
        h, w = mm["have"]["ref"], mm["want"]["ref"]
        if h and w:
            keyed[(h, w, mm["op"])].append(m["id"])
    ne = sum(tier.values()) or 1
    print(f"memes: {len(rm)}  |  op-vocab: " + " ".join(f"{o}×{c}" for o, c in ops.most_common()))
    print("endpoint tiers (%d): " % ne + " · ".join(f"{k} {v} ({v/ne:.0%})" for k, v in tier.most_common()))
    coll = {k: v for k, v in keyed.items() if len(v) > 1}
    print(f"dedupable: {sum(len(v) for v in keyed.values())}; unique (have,want,op) keys: {len(keyed)}; collisions: {len(coll)}")
    print(f"named anchors (unification points): {dict(anchors)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="mark4-70b")
    ap.add_argument("--limit", type=int, default=0, help="0 = all human→agent asks")
    ap.add_argument("--window", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    asks = read_asks(a.limit or None, a.window)
    print(f"asks (auto/system excluded, thread-windowed): {len(asks)}  backend={a.backend}\n")
    rm, errs = [], 0
    for s in asks:
        hints = layer1_tags(s["ask"] + " " + " ".join(s["thread"]))
        try:
            memes = call_openai(s["ask"], s["thread"], hints, a.model) if a.backend == "openai" \
                else call_stub(s["ask"], s["thread"], hints)
        except Exception as e:
            errs += 1
            print(f"  ! {s['id']} extraction failed: {e}")
            continue
        memes = evidence_check(memes, s["ask"], s["thread"], hints)
        for mm in memes:
            rm.append({"id": s["id"], "ask": s["ask"], "provenance": {"project": s["project"], "session": s["session"]},
                       "meme": {**mm, "op": str(mm.get("op", "act")).lower(),
                                "maturity": mm.get("maturity", "open"), "salience_terms": mm.get("salience_terms", [])}})
    out = f"{OUT}/resolved-memes.{a.backend}.json"
    json.dump(rm, open(out, "w"), indent=2)
    if errs:
        print(f"  ({errs} extraction errors)")
    print(f"wrote {out}\n")
    dedup_report(rm)


if __name__ == "__main__":
    main()
