#!/usr/bin/env python3
"""C-MINE (joint) — the BACKWARD dual of meme_mine_joint.py: mine the BELLY, not the methods.

M-goals-and-holes INSTANTIATE (BACKWARD-JOINT card). meme_mine_joint reads human→agent turns for
MEMES (methods, the policy half of EFE). This reads agent→human turns — the 應-voice — for C-ENTRIES
(prior preferences, the C-vector / R19, the preference half). Two C-signals (Joe, 2026-06-25):
  reach       — an unstated goal the AGENT is orienting toward (the assistant turn alone),
  correction  — the HUMAN REPLY overrides/redirects the agent ("not only", "not that — this"): the
                redirected target is a preference the agent's C LACKED. The CLEANEST C-signal —
                Friston's C *is* preference-over-outcomes, and a correction is that, directly. It is
                RELATIONAL: read from the (assistant-turn, human-reply) PAIR, not one role.

Same machinery as the forward joint runner — CPU retriever (concept-tag gazetteer → candidate
missions/patterns for grounding) + GPU 70B reasoner — so it can HOT-SWAP onto the standing forward-run
box (Joe: the box is already on the human→agent pass; swap the code without teardown if we're quick).
Emits the SAME C-entry shape as c_vector.bb (the shared C-ENTRY record), as JSON for the consume side.

Outputs embed verbatim turn spans → gitignored under data/*. Private transcripts stay on dev (tunnel mode).

  futon6/.venv/bin/python scripts/c_mine_joint.py --backend stub --limit 6
  # on box (vLLM, hot-swapped onto the forward run): --backend openai
"""
import argparse, glob, json, os, re, sys, hashlib, urllib.request
import concurrent.futures as cf, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meme_mine_runner import text_of, _sanitize_json_escapes, WRAP, DROP
from meme_mine_joint import registry, retrieve
from mission_concept_tag import gazetteer, spot
from transcript_provenance import is_operator   # the ONE operator/agent/harness test (E-patch leak fix)

HOME = os.path.expanduser("~")
ROOT = "/home/joe/code/futon6"; OUT = f"{ROOT}/data/c-vector"
# A cheap CPU pre-tag (香 salience, the leading edge of a correction) — a HINT for the LLM, never a gate.
CORRECTION_CUE = re.compile(r"\b(not only|not just|not that|actually|no,|nope|isn'?t|wrong|instead|"
                            r"rather|i'?d say|let'?s not|don'?t|shouldn'?t|the issue is|too)\b", re.I)
# The belly is the OPERATOR's preferences. Operator/agent/harness classification is the ONE shared test
# transcript_provenance.is_operator (E-patch-agent-evidence-leaks DERIVE-1). PLUMBING is still used
# locally to drop the AGENT's own bell-narration ASSISTANT turns (not a reach).
PLUMBING = re.compile(r"bell delivered|belled to|auto-?bellback|job-id invoke|verdict belled|"
                      r"bell sent|🔔|finished job|surface:\s*bell|\(state:", re.I)


def read_pairs(limit, window=4):
    """Agent→human turns each PAIRED with the following genuine human reply (finding the belly).
    A correction lives in the pair, not one role; auto/system replies are excluded (finding /noise)."""
    pairs = []
    for f in sorted(glob.glob(f"{HOME}/.claude/projects/*/*.jsonl")):
        proj = os.path.basename(os.path.dirname(f))
        sess = os.path.basename(f)[:8]
        recent, pending = [], None        # pending = the last assistant turn awaiting its human reply
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
                # skip the agent's own plumbing narration (bells/job-ids) — it isn't a reach.
                pending = None if PLUMBING.search(txt) else txt
                recent = (recent + [f"assistant: {txt[:200]}"])[-window:]
                continue
            # user turn: unwrap the surface preamble for the reply text; provenance via the shared test.
            body = txt
            wm = WRAP.search(txt)
            if wm:
                body = wm.group(1).strip()
            reply = body.replace("\n", " ").strip()
            is_human = is_operator(o)   # operator/agent/harness — the ONE shared classifier
            # Emit the pair only on a genuine, substantive human reply to an assistant turn.
            if pending and is_human and not DROP.match(reply) and len(reply) > 8:
                pairs.append({"id": "pair-" + hashlib.sha1((pending[:120] + reply[:120]).encode()).hexdigest()[:8],
                              "project": proj, "session": sess,
                              "assistant": pending.replace("\n", " ").strip()[:1200],
                              "reply": reply[:600], "thread": list(recent),
                              "cue": bool(CORRECTION_CUE.search(reply))})
                if limit and len(pairs) >= limit:
                    return pairs
            pending = None
            recent = (recent + [f"user: {reply[:200]}"])[-window:]
    return pairs


INSTR = """You read ONE agent→human chat turn (what the AGENT did or proposed) together with the
HUMAN'S REPLY, jointly against CANDIDATE MISSIONS (preprints) and CANDIDATE PATTERNS retrieved for it.
You mine the BELLY — prior preferences (Friston's C-vector) — NOT methods. Emit STRICT JSON only:
{"c_entries":[{"flavour":"reach|correction",
  "outcome_ref":{"kind":"goal|preference","referent":<candidate-id-or-short-normalized-id>,"metric":"satisfied|aligned"},
  "preferred":{"op":"satisfied|align","value":<for correction: the REDIRECTED target the human wants; for reach: null>},
  "grounded_ref":<a CANDIDATE mission/pattern id when it fits, else null>,
  "evidence":{"assistant_span":<verbatim span from the AGENT turn>,"reply_span":<verbatim span from the REPLY, or null for reach>}}]}
Rules:
- DEFAULT is reach or []. Most replies are NOT corrections — do not reach for "correction". A reply that
  ACCEPTS the agent's proposal, even while adding scope/detail ("Yes let's X — and also Y", "OK, and we
  could also Z"), is reach or [] — NEVER a correction. Agreement-plus-elaboration is agreement.
- reach: an unstated GOAL the agent is orienting toward (cite the assistant_span). reply_span may be null.
- correction: emit ONLY when the human is steering AWAY from the agent's specific proposal — something the
  agent suggested is being DROPPED, REVERSED, or REPLACED. ALL must hold:
    (a) the AGENT turn proposed/did something specific (not a question);
    (b) the reply explicitly REJECTS or CONTRASTS that proposal — a pivot marker ("not that", "instead",
        "rather", "no,", "actually not", "too abstract", "X by example rather than Y"). A bare "OK"/"yes"/
        "let's" is NOT a pivot;
    (c) you can name BOTH sides: what the agent proposed AND the different thing the human wants instead;
        put the latter in preferred.value as a short phrase. If the agent's proposal is being KEPT, it is
        not a correction.
  NOT corrections (→ reach or []): agreement/approval (even with added detail), continuation ("continue",
  "proceed", "let's take our time", "meanwhile let's…"), description/recap ("so we had an M-typed-bells
  mission…"), or a fresh unrelated request. When in doubt, it is NOT a correction.
- Ground to a CANDIDATE id when it fits (grounded_ref); else null. EVERY c_entry MUST cite a verbatim span
  (no span -> drop it). Emit [] if the turn carries neither a reach nor a genuine correction — [] is the
  common case; do not manufacture a correction from ordinary steering."""


_FEWSHOT_CACHE = None
def _fewshot_messages():
    """Golden few-shot exemplars prepended to every call (gitignored data/c-vector/golden-backward.json) —
    teaches the correction-vs-reach discipline (agreement/recap/continuation are NOT corrections). Validated
    2026-06-25 on a held-out correction-stress batch: correction-precision 3/6-suspect->0/5, gate FAIL->PASS.
    Loaded+cached once. Disable: FEWSHOT_OFF=1. Override: CENTRY_FEWSHOT."""
    global _FEWSHOT_CACHE
    if _FEWSHOT_CACHE is not None:
        return _FEWSHOT_CACHE
    msgs = []
    if os.environ.get("FEWSHOT_OFF") != "1":
        path = os.environ.get("CENTRY_FEWSHOT", f"{OUT}/golden-backward.json")
        try:
            for ex in json.load(open(path)):
                a = ex["input"]["agent_turn"] or ""
                h = ex["input"].get("human_reply") or "(none)"
                u = ("CANDIDATE MISSIONS:\n  (none)\nCANDIDATE PATTERNS:\n  (none)\n"
                     f"THREAD CONTEXT (recent last):\n(none)\nAGENT TURN:\n{a}\nHUMAN REPLY:\n{h}")
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(ex["ideal"])}]
        except FileNotFoundError:
            pass
    _FEWSHOT_CACHE = msgs
    return msgs


def call_openai(p, miss, pat, model):
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    cm = "\n".join(f"  {cid} — {d}" for cid, d in miss) or "  (none)"
    cp = "\n".join(f"  {cid} — {d}" for cid, d in pat) or "  (none)"
    user = (f"CANDIDATE MISSIONS:\n{cm}\nCANDIDATE PATTERNS:\n{cp}\n"
            f"THREAD CONTEXT (recent last):\n" + ("\n".join(p["thread"]) or "(none)") +
            f"\nAGENT TURN:\n{p['assistant']}\nHUMAN REPLY:\n{p['reply']}")
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "system", "content": INSTR}] + _fewshot_messages()
                                   + [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(_sanitize_json_escapes(m.group(0))) if m else {}


# --- Stage-B: correction VERIFY (the structural precision fix, per correction-precision-problem.md) ---
# Keyword cues are brittle both ways (miss implicit pivots; admit non-corrective cue-words). The real
# discriminator is CONTRAST against the agent's just-proposed action — and the model judges that reliably
# when asked NARROWLY. So a candidate `correction` is kept only if a focused second pass says "override".
VERIFY_INSTR = """You judge whether a HUMAN reply OVERRIDES the AGENT's just-proposed action.
**Almost always the answer is `other`.** Answer `override` ONLY when the human REJECTS, REVERSES, or
REPLACES a specific thing the agent proposed to do — telling it to do something DIFFERENT instead (explicit
"use X not Y", or implicit "before that, do Z first" / "actually the issue is …" that changes the plan).
Answer `other` for everything else, including: an instruction/directive that is not a reversal
("set the working directory to …", "do ARGUE as planned"); a decision AMONG options the agent itself
offered ("go with 1"); agreement/approval even with added detail; a continuation ("continue", "also …",
"meanwhile …"); an assessment that doesn't change the next action; a fresh unrelated request.
The test is strictly: is the human telling the agent to DROP / REVERSE / REPLACE what it just proposed?
If unsure, answer `other`. Reply with ONE word only: override | other."""

# Few-shot teaching the boundary on the exact failure classes (correction-precision-problem.md).
_VERIFY_FEWSHOT = [
    ("AGENT: I'll run the ARGUE phase next.\nHUMAN: Let's do ARGUE \"as planned\" per the mission_lifecycle doc, then take stock.", "other"),
    ("AGENT: Two options — (1) a tight fix, (2) a broader rewrite.\nHUMAN: Let's go with 1 — we can do a best-of-class pass later.", "other"),
    ("AGENT: Where are the files?\nHUMAN: Set working directory to ~/code/ and you'll find them.", "other"),
    ("AGENT: I'll start mining the turns now.\nHUMAN: So, before we do that, can we scope a Codex handoff first?", "override"),
    ("AGENT: We can just wing the autoclock-in inside this mission.\nHUMAN: No — make M-autoclock-in its own mission rather than winging it.", "override"),
]


def call_verify(p, model):
    """Focused, few-shot override|other second pass over a candidate correction. Conservative: anything not
    clearly 'override' → 'other' (favours precision). Returns 'override' | 'other'."""
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    shots = []
    for u, ans in _VERIFY_FEWSHOT:
        shots += [{"role": "user", "content": u}, {"role": "assistant", "content": ans}]
    user = f"AGENT (proposed/did):\n{p['assistant']}\nHUMAN REPLY:\n{p['reply']}"
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 4,
                       "messages": [{"role": "system", "content": VERIFY_INSTR}] + shots
                                   + [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    return "override" if re.match(r"\s*override\b", content, re.I) else "other"


def call_stub(p, miss, pat):
    """Plumbing/retrieval stub: a reach grounded to the first candidate mission; a correction ONLY when
    the CPU cue fired (validates pairing + cue + schema, NOT real reasoning)."""
    out = [{"flavour": "reach",
            "outcome_ref": {"kind": "goal", "referent": (miss[0][0] if miss else "(context)"), "metric": "satisfied"},
            "preferred": {"op": "satisfied", "value": None}, "grounded_ref": (miss[0][0] if miss else None),
            "evidence": {"assistant_span": p["assistant"][:60], "reply_span": None}}]
    if p["cue"]:
        out.append({"flavour": "correction",
                    "outcome_ref": {"kind": "preference", "referent": "(redirect)", "metric": "aligned"},
                    "preferred": {"op": "align", "value": p["reply"][:60]}, "grounded_ref": None,
                    "evidence": {"assistant_span": p["assistant"][:60], "reply_span": p["reply"][:60]}})
    return {"c_entries": out}


def to_c_entry(ce, p):
    """Normalize a mined c_entry into the shared C-ENTRY record (cf. c_vector.bb): provenance carries
    the verbatim spans + the source turn; discharge is left open (reach → forward methods; correction →
    adopt-the-redirect). Returns None if no verbatim evidence (I1 provenance / SFC2b defeasible check)."""
    ev = ce.get("evidence") or {}
    flavour = ce.get("flavour")
    a_span, r_span = ev.get("assistant_span"), ev.get("reply_span")
    # I1 per-flavour: a reach must cite the AGENT turn; a correction must cite the REPLY. (Fixes the
    # 1/200 reach-with-no-assistant_span bug — a reach evidenced only by a reply_span is malformed.)
    if flavour == "reach" and not (a_span and str(a_span).strip()):
        return None
    if flavour == "correction" and not (r_span and str(r_span).strip()):
        return None
    if not (a_span or r_span):
        return None
    return {"flavour": flavour, "outcome_ref": ce.get("outcome_ref"),
            "preferred": ce.get("preferred"),
            "weight": {"value": 0.3, "basis": "default-unoriented"},
            "status": "open",
            "provenance": {"source": "c_mine_joint", "channel": "ying-voice",
                           "project": p["project"], "session": p["session"], "derived_from": p["id"],
                           "grounded_ref": ce.get("grounded_ref"),
                           "assistant_span": ev.get("assistant_span"), "reply_span": ev.get("reply_span")},
            "discharged_by": ("adopt-redirect" if flavour == "correction" else None),
            "witness": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="mark4-70b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "8")),
                    help="concurrent in-flight requests (vLLM continuous-batches them; 1 = sequential). "
                         "The single biggest throughput lever — sequential left the GPU mostly idle (2026-06-26).")
    ap.add_argument("--no-verify-corrections", action="store_true",
                    help="disable the Stage-B override|other verify on candidate corrections (the precision fix).")
    a = ap.parse_args()
    verify_corr = (a.backend == "openai") and not a.no_verify_corrections  # Stage-B only meaningful with a real model
    gaz, desc = gazetteer(), registry()
    _fewshot_messages()  # warm the golden-fewshot cache ONCE before threads start (avoid a build race)
    pairs = read_pairs(a.limit or None)
    n_cue = sum(1 for p in pairs if p["cue"])
    print(f"pairs: {len(pairs)}  ({n_cue} with a correction-cue)  backend={a.backend}  "
          f"concurrency={a.concurrency}  registry: {len(desc)}\n")
    os.makedirs(OUT, exist_ok=True)

    def process(p):
        """Worker (I/O-bound → GIL released in urlopen, so threads run concurrently). Times the LLM call
        for the latency instrumentation. Returns (records, latency_s, error_or_None) — never raises."""
        t0 = time.perf_counter()
        try:
            miss, pat = retrieve(p["assistant"] + " " + p["reply"], p["thread"], gaz, desc, a.k)
            r = call_openai(p, miss, pat, a.model) if a.backend == "openai" else call_stub(p, miss, pat)
            recs = [{"id": p["id"], **e}
                    for e in (to_c_entry(ce, p) for ce in (r.get("c_entries") or [])) if e]
            vdrop = 0
            if verify_corr and any(e["flavour"] == "correction" for e in recs):
                kept = []
                for e in recs:
                    if e["flavour"] == "correction":
                        if call_verify(p, a.model) != "override":   # Stage-B: drop the non-override
                            vdrop += 1; continue
                        e["provenance"]["verified"] = "override"    # survivor: gate may trust it
                    kept.append(e)
                recs = kept
            return recs, time.perf_counter() - t0, None, vdrop
        except Exception as e:
            return [], time.perf_counter() - t0, f"{p['id']}: {e}", 0

    out, n_reach, n_corr, n_ground, n_done, last_ckpt, n_vdrop = [], 0, 0, 0, 0, 0, 0
    lats = []
    o = f"{OUT}/c-entries.{a.backend}.json"
    def pct(sorted_lats, q):  # simple percentile; sorted_lats non-empty
        return sorted_lats[min(len(sorted_lats) - 1, int(len(sorted_lats) * q))]
    # ThreadPool fans out `concurrency` requests at once; vLLM batches them on the GPU. Results come
    # back in submission order (executor.map); checkpoint + stats run here on the main thread (no locks).
    with cf.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        for recs, lat, err, vdrop in ex.map(process, pairs):
            n_done += 1; lats.append(lat); n_vdrop += vdrop
            if err:
                print(f"  ! {err}")
            for e in recs:
                n_reach += e["flavour"] == "reach"
                n_corr += e["flavour"] == "correction"
                n_ground += bool(e["provenance"]["grounded_ref"])
                out.append(e)
            # checkpoint every >=200 NEW entries (since-last counter, not % 200 — a pair can add 2; "Y2C").
            if len(out) - last_ckpt >= 200:
                json.dump(out, open(o, "w"), indent=2)
                last_ckpt = len(out)
                print(f"  .. checkpoint: {len(out)} C-entries written")
            if n_done % 100 == 0:
                s = sorted(lats)
                print(f"  [{n_done}/{len(pairs)}] req-latency med={pct(s,0.5):.1f}s p95={pct(s,0.95):.1f}s "
                      f"min={s[0]:.1f}s max={s[-1]:.1f}s | {len(out)} C-entries")
    json.dump(out, open(o, "w"), indent=2)
    s = sorted(lats) or [0.0]
    print(f"C-entries: {len(out)}  (reach {n_reach} · correction {n_corr} · grounded {n_ground})")
    if verify_corr:
        print(f"Stage-B verify: dropped {n_vdrop} candidate corrections as non-override (kept {n_corr} verified)")
    print(f"latency over {len(lats)} calls: med={pct(s,0.5):.1f}s p95={pct(s,0.95):.1f}s max={s[-1]:.1f}s")
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
