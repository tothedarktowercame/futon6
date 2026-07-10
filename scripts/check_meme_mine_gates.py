#!/usr/bin/env python3
"""Gate-checker for the mining artifacts — the reusable go/no-go bands as code (not prose).

Runs over ANY file in the artifact shapes — a smoke output, a full run, OR a curated GOLDEN set whose
ideal-outputs are dumped in the same shape. Author (golden curator) ≠ reviewer (this checker): a golden
that FAILS a gate is mis-curated (would teach the 70B the error); a hand-verified-good golden that fails
reveals the gate is too strict. Pure CPU, no GPU.

  futon6/.venv/bin/python scripts/check_meme_mine_gates.py data/meme-mine/joint-memes.openai.json
  futon6/.venv/bin/python scripts/check_meme_mine_gates.py data/c-vector/c-entries.openai.json
  futon6/.venv/bin/python scripts/check_meme_mine_gates.py <golden.json>   # auto-detects kind

Exit 0 iff all HARD gates pass. Bands derive from holes/meme-mine-runner-spec.md + the F2/F3/F4 review.
"""
import json, re, sys
from collections import Counter

DISCOURSE = {"elaborate","contrast","compare","request","clarify","describe","discuss","acknowledge",
             "evaluate","summarize","summarise","analyze","analyse","consider","explain","note","reflect"}
# the move-class allowlist (the INSTR's operational vocabulary); ops outside this AND not "none" are off-spec
OPERATIONAL = {"build","create","add","update","fix","wire","port","mine","implement","write","extend",
               "refine","execute","run","deploy","dispatch","find","investigate","reuse","relate","assign",
               "preregister","reconstruct","review","verify","commit","commission",
               # the model's wider (legitimate) operational vocabulary, observed in the 400-record run:
               "close","construct","derive","compile","validate","formalise","formalize","identify","install",
               "merge","extract","seed","model","transform","rewrite","schedule","send","read","select",
               "persist","recover","refresh","rebuild","reset","retire","rehome","replicate","scrape","log",
               "iterate","insert","instantiate","join","audit","backfill","activate","adapt","coordinate",
               "draft","drive","expose","improve","initiate","experiment","clean","kill","pair","bank","bell"}
PIVOT = re.compile(r"\b(not only|not that|instead|rather than|rather|no,|actually|too abstract|the issue is|"
                   r"don'?t|shouldn'?t|isn'?t|wrong|by example rather|reverse|drop that|never mind)\b", re.I)
AGREE_OPEN = re.compile(r"^\s*(yes|ok|okay|sure|sounds good|great|yeah|yep|of course|let'?s|will do|agreed)\b", re.I)

def band(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok

def memes_of(records):
    out = []
    for r in records:
        if isinstance(r.get("memes"), list): out += r["memes"]
        elif isinstance(r.get("meme"), dict): out.append(r["meme"])
    return out

def _str(x):  # the model occasionally emits a list/obj where a scalar id/op was asked for
    return x if isinstance(x, (str, type(None))) else (str(x[0]) if isinstance(x, list) and x else str(x))

def check_meme(records):
    print(f"== MEME gates over {len(records)} records ==")
    memes = memes_of(records)
    ops = Counter(_str(m.get("op")) for m in memes)
    disc = sum(c for o,c in ops.items() if o in DISCOURSE)
    oper = sum(c for o,c in ops.items() if o and o not in DISCOURSE and o != "none")
    tiers = Counter(e.get("tier") for m in memes for e in (m.get("have"), m.get("want")) if e)
    has_np = "new_patterns" in records[0] if records else False
    npr = (sum(1 for r in records if r.get("new_patterns")) / len(records)) if (records and has_np) else None
    print(f"  ops: {dict(ops.most_common(10))}")
    offspec = sorted({o for o in ops if o and o != "none" and o not in DISCOURSE and o not in OPERATIONAL})
    if offspec:
        print(f"  [warn] off-spec ops (not discourse, not in move-class allowlist): {offspec} — tighten INSTR if systematic")
    print(f"  tiers: {dict(tiers)}")
    # candidate-aware tier-grounding (F4): a tier="named" endpoint is GENUINE only if its ref is among the
    # record's retrieved candidates OR is literally mentioned in the ask; otherwise it is FORCED grounding.
    tail = lambda x: str(x).split("/")[-1].lower()
    have_cands = any("candidates" in r for r in records)
    named_total = forced = 0; forced_ex = []
    if have_cands:
        for r in records:
            c = r.get("candidates") or {}
            cand_ids = set(c.get("missions") or []) | set(c.get("patterns") or [])
            cand_tails = {tail(x) for x in cand_ids}
            ask = (r.get("ask") or "").lower()
            for m in (r.get("memes") or []):
                for e in (m.get("have"), m.get("want")):
                    if not e or e.get("tier") != "named": continue
                    named_total += 1
                    ref = _str(e.get("ref"))
                    if ref and (ref in cand_ids or tail(ref) in cand_tails or tail(ref) in ask):
                        continue
                    forced += 1; forced_ex.append(ref)
    total_ops = sum(ops.values()) or 1
    disc_share = disc / total_ops
    hard = []
    hard.append(band("op-discourse-share (F3)", disc_share <= 0.02,
                     f"{disc}/{total_ops} discourse-verb ops = {disc_share*100:.1f}% (want <=2%, catches drift not noise); {oper} operational"))
    if npr is not None:
        hard.append(band("new_patterns-rate (F2)", npr <= 0.25, f"{npr*100:.0f}% of records propose >=1 (want <=25%)"))
    else:
        print("  [info] new_patterns: not present in this shape (flat resolved-memes) — skipped")
    if have_cands and named_total:
        if forced_ex: print(f"  forced-named refs (not candidate/ask-backed): {forced_ex[:5]}")
        fr = forced / named_total
        hard.append(band("tier-grounding (F4)", fr <= 0.3, f"{forced}/{named_total} named refs forced (want <=30%)"))
    else:
        print("  [info] tier-grounding: no per-record candidates in this shape — skipped")
    return all(hard)

def check_centry(records):
    print(f"== C-ENTRY gates over {len(records)} records ==")
    fl = Counter(e.get("flavour") for e in records)
    corr = [e for e in records if e.get("flavour") == "correction"]
    corr_share = len(corr) / len(records) if records else 0
    def span(e): pv = e.get("provenance") or {}; return (pv.get("reply_span") or "")
    leak = re.compile(r"(claude-\d|codex-\d)\s*(?:→|->)|task-notification|reply with exactly|caller:\s*(claude|codex)", re.I)
    leaks = sum(1 for e in records if leak.search(span(e) + ((e.get("provenance") or {}).get("assistant_span") or "")))
    suspect = [e for e in corr if (not PIVOT.search(span(e))) or AGREE_OPEN.match(span(e))]
    susp_rate = len(suspect) / len(corr) if corr else 0
    print(f"  flavour: {dict(fl)}")
    if suspect:
        print(f"  suspect corrections (no pivot / agreement-open):")
        for e in suspect[:4]: print(f"     {span(e)[:80]!r}")
    hard = []
    hard.append(band("leak-free", leaks == 0, f"{leaks} inter-agent/harness spans (want 0)"))
    hard.append(band("reach>=correction", corr_share <= 0.5, f"correction share {corr_share*100:.0f}% (want <=50%)"))
    hard.append(band("correction-precision", susp_rate <= 0.2, f"{len(suspect)}/{len(corr)} corrections look non-genuine (want <=20%)"))
    return all(hard)

def main():
    if len(sys.argv) < 2:
        print("usage: check_meme_mine_gates.py <artifact-or-golden.json>"); sys.exit(2)
    ok_all = True
    for path in sys.argv[1:]:
        records = json.load(open(path))
        if not isinstance(records, list) or not records:
            print(f"{path}: empty/unexpected shape"); ok_all = False; continue
        kind = "centry" if "flavour" in records[0] else "meme"
        print(f"\n### {path}  (kind={kind})")
        if kind == "centry":
            print("  [note] for C-entries the richer/canonical validator is check_goals_holes_gates.py"
                  " (adds I1-evidence + correction-target). Running the basic centry bands here:")
        ok = check_centry(records) if kind == "centry" else check_meme(records)
        print(f"  => {'GATES PASS' if ok else 'GATES FAIL'}")
        ok_all = ok_all and ok
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
