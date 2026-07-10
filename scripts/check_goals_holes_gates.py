#!/usr/bin/env python3
"""Gate-checker for the BACKWARD goals-and-holes miner (C-entries) — the belly's go/no-go bands as code.

The dual of check_meme_mine_gates.py (which gates the forward memes). Runs over any C-entry file — a smoke
output, the full run, OR a curated GOLDEN set dumped in C-entry shape. Author (golden curator) ≠ reviewer
(this checker): a golden that FAILS is mis-curated and would teach the 70B the error. Pure CPU.

  futon6/.venv/bin/python scripts/check_goals_holes_gates.py data/c-vector/c-entries.openai.json
  futon6/.venv/bin/python scripts/check_goals_holes_gates.py <golden-centries.json>

Exit 0 iff all HARD gates pass. Bands: holes/goals-holes-readiness.html + the correction-over-fire review.
"""
import json, re, sys
from collections import Counter

PIVOT = re.compile(r"\b(not only|not that|instead|rather than|rather|no,|actually|too abstract|the issue is|"
                   r"don'?t|shouldn'?t|isn'?t|wrong|by example rather|reverse|drop that|never mind|"
                   # precedence/sequencing redirects (a correction can interpose a different FIRST step):
                   r"before we|before that|before doing|hold on|wait,|step back|let'?s first)\b", re.I)
AGREE_OPEN = re.compile(r"^\s*(yes|ok|okay|sure|sounds good|great|yeah|yep|of course|let'?s|will do|agreed)\b", re.I)
LEAK = re.compile(r"(claude-\d|codex-\d)\s*(?:→|->)|task-notification|reply with exactly|caller:\s*(claude|codex)", re.I)

def band(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok

# Shape-tolerant accessors: read either the STORED artifact shape (spans under provenance, grounded_ref
# under provenance — post to_c_entry) OR the RAW-LLM / GOLDEN shape (spans under evidence, grounded_ref
# top-level — what the INSTR schema emits and a curator writes). The gate must judge both identically.
def aspan(e): return ((e.get("provenance") or {}).get("assistant_span") or (e.get("evidence") or {}).get("assistant_span") or "")
def rspan(e): return ((e.get("provenance") or {}).get("reply_span") or (e.get("evidence") or {}).get("reply_span") or "")
def gref(e):  return (e.get("provenance") or {}).get("grounded_ref") or e.get("grounded_ref")

def check(records):
    n = len(records)
    fl = Counter(e.get("flavour") for e in records)
    reach = [e for e in records if e.get("flavour") == "reach"]
    corr  = [e for e in records if e.get("flavour") == "correction"]
    print(f"== C-ENTRY gates over {n} records · flavour {dict(fl)} ==")

    # I1 — every C-entry must cite its required verbatim span (reach→assistant_span, correction→reply_span).
    no_ev = [e for e in records if (e.get("flavour") == "correction" and not rspan(e).strip())
             or (e.get("flavour") == "reach" and not aspan(e).strip())]
    # correction must name a redirected target (preferred.value present, not null/trivial)
    no_target = [e for e in corr if not str((e.get("preferred") or {}).get("value") or "").strip()
                 or len(str((e.get("preferred") or {}).get("value"))) < 4]
    # correction precision: a Stage-B-verified correction (provenance.verified=="override", from
    # c_mine_joint's focused override|other pass) is trusted; only UNVERIFIED corrections fall back to the
    # keyword proxy (a genuine pivot, not agreement-open). This aligns the gate to the runner's discriminator
    # so it stops over-flagging genuine implicit pivots once the runner verifies them.
    def _verified(e): return (e.get("provenance") or {}).get("verified") == "override"
    suspect = [e for e in corr if not _verified(e) and ((not PIVOT.search(rspan(e))) or AGREE_OPEN.match(rspan(e)))]
    leaks = sum(1 for e in records if LEAK.search(rspan(e) + aspan(e)))
    grounded = sum(1 for e in reach if gref(e))
    corr_share = len(corr) / n if n else 0
    susp_rate = len(suspect) / len(corr) if corr else 0
    tgt_rate = len(no_target) / len(corr) if corr else 0
    gr_rate = grounded / len(reach) if reach else 0

    print(f"  reach grounded to a mission/pattern: {grounded}/{len(reach)} = {gr_rate*100:.0f}% (informational)")
    if no_ev: print(f"  missing-evidence examples: {[ (e.get('flavour'), (aspan(e) or rspan(e))[:40]) for e in no_ev[:3] ]}")
    if suspect: print(f"  suspect corrections: {[rspan(e)[:60] for e in suspect[:3]]}")

    hard = []
    hard.append(band("leak-free", leaks == 0, f"{leaks} inter-agent/harness spans (want 0)"))
    hard.append(band("I1-evidence", not no_ev, f"{len(no_ev)} C-entries missing their verbatim span (want 0)"))
    hard.append(band("reach>=correction", corr_share <= 0.5, f"correction share {corr_share*100:.0f}% (want <=50%)"))
    hard.append(band("correction-precision", susp_rate <= 0.2, f"{len(suspect)}/{len(corr)} corrections non-genuine (want <=20%)"))
    hard.append(band("correction-target", tgt_rate <= 0.2, f"{len(no_target)}/{len(corr)} corrections lack a named redirect (want <=20%)"))
    return all(hard)

def main():
    if len(sys.argv) < 2:
        print("usage: check_goals_holes_gates.py <c-entries-or-golden.json>"); sys.exit(2)
    ok_all = True
    for path in sys.argv[1:]:
        records = json.load(open(path))
        if not isinstance(records, list) or not records or "flavour" not in records[0]:
            print(f"{path}: not a C-entry file (expected a list of records with 'flavour')"); ok_all = False; continue
        print(f"\n### {path}")
        ok = check(records)
        print(f"  => {'GATES PASS' if ok else 'GATES FAIL'}")
        ok_all = ok_all and ok
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
