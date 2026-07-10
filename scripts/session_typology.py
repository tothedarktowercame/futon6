#!/usr/bin/env python3
"""DETERMINISTIC TYPOLOGY + SPOTTER — the rigorous core for M-points-de-fuite's "light annotation".

The session-overview chips so far come from the HEAVY mined runs (forward memes / backward C-entries) —
rich but non-deterministic and paid. The mission's graded table (M-points-de-fuite §"do as much as
possible very lightly") says the live markup should instead be a DETERMINISTIC spotter over a CONTROLLED
vocabulary (NNexus-style), with the mining repositioned as the VALIDATION ORACLE that taught us the
op-vocabulary and "which acts are recognizable lightly vs need a model."

This module:
  1. normalise_typology() — folds the mined op-vocabulary into a CLOSED, controlled typology (the act-types,
     each with a glyph/colour and a determinism TIER: explicit > recognized > cued > mined).
  2. spot() — recognizes those types in a turn's raw text with NO model: mission mention/clock (the
     M-autoclock-in rule), pattern refs, the CORRECTION_CUE / reach cue lexicons, explicit !{…} mint signs.
  3. measure() — scores the deterministic spotter against the mined labels for a session: mention-recall
     (Layer-0 recognition), correction-cue recall & precision (Layer-1 — quantifies where override is needed).

  futon6/.venv/bin/python scripts/session_typology.py [--session-id UUID]
  emits data/c-vector/typology.json (the controlled vocabulary the live markup spots against) + a report.
"""
import json, os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_scope_view import parse_turns, _norm

HERE = os.path.dirname(os.path.abspath(__file__))
FWD = os.path.join(HERE, "../data/meme-mine/joint-memes.openai.json")
BWD = os.path.join(HERE, "../data/c-vector/c-entries.openai.json")

# --- Controlled vocabularies that already exist on disk (NNexus discipline: spot against a real set) ---
MISSION_RE = re.compile(r"\b([ME]-[a-z][a-z0-9-]{3,})\b")


def mission_vocab():
    """The controlled mission/excursion set: every [ME]-*.md across the futon holes dirs."""
    v = set()
    for p in glob.glob(os.path.expanduser("~/code/futon*/holes/**/[ME]-*.md"), recursive=True):
        v.add(os.path.basename(p)[:-3])
    return v


def pattern_vocab():
    """The controlled pattern set: every *.flexiarg basename in the libraries."""
    v = set()
    for p in glob.glob(os.path.expanduser("~/code/futon*/library/**/*.flexiarg"), recursive=True):
        v.add(os.path.splitext(os.path.basename(p))[0])
    return v


# --- The light deterministic cue lexicons (Layer 1) — CORRECTION_CUE is the SAME regex the miner used
# as a hint (c_mine_joint.py), here PROMOTED to a deterministic candidate-spotter. ---
CORRECTION_CUE = re.compile(r"\b(not only|not just|not that|actually|no,|nope|isn'?t|wrong|instead|"
                            r"rather|i'?d say|let'?s not|don'?t|shouldn'?t|the issue is|too)\b", re.I)
REACH_CUE = re.compile(r"\b(let'?s|we could|we should|shall we|i'?ll|we can|maybe we|how about|"
                       r"what if|i think we|we want|the goal is|next we)\b", re.I)
# Explicit signs — the writable hidden layer (mission §DSL): emit the sign, don't pay to infer it.
DSL_MINT = re.compile(r"!\{[^}]+\}")
GLYPHS = re.compile(r"[香應咅鹽間専專蒲團]")


def normalise_typology(fwd, bwd):
    """Fold the mined op-vocabulary into a CLOSED, controlled typology. Each act-type carries a
    determinism TIER: explicit (verbatim sign) > recognized (controlled-vocab match) > cued (light
    lexicon, a candidate) > mined (needs the model). This is the normalisation Joe asked for: the raw
    mined ops become a finite, colour-stable vocabulary the markup can spot deterministically."""
    from collections import Counter
    fwd_ops = Counter(m.get("op") for r in fwd for m in (r.get("memes") or []) if m.get("op"))
    bwd_ops = Counter((r.get("flavour"), (r.get("preferred") or {}).get("op")) for r in bwd)
    return {
        "tiers": ["explicit", "recognized", "cued", "mined"],
        "types": [
            {"type": "dsl-mint", "glyph": "!", "colour": "#16a34a", "tier": "explicit",
             "recognizer": "!{A -> B : op} verbatim — minted at the point of flight"},
            {"type": "mission-clock", "glyph": "⊙", "colour": "#0f766e", "tier": "recognized",
             "recognizer": "first on-disk [ME]-… mentioned in the session (M-autoclock-in clocks the first mention)"},
            {"type": "mission-mention", "glyph": "○", "colour": "#475569", "tier": "recognized",
             "recognizer": "on-disk [ME]-… that is NOT the clocked one — mentioned, not clocked-in"},
            {"type": "pattern-ref", "glyph": "◇", "colour": "#7c3aed", "tier": "recognized",
             "recognizer": "*.flexiarg name present in the library"},
            {"type": "reach", "glyph": "◀", "colour": "#7c5cff", "tier": "cued",
             "recognizer": "REACH_CUE proposal lexicon — a light candidate; mining/override confirms",
             "mined_ops": {op: c for (fl, op), c in bwd_ops.items() if fl == "reach"}},
            {"type": "correction", "glyph": "✎", "colour": "#b45309", "tier": "cued",
             "recognizer": "CORRECTION_CUE lexicon — a light candidate; precision is LOW (see measure) so this "
                           "is exactly where the thin explicit-symbol override layer earns its keep",
             "mined_ops": {op: c for (fl, op), c in bwd_ops.items() if fl == "correction"}},
            {"type": "build", "glyph": "▶", "colour": "#0f766e", "tier": "mined",
             "recognizer": "forward meme op — NOT lightly recognizable from text alone; the model's job",
             "mined_ops": dict(fwd_ops)},
        ],
    }


def spot(text, clocked):
    """Deterministic spots in TEXT (no model). CLOCKED is the session's clocked mission (first mention),
    used to split mission tokens into clock vs mention. Returns a list of {type, token, tier}."""
    mv, pv = spot.mvocab, spot.pvocab
    spots = []
    for m in DSL_MINT.findall(text):
        spots.append({"type": "dsl-mint", "token": m[:40], "tier": "explicit"})
    for tok in dict.fromkeys(MISSION_RE.findall(text)):
        if tok in mv:
            if tok == clocked:
                spots.append({"type": "mission-clock", "token": tok, "tier": "recognized"})
            else:
                spots.append({"type": "mission-mention", "token": tok, "tier": "recognized"})
    for tok in set(re.findall(r"\b([a-z][a-z0-9-]{4,})\b", text)):
        if tok in pv:
            spots.append({"type": "pattern-ref", "token": tok, "tier": "recognized"})
    if CORRECTION_CUE.search(text):
        spots.append({"type": "correction", "token": "(cue)", "tier": "cued"})
    if REACH_CUE.search(text):
        spots.append({"type": "reach", "token": "(cue)", "tier": "cued"})
    if GLYPHS.search(text):
        spots.append({"type": "glyph", "token": "".join(dict.fromkeys(GLYPHS.findall(text))), "tier": "explicit"})
    return spots


def clocked_mission(ops, mv):
    """The M-autoclock-in rule: the session is clocked into the FIRST on-disk mission mentioned."""
    for o in ops:
        for tok in MISSION_RE.findall(o["full"]):
            if tok in mv:
                return tok
    return None


def measure(ops, sess, mv):
    """Score the DETERMINISTIC spotter against the MINED labels (the validation oracle).
    Quantifies the graded table: where the light path recovers the heavy path, and where it can't."""
    fwd = [r for r in json.load(open(FWD)) if (r.get("provenance") or {}).get("session", "") == sess]
    bwd = [r for r in json.load(open(BWD)) if (r.get("provenance") or {}).get("session", "") == sess]
    corr = [r for r in bwd if r["flavour"] == "correction"]
    # Layer-0: of the missions the mining grounded, how many are LITERALLY mentioned (deterministic)?
    mined_refs = set()
    for r in fwd:
        for m in (r.get("memes") or []):
            for side in ("have", "want"):
                ref = (m.get(side) or {}).get("ref") or ""
                if ref.startswith("mission/"):
                    mined_refs.add(ref.split("/")[-1])
    for r in bwd:
        ref = (r.get("outcome_ref") or {}).get("referent") or ""
        if ref.startswith("mission/"):
            mined_refs.add(ref.split("/")[-1])
    session_text = " ".join(o["full"] for o in ops)
    mentioned = {tok for tok in MISSION_RE.findall(session_text) if tok in mv}
    recovered = mined_refs & mentioned
    # Layer-1: correction-cue recall (of mined corrections, how many reply_spans fire the cue?)
    cue_hits = sum(1 for r in corr
                   if CORRECTION_CUE.search((r.get("provenance") or {}).get("reply_span") or ""))
    # Layer-1 precision: of OPERATOR turns firing the cue, how many are a mined correction?
    corr_bodies = [_norm((r.get("provenance") or {}).get("reply_span") or "")[:50] for r in corr]
    op_cue_turns = [o for o in ops if o["role"] == "op" and CORRECTION_CUE.search(o["full"])]
    op_cue_is_corr = sum(1 for o in op_cue_turns
                         if any(cb and cb in _norm(o["full"]) for cb in corr_bodies))
    return {
        "mined_mission_refs": len(mined_refs),
        "mention_recall": [len(recovered), len(mined_refs)],
        "missed_refs": sorted(mined_refs - mentioned),
        "correction_cue_recall": [cue_hits, len(corr)],
        "correction_cue_precision": [op_cue_is_corr, len(op_cue_turns)],
    }


def main():
    args = sys.argv[1:]
    sid = None
    if "--session-id" in args:
        sid = args[args.index("--session-id") + 1]
    if sid:
        path = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))[0]
    else:
        path = max(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")), key=os.path.getmtime)
    sess = os.path.basename(path)[:8]
    mv, pv = mission_vocab(), pattern_vocab()
    spot.mvocab, spot.pvocab = mv, pv
    ops, agents = parse_turns(path)
    fwd = [r for r in json.load(open(FWD)) if (r.get("provenance") or {}).get("session", "") == sess]
    bwd = [r for r in json.load(open(BWD)) if (r.get("provenance") or {}).get("session", "") == sess]

    typ = normalise_typology(fwd, bwd)
    out = os.path.join(HERE, "../data/c-vector/typology.json")
    json.dump(typ, open(out, "w"), ensure_ascii=False, indent=1)
    # The controlled vocabulary as a data file the live spotter (session-mode.el) loads — one source of truth.
    vout = os.path.join(HERE, "../data/c-vector/spot-vocab.json")
    json.dump({"missions": sorted(mv), "patterns": sorted(pv)}, open(vout, "w"), ensure_ascii=False)

    clk = clocked_mission(ops, mv)
    m = measure(ops, sess, mv)
    print(f"controlled vocab: {len(mv)} missions · {len(pv)} patterns   →  wrote {os.path.relpath(out, HERE)}")
    print(f"typology: {len(typ['types'])} act-types over tiers {typ['tiers']}")
    print(f"session {sess}: clocked-into (first mention) = {clk}")
    mr, mt = m["mention_recall"]
    print(f"  Layer-0 mention recall (deterministic): {mr}/{mt} mined mission-refs are literally mentioned")
    if m["missed_refs"]:
        print(f"    missed (mined but never literally named): {m['missed_refs']}")
    cr, ct = m["correction_cue_recall"]
    cp, cpt = m["correction_cue_precision"]
    print(f"  Layer-1 correction-cue recall:    {cr}/{ct} mined corrections fire CORRECTION_CUE")
    print(f"  Layer-1 correction-cue precision: {cp}/{cpt} cue-firing operator turns are mined corrections")
    print(f"  → reading: mentions recognize deterministically; corrections need the override symbol (low precision)")


if __name__ == "__main__":
    main()
