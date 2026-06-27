#!/usr/bin/env python3
"""SESSION-SCOPE VIEW — the M-points-de-fuite prototype: a session as a tree of WEAK SCOPES.

Joe's reframe (2026-06-26): a turn IS a scope, weakly typed by the tags it touches; a session's structure
is not missing, just weak. This renders that structure NOW — independent of substrate-2 (whose mission-scope
entities are still empty) — as a foldable org tree, the donor-shape for session-mode.el. Re-run to refresh
"live" as the session grows. Operator turns only (the steering acts), grouped into thematic SUB-ARCS by
shared tags; each turn a leaf scope with its tags. Untyped turns (no tag) are the out-of-band/meta moves.

  futon6/.venv/bin/python scripts/session_scope_view.py [SESSION.jsonl] [OUT.org] [--session-id UUID]
  default: the most-recently-written ~/.claude/projects session → futon2/holes/session-scope-view.org

--session-id pins to a specific session (resolves ~/.claude/projects/*/<UUID>.jsonl) — the robust
"this session" signal the Emacs panel passes (mtime-max ties between concurrently-written peer sessions).
"""
import json, os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcript_provenance import classify, raw_text

CONCEPTS = ["belly", "C-vector", "mining", "gate", "golden", "correction", "sigil", "Arxana", "flight",
            "fuite", "control layer", "substrate", "PROOF", "Pilot", "weav", "蒲團", "recognition",
            "scope", "MAP", "DERIVE", "IDENTIFY", "VERIFY", "INSTANTIATE", "belly", "reflow", "concentration"]


def _known_docs():
    """Real on-disk mission/excursion/campaign doc names (M-/E-/C-*.md), so the comb
    recognises excursions+campaigns too — and a bare regex can't mint phantom teeth
    like 'E-mail' (intersect with this set, the way session-mode.el cross-checks)."""
    names = set()
    for pat in ("M-*.md", "E-*.md", "C-*.md"):
        for f in (glob.glob(f"/home/joe/code/*/holes/{pat}")
                  + glob.glob(f"/home/joe/code/*/holes/**/{pat}", recursive=True)):
            names.add(os.path.basename(f)[:-3])
    return names


_KNOWN_DOCS = _known_docs()


def _scope(body, role, idx):
    # M-/E-/C- mentions that are REAL docs (intersect with _KNOWN_DOCS — no phantoms)
    miss = sorted(set(re.findall(r"\b[MEC]-[a-z][a-z0-9-]{3,}\b", body)) & _KNOWN_DOCS)
    tags = [c for c in dict.fromkeys(CONCEPTS) if re.search(re.escape(c), body, re.I)]
    return {"role": role, "idx": idx, "head": body.replace("\n", " ")[:60],
            "full": body[:1200], "tags": tags, "miss": miss,
            "forward": [], "correction": [], "reach": []}


def parse_turns(path):
    """All steering/voice turns in transcript order: operator turns (the human's steering acts) and
    agent turns (the 應-voice). Each is a weak scope; `idx` gives the shared timeline so reach scopes
    interleave where they fire (M-points-de-fuite v2: in-flow recognition is the same for both voices)."""
    ops, agents, n = [], [], 0
    for line in open(path, errors="replace"):
        if '"type"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get("type")
        if typ == "user" and classify(o) == "operator":
            t = raw_text(o) or ""
            m = re.search(r"User message:\s*(.*)$", t, re.S)
            body = (m.group(1) if m else t).strip()
            role = "op"
        elif typ == "assistant":
            body = (raw_text(o) or "").strip()
            role = "agent"
        else:
            continue
        if len(body) < 12:
            continue
        sc = _scope(body, role, n)
        n += 1
        (ops if role == "op" else agents).append(sc)
    return ops, agents


def turn_scopes(path):
    return parse_turns(path)[0]


def _norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def attach_mined(ops, agents, sess, fwd_path, bwd_path):
    """Reflow the turns against the MINED War-Machine field (M-points-de-fuite).

    The forward run (memes/methods) and backward run (C-entries/belly) over the transcript carry
    `provenance.session`; their verbatim spans rejoin the turns they were mined FROM. Operator acts
    annotate OPERATOR turns: a forward meme (the `ask` IS this turn) → a ▶build move; a correction
    (its `reply_span` is in this turn) → a ✎steer move. Reach is 應-voice — the AGENT's act — so it
    annotates AGENT turns (v2): each reach's `assistant_span` rejoins the agent turn that voiced it,
    pinning the ◀reach where it actually fires rather than aggregating it in the header.

    Returns a session-level `mined` summary; mutates ops (forward/correction) and agents (reach)."""
    from collections import Counter

    def load(path):
        try:
            recs = json.load(open(path))
        except Exception:
            return []
        return [r for r in recs if (r.get("provenance") or {}).get("session", "") == sess]

    fwd, bwd = load(fwd_path), load(bwd_path)
    # Index operator turns by a normalized prefix of their body (forward `ask` == full turn).
    oidx = {}
    for o in ops:
        oidx.setdefault(_norm(o["full"])[:60], o)
    for r in fwd:
        o = oidx.get(_norm(r.get("ask"))[:60])
        if o is not None:
            for m in (r.get("memes") or []):
                ref = (m.get("have") or {}).get("ref") or (m.get("want") or {}).get("ref")
                o["forward"].append({"op": m.get("op"), "ref": ref, "tier": (m.get("have") or {}).get("tier")})
    # Corrections: reply_span is a substring of the operator turn.
    obodies = [(o, _norm(o["full"])) for o in ops]
    for r in bwd:
        if r["flavour"] != "correction":
            continue
        sp = _norm((r.get("provenance") or {}).get("reply_span"))[:50]
        if not sp:
            continue
        for o, body in obodies:
            if sp in body:
                o["correction"].append({"op": (r.get("preferred") or {}).get("op"),
                                        "ref": (r.get("outcome_ref") or {}).get("referent")})
                break
    # Reach: assistant_span rejoins the agent turn that voiced it (v2 — agent acts as scopes).
    abodies = [(a, _norm(a["full"])) for a in agents]
    n_reach_matched = 0
    for r in bwd:
        if r["flavour"] != "reach":
            continue
        sp = _norm((r.get("provenance") or {}).get("assistant_span"))[:60]
        if not sp:
            continue
        for a, body in abodies:
            if sp in body:
                a["reach"].append({"op": (r.get("preferred") or {}).get("op"),
                                   "ref": (r.get("outcome_ref") or {}).get("referent")})
                n_reach_matched += 1
                break
    reach = [r for r in bwd if r["flavour"] == "reach"]
    built = Counter(f["ref"] for o in ops for f in o["forward"] if f["ref"])
    wanted = Counter(rr["ref"] for a in agents for rr in a["reach"] if rr["ref"])
    steered = Counter(c["ref"] for o in ops for c in o["correction"] if c["ref"])
    return {"n_forward": sum(len(o["forward"]) for o in ops),
            "n_reach": len(reach),
            "n_reach_pinned": n_reach_matched,
            "n_correction": sum(len(o["correction"]) for o in ops),
            "built": built.most_common(8),
            "wanted": wanted.most_common(8),
            "steered": steered.most_common(6)}


def attach_threads(ops, full_sid):
    """Number each operator turn (1-based, the context-retrieval turn id the threads use) and attach
    its THREAD membership — keeps the linear list CURRENT and thread-interleaved, with no dependence
    on the stale mining (recent turns get thread chips even where mining never ran)."""
    import urllib.request
    try:
        d = json.load(urllib.request.urlopen(
            f"http://localhost:7070/api/alpha/evidence?tag=context-retrieval&session-id={full_sid}&limit=500",
            timeout=10))
    except Exception:
        return
    rets = {}
    for e in d.get("entries", []):
        b = e.get("evidence/body", {}) or {}
        t = b.get("turn")
        if t is not None and t not in rets:
            rets[t] = b.get("query") or ""
    turn_threads = {}
    try:
        th = json.load(open("/home/joe/code/futon2/holes/session-threads.json"))
        for h in th.get("thread-hyperedges", []):
            sg = (h["sigil"]["truth"] + h["sigil"]["okipona"]).strip() or "·"
            short = h["pattern"].split("/")[-1]
            for end in h["ends"]:
                turn_threads.setdefault(end["turn"], []).append({"sigil": sg, "pattern": short})
    except Exception:
        pass
    obodies = [(o, _norm(o["full"])) for o in ops]
    for turn_num in sorted(rets):
        q = _norm(rets[turn_num])[:40]
        if not q:
            continue
        for o, body in obodies:
            if "turn_num" not in o and q in body:
                o["turn_num"] = turn_num
                o["threads"] = turn_threads.get(turn_num, [])
                break


def recognise_moves(turns, basins_path):
    """Reproducible ▶build/◀reach/✎steer per turn from the mining-distilled move basins
    (mining_recogniser.py) — embed each turn, assign its nearest basin.  No LLM, no paid run;
    works for any session.  Skips gracefully if the basins or the encoder aren't available."""
    if not os.path.exists(basins_path):
        return
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception:
        return
    b = json.load(open(basins_path))
    classes = b["classes"]
    C = np.array([b["centroids"][c] for c in classes])
    try:
        model = SentenceTransformer(b.get("model", "all-MiniLM-L6-v2"))
        V = model.encode([(t["full"] or "")[:400] for t in turns],
                         normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        return
    sims = V @ C.T
    for i, t in enumerate(turns):
        j = int(np.argmax(sims[i]))
        t["move"] = classes[j]
        t["move_score"] = round(float(sims[i][j]), 3)


def sub_arcs(ops):
    """Group consecutive operator turns into thematic sub-arcs (a run sharing >=1 tag; untyped turns
    continue the current arc — they are the meta/out-of-band moves inside it)."""
    arcs, cur, cur_tags = [], [], set()
    for s in ops:
        tagset = set(s["tags"]) | set(s["miss"])
        if cur and tagset and not (tagset & cur_tags):
            arcs.append(cur); cur, cur_tags = [], set()
        cur.append(s); cur_tags |= tagset
    if cur:
        arcs.append(cur)
    return arcs


def _short(ref):
    return ref.split("/")[-1] if ref else None


def org(ops, arcs, sess, mined=None):
    from collections import Counter
    field = Counter(m for s in ops for m in s["miss"]).most_common(8)
    L = [f"#+TITLE: Session scope view — {sess} (live · {len(ops)} operator scopes)",
         "#+TODO: SCOPE | GHOST",
         f"# regenerate: futon6/scripts/session_scope_view.py   ·   the weak-scope reflow (M-points-de-fuite)",
         "",
         f"* Session {sess}   :session:"]
    if mined:
        L.append("  :built: " + " ".join(f"{_short(r)}({c})" for r, c in mined["built"]))
        L.append("  :wanted: " + " ".join(f"{_short(r)}({c})" for r, c in mined["wanted"]))
        L.append(f"  :field: ▶{mined['n_forward']} build · ◀{mined['n_reach']} reach · ✎{mined['n_correction']} steer")
    else:
        L.append("  :concentration-field: " + " ".join(f"{m}({c})" for m, c in field))
    for i, arc in enumerate(arcs, 1):
        from collections import Counter as C
        at = C(t for s in arc for t in (s["tags"] + s["miss"]))
        label = at.most_common(1)[0][0] if at else "untyped"
        tagstr = ":" + ":".join(t.replace(" ", "-") for t, _ in at.most_common(3)) + ":" if at else ""
        L.append(f"** SCOPE arc {i}: {label}  ({len(arc)} turns)  {tagstr}")
        for s in arc:
            tg = (":" + ":".join((s["tags"] + s["miss"])[:4]).replace(" ", "-") + ":") if (s["tags"] or s["miss"]) else ":out-of-band:"
            mv = "".join(f" ▶{f['op']}:{_short(f['ref'])}" for f in s.get("forward", [])) \
               + "".join(f" ✎{c['op']}:{_short(c['ref'])}" for c in s.get("correction", [])) \
               + "".join(f" ◀{r['op']}:{_short(r['ref'])}" for r in s.get("reach", []))
            glyph = "應 " if s.get("role") == "agent" else ""
            L.append(f"*** {glyph}«{s['head']}…»   {tg}{mv}")
    return "\n".join(L) + "\n"


def mission_pivot(out_arcs):
    """The Comb / pivot layer (E-the-dark-tower-2): a long session drifts mission-to-mission, so the
    flat timeline hides the real (non-linear) structure. Re-axis it: each mission is a comb TOOTH whose
    SPINE runs across the arcs, lit where the session touched it and by which move (▶build/◀reach/✎steer).
    A mission touched early, dropped, and resumed shows two lit clusters — the drift made visible."""
    n = len(out_arcs)
    miss = {}

    def bump(ref, kind, ai):
        if not ref:
            return
        d = miss.setdefault(ref, {"build": [0] * n, "reach": [0] * n, "steer": [0] * n, "mention": [0] * n})
        d[kind][ai] += 1

    for ai, arc in enumerate(out_arcs):
        for t in arc["turns"]:
            for f in t["forward"]:
                bump(f["ref"], "build", ai)
            for c in t["correction"]:
                bump(c["ref"], "steer", ai)
            for r in t["reach"]:
                bump(r["ref"], "reach", ai)
            # CURRENT signal (deterministic M-mention) — keeps the comb up to date even where the
            # paid mining never ran; mined move-types (▶◀✎) overlay it where they exist.
            for m in (t.get("missions") or []):
                bump(m, "mention", ai)
    rows = []
    for ref, d in miss.items():
        spine = []
        for ai in range(n):
            if d["build"][ai]:
                spine.append("build")
            elif d["steer"][ai]:
                spine.append("steer")
            elif d["reach"][ai]:
                spine.append("reach")
            elif d["mention"][ai]:
                spine.append("mention")
            else:
                spine.append(None)
        lit = [ai for ai in range(n) if spine[ai]]
        rows.append({"mission": ref, "spine": spine,
                     "n_build": sum(d["build"]), "n_reach": sum(d["reach"]), "n_steer": sum(d["steer"]),
                     "n_mention": sum(d["mention"]),
                     "total": sum(d["build"]) + sum(d["reach"]) + sum(d["steer"]) + sum(d["mention"]),
                     "span": [lit[0], lit[-1]] if lit else None})
    # Missions, excursions AND campaigns (M-/E-/C-…) are comb teeth; drop file/pattern refs.
    rows = [r for r in rows if (r["mission"] or "")[:2] in ("M-", "E-", "C-")]
    rows.sort(key=lambda r: (-r["total"], r["span"][0] if r["span"] else n))
    return {"n_arcs": n, "rows": rows}


def to_json(ops, arcs, sess, path, mined=None):
    """The donor-shape for session-overview.el: session → sub-arcs → leaf turns.
    Mirrors the org tree but machine-readable, with full turn bodies for RET/help-echo,
    and the mined WM field per turn (forward ▶build / correction ✎steer) + session-level."""
    from collections import Counter
    field = Counter(m for s in ops for m in s["miss"]).most_common(8)
    out_arcs = []
    for arc in arcs:
        at = Counter(t for s in arc for t in (s["tags"] + s["miss"]))
        out_arcs.append({
            "label": at.most_common(1)[0][0] if at else "untyped",
            "tags": [t for t, _ in at.most_common(3)],
            "n_forward": sum(len(s.get("forward", [])) for s in arc),
            "n_correction": sum(len(s.get("correction", [])) for s in arc),
            "n_reach": sum(len(s.get("reach", [])) for s in arc),
            "turns": [{"head": s["head"], "full": s["full"], "role": s.get("role", "op"),
                       "turn_num": s.get("turn_num"), "threads": s.get("threads", []),
                       "move": s.get("move"), "move_score": s.get("move_score"),
                       "tags": (s["tags"] + s["miss"])[:4] or ["out-of-band"],
                       "missions": s["miss"],
                       "forward": [{"op": f["op"], "ref": _short(f["ref"])} for f in s.get("forward", [])],
                       "correction": [{"op": c["op"], "ref": _short(c["ref"])} for c in s.get("correction", [])],
                       "reach": [{"op": r["op"], "ref": _short(r["ref"])} for r in s.get("reach", [])]}
                      for s in arc],
        })
    out = {"session": sess, "path": path, "n_ops": len(ops),
           "field": [[m, c] for m, c in field], "arcs": out_arcs,
           "pivot": mission_pivot(out_arcs)}
    if mined:
        out["mined"] = {
            "n_forward": mined["n_forward"], "n_reach": mined["n_reach"],
            "n_reach_pinned": mined["n_reach_pinned"],
            "n_correction": mined["n_correction"],
            "built": [[_short(r), c] for r, c in mined["built"]],
            "wanted": [[_short(r), c] for r, c in mined["wanted"]],
            "steered": [[_short(r), c] for r, c in mined["steered"]]}
    return out


def main():
    args = [a for a in sys.argv[1:]]
    sid = None
    if "--session-id" in args:
        i = args.index("--session-id")
        sid = args[i + 1]
        del args[i:i + 2]
    if args and args[0].endswith(".jsonl"):
        path = args.pop(0)
    elif sid:
        cand = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))
        if not cand:
            sys.stderr.write(f"session_scope_view: no transcript for session-id {sid}\n")
            sys.exit(2)
        path = max(cand, key=os.path.getmtime)
    else:
        path = max(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")), key=os.path.getmtime)
    out = args[0] if args else "/home/joe/code/futon2/holes/session-scope-view.org"
    sess = os.path.basename(path)[:8]
    ops, agents = parse_turns(path)
    here = os.path.dirname(os.path.abspath(__file__))
    mined = attach_mined(ops, agents, sess,
                         os.path.join(here, "../data/meme-mine/joint-memes.openai.json"),
                         os.path.join(here, "../data/c-vector/c-entries.openai.json"))
    # CURRENT: 1-based turn numbers + per-turn thread membership (no mining dependency).
    attach_threads(ops, os.path.basename(path)[:-6])
    # The weak-scope timeline: operator turns + reach-bearing agent turns, in transcript order.
    reach_scopes = [a for a in agents if a["reach"]]
    scopes = sorted(ops + reach_scopes, key=lambda s: s["idx"])
    # REPRODUCIBLE ▶◀✎ per turn from the mining-distilled basins (replaces the mining dependency).
    recognise_moves(scopes, os.path.join(here, "../data/c-vector/move-basins.json"))
    arcs = sub_arcs(scopes)
    open(out, "w").write(org(scopes, arcs, sess, mined))
    jpath = os.path.splitext(out)[0] + ".json"
    json.dump(to_json(scopes, arcs, sess, path, mined), open(jpath, "w"), ensure_ascii=False, indent=1)
    print(f"session {sess}: {len(ops)} operator + {len(reach_scopes)} reach scopes → {len(arcs)} sub-arcs")
    print(f"  mined WM field: ▶{mined['n_forward']} build · ◀{mined['n_reach']} reach "
          f"({mined['n_reach_pinned']} pinned) · ✎{mined['n_correction']} steer")
    print(f"wrote {out}  (open in Emacs org-mode; fold/unfold the arc tree)")
    print(f"wrote {jpath}  (donor-shape for session-overview.el)")


if __name__ == "__main__":
    main()
