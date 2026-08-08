#!/usr/bin/env python3
"""IATC argument-graph -> CLean producer (E-clean next-step 3).

The superpod/Linode runs LLaMA (no Claude/Codex). LLaMA already produces IATC
argument-graphs (mark3_iatc_loop). This turns those graphs into CLean
mechanically, leaving exactly ONE thing for LLaMA to type: the method tag per box.

Mechanical (deterministic) — everything except the method tag:
  inference edge          -> CLean box
    :premise(s)           -> :consumes [claim ids]
    :conclusion           -> :produces  claim id
    :warrant missing      -> :hole {:kind :sorry :discharge :sorryProof :satiety :payoff :wanted …}
    :warrant provided     -> :discharges {:to <slug>}
  claim flows A->B        -> :clean/wires (skip self-loops; they're vacuous steps)
  topological/edge order  -> :clean/seq

LLM-typed (the box-typing LLaMA does) — `:method` per box + `:clean/shape :macro`,
constrained to `clean-method-vocab.edn`. The producer emits the prompt for that.

Usage:
  # 1. mechanical skeleton (untyped methods) + the LLaMA typing prompt
  futon6/.venv/bin/python scripts/iatc_to_clean.py GRAPH.edn --out OUT.clean.edn --emit-prompt OUT.prompt.txt
  # 2. inject LLaMA's returned typing (a JSON {"box": "method", "_macro": "…"})
  futon6/.venv/bin/python scripts/iatc_to_clean.py GRAPH.edn --apply typing.json --out OUT.clean.edn
"""
import argparse
import json
import os
import re
import edn_format as edn
from clean_structure_embed import kw

VOCAB_GLOSS = {
    "construct-auxiliary-object": "build a helper object the rest runs on",
    "reduce-to-known-result": "discharge to a named theorem",
    "quotient-by-irrelevance": "mod out a symmetry / pass to a quotient",
    "local-to-global": "extend a local property to the whole space",
    "transport-along-symmetry": "move a value across the space via an invariance",
    "argue-by-contradiction": "assume the negation; derive an impossibility",
    "count-by-decomposition": "partition into classes/orbits and count",
    "compute-invariant": "compute an invariant that forces/obstructs a conclusion",
    "divisibility-or-parity": "push a divisibility/parity constraint to a conclusion",
    "induct-up-a-tower": "iterate a base step up a chain/quotient tower",
    "cover-and-estimate": "cover the target and bound the total measure",
    "epsilon-of-room": "ε–δ / tail control; conclude by arbitrariness of ε",
}
MACROS = ["construct-exploit-discharge", "count-invariant-obstruct", "cover-estimate",
          "contradiction-reduce", "induct-tower"]


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s[:48] or "warrant"


def cid(x):
    """Claim-id -> CLean keyword name. Integer ids -> c<N>; keyword ids kept."""
    return f"c{x}" if isinstance(x, int) else kw(x)


def bid(x):
    """Edge-id -> CLean box-id name. Integer ids -> e<N>; keyword ids kept."""
    return f"e{x}" if isinstance(x, int) else kw(x)


def as_list(p):
    """Normalize a :premise that may be a scalar id or an (edn) list of ids."""
    if p is None:
        return []
    if isinstance(p, int) or type(p).__name__ == "Keyword":
        return [p]
    return list(p)


def _edn_safe(text):
    """edn_format (strict) rejects tokens that bb (the lenient reader which PRODUCED
    these graphs) accepts, so a graph can pass the bb gates and still be unreadable
    here. Two known classes, both fixed ONLY outside double-quoted strings so :text
    stays verbatim; both substitutions are global and deterministic, so ids and the
    edge refs pointing at them stay aligned:

      ' in symbols/keywords (:phi', common in CT primes)  -> 'prime'
      non-ASCII in symbols/keywords (:hom->cone with a real arrow, :mu-natural)
        -> 'u<hex codepoint>'  (E-superpod-hardening H12, 2026-08-06: 5/98 graphs
        on the Zone e2e run carried unicode ids and were dropped at S7)"""
    from edn_compat import edn_safe   # single source of truth (H12)
    return edn_safe(text)


def load_graph(path):
    m = edn.loads(_edn_safe(open(path).read()))
    d = {kw(k): v for k, v in dict(m).items()}
    nodes = {}
    for n in d.get("nodes", []):
        nd = {kw(k): v for k, v in dict(n).items()}
        nodes[nd["id"]] = str(nd.get("text", ""))
    edges = []
    for e in d.get("edges", []):
        ed = {kw(k): v for k, v in dict(e).items()}
        if kw(ed.get("kind")) != "infer":
            continue
        if "id" not in ed or "conclusion" not in ed:
            continue   # malformed infer-edge: 70B omitted a required field (passes bb, would KeyError) -> skip
        prem = as_list(ed.get("premise"))
        w = ed.get("warrant")
        try:
            warrant = {kw(k): v for k, v in dict(w).items()} if w is not None else {}
        except (TypeError, ValueError):
            warrant = {}   # 70B sometimes emits a bare keyword as :warrant (passes bb, breaks dict()) -> treat as hole
        # A vector :conclusion means the edge establishes SEVERAL claims at once:
        # a biconditional's two directions (:iff), or a lemma with several
        # consequences (:implies / :therefore). Six such edges exist corpus-wide.
        # `premise` was already normalised with as_list; `conclusion` was not, so
        # cid() received a list, rendered its Python repr into the EDN
        # (":produces :[Keyword(a), Keyword(b)]") and the file failed
        # clean_argcheck's G1 "unreadable EDN". Those six proofs were then dropped
        # from the typed corpus and logged as "not a DAG comb (e.g.
        # cyclic-equivalence)" — a guessed reason that was wrong in every case,
        # and which sent a later investigation looking for a cycle disagreement
        # that did not exist.
        #
        # Fanning out preserves both the meaning and the DAG: one inference to N
        # conclusions is N inferences sharing premises and warrant. Ids are
        # suffixed only when there is more than one, so the single-conclusion
        # case — every other edge in the corpus — is byte-identical to before.
        concl = as_list(ed["conclusion"])
        for i, c in enumerate(concl):
            eid = ed["id"] if len(concl) == 1 else f"{ed['id']}__c{i}"
            edges.append({"id": eid, "premise": prem, "conclusion": c,
                          "warrant": warrant})
    return nodes, edges


def build_skeleton(nodes, edges, typing=None):
    """Return the CLean dict. typing: optional {box_id: method, '_macro': macro}."""
    typing = typing or {}
    boxes = []
    for e in edges:
        box_id = bid(e["id"])
        consumes = [cid(p) for p in e["premise"]]
        produces = cid(e["conclusion"])
        ctext = nodes.get(e["conclusion"], "")
        box = {"id": box_id, "consumes": consumes, "produces": produces,
               "text": ctext[:160],
               "method": typing.get(box_id, "untyped-step")}
        wk = kw(e["warrant"].get("kind")) if e["warrant"] else None
        if wk == "missing-warrant":
            box["hole"] = {"satiety": "payoff", "discharge": "sorryProof",
                           "wanted": kw(e["warrant"].get("wanted", "warrant"))}
        elif wk == "claim":
            box["discharges_to"] = slug(e["warrant"].get("text", "warrant"))
        boxes.append(box)

    # wires from claim flow (skip self-loops)
    produced_by = {}
    for b in boxes:
        produced_by.setdefault(b["produces"], []).append(b["id"])
    wires = []
    for b in boxes:
        for c in b["consumes"]:
            for src in produced_by.get(c, []):
                if src != b["id"]:
                    wires.append({"from": src, "to": b["id"], "carries": c})

    seq = [b["method"] for b in boxes]
    holes_at = [b["id"] for b in boxes if "hole" in b]
    disch_at = [b["id"] for b in boxes if "discharges_to" in b]
    macro = typing.get("_macro", "untyped-macro")
    return {"boxes": boxes, "wires": wires, "seq": seq,
            "holes_at": holes_at, "disch_at": disch_at, "macro": macro}


def render_edn(pid, sk, vacuous):
    """Render the skeleton dict as CLean EDN text (gate-compatible)."""
    L = [";; CLean produced from an IATC argument-graph by scripts/iatc_to_clean.py.",
         ";; Mechanical except :method / :clean/shape :macro, which LLaMA types",
         ";; (see the .prompt.txt). typing-source recorded below."]
    if vacuous:
        L.append(f";; NOTE: dropped {vacuous} self-loop inference(s) (vacuous step, X⇒X).")
    typed = not any(b["method"] == "untyped-step" for b in sk["boxes"])
    L.append("{:clean/proof " + edn_str(pid))
    L.append(" :clean/source {:iatc \"" + pid + "\"}")
    L.append(" :clean/typing-source " + (":llama" if typed else ":iatc-stub"))
    L.append(" :clean/seq [" + " ".join(":" + m for m in sk["seq"]) + "]")
    L.append(" :clean/boxes")
    L.append(" [" + "\n  ".join(render_box(b) for b in sk["boxes"]) + "]")
    L.append(" :clean/wires")
    if sk["wires"]:
        L.append(" [" + "\n  ".join(
            "{:from :%s :to :%s :carries :%s}" % (w["from"], w["to"], w["carries"])
            for w in sk["wires"]) + "]")
    else:
        L.append(" []")
    L.append(" :clean/copar [{:reading :informal :is :clean/seq}")
    L.append("               {:reading :formal   :is [:clean/boxes :clean/wires]}]")
    L.append(" :clean/shape {:macro :" + sk["macro"])
    L.append("               :holes-at [" + " ".join(":" + h for h in sk["holes_at"]) + "]")
    L.append("               :discharges-at [" + " ".join(":" + d for d in sk["disch_at"]) + "]}}")
    return "\n".join(L)


def edn_str(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') + '"'


def render_box(b):
    parts = ["{:id :%s :method :%s" % (b["id"], b["method"])]
    parts.append("   :text " + edn_str(b["text"]))
    if b["consumes"]:
        parts.append("   :consumes [" + " ".join(":" + c for c in b["consumes"]) + "]")
    parts.append("   :produces :" + b["produces"])
    if "hole" in b:
        h = b["hole"]
        parts.append("   :hole {:kind :sorry :discharge :%s :satiety :%s :wanted %s}"
                     % (h["discharge"], h["satiety"], edn_str(h["wanted"])))
    if "discharges_to" in b:
        parts.append("   :discharges {:to :%s}" % b["discharges_to"])
    return "\n".join(parts) + "}"


def emit_prompt(pid, nodes, edges, sk):
    L = [f"You are typing the reasoning steps of a mathematical proof (arXiv {pid}).",
         "Assign each STEP exactly one METHOD tag from this controlled vocabulary:",
         ""]
    for m, g in VOCAB_GLOSS.items():
        L.append(f"  {m}  — {g}")
    L += ["", "And assign ONE overall MACRO shape from:",
          "  " + " | ".join(MACROS), "",
          "The steps (each is an inference from its premises to its conclusion):", ""]
    cm = {bid(e["id"]): e for e in edges}
    for b in sk["boxes"]:
        e = cm[b["id"]]
        prem = "; ".join(nodes.get(p, "")[:90] for p in e["premise"])
        L.append(f"  [{b['id']}] FROM: {prem}")
        L.append(f"        THEN: {nodes.get(e['conclusion'],'')[:120]}")
        if "hole" in b:
            L.append(f"        (open obligation: {b['hole']['wanted']})")
        L.append("")
    L += ["Return ONLY JSON: {\"<box-id>\": \"<method-tag>\", ..., \"_macro\": \"<macro>\"}.",
          "Use only tags from the vocabulary above."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit-prompt", default=None)
    ap.add_argument("--apply", default=None, help="JSON typing to inject")
    args = ap.parse_args()

    pid = os.path.basename(args.graph).replace(".edn", "")
    nodes, edges = load_graph(args.graph)
    vacuous = sum(1 for e in edges if all(p == e["conclusion"] for p in e["premise"]))

    typing = json.load(open(args.apply)) if args.apply else None
    sk = build_skeleton(nodes, edges, typing)

    with open(args.out, "w") as fh:
        fh.write(render_edn(pid, sk, vacuous) + "\n")
    typed = bool(typing)
    print(f"wrote {args.out}  ({len(sk['boxes'])} boxes, {len(sk['wires'])} wires, "
          f"{len(sk['holes_at'])} holes, {len(sk['disch_at'])} discharges, "
          f"typing={'llama' if typed else 'stub'})")

    if args.emit_prompt:
        with open(args.emit_prompt, "w") as fh:
            fh.write(emit_prompt(pid, nodes, edges, sk) + "\n")
        print(f"wrote {args.emit_prompt}  (LLaMA box-typing prompt)")


if __name__ == "__main__":
    main()
