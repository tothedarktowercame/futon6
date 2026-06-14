#!/usr/bin/env python3
"""paper_title_card.py — dramatis personae + plot summary for a paper.

Joe's golden-round finding (2026-06-12): the anatomy has organs but no
face. The title card answers at-a-glance: concept signature (top canon
terms), theorem census, per-theorem fingerprints (bound symbols + canon
concepts within each statement's span, after M-canon-fingerprint-store's
Billey-Tenner reading), and the environment-flow plot.
"""
import json
import re
from collections import Counter
from pathlib import Path

NER = Path("/home/joe/code/storage/mark2/ct-handoff/output/ner-terms.json")
_ner_cache = None


def paper_terms(entity_id):
    global _ner_cache
    if _ner_cache is None:
        _ner_cache = {r["entity_id"]: r.get("terms", [])
                      for r in json.load(open(NER))}
    return _ner_cache.get(entity_id, [])


def concept_signature(entity_id, text, top=12):
    counts = Counter()
    for t in paper_terms(entity_id):
        canon = t.get("canon") or t.get("term")
        surface = t.get("term_lower") or t.get("term", "").lower()
        if surface and canon:
            counts[canon] += text.lower().count(surface)
    return counts.most_common(top)


THEOREM_KINDS = ("theorem", "proposition", "lemma", "corollary", "conjecture")


def theorem_fingerprints(entity_id, text, scopes, tex_envs):
    terms = [(t.get("term_lower") or "", t.get("canon") or t.get("term"))
             for t in paper_terms(entity_id)]
    binder_spans = [(s, (s.get("hx/content") or {}))
                    for s in scopes
                    if str(s.get("hx/type", "")).startswith(("bind/", "constrain/"))]
    cards = []
    for env in tex_envs:
        kind = env["hx/type"].removeprefix("env-tex/")
        if kind not in THEOREM_KINDS:
            continue
        c = env["hx/content"]
        lo, hi = c["position"], c["end"]
        body = text[lo:hi]
        symbols = []
        for s, sc in binder_spans:
            if sc.get("position") is not None and lo <= sc["position"] < hi:
                sym = next((e.get("latex") for e in (s.get("hx/ends") or [])
                            if e.get("role") == "symbol"), None)
                typ = next((e.get("text") for e in (s.get("hx/ends") or [])
                            if e.get("role") == "type"), None)
                if sym:
                    symbols.append((sym, (typ or "").strip()[:40]))
        low = body.lower()
        canons = sorted({canon for surf, canon in terms if surf and surf in low})
        cards.append({"kind": kind, "position": lo,
                      "statement_head": re.sub(r"\s+", " ", body[:120]).strip(),
                      "bound_symbols": symbols[:8],
                      "canons": canons[:10]})
    return cards


def plot_summary(tex_envs):
    seq = [e["hx/type"].removeprefix("env-tex/")
           for e in sorted(tex_envs, key=lambda e: e["hx/content"]["position"])]
    out, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        out.append(f"{seq[i]}×{j-i}" if j - i > 1 else seq[i])
        i = j
    return " → ".join(out)
