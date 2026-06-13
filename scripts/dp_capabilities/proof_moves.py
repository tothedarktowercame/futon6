"""Informal proof-move detection for DP paper views."""

from __future__ import annotations

import re

PROOF_MOVES = [
    (r"it is not (?:difficult|hard) to (?:check|see|verify|show|prove)",
     "deferred verification: “not difficult to check”", "deferred-verification"),
    (r"it is (?:easy|straightforward|routine|immediate|trivial) to "
     r"(?:check|see|verify|show|prove)",
     "deferred verification: “easy to see”", "deferred-verification"),
    (r"it is (?:clear|obvious|evident|immediate)(?:\s+from[^.,;]{0,40})? that",
     "deferred verification: “it is clear that”", "deferred-verification"),
    (r"(?:one|the reader) (?:can|may) (?:easily |readily )?"
     r"(?:check|verify|see|show)",
     "deferred verification: “one can check”", "deferred-verification"),
    (r"(?:by )?a (?:direct|routine|straightforward|simple|short) "
     r"(?:computation|calculation|verification|argument|check)",
     "deferred verification: routine computation", "deferred-verification"),
    (r"(?:is|are) (?:easily|readily) (?:seen|checked|verified|shown)",
     "deferred verification: “readily checked”", "deferred-verification"),
    (r"(?:we )?leave[^.,;]{0,30} to the reader|left to the reader",
     "deferred verification: left to the reader", "deferred-verification"),
    (r"the (?:proofs?|verifications?|details?|computations?) (?:is|are) "
     r"(?:omitted|left|straightforward|routine|similar|analogous)",
     "deferred verification: details omitted", "deferred-verification"),
    (r"\bclearly\b|\bobviously\b|\btrivially\b",
     "hedge adverb", "deferred-verification"),
    (r"it (?:suffices|is enough) to (?:show|prove|check|consider|find)",
     "reduction: it suffices to show", "sufficiency-reduction"),
    (r"without loss of generality|\bWLOG\b|\bw\.l\.o\.g\.|by symmetry",
     "WLOG / symmetry reduction", "wlog-symmetry"),
]
PROOF_MOVES_C = [(re.compile(p, re.I), lbl, fam) for p, lbl, fam in PROOF_MOVES]


def detect_proof_moves(ftext, base):
    out = []
    for rx, label, family in PROOF_MOVES_C:
        for pm in rx.finditer(ftext):
            out.append({
                "start": base + pm.start(), "end": base + pm.end(),
                "layer": "dp", "kind": "proof-move",
                "tip": f"informal proof move · {label}",
                "fields": [["move", label], ["family", family],
                           ["text", pm.group(0)[:50]]],
            })
    return out
