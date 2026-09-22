"""S4 output contract: the model returns schema-constrained JSON; code writes EDN.

The model classifies an expository region into scopes from the finalized arXiv
vocabulary and either fills each scope's typed hole with source-anchored text or
holds it with a reason. The JSON schema fixes the kind vocabulary (out-of-scope
kinds excluded) and bounds line numbers to the region. Code checks the rest
(ordered ranges; exactly one of fill or held reason), looks up the hole's slot
from the vocabulary, and writes the EDN scope graph that expository_argcheck and
the S6/S8/S10 readers consume. Nothing is repaired; violations reject the item.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import iatc_json
import region_units

GENERATOR = "expository-json/v2"
VOCAB = Path(__file__).resolve().parent.parent / "holes" / "excursions" / "expository-superpod-vocab.edn"
MAX_SCOPES = 8


def vocabulary(path: Path = VOCAB) -> dict[str, str]:
    """{scope kind: hole slot}, excluding kinds listed as out of scope for arXiv.

    Read with babashka, the reader expository_argcheck uses. Python edn_format parses
    `:rationale/telos/organization-roadmap` as `:rationale/telos` plus a stray symbol,
    which silently merged two kinds and gave one of them the other's slot.
    """
    program = ("(require '[cheshire.core :as json]) "
               "(let [v (clojure.edn/read-string (slurp (first *command-line-args*))) "
               "      out (set (map :iatc (:out-of-scope-arxiv v)))] "
               "  (println (json/generate-string "
               "    (into {} (for [s (:scopes v) :when (not (out (:kind s)))] "
               "      [(subs (str (:kind s)) 1) (name (get-in s [:hole :slot]))])))))")
    result = subprocess.run(["bb", "-e", program, str(path)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def schema(lo: int, hi: int, kinds, units=()) -> dict:
    """The scope shape. With units, a scope cites them and quotes from them.

    Line numbers were the only anchor a scope had, and the fill was a string the
    model typed: in mark7master-20260921 only 38% of filled scopes said anything
    that appears in the passage word for word, and a scope could only ever be shown
    a whole line at a time. Citing units gives the scope an extent, and quoting from
    them gives the fill one.
    """
    scope = {"type": "object", "additionalProperties": False,
             "required": ["kind", "units", "fill", "held_reason"],
             "properties": {"kind": {"type": "string", "enum": sorted(kinds)},
                            "units": {"type": "array", "minItems": 1, "maxItems": 3,
                                      "items": {"type": "string", "enum": [u["id"] for u in units]}},
                            "fill": {"type": "string", "maxLength": 400},
                            "held_reason": {"type": "string", "maxLength": 200}}}
    if not units:                                   # pre-v2 candidates: lines, as before
        line = {"type": "integer", "minimum": lo, "maximum": hi}
        scope["required"] = ["kind", "first_line", "last_line", "fill", "held_reason"]
        scope["properties"] = {"kind": scope["properties"]["kind"],
                               "first_line": line, "last_line": line,
                               "fill": scope["properties"]["fill"],
                               "held_reason": scope["properties"]["held_reason"]}
    return {"type": "object", "additionalProperties": False, "required": ["scopes"],
            "properties": {"scopes": {"type": "array", "minItems": 1, "maxItems": MAX_SCOPES, "items": scope}}}


def problems(doc, lo: int, hi: int, kinds, units=()) -> list[str]:
    if not isinstance(doc, dict) or not isinstance(doc.get("scopes"), list) or not doc["scopes"]:
        return ["output is not an object with a nonempty scopes list (the endpoint did not enforce the schema)"]
    found = []
    for i, s in enumerate(doc["scopes"], 1):
        if not isinstance(s, dict) or s.get("kind") not in kinds:
            found.append(f"scope {i}: kind {s.get('kind') if isinstance(s, dict) else s!r} not in the vocabulary")
            continue
        fill, held = str(s.get("fill", "")).strip(), str(s.get("held_reason", "")).strip()
        if bool(fill) == bool(held):
            found.append(f"scope {i}: needs exactly one of fill or held_reason")
        if units:
            by_id = {u["id"]: u for u in units}
            cited = [by_id[x] for x in (s.get("units") or []) if x in by_id]
            if not cited:
                found.append(f"scope {i}: cites no unit of this region")
            elif fill and region_units.locate_in_units(fill, cited) is None:
                # The fill must be the passage's words, not the model's about them.
                found.append(f"scope {i}: fill {fill[:60]!r} is not in the unit(s) it cites")
            continue
        a, b = s.get("first_line"), s.get("last_line")
        if not (isinstance(a, int) and isinstance(b, int) and lo <= a <= b <= hi):
            found.append(f"scope {i}: lines {a}-{b} not an ordered range inside {lo}-{hi}")
    return found


def to_edn(doc: dict, candidate: dict, kinds: dict[str, str], model: str) -> str:
    lo, hi = candidate["window-lines"]
    s = iatc_json.edn_string
    units = {u["id"]: u for u in candidate.get("units") or ()}
    scopes = []
    for i, scope in enumerate(doc["scopes"], 1):
        fill, held = scope["fill"].strip(), scope["held_reason"].strip()
        body = (f":slot-fill {{:{kinds[scope['kind']]} {s(fill)}}}" if fill
                else f":held {{:reason {s(held)}}}")
        if units and scope.get("units"):
            cited = [units[x] for x in scope["units"] if x in units]
            first, last = min(u["line"] for u in cited), max(u["line"] for u in cited)
            extent = [min(u["start"] for u in cited), max(u["end"] for u in cited)]
            ids = " ".join(s(u["id"]) for u in cited)
            where = (f":source {{:lines [{first} {last}] :units [{ids}] "
                     f":span [{extent[0]} {extent[1]}]}}")
            # The fill's own extent, so a reader can be shown the words, not the line.
            at = region_units.locate_in_units(fill, cited) if fill else None
            if at:
                body += f" :fill-span [{at[0]} {at[1]}]"
        else:
            where = f":source {{:lines [{scope['first_line']} {scope['last_line']}]}}"
        scopes.append(f"{{:id :s{i} :kind :{scope['kind']} {where} {body}}}")
    return ("{:paper/id " + s(candidate["paper-id"]) + "\n :passage/id " + s(candidate["passage-id"])
            + f"\n :source {{:lines [{lo} {hi}] :kind :expository}}"
            + f"\n :provenance {{:generator {s(GENERATOR)} :model {s(model)}}}"
            + "\n :scopes [" + "\n  ".join(scopes) + "]}\n")
