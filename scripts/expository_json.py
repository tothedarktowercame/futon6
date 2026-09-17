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

GENERATOR = "expository-json/v1"
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


def schema(lo: int, hi: int, kinds) -> dict:
    line = {"type": "integer", "minimum": lo, "maximum": hi}
    scope = {"type": "object", "additionalProperties": False,
             "required": ["kind", "first_line", "last_line", "fill", "held_reason"],
             "properties": {"kind": {"type": "string", "enum": sorted(kinds)},
                            "first_line": line, "last_line": line,
                            "fill": {"type": "string", "maxLength": 400},
                            "held_reason": {"type": "string", "maxLength": 200}}}
    return {"type": "object", "additionalProperties": False, "required": ["scopes"],
            "properties": {"scopes": {"type": "array", "minItems": 1, "maxItems": MAX_SCOPES, "items": scope}}}


def problems(doc, lo: int, hi: int, kinds) -> list[str]:
    if not isinstance(doc, dict) or not isinstance(doc.get("scopes"), list) or not doc["scopes"]:
        return ["output is not an object with a nonempty scopes list (the endpoint did not enforce the schema)"]
    found = []
    for i, s in enumerate(doc["scopes"], 1):
        if not isinstance(s, dict) or s.get("kind") not in kinds:
            found.append(f"scope {i}: kind {s.get('kind') if isinstance(s, dict) else s!r} not in the vocabulary")
            continue
        a, b = s.get("first_line"), s.get("last_line")
        if not (isinstance(a, int) and isinstance(b, int) and lo <= a <= b <= hi):
            found.append(f"scope {i}: lines {a}-{b} not an ordered range inside {lo}-{hi}")
        fill, held = str(s.get("fill", "")).strip(), str(s.get("held_reason", "")).strip()
        if bool(fill) == bool(held):
            found.append(f"scope {i}: needs exactly one of fill or held_reason")
    return found


def to_edn(doc: dict, candidate: dict, kinds: dict[str, str], model: str) -> str:
    lo, hi = candidate["window-lines"]
    s = iatc_json.edn_string
    scopes = []
    for i, scope in enumerate(doc["scopes"], 1):
        fill, held = scope["fill"].strip(), scope["held_reason"].strip()
        body = (f":slot-fill {{:{kinds[scope['kind']]} {s(fill)}}}" if fill
                else f":held {{:reason {s(held)}}}")
        scopes.append(f"{{:id :s{i} :kind :{scope['kind']} :source {{:lines [{scope['first_line']} "
                      f"{scope['last_line']}]}} {body}}}")
    return ("{:paper/id " + s(candidate["paper-id"]) + "\n :passage/id " + s(candidate["passage-id"])
            + f"\n :source {{:lines [{lo} {hi}] :kind :expository}}"
            + f"\n :provenance {{:generator {s(GENERATOR)} :model {s(model)}}}"
            + "\n :scopes [" + "\n  ".join(scopes) + "]}\n")
