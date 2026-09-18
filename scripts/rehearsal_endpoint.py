#!/usr/bin/env python3
"""Local stand-in for an OpenAI-compatible model server, for runner rehearsals only.

It answers every schema-constrained request with a schema-valid document derived
deterministically from the prompt, and plain requests with fixed text. It exists
so that preflight, conformance, S1–S12, resume and retrieval can be exercised
end to end on a machine with no GPU before a billed host is allocated.

Nothing produced against it is evidence about extraction quality or acceptance:
its answers are synthetic. Runs against it should use MODEL=rehearsal-double, so
the run manifest records that no model was involved.

  python scripts/rehearsal_endpoint.py --port 8765
  OPENAI_BASE_URL=http://127.0.0.1:8765/v1 MODEL=rehearsal-double ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def instance(schema: dict, seed: int, path: str = ""):
    """A schema-valid value (object/array/string/integer/enum), varied by seed."""
    if "enum" in schema:
        return schema["enum"][seed % len(schema["enum"])]
    kind = schema.get("type")
    if kind == "object":
        return {k: instance(v, _seed(f"{seed}/{path}/{k}"), f"{path}/{k}")
                for k, v in schema.get("properties", {}).items() if k in schema.get("required", [])}
    if kind == "array":
        lo, hi = schema.get("minItems", 0), schema.get("maxItems", 3)
        n = lo + seed % (max(hi, lo) - lo + 1) if hi > lo else lo
        return [instance(schema["items"], _seed(f"{seed}/{i}"), f"{path}[{i}]") for i in range(n)]
    if kind == "integer":
        lo, hi = schema.get("minimum", 0), schema.get("maximum", schema.get("minimum", 0) + 3)
        return lo + seed % (hi - lo + 1)
    if kind == "string":
        text = f"rehearsal {path.rsplit('/', 1)[-1]} {seed % 100000}"
        return text[: schema.get("maxLength", len(text))] or "x"
    return None


def nodes_document(schema: dict, seed: int) -> dict:
    """S3 phase 1: the nodes an argument is made of."""
    line = schema["properties"]["nodes"]["items"]["properties"]["first_line"]
    lo, hi = line["minimum"], line["maximum"]
    n = 2 + seed % 4
    return {"nodes": [{"kind": "claim", "text": f"rehearsal claim {i} of {seed % 100000}", "citation": "",
                       "first_line": lo + (i * (hi - lo)) // max(n - 1, 1),
                       "last_line": lo + (i * (hi - lo)) // max(n - 1, 1)}
                      for i in range(n)]}


# Every Nth proof comes back circular, to rehearse the refusal path. Off by default.
CYCLE_EVERY = int(os.environ.get("FUTON6_REHEARSAL_CYCLE_EVERY", "0") or "0")


def derivations_document(schema: dict, seed: int) -> dict:
    """S3 phase 2: a chain over the nodes that exist, one derivation per derived node."""
    properties = schema["properties"]["derivations"]["properties"]
    nodes = sorted(int(k) for k in properties)
    kinds = ["stated", "citation", "missing"]
    derivations = {}
    for i, node in enumerate(nodes[1:], start=1):
        entry = properties[str(node)]["items"]["properties"]
        line = entry["first_line"]
        derivations[str(node)] = [{"relation": "implies", "premises": [nodes[i - 1]],
                                   "warrant_kind": kinds[(seed + i) % 3],
                                   "warrant": f"rehearsal warrant {seed % 100000}-{i}",
                                   "first_line": line["minimum"], "last_line": line["maximum"]}]
    if CYCLE_EVERY and len(nodes) > 2 and seed % CYCLE_EVERY == 0:
        # Derive the first node from the second as well, so the pair is circular.
        # A double that can only return acceptable output cannot rehearse what the
        # run does with a refusal, which is the half of the pipeline that failed.
        first = properties[str(nodes[0])]["items"]["properties"]["first_line"]
        derivations[str(nodes[0])] = [{"relation": "implies", "premises": [nodes[1]],
                                       "warrant_kind": "stated",
                                       "warrant": f"rehearsal cycle {seed % 100000}",
                                       "first_line": first["minimum"], "last_line": first["maximum"]}]
    return {"derivations": derivations}


def scopes_document(schema: dict, seed: int) -> dict:
    item = schema["properties"]["scopes"]["items"]["properties"]
    lo, hi = item["first_line"]["minimum"], item["first_line"]["maximum"]
    kinds = item["kind"]["enum"]
    return {"scopes": [{"kind": kinds[seed % len(kinds)], "first_line": lo, "last_line": hi,
                        "fill": f"rehearsal fill {seed % 100000}", "held_reason": ""}]}


class Handler(BaseHTTPRequestHandler):
    model = "rehearsal-double"

    def _send(self, body: dict, code: int = 200):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send({"data": [{"id": self.model, "object": "model"}]})
        if self.path.rstrip("/").endswith("/health"):
            return self._send({"status": "ok"})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        prompt = json.dumps(body.get("messages", []))
        seed = _seed(prompt)
        fmt = (body.get("response_format") or {}).get("json_schema")
        if fmt:
            schema, name = fmt["schema"], fmt.get("name")
            if name == "iatc_proof" and "nodes" in schema["properties"]:
                content = nodes_document(schema, seed)
            elif name == "iatc_proof":
                content = derivations_document(schema, seed)
            elif name == "expository_region":
                content = scopes_document(schema, seed)
            else:
                content = instance(schema, seed)
            text, tokens = json.dumps(content), 64
        else:
            tokens = int(body.get("max_tokens") or 16)
            text = "rehearsal " * tokens
        self._send({"model": self.model, "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": text}}],
                    "usage": {"completion_tokens": tokens}})

    def log_message(self, *args):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
