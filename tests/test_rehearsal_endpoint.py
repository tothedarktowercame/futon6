"""The stand-in endpoint must answer every schema the run sends with a document the run accepts."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expository_json  # noqa: E402
import iatc_json  # noqa: E402
import mark3_iatc_loop as loop  # noqa: E402
import rehearsal_endpoint  # noqa: E402

KINDS = {"definition": "term", "remark": "note"}
UNITS = [{"id": "u1", "line": 3, "start": 40, "end": 76, "text": "Let  L(s) be the completed\n L-function."},
         {"id": "u2", "line": 4, "start": 77, "end": 103, "text": "It is entire of order one."}]


class ExpositoryScopeTests(unittest.TestCase):
    def test_unit_citing_scopes_quote_the_unit_they_cite(self):
        schema = expository_json.schema(3, 4, KINDS, UNITS)
        prompt = "Units of this region:\n" + "\n".join(
            f"  {u['id']}  {' '.join(u['text'].split())}" for u in UNITS)
        for seed in range(6):
            doc = rehearsal_endpoint.scopes_document(schema, seed, prompt)
            self.assertEqual(expository_json.problems(doc, 3, 4, KINDS, UNITS), [], doc)
            self.assertTrue(doc["scopes"][0]["fill"])

    def test_a_unit_the_prompt_does_not_list_is_held_not_invented(self):
        schema = expository_json.schema(3, 4, KINDS, UNITS)
        doc = rehearsal_endpoint.scopes_document(schema, 0, "")
        self.assertEqual(expository_json.problems(doc, 3, 4, KINDS, UNITS), [], doc)
        self.assertEqual(doc["scopes"][0]["fill"], "")

    def test_line_scopes_still_answered_for_candidates_without_units(self):
        schema = expository_json.schema(3, 4, KINDS)
        doc = rehearsal_endpoint.scopes_document(schema, 1)
        self.assertEqual(expository_json.problems(doc, 3, 4, KINDS), [], doc)


class ProofGraphTests(unittest.TestCase):
    def test_each_proof_answer_passes_the_s3_contract(self):
        """As mark3_iatc_loop judges it: the nodes answer, then the derivations, then the contract."""
        for seed in range(12):
            doc = rehearsal_endpoint.nodes_document(iatc_json.nodes_schema(10, 14), seed)
            doc.update(rehearsal_endpoint.derivations_document(
                iatc_json.steps_schema(10, 14, len(doc["nodes"])), seed))
            doc["steps"] = iatc_json.steps_of(doc)
            self.assertEqual(iatc_json.problems(doc, 10, 14), [], doc)


class TokenPreflightTests(unittest.TestCase):
    """S3 asks the server's /tokenize before every call, so a rehearsal needs the stand-in to answer it."""

    def setUp(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), rehearsal_endpoint.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        env = mock.patch.dict(os.environ, {"OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1"})
        env.start()
        self.addCleanup(env.stop)

    def test_s3_preflight_and_call_pass_against_the_stand_in(self):
        prompt = "Show that the completed L-function is entire of order one. " * 20
        answer = loop.call_openai(prompt, {}, "rehearsal-double", iatc_json.nodes_schema(10, 14))
        self.assertEqual(answer.prompt_tokens, len(prompt) // 4)
        self.assertEqual(answer.context_tokens, loop.MODEL_CONTEXT_TOKENS)
        self.assertTrue(json.loads(answer.text)["nodes"])

    def test_a_prompt_the_context_cannot_answer_is_refused_before_the_call(self):
        prompt = "x" * 4 * loop.MODEL_CONTEXT_TOKENS
        with self.assertRaisesRegex(loop.ModelCallError, "token preflight refused"):
            loop.call_openai(prompt, {}, "rehearsal-double", iatc_json.nodes_schema(10, 14))


if __name__ == "__main__":
    unittest.main()
