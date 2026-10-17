"""An endpoint that fails while the answer is awaited is an error of that item, not of the stage.

urllib wraps a failure while sending a request as URLError, but a timeout or a dropped connection
while waiting for the answer arrives bare. S3's loop let that escape and ended the whole stage
(run mark7rhgrh12-20260927: one 600 s timeout on the run's largest candidate). S3, S4 and S7 now
record it against the item, as they already did for HTTP and connection errors.
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_box_typing as s7  # noqa: E402
import mark3_expository_loop as s4  # noqa: E402
import mark3_iatc_loop as s3  # noqa: E402


class SilentEndpoint:
    """Accepts a request, reads it, then either never answers or hangs up without answering."""

    def __init__(self, hang_up: bool):
        self.hang_up = hang_up
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            conn.recv(65536)
            if self.hang_up:
                conn.close()
            else:
                time.sleep(3)
                conn.close()

    def close(self):
        self.server.close()


def refused_port() -> int:
    with socket.create_server(("127.0.0.1", 0)) as probe:
        return probe.getsockname()[1]


class ResponsePhaseFailures(unittest.TestCase):
    def calls(self, base: str):
        env = {"OPENAI_BASE_URL": base, "FUTON6_LLM_TIMEOUT": "1"}
        return [
            (s3.ModelCallError, lambda: s3.call_openai("p", {}, "m", {"type": "object"})),
            (s4.ModelCallError, lambda: s4.call_openai("p", {"window-lines": [1, 2]}, {}, "m")),
            (s7.TypingCallError, lambda: s7.query_model(f"{base}/chat/completions", "m", "p", {"boxes": []}, [])),
        ], env

    def assert_item_errors(self, base: str):
        calls, env = self.calls(base)
        with mock.patch.dict(os.environ, env):
            for error, call in calls:
                with self.assertRaises(error):
                    call()

    def test_no_answer_within_the_timeout(self):
        endpoint = SilentEndpoint(hang_up=False)
        try:
            self.assert_item_errors(f"http://127.0.0.1:{endpoint.port}/v1")
        finally:
            endpoint.close()

    def test_connection_dropped_before_the_answer(self):
        endpoint = SilentEndpoint(hang_up=True)
        try:
            self.assert_item_errors(f"http://127.0.0.1:{endpoint.port}/v1")
        finally:
            endpoint.close()

    def test_refused_connection_still_an_item_error(self):
        self.assert_item_errors(f"http://127.0.0.1:{refused_port()}/v1")


if __name__ == "__main__":
    unittest.main()
