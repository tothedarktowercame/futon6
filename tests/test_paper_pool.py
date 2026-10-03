"""Per-paper stages run their papers on every CPU the job may use, reporting in list order."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import futon6_config as config  # noqa: E402
import paper_pool  # noqa: E402


class PaperPoolTests(unittest.TestCase):
    def test_results_come_back_in_list_order_from_several_workers(self):
        papers = ["p" + "x" * n for n in (5, 1, 3, 0, 2)]
        self.assertEqual(list(paper_pool.map_papers(len, papers, workers=3)),
                         [(paper, len(paper)) for paper in papers])

    def test_one_worker_and_an_empty_list_need_no_pool(self):
        self.assertEqual(list(paper_pool.map_papers(len, ["ab"], workers=1)), [("ab", 2)])
        self.assertEqual(list(paper_pool.map_papers(len, [])), [])

    def test_workers_default_to_the_cpus_this_job_may_use(self):
        environment = {k: v for k, v in os.environ.items() if k != "FUTON6_WORKERS"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(config.workers(), len(os.sched_getaffinity(0)))
        with patch.dict(os.environ, {**environment, "FUTON6_WORKERS": "3"}, clear=True):
            self.assertEqual(config.workers(), 3)


if __name__ == "__main__":
    unittest.main()
