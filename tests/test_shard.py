"""Tests for shard filter and merge functionality."""

import json
import os
import tempfile
from pathlib import Path

from futon6.stackexchange import (
    build_qa_pairs_streaming,
    build_threads_streaming,
)


FIXTURE_POSTS = str(Path(__file__).parent / "fixtures" / "se-mini" / "Posts.xml")
FIXTURE_COMMENTS = str(Path(__file__).parent / "fixtures" / "se-mini" / "Comments.xml")


class TestShardFilter:
    """Verify shard filter on build_qa_pairs_streaming."""

    def test_shards_are_disjoint_and_cover_all(self):
        """Union of all shards equals the full (unsharded) set."""
        all_pairs = build_qa_pairs_streaming(FIXTURE_POSTS, min_score=0)
        all_qids = {p.question.id for p in all_pairs}

        for n in [2, 3, 4]:
            union = set()
            for i in range(n):
                pairs = build_qa_pairs_streaming(
                    FIXTURE_POSTS, min_score=0,
                    shard_index=i, num_shards=n)
                qids = {p.question.id for p in pairs}
                assert qids.isdisjoint(union), (
                    f"Shard {i}/{n} overlaps with previous shards")
                union |= qids
            assert union == all_qids, (
                f"Shards don't cover all qids for n={n}: "
                f"missing={all_qids - union}, extra={union - all_qids}")

    def test_single_shard_equals_full(self):
        """One shard of one equals the full set."""
        all_pairs = build_qa_pairs_streaming(FIXTURE_POSTS, min_score=0)
        shard_pairs = build_qa_pairs_streaming(
            FIXTURE_POSTS, min_score=0,
            shard_index=0, num_shards=1)
        assert len(shard_pairs) == len(all_pairs)

    def test_shard_respects_min_score(self):
        """Shard filter composes with min_score filter."""
        for i in range(2):
            pairs = build_qa_pairs_streaming(
                FIXTURE_POSTS, min_score=3,
                shard_index=i, num_shards=2)
            for p in pairs:
                assert p.question.score >= 3

    def test_no_shard_args_gives_all(self):
        """Omitting shard args returns everything (backward compat)."""
        all_pairs = build_qa_pairs_streaming(FIXTURE_POSTS, min_score=0)
        assert len(all_pairs) >= 3  # fixture has 4 questions


class TestShardFilterThreads:
    """Verify shard filter on build_threads_streaming."""

    def test_thread_shards_disjoint_and_cover(self):
        """Union of thread shards equals the full set."""
        all_threads = build_threads_streaming(
            FIXTURE_POSTS, FIXTURE_COMMENTS, min_score=0)
        all_qids = {t.question.id for t in all_threads}

        for n in [2, 3]:
            union = set()
            for i in range(n):
                threads = build_threads_streaming(
                    FIXTURE_POSTS, FIXTURE_COMMENTS, min_score=0,
                    shard_index=i, num_shards=n)
                qids = {t.question.id for t in threads}
                assert qids.isdisjoint(union)
                union |= qids
            assert union == all_qids

    def test_thread_shard_has_answers_and_comments(self):
        """Sharded threads still have their answers and comments."""
        for i in range(2):
            threads = build_threads_streaming(
                FIXTURE_POSTS, FIXTURE_COMMENTS, min_score=0,
                shard_index=i, num_shards=2)
            for t in threads:
                # Every question in fixture has at least one answer
                assert len(t.answers) >= 1, (
                    f"Thread {t.question.id} in shard {i} has no answers")


class TestMerge:
    """Test merge logic from superpod-shard.py."""

    def test_merge_json_arrays(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from importlib import import_module
        shard_mod = import_module("superpod-shard")

        with tempfile.TemporaryDirectory() as tmpdir:
            d0 = Path(tmpdir) / "shard-0"
            d1 = Path(tmpdir) / "shard-1"
            d0.mkdir()
            d1.mkdir()

            # Write shard files in pipeline format
            with open(d0 / "entities.json", "w") as f:
                f.write('[\n{"id": 1},\n{"id": 2}\n]')
            with open(d1 / "entities.json", "w") as f:
                f.write('[\n{"id": 3}\n]')

            out = Path(tmpdir) / "merged"
            out.mkdir()
            shard_mod.merge_json_array_files([d0, d1], "entities.json",
                                              out / "entities.json")

            with open(out / "entities.json") as f:
                result = json.load(f)
            assert len(result) == 3
            assert [r["id"] for r in result] == [1, 2, 3]

    def test_merge_tags_as_array(self):
        """tags.json is a JSON array of tag objects, merged like other arrays."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from importlib import import_module
        shard_mod = import_module("superpod-shard")

        with tempfile.TemporaryDirectory() as tmpdir:
            d0 = Path(tmpdir) / "shard-0"
            d1 = Path(tmpdir) / "shard-1"
            d0.mkdir()
            d1.mkdir()

            with open(d0 / "tags.json", "w") as f:
                f.write('[\n{"name": "algebra"},\n{"name": "topology"}\n]')
            with open(d1 / "tags.json", "w") as f:
                f.write('[\n{"name": "analysis"}\n]')

            out = Path(tmpdir) / "merged"
            out.mkdir()
            shard_mod.merge_json_array_files([d0, d1], "tags.json",
                                              out / "tags.json")

            with open(out / "tags.json") as f:
                result = json.load(f)
            assert len(result) == 3
            names = {r["name"] for r in result}
            assert names == {"algebra", "topology", "analysis"}

    def test_merge_stats(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from importlib import import_module
        shard_mod = import_module("superpod-shard")

        with tempfile.TemporaryDirectory() as tmpdir:
            d0 = Path(tmpdir) / "shard-0"
            d1 = Path(tmpdir) / "shard-1"
            d0.mkdir()
            d1.mkdir()

            with open(d0 / "stats.json", "w") as f:
                json.dump({"qa_pairs": 100, "unique_tags": 20}, f)
            with open(d1 / "stats.json", "w") as f:
                json.dump({"qa_pairs": 150, "unique_tags": 25}, f)

            out = Path(tmpdir) / "merged"
            out.mkdir()
            result = shard_mod.merge_stats([d0, d1], out / "stats.json")
            assert result["qa_pairs"] == 250
            assert result["unique_tags"] == 45

    def test_merge_empty_shard_skipped(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from importlib import import_module
        shard_mod = import_module("superpod-shard")

        with tempfile.TemporaryDirectory() as tmpdir:
            d0 = Path(tmpdir) / "shard-0"
            d1 = Path(tmpdir) / "shard-1"
            d0.mkdir()
            d1.mkdir()

            with open(d0 / "entities.json", "w") as f:
                f.write('[\n{"id": 1}\n]')
            # d1 has no entities.json

            out = Path(tmpdir) / "merged"
            out.mkdir()
            shard_mod.merge_json_array_files([d0, d1], "entities.json",
                                              out / "entities.json")

            with open(out / "entities.json") as f:
                result = json.load(f)
            assert len(result) == 1

    def test_merge_npy(self):
        import numpy as np
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from importlib import import_module
        shard_mod = import_module("superpod-shard")

        with tempfile.TemporaryDirectory() as tmpdir:
            d0 = Path(tmpdir) / "shard-0"
            d1 = Path(tmpdir) / "shard-1"
            d0.mkdir()
            d1.mkdir()

            np.save(str(d0 / "embeddings.npy"), np.ones((3, 4)))
            np.save(str(d1 / "embeddings.npy"), np.zeros((2, 4)))

            out = Path(tmpdir) / "merged"
            out.mkdir()
            shape = shard_mod.merge_npy_files([d0, d1], "embeddings.npy",
                                               out / "embeddings.npy")
            assert shape == (5, 4)
            merged = np.load(str(out / "embeddings.npy"))
            assert merged.shape == (5, 4)
            assert merged[:3].sum() == 12.0  # 3*4 ones
            assert merged[3:].sum() == 0.0   # 2*4 zeros
