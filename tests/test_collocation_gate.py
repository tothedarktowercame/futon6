"""Tests for the collocation-coherence gate (M-prior-mathematics, inline Stage 5).

Runs against the REAL CT prior (futon6/data/ct-term-prior.json) if present,
plus a synthetic mini-prior so the core logic is testable without the 50MB file.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "superpod_job",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "superpod-job.py"),
)
superpod_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(superpod_job)

_collocation_incoherent = superpod_job._collocation_incoherent
_keep_multi = superpod_job._discovery_keep_multiword_term

REAL_PRIOR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ct-term-prior.json")

# Synthetic prior mirroring the measured CT shape, for prior-free CI.
SYNTH = {
    "n_docs": 10000,
    "unigram_df": {
        "stable": 4140, "cartesian": 3950, "marriage": 23, "problem": 5640,
        "category": 9450, "homotopy": 5270, "lextensive": 114, "completion": 2790,
    },
    "bigram_df": {
        "stable category": 728, "stable homotopy": 833, "lextensive category": 45,
    },
}


def test_noop_without_prior():
    assert _collocation_incoherent("stable marriage", None) is False
    assert _collocation_incoherent("stable marriage", {}) is False


def test_single_token_never_incoherent():
    assert _collocation_incoherent("functor", SYNTH) is False


def test_synthetic_rejects_stable_marriage():
    # common head (stable), alien unlicensed tail (marriage)
    assert _collocation_incoherent("stable marriage", SYNTH) is True
    # trigram: the stable->marriage adjacent pair still trips it
    assert _collocation_incoherent("stable marriage problem", SYNTH) is True


def test_synthetic_passes_licensed_collocations():
    assert _collocation_incoherent("stable category", SYNTH) is False
    assert _collocation_incoherent("stable homotopy", SYNTH) is False


def test_synthetic_abstains_on_novel_low_prior_head():
    # lextensive is low-prior head -> never rejected even though
    # "lextensive completion" is not a licensed bigram
    assert _collocation_incoherent("lextensive completion", SYNTH) is False


def test_keep_multiword_seed_known_bypasses_gate():
    # seed-known terms must pass regardless of collocation (novelty/known safety)
    assert _keep_multi(
        "stable marriage",
        known_in_pm_seed=True, known_in_nlab_seed=False, known_in_nnexus_snapshot=False,
        nnexus_stopwords=set(), collocation_prior=SYNTH,
    ) is True


def test_keep_multiword_unknown_incoherent_rejected():
    assert _keep_multi(
        "stable marriage",
        known_in_pm_seed=False, known_in_nlab_seed=False, known_in_nnexus_snapshot=False,
        nnexus_stopwords=set(), collocation_prior=SYNTH,
    ) is False


def test_against_real_prior_if_present():
    if not os.path.exists(REAL_PRIOR_PATH):
        return  # skip silently when the 50MB artefact isn't local
    prior = superpod_job._load_collocation_prior(REAL_PRIOR_PATH)
    assert prior is not None
    # The target case and its guards, on real measured data.
    assert _collocation_incoherent("stable marriage", prior) is True
    assert _collocation_incoherent("cartesian marriage", prior) is True
    assert _collocation_incoherent("stable category", prior) is False
    assert _collocation_incoherent("stable homotopy", prior) is False
    assert _collocation_incoherent("abelian group", prior) is False
    assert _collocation_incoherent("lextensive completion", prior) is False
