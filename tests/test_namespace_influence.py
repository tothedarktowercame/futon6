"""Fixtures-only tests for namespace_influence.py (M-live-efe-map)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import namespace_influence as ni


def test_parse_endpoint_basic():
    repo, ns, var = ni.parse_endpoint("futon3c-d/agency.registry/ensure-registered")
    assert repo == "futon3c-d"
    assert ns == "agency.registry"
    assert var == "ensure-registered"


def test_parse_endpoint_var_with_slash_stays_whole():
    repo, ns, var = ni.parse_endpoint("futon0-d/some-ns/weird/var/name")
    assert repo == "futon0-d"
    assert ns == "some-ns"
    assert var == "weird/var/name"


def test_base_repo():
    assert ni.base_repo("futon3c-d") == "futon3c"
    assert ni.base_repo("futon6-py-d") == "futon6"


def test_aggregation_weights_and_self_loops():
    calls = [("r/a/x", "s/b/y"), ("r/a/z", "s/b/y"), ("s/b/y", "r/a/x")]
    namespaces, graph = ni.aggregate(calls)
    assert graph[("r", "a")][("s", "b")] == 2
    assert graph[("s", "b")][("r", "a")] == 1
    assert ("r", "a") not in graph.get(("r", "a"), {})  # self loops dropped
    assert namespaces[("s", "b")]["calls_in"] == 2
    assert namespaces[("s", "b")]["calls_in_cross_ns"] == 2
    assert namespaces[("s", "b")]["calls_in_cross_repo"] == 2
    assert namespaces[("r", "a")]["calls_out"] == 2
    assert namespaces[("r", "a")]["vars"] == {"x", "z"}


def test_aggregation_drops_self_calls_from_graph():
    calls = [("r/a/x", "r/a/y")]
    namespaces, graph = ni.aggregate(calls)
    assert dict(graph.get(("r", "a"), {})) == {}
    assert namespaces[("r", "a")]["calls_in"] == 1
    assert namespaces[("r", "a")]["calls_in_cross_ns"] == 0


def test_pagerank_sums_to_one_and_sink_wins():
    # ns "hub" has many callers; "lonely" is isolated.
    calls = [(f"r/caller{i}/f", "r/hub/g") for i in range(10)]
    namespaces, graph = ni.aggregate(calls)
    # ensure the isolated namespace exists in the graph node set
    graph[("r", "lonely")]  # noqa: B018 - touch to create node via pagerank
    ranks = ni.pagerank({("r", "hub"): dict(graph.get(("r", "hub"), {})),
                         **{("r", f"caller{i}"): dict(graph[("r", f"caller{i}")])
                            for i in range(10)},
                         ("r", "lonely"): {}})
    assert abs(sum(ranks.values()) - 1.0) < 1e-9
    assert ranks[("r", "hub")] > ranks[("r", "lonely")]
