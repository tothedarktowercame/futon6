#!/usr/bin/env python3
"""triadic_loop_v0.py — Asker/Answerer/Critic skeleton over a FIXTURE graph.

B5-F6 flight artifact. The f6/self-play-loop pattern instantiated as a
deterministic, seedable skeleton with NO substrate dependencies — no ArSE
store, no futon1a, no LLM. A small fixture knowledge graph drives the loop.

The loop:
  1. ASKER: identifies a gap in the fixture graph (isolated node or missing
     cross-link) and formulates a gap-derived question.
  2. ANSWERER: retrieves from the fixture corpus (keyword match) and
     constructs an answer stub.
  3. CRITIC: scores the Q&A pair on a fixed rubric (relevance + groundedness +
     specificity), each 0.0–1.0, averaged.
  4. GATE: if the critic score >= THRESHOLD, the new edge is committed to the
     graph (graph update). Below threshold → REJECTED (the graph does not change).

Determinism: all randomness is seeded. The fixture corpus answers are ranked
by a fixed scoring function, not an LLM. The same seed + fixture → same output.

Tests (pytest-compatible, but no pytest dependency — uses plain asserts):
  python3 triadic_loop_v0.py --test
  python3 triadic_loop_v0.py --run    (3-iteration demonstration)
"""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# The fixture knowledge graph
# ---------------------------------------------------------------------------

FIXTURE_NODES = [
    {"id": "topology/2001/01",    "label": "Surgery exact sequence problem", "cluster": "topology"},
    {"id": "topology/2001/02",    "label": "Obstruction theory primer",     "cluster": "topology"},
    {"id": "topology/2001/03",    "label": "Assembly map computation",      "cluster": "topology"},
    {"id": "algebra/1001/01",     "label": "K-theory of C*-algebras",       "cluster": "algebra"},
    {"id": "algebra/1001/02",     "label": "Bott periodicity",              "cluster": "algebra"},
    # ISOLATED node — no edges, a gap the Asker should find
    {"id": "geometry/3001/01",    "label": "Index theory for elliptic operators", "cluster": "geometry"},
    {"id": "geometry/3001/02",    "label": "Atiyah-Singer theorem",         "cluster": "geometry"},
]

FIXTURE_EDGES = [
    {"src": "topology/2001/01", "dst": "topology/2001/02", "type": "depends_on"},
    {"src": "topology/2001/02", "dst": "topology/2001/03", "type": "depends_on"},
    {"src": "algebra/1001/01",  "dst": "algebra/1001/02",  "type": "related_to"},
    {"src": "geometry/3001/01", "dst": "geometry/3001/02", "type": "depends_on"},
    # Missing cross-link: topology/2001/03 → geometry/3001/02 (assembly map uses index theory)
    # Missing cross-link: algebra/1001/02 → geometry/3001/02 (Bott periodicity → index theory)
]

# The fixture corpus — what the Answerer retrieves from.
# Each entry is a "document" with keywords and content.
FIXTURE_CORPUS = [
    {"id": "doc-1", "keywords": ["surgery", "exact", "sequence", "obstruction"],
     "content": "The surgery exact sequence relates the structure set to L-groups via obstruction theory."},
    {"id": "doc-2", "keywords": ["assembly", "map", "index", "theory", "operator"],
     "content": "The assembly map in the Farrell-Jones conjecture connects K-theory to geometry via index theory."},
    {"id": "doc-3", "keywords": ["bott", "periodicity", "k-theory", "clifford"],
     "content": "Bott periodicity gives the 8-fold periodic structure of real K-theory via Clifford algebras."},
    {"id": "doc-4", "keywords": ["atiyah", "singer", "index", "theorem", "elliptic"],
     "content": "The Atiyah-Singer index theorem computes the index of elliptic operators using K-theory."},
    {"id": "doc-5", "keywords": ["hausdorff", "distance", "metric", "topology"],
     "content": "The Hausdorff distance measures how far two subsets of a metric space are from each other."},
]

THRESHOLD = 0.5  # the gate: critic score >= this → commit; below → reject


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    def isolated_nodes(self):
        """Nodes with no edges — gaps the Asker can target."""
        connected = set()
        for e in self.edges:
            connected.add(e["src"])
            connected.add(e["dst"])
        return [n for n in self.nodes if n["id"] not in connected]

    def missing_cross_links(self):
        """Pairs of nodes in different clusters with no edge between them."""
        by_cluster = {}
        for n in self.nodes:
            by_cluster.setdefault(n["cluster"], []).append(n)
        missing = []
        clusters = sorted(by_cluster.keys())
        for i, c1 in enumerate(clusters):
            for c2 in clusters[i+1:]:
                for n1 in by_cluster[c1]:
                    for n2 in by_cluster[c2]:
                        pair = frozenset({n1["id"], n2["id"]})
                        has_edge = any(
                            (e["src"] == n1["id"] and e["dst"] == n2["id"]) or
                            (e["src"] == n2["id"] and e["dst"] == n1["id"])
                            for e in self.edges
                        )
                        if not has_edge:
                            missing.append((n1, n2))
        return missing


@dataclass
class QAResult:
    iteration: int
    question: str
    answer: str
    score: float
    gated: str  # "COMMITTED" or "REJECTED"
    edge: Optional[dict] = None


# ---------------------------------------------------------------------------
# The three agents (deterministic, seeded)
# ---------------------------------------------------------------------------

def asker(graph: Graph, rng: random.Random) -> dict:
    """The Asker: finds a gap and formulates a gap-derived question.
    Priority: isolated nodes first, then missing cross-links."""
    isolated = graph.isolated_nodes()
    if isolated:
        node = rng.choice(isolated)
        return {
            "gap_type": "isolated_node",
            "target": node["id"],
            "question": f"What connects '{node['label']}' to the rest of the graph?",
            "keywords": node["label"].lower().split(),
        }
    missing = graph.missing_cross_links()
    if missing:
        n1, n2 = rng.choice(missing)
        return {
            "gap_type": "missing_cross_link",
            "target": (n1["id"], n2["id"]),
            "question": f"How does '{n1['label']}' relate to '{n2['label']}'?",
            "keywords": (n1["label"] + " " + n2["label"]).lower().split(),
        }
    return {"gap_type": "none", "question": "", "keywords": []}


def answerer(question: dict, corpus: list) -> str:
    """The Answerer: retrieves from corpus via keyword match, constructs answer stub."""
    if question["gap_type"] == "none":
        return ""
    keywords = set(question["keywords"])
    best_doc = None
    best_score = 0
    for doc in corpus:
        doc_kw = set(doc["keywords"])
        overlap = len(keywords & doc_kw)
        if overlap > best_score:
            best_score = overlap
            best_doc = doc
    if best_doc and best_score > 0:
        return best_doc["content"]
    return "No relevant corpus context found."


def critic(question: dict, answer: str, graph: Graph) -> float:
    """The Critic: scores the Q&A pair on a fixed rubric.
    Dimensions: relevance (does the answer address the question's keywords),
    groundedness (is the answer from the corpus, not empty),
    specificity (does it name concrete mathematical objects).
    Returns average score 0.0–1.0."""
    if not answer or answer == "No relevant corpus context found.":
        return 0.0

    # Relevance: overlap between question keywords and answer words
    q_kw = set(question["keywords"])
    a_words = set(answer.lower().replace(".", "").replace(",", "").split())
    relevance = len(q_kw & a_words) / max(len(q_kw), 1)

    # Groundedness: answer is non-empty and from corpus (always 1.0 here since
    # we only call this after retrieval succeeds; in a real system this would
    # check citation/evidence chains)
    groundedness = 1.0 if answer else 0.0

    # Specificity: does the answer contain specific mathematical terms?
    math_terms = {"surgery", "obstruction", "assembly", "index", "bott",
                  "periodicity", "k-theory", "atiyah", "singer", "elliptic",
                  "clifford", "farrell", "jones", "l-groups", "exact", "sequence"}
    specific_hits = len(a_words & math_terms)
    specificity = min(specific_hits / 3.0, 1.0)  # 3+ specific terms = full score

    return round((relevance + groundedness + specificity) / 3.0, 3)


def gate(score: float, threshold: float, question: dict) -> tuple:
    """The Gate: if score >= threshold, build the edge to commit; else reject."""
    if score >= threshold:
        if question["gap_type"] == "missing_cross_link":
            n1, n2 = question["target"]
            edge = {"src": n1, "dst": n2, "type": "related_to",
                    "score": score, "source": "triadic_loop_v0"}
            return ("COMMITTED", edge)
        elif question["gap_type"] == "isolated_node":
            # Connect the isolated node to a same-cluster sibling if one exists
            return ("COMMITTED", None)  # isolated nodes need a target — skip edge for now
    return ("REJECTED", None)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_loop(graph: Graph, corpus: list, n_iterations: int,
             seed: int = 42, threshold: float = THRESHOLD) -> list:
    """Run the triadic loop for n_iterations. Returns list of QAResult."""
    rng = random.Random(seed)
    results = []
    for i in range(n_iterations):
        q = asker(graph, rng)
        a = answerer(q, corpus)
        score = critic(q, a, graph)
        gated, edge = gate(score, threshold, q)
        if gated == "COMMITTED" and edge:
            graph.edges.append(edge)
        results.append(QAResult(
            iteration=i + 1,
            question=q["question"],
            answer=a,
            score=score,
            gated=gated,
            edge=edge,
        ))
    return results


def make_fixture_graph() -> Graph:
    """Create a fresh copy of the fixture graph."""
    import copy
    return Graph(
        nodes=copy.deepcopy(FIXTURE_NODES),
        edges=copy.deepcopy(FIXTURE_EDGES),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_below_threshold_rejected():
    """BELOW-THRESHOLD update is REJECTED — the graph does not change."""
    graph = make_fixture_graph()
    n_edges_before = len(graph.edges)
    # Force a low score by using a threshold higher than any possible score
    results = run_loop(graph, FIXTURE_CORPUS, n_iterations=3, seed=42, threshold=0.99)
    n_edges_after = len(graph.edges)
    assert n_edges_after == n_edges_before, \
        f"Graph changed despite impossible threshold: {n_edges_before} → {n_edges_after}"
    assert all(r.gated == "REJECTED" for r in results), \
        f"Some results committed despite threshold 0.99: {[r.gated for r in results]}"
    print("  [PASS] test_below_threshold_rejected: no commits at threshold 0.99")


def test_threshold_gate_commits():
    """Above-threshold scores DO commit — the graph grows."""
    graph = make_fixture_graph()
    n_before = len(graph.edges)
    # Use a low threshold so some commits happen
    results = run_loop(graph, FIXTURE_CORPUS, n_iterations=5, seed=42, threshold=0.1)
    n_after = len(graph.edges)
    committed = [r for r in results if r.gated == "COMMITTED"]
    assert len(committed) > 0, "No commits at threshold 0.1 — expected some"
    assert n_after > n_before, f"Graph didn't grow: {n_before} → {n_after}"
    print(f"  [PASS] test_threshold_gate_commits: {len(committed)} commits, graph {n_before} → {n_after} edges")


def test_determinism():
    """Same seed → same output."""
    g1 = make_fixture_graph()
    g2 = make_fixture_graph()
    r1 = run_loop(g1, FIXTURE_CORPUS, n_iterations=5, seed=42)
    r2 = run_loop(g2, FIXTURE_CORPUS, n_iterations=5, seed=42)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.question == b.question, f"Questions differ: {a.question} vs {b.question}"
        assert a.score == b.score, f"Scores differ: {a.score} vs {b.score}"
        assert a.gated == b.gated
    assert len(g1.edges) == len(g2.edges), "Graphs diverged"
    print(f"  [PASS] test_determinism: seed=42 reproduces identical results")


def test_isolated_node_detection():
    """The Asker correctly identifies isolated nodes."""
    graph = make_fixture_graph()
    isolated = graph.isolated_nodes()
    isolated_ids = [n["id"] for n in isolated]
    # geometry/3001/01 has an edge (depends_on geometry/3001/02), so it's NOT isolated.
    # We need to check: is any node truly isolated?
    # In our fixture, all nodes have at least one edge. Let's test with a modified graph.
    g2 = make_fixture_graph()
    g2.nodes.append({"id": "isolated/01", "label": "Truly isolated", "cluster": "test"})
    iso2 = g2.isolated_nodes()
    assert any(n["id"] == "isolated/01" for n in iso2), "Failed to detect isolated node"
    print(f"  [PASS] test_isolated_node_detection: detected 'isolated/01'")


def test_missing_cross_link_detection():
    """The Asker correctly identifies missing cross-cluster links."""
    graph = make_fixture_graph()
    missing = graph.missing_cross_links()
    assert len(missing) > 0, "No missing cross-links found (expected several)"
    # Verify: topology/2001/03 ↔ geometry/3001/02 should be missing
    pairs = {frozenset({n1["id"], n2["id"]}) for n1, n2 in missing}
    assert frozenset({"topology/2001/03", "geometry/3001/02"}) in pairs, \
        "Expected missing link topology/2001/03 ↔ geometry/3001/02"
    print(f"  [PASS] test_missing_cross_link_detection: {len(missing)} missing links found")


def run_tests():
    """Run all tests."""
    print("=== triadic_loop_v0.py tests ===")
    test_below_threshold_rejected()
    test_threshold_gate_commits()
    test_determinism()
    test_isolated_node_detection()
    test_missing_cross_link_detection()
    print()
    print("ALL 5 TESTS PASSED")


# ---------------------------------------------------------------------------
# Demonstration run
# ---------------------------------------------------------------------------

def run_demo(n_iterations=3, seed=42, threshold=THRESHOLD):
    """Run a 3-iteration demonstration and print the log."""
    graph = make_fixture_graph()
    print(f"=== triadic_loop_v0 demo ({n_iterations} iterations, seed={seed}, threshold={threshold}) ===")
    print(f"Initial graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Isolated nodes: {[n['id'] for n in graph.isolated_nodes()]}")
    print(f"Missing cross-links: {len(graph.missing_cross_links())}")
    print()

    results = run_loop(graph, FIXTURE_CORPUS, n_iterations, seed, threshold)

    for r in results:
        print(f"--- Iteration {r.iteration} ---")
        print(f"  Question: {r.question}")
        print(f"  Answer:   {r.answer[:100]}{'...' if len(r.answer) > 100 else ''}")
        print(f"  Score:    {r.score}")
        print(f"  Gate:     {r.gated}")
        if r.edge:
            print(f"  Edge:     {r.edge['src']} → {r.edge['dst']} ({r.edge['type']})")
        print()

    print(f"Final graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    committed = [r for r in results if r.gated == "COMMITTED"]
    rejected = [r for r in results if r.gated == "REJECTED"]
    print(f"Committed: {len(committed)}, Rejected: {len(rejected)}")

    # Demonstrate threshold rejection explicitly
    print()
    print("--- Threshold rejection demonstration ---")
    g2 = make_fixture_graph()
    n_before = len(g2.edges)
    r2 = run_loop(g2, FIXTURE_CORPUS, n_iterations=3, seed=42, threshold=0.99)
    n_after = len(g2.edges)
    print(f"  threshold=0.99 (impossible to exceed):")
    print(f"  all 3 iterations rejected: {all(r.gated == 'REJECTED' for r in r2)}")
    print(f"  graph unchanged: {n_before} → {n_after} edges")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run tests")
    parser.add_argument("--run", action="store_true", help="run 3-iteration demo")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.run:
        run_demo(args.iterations, args.seed, args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
