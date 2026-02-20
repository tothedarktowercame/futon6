"""IATC performative detection and thread wiring diagram construction.

Models MathOverflow/SE threads as argument diagrams following Corneli 2017:
each post/comment is a node, each relationship (challenge, reform, clarify, ...)
is a typed edge. Detects performatives via regex bank (classical detection)
and produces wiring diagrams compatible with futon4 hyperedge schema.

Performative types follow IATC vocabulary adapted for futon:
  assert, challenge, query, clarify, reform, exemplify, reference, agree, retract
"""

import re
import json
from dataclasses import dataclass, field, asdict


# --- Data structures ---

@dataclass
class ThreadNode:
    id: str             # "t42-q", "t42-a123", "t42-c456"
    node_type: str      # "question" | "answer" | "comment"
    post_id: int
    body_text: str
    score: int
    creation_date: str
    parent_post_id: int | None = None
    tags: dict[str, str] | None = None


@dataclass
class ThreadEdge:
    source: str         # node id
    target: str         # node id
    edge_type: str      # IATC performative
    evidence: str       # matched text
    detection: str      # "structural" | "classical" | "llm"


@dataclass
class ThreadWiringDiagram:
    thread_id: str
    nodes: list[ThreadNode] = field(default_factory=list)
    edges: list[ThreadEdge] = field(default_factory=list)


# --- IATC performative regex bank ---

PERFORMATIVE_REGEXES: dict[str, list[re.Pattern]] = {
    "challenge": [
        re.compile(r"\bbut\b", re.IGNORECASE),
        re.compile(r"\bhowever\b", re.IGNORECASE),
        re.compile(r"\bnot quite\b", re.IGNORECASE),
        re.compile(r"\bincorrect\b", re.IGNORECASE),
        re.compile(r"\bwrong\b", re.IGNORECASE),
        re.compile(r"\bdisagree\b", re.IGNORECASE),
        re.compile(r"\bthis is false\b", re.IGNORECASE),
        re.compile(r"\bnot true\b", re.IGNORECASE),
        re.compile(r"\bmistake\b", re.IGNORECASE),
    ],
    "query": [
        re.compile(r"\bwhat about\b", re.IGNORECASE),
        re.compile(r"\bdoes this work\b", re.IGNORECASE),
        re.compile(r"\bcan you explain\b", re.IGNORECASE),
        re.compile(r"\bcould you clarify\b", re.IGNORECASE),
        re.compile(r"\bwhy does\b", re.IGNORECASE),
        re.compile(r"\bhow does\b", re.IGNORECASE),
        re.compile(r"\bwhat if\b", re.IGNORECASE),
        re.compile(r"\bwhat do you mean\b", re.IGNORECASE),
        re.compile(r"\?$", re.MULTILINE),
    ],
    "clarify": [
        re.compile(r"\bto be precise\b", re.IGNORECASE),
        re.compile(r"\bin other words\b", re.IGNORECASE),
        re.compile(r"\bi\.e\.\b", re.IGNORECASE),
        re.compile(r"\bthat is,\b", re.IGNORECASE),
        re.compile(r"\bmore precisely\b", re.IGNORECASE),
        re.compile(r"\bto clarify\b", re.IGNORECASE),
        re.compile(r"\bspecifically\b", re.IGNORECASE),
        re.compile(r"\bmeaning that\b", re.IGNORECASE),
    ],
    "reform": [
        re.compile(r"\bbetter way\b", re.IGNORECASE),
        re.compile(r"\breal question\b", re.IGNORECASE),
        re.compile(r"\bequivalently\b", re.IGNORECASE),
        re.compile(r"\breduces to\b", re.IGNORECASE),
        re.compile(r"\bis equivalent to\b", re.IGNORECASE),
        re.compile(r"\bcan be rephrased\b", re.IGNORECASE),
        re.compile(r"\banother way to see\b", re.IGNORECASE),
        re.compile(r"\binstead of\b", re.IGNORECASE),
    ],
    "exemplify": [
        re.compile(r"\bfor example\b", re.IGNORECASE),
        re.compile(r"\bconsider the case\b", re.IGNORECASE),
        re.compile(r"\bto illustrate\b", re.IGNORECASE),
        re.compile(r"\be\.g\.\b", re.IGNORECASE),
        re.compile(r"\bfor instance\b", re.IGNORECASE),
        re.compile(r"\bas an example\b", re.IGNORECASE),
        re.compile(r"\btake for example\b", re.IGNORECASE),
    ],
    "reference": [
        re.compile(r"\bsee\s+(?:also\s+)?(?:the\s+)?(?:e\.g\.\s+)?(?:\[|theorem|lemma|proposition|section|chapter)", re.IGNORECASE),
        re.compile(r"\bcf\.\b", re.IGNORECASE),
        re.compile(r"\barXiv:", re.IGNORECASE),
        re.compile(r"https?://"),
        re.compile(r"\bMR\d{5,}", re.IGNORECASE),
        re.compile(r"\bdoi:", re.IGNORECASE),
    ],
    "agree": [
        re.compile(r"\b\+1\b"),
        re.compile(r"\byes exactly\b", re.IGNORECASE),
        re.compile(r"\bthis is correct\b", re.IGNORECASE),
        re.compile(r"\bgreat answer\b", re.IGNORECASE),
        re.compile(r"\bnice answer\b", re.IGNORECASE),
        re.compile(r"\bI agree\b", re.IGNORECASE),
        re.compile(r"\bexactly right\b", re.IGNORECASE),
        re.compile(r"\bthat's right\b", re.IGNORECASE),
    ],
    "retract": [
        re.compile(r"\bI was wrong\b", re.IGNORECASE),
        re.compile(r"\bignore my earlier\b", re.IGNORECASE),
        re.compile(r"\bedit:\s*fixed\b", re.IGNORECASE),
        re.compile(r"\bmy mistake\b", re.IGNORECASE),
        re.compile(r"\bI stand corrected\b", re.IGNORECASE),
        re.compile(r"\bretracted\b", re.IGNORECASE),
        re.compile(r"\bnever\s?mind\b", re.IGNORECASE),
    ],
}


def detect_performatives(text: str) -> list[tuple[str, str]]:
    """Detect IATC performatives in text via regex bank.

    Returns list of (performative_type, evidence_text) tuples.
    A text can match multiple performative types.
    """
    hits = []
    for ptype, patterns in PERFORMATIVE_REGEXES.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                hits.append((ptype, m.group()))
                break  # one match per type is enough
    return hits


# --- Wiring diagram construction ---

def build_thread_wiring_diagram(thread) -> ThreadWiringDiagram:
    """Build a wiring diagram from an SEThread.

    1. Structural edges (always present):
       - answer → question = "assert"
       - comment → parent_post = "comment-on"
    2. Classical detection: scan text with PERFORMATIVE_REGEXES;
       override structural edge type when detected.

    Args:
        thread: an SEThread object (from stackexchange.py)

    Returns:
        ThreadWiringDiagram with nodes and edges
    """
    qid = thread.question.id
    tid = f"t{qid}"

    diagram = ThreadWiringDiagram(thread_id=tid)

    # Question node
    q_node_id = f"{tid}-q"
    diagram.nodes.append(ThreadNode(
        id=q_node_id,
        node_type="question",
        post_id=qid,
        body_text=thread.question.body_text,
        score=thread.question.score,
        creation_date=thread.question.creation_date,
    ))

    # Answer nodes + structural edges
    for ans in thread.answers:
        a_node_id = f"{tid}-a{ans.id}"
        diagram.nodes.append(ThreadNode(
            id=a_node_id,
            node_type="answer",
            post_id=ans.id,
            body_text=ans.body_text,
            score=ans.score,
            creation_date=ans.creation_date,
            parent_post_id=qid,
        ))

        # Structural edge: answer → question = "assert"
        # Check for classical override
        perfs = detect_performatives(ans.body_text)
        if perfs:
            # Use the first detected performative as the edge type
            edge_type = perfs[0][0]
            evidence = perfs[0][1]
            detection = "classical"
        else:
            edge_type = "assert"
            evidence = ""
            detection = "structural"

        diagram.edges.append(ThreadEdge(
            source=a_node_id,
            target=q_node_id,
            edge_type=edge_type,
            evidence=evidence,
            detection=detection,
        ))

    # Comment nodes + edges
    for post_id, comments in thread.comments.items():
        # Determine target node id for comments on this post
        if post_id == qid:
            target_node_id = q_node_id
        else:
            target_node_id = f"{tid}-a{post_id}"

        for cmt in comments:
            c_node_id = f"{tid}-c{cmt.id}"
            diagram.nodes.append(ThreadNode(
                id=c_node_id,
                node_type="comment",
                post_id=cmt.id,
                body_text=cmt.text,
                score=cmt.score,
                creation_date=cmt.creation_date,
                parent_post_id=post_id,
            ))

            # Structural edge: comment → parent_post = "comment-on"
            # Check for classical override
            perfs = detect_performatives(cmt.text)
            if perfs:
                edge_type = perfs[0][0]
                evidence = perfs[0][1]
                detection = "classical"
            else:
                edge_type = "comment-on"
                evidence = ""
                detection = "structural"

            diagram.edges.append(ThreadEdge(
                source=c_node_id,
                target=target_node_id,
                edge_type=edge_type,
                evidence=evidence,
                detection=detection,
            ))

    return diagram


# --- Hyperedge conversion (futon4-compatible) ---

def diagram_to_hyperedges(diagram: ThreadWiringDiagram) -> list[dict]:
    """Convert a wiring diagram to futon4-compatible hyperedge records.

    Each edge becomes a hyperedge with:
      hx/type  — edge type (IATC performative or structural)
      hx/ends  — source and target node refs with roles
      hx/content — evidence text and metadata
      hx/labels — tags for filtering
    """
    hyperedges = []
    for i, edge in enumerate(diagram.edges):
        hx_id = f"{diagram.thread_id}:edge-{i:03d}"

        # Look up source and target nodes for metadata
        source_node = next((n for n in diagram.nodes if n.id == edge.source), None)
        target_node = next((n for n in diagram.nodes if n.id == edge.target), None)

        ends = [
            {
                "role": "source",
                "ident": edge.source,
                "node_type": source_node.node_type if source_node else "unknown",
            },
            {
                "role": "target",
                "ident": edge.target,
                "node_type": target_node.node_type if target_node else "unknown",
            },
        ]

        hyperedges.append({
            "hx/id": hx_id,
            "hx/type": f"thread/{edge.edge_type}",
            "hx/ends": ends,
            "hx/content": {
                "evidence": edge.evidence,
                "detection": edge.detection,
            },
            "hx/labels": ["thread", edge.edge_type, edge.detection],
        })

    return hyperedges


def diagram_to_dict(diagram: ThreadWiringDiagram) -> dict:
    """Serialize a wiring diagram to a JSON-safe dict."""
    return {
        "thread_id": diagram.thread_id,
        "nodes": [asdict(n) for n in diagram.nodes],
        "edges": [asdict(e) for e in diagram.edges],
        "hyperedges": diagram_to_hyperedges(diagram),
        "stats": diagram_stats(diagram),
    }


def diagram_stats(diagram: ThreadWiringDiagram) -> dict:
    """Compute summary statistics for a wiring diagram."""
    edge_types = {}
    detection_types = {}
    for edge in diagram.edges:
        edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
        detection_types[edge.detection] = detection_types.get(edge.detection, 0) + 1

    node_types = {}
    for node in diagram.nodes:
        node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

    return {
        "n_nodes": len(diagram.nodes),
        "n_edges": len(diagram.edges),
        "node_types": node_types,
        "edge_types": edge_types,
        "detection_types": detection_types,
    }


# --- Moist-run prompt builder ---

def build_thread_performative_prompt(diagram: ThreadWiringDiagram) -> str:
    """Build an LLM prompt for performative classification of a thread.

    Generates a prompt that asks the LLM to classify each edge in the
    thread wiring diagram with an IATC performative type.
    """
    lines = []
    lines.append("You are a discourse analyst studying mathematical argumentation online.")
    lines.append("")
    lines.append("Given a thread from a math/science Q&A site, classify each ")
    lines.append("post/comment relationship using IATC performative types.")
    lines.append("")
    lines.append("Performative types:")
    lines.append("  assert    — states a claim or provides an answer")
    lines.append("  challenge — disputes, contradicts, or corrects")
    lines.append("  query     — asks a question or requests clarification")
    lines.append("  clarify   — restates more precisely or explains meaning")
    lines.append("  reform    — reframes the problem or proposes equivalent formulation")
    lines.append("  exemplify — provides an example or illustration")
    lines.append("  reference — cites external source or prior work")
    lines.append("  agree     — endorses or confirms")
    lines.append("  retract   — withdraws a prior claim")
    lines.append("")
    lines.append("For each relationship, classify the performative type.")
    lines.append("Reply as JSON list:")
    lines.append('[{"source": "<node_id>", "target": "<node_id>", '
                 '"performative": "<type>", "reasoning": "<brief explanation>"}]')
    lines.append("")
    lines.append("Thread:")

    # Question
    q_node = next((n for n in diagram.nodes if n.node_type == "question"), None)
    if q_node:
        lines.append(f"  [Q] (score={q_node.score}): {q_node.body_text[:400]}")

    # Answers
    for node in diagram.nodes:
        if node.node_type == "answer":
            lines.append(f"  [A{node.post_id}] (score={node.score}): {node.body_text[:400]}")

    # Comments
    for node in diagram.nodes:
        if node.node_type == "comment":
            lines.append(f"  [C{node.post_id} on post {node.parent_post_id}]: {node.body_text[:200]}")

    return "\n".join(lines)


VALID_PERFORMATIVES = {
    "assert", "challenge", "query", "clarify", "reform",
    "exemplify", "reference", "agree", "retract",
}


def merge_llm_edges(diagram: ThreadWiringDiagram, llm_edges: list[dict]):
    """Merge LLM-classified edge types back into a wiring diagram.

    For each LLM edge classification, finds the matching edge (by source+target)
    and updates its edge_type, evidence, and detection to reflect the LLM result.
    Skips edges with invalid performative types or unmatched source/target.

    Modifies diagram in place.
    """
    edge_lookup = {(e.source, e.target): e for e in diagram.edges}

    for llm_edge in llm_edges:
        if not isinstance(llm_edge, dict):
            continue
        source = llm_edge.get("source", "")
        target = llm_edge.get("target", "")
        performative = llm_edge.get("performative", "").lower().strip()
        reasoning = llm_edge.get("reasoning", "")

        if performative not in VALID_PERFORMATIVES:
            continue

        existing = edge_lookup.get((source, target))
        if existing:
            existing.edge_type = performative
            existing.evidence = reasoning[:120] if reasoning else existing.evidence
            existing.detection = "llm"


def write_thread_wiring_json(diagrams: list[ThreadWiringDiagram], output_path: str):
    """Write thread wiring diagrams to a JSON file (streaming)."""
    with open(output_path, "w") as f:
        f.write("[\n")
        for i, diagram in enumerate(diagrams):
            sep = ",\n" if i > 0 else ""
            f.write(sep + json.dumps(diagram_to_dict(diagram), ensure_ascii=False))
        f.write("\n]")


# --- Batch processing ---

def process_threads_to_diagrams(
    threads,
    output_path: str | None = None,
) -> tuple[list[ThreadWiringDiagram], dict]:
    """Process a list of SEThread objects into wiring diagrams.

    Optionally writes results as streaming JSON to output_path.

    Returns (diagrams, aggregate_stats).
    """
    from collections import Counter

    diagrams = []
    total_edges = 0
    total_nodes = 0
    classical_edges = 0
    structural_edges = 0
    performative_freq = Counter()
    threads_with_classical = 0

    f = None
    if output_path:
        f = open(output_path, "w")
        f.write("[\n")

    try:
        for i, thread in enumerate(threads):
            diagram = build_thread_wiring_diagram(thread)
            diagrams.append(diagram)

            stats = diagram_stats(diagram)
            total_nodes += stats["n_nodes"]
            total_edges += stats["n_edges"]
            classical_count = stats["detection_types"].get("classical", 0)
            classical_edges += classical_count
            structural_edges += stats["detection_types"].get("structural", 0)
            if classical_count > 0:
                threads_with_classical += 1
            for etype, count in stats["edge_types"].items():
                performative_freq[etype] += count

            if f:
                sep = ",\n" if i > 0 else ""
                f.write(sep + json.dumps(diagram_to_dict(diagram), ensure_ascii=False))

        if f:
            f.write("\n]")
    finally:
        if f:
            f.close()

    agg_stats = {
        "threads_processed": len(threads),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "classical_edges": classical_edges,
        "structural_edges": structural_edges,
        "classical_edge_rate": classical_edges / total_edges if total_edges else 0,
        "threads_with_classical": threads_with_classical,
        "threads_with_classical_rate": threads_with_classical / len(threads) if threads else 0,
        "performative_freq": dict(performative_freq.most_common()),
        "unique_performatives": len(performative_freq),
    }

    return diagrams, agg_stats
