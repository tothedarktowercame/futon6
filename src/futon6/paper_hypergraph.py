"""Stage 5d: Paper hypergraph.

Lifts a paper into a structure-and-terminology-first semantic object: the
argumentative skeleton (theorem/lemma/proof blocks + their dependencies)
plus terminological anchors (concepts from stage 5, techniques from 5c).

Two extraction arms are kept distinct for the batch-002 natural experiment
(spec: futon6/holes/missions/M-paper-reverse-morphogenesis.md §5d):

  extract_paper_hypergraph_classical(text, concepts=..., techniques=..., ...)
  extract_paper_hypergraph_llm(text, classical_hg=..., pipe=..., tokenizer=...)

Output shape matches the thread hypergraph (src/futon6/hypergraph.py) so
downstream consumers (FAISS index, R-GCN training, stage 6 reconstruction)
can treat both uniformly. Per-edge provenance is recorded in attrs.

The classical arm is deterministic LaTeX-block parsing plus term-location
indexing. It produces:
  - nodes: section, definition, theorem/lemma/proposition/corollary, proof,
           equation, citation, concept, technique
  - edges: derivation, definition-use, structural-cooccurrence,
           citation-grounding
The motivation-link edge type is LLM-arm territory (requires reading the
intro's prose to connect stated motivation to a technique node).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# --- LaTeX block parsing ---------------------------------------------------

CLAIM_ENVS = ("theorem", "lemma", "proposition", "corollary")
DEF_ENVS = ("definition",)
PROOF_ENVS = ("proof",)
EQ_ENVS = ("equation", "align", "equation*", "align*", "gather", "gather*")

_ENV_PATTERN = re.compile(
    r"\\begin\{(?P<env>"
    + "|".join(re.escape(e) for e in CLAIM_ENVS + DEF_ENVS + PROOF_ENVS + EQ_ENVS)
    + r")\}"
    r"(?:\[(?P<title>[^\]]*)\])?"
    r"(?P<body>.*?)"
    r"\\end\{(?P=env)\}",
    re.DOTALL,
)

_SECTION_PATTERN = re.compile(
    r"\\(?P<level>section|subsection|subsubsection)\*?\{(?P<title>[^}]*)\}"
)

_LABEL_PATTERN = re.compile(r"\\label\{(?P<label>[^}]+)\}")
_CITE_PATTERN = re.compile(r"\\(?:cite|citep|citet|citeauthor)\{(?P<keys>[^}]+)\}")
_REF_PATTERN = re.compile(r"\\(?:ref|eqref|cref|autoref)\{(?P<label>[^}]+)\}")


@dataclass
class Block:
    r"""A parsed LaTeX environment block.

    env: theorem|lemma|proposition|corollary|definition|proof|equation|...
    title: optional title from \begin{env}[title]
    label: optional \label{} inside the block
    body: raw body text
    span: (start, end) char offsets in the full paper text
    section: section id this block sits in
    number: positional number within its env type (1-indexed)
    """
    env: str
    body: str
    span: tuple[int, int]
    title: str | None = None
    label: str | None = None
    section: str = "0"
    number: int = 0


def _parse_section_spans(text: str) -> list[tuple[int, int, str, str]]:
    """Return list of (start, end, section_id, title). IDs are 1-indexed
    per section level; preamble (if any) is section '0'."""
    spans: list[tuple[int, int, str, str]] = []
    matches = list(_SECTION_PATTERN.finditer(text))
    if not matches:
        return [(0, len(text), "0", "")]

    counters: dict[str, int] = {"section": 0, "subsection": 0, "subsubsection": 0}
    last_section = 0
    last_subsection = 0

    entries: list[tuple[int, int, str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        level = m.group("level")
        title = m.group("title").strip()
        if level == "section":
            counters["section"] += 1
            counters["subsection"] = 0
            counters["subsubsection"] = 0
            last_section = counters["section"]
            sid = f"{last_section}"
        elif level == "subsection":
            counters["subsection"] += 1
            counters["subsubsection"] = 0
            last_subsection = counters["subsection"]
            sid = f"{last_section}.{last_subsection}"
        else:  # subsubsection
            counters["subsubsection"] += 1
            sid = (f"{last_section}.{last_subsection}."
                   f"{counters['subsubsection']}")
        entries.append((start, end, sid, title))

    if matches[0].start() > 0:
        entries.insert(0, (0, matches[0].start(), "0", ""))
    return entries


def _section_for_offset(section_spans: Sequence[tuple[int, int, str, str]],
                        offset: int) -> str:
    for start, end, sid, _title in section_spans:
        if start <= offset < end:
            return sid
    return "0"


def parse_latex_blocks(text: str) -> tuple[list[Block], list[tuple[int, int, str, str]]]:
    """Parse all recognised LaTeX environments. Returns (blocks, section_spans).

    Blocks are returned in document order. Section spans are (start, end, id,
    title) tuples.
    """
    section_spans = _parse_section_spans(text)
    blocks: list[Block] = []
    env_counts: dict[str, int] = defaultdict(int)

    for m in _ENV_PATTERN.finditer(text):
        env = m.group("env")
        body = m.group("body") or ""
        title = m.group("title")
        label_m = _LABEL_PATTERN.search(body)
        label = label_m.group("label") if label_m else None
        section = _section_for_offset(section_spans, m.start())
        env_counts[env] += 1
        blocks.append(Block(
            env=env,
            body=body,
            span=(m.start(), m.end()),
            title=title,
            label=label,
            section=section,
            number=env_counts[env],
        ))
    return blocks, section_spans


# --- Classical hypergraph builder ------------------------------------------


def _node_id(type_: str, key: str) -> str:
    safe = re.sub(r"\s+", "-", key.strip().lower())
    safe = re.sub(r"[^a-z0-9\-._]", "", safe)
    return f"{type_}:{safe}" if safe else f"{type_}:_"


def _find_term_occurrences(body: str, terms: Iterable[str]) -> list[tuple[str, int]]:
    """For each term, find char offsets in `body` (case-insensitive).

    Returns list of (canonical_term, offset_in_body)."""
    hits: list[tuple[str, int]] = []
    lowered = body.lower()
    for term in terms:
        t = term.lower()
        if not t:
            continue
        start = 0
        while True:
            idx = lowered.find(t, start)
            if idx < 0:
                break
            hits.append((term, idx))
            start = idx + len(t)
    return hits


def extract_paper_hypergraph_classical(
    text: str,
    paper_id: str,
    concepts: Iterable[str] | None = None,
    techniques: Iterable[str] | None = None,
) -> dict:
    """Build paper-level hypergraph from LaTeX block structure + term indexes.

    Args:
        text: paper body (LaTeX preferred; falls back to prose gracefully).
        paper_id: stable identifier for this paper.
        concepts: canonical concept terms (from stage 5 NER).
        techniques: canonical technique terms (from stage 5c).

    Returns a dict with keys:
        paper_id, nodes, edges, sectional, meta.
    """
    concepts = list(concepts or [])
    techniques = list(techniques or [])

    blocks, section_spans = parse_latex_blocks(text)

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def _add_node(node_id: str, type_: str, subtype: str | None = None,
                  attrs: dict | None = None):
        if node_id in nodes:
            return nodes[node_id]
        node = {
            "id": node_id,
            "type": type_,
            "subtype": subtype,
            "attrs": attrs or {},
        }
        nodes[node_id] = node
        return node

    def _add_edge(type_: str, ends: list[str], roles: dict[str, str] | None = None,
                  attrs: dict | None = None):
        edge = {
            "type": type_,
            "ends": list(ends),
            "roles": dict(roles) if roles else {},
            "attrs": attrs or {},
        }
        edge["attrs"].setdefault("provenance", "classical")
        edges.append(edge)

    # Section nodes
    for start, end, sid, title in section_spans:
        if sid == "0" and not title:
            continue
        _add_node(
            _node_id("section", sid),
            "section",
            subtype="subsection" if "." in sid else "section",
            attrs={"id": sid, "title": title, "char_span": [start, end]},
        )

    # Concept / technique term nodes
    for c in concepts:
        _add_node(_node_id("concept", c), "concept", attrs={"term": c})
    for t in techniques:
        _add_node(_node_id("technique", t), "technique", attrs={"term": t})

    # Block nodes (definition / claim / proof / equation)
    claim_nodes_by_label: dict[str, str] = {}   # label -> node id
    for b in blocks:
        if b.env in CLAIM_ENVS:
            bid = _node_id("claim", f"{b.env}-{b.number}")
            _add_node(bid, "claim", subtype=b.env, attrs={
                "env": b.env,
                "number": b.number,
                "title": b.title,
                "label": b.label,
                "section": b.section,
                "char_span": list(b.span),
                "statement": b.body.strip()[:800],
            })
            if b.label:
                claim_nodes_by_label[b.label] = bid
        elif b.env in DEF_ENVS:
            bid = _node_id("definition", f"def-{b.number}")
            _add_node(bid, "definition", attrs={
                "number": b.number,
                "title": b.title,
                "label": b.label,
                "section": b.section,
                "char_span": list(b.span),
                "text": b.body.strip()[:800],
            })
            if b.label:
                claim_nodes_by_label[b.label] = bid
        elif b.env in PROOF_ENVS:
            bid = _node_id("proof", f"proof-{b.number}")
            _add_node(bid, "proof", attrs={
                "number": b.number,
                "section": b.section,
                "char_span": list(b.span),
                "text": b.body.strip()[:800],
            })
        elif b.env in EQ_ENVS:
            bid = _node_id("equation", f"eq-{b.number}")
            _add_node(bid, "equation", attrs={
                "env": b.env,
                "number": b.number,
                "label": b.label,
                "section": b.section,
                "char_span": list(b.span),
                "tex": b.body.strip()[:400],
            })
            if b.label:
                claim_nodes_by_label[b.label] = bid

    # derivation edge: consecutive claim -> proof in document order
    prev_claim: dict | None = None
    for b in blocks:
        if b.env in CLAIM_ENVS:
            prev_claim = b
        elif b.env in PROOF_ENVS and prev_claim is not None:
            target = _node_id("claim", f"{prev_claim.env}-{prev_claim.number}")
            proof_id = _node_id("proof", f"proof-{b.number}")
            # Techniques + definitions mentioned in the proof body become
            # "depends_on" ends. This is the core derivation hyperedge.
            ends = [target, proof_id]
            roles = {target: "target", proof_id: "proof_of_target"}

            proof_techs = _find_term_occurrences(b.body, techniques)
            for term, _ in proof_techs:
                tid = _node_id("technique", term)
                if tid not in ends:
                    ends.append(tid)
                    roles[tid] = "depends_on"

            # \ref{} inside proof pointing at a labelled definition/claim
            for m in _REF_PATTERN.finditer(b.body):
                label = m.group("label")
                ref_id = claim_nodes_by_label.get(label)
                if ref_id and ref_id not in ends:
                    ends.append(ref_id)
                    roles[ref_id] = "depends_on"

            _add_edge(
                "derivation",
                ends,
                roles=roles,
                attrs={"claim_env": prev_claim.env,
                       "proof_number": b.number,
                       "section": b.section},
            )
            prev_claim = None  # one proof per claim in this classical pass

    # definition-use edge: term defined in a \begin{definition} block and
    # mentioned elsewhere.
    for b in blocks:
        if b.env not in DEF_ENVS:
            continue
        def_id = _node_id("definition", f"def-{b.number}")
        defined_terms: list[str] = []
        for c in concepts:
            if c.lower() in b.body.lower():
                defined_terms.append(c)
        for t in techniques:
            if t.lower() in b.body.lower():
                defined_terms.append(t)

        for term in defined_terms:
            use_spans: list[int] = []
            needle = term.lower()
            lowered = text.lower()
            start = b.span[1]  # search after the definition
            while True:
                idx = lowered.find(needle, start)
                if idx < 0:
                    break
                use_spans.append(idx)
                start = idx + len(needle)
            if not use_spans:
                continue
            term_node = (_node_id("concept", term)
                         if term in concepts
                         else _node_id("technique", term))
            ends = [def_id, term_node]
            roles = {def_id: "defines", term_node: "term"}
            for i, offset in enumerate(use_spans[:10]):
                use_section = _section_for_offset(section_spans, offset)
                use_id = f"{term_node}@{use_section}#{i}"
                _add_node(use_id, "term-use",
                          attrs={"term": term, "offset": offset,
                                 "section": use_section})
                ends.append(use_id)
                roles[use_id] = "used_at"
            _add_edge("definition-use", ends, roles=roles,
                      attrs={"term": term, "section": b.section})

    # structural-cooccurrence: concepts+techniques co-mentioned in the same
    # claim/proof/definition block form a hyperedge.
    for b in blocks:
        if b.env not in CLAIM_ENVS + DEF_ENVS + PROOF_ENVS:
            continue
        mentioned: list[str] = []
        for c in concepts:
            if c.lower() in b.body.lower():
                mentioned.append(_node_id("concept", c))
        for t in techniques:
            if t.lower() in b.body.lower():
                mentioned.append(_node_id("technique", t))
        if len(mentioned) < 2:
            continue
        if b.env in CLAIM_ENVS:
            block_id = _node_id("claim", f"{b.env}-{b.number}")
        elif b.env in DEF_ENVS:
            block_id = _node_id("definition", f"def-{b.number}")
        else:
            block_id = _node_id("proof", f"proof-{b.number}")
        _add_edge("structural-cooccurrence",
                  [block_id] + mentioned,
                  roles={block_id: "container",
                         **{n: "cooccurs" for n in mentioned}},
                  attrs={"env": b.env, "section": b.section})

    # citation-grounding: \cite{K} inside a block links the block's terms to
    # citation node K.
    for b in blocks:
        if b.env not in CLAIM_ENVS + PROOF_ENVS:
            continue
        cite_keys: list[str] = []
        for m in _CITE_PATTERN.finditer(b.body):
            for k in m.group("keys").split(","):
                key = k.strip()
                if key:
                    cite_keys.append(key)
        if not cite_keys:
            continue
        if b.env in CLAIM_ENVS:
            block_id = _node_id("claim", f"{b.env}-{b.number}")
        else:
            block_id = _node_id("proof", f"proof-{b.number}")
        block_techs = [_node_id("technique", t) for t in techniques
                       if t.lower() in b.body.lower()]
        for key in cite_keys:
            cite_id = _node_id("citation", key)
            _add_node(cite_id, "citation", attrs={"key": key})
            ends = [block_id, cite_id] + block_techs
            roles = {block_id: "uses_citation", cite_id: "cited"}
            for tn in block_techs:
                roles[tn] = "via"
            _add_edge("citation-grounding", ends, roles=roles,
                      attrs={"citation_key": key, "section": b.section})

    meta = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_blocks": len(blocks),
        "n_sections": len(section_spans),
        "has_theorem_blocks": any(b.env in CLAIM_ENVS for b in blocks),
        "has_proof_blocks": any(b.env in PROOF_ENVS for b in blocks),
    }

    return {
        "paper_id": paper_id,
        "nodes": list(nodes.values()),
        "edges": edges,
        "sectional": [
            {"id": sid, "title": title, "char_span": [start, end]}
            for (start, end, sid, title) in section_spans
        ],
        "meta": meta,
    }


# --- LLM arm ---------------------------------------------------------------


LLM_PROMPT_PREFIX = (
    "You are analyzing the argumentative skeleton of a mathematics paper. "
    "A classical parser has already extracted the explicit LaTeX structure: "
    "numbered theorems, lemmas, proofs, definitions, equations, and the "
    "technique terms named in each block. Your job is to add IMPLICIT "
    "hyperedges that the classical parser missed — specifically:\n\n"
    "1. `derivation`: a proof that uses another lemma's construction WITHOUT "
    "an explicit \\ref or \\cite.\n"
    "2. `motivation-link`: the intro or abstract gestures at a problem whose "
    "resolution is a theorem/technique elsewhere in the paper.\n\n"
    "Return ONLY a JSON array of edges. Each edge has keys: type, ends "
    "(list of node IDs from the classical hypergraph), roles (map of "
    "node_id -> role name), rationale (one sentence).\n\n"
    "If you cannot identify ANY implicit edges, return [].\n\n"
)


def _summarize_hypergraph_for_llm(hg: dict, max_nodes: int = 80) -> str:
    """Produce a compact text summary of the classical hypergraph for the LLM."""
    lines = [f"PAPER: {hg['paper_id']}"]
    lines.append(f"SECTIONS: {[s['id'] + ' - ' + s['title'] for s in hg['sectional']][:20]}")
    lines.append("\nNODES (abbreviated):")
    for node in hg["nodes"][:max_nodes]:
        typ = node["type"]
        nid = node["id"]
        if typ == "claim":
            stmt = (node["attrs"].get("statement") or "")[:160]
            lines.append(f"  [{nid}] {node['subtype']}: {stmt}")
        elif typ == "definition":
            txt = (node["attrs"].get("text") or "")[:160]
            lines.append(f"  [{nid}] definition: {txt}")
        elif typ == "proof":
            txt = (node["attrs"].get("text") or "")[:120]
            lines.append(f"  [{nid}] proof: {txt}")
        elif typ == "technique":
            lines.append(f"  [{nid}] technique: {node['attrs'].get('term')}")
        elif typ == "concept":
            lines.append(f"  [{nid}] concept: {node['attrs'].get('term')}")
        elif typ == "citation":
            lines.append(f"  [{nid}] cite: {node['attrs'].get('key')}")
    lines.append("\nEXISTING EDGES (classical):")
    for e in hg["edges"][:max_nodes]:
        lines.append(f"  {e['type']}: ends={e['ends']}  roles={e.get('roles', {})}")
    return "\n".join(lines)


def _build_llm_prompt(text: str, classical_hg: dict,
                      prose_cap_chars: int = 4000,
                      hg_cap_nodes: int = 80) -> str:
    prose = text[:prose_cap_chars]
    hg_summary = _summarize_hypergraph_for_llm(classical_hg, max_nodes=hg_cap_nodes)
    return (
        LLM_PROMPT_PREFIX
        + "CLASSICAL HYPERGRAPH SUMMARY:\n" + hg_summary + "\n\n"
        + "PAPER PROSE (abstract + early sections):\n" + prose + "\n\n"
        + "IMPLICIT EDGES (JSON array only):"
    )


def _parse_llm_edges(response: str) -> list[dict]:
    """Parse a JSON array of edge dicts from the LLM response."""
    start = response.find("[")
    end = response.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(response[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for e in parsed:
        if not isinstance(e, dict):
            continue
        if "type" not in e or "ends" not in e:
            continue
        if not isinstance(e["ends"], list):
            continue
        out.append({
            "type": str(e["type"]),
            "ends": [str(x) for x in e["ends"]],
            "roles": dict(e.get("roles") or {}),
            "attrs": {
                "provenance": "llm",
                "rationale": str(e.get("rationale", ""))[:400],
            },
        })
    return out


def _filter_llm_edges(classical_hg: dict, edges: list[dict]) -> list[dict]:
    """Keep only LLM edges whose endpoint node ids exist classically."""
    valid_ids = {n["id"] for n in classical_hg["nodes"]}
    filtered = []
    for e in edges:
        if all(end in valid_ids for end in e["ends"]):
            filtered.append(e)
    return filtered


def extract_paper_hypergraph_llm_batch(
    texts: list[str],
    classical_hgs: list[dict],
    pipe,
    tokenizer,
    max_new_tokens: int = 700,
    prose_cap_chars: int = 4000,
    hg_cap_nodes: int = 80,
    batch_size: int = 8,
    loader_workers: int = 0,
) -> list[list[dict]]:
    """Batched LLM implicit-edge extraction for paper hypergraphs.

    Uses a Dataset-backed transformers pipeline call so GPU inference is
    streamed in batches instead of invoked once per paper.
    """
    from torch.utils.data import Dataset as TorchDataset

    if len(texts) != len(classical_hgs):
        raise ValueError(
            f"text/classical_hg length mismatch: {len(texts)} != {len(classical_hgs)}"
        )

    class _PaperHypergraphPromptDataset(TorchDataset):
        def __init__(self, paper_texts, hgs, tok):
            self.paper_texts = paper_texts
            self.hgs = hgs
            self.tok = tok

        def __len__(self):
            return len(self.paper_texts)

        def __getitem__(self, idx):
            prompt = _build_llm_prompt(
                self.paper_texts[idx],
                self.hgs[idx],
                prose_cap_chars=prose_cap_chars,
                hg_cap_nodes=hg_cap_nodes,
            )
            messages = [{"role": "user", "content": prompt}]
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    if not texts:
        return []

    prompt_dataset = _PaperHypergraphPromptDataset(texts, classical_hgs, tokenizer)
    outputs = pipe(
        prompt_dataset,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        num_workers=loader_workers,
    )

    results: list[list[dict]] = []
    for classical_hg, out in zip(classical_hgs, outputs):
        if isinstance(out, list):
            item = out[0] if out else {}
        elif isinstance(out, dict):
            item = out
        else:
            item = {}
        raw = str(item.get("generated_text", ""))
        results.append(_filter_llm_edges(classical_hg, _parse_llm_edges(raw)))
    return results


def extract_paper_hypergraph_llm(
    text: str,
    classical_hg: dict,
    pipe,
    tokenizer,
    max_new_tokens: int = 700,
    prose_cap_chars: int = 4000,
    hg_cap_nodes: int = 80,
) -> list[dict]:
    """Call the LLM to propose implicit hyperedges. Returns edge list only.

    Caller is expected to merge these into the classical hypergraph via
    merge_paper_hypergraphs().
    """
    return extract_paper_hypergraph_llm_batch(
        [text],
        [classical_hg],
        pipe=pipe,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        prose_cap_chars=prose_cap_chars,
        hg_cap_nodes=hg_cap_nodes,
        batch_size=1,
        loader_workers=0,
    )[0]


def merge_paper_hypergraphs(classical_hg: dict, llm_edges: list[dict]) -> dict:
    """Merge LLM edges into the classical hypergraph. Duplicates (same type +
    same end set) are marked provenance='both' on the classical edge; new
    edges carry provenance='llm'."""
    merged = {
        "paper_id": classical_hg["paper_id"],
        "nodes": list(classical_hg["nodes"]),
        "edges": [dict(e) for e in classical_hg["edges"]],
        "sectional": list(classical_hg.get("sectional", [])),
        "meta": dict(classical_hg.get("meta", {})),
    }
    existing_keys = {
        (e["type"], frozenset(e["ends"])): i
        for i, e in enumerate(merged["edges"])
    }

    n_new = 0
    n_both = 0
    for llm_edge in llm_edges:
        key = (llm_edge["type"], frozenset(llm_edge["ends"]))
        if key in existing_keys:
            i = existing_keys[key]
            merged["edges"][i]["attrs"]["provenance"] = "both"
            rationale = llm_edge.get("attrs", {}).get("rationale")
            if rationale:
                merged["edges"][i]["attrs"]["llm_rationale"] = rationale
            n_both += 1
        else:
            merged["edges"].append(llm_edge)
            n_new += 1

    merged["meta"]["n_edges"] = len(merged["edges"])
    merged["meta"]["n_llm_new_edges"] = n_new
    merged["meta"]["n_llm_confirmed_edges"] = n_both
    return merged
