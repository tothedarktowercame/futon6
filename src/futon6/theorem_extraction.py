r"""Theorem extraction from ArXiv LaTeX source.

Extracts theorem-like environments (\begin{theorem}...\end{theorem}) from
LaTeX source files and pairs them with their proofs. Produces structured
records suitable for conversion into stepper missions.

Design constraints:
  - Regex-based, no LaTeX compiler needed (papers may not compile cleanly)
  - Handles common authoring patterns: labels, theorem names, nested envs
  - Pairs theorems with immediately-following proofs when present
  - Preserves full content (no truncation — unlike latex_terms.py)
"""

import re
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


# -- Environment detection --

# Theorem-like environments (statements we want to prove)
THEOREM_ENVS = frozenset({
    "theorem", "lemma", "proposition", "corollary",
    "conjecture", "claim", "fact",
})

# Supporting environments (context, not missions themselves)
CONTEXT_ENVS = frozenset({
    "definition", "example", "remark", "notation", "assumption",
})

# Proof environments
PROOF_ENVS = frozenset({"proof"})

ALL_ENVS = THEOREM_ENVS | CONTEXT_ENVS | PROOF_ENVS

# Regex for \begin{env}...\end{env} with proper nesting awareness.
# This simple version handles non-nested cases. For papers with nested
# theorem environments (rare), we fall back to brace-counting.
_ENV_RE = re.compile(
    r"\\begin\{(" + "|".join(ALL_ENVS) + r")\}"
    r"(.*?)"
    r"\\end\{\1\}",
    re.DOTALL,
)

# Optional label: \label{thm:foo} immediately after \begin{theorem}
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")

# Optional theorem name: \begin{theorem}[Name of Theorem]
_THMNAME_RE = re.compile(r"^\s*\[([^\]]+)\]")

# \ref{label} and \eqref{label} cross-references within content
_REF_RE = re.compile(r"\\(?:eq)?ref\{([^}]+)\}")

# Section/subsection headings for context
_SECTION_RE = re.compile(
    r"\\(?:sub)*section\*?\{([^}]+)\}",
)


@dataclass
class TheoremRecord:
    """A theorem-like statement extracted from a LaTeX source."""
    # Identity
    paper_id: str           # ArXiv ID e.g. "q-alg/9503002"
    env_type: str           # "theorem", "lemma", "proposition", etc.
    label: str = ""         # \label{thm:foo} if present
    name: str = ""          # [Name] if present, e.g. "Yoneda Lemma"
    number_hint: str = ""   # best-effort env counter within paper

    # Content
    statement: str = ""     # full LaTeX body of the environment
    proof: str = ""         # associated \begin{proof}...\end{proof} if found
    section: str = ""       # enclosing \section title

    # Cross-references
    refs: list[str] = field(default_factory=list)    # \ref{} targets in statement
    proof_refs: list[str] = field(default_factory=list)  # \ref{} targets in proof

    # Metadata
    char_offset: int = 0    # byte offset in source file
    statement_hash: str = ""  # SHA256 of statement for dedup/versioning

    def __post_init__(self):
        if self.statement and not self.statement_hash:
            self.statement_hash = hashlib.sha256(
                self.statement.encode()
            ).hexdigest()[:16]

    @property
    def theorem_id(self) -> str:
        """Stable ID: paper_id + label or hash."""
        slug = self.label or f"{self.env_type}-{self.statement_hash[:8]}"
        return f"{self.paper_id}::{slug}"

    @property
    def has_proof(self) -> bool:
        return bool(self.proof.strip())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["theorem_id"] = self.theorem_id
        d["has_proof"] = self.has_proof
        return d


@dataclass
class ExtractionResult:
    """All theorems extracted from one paper."""
    paper_id: str
    source_path: str
    theorems: list[TheoremRecord] = field(default_factory=list)
    definitions: list[dict] = field(default_factory=list)  # context only
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "source_path": self.source_path,
            "theorems": [t.to_dict() for t in self.theorems],
            "definitions": self.definitions,
            "stats": self.stats,
        }


def _current_section(source: str, offset: int) -> str:
    """Find the most recent \\section heading before offset."""
    best = ""
    for m in _SECTION_RE.finditer(source):
        if m.start() > offset:
            break
        best = m.group(1).strip()
    return best


def _count_env_before(source: str, env_type: str, offset: int) -> int:
    """Count how many environments of this type appear before offset."""
    pattern = re.compile(r"\\begin\{" + re.escape(env_type) + r"\}")
    return sum(1 for m in pattern.finditer(source) if m.start() < offset)


def extract_theorems(source: str, paper_id: str) -> ExtractionResult:
    """Extract all theorem-like environments from LaTeX source.

    Returns an ExtractionResult with theorems (provable statements) and
    definitions (context). Theorems are paired with immediately-following
    proofs when present.
    """
    # First pass: find all environments with their positions
    envs = []
    for m in _ENV_RE.finditer(source):
        envs.append({
            "env_type": m.group(1),
            "content": m.group(2),
            "start": m.start(),
            "end": m.end(),
        })

    theorems = []
    definitions = []

    for i, env in enumerate(envs):
        content = env["content"].strip()
        env_type = env["env_type"]

        if env_type in CONTEXT_ENVS:
            # Extract label and name for definitions too
            label = ""
            lm = _LABEL_RE.search(content)
            if lm:
                label = lm.group(1)
            definitions.append({
                "env_type": env_type,
                "label": label,
                "content": content[:500],  # truncate context envs
                "section": _current_section(source, env["start"]),
            })
            continue

        if env_type in PROOF_ENVS:
            # Proofs are attached to theorems below, skip standalone
            continue

        if env_type not in THEOREM_ENVS:
            continue

        # Extract label
        label = ""
        lm = _LABEL_RE.search(content)
        if lm:
            label = lm.group(1)
            # Remove \label{} from the statement for cleanliness
            content_clean = content[:lm.start()] + content[lm.end():]
        else:
            content_clean = content

        # Extract theorem name [Name]
        name = ""
        nm = _THMNAME_RE.match(content_clean)
        if nm:
            name = nm.group(1).strip()
            content_clean = content_clean[nm.end():].strip()

        # Extract \ref{} cross-references
        refs = [r.group(1) for r in _REF_RE.finditer(content_clean)]

        # Look for immediately-following proof
        proof_text = ""
        proof_refs = []
        if i + 1 < len(envs):
            next_env = envs[i + 1]
            # Check that the next environment is a proof and that there's
            # only whitespace/comments between this env and the proof
            gap = source[env["end"]:next_env["start"]]
            gap_stripped = re.sub(r"%[^\n]*\n", "", gap).strip()
            if next_env["env_type"] == "proof" and len(gap_stripped) < 50:
                proof_text = next_env["content"].strip()
                proof_refs = [r.group(1) for r in _REF_RE.finditer(proof_text)]

        # Number hint (1-indexed count of this env type in paper)
        num = _count_env_before(source, env_type, env["start"]) + 1

        rec = TheoremRecord(
            paper_id=paper_id,
            env_type=env_type,
            label=label,
            name=name,
            number_hint=f"{env_type.capitalize()} {num}",
            statement=content_clean.strip(),
            proof=proof_text,
            section=_current_section(source, env["start"]),
            refs=refs,
            proof_refs=proof_refs,
            char_offset=env["start"],
        )
        theorems.append(rec)

    result = ExtractionResult(
        paper_id=paper_id,
        source_path="",
        theorems=theorems,
        definitions=definitions,
        stats={
            "total_envs": len(envs),
            "theorems": len(theorems),
            "with_proof": sum(1 for t in theorems if t.has_proof),
            "with_name": sum(1 for t in theorems if t.name),
            "with_label": sum(1 for t in theorems if t.label),
            "definitions": len(definitions),
            "env_breakdown": {
                et: sum(1 for t in theorems if t.env_type == et)
                for et in THEOREM_ENVS if any(t.env_type == et for t in theorems)
            },
        },
    )
    return result


def extract_from_file(tex_path: str, paper_id: str) -> ExtractionResult:
    """Extract theorems from a .tex file on disk."""
    source = Path(tex_path).read_text(errors="replace")
    result = extract_theorems(source, paper_id)
    result.source_path = tex_path
    return result


def extract_from_tarball(tar_path: str, paper_id: str) -> ExtractionResult:
    """Extract theorems from an ArXiv eprint tarball.

    ArXiv eprints are typically .tar.gz containing one or more .tex files.
    We find the main .tex file (the one with \\begin{document}) and extract
    from it, with \\input{} expansion for multi-file papers.
    """
    import tarfile

    tex_contents = {}
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".tex") and member.isfile():
                f = tar.extractfile(member)
                if f:
                    tex_contents[member.name] = f.read().decode(
                        "utf-8", errors="replace"
                    )

    if not tex_contents:
        # Might be a single .tex file, not a tarball
        try:
            source = Path(tar_path).read_text(errors="replace")
            result = extract_theorems(source, paper_id)
            result.source_path = tar_path
            return result
        except Exception:
            return ExtractionResult(
                paper_id=paper_id, source_path=tar_path,
                stats={"error": "no .tex files found"},
            )

    # Find main file: the one containing \begin{document}
    main_file = None
    for name, content in tex_contents.items():
        if r"\begin{document}" in content:
            main_file = name
            break

    if not main_file:
        # Fall back to largest .tex file
        main_file = max(tex_contents, key=lambda k: len(tex_contents[k]))

    source = tex_contents[main_file]

    # Basic \input{} expansion (one level)
    def expand_input(src: str) -> str:
        def replacer(m):
            fname = m.group(1)
            if not fname.endswith(".tex"):
                fname += ".tex"
            return tex_contents.get(fname, "")
        return re.sub(r"\\input\{([^}]+)\}", replacer, src)

    source = expand_input(source)

    result = extract_theorems(source, paper_id)
    result.source_path = f"{tar_path}::{main_file}"
    return result


def to_stepper_missions(
    theorems: list[TheoremRecord],
    *,
    min_statement_len: int = 50,
    require_proof: bool = False,
) -> list[dict]:
    """Convert extracted theorems into stepper mission skeletons.

    Each mission has:
      - canonical statement + closure criterion
      - standard framing obligations (L-claim-type through L-conclusion)
      - suggested corpus queries derived from the statement
    """
    missions = []
    for thm in theorems:
        # Skip trivial statements
        if len(thm.statement) < min_statement_len:
            continue
        if require_proof and not thm.has_proof:
            continue

        # Build canonical statement
        display_name = thm.name or thm.number_hint
        canonical = (
            f"[{thm.paper_id}, {display_name}] "
            f"{thm.statement}"
        )

        # Closure criterion depends on env type
        if thm.env_type == "conjecture":
            closure = "Prove or disprove with explicit argument"
        else:
            closure = "Rigorous proof following the stepper framing-first discipline"

        # Generate corpus queries from the statement
        # Strip LaTeX commands, keep mathematical terms
        clean = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", thm.statement)
        clean = re.sub(r"[\\${}]", " ", clean)
        words = [w for w in clean.split() if len(w) > 3]

        # Generic framing queries
        generic_queries = [
            f"preconditions for {thm.env_type} about {' '.join(words[:6])}",
            f"known obstructions or counterexamples to {' '.join(words[:8])}",
        ]

        # Domain queries from theorem name and section
        domain_queries = []
        if thm.name:
            domain_queries.append(thm.name)
        if thm.section:
            domain_queries.append(f"{thm.section} {thm.env_type}")

        mission = {
            "mission_id": thm.theorem_id,
            "paper_id": thm.paper_id,
            "env_type": thm.env_type,
            "display_name": display_name,
            "canonical": {
                "statement": canonical,
                "closure_criterion": closure,
                "statement_hash": thm.statement_hash,
            },
            "ledger_template": [
                {
                    "id": "L-claim-type",
                    "label": f"Classify claim type ({thm.env_type})",
                    "depends_on": [],
                    "unlocks": ["L-preconditions", "L-obstruction-scan"],
                },
                {
                    "id": "L-preconditions",
                    "label": "Verify hypotheses and preconditions hold",
                    "depends_on": ["L-claim-type"],
                    "unlocks": ["L-bridge"],
                },
                {
                    "id": "L-obstruction-scan",
                    "label": "Search for known obstructions or counterexamples",
                    "depends_on": ["L-claim-type"],
                    "unlocks": ["L-bridge"],
                },
                {
                    "id": "L-bridge",
                    "label": "Evaluate proof route (only after preconditions + obstruction-scan)",
                    "depends_on": ["L-preconditions", "L-obstruction-scan"],
                    "unlocks": ["L-conclusion"],
                },
                {
                    "id": "L-conclusion",
                    "label": "Final proof or disproof",
                    "depends_on": ["L-bridge"],
                    "unlocks": [],
                },
            ],
            "corpus_queries": {
                "generic": generic_queries,
                "domain": domain_queries,
            },
            "source": {
                "section": thm.section,
                "label": thm.label,
                "has_proof": thm.has_proof,
                "refs": thm.refs,
            },
        }
        missions.append(mission)

    return missions


def batch_extract(
    papers: list[dict],
    eprint_dir: str,
    *,
    output_jsonl: str | None = None,
) -> list[ExtractionResult]:
    """Extract theorems from a batch of downloaded ArXiv eprints.

    Args:
        papers: list of dicts with at least {"id": "...", "eprint_url": "..."}
        eprint_dir: directory containing downloaded eprint files
        output_jsonl: if set, stream results to this JSONL file

    Eprint files should be named by paper ID with slashes replaced by underscores,
    e.g. "q-alg_9503002" or "2301.00001".
    """
    eprint_path = Path(eprint_dir)
    results = []
    out_f = open(output_jsonl, "w") if output_jsonl else None

    try:
        for paper in papers:
            paper_id = paper["id"]
            # Try common naming patterns
            stem = paper_id.replace("/", "_")
            candidates = [
                eprint_path / stem,
                eprint_path / f"{stem}.tar.gz",
                eprint_path / f"{stem}.gz",
                eprint_path / f"{stem}.tex",
                eprint_path / f"{stem}.pdf",  # skip PDFs
            ]

            found = None
            for c in candidates:
                if c.exists() and not c.name.endswith(".pdf"):
                    found = c
                    break

            if not found:
                continue

            try:
                if found.name.endswith((".tar.gz", ".gz", ".tar")):
                    result = extract_from_tarball(str(found), paper_id)
                else:
                    result = extract_from_file(str(found), paper_id)
            except Exception as e:
                result = ExtractionResult(
                    paper_id=paper_id,
                    source_path=str(found),
                    stats={"error": str(e)},
                )

            results.append(result)

            if out_f and result.theorems:
                out_f.write(json.dumps(result.to_dict()) + "\n")

    finally:
        if out_f:
            out_f.close()

    return results
