"""Mark 3 coverage-discipline records (DERIVE item: coverage-record schema).

Mission M-superpod-mark3 §MAP F-1 found two *silent* failure modes that
mark2 could not tell apart:

  * triple-level  — reverse-morphogenesis emits a null ``analysis`` for
    the paper (≈4 % of papers);
  * hypergraph-level — the hypergraph has zero claims (``n_claims == 0``,
    ≈32 % of papers), so Stage 6 has nothing to analyse even when the
    triple itself is fine.

The coverage property (no nulls) requires every paper to leave the batch
with an explicit record: either covered, or failed with a *named* reason.
This module builds those records and summarizes a batch. Nothing here
calls the LLM or touches the network; it consumes the outputs of
``arxiv_pattern_prompt.parse_arxiv_pattern_response`` and the
reverse-morphogenesis stage.

Failure-mode precedence (first match wins):

  1. ``stage3-parse-error``      — the pattern response did not parse.
  2. ``triple-null-analysis``    — the triple's ``analysis`` is null
                                   (the triple-level silent fail).
  3. ``slot-collapse``           — situation / salience / constraint
                                   slots are not three distinct values.
  4. ``hypergraph-zero-claims``  — ``n_claims == 0`` (the hypergraph-level
                                   silent fail).
  5. otherwise ``covered``.

All reasons are distinct, non-null strings — a batch summary that counts
them cannot fold the two silent modes back together.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

FAILED = "failed"
COVERED = "covered"

# The named failure modes, in precedence order. Exported so tests and the
# batch runner share one spelling.
REASON_STAGE3_PARSE_ERROR = "stage3-parse-error"
REASON_TRIPLE_NULL_ANALYSIS = "triple-null-analysis"
REASON_SLOT_COLLAPSE = "slot-collapse"
REASON_HYPERGRAPH_ZERO_CLAIMS = "hypergraph-zero-claims"

SLOT_KEYS = ("situation", "salience", "constraint")


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _slots_distinct(analysis: Mapping[str, Any]) -> bool:
    """Slot-distinctness over the situation_S / xiang_salience /
    arrow_constraint triple: all three present and pairwise distinct.

    A collapsed record (same string in two slots, or a missing slot) is
    rejected here — the mission's repeatable property depends on the three
    slots carrying distinguishable content.
    """
    values = []
    for key in SLOT_KEYS:
        value = analysis.get(key)
        if _is_null(value):
            return False
        values.append(str(value).strip())
    return len(set(values)) == len(values)


def build_coverage_record(
    paper_id: str,
    parse_result: Optional[Mapping[str, Any]],
    analysis: Optional[Mapping[str, Any]] = None,
    n_claims: Optional[int] = None,
) -> dict:
    """Build one explicit coverage record for a paper.

    ``parse_result`` is the dict from
    ``parse_arxiv_pattern_response`` (``ok`` False => stage-3 parse
    failure). ``analysis`` is the reverse-morphogenesis triple record
    (``None`` or empty => null-analysis failure). ``n_claims`` is the
    hypergraph claim count.
    """
    record: dict[str, Any] = {"paper_id": paper_id}

    if parse_result is None or not parse_result.get("ok"):
        error = (parse_result or {}).get("error") or "missing-parse-result"
        record.update(
            status=FAILED,
            reason=REASON_STAGE3_PARSE_ERROR,
            detail=str(error),
        )
        return record

    if not isinstance(analysis, Mapping) or not analysis:
        record.update(
            status=FAILED,
            reason=REASON_TRIPLE_NULL_ANALYSIS,
            detail="reverse-morphogenesis analysis is null or absent",
        )
        return record

    if not _slots_distinct(analysis):
        record.update(
            status=FAILED,
            reason=REASON_SLOT_COLLAPSE,
            detail="situation/salience/constraint slots collapsed or incomplete",
        )
        return record

    if n_claims is None or n_claims <= 0:
        record.update(
            status=FAILED,
            reason=REASON_HYPERGRAPH_ZERO_CLAIMS,
            detail=f"n_claims={n_claims!r}",
        )
        return record

    record.update(
        status=COVERED,
        reason=None,
        family=parse_result.get("family"),
        leaf=parse_result.get("leaf"),
        n_claims=n_claims,
    )
    return record


def summarize_batch(records: Iterable[Mapping[str, Any]]) -> dict:
    """Summarize coverage over a batch.

    The summary counts each failure reason separately, so the two
    silent-fail modes (triple-null-analysis, hypergraph-zero-claims)
    stay distinguishable in the batch record — the exact defect the
    mission's coverage property exists to fix.
    """
    counts: dict[str, int] = {}
    total = 0
    for record in records:
        total += 1
        if record.get("status") == COVERED:
            counts[COVERED] = counts.get(COVERED, 0) + 1
        else:
            reason = record.get("reason")
            if not isinstance(reason, str) or not reason:
                # A failed record with no named reason is itself a defect;
                # surface it instead of silently counting it as covered.
                reason = "unnamed-failure"
            counts[reason] = counts.get(reason, 0) + 1
    covered = counts.get(COVERED, 0)
    return {
        "total": total,
        "covered": covered,
        "coverage_rate": (covered / total) if total else 0.0,
        "failure_counts": {
            key: value for key, value in counts.items() if key != COVERED
        },
    }
