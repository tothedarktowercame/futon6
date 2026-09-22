# T-argument-outside-of-proof — statements no layer reads

**Type:** T-prefix ticket (first of its kind in this repo; E- is an excursion,
TN- a tech note, M- a mission). **Raised:** 2026-09-22 by Joe, reviewing
`/wip/mark7-0806.1324-margin.html`. **Status:** open, parked deliberately.

## What was seen

In Krause, *Localization theory for triangulated categories* (0806.1324), the
theorem at L240–250 lists equivalent conditions, among them:

> (3) There exists a cohomological functor $H\colon\T\to\A$ into a locally
> presentable abelian category such that $H$ preserves small coproducts and
> $\Sc=\Ker H$.

Joe: *"'There exists' isn't marked up. I'd assume existential claims are
related to IATC, but maybe IATC isn't running on theorem statements?"*

Two separate faults were behind that one line. The first is fixed; this ticket
is the second.

1. **The existential detector saw almost nothing** — fixed in `4cdedaf`. It
   required the formula to follow "There exists" immediately and "There" to be
   capitalised, so 2 existential scopes were recorded across the whole run.
   Now 725 in 9 of 12 papers. The universal quantifier had the same shape of
   defect and was fixed with it.

2. **No layer reads this statement at all.** This is the open part.

## The gap

S3 (IATC) builds one candidate per S1 **proof** region, and attaches the
statement above it when the proof follows within `STATEMENT_GAP` (20 lines).
A statement whose proof is elsewhere, or absent because the result is quoted
from the literature, is therefore never read by S3.

S4 does not pick it up either: a statement environment is a formal block, so
the region carving excludes it from exposition by construction.

Measured over mark7master-20260921, per paper, counting S1's
`env/{theorem,lemma,proposition,corollary}` marks:

| | statements |
|---|---|
| with a proof within 20 lines (S3 reads them) | 124 |
| no proof, but inside an S4 region | 5 |
| **read by neither** | **20** |

13% of the run's statements are mined by nothing. Krause's theorem is one.

Reproduce:

    futon6/.venv/bin/python - <<'PY'
    # counts the three rows above from a run's marks + the current region carving
    PY

(the snippet is in the turn log for 2026-09-22; it is six lines over
`expository_region_extract.extract_regions` and the `env/*` marks.)

## Why it is not obvious what to do

Three options, none free:

1. **Statement-only S3 candidates.** S3's task is "reconstruct the argument of
   this proof". A statement has no argument to reconstruct, so the task would
   have to change for these candidates, and a run's S3 accounting would mix two
   kinds of item.
2. **Let S4 read unproved statements.** Cheap — one predicate in the carving —
   but they are not exposition, and the S4 vocabulary (rationale, connection,
   heuristic) has nothing to say about a theorem statement.
3. **A statement-reading step of its own**, whose product is the claim's
   structure: what is quantified over, what is assumed, what is concluded, what
   it cites. This is what the example calls for — the existential in condition
   (3) is exactly such a structure — and it would consume the quantifier scopes
   fixed in `4cdedaf`. It is also a new stage, a new contract term, and a new
   schema.

Claude-9's recommendation is (3), narrow: no new model call where a proof
already exists, only for statements S3 does not reach.

## Next step

Joe to choose between the three, or to reject the premise (a statement with no
proof in the paper may simply not be this pipeline's business). Nothing is
being built in the meantime.
