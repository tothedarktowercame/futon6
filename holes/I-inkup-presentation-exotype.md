# I-InkUp — proof-presentation exotype before prose realization

Status: proposed, report-only

## Claim

Put a presentation-planning layer between typed proof emission and prose
realization:

`LeanEmit -> InkUp -> PaperPrinter`

The provisional name is **InkUp**.  The architectural boundary, not the name,
is the proposal.  LeanEmit says what is justified.  InkUp accounts for what a
reader must be shown, where, and at what resolution.  PaperPrinter realizes
that plan as text.  InkUp must not invent a proof fact, and PaperPrinter must
not silently change the plan.

This makes proof presentation a generative object rather than a score assigned
after prose exists.  Initially InkUp is report-only: it annotates an existing
human or generated proof and emits diagnostics, but it neither suppresses
content nor steers generation.

## Why there is enough data for a first slice

The archived Mark7 probe contains 125 candidates, 724 graphs, 1,111
expository artifacts, 2,420 expository candidates, and 12 marked papers.  Its
candidate records preserve human `source-window` text together with proof
lines, binder context, citations, and typed enrichment.  Mark8 already
requires frozen pre-call features, immutable accounting, deterministic replay,
and report-only status until an offline and prospective gate is met.

Those artifacts support comparative measurements of human and generated local
scopes.  They do **not** yet support the assertion that a proof is clearly
written.  The first slice therefore produces inspectable measurements and
specific objections, not a scalar quality verdict.

## CMYK scope account

Each local proof scope receives four non-exclusive ink channels.  The names
are provisional; keeping the four-channel vector and its evidence is the
contract.

| Channel | Reader-facing work | Typical material |
|---|---|---|
| C — context | Establish what is locally in force | assumptions, definitions, notation, branch conditions |
| M — method | Make the proof strategy and reductions visible | overview, decomposition, lemma choice, why a move advances the proof |
| Y — yield | State what the scope contributes | intermediate claims, examples, consequences, discharged goals |
| K — kernel | Expose or compress formal support | calculations, rewrites, citations, routine closure, administrative steps |

An item can contribute to several channels.  Ink is measured from spans or
typed nodes and is never inferred merely from a colour label.  Every LeanEmit
node must be accounted for as `show`, `compress`, or `omit`; compression and
omission require a reason and preserve a link to the typed source.

The plan also records rhetorical roles beneath the channels, including
`overview`, `example`, `reduction`, `bridge`, `calculation`, `citation`, and
`closure`.  This is what permits two proofs with the same CMYK proportions to
be distinguished: for example, Y ink may consist entirely of intermediate
claims and contain no example.

## First diagnostics (specific, not “clear”)

1. **Example scarcity.** Report example-role ink and example count relative to
   theorem, section, and comparison-corpus strata.  Say “no example occurs in
   this scope” before learning any threshold for “thin on examples.”
2. **Sufficiency wall.** Find runs of reduction/sufficiency moves with no
   preceding method overview and little measured frontier reduction.  Report
   the run, the goals before and after it, and the candidate overview site.
3. **Unintroduced dependency.** A shown claim depends on a hypothesis not
   introduced in its visible context.  This generalizes the failure exposed by
   the PaperPrinter C23 audit.
4. **Strategy deficit.** The proof contains many K/Y details but no M span that
   covers their common branch or decomposition.
5. **Administrative dominance.** K ink dominates a local scope while its
   typed steps admit a named compression boundary.

Diagnostics are objections with witnesses, not penalties.  Several may be
true simultaneously, and absence of these five is not evidence of clarity.

## Machine boundary

`holes/inkup/presentation-plan.schema.json` is the first interchange contract.
One plan names its source kind and immutable source reference, contains a tree
of scopes, accounts for typed nodes, assigns evidenced CMYK weights and roles,
and records report-only diagnostics.  A concrete valid instance is
`holes/inkup/example-plan.json`.

PaperPrinter may consume `show` and `compress` directives only after a later
gate.  Until then the same schema is used to annotate both human Mark7 source
windows and generated proofs, which prevents the measurement vocabulary from
being tailored only to PaperPrinter's current pathologies.

## Calibration design

The unit of comparison is a local scope, nested in a proof and paper—not a
whole-paper bag of words.  Initial strata are source kind (human/generated),
proof role, scope depth, proof length, and mathematical domain.  Retain paired
views whenever a generated proof and its human source or reference proof are
available.

The first annotation exercise should sample scopes selected independently of
the five diagnostics, then ask readers for pairwise, actionable judgements:
which version better exposes the strategy, which dependency is missing, and
whether an example changes understanding.  Do not initially ask for a single
“clearly written” label.  Pairwise disagreements become evidence for refining
the roles and channels rather than noise hidden by an average.

## Promotion gates

InkUp may influence generated presentation only after all of these hold:

1. deterministic replay from the same LeanEmit/Mark7 inputs;
2. complete typed-node accounting, with no unexplained omission;
3. span or node precision reviewed separately for every diagnostic;
4. frozen human/generated calibration and a held-out paper set;
5. comparison against simple baselines such as token count, proof length, and
   repeated connective count;
6. prospective evidence that a proposed change improves the targeted reader
   judgement without increasing dependency loss or proof error;
7. an exploration share remains, so a preferred CMYK profile cannot become a
   self-confirming house style.

Only after these gates should a flexiarg select presentation actions.  A later
proof-generation use is then structurally straightforward: generation proposes
typed proof branches, InkUp evaluates alternative presentation plans, and
PaperPrinter realizes the selected plan.  Proof validity remains outside and
prior to that choice.

## First bounded implementation slice

1. Adapt Mark7 candidate records into the plan schema without changing their
   content.
2. Adapt one accepted PaperPrinter proof and retain its typed-node identities.
3. Hand-annotate CMYK weights and roles for ten matched or comparable local
   scopes, including an example-rich scope and a sufficiency wall.
4. Render a side-by-side scope tree with channel proportions and diagnostic
   witnesses.
5. Freeze that set as the v0 fixture before implementing an automatic
   detector.

The output of this slice is a calibration surface and an accounting test, not
a quality score and not a prose rewrite.
