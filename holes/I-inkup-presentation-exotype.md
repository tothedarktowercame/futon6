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

## Inherited CMY scope account

The Mark7 margin-scope example fixes what the colours mean.  They are pipeline
provenance layers over shared character spans, not abbreviations for
reader-facing qualities:

| Ink | Mark7 layer | Contents in the 0708.2185 example |
|---|---|---|
| C — cyan | S1 marks | symbols, terms, clauses, environments, displays, and structure |
| M — magenta | S3 proof graph | proof bodies, claims, objects, references, and definitions |
| Y — yellow | S4 expository scopes | connections, characterizations, auxiliary constructions, status, generalisations, difficulty assessments, and carved regions |

The renderer mixes these as subtractive ink.  Overlap is therefore the
important observation: a passage can be read simultaneously by several
stages.  Mark7 has no K layer.  Adding one merely to complete the printing
acronym would erase the empirical meaning of the display.

InkUp inherits this CMY observation coordinate system as `mining_layers`; it
does not rename the colours.  Presentation planning is an orthogonal axis.
Every LeanEmit node must be accounted for as `show`, `compress`, or `omit`;
compression and omission require a reason and preserve a link to the typed
source.

The plan records rhetorical roles separately, including
`overview`, `example`, `reduction`, `bridge`, `calculation`, `citation`, and
`closure`.  This is what permits two proofs with the same CMY proportions to
be distinguished: identical stage coverage may conceal very different amounts
of overview, calculation, and example work.

## Mark8 boundary

Mark8 improves selection and auditability around the existing stages: it pins
the model-call budget and allocation policy, adds deterministic pre-call
features, compares S4 selection policies at equal budget, preserves recovery
reasons, and gates new signals before they can steer allocation.  Its declared
mission and implemented slices do not replace the S1/S3/S4 scope vocabularies
or define a new composition law among their spans and proof-graph edges.

Accordingly, InkUp should inherit two different things without conflating
them: Mark7 supplies the observed scope kinds, shared spans, and CMY overlay;
Mark8 supplies the experimental protocol by which a proposed presentation
signal can graduate from report-only observation to generative control.

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
4. **Strategy deficit.** The planned proof contains many calculation/yield
   roles but no overview that covers their common branch or decomposition.
5. **Administrative dominance.** Calculation or closure roles dominate a
   local scope while its typed steps admit a named compression boundary.

Diagnostics are objections with witnesses, not penalties.  Several may be
true simultaneously, and absence of these five is not evidence of clarity.

## Machine boundary

`holes/inkup/presentation-plan.schema.json` is the first interchange contract.
One plan names its source kind and immutable source reference, contains a tree
of scopes, accounts for typed nodes, retains evidenced CMY mining coverage,
assigns presentation roles, and records report-only diagnostics.  A concrete valid instance is
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
the roles and evidence model rather than noise hidden by an average.

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
7. an exploration share remains, so a preferred presentation profile cannot
   become a self-confirming house style.

Only after these gates should a flexiarg select presentation actions.  A later
proof-generation use is then structurally straightforward: generation proposes
typed proof branches, InkUp evaluates alternative presentation plans, and
PaperPrinter realizes the selected plan.  Proof validity remains outside and
prior to that choice.

## First bounded implementation slice

1. Adapt Mark7 candidate records into the plan schema without changing their
   content.
2. Adapt one accepted PaperPrinter proof and retain its typed-node identities.
3. Retain CMY mining coverage and hand-annotate presentation roles for ten
   matched or comparable local scopes, including an example-rich scope and a
   sufficiency wall.
4. Render a side-by-side scope tree with CMY coverage, presentation roles, and
   diagnostic witnesses.
5. Freeze that set as the v0 fixture before implementing an automatic
   detector.

The output of this slice is a calibration surface and an accounting test, not
a quality score and not a prose rewrite.
