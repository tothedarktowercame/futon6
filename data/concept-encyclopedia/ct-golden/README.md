# Golden concept seeds (hand-authored, APM-Xi level)

Five fully-formalised CT concept definitions — the **noun-side few-shot set** for
the mark3 superpod concept-formaliser, mirroring the 13 IATC argument graphs on
the verb side. These are what `data/concept-encyclopedia/ct/*.edn` (the cheap,
auto-assembled scaffold) should be *deepened into*: its `:gloss` + clause-level
`:components` carry a `:formalise` hole; these golden entries show the target with
**no holes**.

The APM-Xi structure-first form (cf. Joe's `~/Xi.tex`, e.g. *solvable group*):
- `:genus` — what the concept specialises.
- `:given` — the typed inputs (`{:var … :type …}`), e.g. `C : category`.
- `:data` — the structural data the concept carries (a functor's object/morphism
  maps; a monad's `T, μ, η`).
- `:axioms` — the defining conditions as **typed ∀/∃ statements**, each with
  `:refs` to the concepts it depends on.
- (`:equivalent-to` for alternative characterisations, e.g. unit/counit adjunctions.)

Seeds: `functor`, `natural-transformation`, `adjunction`, `monad`,
`abelian-category` — chosen to span the forms: structure-with-axioms (functor,
monad), morphism-of-functors with a naturality square (natural transformation),
a universal property (adjunction), and a property-defined object (abelian
category).

## Theorems are scopes too (a class of definition)

A **scope** is: typed `:given` (with relations among them) → a produced output.
A *definition* produces defining `:axioms`; a **theorem** produces a `:conclusion`
relation, justified by a `:proof`. Same shape, `:kind :theorem`.

`yoneda-lemma.edn` is the golden theorem seed. The schema variant:
- `:given` — the hypotheses (typed vars + their relations).
- `:conclusion` — the produced relation (`Nat(C(c,−),F) ≅ F(c)`), with `:natural-in`
  and `:refs`.
- `:proof` — `:via :iatc-argument-graph`: a theorem's proof IS a verb-side IATC
  graph; `:uses` lists the lemmas it **imports**.
- `:exports` — the downstream constructs the theorem enables.

This makes the **interface** explicit: a theorem imports its hypotheses + `:uses`
lemmas and exports its `:conclusion` + `:exports` — so it is a first-class node in
the dependency tapestry, and a proof's `:missing-warrant` hole is just an
unresolved lemma-import. Theorem *statements* are already detected as
`env/theorem`/`env/lemma` scopes; this is the structure-first treatment of the
statement, the noun/verb hinge between the encyclopedia (nouns) and the IATC
graphs (verbs).

**Handoff contract for the superpod:** given a scaffold entry (gloss + provenance
+ corpus definition passages + concept-dependency edges) and these seeds as
few-shot examples, emit the same schema with `:given` / `:data` / `:axioms`
filled and `:holes []`. Self-gate against a checker (the noun-side analogue of the
IATC checker): every `:refs` target must resolve to another encyclopedia entry;
every axiom must be a well-formed typed statement.
