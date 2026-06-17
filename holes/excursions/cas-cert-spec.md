# CAS-CERT — conformance certificate + residual-sorry map (spec)

*The last Rank-D breakdown, now specified. Author: claude-1, 2026-06-17. Converged from three
directions — Joe + claude-loop's "the certificate IS the substrate made per-paper" note in
`E-informal-proof-checking.md`, my port-ledger framing, and the discursive-core CP. This spec
records the synthesis + the machine schema; the 4-facet human guide is its display.*

## 0. What's settled (the convergence)

- **CAS-CERT is not a new check — it's a deterministic per-paper *aggregator*** over the
  answerable substrate the rungs already produce (the capstone of "the gate is also a describer").
- **Two faces of one object:** machine certificate (the port ledger) + human reader's-guide (the
  4-facet display). **RENDER displays CAS-CERT** — render_run/DEMO-COMPOSE are the surface, the
  certificate is the content. One artifact.
- **Granularity = by grain** (resolves the open fork). Ports organize by the **grain ladder** —
  symbol ⊂ concept ⊂ technique ⊂ proof — i.e. the nested scopes of the discursive core, *not*
  CAS-SEL's per-step unit. Per-item detail lives under each grain. ("coverage-by-grain", Joe.)
- **Conformance is a vector-by-grain, not a single grade** — a single % would launder very
  different gaps together.
- **Gate fails only on a mis-wire** (present-but-wrong), never on an empty port (N/A≠FAIL).
- **Ships now over the built grains;** not-yet-built grains (rung-3 technique, SFC2b symbol) are
  **N/A**, and the cert grows automatically as they land.
- **Honesty boundary (load-bearing):** the certificate asserts *"this is a well-formed wiring —
  every port that should be filled is filled or flagged,"* **not** *"this proof is correct."* No gold
  exists (the parent question); a cert implying "verified" would be exactly the overclaim we keep
  catching. The residual map is the honest other half.

## 1. The port ledger (machine face)

A **port** = a slot some rung asks a yes/no/NA about, tagged with its **grain**:

| grain | ports | producing rung | state today |
|---|---|---|---|
| **symbol** | each symbol → bound to a per-paper domain? | SFC2b | **N/A** (not built) |
| **concept** | each concept → defined/known/imported? | R2d | **live** (8d9acad) |
| **technique** | each move → grounded-by-pattern/citation? | rung-3 / CAS-SEL sorry | rung-3 **N/A**; CAS-SEL `sorry`/`thin` **live** |
| **proof** | node→anchor (R2a) · graph→closure (R2b) · edge→warrant (R2c) | rung-0/1/2 | **live** |

Each port has a **state**: `filled` · `empty` · `miswired` · `na`.
- `filled` = the rung answered the scoped query (concept defined, warrant resolved, anchor faithful).
- `empty` = the scoped query returned nothing → a real open question (R2d undefined, CAS-SEL thin,
  missing warrant, orphan node). **Not a failure.**
- `miswired` = present-but-wrong (R2a flag: node anchored to lines lacking its terms; substance
  self-loop). **The only FAIL condition.**
- `na` = the grain isn't built yet, or the structure isn't present at this resolution.

## 2. Schema (`futon6/cas-cert/v1`)

```json
{ "paper_id": "0706.1286",
  "schema": "futon6/cas-cert/v1",
  "conformance": {                                  // VECTOR by grain, not one grade
    "by_grain": {
      "symbol":    {"filled":0,"empty":0,"miswired":0,"na":true,  "rate":null, "rung":"SFC2b"},
      "concept":   {"filled":2,"empty":2,"miswired":0,"na":false, "rate":0.5,  "rung":"R2d"},
      "technique": {"filled":3,"empty":1,"miswired":0,"na":false, "rate":0.75, "rung":"CAS-SEL/rung-3"},
      "proof":     {"anchor":{"filled":8,"miswired":1},"closure":"PASS","warrant":{"filled":1,"total":5}}
    },
    "headline": "filled across built grains; symbol grain N/A (SFC2b unbuilt)"   // derived, never replaces the vector
  },
  "ports": [ {"grain":"concept","item":"calmod-like bicategory","state":"empty",
              "rung":"R2d","scoped_query":"definition of 'calmod-like bicategory' in substrate",
              "evidence":null} ],
  "residual_sorries": [ {"grain":"concept","kind":"undefined","item":"calmod-like bicategory",
                         "scoped_query":"...","arse_seed":"What defines a calmod-like bicategory?"} ],
  "value_signals": {"centrality":0.0,"novelty":"introduces|applies","connections":[],
                    "conjectures":[],"pct_grounded":0.5},
  "verdict": {"well_wired": true, "miswires": ["0709-style anchor flag if any"], "gate": "PASS"}
}
```
`gate` = `FAIL` iff `miswires` is non-empty; empty/na ports never fail it.

## 2b. Confidence — the reviewer's second axis (Joe, 2026-06-17)

A reviewer rates the paper **and** rates their own confidence. CAS-CERT's `verdict` is the first
axis (kept binary — signal-tuning is premature). **Confidence is a separate, *derived* axis** — not
a new signal to tune, but a deterministic readout of **how complete our analysis was**. The
nested grain ladder (symbol ⊂ concept ⊂ technique ⊂ proof) gives it for free:

> **Confidence in a verdict = the solidity of the grains *below* the one where the verdict's
> determining issue lives.** If the foundation grains are fully analysed and the higher grain still
> fails, the failure is attributable to the paper/reconstruction, not to our incomplete analysis →
> high confidence. If a foundation grain is N/A (we never ran it), the verdict may be an artifact of
> that gap → low confidence.

Joe's worked intuition: *"confident we could ground the symbols, find the scopes, define the
concepts, and **still** couldn't make sense of the claim → confidence is high."*

**Derivation (deterministic, no tuning):**
- A grain is **analysed** if it is not `na` (the rung ran). `solidity(grain)` = fraction of its
  ports that reached a *definitive* state (`filled` or `miswired`) vs indeterminate (`empty`).
- `confidence` = a level (`low`/`medium`/`high`) from (a) how many grains *below the verdict's
  locus* are analysed at all, and (b) their solidity. The cert also names the **limiting factor**
  (which grain is N/A or unresolved), so the report reads like a reviewer:
  *"FAIL · confidence LIMITED — symbol grounding (SFC2b) + technique grounding (rung-3) not yet
  performed."*

**Honest consequence today:** symbol (SFC2b) and technique-full (rung-3) grains are **N/A**, so
**every current certificate is capped at medium confidence**, and must say which grains it couldn't
analyse. Confidence rises **automatically** as SFC2b + rung-3 land — no re-tuning, just more of the
ladder filled. This keeps the honesty boundary sharp: a high-confidence FAIL (full ladder analysed,
still broken) is a strong claim; a low-confidence FAIL is "we couldn't finish looking."

*Status: implemented in `cas_cert.py`; the `confidence` field is a small derived addition
(reads grain N/A + solidity off the existing port ledger). Verdict semantics unchanged.
On the current `loop-run-70b` set, all 9 certificates are `medium` because symbol
(`SFC2b`) and technique (`rung-3`) grains are N/A; those two grains are named as
the limiting factors.*

## 3. The residual-sorry map = facet (3) = the empty ports

The set of `empty` ports IS the residual-sorry map and the ArSE-question seeds, typed by grain:
`thin` (technique) · `undefined` (concept) · `ungrounded` (symbol) · `orphan`/`missing-warrant`
(proof) · `conjecture` (author-declared — credited, a corpus open-problem map). This is the
load-bearing output; rung-3 phrases these as questions via the RM question-pattern menu.

## 4. The human guide (display) — the 4 facets, each a projection of the ledger

| facet | reads from |
|---|---|
| **READ THIS FIRST** (what it's about) | `filled` symbol+concept ports + region skeleton |
| **START HERE** (entry point) | the wiring DAG roots→goal (R2b) + import-descent order (phylogeny) |
| **OPEN QUESTIONS WE COULDN'T FIGURE OUT YET** | `residual_sorries` (§3) |
| **WHY IT'S LIKELY VALUABLE** | `value_signals` (centrality · novelty · connections · conjectures · %grounded) |

RENDER renders this; DEMO-COMPOSE already shows facet (3) (residual sorries highlighted) → it's a
partial guide already, so RENDER↔CAS-CERT is generalization, not new surface.

## 5. Build shape (deterministic aggregator, no model)

- Input: the per-paper rung outputs already on disk — `iatc_semcheck` profile (R2a/R2b/R2c + R2d),
  `cas_select` output (topology/sorry/induce → technique ports), and (when built) SFC2b symbol
  grounding + rung-3 technique verdicts.
- Output: the schema above (`.edn` + the guide as `.md`/HTML via RENDER).
- No LLM: it reads verdicts and partitions ports. `na` for absent grains.
- `--gate`: exit non-zero iff any `miswired` port (mirrors rung-0/1/2 so it can self-gate).

## 6. Dependencies / readiness

- **Buildable now** over the live grains {concept (R2d), proof (R2a/b/c), technique-partial
  (CAS-SEL sorry/thin)}. Symbol (SFC2b) + technique-full (rung-3) ports = N/A until those land.
- Sits downstream of `iatc_semcheck` + `cas_select`; it's their union-aggregator, the same way
  `iatc_semcheck` aggregates R2a/b/c.
- **CAS-SEL-2** (executable check registry) and CAS-CERT share the "matched-pattern → check" map;
  CAS-CERT consumes the `checks` field cas_select already emits.

## 7. Handoff readiness
Dispatchable as one Codex build (`scripts/cas_cert.py`, check-graph-aggregator shape; reuse
`iatc_semcheck` + `cas_select` outputs, no re-check). Acceptance: per-paper port ledger over the
9 loop-run-70b proofs; conformance vector-by-grain; residual map = empty ports; `--gate` FAILs only
on mis-wire; symbol/rung-3 grains N/A; deterministic. Gates: PY (py_compile + pytest) + report the
per-paper vectors. The 4-facet guide display is a RENDER follow-on, not this build.

## Findings — CAS-CERT (codex-1)

Implemented `scripts/cas_cert.py` as a deterministic reader/aggregator over emitted
rung outputs:

- `iatc_semcheck` supplies proof-grain ports from R2a/R2b/R2c and concept-grain
  ports from R2d.
- optional `cas_select.py` JSON supplies partial technique-grain CAS-SEL ports;
  absent per-paper CAS-SEL output leaves technique as N/A except for the full
  rung-3 technique port, which is always N/A for now.
- symbol grain is explicitly N/A (`SFC2b` not wired).

Port mapping:

- R2a anchor `:pass` -> `filled`; R2a anchor `:fail` -> `miswired`.
- R2b orphan nodes -> `empty` residuals of kind `orphan`; R2b cycles -> `miswired`.
- R2c resolved warrants -> `filled`; missing/absent warrants -> `empty` residuals
  of kind `missing-warrant`.
- R2d `defined`/`known`/`imported` -> `filled`; `undefined` -> `empty` residuals
  of kind `undefined`.
- CAS-SEL declared sorries -> `filled`; CAS-SEL thin sorries -> `empty` residuals
  of kind `thin`.

Live run over `data/iatc-argument-graphs/loop-run-70b` using the emitted semcheck
profile:

| paper | gate | miswires | residual sorries | concept rate | proof rate |
|---|---:|---:|---:|---:|---:|
| 0705.0452 | FAIL | 4 | 9 | 1.000 | 0.235 |
| 0706.1286 | PASS | 0 | 6 | 0.500 | 0.692 |
| 0708.1921 | PASS | 0 | 7 | 0.500 | 0.000 |
| 0708.2067 | FAIL | 2 | 7 | 1.000 | 0.357 |
| 0709.0248 | FAIL | 2 | 4 | 0.800 | 0.600 |
| 0711.0473 | FAIL | 1 | 2 | 1.000 | 0.667 |
| 0712.0724 | FAIL | 1 | 8 | 1.000 | 0.250 |
| 0801.0199 | PASS | 0 | 2 | 1.000 | 0.818 |
| 0801.3843 | FAIL | 1 | 1 | 1.000 | 0.800 |

`--gate` exits non-zero on the live set because miswired ports exist. Empty ports
do not fail the gate: `0708.1921` has proof empty ports and residual sorries but
no miswires, so its certificate gate is `PASS`.

Gates passed:

- `python3 -m py_compile scripts/cas_cert.py`
- `pytest -q tests/test_cas_cert.py` (`4` passed)

## Review — claude-1 (author ≠ reviewer), 2026-06-17 · commit 85ce000

**Verdict: PASS, clean — no amendments.** Checked: re-ran gates (py_compile OK; pytest 4/4);
read the diff. It's a genuine deterministic reader — no model, no re-checking; the lone
`subprocess` reads `iatc_semcheck.bb --out` (reads verdicts, doesn't re-implement them). Gate
logic correct: `gate=FAIL` iff a `miswired` port exists; `--gate` exits 1 only then; empty/na
never fail. Conformance is a vector-by-grain (concept/proof separate), not a single grade.

**Spot-checked port states against ground truth (the CAS-SEL-3 lesson):**
- `0709.0248` → FAIL, miswires `anchor::extensional-category` + `anchor::reflexivity-term` —
  trace exactly to the known R2a proposition-anchor flags. Correct.
- `0708.1921` → **PASS despite proof_rate 0.000** — its 6 proof ports are all *empty* (orphan +
  missing-warrant), zero miswired. This is the honesty boundary working as designed: a
  0%-grounded graph passes the conformance gate because nothing is *mis-wired*; the residual map
  carries all 6 open ports. Flagging it loudly here so it's not later mistaken for a bug — it's
  the spec's load-bearing distinction (cert = "well-wired or flagged", NOT "verified").

Residual kinds typed correctly (undefined / orphan / missing-warrant / thin). Aggregate gate
FAIL over loop-run-70b (miswires exist in 6/9) — the honest, expected result.

**CAS-CERT COMPLETE.** The capstone aggregator is live over the proof + concept grains; technique
(rung-3) + symbol (SFC2b) grains wire in as N/A and grow automatically when those land. Feeds the
4-facet RENDER guide + the ArSE-question seeds.

### Confidence field — REVIEWED PASS (claude-1, 2026-06-17) · commit 822294e
Re-ran gates (py_compile OK; pytest 6/6). Verified the load-bearing invariant: **verdict
byte-unchanged** — PASS set still exactly {0706.1286, 0708.1921, 0801.0199}, aggregate FAIL,
identical to pre-confidence. All 9 certs = `medium`, each naming exactly {symbol grain N/A —
SFC2b, technique grain N/A — rung-3} as limiting factors — the reviewer voice the design called
for. Note: codex treated symbol/concept/**technique** as foundation grains (I'd said symbol/concept);
that's *more* correct — technique is below proof in the nested ladder, so a current verdict honestly
rests on two unanalysed foundation grains → capped at medium. Rises automatically as SFC2b/rung-3 land.
