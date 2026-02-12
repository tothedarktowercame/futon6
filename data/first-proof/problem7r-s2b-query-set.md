# p7r-s2b Query Set (Order-2 Lattice Instantiation Hunt)

## Goal

Find paper-level routes to complete node `p7r-s2b`:

- explicit uniform lattice family with order-2 torsion;
- finite-group action model compatible with Fowler's `FH(Q)` criterion;
- fixed-set Euler-vanishing conditions verifiable.

## Reproducible search

Run:

```bash
python3 scripts/search-arxiv-p7r-s2b.py \
  --output data/first-proof/problem7r-s2b-arxiv-results.json \
  --max-results 20
```

Current result count: 19 unique arXiv hits.

## Query bundles

### Bundle A: direct group-realization / lattice witnesses

- `all:"Finite Groups and Hyperbolic Manifolds"`
- `all:"Belolipetsky" AND all:"Lubotzky"`
- `all:"uniform lattice" AND all:"order 2" AND all:"SO(n,1)"`
- `all:"arithmetic lattice" AND all:"SO(n,1)" AND (all:"involution" OR all:"Z/2")`

### Bundle B: fixed-set geometry for involutions/symmetries

- `all:"finite group actions" AND all:"hyperbolic manifolds"`
- `all:"involution" AND all:"hyperbolic manifold" AND all:"fixed points"`
- `all:"compact hyperbolic" AND all:"involution"`
- `all:"reflection" AND all:"hyperbolic manifold" AND all:"fixed"`

### Bundle C: orbifold/Euler interface and subgroup control

- `all:"orbifold fundamental group" AND all:"finite group action" AND all:"Euler characteristic"`
- `all:"uniform lattice" AND all:"torsion" AND all:"orbifold"`
- `all:"On the number of finite subgroups of a lattice"`
- `all:"Smith theory" AND all:"locally symmetric manifolds"`

## First-pass top candidates (for p7r-s2b)

### A-tier (most likely compositional value)

1. `math/0406607` — *Finite Groups and Hyperbolic Manifolds* (Belolipetsky-Lubotzky)
   - Why: gives realization of prescribed finite groups as isometry groups of compact hyperbolic manifolds; strongest direct route toward order-2 witness families.
2. `2506.23994` — *On reflections of congruence hyperbolic manifolds*
   - Why: explicit reflective symmetries with geometric information on fixed sets; useful for checking fixed-set Euler behavior.
3. `1106.1704` — *Smith theory, L2 cohomology, isometries...*
   - Why: periodic-action constraints on locally symmetric/aspherical spaces; can help determine when fixed-point behavior forces/forbids candidate configurations.

### B-tier (supporting structure)

1. `1209.2484` — *On the number of finite subgroups of a lattice*
   - Why: finite subgroup/isotropy control in lattice settings.
2. `2012.15322` — *Homology bounds for hyperbolic orbifolds*
   - Why: orbifold-side homological control; may help packaging finite-group actions.
3. `1901.00815` — *Symmetries of exotic negatively curved manifolds*
   - Why: symmetry realization mechanisms in negatively curved settings.
4. `1804.03777` — *Equivariant hyperbolization of 3-manifolds...*
   - Why: explicit finite-group equivariant constructions (dimension-3 biased).

## Extraction questions per candidate

For each paper, extract answers to:

1. Does it explicitly produce an order-2 action (or allow choosing `G=Z/2`)?
2. Is the action on a compact manifold/orbifold with lattice-derived `pi_1`?
3. Are fixed sets described (empty, totally geodesic, codimension-1, etc.)?
4. Can one prove `χ(component)=0` for nontrivial subgroup fixed sets?
5. Does the construction preserve cocompact/uniform lattice context in `SO(n,1)` or another semisimple group?

## Immediate execution plan

1. Deep-read `math/0406607` for a concrete `G=Z/2` specialization and fixed-set behavior.
2. Pair with `2506.23994` to get explicit fixed-set geometry candidates.
3. Use `1106.1704` as a constraint filter (eliminate impossible fixed-set/Euler scenarios).
4. Attempt a concrete `p7r-s2b` instantiation draft from those three.
