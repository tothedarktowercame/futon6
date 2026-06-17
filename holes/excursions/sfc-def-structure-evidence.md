# SFC Definition Structure Evidence

H-SFC2a transducer: `scripts/sfc_def_structure.bb`

## Inputs and emitted structures

### L-closure worked example

Source: live D4 worked example.

Formula:

```tex
\overline{M}=\{x\in X\mid \forall f,g:X\to Y\,.\,(f|_M=g|_M\,\Rw\,f\cdot x\cong g\cdot x)\}
```

Emitted `:structure`:

```clojure
(= (overline M)
   (conditional-set
    (∈ x X)
    (forall [f g] (: (→ X Y))
      (implies
       (= (restrict f M) (restrict g M))
       (cong (· f x) (· g x))))))
```

Ungrounded:

```clojure
[{:symbol "M", :grounding :hole}
 {:symbol "X", :grounding :hole}
 {:symbol "Y", :grounding :hole}
 {:symbol "f", :grounding :hole}
 {:symbol "g", :grounding :hole}
 {:symbol "x", :grounding :hole}
 {:symbol "·", :grounding :hole}]
```

### fibrant replacement

Source: `data/warp/def-snippets.json`, `fibrant replacement`, paper `0906.4087`.

Formula:

```tex
f:X\to Y
```

Emitted `:structure`:

```clojure
(: f (→ X Y))
```

### homotopy category

Source: `data/warp/def-snippets.json`, `homotopy category`, paper `0707.0300`.

Formula:

```tex
\gamma:\mathcal M\to\Ho(\mathcal M)
```

Emitted `:structure`:

```clojure
(: γ (→ M (* \Ho M)))
```

### homotopy equivalence

Source: `data/warp/def-snippets.json`, `homotopy equivalence`, paper `0708.2185`.

Formula:

```tex
Qf\cong Qg
```

Emitted `:structure`:

```clojure
(cong (* Q f) (* Q g))
```

## Gates

- `clj-kondo --lint scripts/sfc_def_structure.bb`
- `emacs --batch -l /home/joe/code/futon4/dev/check-parens.el scripts/sfc_def_structure.bb`
- `bb tests/sfc_def_structure_test.clj`

