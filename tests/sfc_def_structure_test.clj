(require '[clojure.test :refer [deftest is run-tests]])

(load-file "scripts/sfc_def_structure.bb")

(declare transduce-formula)

(def l-closure
  "\\overline{M}=\\{x\\in X\\mid \\forall f,g:X\\to Y\\,.\\,(f|_M=g|_M\\,\\Rw\\,f\\cdot x\\cong g\\cdot x)\\}")

(def l-target
  (list '= (list 'overline 'M)
        (list 'conditional-set
              (list '∈ 'x 'X)
              (list 'forall ['f 'g] (list (symbol ":") (list '→ 'X 'Y))
                    (list 'implies
                          (list '= (list 'restrict 'f 'M) (list 'restrict 'g 'M))
                          (list 'cong (list '· 'f 'x) (list '· 'g 'x)))))))

(def even-naturals
  "\\{ n \\in \\mathbb{N} \\mid \\exists k . n = 2 k \\}")

(def even-naturals-target
  (list 'conditional-set
        (list '∈ 'n 'ℕ)
        (list 'exists ['k]
              (list '= 'n (list '* 2 'k)))))

(defn contains-symbol? [needle form]
  (cond
    (= needle form) true
    (seq? form) (boolean (some #(contains-symbol? needle %) form))
    (vector? form) (boolean (some #(contains-symbol? needle %) form))
    :else false))

(deftest l-closure-yields-d4-target
  (let [result (transduce-formula l-closure)]
    (is (= l-target (:structure result)))
    (is (= [{:vars ["f" "g"], :type "X\\to Y"}] (:binder-captures result)))
    (is (some #(= {:symbol "·" :grounding :hole} %) (:ungrounded result)))))

(deftest deterministic-output
  (is (= (select-keys (transduce-formula l-closure) [:structure :ungrounded])
         (select-keys (transduce-formula l-closure) [:structure :ungrounded]))))

(deftest parses-simple_formula_from_snippet_feedstock
  (let [result (transduce-formula "f:X\\to Y")]
    (is (= (list (symbol ":") 'f (list '→ 'X 'Y)) (:structure result)))
    (is (= #{"X" "Y" "f"} (set (map :symbol (:ungrounded result)))))))

(deftest preserves-exists-and-mathbb-in-set-builder
  (let [result (transduce-formula even-naturals)]
    (is (= even-naturals-target (:structure result)))
    (is (not (contains-symbol? 'formulae-sequence (:structure result))))
    (is (not (contains-symbol? (symbol "\\mathbb") (:structure result))))
    (is (some #(= {:symbol "k" :grounding :hole} %) (:ungrounded result)))))

(deftest normalizes-typed-multi-var-exists
  (let [result (transduce-formula "\\exists x,y:X\\to Y\\,. x=y")]
    (is (= (list 'exists ['x] (list 'exists ['y] (list '= 'x 'y)))
           (:structure result)))
    (is (= [{:vars ["x" "y"], :type "X\\to Y"}]
           (:binder-captures result)))))

(let [summary (run-tests)]
  (when (pos? (+ (:fail summary) (:error summary)))
    (System/exit 1)))
