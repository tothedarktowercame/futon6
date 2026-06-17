(require '[clojure.test :refer [deftest is run-tests]])

(load-file "scripts/sfc_def_structure.bb")

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

(let [summary (run-tests)]
  (when (pos? (+ (:fail summary) (:error summary)))
    (System/exit 1)))
