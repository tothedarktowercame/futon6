(ns iatc-anchor-faithfulness-test
  (:require [clojure.test :refer [deftest is run-tests]]))

(load-file "scripts/iatc_anchor_faithfulness.bb")

(defn script-fn [sym]
  (or (resolve sym)
      (throw (ex-info "script var did not load" {:symbol sym}))))

(defn text-terms* [s]
  ((script-fn 'text-terms) s))

(defn check-graph* [graph ctx opts]
  ((script-fn 'check-graph) graph ctx opts))

(deftest extracts-content-and-math-terms
  (is (= ["locally" "cartesian" "closed" "category" "extensional"]
         (text-terms* "every locally cartesian closed category is extensional")))
  (is (= ["calmod" "bicategories" "ring" "isomorphisms"]
         (text-terms* "\\calMod-like bicategories ought to include ring isomorphisms"))))

(deftest check-graph-contract-shape
  (let [graph {:nodes [{:id :x
                        :kind :claim
                        :text "locally cartesian category"
                        :source {:lines [1 1]}}]}
        ctx {:paper-id "toy"
             :source "memory"
             :lines ["A locally cartesian category appears here."]}
        result (check-graph* graph ctx {:k 2 :tau 0.45 :floor 0.30})]
    (is (= :anchor-faithfulness (:check result)))
    (is (:pass result))
    (is (= 1.0 (:rate result)))
    (is (empty? (:reasons result)))
    (is (= [:x] (mapv :id (:per-item result))))))

(deftest flags-empty-anchor-for-extensional-claim
  (let [result (check-graph*
                "data/iatc-argument-graphs/loop-run-70b/0709.0248.edn"
                {}
                {:k 2 :tau 0.45 :floor 0.30})
        flagged-ids (set (map :id (:reasons result)))]
    (is (contains? flagged-ids :extensional-category))
    (is (some #(= "extensional" %) (:missing (first (filter #(= :extensional-category (:id %))
                                                             (:per-item result))))))))

(deftest scores-0706-high
  (let [result (check-graph*
                "data/iatc-argument-graphs/loop-run-70b/0706.1286.edn"
                {}
                {:k 2 :tau 0.45 :floor 0.30})]
    (is (>= (:rate result) 0.85))))

(let [{:keys [fail error]} (run-tests)]
  (when (pos? (+ fail error))
    (System/exit 1)))
