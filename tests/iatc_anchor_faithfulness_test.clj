(ns iatc-anchor-faithfulness-test
  (:require [clojure.test :refer [deftest is run-tests]]
            [clojure.java.io :as io]
            [cheshire.core :as json]))

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
         (text-terms* "\\calMod-like bicategories ought to include ring isomorphisms")))
  (is (= ["homomorphism" "maps" "group"]
         (text-terms* "\\textbf{homomorphism} $t\\maps H\\to \\G$")))
  (is (= ["object"]
         (text-terms* "\\Ob(\\mathcal{C})"))))

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

(deftest sparse-node-is-na-not-fail
  (let [graph {:nodes [{:id :h
                        :kind :object
                        :text "group H"
                        :source {:lines [1 1]}}]}
        ctx {:paper-id "toy"
             :source "memory"
             :lines ["Let H be a group."]}
        result (check-graph* graph ctx {:k 2 :tau 0.45 :floor 0.30})
        item (first (:per-item result))]
    (is (= :na (:status item)))
    (is (false? (:scorable item)))
    (is (= 1.0 (:rate result)))
    (is (empty? (:reasons result)))))

(deftest neighbor-line-tolerance-recovers-crossed-module-anchor
  (let [graph {:nodes [{:id :crossed-module
                        :kind :claim
                        :text "crossed module (G,H,t,a)"
                        :source {:lines [2 2]}}]}
        ctx {:paper-id "toy"
             :source "memory"
             :lines [""
                     "$(G,H,t,a)$ by setting $G$ to be $\\Ob(\\G)$"
                     "Conversely, given a topological $2$-group $\\G$, we define a crossed module"]}
        result (check-graph* graph ctx {:k 2 :tau 0.45 :floor 0.30})]
    (is (:pass result))
    (is (= :pass (:status (first (:per-item result)))))))

(deftest flags-empty-anchor-for-extensional-claim
  (let [result (check-graph*
                {:nodes [{:id :extensional-category :text "extensional category"
                          :source {:lines [1 1]}}]}
                {:lines ["finite dimensional vector spaces"]} {})]
    (is (false? (:pass result)))
    (is (= [:extensional-category] (mapv :id (:reasons result))))))

(deftest candidate-window-coordinates-and-real-file-resolution
  (let [dir (.toFile (java.nio.file.Files/createTempDirectory
                      "anchor-faithfulness" (make-array java.nio.file.attribute.FileAttribute 0)))
        graphs (io/file dir "graphs")
        candidates (io/file dir "candidates")
        file (io/file graphs "toy__p0.edn")
        candidate-file (io/file candidates "toy__p0.candidate.json")
        graph {:paper/id "toy" :passage/id "toy:proof0:L101-104"
               :nodes [{:id :claim :text "locally cartesian category"
                        :source {:lines [101 101]}}]}
        candidate {:paper-id "toy" :passage-id (:passage/id graph)
                   :window-lines [101 104]
                   :source-window "locally cartesian category\n\n\nfinite dimensional vector spaces"}
        check-file* (script-fn 'check-file)]
    (try
      (.mkdirs graphs)
      (.mkdirs candidates)
      (spit file (pr-str graph))
      (spit candidate-file (json/generate-string candidate))
      ;; The old 1-based file slice looks at line 101 of a four-line window.
      (is (zero? (:rate (check-graph* graph {:lines ["locally cartesian category" "" "" "finite dimensional vector spaces"]} {}))))
      (let [good (check-file* {} file)]
        (is (:pass good))
        (is (= 1.0 (:rate good)))
        (is (re-find #"index 0" (:coordinate-convention good))))
      ;; Same words, genuinely wrong anchor, farther than neighbor tolerance.
      (spit file (pr-str (assoc-in graph [:nodes 0 :source :lines] [104 104])))
      (is (false? (:pass (check-file* {} file))))
      (spit file (pr-str (assoc-in graph [:nodes 0 :source :lines] [100 101])))
      (is (false? (:pass (check-file* {} file))))
      (spit file (pr-str graph))
      (spit candidate-file (json/generate-string (assoc candidate :passage-id "wrong")))
      (is (re-find #"identity" (get-in (check-file* {} file) [:reasons 0 :reason])))
      (spit candidate-file (json/generate-string (assoc candidate :window-lines [101 105])))
      (is (nil? (:rate (check-file* {} file))))
      (is (re-find #"bounds exceed" (get-in (check-file* {} file) [:reasons 0 :reason])))
      (.delete candidate-file)
      (is (re-find #"source-window not found" (get-in (check-file* {} file) [:reasons 0 :reason])))
      (finally
        (doseq [f (reverse (file-seq dir))] (.delete f))))))

(let [{:keys [fail error]} (run-tests)]
  (when (pos? (+ fail error))
    (System/exit 1)))
