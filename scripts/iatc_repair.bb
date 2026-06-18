#!/usr/bin/env bb
;; Mechanical IATC-graph canonicalization (deterministic, no LLM judgment) —
;; closes the conformance gaps the model slips on, all checkable from the graph
;; itself, so no consent gate is needed:
;;   1. coerce invalid node :kinds to the valid set {:object :claim :ref}: a
;;      :citation node -> :ref + :citation field (so ref-resolved? holds); any
;;      other unknown kind -> :claim (it carries :text, i.e. an assertion);
;;   2. back-fill every edge's :source from its premise/conclusion node :source
;;      lines (span min..max) when the edge omits :source;
;;   3. mirror every :missing-warrant edge into the top-level :holes vector with a
;;      {:kind :missing-warrant :edge <edge-id> :wanted X} entry — argcheck matches
;;      a hole to an edge by :edge/:id/:target == edge :id (not by :wanted alone).
;; Reads + rewrites the EDN file in place. No-op if the text isn't a parseable map.
(require '[clojure.edn :as edn])

(let [path (first *command-line-args*)
      g    (try (edn/read-string (slurp path)) (catch Exception _ nil))]
  (when (map? g)
    (let [fix-kind  (fn [n]
                      (case (:kind n)
                        (:object :claim :ref) n
                        :citation (assoc n :kind :ref :citation (or (:citation n) (:text n) true))
                        (assoc n :kind :claim)))
          nodes     (mapv fix-kind (:nodes g))
          node-src  (into {} (map (juxt :id :source)) nodes)
          backfill  (fn [e]
                      (if (:source e)
                        e
                        (let [ls (->> [(:premise e) (:conclusion e)]
                                      (mapcat #(get-in node-src [% :lines]))
                                      (remove nil?))]
                          (if (seq ls)
                            (assoc e :source {:lines [(apply min ls) (apply max ls)]})
                            e))))
          edges     (mapv backfill (:edges g))
          holes     (vec (:holes g))
          covered   (set (mapcat (fn [h]
                                   (when (= :missing-warrant (:kind h))
                                     (keep h [:edge :id :target])))
                                 holes))
          new-holes (for [e edges
                          :when (= :missing-warrant (get-in e [:warrant :kind]))
                          :when (not (contains? covered (:id e)))]
                      {:kind :missing-warrant
                       :edge (:id e)
                       :wanted (get-in e [:warrant :wanted])})]
      (spit path (pr-str (assoc g :nodes nodes :edges edges :holes (into holes new-holes)))))))
