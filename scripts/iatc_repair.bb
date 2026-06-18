#!/usr/bin/env bb
;; Mechanical IATC-graph canonicalization (deterministic, no LLM judgment) —
;; closes the conformance gaps the model slips on, all checkable from the graph
;; itself, so no consent gate is needed:
;;   0. sanitize invalid EDN string escapes (text-level, BEFORE any parse) so raw
;;      LaTeX the model embedded (\circ, \otimes, \times, \nabla, ...) doesn't
;;      either crash the reader or silently corrupt the text — see below;
;;   1. coerce invalid node :kinds to the valid set {:object :claim :ref}: a
;;      :citation node -> :ref + :citation field (so ref-resolved? holds); any
;;      other unknown kind -> :claim (it carries :text, i.e. an assertion);
;;   2. back-fill every edge's :source from its premise/conclusion node :source
;;      lines (span min..max) when the edge omits :source;
;;   3. mirror every :missing-warrant edge into the top-level :holes vector with a
;;      {:kind :missing-warrant :edge <edge-id> :wanted X} entry — argcheck matches
;;      a hole to an edge by :edge/:id/:target == edge :id (not by :wanted alone).
;; Reads + rewrites the EDN file in place. No-op if the text isn't a parseable map
;; (after escape-sanitization).
(require '[clojure.edn :as edn])

;; --- EDN string-escape sanitization (text-level, pre-parse) -----------------
;; The model frequently embeds raw LaTeX inside description strings — e.g.
;; "u \circ \phi", "A \otimes B", "\nabla f". Two distinct hazards:
;;  (a) LOUD: a backslash followed by a char that isn't a legal EDN escape makes
;;      the reader throw ("Unsupported escape character: \c"), so the whole graph
;;      fails to parse and is dropped at the gate (found live on 0712.0724, a
;;      category-theory paper, 2026-06-18: \circ -> \c).
;;  (b) SILENT: a few LaTeX commands START with a legal escape letter
;;      (\times -> \t = TAB, \nabla -> \n = newline, \beta -> \b, \rho -> \r,
;;      \frac -> \f), so they parse WITHOUT error and silently corrupt the text —
;;      an anchor-faithfulness (L4) defect the gate never catches.
;; Fix for both: inside double-quoted strings, double the backslash of ANYTHING
;; that isn't a genuine \" , \\ or \uXXXX. This faithfully preserves the literal
;; LaTeX while making the EDN parseable. Mechanical + checkable, so no consent
;; gate (futon3 mechanical-vs-semantic-consent). Known LIMITATIONS and the
;; thornier cases (char literals outside strings; LaTeX->unicode nicety; a proper
;; EDN-aware tokenizer) are tracked in holes/excursions/E-sanitize-invalid-EDN.md.
(defn- hex? [c]
  (or (<= (int \0) (int c) (int \9))
      (<= (int \a) (int c) (int \f))
      (<= (int \A) (int c) (int \F))))

(defn sanitize-edn-escapes
  "Inside double-quoted strings, double the backslash of any non-EDN escape so the
   reader accepts embedded LaTeX. Preserves the genuine escapes \\\" \\\\ \\uXXXX
   (4 hex). Backslashes outside strings are left untouched."
  [s]
  (let [n (count s) sb (StringBuilder.)]
    (loop [i 0 in-str? false]
      (if (>= i n)
        (.toString sb)
        (let [c (.charAt s i)]
          (cond
            (not in-str?)
            (do (.append sb c) (recur (inc i) (= c \")))

            (and (= c \\) (< (inc i) n))
            (let [d (.charAt s (inc i))]
              (cond
                (= d \") (do (.append sb "\\\"") (recur (+ i 2) true))
                (= d \\) (do (.append sb "\\\\") (recur (+ i 2) true))
                (and (= d \u) (<= (+ i 6) n) (every? hex? (subs s (+ i 2) (+ i 6))))
                (do (.append sb (subs s i (+ i 6))) (recur (+ i 6) true))
                :else (do (.append sb "\\\\") (.append sb d) (recur (+ i 2) true))))

            (= c \")
            (do (.append sb c) (recur (inc i) false))

            :else
            (do (.append sb c) (recur (inc i) true))))))))

(let [path (first *command-line-args*)
      raw  (slurp path)
      text (sanitize-edn-escapes raw)
      ;; persist the escape fix even if the structural parse below still fails for
      ;; some OTHER reason — the downstream gate must see clean escapes regardless.
      _    (when (not= text raw) (spit path text))
      g    (try (edn/read-string text) (catch Exception _ nil))]
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
