#!/usr/bin/env bb
;; G-method-vocab gate (linde-stepper-contract S4 postcondition).
;;
;; Every CLean :method (per box + in :clean/seq) and :clean/shape :macro must be in
;; the controlled vocabulary (holes/clean/clean-method-vocab.edn). Catches LLaMA
;; off-vocabulary tags AND un-typed placeholders (:untyped-step / :untyped-macro) —
;; either of which would silently pollute the structure-embedding space if it
;; reached S7. The well-formedness gate (clean_argcheck.bb) checks copar/ports/DAG;
;; THIS gate checks controlled-vocabulary conformance. Both must pass at S4.
;;
;; Usage:
;;   bb scripts/clean_vocab_gate.bb holes/clean/        ;; all *.clean.edn
;;   bb scripts/clean_vocab_gate.bb path/to/x.clean.edn
;; Exits nonzero if any file uses an off-vocabulary or un-typed method/macro.

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def vocab (edn/read-string (slurp "holes/clean/clean-method-vocab.edn")))
(def valid-methods (set (keys (:clean/method-vocab vocab))))
(def valid-macros  (set (keys (:clean/macro-shapes vocab))))

(defn check [m]
  (let [errs (atom [])
        methods (map :method (:clean/boxes m))
        seqm    (:clean/seq m)
        macro   (get-in m [:clean/shape :macro])]
    (doseq [mt (distinct (concat methods seqm))]
      (when-not (valid-methods mt)
        (swap! errs conj (str "off-vocab/un-typed :method " mt))))
    (when-not (valid-macros macro)
      (swap! errs conj (str "off-vocab/un-typed :macro " macro)))
    @errs))

(defn clean-files [path]
  (let [f (io/file path)]
    (cond
      (.isDirectory f) (->> (file-seq f)
                            (filter #(str/ends-with? (.getName %) ".clean.edn"))
                            (sort-by #(.getName %)))
      (.isFile f) [f]
      :else [])))

(let [paths (if (seq *command-line-args*) *command-line-args* ["holes/clean"])
      files (mapcat clean-files paths)
      results (for [f files]
                [f (try (check (edn/read-string (slurp f)))
                        (catch Exception e [(str "unreadable: " (.getMessage e))]))])
      fails (filter (fn [[_ e]] (seq e)) results)]
  (doseq [[f errs] results]
    (if (seq errs)
      (do (println "FAIL" (.getName f)) (doseq [e errs] (println "     " e)))
      (println "PASS" (.getName f))))
  (println (format "\n%d/%d vocab-conformant  (%d methods, %d macros in vocab)"
                   (- (count results) (count fails)) (count results)
                   (count valid-methods) (count valid-macros)))
  (System/exit (if (seq fails) 1 0)))
