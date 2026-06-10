#!/usr/bin/env bb
;; closure_recorder.bb — the real track record of closures (E-ground-G).
;;
;; "Not a made-up history, but a real track record of closures." (Joe, 2026-06-10.)
;; THE ANTI-FABRICATION RULE: every closure entry MUST carry real provenance
;; (:evidence — a commit-sha, a :send-witness, a discharge-event ref). An entry
;; without evidence is REFUSED. A closure is a hole that was open and got closed,
;; recorded WHEN it happened with WHAT closed it. This is the dense per-move
;; realized signal E-ground-G grounds G in (closure = anamnesis-discharge).
;;
;; This ledger is the broad-grain sibling of the car's CH2 discharge sink
;; (futon3a/data/ch2-discharge-events.edn): the discharge-emission feeds closures
;; at the meme.step promote! grain; this recorder also captures closures at the
;; work grain (a sorry discharged, a mission phase anchored) — each with evidence.

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def ledger-path
  ;; tracked under holes/ (NOT data/, which is gitignored) — the track record is
  ;; durable evidence and must be versioned.
  (str (io/file (.getParent (io/file *file*)) ".." "holes" "closure-ledger.edn")))

(def required-keys #{:closed/scope :kind :at :by :evidence})
;; :closed/scope — the hole id (scope-id / sorry-id / want-endpoint), maps to a move target
;; :move/id      — optional, the (have->want) move if this closure is move-attributable
;; :kind         — :phase | :sorry | :arrow | :mission-phase
;; :at           — iso-ts (pass in; no Date.now in bb here, caller stamps)
;; :by           — who/what closed it (agent / move / commit)
;; :evidence     — REAL provenance: a commit-sha, a :send-witness, a discharge-event ref

(defn read-closures
  ([] (read-closures ledger-path))
  ([path]
   (let [f (io/file path)]
     (if (.exists f)
       (->> (str/split-lines (slurp f)) (remove str/blank?) (mapv edn/read-string))
       []))))

(defn valid-closure?
  "A closure is recordable iff it carries every required key AND its :evidence is
   non-blank (the anti-fabrication rule). Returns [ok? reason]."
  [m]
  (cond
    (not (every? #(contains? m %) required-keys))
    [false (str "missing keys: " (vec (remove #(contains? m %) required-keys)))]
    (or (nil? (:evidence m)) (and (string? (:evidence m)) (str/blank? (:evidence m))))
    [false "no :evidence — refused (a real track record needs real provenance)"]
    :else [true :ok]))

(defn append-closure!
  "Append one real closure to the ledger, or refuse a fabricated/incomplete one."
  ([m] (append-closure! ledger-path m))
  ([path m]
   (let [[ok reason] (valid-closure? m)]
     (if ok
       (do (io/make-parents path)
           (spit path (str (pr-str m) "\n") :append true)
           {:recorded m})
       {:refused reason :entry m}))))

;; CLI: bb closure_recorder.bb count        -> how many closures recorded
;;      bb closure_recorder.bb show         -> print the ledger
(let [mode (or (first *command-line-args*) "count")]
  (case mode
    "count" (println (count (read-closures)) "closures recorded in" ledger-path)
    "show"  (doseq [c (read-closures)] (prn c))
    (println "usage: bb closure_recorder.bb [count|show]")))
