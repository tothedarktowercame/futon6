#!/usr/bin/env bb
;; Strict well-formedness checker for CLean EDN files (the CT treatment of a
;; proof). The proof-side analogue of iatc_argcheck.bb.
;;
;; Usage:
;;   bb scripts/clean_argcheck.bb holes/clean/a93J05.clean.edn
;;   bb scripts/clean_argcheck.bb holes/clean/        ;; all *.clean.edn in dir
;;
;; The gates (a CLean is well-formed iff ALL pass):
;;   G1 parse        — valid EDN, required keys present
;;   G2 boxes        — unique :id, every box has :method, :text
;;   G3 copar        — :clean/seq equals the in-order vector of box :method
;;                     (the M-typed-holes copar coherence: informal ∥ formal cohere)
;;   G4 wires        — every :from/:to is an existing box id; :carries present
;;   G5 ports        — wire :carries is the :produces of :from and is in the
;;                     :consumes of :to (the comb wiring actually type-checks)
;;   G6 holes        — :satiety ∈ DarkTower SatietyGrade; :discharge ∈ DischargeKind
;;   G7 dag          — the wire graph is acyclic
;;   G8 shape        — :holes-at / :discharges-at agree with the boxes
;;
;; Exits nonzero if any file fails any gate.

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def satiety-grades   #{:parse :payoff :canon :bundling :role})
(def discharge-kinds  #{:sorryProof :queryAnswer :ungroundedBinder})
(def required-keys    [:clean/proof :clean/seq :clean/boxes :clean/wires
                       :clean/copar :clean/shape])

(defn fail [errs gate msg] (conj errs (str gate ": " msg)))

(defn check-clean [m]
  (let [errs (atom [])
        boxes (:clean/boxes m)
        ids   (mapv :id boxes)
        idset (set ids)]
    ;; G1 required keys
    (doseq [k required-keys]
      (when-not (contains? m k)
        (swap! errs fail "G1" (str "missing key " k))))
    ;; G2 boxes
    (when (not= (count ids) (count idset))
      (swap! errs fail "G2" "duplicate box :id"))
    (doseq [b boxes]
      (when-not (:method b) (swap! errs fail "G2" (str "box " (:id b) " has no :method")))
      (when-not (:text b)   (swap! errs fail "G2" (str "box " (:id b) " has no :text"))))
    ;; G3 copar coherence
    (let [seq-methods (:clean/seq m)
          box-methods (mapv :method boxes)]
      (when (not= seq-methods box-methods)
        (swap! errs fail "G3"
               (str "copar incoherent: :clean/seq " seq-methods
                    " ≠ box methods " box-methods))))
    ;; G4 + G5 wires & ports
    (doseq [{:keys [from to carries]} (:clean/wires m)]
      (when-not (idset from) (swap! errs fail "G4" (str "wire :from " from " is not a box id")))
      (when-not (idset to)   (swap! errs fail "G4" (str "wire :to " to " is not a box id")))
      (when (nil? carries)   (swap! errs fail "G4" (str "wire " from "→" to " has no :carries")))
      (let [bf (first (filter #(= from (:id %)) boxes))
            bt (first (filter #(= to (:id %)) boxes))
            produced (let [p (:produces bf)] (if (coll? p) (set p) #{p}))
            consumed (set (:consumes bt))]
        (when (and bf carries (not (produced carries)))
          (swap! errs fail "G5" (str "wire " from "→" to " carries " carries
                                     " but " from " does not :produce it")))
        (when (and bt carries (not (consumed carries)))
          (swap! errs fail "G5" (str "wire " from "→" to " carries " carries
                                     " but " to " does not :consume it")))))
    ;; G6 holes
    (doseq [b boxes]
      (when-let [h (:hole b)]
        (when-not (satiety-grades (:satiety h))
          (swap! errs fail "G6" (str "box " (:id b) " hole :satiety " (:satiety h) " invalid")))
        (when-not (discharge-kinds (:discharge h))
          (swap! errs fail "G6" (str "box " (:id b) " hole :discharge " (:discharge h) " invalid")))
        (when-not (:kind h)
          (swap! errs fail "G6" (str "box " (:id b) " hole missing :kind")))))
    ;; G7 acyclicity (Kahn)
    (let [edges (mapv (juxt :from :to) (:clean/wires m))
          succ  (reduce (fn [acc [a b]] (update acc a (fnil conj #{}) b)) {} edges)
          indeg (reduce (fn [acc [_ b]] (update acc b (fnil inc 0) )) (zipmap ids (repeat 0)) edges)]
      (loop [indeg indeg, q (vec (filter #(zero? (indeg %)) ids)), seen 0]
        (if (empty? q)
          (when (not= seen (count ids))
            (swap! errs fail "G7" "wire graph has a cycle"))
          (let [n (peek q)
                q (pop q)
                [indeg q] (reduce (fn [[ind qq] m']
                                    (let [d (dec (ind m'))]
                                      [(assoc ind m' d) (if (zero? d) (conj qq m') qq)]))
                                  [indeg q] (succ n))]
            (recur indeg q (inc seen))))))
    ;; G8 shape cross-check
    (let [shape (:clean/shape m)
          actual-holes     (set (keep #(when (:hole %) (:id %)) boxes))
          actual-discharge (set (keep #(when (:discharges %) (:id %)) boxes))
          declared-holes   (set (:holes-at shape))
          declared-disch   (set (:discharges-at shape))]
      (when (not= actual-holes declared-holes)
        (swap! errs fail "G8" (str ":holes-at " declared-holes " ≠ boxes with holes " actual-holes)))
      (when (not= actual-discharge declared-disch)
        (swap! errs fail "G8" (str ":discharges-at " declared-disch " ≠ boxes with discharges " actual-discharge)))
      (when-not (:macro shape)
        (swap! errs fail "G8" "shape missing :macro")))
    @errs))

(defn clean-files [path]
  (let [f (io/file path)]
    (cond
      (.isDirectory f) (->> (file-seq f)
                            (filter #(str/ends-with? (.getName %) ".clean.edn"))
                            (sort-by #(.getName %)))
      (.isFile f) [f]
      :else [])))

(let [args *command-line-args*
      paths (if (seq args) args ["holes/clean"])
      files (mapcat clean-files paths)
      results (for [f files]
                (let [errs (try (check-clean (edn/read-string (slurp f)))
                                (catch Exception e [(str "G1: unreadable EDN — " (.getMessage e))]))]
                  [f errs]))
      failures (filter (fn [[_ e]] (seq e)) results)]
  (doseq [[f errs] results]
    (if (seq errs)
      (do (println "FAIL" (.getName f))
          (doseq [e errs] (println "     " e)))
      (println "PASS" (.getName f))))
  (println (format "\n%d/%d well-formed" (- (count results) (count failures)) (count results)))
  (System/exit (if (seq failures) 1 0)))
