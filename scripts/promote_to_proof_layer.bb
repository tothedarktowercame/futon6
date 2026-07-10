#!/usr/bin/env bb
;; PROMOTE-TO-PROOF-LAYER (DRY RUN) — D4 promotion-readiness prototype for M-populate-substrate-2.
;;
;; D4's deliverable narrows to "the promotion path overlay->store for what M-operational-vocabulary
;; (forward methods) + M-goals-and-holes (backward C-vector) produce" — NOT inventing relation content.
;; This maps our sim-only overlays into CANDIDATE substrate-2 PROOF-layer edges and runs each through the
;; E-substrate-2-sorry-typing **T-A4 gate** (promote only when ALL: (a) a CANONICAL source field/artifact —
;; not a heuristic free-text parse; (b) the target is a normalized typed endpoint; (c) a NAMED consumer needs
;; it AS a relation). Borrowed-prior moves stay `:mined-structural` — never laundered as proofs.
;;
;; ZERO :7071 writes. Emits candidate-proof-edges.edn + a per-feeder T-A4 report. Read-only on :7071.
;;   bb scripts/promote_to_proof_layer.bb
(require '[clojure.edn :as edn] '[clojure.pprint :refer [pprint]]
         '[babashka.http-client :as http] '[cheshire.core :as json])

(def ROOT "/home/joe/code/futon6")
(def OUT (str ROOT "/data/c-vector/candidate-proof-edges.edn"))
(def MOVES (str ROOT "/data/diffsub-moves-mined.edn"))
(def MEMES (str ROOT "/data/meme-mine/resolved-memes.openai.json"))
(def SUBSTRATE "http://localhost:7071")

;; T-A4 — promote iff all three. Each feeder declares its (a)/(b)/(c) honestly.
(defn ta4 [{:keys [canonical-source? normalized-target? named-consumer?]}]
  (boolean (and canonical-source? normalized-target? named-consumer?)))
(defn norm-target? [t]
  (boolean (and t (re-find #"^(mission/|scope/|pattern/|hole/|artifact/|[a-z0-9]+/[A-Za-z0-9._-]+|M-[a-z0-9-]+)$" (str t)))))

(def CONSTRUCTS-OPS #{"build" "create" "add" "implement" "write" "wire" "extend" "port" "mine" "reconstruct" "commission"})
(def CLOSES-OPS     #{"fix" "close" "refine" "update" "commit" "verify" "review" "execute"})

;; ---- Feeder (b1): mission-miner overlay :close-hole moves -> :closes/:constructs (BLOCKED, :mined-structural)
(defn from-moves []
  (for [m (:moves (edn/read-string (slurp MOVES)))
        :when (= :close-hole (:move/class m))]
    (let [ta {:canonical-source? false :normalized-target? (norm-target? (:want m)) :named-consumer? true}]  ; (a) FALSE: borrowed structural prior, not a canonical source
      {:type :closes :source (:have m) :target (:want m) :consumer :rollout-act-gate :feeder :miner-overlay
       :provenance {:from :diffsub-moves-mined :move-id (:move/id m) :confidence (:confidence m) :prior (:prior m)}
       :tag :mined-structural :ta4 ta :promotable (ta4 ta)})))

;; ---- Feeder (b2): GPU named-tier memes -> :constructs/:closes (canonical = named ref + verbatim evidence)
(defn from-memes []
  (when (.exists (java.io.File. MEMES))
    (for [r (json/parse-string (slurp MEMES) true)
          :let [m (:meme r) h (:have m) w (:want m) op (str (:op m))
                src (when (= "named" (:tier h)) (:ref h)) tgt (when (= "named" (:tier w)) (:ref w))]
          ;; a method->goal edge needs DISTINCT named source + target; have-only memes (want=null) are
          ;; not constructs/closes edges (smoke yields none — real edges await the full rerun's have->want)
          :when (and src tgt (not= src tgt) (or (CONSTRUCTS-OPS op) (CLOSES-OPS op)))]
      (let [ta {:canonical-source? true :normalized-target? (norm-target? tgt) :named-consumer? true}]  ; (a) named ref + verbatim evidence
        {:type (if (CLOSES-OPS op) :closes :constructs) :source src :target tgt
         :consumer :proof-propagation :feeder :gpu-meme
         :provenance {:from :resolved-memes :op op :tier-have (:tier h) :tier-want (:tier w)
                      :evidence (or (:evidence h) (:evidence w)) :ask-id (:id r)}
         :tag :gpu-mined :ta4 ta :promotable (ta4 ta)}))))

;; ---- Feeder (a): sorries with a canonical :depends-on-raw field -> :depends-on-sorry
(defn from-sorries []
  (let [ss (-> (http/get (str SUBSTRATE "/api/alpha/entities/latest?type=sorry&limit=2000")) :body edn/read-string :entities)]
    (for [s ss
          :let [p (:props s) sid (:sorry/id p) deps (:sorry/depends-on-raw p)]
          dep (when (seq (str deps)) deps)]
      (let [ta {:canonical-source? true :normalized-target? (norm-target? dep) :named-consumer? true}]  ; (a) canonical FIELD, not a free-text parse
        {:type :depends-on-sorry :source sid :target dep :consumer :close-propagation :feeder :sorry-typing
         :provenance {:from :sorry/depends-on-raw :sorry-status (:sorry/status p)}
         :tag :sourced :ta4 ta :promotable (ta4 ta)}))))

(defn report [label edges]
  (let [prom (filter :promotable edges) blk (remove :promotable edges)]
    (println (format "  %-16s candidates %3d · T-A4 PASS %3d · BLOCKED %3d" label (count edges) (count prom) (count blk)))
    edges))

(defn -main [& _]
  (println "=== D4 promotion-readiness (DRY RUN — zero :7071 writes) ===")
  (let [moves (report "miner-overlay" (from-moves))
        memes (report "gpu-meme" (vec (from-memes)))
        sorries (report "sorry-typing" (from-sorries))
        all (concat moves memes sorries)
        prom (filter :promotable all)]
    (.mkdirs (java.io.File. (str ROOT "/data/c-vector")))
    (spit OUT (with-out-str (pprint {:source "promote_to_proof_layer.bb (DRY RUN — candidates only, sim-only)"
                                     :n (count all) :promotable (count prom)
                                     :by-type (frequencies (map :type all))
                                     :promotable-by-type (frequencies (map :type prom))
                                     :edges (vec all)})))
    (println (format "\nTOTAL: %d candidate edges · %d pass T-A4 (promotable) · %d held as :mined-structural/blocked"
                     (count all) (count prom) (- (count all) (count prom))))
    (println "promotable by type:" (into (sorted-map) (frequencies (map :type prom))))
    (println "blocked by feeder:" (into (sorted-map) (frequencies (map :feeder (remove :promotable all)))))
    (println (str "wrote " OUT " (candidates only — promotion to :7071 is gated, claude-2's D4 store-side)"))))

(apply -main *command-line-args*)
