#!/usr/bin/env bb
;; C-VECTOR — the belly (Friston's C, R19 of futon-aif-completeness) as data.
;; INSTANTIATE for M-goals-and-holes.md: builds the all-CPU static loop
;;   C-ENTRY (the shared record)  →  C-MESS (a producer)  →  C-RISK (static risk = distance).
;;
;; A C-entry is ONE preferred outcome the agent wants satisfied (DERIVE §3). The C-vector =
;; the open C-entries. Risk = Σ weight·divergence(current-outcome, preferred) — the R5 term the
;; C-less EFE drops (Da Costa et al.: no C ⇒ G collapses to pure info-gain). Static risk uses the
;; CURRENT observable outcome (computable now); PREDICTIVE risk (outcome under a policy) is W1-gated
;; (the goal↔method join — see goals-holes-readiness.html W1-JOIN).
;;
;; The mess channel (Joe: "a messy mission IS implicitly a hole"): a low-coherence mission's
;; preferred outcome is "coherence rises to the alive-class norm". Salingaros liveness L=T·H is the
;; observable; the alive-class median L is the preferred value. NB the EDN's :C field is the LEGACY
;; Salingaros mess number — we read :L/:class only, never :C, to keep it apart from Friston's C.
;; Discharge already exists for this channel (:centre-mess), which is why it is the cleanest one.
;;
;;   bb scripts/c_vector.bb            # produce mess C-entries + print the static-risk validation
;;   bb scripts/c_vector.bb --quiet    # write only
(require '[clojure.edn :as edn]
         '[clojure.string :as str]
         '[clojure.pprint :refer [pprint]]
         '[babashka.http-client :as http]
         '[babashka.fs :as fs]
         '[clojure.set :as set]
         '[cheshire.core :as json]
         '[babashka.classpath :as cp])

;; R19-UNIFY: the stated-channel entry logic has ONE source of truth —
;; `futon2.aif.c-vector` (the live belly producer; babashka-compatible). This
;; script (the snapshot producer) delegates to it so the two cannot drift.
(cp/add-classpath "/home/joe/code/futon2/src")
(require '[futon2.aif.c-vector :as cv])

(def ROOT "/home/joe/code/futon6")
(def WHOLENESS (str ROOT "/data/mission-wholeness.edn"))
(def SCOPE-TREES (str ROOT "/data/mission-scope-trees"))
(def OUT-DIR (str ROOT "/data/c-vector"))
(def LATE-PHASES #{"verify" "instantiate" "document"})    ; a mission that reached none = not yet run
(def SUBSTRATE "http://localhost:7071")
;; 33 open sorries share this templated :if — the boilerplate the audit found (143 open → 110 clean).
(def BOILERPLATE-IF "Work requires a structured plan")

;; ---- C-ENTRY : the shared record (every producer emits this shape) --------------------
(defn c-entry
  "One preferred outcome. Keys mirror M-goals-and-holes DERIVE §3. `:status` flips open→met only
   with a `:witness` (I3 earned-closure). `:weight` carries its `:basis` so orientation is auditable
   (I2): un-oriented entries take a low default — never silently dropped (I4), never the local text."
  [{:keys [flavour outcome-ref preferred weight status provenance discharged-by witness]
    :or   {status :open weight {:value 0.3 :basis :default-unoriented} witness nil}}]
  {:pre [flavour outcome-ref preferred provenance]}             ; I1: no preference without provenance
  {:flavour flavour :outcome-ref outcome-ref :preferred preferred
   :weight weight :status status :provenance provenance
   :discharged-by discharged-by :witness witness})

(defn- divergence
  "Static one-sided distance of the current outcome from the preferred floor (we only want it raised;
   being already above the floor is no risk). Returns 0.0 when satisfied. `:becomes` = a binary goal
   (a cap attested / a sorry closed): full unit risk while unmet, 0 when it reaches the preferred state."
  [current {:keys [op value]}]
  (case op
    :>=     (max 0.0 (- value current))
    :<=     (max 0.0 (- current value))
    :becomes (if (= current value) 0.0 1.0)
    (Math/abs (double (- current value)))))

(defn risk-of
  "C-RISK (static): weight·divergence(current, preferred). The R5 contribution of one open C-entry."
  [{:keys [preferred weight status]} current]
  (if (= status :open)
    (* (double (:value weight)) (divergence current preferred))
    0.0))

;; ---- C-MESS : the producer (mess channel) ---------------------------------------------
(defn- median [xs]
  (let [s (vec (sort xs))] (nth s (quot (count s) 2))))

(defn produce-mess
  "Read mission-wholeness.edn → (i) one per-mission C-entry for each mess-class mission (coherence
   rises to the alive-class median) + (ii) one STANDING global-coherence regularizer (DERIVE §3 ii):
   not a hole to close but a preference that EVERY policy keep stack coherence high — the 'don't make
   a bigger mess while fixing one thing' safety property (ARGUE; the pattern-harness telos)."
  [wholeness]
  (let [ms      (:missions wholeness)
        alive   (filter #(= :alive (:class %)) ms)
        med     (double (median (map :L alive)))
        mess    (filter #(= :mess (:class %)) ms)
        per     (for [m mess]
                  (c-entry {:flavour :mess
                            :outcome-ref {:kind :coherence :mission (:mission m) :metric :L}
                            :preferred   {:op :>= :value med :basis :alive-class-median}
                            :provenance  {:source "mission-wholeness.edn" :class :mess
                                          :L (double (:L m)) :H (:H m) :derived-from (:mission m)}
                            :discharged-by :centre-mess}))      ; the one channel whose method exists
        standing (c-entry {:flavour :mess
                           :outcome-ref {:kind :coherence :scope :stack :metric :median-L}
                           :preferred   {:op :>= :value med :basis :stays-high}
                           :weight      {:value 1.0 :basis :standing-regularizer}
                           :provenance  {:source "mission-wholeness.edn" :stat :alive-median-L :value med}
                           :discharged-by :preserve-coherence})]
    {:median-L med :per-mission (vec per) :standing standing}))

;; ---- C-STATED : the producer (stated channel — caps · clean open sorries) --------------
(defn- fetch
  "Read entities from substrate-2 (:7071, EDN). Read-only — sim-only, zero writes."
  [type]
  (-> (http/get (str SUBSTRATE "/api/alpha/entities/latest?type=" type "&limit=2000"))
      :body edn/read-string :entities))

(defn produce-stated
  "The stated channel — goals that are already structured (the backward asymmetry: reads, not mining).
   Capabilities not yet attested + clean open sorries (the 33 boilerplate dropped). Each → a stated
   C-entry whose preferred outcome is its satisfaction (cap attested / sorry closed). Weight: caps are
   star-map-oriented (status drives it, I2); un-oriented sorries take the default-low."
  []
  (let [caps    (fetch "capability")
        sorries (fetch "sorry")
        ;; ENTRIES come from the single source of truth (R19-UNIFY).
        entries  (cv/entries-from-corpus caps sorries)
        cap-es   (filter #(= :capability (-> % :outcome-ref :kind)) entries)
        sorry-es (filter #(= :sorry (-> % :outcome-ref :kind)) entries)
        ;; stats (presentation only) — mirror the producer's own filters.
        meta?   #{"scope/capability/capabilities" "scope/capability/capability"}
        unmet   (->> caps (remove #(get-in % [:props :capability/attested?])) (remove #(meta? (:id %))))
        open    (filter #(= "open" (get-in % [:props :sorry/status])) sorries)
        clean   (remove #(str/starts-with? (str (get-in % [:props :sorry/if])) BOILERPLATE-IF) open)]
    {:caps {:total (count caps) :unmet (count unmet) :entries (vec cap-es)}
     :sorries {:open (count open) :clean (count clean) :boilerplate (- (count open) (count clean))
               :entries (vec sorry-es)}}))

;; ---- C-INCOMPLETE : the producer (incompleteness channel — started-but-not-run) -------
(defn- phases-of [scope-tree]
  (set (for [he (get scope-tree "scope-hyperedges") e (get he "ends") :let [p (get e "phase")] :when p] p)))

(defn produce-incomplete
  "The incompleteness channel — a mission begun but never run is a hole stated by its own silence
   (Joe's not-talking tell). v1 signal = the phase-spine: has identify/map ∧ reached none of
   verify/instantiate/document. I4: surfaced (never silently dropped — that would re-hide the hole);
   un-oriented → default-low weight until the star-map∪devmap orientation join lands."
  []
  (let [trees (->> (fs/glob SCOPE-TREES "M-*.json") (map (comp #(json/parse-string (slurp %)) str)))
        started-not-run (for [t trees
                              :let [ph (phases-of t)]
                              :when (and (or (ph "identify") (ph "map")) (empty? (set/intersection ph LATE-PHASES)))]
                          (c-entry {:flavour :incompleteness
                                    :outcome-ref {:kind :mission-completion :mission (get t "mission") :metric :run}
                                    :preferred   {:op :becomes :value :run}
                                    :provenance  {:source "mission-scope-trees" :mission (get t "mission")
                                                  :phases-present (vec (sort ph)) :oriented? false}}))]
    {:total (count trees) :entries (vec started-not-run)}))

;; ---- 應-voice : the GPU channel (reach + correction), SALVAGED from the pre-fix run ----
(def SALVAGED (str OUT-DIR "/c-entries.salvaged.json"))
(defn produce-yingvoice
  "Fold the salvaged 應-voice C-entries (gate-passing reach + genuine correction, from salvage_c_entries.py)
   into the belly as its first GPU-mined content. PROVISIONAL — the clean rerun supersedes this. Each mined
   record → the shared c-entry shape (binary :becomes outcome so it slots into risk-of). [] if no salvage."
  []
  (if-not (.exists (java.io.File. SALVAGED))
    []
    (vec (for [r (json/parse-string (slurp SALVAGED) true)
               :let [pv (:provenance r) fl (keyword (:flavour r))
                     g (:grounded_ref pv) a (:assistant_span pv) rep (:reply_span pv)]]
           (c-entry {:flavour fl
                     :outcome-ref (if (= fl :correction)
                                    {:kind :preference :referent g :target (get-in r [:preferred :value]) :metric :aligned}
                                    {:kind :goal :referent g :metric :satisfied})
                     :preferred {:op :becomes :value (if (= fl :correction) :adopted :reached)}
                     :provenance {:source ":ying-voice-salvage" :provisional true :channel :ying-voice
                                  :derived-from (:id r) :grounded-ref g :assistant-span a :reply-span rep}
                     :discharged-by (when (= fl :correction) :adopt-redirect)})))))

;; ---- C-STORE : substrate-2 overlay (sim-only — zero :7071 writes) ----------------------
(defn- entity-id
  "A stable, unique, promotable substrate id. For stated entries key off the source's UNIQUE substrate
   id (the prop-level :capability/id can be nil → collisions); for mess/incompleteness key off the
   mission. Idempotent across runs; promotable to :7071 as-is."
  [{:keys [flavour outcome-ref provenance]}]
  (if-let [sid (:substrate-id provenance)]
    (str "scope/c-entry/" (str/replace (str sid) #"^scope/" ""))
    (let [{:keys [kind mission scope]} outcome-ref]
      (str "scope/c-entry/"
           (case kind
             :coherence          (str "mess/" (or mission (when scope (name scope)) "STANDING"))
             :mission-completion (str "incomplete/" mission)
             ;; 應-voice (goal/preference) → turn id + a content hash (one turn can yield several entries)
             (str (name flavour) "/" (or (:derived-from provenance) "x") "/"
                  (format "%x" (bit-and (hash [(:referent outcome-ref) (:assistant-span provenance)
                                               (:reply-span provenance)]) 0xffffff))))))))

(defn ->substrate-entity
  "Map a C-entry to the substrate-2 entity shape {:id :name :type :props} (cf. the :7071 capability/
   sorry entities). The DERIVE relations are inlined in :props as references: `:c-entry/derived-from`
   (provenance, I1) + `:c-entry/discharged-by` (the PROOF-layer join, the goal↔method bridge)."
  [e]
  (let [p (:provenance e)
        df (or (:substrate-id p)
               (some->> (:derived-from p) (str "mission/"))
               (some->> (:mission p) (str "mission/")))]
    {:id (entity-id e) :name (str (name (:flavour e)) ":" (-> e :outcome-ref :kind name))
     :type :c-entry
     :props {:c-entry/flavour (:flavour e) :c-entry/outcome-ref (:outcome-ref e)
             :c-entry/preferred (:preferred e) :c-entry/weight (:weight e)
             :c-entry/status (:status e) :c-entry/derived-from df
             :c-entry/discharged-by (:discharged-by e) :c-entry/witness (:witness e)
             :scope/role "preference" :scope/source "c-vector"}}))

;; ---- run --------------------------------------------------------------------------------
(defn -main [& args]
  (let [quiet? (some #{"--quiet"} args)
        whole  (edn/read-string (slurp WHOLENESS))
        {:keys [median-L per-mission standing]} (produce-mess whole)
        ;; static risk: read each per-mission entry's CURRENT outcome from its own provenance L
        scored (->> per-mission
                    (map (fn [e] (assoc e :static-risk (risk-of e (-> e :provenance :L)))))
                    (sort-by :static-risk >))
        mess-entries (cons standing scored)
        ;; stated channel — guarded: :7071 may be down (the mess channel must still produce)
        stated (try (produce-stated) (catch Exception e {:error (.getMessage e)}))]
    (.mkdirs (java.io.File. OUT-DIR))
    (spit (str OUT-DIR "/c-entries.mess.edn")
          (with-out-str (pprint {:source "c_vector.bb/produce-mess"
                                 :preferred-coherence median-L
                                 :n (count mess-entries) :entries mess-entries})))
    (when-not quiet?
      (println (format "mess C-entries: %d per-mission + 1 standing regularizer = %d"
                       (count per-mission) (count mess-entries)))
      (println (format "preferred coherence (alive-class median L) = %.1f" median-L))
      (println "top by static risk (the messiest = the biggest holes):")
      (doseq [e (take 4 scored)]
        (println (format "  %-38s L=%.1f  risk=%.1f  → %s"
                         (-> e :outcome-ref :mission) (-> e :provenance :L)
                         (:static-risk e) (name (:discharged-by e)))))
      (println (str "wrote " OUT-DIR "/c-entries.mess.edn")))
    ;; ---- stated channel ----
    (if (:error stated)
      (println "\nstated channel SKIPPED (:7071 unreachable):" (:error stated))
      (let [{:keys [caps sorries]} stated
            all-stated (concat (:entries caps) (:entries sorries))
            ranked (->> all-stated (map #(assoc % :static-risk (risk-of % :open))) (sort-by :static-risk >))]
        (spit (str OUT-DIR "/c-entries.stated.edn")
              (with-out-str (pprint {:source "c_vector.bb/produce-stated"
                                     :n (count all-stated) :entries ranked})))
        (when-not quiet?
          (println (format "\nstated C-entries: %d caps (of %d, %d unmet) + %d clean sorries (%d open − %d boilerplate) = %d"
                           (count (:entries caps)) (:total caps) (:unmet caps)
                           (count (:entries sorries)) (:open sorries) (:boilerplate sorries)
                           (count all-stated)))
          (println "top stated by weight (held caps > frontier > sorries):")
          (doseq [e (take 4 ranked)]
            (println (format "  %-10s w=%.1f  %s"
                             (name (-> e :outcome-ref :kind))
                             (-> e :weight :value)
                             (or (-> e :provenance :id) (-> e :provenance :title)))))
          (println (str "wrote " OUT-DIR "/c-entries.stated.edn")))))
    ;; ---- incompleteness channel ----
    (let [inc (produce-incomplete)]
      (spit (str OUT-DIR "/c-entries.incomplete.edn")
            (with-out-str (pprint {:source "c_vector.bb/produce-incomplete"
                                   :n (count (:entries inc)) :entries (:entries inc)})))
      (when-not quiet?
        (println (format "\nincompleteness C-entries: %d started-but-not-run (of %d missions, default-low weight per I4)"
                         (count (:entries inc)) (:total inc)))
        (println (str "wrote " OUT-DIR "/c-entries.incomplete.edn"))
        ;; ---- the assembled C-vector readout (the belly: open preferences ranked by weight·risk) ----
        (let [mess-open (map #(assoc % :static-risk (risk-of % (-> % :provenance :L))) scored)
              stated-open (when-not (:error stated)
                            (map #(assoc % :static-risk (risk-of % :open))
                                 (concat (-> stated :caps :entries) (-> stated :sorries :entries))))
              inc-open (map #(assoc % :static-risk (risk-of % :open)) (:entries inc))
              yv (map #(assoc % :static-risk (risk-of % :open)) (produce-yingvoice))
              all (concat mess-open stated-open inc-open yv)]
          (when (seq yv)
            (spit (str OUT-DIR "/c-entries.yingvoice.edn")
                  (with-out-str (pprint {:source "c_vector.bb/produce-yingvoice (salvaged, PROVISIONAL — rerun supersedes)"
                                         :n (count yv) :entries (vec yv)})))
            (println (format "\n應-voice C-entries (salvaged, PROVISIONAL): %d folded (reach + genuine correction)" (count yv))))
          (println (format "\n=== C-VECTOR (the belly) : %d open preferences across %d channels ==="
                           (count all) (count (frequencies (map :flavour all)))))
          (println "by flavour:" (into (sorted-map) (frequencies (map :flavour all))))
          (println (format "all derived-from a source (I1): %s · all carry a weight basis (I2): %s"
                           (every? :provenance all) (every? #(-> % :weight :basis) all)))
          ;; ---- C-STORE: substrate-2 overlay (sim-only) ----
          (let [ents (mapv ->substrate-entity all)
                dup (->> ents (map :id) frequencies (filter #(> (val %) 1)) (map key))
                linked (count (filter #(-> % :props :c-entry/derived-from) ents))]
            (spit (str OUT-DIR "/c-store-overlay.edn")
                  (with-out-str (pprint {:note "sim-only overlay — substrate-2 :c-entry shape; promote to :7071 PROOF layer only when greenlit (M-populate-substrate-2)"
                                         :type :c-entry :profile "default"
                                         :n (count ents) :entities ents})))
            (println (format "\nC-STORE: %d C-entries → substrate-2 :c-entry entities (%d with derived-from link, %d id-collisions)"
                             (count ents) linked (count dup)))
            (println "  ZERO :7071 writes — wrote LOCAL overlay" (str OUT-DIR "/c-store-overlay.edn"))))))))

(apply -main *command-line-args*)
