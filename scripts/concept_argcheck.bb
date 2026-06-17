#!/usr/bin/env bb
;; Strict checker for concept-encyclopedia EDN entries.
;;
;; Usage:
;;   bb scripts/concept_argcheck.bb data/concept-encyclopedia/ct/
;;   bb scripts/concept_argcheck.bb --self-check
;;
;; Mirrors scripts/iatc_argcheck.bb: print PASS/FAIL per file and exit nonzero
;; if any file fails a gate.

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.set :as set]
         '[clojure.string :as str])

(def repo-root
  (.getParentFile (.getParentFile (io/file *file*))))

(def ct-dir (io/file repo-root "data/concept-encyclopedia/ct"))
(def golden-dir (io/file repo-root "data/concept-encyclopedia/ct-golden"))
(def fixture-dir (io/file repo-root "holes/concept-argcheck/fixtures"))

(def allowed-kinds #{:object :morphism :property :construction :operation :theorem})

;; These are the unresolved primitive imports already used by the six curated
;; seeds. They are the checker's explicit core vocabulary, not generated entries.
(def core-targets
  #{:category :morphism :object :composition :structure-preserving-map
    :kernel :cokernel :monomorphism :epimorphism
    :universal-property :hom-set :morphism-of-functors
    :hom-functor :representable-functor :universal-element
    :representability-theorem :locally-small-category :set
    :monoid-in-endofunctors :endofunctor})

(def ref-keys #{:depends-on :refs :uses :genus :exports})

(defn usage! []
  (binding [*out* *err*]
    (println "Usage: bb scripts/concept_argcheck.bb [--self-check] <file-or-directory> [...]"))
  (System/exit 2))

(defn edn-files [path]
  (let [f (io/file path)]
    (cond
      (not (.exists f)) []
      (.isDirectory f) (->> (file-seq f)
                            (filter #(.isFile %))
                            (filter #(str/ends-with? (.getName %) ".edn"))
                            (sort-by #(.getPath %)))
      :else [f])))

(defn read-one-edn [file]
  (with-open [r (java.io.PushbackReader. (io/reader file))]
    (let [form (edn/read {:eof ::eof} r)]
      (if (= ::eof form)
        (throw (ex-info "empty EDN file" {}))
        (let [tail (edn/read {:eof ::eof} r)]
          (when-not (= ::eof tail)
            (throw (ex-info "trailing EDN forms after entry" {:tail tail})))
          form)))))

(defn concept-files []
  (concat (edn-files ct-dir) (edn-files golden-dir)))

(defn concept-index []
  (->> (concept-files)
       (map read-one-edn)
       (map :concept/id)
       (remove nil?)
       set
       (set/union core-targets)))

(defn locus [file entry x]
  (str (.getPath file)
       (when-let [cid (:concept/id entry)] (str " concept=" cid))
       (when-let [id (:id x)] (str " id=" id))))

(defn fail-entry [gate file entry x reason]
  {:gate gate
   :file (.getPath file)
   :locus (locus file entry x)
   :reason reason})

(defn blank? [x]
  (str/blank? (str (or x ""))))

(defn seqify [x]
  (cond
    (nil? x) []
    (sequential? x) x
    :else [x]))

(defn collect-targets
  ([entry] (collect-targets [] entry))
  ([path x]
   (cond
     (map? x)
     (mapcat (fn [[k v]]
               (if (contains? ref-keys k)
                 (for [target (seqify v)
                       :when (keyword? target)]
                   {:path (conj path k)
                    :key k
                    :target target})
                 (collect-targets (conj path k) v)))
             x)

     (sequential? x)
     (mapcat (fn [[idx v]] (collect-targets (conj path idx) v))
             (map-indexed vector x))

     :else [])))

(defn hole-mentions-target? [holes target]
  (let [needle (name target)]
    (some (fn [h]
            (and (map? h)
                 (some (fn [v]
                         (cond
                           (= v target) true
                           (string? v) (str/includes? v needle)
                           (sequential? v) (some #(or (= % target)
                                                      (and (string? %)
                                                           (str/includes? % needle)))
                                                  v)
                           :else false))
                       (vals h))))
          holes)))

(defn typed-hole? [h]
  (and (map? h)
       (keyword? (:kind h))
       (contains? h :wanted)
       (not (blank? (:wanted h)))))

(defn typed-var? [v]
  (and (map? v)
       (keyword? (:var v))
       (keyword? (:type v))))

(defn typed-axiom? [a]
  (and (map? a)
       (keyword? (:id a))
       (string? (:statement a))
       (not (blank? (:statement a)))))

(defn typed-conclusion? [c]
  (and (map? c)
       (or (and (string? (:relation c)) (not (blank? (:relation c))))
           (and (string? (:statement c)) (not (blank? (:statement c)))))))

(defn check-entry [file entry idx]
  (let [holes (vec (:holes entry))
        failures (atom [])]
    (when-not (contains? allowed-kinds (:kind entry))
      (swap! failures conj
             (fail-entry :kind file entry entry
                         (str "unknown :kind " (pr-str (:kind entry))
                              "; expected one of " (pr-str allowed-kinds)))))

    (doseq [g (:given entry)]
      (when-not (typed-var? g)
        (swap! failures conj
               (fail-entry :typed-var file entry g
                           ":given entries must be maps with keyword :var and keyword :type"))))

    (doseq [a (:axioms entry)]
      (when-not (typed-axiom? a)
        (swap! failures conj
               (fail-entry :typed-statement file entry a
                           ":axioms entries must be maps with keyword :id and nonblank string :statement"))))

    (when (and (:conclusion entry)
               (not (typed-conclusion? (:conclusion entry))))
      (swap! failures conj
             (fail-entry :typed-statement file entry (:conclusion entry)
                         ":conclusion must be a map with nonblank string :relation or :statement")))

    (when (and (= :theorem (:kind entry))
               (not (typed-conclusion? (:conclusion entry))))
      (swap! failures conj
             (fail-entry :typed-statement file entry entry
                         ":kind :theorem requires a well-formed :conclusion")))

    (if (= :golden (:status entry))
      (when (seq holes)
        (swap! failures conj
               (fail-entry :golden-holes file entry entry
                           ":status :golden entries must have :holes []")))
      (doseq [h holes]
        (when-not (typed-hole? h)
          (swap! failures conj
                 (fail-entry :hole file entry h
                             "non-golden holes must be typed maps with :kind and :wanted")))))

    (doseq [{:keys [target key]} (collect-targets entry)]
      (when-not (or (contains? idx target)
                    (hole-mentions-target? holes target))
        (swap! failures conj
               (fail-entry :unresolved-target file entry entry
                           (str key " target " target
                                " does not resolve to ct/ or ct-golden/ and is not named in :holes")))))
    @failures))

(defn check-file [idx file]
  (try
    (let [entry (read-one-edn file)
          failures (check-entry file entry idx)]
      {:file (.getPath file)
       :ok? (empty? failures)
       :failures failures})
    (catch Exception e
      {:file (.getPath file)
       :ok? false
       :failures [{:gate :edn-parse
                   :file (.getPath file)
                   :locus (.getPath file)
                   :reason (.getMessage e)}]})))

(defn print-result [{:keys [file ok? failures]}]
  (println (str (if ok? "PASS " "FAIL ") file))
  (doseq [{:keys [gate locus reason]} failures]
    (println (str "  [" (name gate) "] " locus " :: " reason))))

(defn expected-gate [file]
  (try
    (:expect/gate (read-one-edn file))
    (catch Exception _ nil)))

(defn run-files! [paths]
  (let [idx (concept-index)
        files (mapcat edn-files paths)]
    (when (empty? files)
      (binding [*out* *err*]
        (println "No .edn files found in input paths:" (str/join " " paths)))
      (System/exit 2))
    (let [results (mapv #(check-file idx %) files)]
      (doseq [r results] (print-result r))
      (when (some (complement :ok?) results)
        (System/exit 1)))))

(defn run-self-check! []
  (let [idx (concept-index)
        golden-files (edn-files (io/file fixture-dir "golden"))
        negative-files (edn-files (io/file fixture-dir "negative"))
        golden-results (mapv #(check-file idx %) golden-files)
        negative-results (mapv #(check-file idx %) negative-files)
        negative-ok? (fn [file result]
                       (let [want (expected-gate file)
                             got (set (map :gate (:failures result)))]
                         (and (not (:ok? result))
                              (contains? got want))))
        bad-negatives (keep (fn [[file result]]
                              (when-not (negative-ok? file result)
                                (let [reason (str "expected failure gate "
                                                  (pr-str (expected-gate file))
                                                  ", got "
                                                  (pr-str (map :gate (:failures result))))]
                                  (assoc result :ok? false
                                         :failures [{:gate :self-check
                                                     :file (.getPath file)
                                                     :locus (.getPath file)
                                                     :reason reason}]))))
                            (map vector negative-files negative-results))
        failures (concat (remove :ok? golden-results) bad-negatives)]
    (doseq [r (concat golden-results negative-results)] (print-result r))
    (if (empty? failures)
      (do (println "SELF-CHECK PASS") (System/exit 0))
      (do (println "SELF-CHECK FAIL")
          (doseq [r failures] (print-result r))
          (System/exit 1)))))

(let [args *command-line-args*]
  (cond
    (empty? args) (usage!)
    (= ["--self-check"] args) (run-self-check!)
    :else (run-files! args)))
