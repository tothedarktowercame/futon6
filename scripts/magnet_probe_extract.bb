#!/usr/bin/env bb
;; magnet_probe_extract.bb (claude-6, E-have-want-pairs Q-B) — flatten the real
;; (have,want) sources into ONE json the python cascade-scorer reads, so EDN parsing
;; (bb, correct) is separate from cascade scoring (futon3a venv). Sim-only reads.
;;
;;   Emits per item: {channel, key, idstem, want_raw, have_raw, title, delta_g}
;;   Tokenization + magnet assembly happen python-side.
(require '[clojure.edn :as edn]
         '[cheshire.core :as json]
         '[clojure.string :as str])

(def CV "/home/joe/code/futon6/data/c-vector")
(def DIFFSUB "/home/joe/code/futon6/data/diffsub-moves.edn")

(defn slurp-edn [p] (edn/read-string (slurp p)))

(defn stated-item [{:keys [outcome-ref provenance]}]
  (let [{:keys [kind id mission title]} outcome-ref
        prov-title (:title provenance)
        prov-status (:status provenance)]
    {:channel (str "stated-" (name (or kind :unknown)))
     :key (str (or id mission ""))
     ;; baseline weak magnet = the id-stem the live lane would strip to
     :idstem (str (or id mission ""))
     ;; want = the goal described (richest text available for the outcome)
     :want_raw (str/join " " (remove nil? [(or title prov-title) (str (or id mission ""))]))
     ;; have = current mission state (thin for stated: just the held/open status)
     :have_raw (str (some-> prov-status name))
     :title (or title prov-title)
     :delta_g nil}))

(defn incomplete-item [{:keys [outcome-ref provenance]}]
  (let [{:keys [mission]} outcome-ref
        phases (:phases-present provenance)]
    {:channel "incomplete"
     :key (str mission)
     :idstem (str mission)
     ;; want = reach the mission's terminal outcome (the mission itself, run)
     :want_raw (str mission " run complete")
     ;; have = the phases actually reached = the reconstructed CURRENT state
     :have_raw (str/join " " (or phases []))
     :title nil
     :delta_g nil}))

(defn mess-item [{:keys [outcome-ref provenance]}]
  {:channel "mess"
   :key (str (:mission provenance) "|" (name (:kind outcome-ref)))
   :idstem (str (or (:mission provenance) "coherence"))
   :want_raw (str (or (:mission provenance) "") " coherence wholeness rise")
   :have_raw "low coherence mess"
   :title nil
   :delta_g nil})

(defn diffsub-item [{:keys [have want delta-g] :as m}]
  {:channel "diffsub"
   :key (str (:move/id m))
   :idstem (str have)
   :want_raw (str want)
   :have_raw (str have)
   :title nil
   :delta_g delta-g})

(let [stated     (map stated-item     (:entries (slurp-edn (str CV "/c-entries.stated.edn"))))
      incomplete (map incomplete-item (:entries (slurp-edn (str CV "/c-entries.incomplete.edn"))))
      mess       (map mess-item       (:entries (slurp-edn (str CV "/c-entries.mess.edn"))))
      diffsub    (map diffsub-item    (:moves   (slurp-edn DIFFSUB)))
      all        (concat stated incomplete mess diffsub)]
  (binding [*out* *err*]
    (println (format "extracted: stated=%d incomplete=%d mess=%d diffsub=%d total=%d"
                     (count stated) (count incomplete) (count mess) (count diffsub) (count all))))
  (println (json/generate-string all)))
