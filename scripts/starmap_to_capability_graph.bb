#!/usr/bin/env bb
;; starmap_to_capability_graph.bb — derive data/capability-graph.json from the
;; curated star-map EDN (futon0/holes/missions/M-capability-star-map.graph.edn).
;; The EDN is the source of truth (WM reads it directly); this JSON is the
;; EFE-field projection. Mechanically verified on first run by reproducing the
;; prior JSON for untouched capabilities.
(require '[clojure.edn :as edn]
         '[cheshire.core :as json])
(let [graph (edn/read-string
             (slurp "/home/joe/code/futon0/holes/missions/M-capability-star-map.graph.edn"))
      caps (:capabilities graph)
      out (into (sorted-map)
                (map (fn [[k v]]
                       [(name k)
                        {:title (:title v)
                         :status (str (:status v))
                         :claimed (boolean (and (= :satisfied (:status v))
                                                (seq (:minted-by v))))
                         :minted_by (vec (or (:minted-by v) []))
                         :scope (mapv name (or (:scope v) []))
                         :frontier (boolean (or (:frontier v) (:pre-registered? v)))}]))
                caps)]
  (spit "data/capability-graph.json" (json/generate-string out {:pretty true}))
  (println "wrote" (count out) "capabilities;"
           (count (filter :claimed (vals out))) "claimed"))
