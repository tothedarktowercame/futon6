#!/usr/bin/env bb
;; starmap_to_capability_graph.bb — derive data/capability-graph.json from the
;; curated star-map EDN (futon0/holes/missions/M-capability-star-map.graph.edn).
;; The EDN is the source of truth (WM reads it directly); this JSON is the
;; EFE-field projection. Mechanically verified on first run by reproducing the
;; prior JSON for untouched capabilities.
(require '[clojure.edn :as edn]
         '[cheshire.core :as json])
(require '[clojure.java.io])
;; Roots derive from THIS script's location; the documented environment variables
;; override them. No absolute path belonging to one host is baked in here.
(def ^:private repo (.getParentFile (.getParentFile (clojure.java.io/file *file*))))
(defn- configured [env fallback]
  (or (System/getenv env) fallback))
(def ROOT (configured "FUTON6_CHECKOUT" (.getPath repo)))
(def CODE-ROOT (configured "FUTON_CODE_ROOT" (.getPath (.getParentFile repo))))
(defn- sibling [env name] (configured env (str CODE-ROOT "/" name)))
(let [graph (edn/read-string
             (slurp (str (sibling "FUTON0_ROOT" "futon0") "/holes/missions/M-capability-star-map.graph.edn")))
      caps (:capabilities graph)
      ;; Visual coalescing (Joe, 2026-06-12): CLAIMED members of a named
      ;; cluster merge into ONE star on the map; unclaimed members remain
      ;; individual islands and are absorbed when claimed. The curated EDN
      ;; keeps every entry individual — this is projection-layer only.
      clusters {"pudding-kit" #(re-matches #"kit-.*" (name %))}
      claimed? (fn [v] (boolean (and (= :satisfied (:status v))
                                     (seq (:minted-by v)))))
      cluster-of (fn [k] (some (fn [[cname pred]] (when (pred k) cname)) clusters))
      grouped (group-by (fn [[k v]] (when (claimed? v) (cluster-of k))) caps)
      singles (get grouped nil)
      out (into (sorted-map)
                (map (fn [[k v]]
                       [(name k)
                        {:title (:title v)
                         :status (str (:status v))
                         :claimed (claimed? v)
                         :minted_by (vec (or (:minted-by v) []))
                         :scope (mapv name (or (:scope v) []))
                         :frontier (boolean (or (:frontier v) (:pre-registered? v)))}]))
                singles)
      out (reduce (fn [acc [cname _]]
                    (let [members (get grouped cname)]
                      (if (seq members)
                        (assoc acc cname
                               {:title (format "%s — %d built kits coalesced: %s"
                                               cname (count members)
                                               (clojure.string/join ", " (map (comp name first) members)))
                                :status ":satisfied"
                                :claimed true
                                :coalesced (mapv (comp name first) members)
                                :minted_by (vec (distinct (mapcat (comp :minted-by second) members)))
                                :scope (vec (distinct (mapv name (mapcat (comp :scope second) members))))
                                :frontier false})
                        acc)))
                  out clusters)]
  (spit "data/capability-graph.json" (json/generate-string out {:pretty true}))
  (println "wrote" (count out) "capabilities;"
           (count (filter :claimed (vals out))) "claimed"))
