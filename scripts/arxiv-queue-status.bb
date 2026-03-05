#!/usr/bin/env bb
(ns arxiv-queue-status
  (:require [cheshire.core :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(def default-manifest "data/arxiv-math-ct/entities.json")

(defn load-entities [path]
  (json/parse-string (slurp path) true))

(defn fmt-num [n]
  (format "%,d" (long n)))

(defn describe-categories [entities]
  (->> entities
       (mapcat #(or (:categories %) []))
       (remove str/blank?)
       (frequencies)
       (sort-by val >)
       (take 5)
       (map (fn [[cat n]] (str cat "=" (fmt-num n))))
       (str/join ", ")))

(defn main [& args]
  (let [manifest (or (first args) default-manifest)
        file (io/file manifest)]
    (when-not (.exists file)
      (binding [*out* *err*]
        (println "Manifest not found:" manifest))
      (System/exit 1))
    (let [entities (load-entities file)
          total (count entities)
          total-bytes (reduce + (map #(or (:body_length %) 0) entities))
          avg (if (pos? total) (Math/round (double (/ total-bytes total))) 0)
          sample (first entities)
          categories (describe-categories entities)]
      (println (format "[arxiv-queue] %s entries, avg body_length=%s bytes"
                       (fmt-num total) (fmt-num avg)))
      (println (format "[arxiv-queue] category mix: %s"
                       (if (seq categories) categories "<none>")))
      (when sample
        (let [preview (or (:body_preview sample) "")]
          (println (format "[arxiv-queue] sample %s — %s..."
                           (:entity_id sample)
                           (subs preview 0 (min 120 (count preview))))))))))

(main)
