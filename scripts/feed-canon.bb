#!/usr/bin/env bb
;; Feed hungry-for-canon scopes from canon-links.edn (THE KILLER IDEA loop:
;; census -> feed -> re-measure). Match by node :form; on feed, :canon gets
;; the URL and :satiety flips to :full. Unmatched stay hungry.
(require '[clojure.edn :as edn] '[clojure.pprint :as pp]
         '[clojure.string :as str] '[clojure.java.io :as io])
(def links (edn/read-string (slurp "holes/golden-graphs/canon-links.edn")))
(defn feed [n]
  (if-let [url (and (= {:hungry-for :canon} (:satiety n)) (links (:form n)))]
    (assoc n :canon {:nlab url} :satiety :full :fed-at "2026-06-12")
    n))
(doseq [f (sort (filter #(and (str/ends-with? (.getName %) ".edn")
                              (not= "canon-links.edn" (.getName %)))
                        (file-seq (io/file "holes/golden-graphs"))))]
  (let [text (slurp f)
        header (->> (str/split-lines text)
                    (take-while #(or (str/starts-with? % ";;") (str/blank? %)))
                    (str/join "\n"))
        g (edn/read-string text)
        g' (update g :nodes #(mapv feed %))
        fed (- (count (filter #(not= :full (:satiety %)) (:nodes g)))
               (count (filter #(not= :full (:satiety %)) (:nodes g'))))]
    (when (pos? fed)
      (spit f (str header "\n" (with-out-str (pp/pprint g')))))
    (println (format "%-50s fed=%d still-hungry=%d" (.getName (io/file f)) fed
                     (count (filter #(not= :full (:satiety %)) (:nodes g')))))))
