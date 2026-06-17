import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mark3_diachronic", ROOT / "scripts" / "mark3_diachronic.py")
mark3_diachronic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mark3_diachronic
SPEC.loader.exec_module(mark3_diachronic)


def test_parse_arxiv_month_new_and_old_styles():
    assert mark3_diachronic.parse_arxiv_month("0705.0452") == "2007-05"
    assert mark3_diachronic.parse_arxiv_month("2401.14311") == "2024-01"
    assert mark3_diachronic.parse_arxiv_month("math/9811139") == "1998-11"
    assert mark3_diachronic.parse_arxiv_month("math__0210114") == "2002-10"
    assert mark3_diachronic.parse_arxiv_month("fable-math__0304277-dp-emacs.json") == "2003-04"


def test_trend_scores_rising_above_flat_series():
    buckets = {str(year): 10 for year in range(2000, 2006)}
    rising = {str(year): max(0, year - 2001) for year in range(2000, 2006)}
    flat = {str(year): 2 for year in range(2000, 2006)}

    rising_trend = mark3_diachronic.trend_for_series("rising", rising, buckets)
    flat_trend = mark3_diachronic.trend_for_series("flat", flat, buckets)

    assert rising_trend.trend["slope"] > 0
    assert abs(flat_trend.trend["slope"]) < 1e-12
    assert rising_trend.emergence_score > flat_trend.emergence_score
    assert rising_trend.first_seen == "2002"
    assert rising_trend.peak_year == "2005"


def test_rank_emerging_terms_prefers_recent_growth():
    bucket_docs = {str(year): 10 for year in range(2000, 2006)}
    term_counts = {
        "rising term": {"2000": 0, "2001": 0, "2002": 1, "2003": 2, "2004": 4, "2005": 5},
        "flat term": {"2000": 2, "2001": 2, "2002": 2, "2003": 2, "2004": 2, "2005": 2},
    }

    ranked = mark3_diachronic.rank_emerging_terms(term_counts, bucket_docs, min_df=4, top_n=2)

    assert ranked[0].term == "rising term"
    assert ranked[0].emergence_score > 0


def test_candidate_terms_limit_ranked_surface():
    bucket_docs = {str(year): 10 for year in range(2000, 2006)}
    term_counts = {
        "rising term": {"2000": 0, "2001": 0, "2002": 1, "2003": 2, "2004": 4, "2005": 5},
        "other rising": {"2000": 0, "2001": 0, "2002": 1, "2003": 2, "2004": 4, "2005": 5},
    }

    ranked = mark3_diachronic.rank_emerging_terms(
        term_counts,
        bucket_docs,
        min_df=4,
        top_n=5,
        candidate_terms={"rising term"},
    )

    assert [r.term for r in ranked] == ["rising term"]


def test_texish_terms_filtered_by_default():
    bucket_docs = {str(year): 10 for year in range(2000, 2006)}
    term_counts = {
        "begin proof": {"2000": 0, "2001": 0, "2002": 1, "2003": 2, "2004": 4, "2005": 5},
        "derived stack": {"2000": 0, "2001": 0, "2002": 1, "2003": 2, "2004": 4, "2005": 5},
    }

    ranked = mark3_diachronic.rank_emerging_terms(term_counts, bucket_docs, min_df=4, top_n=5)
    raw = mark3_diachronic.rank_emerging_terms(
        term_counts, bucket_docs, min_df=4, top_n=5, include_texish=True
    )

    assert [r.term for r in ranked] == ["derived stack"]
    assert {r.term for r in raw} == {"begin proof", "derived stack"}
