import numpy as np

from gesn.evaluate import bootstrap_ci, per_query_metrics, table_of, to_tables
from gesn.retrievers import HybridRetriever, tokenize


def test_per_query_metrics_hit_at_rank_3():
    m = per_query_metrics(["a", "b", "c", "d"], {"c"})
    assert m == {"R@1": 0.0, "R@5": 1.0, "R@10": 1.0, "MRR@10": 1 / 3}


def test_per_query_metrics_uses_best_of_several_relevant():
    m = per_query_metrics(["a", "b", "c"], {"c", "b"})
    assert m["MRR@10"] == 0.5


def test_per_query_metrics_miss():
    m = per_query_metrics(["a", "b"], {"z"})
    assert m["R@10"] == 0.0 and m["MRR@10"] == 0.0


def test_to_tables_dedupes_preserving_order():
    codes = ["15-04-005-02", "15-04-005-01", "11-01-011-01", "15-04-005-03"]
    assert to_tables(codes) == ["15-04-005", "11-01-011"]
    assert table_of("06-01-001-12") == "06-01-001"


def test_bootstrap_ci_contains_mean():
    values = np.array([0, 1] * 50, dtype=float)
    lo, hi = bootstrap_ci(values)
    assert lo < 0.5 < hi


def test_tokenize_lemmatizes_and_drops_stopwords():
    assert tokenize("Покрасить стены в квартире") == ["покрасить", "стена", "квартира"]
    assert tokenize("стяжка 50 мм") == ["стяжка", "50", "мм"]


class Fixed:
    def __init__(self, name, ranking):
        self.name, self.ranking = name, ranking

    def search(self, query, k=10):
        return self.ranking[:k]


def test_hybrid_rrf_rewards_agreement():
    hybrid = HybridRetriever([Fixed("a", ["x", "y", "z"]), Fixed("b", ["y", "z", "x"])])
    assert hybrid.search("q", 3)[0] == "y"
