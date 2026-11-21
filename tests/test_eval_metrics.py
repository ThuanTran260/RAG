import pytest

from eval.metrics import (
    PairTable,
    ScoreSummary,
    count_above,
    format_score,
    hit_rank,
    mrr,
    pair_table,
    recall_at,
    summarize,
)

RANKS = [1, 5, 6, 10, None]


def test_hit_rank_returns_first_hit_one_based():
    assert hit_rank(["a", "b", "c"], lambda x: x in {"b", "c"}) == 2


def test_hit_rank_returns_none_without_hit():
    assert hit_rank(["a", "b"], lambda x: x == "z") is None
    assert hit_rank([], lambda x: True) is None


def test_recall_at_counts_ranks_within_k():
    assert recall_at(RANKS, 5) == 0.4
    assert recall_at(RANKS, 10) == 0.8


def test_mrr_averages_reciprocal_ranks():
    assert mrr(RANKS) == pytest.approx((1 + 0.2 + 1 / 6 + 0.1) / 5)


def test_recall_and_mrr_reject_empty():
    with pytest.raises(ValueError):
        recall_at([], 5)
    with pytest.raises(ValueError):
        mrr([])


def test_pair_table_fills_four_cells():
    hits = {
        "p01": (True, True),
        "p02": (True, False),
        "p03": (False, True),
        "p04": (False, False),
        "p05": (True, False),
    }
    assert pair_table(hits) == PairTable(1, 2, 1, 1, ["p02", "p05"])


def test_pair_table_sorts_direct_only_pairs():
    hits = {"p09": (True, False), "p02": (True, False)}
    assert pair_table(hits).direct_only_pairs == ["p02", "p09"]


def test_summarize_and_count_above():
    scores = [0.3, 0.5, 0.4]
    assert summarize(scores) == ScoreSummary(3, 0.3, 0.4, 0.5)
    assert count_above(scores, 0.4) == 1


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_format_score_uses_decimal_comma():
    assert format_score(0.7125) == "0,713"
    assert format_score(1.0) == "1,000"
    assert format_score(0.0) == "0,000"
