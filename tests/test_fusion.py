"""Reciprocal rank fusion against values computed by hand (spec §6.3)."""

from fractions import Fraction

import pytest

from app.rag.fusion import RRF_K, FusedRank, reciprocal_rank_fusion


def keys(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{number}" for number in range(1, count + 1)]


def ranking_with(length: int, filler: str, **placed: int) -> list[str]:
    """A ranking of ``length`` keys in which each named key sits at the given 1-based rank."""
    ranking = keys(filler, length)
    for key, rank in placed.items():
        ranking[rank - 1] = key
    return ranking


def by_key(fused: list[FusedRank]) -> dict[str, FusedRank]:
    return {item.key: item for item in fused}


def test_scores_and_order_match_the_hand_computed_example():
    fused = reciprocal_rank_fusion(["a", "b", "c", "d", "e"], ["c", "a", "e"])

    assert RRF_K == 60
    # a: 1/61 + 1/62, c: 1/63 + 1/61, e: 1/65 + 1/63, b: 1/62, d: 1/64.
    assert fused == [
        FusedRank("a", 123 / 3782, dense_rank=1, sparse_rank=2),
        FusedRank("c", 124 / 3843, dense_rank=3, sparse_rank=1),
        FusedRank("e", 128 / 4095, dense_rank=5, sparse_rank=3),
        FusedRank("b", 1 / 62, dense_rank=2, sparse_rank=None),
        FusedRank("d", 1 / 64, dense_rank=4, sparse_rank=None),
    ]


def test_equal_scores_are_ordered_by_dense_rank():
    # Both score 1/61 + 1/62. "y" comes first in the sparse ranking and sorts first as a name
    # in the reverse order, so only the dense rank can put it second.
    fused = reciprocal_rank_fusion(["z", "y"], ["y", "z"])

    assert [(item.key, item.score) for item in fused] == [
        ("z", 123 / 3782),
        ("y", 123 / 3782),
    ]


def test_a_tie_is_exact_even_when_float_sums_differ():
    # 1/72 + 1/120 and 1/90 + 1/90 are both 1/45, but the float sums differ in the last bit.
    assert 1 / 72 + 1 / 120 < 1 / 90 + 1 / 90
    assert Fraction(1, 72) + Fraction(1, 120) == Fraction(1, 90) + Fraction(1, 90)
    dense = ranking_with(30, "d", first=12, second=30)
    sparse = ranking_with(60, "s", first=60, second=30)

    fused = reciprocal_rank_fusion(dense, sparse)

    order = [item.key for item in fused]
    assert order.index("first") < order.index("second")
    items = by_key(fused)
    assert items["first"].score == items["second"].score == 1 / 45


def test_a_dense_only_item_wins_a_tie_by_its_dense_rank():
    # 1/70 for the dense-only item, 1/140 + 1/140 for the item at rank 80 of both rankings.
    dense = ranking_with(80, "d", alone=10, both=80)
    sparse = ranking_with(80, "s", both=80)

    fused = reciprocal_rank_fusion(dense, sparse)

    order = [item.key for item in fused]
    assert order.index("alone") + 1 == order.index("both")
    assert by_key(fused)["alone"].score == by_key(fused)["both"].score == 1 / 70


def test_an_item_without_a_dense_rank_loses_a_tie():
    fused = reciprocal_rank_fusion(["b"], ["a"])

    assert fused == [
        FusedRank("b", 1 / 61, dense_rank=1, sparse_rank=None),
        FusedRank("a", 1 / 61, dense_rank=None, sparse_rank=1),
    ]


def test_another_constant_changes_the_scores():
    assert reciprocal_rank_fusion(["a"], ["a"], k=1) == [
        FusedRank("a", 1.0, dense_rank=1, sparse_rank=1)
    ]


@pytest.mark.parametrize(
    ("dense", "sparse", "expected"),
    [
        (["b", "a", "c"], [], ["b", "a", "c"]),
        ([], ["c", "a"], ["c", "a"]),
        ([], [], []),
    ],
)
def test_an_empty_ranking_keeps_the_other_order(dense, sparse, expected):
    assert [item.key for item in reciprocal_rank_fusion(dense, sparse)] == expected


@pytest.mark.parametrize(("dense", "sparse"), [(["a", "a"], []), (["a"], ["b", "b"])])
def test_a_repeated_key_is_rejected(dense, sparse):
    with pytest.raises(ValueError, match="repeated"):
        reciprocal_rank_fusion(dense, sparse)
