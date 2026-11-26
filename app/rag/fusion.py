"""Reciprocal rank fusion of a dense and a sparse ranking (spec §6.3).

The fusion runs in Python on full rankings. Qdrant's server-side RRF uses another constant and
counts ranks from 0, so it is never used.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

RRF_K = 60


@dataclass(frozen=True)
class FusedRank:
    key: str
    score: float  # the exact RRF sum, rounded once
    dense_rank: int | None  # 1-based; None when the dense ranking lacks the key
    sparse_rank: int | None


def _ranks(ranking: Sequence[str], name: str) -> dict[str, int]:
    """The 1-based rank of every key of one ranking, which must not repeat a key."""
    ranks = {key: rank for rank, key in enumerate(ranking, start=1)}
    if len(ranks) != len(ranking):
        raise ValueError(f"the {name} ranking holds a repeated key")
    return ranks


def reciprocal_rank_fusion(
    dense: Sequence[str], sparse: Sequence[str], k: int = RRF_K
) -> list[FusedRank]:
    """Fuse two rankings of keys, each best first, with equal weights: sum of 1 / (k + rank).

    Equal scores are ordered by dense rank (a key without one comes last), then by key.
    """
    dense_ranks = _ranks(dense, "dense")
    sparse_ranks = _ranks(sparse, "sparse")
    # Exact fractions: float sums split true ties, e.g. ranks (12, 60) and (30, 30), which
    # would then be ordered by rounding error instead of by the dense rank.
    scores = {
        key: sum(
            Fraction(1, k + ranks[key]) for ranks in (dense_ranks, sparse_ranks) if key in ranks
        )
        for key in dense_ranks.keys() | sparse_ranks.keys()
    }
    order = sorted(scores, key=lambda key: (-scores[key], dense_ranks.get(key, math.inf), key))
    return [
        FusedRank(key, float(scores[key]), dense_ranks.get(key), sparse_ranks.get(key))
        for key in order
    ]
