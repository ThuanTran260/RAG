"""Pure retrieval metrics: no I/O, no retriever, no model."""

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


def hit_rank[T](retrieved: Sequence[T], is_hit: Callable[[T], bool]) -> int | None:
    """Return the 1-based rank of the first hit, or None when nothing hits."""
    for rank, item in enumerate(retrieved, start=1):
        if is_hit(item):
            return rank
    return None


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of questions whose first hit is within the top k."""
    if not ranks:
        raise ValueError("ranks must not be empty")
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr(ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank; a missing hit counts as 0."""
    if not ranks:
        raise ValueError("ranks must not be empty")
    return sum(0.0 if rank is None else 1 / rank for rank in ranks) / len(ranks)


@dataclass(frozen=True)
class PairTable:
    """2x2 outcome counts for direct/paraphrase question pairs."""

    both: int
    direct_only: int
    paraphrase_only: int
    neither: int
    direct_only_pairs: list[str]


def pair_table(hits: Mapping[str, tuple[bool, bool]]) -> PairTable:
    """Count pairs by (direct_hit, paraphrase_hit); keys are pair ids."""
    both = paraphrase_only = neither = 0
    direct_only_pairs: list[str] = []
    for pair_id, (direct_hit, paraphrase_hit) in hits.items():
        if direct_hit and paraphrase_hit:
            both += 1
        elif direct_hit:
            direct_only_pairs.append(pair_id)
        elif paraphrase_hit:
            paraphrase_only += 1
        else:
            neither += 1
    return PairTable(
        both=both,
        direct_only=len(direct_only_pairs),
        paraphrase_only=paraphrase_only,
        neither=neither,
        direct_only_pairs=sorted(direct_only_pairs),
    )


@dataclass(frozen=True)
class ScoreSummary:
    count: int
    minimum: float
    median: float
    maximum: float


def summarize(scores: Sequence[float]) -> ScoreSummary:
    """Return count, min, median and max of a non-empty score list."""
    if not scores:
        raise ValueError("scores must not be empty")
    return ScoreSummary(
        count=len(scores),
        minimum=min(scores),
        median=statistics.median(scores),
        maximum=max(scores),
    )


def count_above(scores: Sequence[float], threshold: float) -> int:
    """Count scores strictly greater than the threshold."""
    return sum(1 for score in scores if score > threshold)


def format_score(value: float) -> str:
    """Format with 3 decimals and a decimal comma, e.g. 0.7125 -> '0,713'."""
    return f"{value:.3f}".replace(".", ",")
