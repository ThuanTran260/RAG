"""Paired recall@1 statistics for comparing two retrieval runs. Standard library only."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Literal

# Width of the final bisection bracket; the returned midpoint is closer than this to the root.
BISECTION_TOLERANCE = Fraction(1, 10**12)


@dataclass(frozen=True)
class McNemarResult:
    b: int  # B correct, A wrong (gains of B)
    c: int  # A correct, B wrong (losses of B)
    n: int  # b + c
    exact_p: float  # two-sided exact conditional (binomial) p-value
    mid_p: float  # two-sided mid-p


def mcnemar(b: int, c: int) -> McNemarResult:
    """Exact conditional and mid-p McNemar tests from the two discordant counts.

    Under H0, X ~ Binomial(n = b + c, 1/2). Extremeness is the distance from the centre,
    D(x) = |x - n/2|, and d_obs = |b - n/2|:

        exact_p = P(D(X) >= d_obs)
        mid_p   = P(D(X) > d_obs) + 0.5 * P(D(X) = d_obs)

    With n = 0 both p-values are 1.0. For b != c this equals the exact conditional and mid-p
    tests of Fagerland, Lydersen & Laake (2013); for b == c it is the symmetric-distance
    convention.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must not be negative")
    n = b + c
    if n == 0:
        return McNemarResult(b, c, n, exact_p=1.0, mid_p=1.0)
    # Distances are doubled, |2x - n|, so that they stay integers.
    observed = abs(b - c)
    beyond = sum(comb(n, x) for x in range(n + 1) if abs(2 * x - n) > observed)
    at = sum(comb(n, x) for x in range(n + 1) if abs(2 * x - n) == observed)
    exact_p = Fraction(beyond + at, 2**n)
    mid_p = Fraction(2 * beyond + at, 2 ** (n + 1))
    return McNemarResult(b, c, n, float(exact_p), float(mid_p))


def _upper_tail(k: int, n: int, p: Fraction) -> Fraction:
    """P(X >= k) for X ~ Binomial(n, p), computed exactly."""
    q = 1 - p
    return sum((comb(n, j) * p**j * q ** (n - j) for j in range(k, n + 1)), Fraction(0))


def _solve_upper_tail(k: int, n: int, target: Fraction) -> float:
    """The p with P(X >= k | n, p) = target, by bisection; the tail increases with p (k >= 1)."""
    low, high = Fraction(0), Fraction(1)
    while high - low > BISECTION_TOLERANCE:
        middle = (low + high) / 2
        if _upper_tail(k, n, middle) < target:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def clopper_pearson(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided (Clopper-Pearson) interval for x successes out of n trials.

    With alpha = 1 - level, the lower bound is the p with P(X >= x | n, p) = alpha/2
    (0.0 when x == 0) and the upper bound is the p with P(X <= x | n, p) = alpha/2
    (1.0 when x == n). Both are accurate to an absolute error below 1e-12.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= x <= n:
        raise ValueError("x must be between 0 and n")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    half_alpha = (1 - Fraction(level)) / 2
    lower = 0.0 if x == 0 else _solve_upper_tail(x, n, half_alpha)
    # P(X <= x) = alpha/2 is the same equation as P(X >= x + 1) = 1 - alpha/2.
    upper = 1.0 if x == n else _solve_upper_tail(x + 1, n, 1 - half_alpha)
    return lower, upper


def load_run(
    path: Path, level: Literal["article", "clause"] = "article"
) -> dict[str, tuple[str, int | None]]:
    """Read a run file written by `eval/run_eval.py --out` as question id -> (group, rank)."""
    rank_key = f"rank_{level}"
    run: dict[str, tuple[str, int | None]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            run[row["id"]] = (row["group"], row[rank_key])
    return run


@dataclass(frozen=True)
class PairedR1:
    group: str
    n: int  # questions of that group
    hits_a: int  # rank == 1 in run A
    hits_b: int
    gains: int  # rank 1 in B but not in A   (= McNemar b)
    losses: int  # rank 1 in A but not in B   (= McNemar c)


def paired_r1(
    run_a: Mapping[str, tuple[str, int | None]],
    run_b: Mapping[str, tuple[str, int | None]],
    group: str,
) -> PairedR1:
    """Count recall@1 hits of two runs over the questions of one group.

    Raises ValueError unless both runs hold exactly the same question ids with the same groups.
    """
    groups_a = {question_id: entry[0] for question_id, entry in run_a.items()}
    groups_b = {question_id: entry[0] for question_id, entry in run_b.items()}
    if groups_a != groups_b:
        differing = sorted(
            question_id
            for question_id in groups_a.keys() | groups_b.keys()
            if groups_a.get(question_id) != groups_b.get(question_id)
        )
        raise ValueError(f"runs differ in question ids or groups: {', '.join(differing)}")
    ids = [question_id for question_id, name in groups_a.items() if name == group]
    pairs = [(run_a[i][1] == 1, run_b[i][1] == 1) for i in ids]
    return PairedR1(
        group=group,
        n=len(pairs),
        hits_a=sum(a for a, _ in pairs),
        hits_b=sum(b for _, b in pairs),
        gains=sum(b and not a for a, b in pairs),
        losses=sum(a and not b for a, b in pairs),
    )
