import json

import pytest

from eval.stats import (
    McNemarResult,
    PairedR1,
    clopper_pearson,
    load_run,
    mcnemar,
    paired_r1,
)
from ingestion.documents import REPO_ROOT

BASELINE = REPO_ROOT / "eval" / "runs" / "2026-11-22-dense-baseline.jsonl"

# (b, c, exact_p, mid_p); every value here is an exact binary fraction
EXACT_VECTORS = [
    (0, 0, 1.0, 1.0),
    (5, 0, 0.0625, 0.03125),
    (6, 0, 0.03125, 0.015625),
    (2, 0, 0.5, 0.25),
    (3, 3, 1.0, 0.84375),
    (8, 1, 0.0390625, 0.021484375),
    (1, 7, 0.0703125, 0.0390625),
    (8, 2, 0.109375, 0.0654296875),
]

# (b, c, exact_p, mid_p), rounded to 10 decimals
ROUNDED_VECTORS = [
    (6, 16, 0.0524787903, 0.0346896648),
    (10, 3, 0.0922851562, 0.0573730469),
    (12, 4, 0.0768127441, 0.0490417480),
]


@pytest.mark.parametrize(("b", "c", "exact_p", "mid_p"), EXACT_VECTORS)
def test_mcnemar_exact_vectors(b, c, exact_p, mid_p):
    assert mcnemar(b, c) == McNemarResult(b, c, b + c, exact_p, mid_p)
    assert mcnemar(c, b) == McNemarResult(c, b, b + c, exact_p, mid_p)


@pytest.mark.parametrize(("b", "c", "exact_p", "mid_p"), ROUNDED_VECTORS)
def test_mcnemar_rounded_vectors(b, c, exact_p, mid_p):
    for result in (mcnemar(b, c), mcnemar(c, b)):
        assert result.n == b + c
        assert result.exact_p == pytest.approx(exact_p, rel=0, abs=1e-10)
        assert result.mid_p == pytest.approx(mid_p, rel=0, abs=1e-10)


@pytest.mark.parametrize(("b", "c"), [(-1, 0), (0, -1), (-2, 5)])
def test_mcnemar_rejects_negative_counts(b, c):
    with pytest.raises(ValueError):
        mcnemar(b, c)


@pytest.mark.parametrize(
    ("x", "n", "lower", "upper"),
    [
        (0, 44, 0.0, 0.0804),
        (0, 32, 0.0, 0.1089),
        (1, 44, 0.0006, 0.1202),
        (0, 45, 0.0, 0.0787),
    ],
)
def test_clopper_pearson_vectors(x, n, lower, upper):
    assert clopper_pearson(x, n) == pytest.approx((lower, upper), rel=0, abs=1e-4)


def test_clopper_pearson_pins_the_closed_ends():
    assert clopper_pearson(0, 44)[0] == 0.0
    assert clopper_pearson(44, 44)[1] == 1.0


@pytest.mark.parametrize("level", [0.9, 0.95, 0.99])
def test_clopper_pearson_matches_closed_form_at_the_ends(level):
    # With x = 0 or x = n the tail equation has an algebraic solution.
    n = 44
    root = ((1 - level) / 2) ** (1 / n)
    assert clopper_pearson(0, n, level)[1] == pytest.approx(1 - root, rel=0, abs=1e-12)
    assert clopper_pearson(n, n, level)[0] == pytest.approx(root, rel=0, abs=1e-12)


def test_clopper_pearson_is_symmetric_in_successes_and_failures():
    lower, upper = clopper_pearson(7, 45)
    mirrored = clopper_pearson(38, 45)
    assert mirrored == pytest.approx((1 - upper, 1 - lower), rel=0, abs=2e-12)


@pytest.mark.parametrize(
    ("x", "n", "level"),
    [
        (0, 0, 0.95),
        (0, -1, 0.95),
        (-1, 10, 0.95),
        (11, 10, 0.95),
        (5, 10, 0.0),
        (5, 10, 1.0),
        (5, 10, -0.5),
        (5, 10, 1.5),
    ],
)
def test_clopper_pearson_rejects_invalid_arguments(x, n, level):
    with pytest.raises(ValueError):
        clopper_pearson(x, n, level)


def _write_run(path, rows):
    """Write a run file in the run_eval --out format; rows are (id, group, article, clause)."""
    lines = [
        json.dumps(
            {
                "id": question_id,
                "group": group,
                "pair_id": None,
                "rank_article": rank_article,
                "rank_clause": rank_clause,
                "top": [],
            }
        )
        for question_id, group, rank_article, rank_clause in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_run_picks_the_rank_of_the_requested_level(tmp_path):
    path = _write_run(
        tmp_path / "run.jsonl",
        [("q1", "direct", 1, 3), ("q2", "out_of_scope", None, None)],
    )
    article = {"q1": ("direct", 1), "q2": ("out_of_scope", None)}
    assert load_run(path) == article
    assert load_run(path, "article") == article
    assert load_run(path, "clause") == {"q1": ("direct", 3), "q2": ("out_of_scope", None)}


def test_load_run_rejects_a_duplicate_question_id(tmp_path):
    path = _write_run(
        tmp_path / "duplicated.jsonl",
        [("q1", "direct", 1, 1), ("q2", "direct", 1, 1), ("q1", "direct", 2, 2)],
    )
    with pytest.raises(ValueError) as error:
        load_run(path)
    assert "q1" in str(error.value)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("level", ["Article", "chunk", ""])
def test_load_run_rejects_an_unknown_level(tmp_path, level):
    path = _write_run(tmp_path / "run.jsonl", [("q1", "direct", 1, 1)])
    with pytest.raises(ValueError, match="level"):
        load_run(path, level)


def test_paired_r1_counts_gains_and_losses_within_the_group(tmp_path):
    # (id, group, rank in A, rank in B)
    ranks = [
        ("p1", "paraphrase", 2, 1),  # gain
        ("p2", "paraphrase", None, 1),  # gain
        ("p3", "paraphrase", 10, 1),  # gain
        ("p4", "paraphrase", 1, 2),  # loss
        ("p5", "paraphrase", 1, 1),  # hit in both
        ("p6", "paraphrase", 3, None),  # miss in both
        ("d1", "direct", 1, None),  # another group: must not be counted
        ("d2", "direct", 4, 1),
    ]
    run_a = load_run(_write_run(tmp_path / "a.jsonl", [(i, g, a, a) for i, g, a, _ in ranks]))
    run_b = load_run(
        _write_run(tmp_path / "b.jsonl", [(i, g, b, b) for i, g, _, b in reversed(ranks)])
    )

    result = paired_r1(run_a, run_b, "paraphrase")

    assert result == PairedR1("paraphrase", n=6, hits_a=2, hits_b=4, gains=3, losses=1)
    assert paired_r1(run_b, run_a, "paraphrase") == PairedR1(
        "paraphrase", n=6, hits_a=4, hits_b=2, gains=1, losses=3
    )


@pytest.mark.parametrize(
    "run_b",
    [
        {"q1": ("direct", 1)},  # id missing
        {"q1": ("direct", 1), "q2": ("paraphrase", 1), "q3": ("direct", 1)},  # extra id
        {"q1": ("direct", 1), "q9": ("paraphrase", 1)},  # different id
        {"q1": ("direct", 1), "q2": ("direct", 1)},  # different group
    ],
)
@pytest.mark.parametrize("group", ["direct", "paraphrase"])
def test_paired_r1_rejects_runs_over_different_questions(run_b, group):
    run_a = {"q1": ("direct", 1), "q2": ("paraphrase", 2)}
    with pytest.raises(ValueError):
        paired_r1(run_a, run_b, group)
    with pytest.raises(ValueError):
        paired_r1(run_b, run_a, group)


@pytest.mark.parametrize("group", ["paraphase", "article_ref"])
def test_paired_r1_rejects_a_group_without_questions(group):
    run = {"q1": ("direct", 1), "q2": ("paraphrase", 2)}
    with pytest.raises(ValueError, match=group):
        paired_r1(run, run, group)


@pytest.mark.parametrize(
    ("group", "n", "hits"),
    [
        ("direct", 45, 44),
        ("paraphrase", 45, 32),
        ("article_ref", 12, 9),
        ("out_of_scope", 25, 0),
    ],
)
def test_baseline_run_against_itself_at_article_level(group, n, hits):
    base = load_run(BASELINE)
    assert paired_r1(base, base, group) == PairedR1(group, n, hits, hits, gains=0, losses=0)


def test_baseline_run_against_itself_at_clause_level():
    base = load_run(BASELINE, "clause")
    results = {
        group: paired_r1(base, base, group)
        for group in ("direct", "paraphrase", "article_ref", "out_of_scope")
    }
    assert results["direct"].hits_a == 41
    assert results["paraphrase"].hits_a == 29
    assert results["article_ref"].hits_a == 9
    for result in results.values():
        assert result.hits_b == result.hits_a
        assert (result.gains, result.losses) == (0, 0)
