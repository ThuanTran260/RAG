"""Retrieval evaluation: uv run python -m eval.run_eval [--questions PATH] [--out PATH] ...

Runs every question through one retriever once (k = 10), then prints a report and one line for
eval/results.md. With no retrieval flag that is default_retriever(), the dense baseline;
--retriever, --collection and --promote pick another configuration. --out saves the run file and
its .meta.json, and --reference compares this run with a saved one (Week 4 spec §5). The numbers
use a decimal comma (0,713), like README.md.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.rag.article_refs import article_refs
from app.rag.embedder import EmbeddingModelUnavailableError
from app.rag.fusion import RRF_K
from app.rag.retriever import (
    RETRIEVAL_MODES,
    CollectionNotIndexedError,
    InvalidCollectionDataError,
    InvalidQueryError,
    RetrievalMode,
    RetrievedChunk,
    Retriever,
    default_retriever,
)
from app.rag.store import QdrantUnavailableError, qdrant_errors, rejection_message
from eval.metrics import (
    count_above,
    format_score,
    hit_rank,
    mrr,
    pair_table,
    recall_at,
    summarize,
)
from eval.questions import (
    IN_SCOPE_MAIN,
    Group,
    Question,
    QuestionFileError,
    check_gold_ids,
    load_questions,
)
from eval.stats import clopper_pearson, mcnemar, paired_r1
from ingestion.documents import REPO_ROOT
from ingestion.index import load_chunks

EVAL_K = 10
ARTICLE_HIT_K = 5  # a question counts as found if its article is in the top 5
TOP_SHOWN_FOR_MISSES = 3
NA = "N/A"
COLLECTION_CHOICES = ("default", "hybrid")
# The fields of a result that a run file stores only when they are set (sparse and hybrid modes).
OPTIONAL_FIELDS = ("fusion_score", "dense_rank", "sparse_rank")

# The comparison with a reference run (Week 4 spec §5).
ALPHA = 0.05  # C1: McNemar mid-p at most this
GUARD_MAX_LOST = 1  # direct questions that may lose rank 1
MISSING_RANK = EVAL_K + 1  # how a rank regression counts an article outside the top 10
NEAR_TIE_GAP = 1e-4  # a fusion score this close behind the chunk above explains a regression
TOLERANCE = 1e-4  # reproduction tolerance of scores
TOLERANCE_MODES = ("dense", "hybrid")


class SupportsRetrieve(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    top: list[tuple[str, int, float]]  # (chunk_id, article, score), in rank order
    rank_article: int | None
    rank_clause: int | None
    # For each item of top, those of OPTIONAL_FIELDS that are set; empty when none was recorded.
    extras: Sequence[Mapping[str, float | int]] = ()
    dense_top1: float | None = None  # cosine of the rank-1 chunk of a plain dense query


def _first_rank(top: Sequence[tuple[str, int, float]], column: int, gold: set) -> int | None:
    """Rank of the first item whose chunk id (column 0) or article (column 1) is in gold."""
    return hit_rank(top, lambda item: item[column] in gold)


def evaluate(
    questions: Sequence[Question],
    retriever: SupportsRetrieve,
    dense_probe: SupportsRetrieve | None = None,
) -> list[QuestionResult]:
    """Retrieve once per question. Ranks are 1-based positions in the raw chunk list.

    ``dense_probe`` is a plain dense retriever on the same collection. It is needed when the first
    result of ``retriever`` may not be the dense top-1, and is then asked once per question too.
    """
    results: list[QuestionResult] = []
    for question in questions:
        retrieved = retriever.retrieve(question.question, EVAL_K)
        top = [(r.chunk.id, r.chunk.article, r.score) for r in retrieved]
        extras = [
            {name: value for name in OPTIONAL_FIELDS if (value := getattr(r, name)) is not None}
            for r in retrieved
        ]
        if dense_probe is None:
            dense = retrieved
        else:
            dense = dense_probe.retrieve(question.question, EVAL_K)
        dense_top1 = dense[0].score if dense else None
        if question.group == "out_of_scope":
            rank_article = rank_clause = None
        else:
            rank_article = _first_rank(top, 1, set(question.gold_articles))
            rank_clause = _first_rank(top, 0, set(question.gold_ids))
        results.append(QuestionResult(question, top, rank_article, rank_clause, extras, dense_top1))
    return results


def questions_digest(path: Path) -> str:
    """First 12 hex characters of the sha256 of the file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _group(results: Sequence[QuestionResult], *groups: str) -> list[QuestionResult]:
    return [r for r in results if r.question.group in groups]


def _scores(ranks: Sequence[int | None], clause_level: bool = True) -> list[str]:
    """recall@5, recall@10 and MRR@10 as text; N/A when unavailable or there are no questions."""
    if not clause_level or not ranks:
        return [NA, NA, NA]
    return [
        format_score(recall_at(ranks, 5)),
        format_score(recall_at(ranks, 10)),
        format_score(mrr(ranks)),
    ]


def _scores_with_r1(ranks: Sequence[int | None], clause_level: bool = True) -> list[str]:
    """recall@1 followed by _scores(); used by the report and the results.md row."""
    if not clause_level or not ranks:
        return [NA, NA, NA, NA]
    return [format_score(recall_at(ranks, 1)), *_scores(ranks)]


def _article_ranks(results: Sequence[QuestionResult]) -> list[int | None]:
    return [r.rank_article for r in results]


def _clause_ranks(results: Sequence[QuestionResult]) -> list[int | None]:
    return [r.rank_clause for r in results]


def _is_hit(rank: int | None) -> bool:
    return rank is not None and rank <= ARTICLE_HIT_K


def _top_scores(results: Sequence[QuestionResult]) -> list[float]:
    return [r.dense_top1 for r in results if r.dense_top1 is not None]


def _table_section(results: Sequence[QuestionResult], clause_level: bool) -> list[str]:
    rows: list[tuple[str, list[QuestionResult]]] = [
        ("số chính", _group(results, *IN_SCOPE_MAIN)),
        ("direct", _group(results, "direct")),
        ("paraphrase", _group(results, "paraphrase")),
        ("article_ref", _group(results, "article_ref")),
    ]
    header = (
        f"{'':<12}{'n':>4}  {'Điều: R@1':>10} {'R@5':>7} {'R@10':>7} {'MRR@10':>7}"
        f"  {'Khoản: R@1':>11} {'R@5':>7} {'R@10':>7} {'MRR@10':>7}"
    )
    lines = [header]
    for name, group in rows:
        article = _scores_with_r1(_article_ranks(group))
        clause = _scores_with_r1(_clause_ranks(group), clause_level)
        lines.append(
            f"{name:<12}{len(group):>4}  {article[0]:>10} {article[1]:>7} {article[2]:>7}"
            f" {article[3]:>7}  {clause[0]:>11} {clause[1]:>7} {clause[2]:>7} {clause[3]:>7}"
        )
    return lines


def _histogram_section(results: Sequence[QuestionResult]) -> list[str]:
    """How many questions found their Điều at each rank 1..EVAL_K; 'trượt' = not in the top 10."""
    header = f"{'':<12}" + "".join(f"{rank:>4}" for rank in range(1, EVAL_K + 1)) + f"{'trượt':>7}"
    lines = [header]
    for name in ("direct", "paraphrase"):
        ranks = _article_ranks(_group(results, name))
        counts = [sum(1 for r in ranks if r == rank) for rank in range(1, EVAL_K + 1)]
        misses = sum(1 for r in ranks if r is None)
        lines.append(f"{name:<12}" + "".join(f"{count:>4}" for count in counts) + f"{misses:>7}")
    return lines


def _misses_section(results: Sequence[QuestionResult]) -> list[str]:
    misses = [r for r in _group(results, *IN_SCOPE_MAIN, "article_ref") if r.rank_article != 1]
    if not misses:
        return ["(không có)"]
    lines: list[str] = []
    for r in misses:
        shown = ", ".join(
            f"{chunk_id} ({format_score(score)})"
            for chunk_id, _article, score in r.top[:TOP_SHOWN_FOR_MISSES]
        )
        articles = ", ".join(str(a) for a in r.question.gold_articles)
        lines.append(f"- {r.question.id} [{r.question.group}] {r.question.question}")
        # A sparse run without a BM25 hit has nothing to show.
        lines.append(
            f"  Điều đúng: {articles}; top {TOP_SHOWN_FOR_MISSES}: {shown or '(không có)'}"
        )
    return lines


def _summary_text(scores: Sequence[float]) -> str:
    if not scores:
        return NA
    s = summarize(scores)
    return (
        f"n={s.count}, thấp nhất {format_score(s.minimum)}, "
        f"trung vị {format_score(s.median)}, cao nhất {format_score(s.maximum)}"
    )


def _score_section(results: Sequence[QuestionResult]) -> list[str]:
    in_scope = _top_scores(_group(results, *IN_SCOPE_MAIN, "article_ref"))
    out_of_scope = _top_scores(_group(results, "out_of_scope"))
    if in_scope and out_of_scope:
        above = f"{count_above(out_of_scope, min(in_scope))}/{len(out_of_scope)}"
    else:
        above = NA
    return [
        f"Trong phạm vi: {_summary_text(in_scope)}",
        f"Ngoài phạm vi: {_summary_text(out_of_scope)}",
        f"Câu ngoài phạm vi có điểm cao hơn điểm thấp nhất trong phạm vi: {above}",
    ]


def _pair_section(results: Sequence[QuestionResult]) -> list[str]:
    hits: dict[str, list[bool]] = {}
    for r in _group(results, *IN_SCOPE_MAIN):
        pair_id = r.question.pair_id
        if pair_id is None:
            continue
        slot = hits.setdefault(pair_id, [False, False])
        slot[0 if r.question.group == "direct" else 1] = _is_hit(r.rank_article)
    table = pair_table({pair_id: (slot[0], slot[1]) for pair_id, slot in hits.items()})
    only_direct = ", ".join(table.direct_only_pairs) or "không có"
    return [
        f"cả hai trúng: {table.both}",
        f"chỉ direct trúng: {table.direct_only} (cặp: {only_direct})",
        f"chỉ paraphrase trúng: {table.paraphrase_only}",
        f"cả hai trượt: {table.neither}",
    ]


def _on_off(promotion: bool) -> str:
    return "bật" if promotion else "tắt"


def render_report(
    results: Sequence[QuestionResult],
    *,
    clause_level: bool,
    collection: str,
    model: str,
    digest: str,
    mode: RetrievalMode = "dense",
    promotion: bool = False,
) -> str:
    counts = ", ".join(
        f"{group} {len(_group(results, group))}"
        for group in ("direct", "paraphrase", "article_ref", "out_of_scope")
    )
    sections = [
        [
            "## 1. Tổng quan",
            f"Collection: {collection}",
            f"Chế độ truy xuất: {mode}",
            f"Promotion số Điều: {_on_off(promotion)}",
            f"Model: {model}",
            f"Bộ câu: {digest}",
            f"Số câu: {counts}",
        ],
        ["## 2. Bảng kết quả (hạng tính trên top 10)"] + _table_section(results, clause_level),
        ["## 3. Câu trượt ở hạng 1 (mức Điều)"] + _misses_section(results),
        ["## 4. Điểm top-1"] + _score_section(results),
        [f"## 5. Bảng cặp direct/paraphrase (trúng = Điều đúng trong top {ARTICLE_HIT_K})"]
        + _pair_section(results),
        ["## 6. Phân bố hạng của Điều đúng (số câu, mức Điều)"] + _histogram_section(results),
    ]
    return "\n\n".join("\n".join(lines) for lines in sections)


def results_row(
    results: Sequence[QuestionResult],
    *,
    clause_level: bool,
    digest: str,
    label: str,
    day: date,
) -> str:
    """One line for the table in eval/results.md."""
    main = _group(results, *IN_SCOPE_MAIN)
    article = " | ".join(_scores_with_r1(_article_ranks(main)))
    clause = " / ".join(_scores_with_r1(_clause_ranks(main), clause_level))
    rank1_by_group = []
    for name in IN_SCOPE_MAIN:
        ranks = _article_ranks(_group(results, name))
        rank1_by_group.append(f"{name} {sum(1 for r in ranks if r == 1)}/{len(ranks)}")
    refs = len(_group(results, "article_ref"))
    out_of_scope = len(_group(results, "out_of_scope"))
    return (
        f"| {day.isoformat()} | {label} | {article} | "
        f"Mức Khoản: {clause}; "
        f"R@1 theo nhóm (mức Điều): {', '.join(rank1_by_group)}; bộ câu {digest}; "
        f"n={len(main)} (+{refs} article_ref, {out_of_scope} ngoài phạm vi) |"
    )


# --- run files ----------------------------------------------------------------------


class RunEntry(BaseModel):
    """One retrieved chunk as a run file stores it."""

    id: str
    score: float  # dense cosine
    fusion_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


class RunRow(BaseModel):
    """One question as a run file stores it."""

    id: str
    group: Group
    rank_article: int | None
    rank_clause: int | None
    top: list[RunEntry]


class RunMeta(BaseModel):
    """What the comparison reads from a meta file."""

    questions_digest: str
    mode: RetrievalMode


class ReferenceRunError(ValueError):
    """The reference run cannot be read, compared with this run or kept from being overwritten."""


@dataclass(frozen=True)
class ReferenceRun:
    path: Path
    rows: dict[str, RunRow]  # by question id
    mode: RetrievalMode
    digest: str | None  # None without a meta file


def meta_path(run_path: Path) -> Path:
    """Where the meta file of a run file sits: X.jsonl has X.meta.json beside it."""
    return run_path.with_name(f"{run_path.stem}.meta.json")


def _invalid_fields(error: ValidationError) -> str:
    return ", ".join(
        sorted(
            {
                f"{'.'.join(str(part) for part in item['loc']) or 'content'} ({item['type']})"
                for item in error.errors()
            }
        )
    )


def load_reference(path: Path) -> ReferenceRun:
    """Read a run file and, when there is one, its meta file.

    A run without a meta file counts as dense: the 2026-11-22 baseline predates meta files.
    """
    rows: dict[str, RunRow] = {}
    try:
        with path.open(encoding="utf-8") as lines:
            for number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    row = RunRow.model_validate_json(line)
                except ValidationError as error:
                    raise ReferenceRunError(
                        f"{path}:{number}: not a run file line: {_invalid_fields(error)}"
                    ) from None
                if row.id in rows:
                    raise ReferenceRunError(f"{path}:{number}: duplicate question id {row.id}")
                rows[row.id] = row
    except UnicodeDecodeError:
        raise ReferenceRunError(f"{path}: not a UTF-8 file") from None
    sidecar = meta_path(path)
    if not sidecar.exists():
        return ReferenceRun(path, rows, mode="dense", digest=None)
    try:
        meta = RunMeta.model_validate_json(sidecar.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ReferenceRunError(
            f"{sidecar}: not a run meta file: {_invalid_fields(error)}"
        ) from None
    except UnicodeDecodeError:
        raise ReferenceRunError(f"{sidecar}: not a UTF-8 file") from None
    return ReferenceRun(path, rows, mode=meta.mode, digest=meta.questions_digest)


def _check_reference(reference: ReferenceRun, questions: Sequence[Question], digest: str) -> None:
    """Both runs must be about the same questions."""
    ours = {question.id: question.group for question in questions}
    theirs = {row.id: row.group for row in reference.rows.values()}
    if ours != theirs:
        differing = sorted(i for i in ours.keys() | theirs.keys() if ours.get(i) != theirs.get(i))
        raise ReferenceRunError(
            f"reference {reference.path} does not have the same question ids and groups as this "
            f"run: {', '.join(differing)}"
        )
    if reference.digest not in (None, digest):
        raise ReferenceRunError(
            f"reference {reference.path} was run on question set {reference.digest}, "
            f"this run uses {digest}"
        )


def _same_file(first: Path, second: Path) -> bool:
    if first.exists() and second.exists():
        return first.samefile(second)
    return os.path.normcase(first.resolve()) == os.path.normcase(second.resolve())


def _check_out_spares_reference(out: Path, reference: Path) -> None:
    written = (out, meta_path(out))
    kept = (reference, meta_path(reference))
    if any(_same_file(new, old) for new in written for old in kept):
        raise ReferenceRunError(
            f"--out {out} would overwrite the reference {reference} or its meta file; "
            "choose another name"
        )


def _run_line(result: QuestionResult) -> dict:
    extras = result.extras or [{}] * len(result.top)
    return {
        "id": result.question.id,
        "group": result.question.group,
        "pair_id": result.question.pair_id,
        "rank_article": result.rank_article,
        "rank_clause": result.rank_clause,
        "top": [
            {"id": chunk_id, "score": score, **extra}
            for (chunk_id, _article, score), extra in zip(result.top, extras, strict=True)
        ],
    }


def _write_files(files: Sequence[tuple[Path, str]]) -> None:
    """Write via .tmp files so a failure never leaves a half-written output.

    The files are moved into place once all of them are written. When one cannot be moved, those
    already moved are removed, so a run file never stays without its meta file.
    """
    pending = [(path, path.with_name(path.name + ".tmp"), text) for path, text in files]
    moved: list[Path] = []
    try:
        for _path, tmp, text in pending:
            with tmp.open("w", encoding="utf-8", newline="\n") as file:
                file.write(text)
        for path, tmp, _text in pending:
            os.replace(tmp, path)
            moved.append(path)
    except BaseException:
        for leftover in [tmp for _path, tmp, _text in pending] + moved:
            leftover.unlink(missing_ok=True)
        raise


# --- comparison with a reference run (Week 4 spec §5) --------------------------------
# A is the reference and B the new run; both map a question id to its RunRow.


def _rank_text(rank: int | None) -> str:
    return f"ngoài top {EVAL_K}" if rank is None else str(rank)


def _yes_no(value: bool) -> str:
    return "có" if value else "không"


def _decimal(value: float, places: int) -> str:
    return f"{value:.{places}f}".replace(".", ",")


@dataclass(frozen=True)
class Guard:
    """The guard for the direct questions (§5.3)."""

    size: int  # |D|: direct questions with their article at rank 1 in A
    lost: list[str]  # those of D not at rank 1 in B
    upper: float | None  # upper 95 % Clopper-Pearson bound of lost / |D|; None when D is empty
    left_top5: list[str]  # direct questions in the top 5 in A but not in B
    holds: bool


def guard_check(
    questions: Sequence[Question], run_a: Mapping[str, RunRow], run_b: Mapping[str, RunRow]
) -> Guard:
    direct = [question.id for question in questions if question.group == "direct"]
    first_in_a = [i for i in direct if run_a[i].rank_article == 1]
    lost = [i for i in first_in_a if run_b[i].rank_article != 1]
    left = [
        i for i in direct if _is_hit(run_a[i].rank_article) and not _is_hit(run_b[i].rank_article)
    ]
    upper = clopper_pearson(len(lost), len(first_in_a), 0.95)[1] if first_in_a else None
    holds = len(lost) <= GUARD_MAX_LOST and not left
    return Guard(len(first_in_a), lost, upper, left, holds)


@dataclass(frozen=True)
class Regression:
    question_id: str
    group: str
    rank_a: int | None
    rank_b: int | None
    cause: str | None  # None for a sparse run B
    detail: str = ""


def _article_set(articles: set[int]) -> str:
    return "{" + ", ".join(str(a) for a in sorted(articles)) + "}" if articles else "∅"


def _near_tie_gap(row: RunRow) -> float | None:
    """How far the first gold chunk is behind the chunk directly above it, if that is a near tie."""
    if row.rank_article is None or row.rank_article < 2:
        return None
    above, gold = row.top[row.rank_article - 2], row.top[row.rank_article - 1]
    if above.fusion_score is None or gold.fusion_score is None:
        return None
    gap = above.fusion_score - gold.fusion_score
    # A negative gap is a promoted chunk with a lower score: the gold chunk is not behind it.
    return gap if 0 <= gap < NEAR_TIE_GAP else None


def _cause(
    question: Question, row_b: RunRow, mode_b: RetrievalMode, promotion_b: bool
) -> tuple[str | None, str]:
    """The first cause of §5.9 that fits, judged by the flags of run B, and its detail."""
    if mode_b == "sparse":
        return None, ""
    if promotion_b:
        found, gold = article_refs(question.question), set(question.gold_articles)
        if found != gold:
            return "parser", f"article_refs = {_article_set(found)}, gold = {_article_set(gold)}"
    if mode_b == "hybrid":
        gap = _near_tie_gap(row_b)
        if gap is not None:
            return "gần hòa điểm", f"gap {gap:.2e}"
        return "chunk khác trùng từ khóa", ""
    return "khác", ""


def rank_regressions(
    questions: Sequence[Question],
    run_a: Mapping[str, RunRow],
    run_b: Mapping[str, RunRow],
    *,
    mode_b: RetrievalMode,
    promotion_b: bool,
) -> list[Regression]:
    """Questions whose article ranks lower in B than in A (§5.9)."""
    found: list[Regression] = []
    for question in questions:
        rank_a = run_a[question.id].rank_article
        rank_b = run_b[question.id].rank_article
        if (rank_b or MISSING_RANK) <= (rank_a or MISSING_RANK):
            continue
        cause, detail = _cause(question, run_b[question.id], mode_b, promotion_b)
        found.append(Regression(question.id, question.group, rank_a, rank_b, cause, detail))
    return found


@dataclass(frozen=True)
class TopCheck:
    """Conditions (b) and (c) of the reproduction tolerance (§5.7) over some questions."""

    questions: int
    identical: int  # questions with the same top-10 ids in the same order
    swapped: int  # questions whose top 10 differ in tolerated ways only
    violated: int  # questions with at least one difference that is not tolerated
    swaps: list[str]
    violations: list[str]
    scores_checked: int  # chunks present in both top 10
    score_violations: list[str]


@dataclass(frozen=True)
class Tolerance:
    in_scope: int
    rank_violations: list[str]  # condition (a): in-scope questions whose ranks differ
    everything: TopCheck  # all questions
    without_refs: TopCheck  # condition (d): the questions that are not article_ref


def _ranking_score(entry: RunEntry, mode: RetrievalMode) -> float:
    """The score that orders a run: the fusion score in hybrid mode, else the cosine."""
    if mode == "hybrid" and entry.fusion_score is not None:
        return entry.fusion_score
    return entry.score  # dense rows, and the dense fallback of hybrid mode


def _order_differences(
    row_a: RunRow, row_b: RunRow, mode: RetrievalMode
) -> tuple[list[str], list[str]]:
    """Compare two top 10 and return (tolerated swaps, violations), condition (b) of §5.7."""
    top_a, top_b = row_a.top[:EVAL_K], row_b.top[:EVAL_K]
    score_a = {entry.id: _ranking_score(entry, mode) for entry in top_a}
    score_b = {entry.id: _ranking_score(entry, mode) for entry in top_b}
    place_b = {entry.id: place for place, entry in enumerate(top_b)}
    swaps: list[str] = []
    violations: list[str] = []
    # Chunks of both lists that the two runs order differently: a near tie in each run.
    shared = [entry.id for entry in top_a if entry.id in place_b]  # in the order of A
    for first, second in combinations(shared, 2):
        if place_b[first] < place_b[second]:
            continue
        gap_a = abs(score_a[first] - score_a[second])
        gap_b = abs(score_b[first] - score_b[second])
        line = f"{row_a.id}: đổi chỗ {first} <-> {second} (gap A {gap_a:.2e}, B {gap_b:.2e})"
        (swaps if gap_a < TOLERANCE and gap_b < TOLERANCE else violations).append(line)
    # Chunks of one list only: near the 10th score of the other run, so at the cut edge.
    for name, own, other_name, other in (("A", top_a, "B", top_b), ("B", top_b, "A", top_a)):
        other_ids = {entry.id for entry in other}
        for entry in own:
            if entry.id in other_ids:
                continue
            if len(other) < EVAL_K:
                violations.append(
                    f"{row_a.id}: {entry.id} chỉ có ở {name}, {other_name} chỉ có {len(other)} "
                    f"chunk nên không có biên top {EVAL_K}"
                )
                continue
            gap = abs(_ranking_score(entry, mode) - _ranking_score(other[-1], mode))
            line = (
                f"{row_a.id}: biên top {EVAL_K}, {entry.id} chỉ có ở {name} "
                f"(gap {gap:.2e} tới hạng {EVAL_K} của {other_name})"
            )
            (swaps if gap < TOLERANCE else violations).append(line)
    return swaps, violations


def _score_differences(row_a: RunRow, row_b: RunRow) -> tuple[int, list[str]]:
    """(chunks in both top 10, scores that moved too far), condition (c) of §5.7."""
    in_b = {entry.id: entry for entry in row_b.top[:EVAL_K]}
    checked = 0
    violations: list[str] = []
    for entry in row_a.top[:EVAL_K]:
        other = in_b.get(entry.id)
        if other is None:
            continue
        checked += 1
        for name in ("score", "fusion_score"):
            before, after = getattr(entry, name), getattr(other, name)
            if before is None or after is None:
                continue
            delta = abs(after - before)
            if delta >= TOLERANCE:
                violations.append(f"{row_a.id}: {entry.id} |Δ {name}| = {delta:.2e}")
    return checked, violations


def _top_check(
    ids: Sequence[str],
    run_a: Mapping[str, RunRow],
    run_b: Mapping[str, RunRow],
    mode: RetrievalMode,
) -> TopCheck:
    identical = swapped = violated = checked = 0
    swaps: list[str] = []
    violations: list[str] = []
    score_violations: list[str] = []
    for i in ids:
        row_a, row_b = run_a[i], run_b[i]
        count, moved = _score_differences(row_a, row_b)
        checked += count
        score_violations += moved
        if [e.id for e in row_a.top[:EVAL_K]] == [e.id for e in row_b.top[:EVAL_K]]:
            identical += 1
            continue
        tolerated, broken = _order_differences(row_a, row_b, mode)
        swaps += tolerated
        violations += broken
        if broken:
            violated += 1
        else:
            swapped += 1
    return TopCheck(
        len(ids), identical, swapped, violated, swaps, violations, checked, score_violations
    )


def tolerance_check(
    questions: Sequence[Question],
    run_a: Mapping[str, RunRow],
    run_b: Mapping[str, RunRow],
    mode: RetrievalMode,
) -> Tolerance:
    """The reproduction tolerance (§5.7) between two runs of the same dense or hybrid mode."""
    in_scope = [question.id for question in questions if question.group != "out_of_scope"]
    rank_violations = []
    for i in in_scope:
        a, b = run_a[i], run_b[i]
        if (a.rank_article, a.rank_clause) != (b.rank_article, b.rank_clause):
            rank_violations.append(
                f"{i}: rank_article {_rank_text(a.rank_article)} → {_rank_text(b.rank_article)}, "
                f"rank_clause {_rank_text(a.rank_clause)} → {_rank_text(b.rank_clause)}"
            )
    everything = [question.id for question in questions]
    without_refs = [question.id for question in questions if question.group != "article_ref"]
    return Tolerance(
        len(in_scope),
        rank_violations,
        _top_check(everything, run_a, run_b, mode),
        _top_check(without_refs, run_a, run_b, mode),
    )


def _r1_lines(
    questions: Sequence[Question],
    run_a: Mapping[str, RunRow],
    run_b: Mapping[str, RunRow],
    clause_level: bool,
) -> list[str]:
    lines = ["### 7.1 R@1 theo nhóm (mức Điều)"]
    for group in (*IN_SCOPE_MAIN, "article_ref"):
        ids = [question.id for question in questions if question.group == group]
        hits = [sum(run[i].rank_article == 1 for i in ids) for run in (run_a, run_b)]
        lines.append(f"{group}: A {hits[0]}/{len(ids)}, B {hits[1]}/{len(ids)}")
        if group != "article_ref":
            continue
        for i in ids:
            a, b = run_a[i], run_b[i]
            clause = (
                f"A {_rank_text(a.rank_clause)}, B {_rank_text(b.rank_clause)}"
                if clause_level
                else NA
            )
            lines.append(
                f"  {i}: hạng Điều A {_rank_text(a.rank_article)}, "
                f"B {_rank_text(b.rank_article)}; hạng Khoản {clause}"
            )
    return lines


def _mcnemar_lines(
    questions: Sequence[Question], run_a: Mapping[str, RunRow], run_b: Mapping[str, RunRow]
) -> list[str]:
    """The inputs of the decision rule (§5.2, §5.4); the decision itself is not made here."""
    lines = ["### 7.2 McNemar trên câu paraphrase (mức Điều, hạng 1)"]
    if not any(question.group == "paraphrase" for question in questions):
        return lines + ["Không có câu paraphrase."]
    pairs_a, pairs_b = (
        {q.id: (q.group, run[q.id].rank_article) for q in questions} for run in (run_a, run_b)
    )
    paired = paired_r1(pairs_a, pairs_b, "paraphrase")
    test = mcnemar(paired.gains, paired.losses)
    c1 = test.mid_p <= ALPHA and test.b > test.c
    alternative = test.b - test.c >= 3 and test.c <= 1
    return lines + [
        f"n = {paired.n}; b = {test.b} (B đúng hạng 1, A sai); c = {test.c} (A đúng hạng 1, B sai)",
        f"p chính xác = {_decimal(test.exact_p, 6)}; mid-p = {_decimal(test.mid_p, 6)}",
        f"C1 (mid-p ≤ {_decimal(ALPHA, 2)} và b > c): {_yes_no(c1)}",
        f"b − c ≥ 3 và c ≤ 1: {_yes_no(alternative)}",
    ]


def _guard_lines(
    questions: Sequence[Question], run_a: Mapping[str, RunRow], run_b: Mapping[str, RunRow]
) -> list[str]:
    guard = guard_check(questions, run_a, run_b)

    def moves(ids: Sequence[str]) -> str:
        return (
            ", ".join(
                f"{i} ({_rank_text(run_a[i].rank_article)} → {_rank_text(run_b[i].rank_article)})"
                for i in ids
            )
            or "(không có)"
        )

    upper = NA if guard.upper is None else _decimal(guard.upper, 4)
    return [
        "### 7.3 Guard cho câu direct (mức Điều)",
        f"|D| = {guard.size} (câu direct có Điều đúng ở hạng 1 trong A)",
        f"Mất hạng 1 ở B: {len(guard.lost)}/{guard.size}: {moves(guard.lost)}",
        f"Cận trên Clopper–Pearson hai phía 95 % của tỷ lệ mất: {upper}",
        (
            f"Rời top {ARTICLE_HIT_K} (direct, A trong top {ARTICLE_HIT_K}, B ngoài): "
            f"{len(guard.left_top5)}: {moves(guard.left_top5)}"
        ),
        (
            f"Guard đạt (mất tối đa {GUARD_MAX_LOST} và không câu nào rời top {ARTICLE_HIT_K}): "
            f"{_yes_no(guard.holds)}"
        ),
    ]


def _regression_lines(regressions: Sequence[Regression]) -> list[str]:
    lines = [
        f"### 7.4 Câu tụt hạng (mức Điều, ngoài top {EVAL_K} tính là {MISSING_RANK})",
        f"Số câu tụt hạng: {len(regressions)}",
    ]
    for r in regressions:
        line = f"- {r.question_id} [{r.group}] {_rank_text(r.rank_a)} → {_rank_text(r.rank_b)}"
        if r.cause is not None:
            line += f"; nguyên nhân: {r.cause}" + (f" ({r.detail})" if r.detail else "")
        lines.append(line)
    return lines


def _listed(lines: Sequence[str]) -> list[str]:
    return [f"  - {line}" for line in lines]


def _top_check_lines(check: TopCheck) -> list[str]:
    return [
        (
            f"(b) top {EVAL_K} ({check.questions} câu): giống hệt {check.identical}, "
            f"chỉ đổi chỗ trong dung sai {check.swapped}, có vi phạm {check.violated}; "
            f"đạt: {_yes_no(not check.violations)}"
        ),
        f"  Đổi chỗ trong dung sai: {len(check.swaps)}",
        *_listed(check.swaps),
        f"  Vi phạm: {len(check.violations)}",
        *_listed(check.violations),
        (
            f"(c) |Δ điểm| < 1e-4: {check.scores_checked} chunk chung, "
            f"vi phạm {len(check.score_violations)}; đạt: {_yes_no(not check.score_violations)}"
        ),
        *_listed(check.score_violations),
    ]


def _tolerance_lines(
    questions: Sequence[Question],
    reference: ReferenceRun,
    run_b: Mapping[str, RunRow],
    mode_b: RetrievalMode,
) -> list[str]:
    lines = ["### 7.5 Dung sai tái lập"]
    if reference.mode != mode_b or mode_b not in TOLERANCE_MODES:
        skipped = (
            f"Bỏ qua kiểm tra dung sai: A là {reference.mode}, B là {mode_b} "
            "(chỉ kiểm khi hai run cùng mode dense hoặc cùng hybrid)."
        )
        return lines + [skipped]
    check = tolerance_check(questions, reference.rows, run_b, mode_b)
    same = check.in_scope - len(check.rank_violations)
    return lines + [
        (
            f"(a) rank_article và rank_clause giống hệt: {same}/{check.in_scope} câu trong "
            f"phạm vi; đạt: {_yes_no(not check.rank_violations)}"
        ),
        *_listed(check.rank_violations),
        *_top_check_lines(check.everything),
        f"(d) (b) và (c) trên {check.without_refs.questions} câu không phải article_ref:",
        *_top_check_lines(check.without_refs),
    ]


def render_comparison(
    questions: Sequence[Question],
    reference: ReferenceRun,
    run_b: Mapping[str, RunRow],
    *,
    mode_b: RetrievalMode,
    promotion_b: bool,
    clause_level: bool,
) -> str:
    """Section 7: run B (this run) against the reference run A, at article level."""
    run_a = reference.rows
    without_meta = "" if reference.digest is not None else " (không có file meta)"
    sections = [
        [
            "## 7. So với run tham chiếu",
            f"A (tham chiếu): {reference.path}; mode {reference.mode}{without_meta}",
            f"B (run này): mode {mode_b}, promotion {_on_off(promotion_b)}",
        ],
        _r1_lines(questions, run_a, run_b, clause_level),
        _mcnemar_lines(questions, run_a, run_b),
        _guard_lines(questions, run_a, run_b),
        _regression_lines(
            rank_regressions(questions, run_a, run_b, mode_b=mode_b, promotion_b=promotion_b)
        ),
        _tolerance_lines(questions, reference, run_b, mode_b),
    ]
    return "\n\n".join("\n".join(lines) for lines in sections)


# --- command line --------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    flagged: bool  # False without a retrieval flag: default_retriever() is measured as it is
    mode: RetrievalMode
    hybrid_collection: bool
    promotion: bool

    @property
    def label(self) -> str:
        """The default text of the results.md row."""
        if self.mode == "dense" and not self.hybrid_collection and not self.promotion:
            return "dense baseline"
        label = self.mode + (" + promotion" if self.promotion else "")
        # Sparse and hybrid imply the hybrid collection; a dense run on it has to say so.
        if self.mode == "dense" and self.hybrid_collection:
            label += ", hybrid collection"
        return label


def _run_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> RunConfig:
    """Resolve the retrieval flags; parser.error() exits on a combination that cannot work."""
    mode = args.retriever or "dense"
    collection = args.collection or ("default" if mode == "dense" else "hybrid")
    if mode != "dense" and collection == "default":
        parser.error(
            f"--retriever {mode} needs --collection hybrid: "
            "the default collection has no BM25 vectors"
        )
    if args.promote and mode == "sparse":
        parser.error("--promote has no effect with --retriever sparse; use dense or hybrid")
    flagged = args.retriever is not None or args.collection is not None or args.promote
    return RunConfig(flagged, mode, collection == "hybrid", args.promote)


def _retrievers(
    config: RunConfig, collection: str
) -> tuple[Retriever, Retriever, Retriever | None]:
    """(the default retriever, the one to measure, the dense probe if section 4 needs one).

    All of them share the client and the embedder of the default retriever, so the model loads once.
    """
    base = default_retriever()
    if not config.flagged:
        return base, base, None
    measured = Retriever(
        base.client, base.embedder, collection, config.mode, promote=config.promotion
    )
    probe = None
    if config.mode != "dense" or config.promotion:
        probe = Retriever(base.client, base.embedder, collection)
    return base, measured, probe


def _git_sha() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        print("warning: cannot read the git commit; git_sha is null", file=sys.stderr)
        return None
    return done.stdout.strip()


def _run_meta(config: RunConfig, collection: str, base: Retriever, digest: str, day: date) -> dict:
    """The meta file of a run: these keys only, and never a dump of the settings."""
    metadata = None  # the dense-only collection stores none
    if config.hybrid_collection:
        with qdrant_errors():
            metadata = base.client.get_collection(collection).config.metadata or None
    return {
        "questions_digest": digest,
        "collection": collection,
        "collection_metadata": metadata,
        "mode": config.mode,
        "k": EVAL_K,
        "rrf_k": RRF_K if config.mode == "hybrid" else None,
        "promotion": config.promotion,
        # Where the model ran, which the settings do not say: they may leave the choice open.
        "device": getattr(getattr(base, "embedder", None), "device", None),
        "git_sha": _git_sha(),
        "date": day.isoformat(),
    }


def _run(args: argparse.Namespace, config: RunConfig) -> int:
    questions = load_questions(args.questions)
    if not questions:
        print(f"no questions in {args.questions}", file=sys.stderr)
        return 1
    chunks = load_chunks(args.chunks)
    missing = check_gold_ids(questions, chunks)
    clause_level = not missing
    if missing:
        print(
            f"warning: {len(missing)} gold ids are not in {args.chunks}; "
            "clause-level metrics are N/A",
            file=sys.stderr,
        )
    digest = questions_digest(args.questions)
    # The reference is read before any retrieval and before --out is written.
    reference = None
    if args.reference is not None:
        if args.out is not None:
            _check_out_spares_reference(args.out, args.reference)
        reference = load_reference(args.reference)
        _check_reference(reference, questions, digest)
    settings = get_settings()
    collection = (
        settings.hybrid_collection_name if config.hybrid_collection else settings.collection_name
    )
    base, retriever, probe = _retrievers(config, collection)
    results = evaluate(questions, retriever, probe)
    day = date.today()
    run_lines = [_run_line(result) for result in results]
    output = [
        render_report(
            results,
            clause_level=clause_level,
            collection=collection,
            model=settings.embedding_model,
            digest=digest,
            mode=config.mode,
            promotion=config.promotion,
        )
    ]
    if reference is not None:
        output.append(
            render_comparison(
                questions,
                reference,
                {line["id"]: RunRow.model_validate(line) for line in run_lines},
                mode_b=config.mode,
                promotion_b=config.promotion,
                clause_level=clause_level,
            )
        )
    label = config.label if args.label is None else args.label
    output.append(
        results_row(results, clause_level=clause_level, digest=digest, label=label, day=day)
    )
    # Render first, so a failure while rendering leaves no run or meta file.
    if args.out is not None:
        meta = _run_meta(config, collection, base, digest, day)
        _write_files(
            [
                (
                    args.out,
                    "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in run_lines),
                ),
                (meta_path(args.out), json.dumps(meta, ensure_ascii=False, indent=2) + "\n"),
            ]
        )
    print("\n\n".join(output))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not use UTF-8; Vietnamese output must not crash.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the question set.")
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "eval" / "questions.jsonl")
    parser.add_argument(
        "--chunks", type=Path, default=REPO_ROOT / "data" / "processed" / "chunks.jsonl"
    )
    parser.add_argument("--retriever", choices=RETRIEVAL_MODES, help="retrieval mode")
    parser.add_argument(
        "--collection",
        choices=COLLECTION_CHOICES,
        help="collection to search; sparse and hybrid use the hybrid one, dense the default one",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="move chunks of the articles a question names to the top (dense or hybrid)",
    )
    parser.add_argument(
        "--out", type=Path, help="write per-question results (JSONL) and a .meta.json beside it"
    )
    parser.add_argument("--reference", type=Path, help="run file to compare this run with")
    parser.add_argument("--label", help="text for the results.md row (default: from the flags)")
    args = parser.parse_args(argv)
    config = _run_config(args, parser)

    try:
        return _run(args, config)
    except (
        QuestionFileError,
        ReferenceRunError,
        QdrantUnavailableError,
        CollectionNotIndexedError,
        InvalidCollectionDataError,
        InvalidQueryError,
        EmbeddingModelUnavailableError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    except UnexpectedResponse as error:
        print(rejection_message(error, searching=True), file=sys.stderr)
        return 1
    except ValidationError as error:
        print(f"invalid settings or chunks file: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"cannot read or write a file: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
