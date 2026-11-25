"""Retrieval evaluation: uv run python -m eval.run_eval [--questions PATH] [--out PATH] ...

Runs every question through the dense retriever once (k = 10), then prints a report and one
line for eval/results.md. The numbers use a decimal comma (0,713), like README.md.
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.rag.embedder import EmbeddingModelUnavailableError
from app.rag.retriever import (
    CollectionNotIndexedError,
    InvalidCollectionDataError,
    InvalidQueryError,
    RetrievedChunk,
    default_retriever,
)
from app.rag.store import QdrantUnavailableError, rejection_message
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
    Question,
    QuestionFileError,
    check_gold_ids,
    load_questions,
)
from ingestion.documents import REPO_ROOT
from ingestion.index import load_chunks

EVAL_K = 10
ARTICLE_HIT_K = 5  # the headline number: a question counts as found if its article is in the top 5
TOP_SHOWN_FOR_MISSES = 3
NA = "N/A"


class SupportsRetrieve(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    top: list[tuple[str, int, float]]  # (chunk_id, article, score), in rank order
    rank_article: int | None
    rank_clause: int | None


def _first_rank(top: Sequence[tuple[str, int, float]], column: int, gold: set) -> int | None:
    """Rank of the first item whose chunk id (column 0) or article (column 1) is in gold."""
    return hit_rank(top, lambda item: item[column] in gold)


def evaluate(questions: Sequence[Question], retriever: SupportsRetrieve) -> list[QuestionResult]:
    """Retrieve once per question. Ranks are 1-based positions in the raw chunk list."""
    results: list[QuestionResult] = []
    for question in questions:
        retrieved = retriever.retrieve(question.question, EVAL_K)
        top = [(r.chunk.id, r.chunk.article, r.score) for r in retrieved]
        if question.group == "out_of_scope":
            rank_article = rank_clause = None
        else:
            rank_article = _first_rank(top, 1, set(question.gold_articles))
            rank_clause = _first_rank(top, 0, set(question.gold_ids))
        results.append(QuestionResult(question, top, rank_article, rank_clause))
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


def _top_score(result: QuestionResult) -> float | None:
    return result.top[0][2] if result.top else None


def _top_scores(results: Sequence[QuestionResult]) -> list[float]:
    return [score for r in results if (score := _top_score(r)) is not None]


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
    misses = [
        r for r in _group(results, *IN_SCOPE_MAIN, "article_ref") if not _is_hit(r.rank_article)
    ]
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
        lines.append(f"  Điều đúng: {articles}; top {TOP_SHOWN_FOR_MISSES}: {shown}")
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


def render_report(
    results: Sequence[QuestionResult],
    *,
    clause_level: bool,
    collection: str,
    model: str,
    digest: str,
) -> str:
    counts = ", ".join(
        f"{group} {len(_group(results, group))}"
        for group in ("direct", "paraphrase", "article_ref", "out_of_scope")
    )
    sections = [
        ["## 1. Tổng quan", f"Collection: {collection}", f"Model: {model}", f"Bộ câu: {digest}"]
        + [f"Số câu: {counts}"],
        ["## 2. Bảng kết quả (hạng tính trên top 10)"] + _table_section(results, clause_level),
        [f"## 3. Câu trượt ở top {ARTICLE_HIT_K} (mức Điều)"] + _misses_section(results),
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


def _write_out(path: Path, results: Sequence[QuestionResult]) -> None:
    """Write via a .tmp file so a failure never leaves a half-written output."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as file:
            for r in results:
                row = {
                    "id": r.question.id,
                    "group": r.question.group,
                    "pair_id": r.question.pair_id,
                    "rank_article": r.rank_article,
                    "rank_clause": r.rank_clause,
                    "top": [{"id": chunk_id, "score": score} for chunk_id, _a, score in r.top],
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _run(args: argparse.Namespace) -> int:
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
    settings = get_settings()
    results = evaluate(questions, default_retriever())
    digest = questions_digest(args.questions)
    if args.out is not None:
        _write_out(args.out, results)
    print(
        render_report(
            results,
            clause_level=clause_level,
            collection=settings.collection_name,
            model=settings.embedding_model,
            digest=digest,
        )
    )
    print()
    print(
        results_row(
            results, clause_level=clause_level, digest=digest, label=args.label, day=date.today()
        )
    )
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
    parser.add_argument("--out", type=Path, default=None, help="write per-question results (JSONL)")
    parser.add_argument("--label", default="dense baseline", help="text for the results.md row")
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except (
        QuestionFileError,
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
