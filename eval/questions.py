"""Schema and validation for eval/questions.jsonl (one JSON object per line)."""

import json
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ingestion.models import Chunk

Group = Literal["direct", "paraphrase", "article_ref", "out_of_scope"]
IN_SCOPE_MAIN: tuple[Group, ...] = ("direct", "paraphrase")

MIN_ARTICLE = 1
MAX_ARTICLE = 220  # the corpus has 220 articles


class Question(BaseModel):
    """One eval question. Checks the format of each field; cross-field rules live in load_questions."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str = Field(pattern=r"^q[0-9]{3}$")
    group: Group
    pair_id: str | None = Field(pattern=r"^p[0-9]{2}$")
    question: str
    gold_ids: list[str]
    gold_articles: list[int]
    note: str = ""


class QuestionFileError(ValueError):
    """The question file breaks a rule; the message is "{path}:{line}: {reason}"."""


def _fail(where: object, reason: str) -> QuestionFileError:
    return QuestionFileError(f"{where}: {reason}")


def _parse_line(raw: str, where: str) -> Question:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail(where, f"invalid JSON ({exc.msg} at column {exc.colno})") from None
    try:
        return Question.model_validate(data)
    except ValidationError as exc:
        # Name the failing fields only; never echo the line content.
        problems = sorted(
            {
                f"{'.'.join(str(part) for part in err['loc']) or 'line'} ({err['type']})"
                for err in exc.errors()
            }
        )
        raise _fail(where, "invalid fields: " + ", ".join(problems)) from None


def _check_line_rules(q: Question, where: str) -> None:
    if not unicodedata.is_normalized("NFC", q.question):
        raise _fail(where, "question is not NFC-normalized")
    if not unicodedata.is_normalized("NFC", q.note):
        raise _fail(where, "note is not NFC-normalized")
    if not q.question.strip():
        raise _fail(where, "question is empty")
    if len(set(q.gold_ids)) != len(q.gold_ids):
        raise _fail(where, "gold_ids contains duplicates")
    if any(not MIN_ARTICLE <= a <= MAX_ARTICLE for a in q.gold_articles):
        raise _fail(where, f"gold_articles must be within {MIN_ARTICLE}-{MAX_ARTICLE}")
    if any(a >= b for a, b in zip(q.gold_articles, q.gold_articles[1:])):
        raise _fail(where, "gold_articles must be strictly increasing")
    if bool(q.gold_articles) != bool(q.gold_ids):
        raise _fail(where, "gold_articles must be empty if and only if gold_ids is empty")

    if q.group == "out_of_scope":
        if q.gold_ids or q.pair_id is not None:
            raise _fail(where, "out_of_scope needs empty gold_ids and pair_id null")
    elif q.group == "article_ref":
        if not q.gold_ids or q.pair_id is not None:
            raise _fail(where, "article_ref needs gold_ids and pair_id null")
    elif not q.gold_ids or q.pair_id is None:
        raise _fail(where, f"{q.group} needs gold_ids and a pair_id")


def _check_pair(pair_id: str, members: list[tuple[int, Question]], path: Path) -> None:
    where = f"{path}:{members[0][0]}"
    groups = sorted(q.group for _, q in members)
    if groups != ["direct", "paraphrase"]:
        raise _fail(where, f"pair {pair_id} needs exactly one direct and one paraphrase")
    first, second = members[0][1], members[1][1]
    if set(first.gold_ids) != set(second.gold_ids):
        raise _fail(where, f"pair {pair_id} has different gold_ids")


def load_questions(path: Path) -> list[Question]:
    """Read and validate the question file. Accepts a BOM, CRLF and blank lines."""
    questions: list[Question] = []
    seen_ids: set[str] = set()
    pairs: dict[str, list[tuple[int, Question]]] = {}

    with path.open(encoding="utf-8-sig") as lines:
        for line_no, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            where = f"{path}:{line_no}"
            question = _parse_line(raw, where)
            if question.id in seen_ids:
                raise _fail(where, f"duplicate id {question.id}")
            seen_ids.add(question.id)
            _check_line_rules(question, where)
            questions.append(question)
            if question.pair_id is not None:
                pairs.setdefault(question.pair_id, []).append((line_no, question))

    for pair_id, members in pairs.items():
        _check_pair(pair_id, members, path)
    return questions


def _corpus_order(chunks: Sequence[Chunk]) -> dict[str, int]:
    return {chunk.id: position for position, chunk in enumerate(chunks)}


def gold_articles_for(gold_ids: Sequence[str], chunks: Sequence[Chunk]) -> list[int]:
    """Sorted, unique article numbers of the given chunk ids. Raises KeyError for an unknown id."""
    article_of = {chunk.id: chunk.article for chunk in chunks}
    return sorted({article_of[gold_id] for gold_id in gold_ids})


def order_gold_ids(gold_ids: Iterable[str], chunks: Sequence[Chunk]) -> list[str]:
    """Drop duplicates and sort by corpus order. Raises KeyError for an unknown id."""
    position = _corpus_order(chunks)
    return sorted(set(gold_ids), key=lambda gold_id: position[gold_id])


def check_gold_ids(questions: Sequence[Question], chunks: Sequence[Chunk]) -> list[str]:
    """Return the sorted, unique gold ids that are not in chunks.

    A question whose gold ids are all present must list them in corpus order and carry
    the matching gold_articles; otherwise QuestionFileError names the question id.
    """
    position = _corpus_order(chunks)
    missing: set[str] = set()
    for q in questions:
        absent = [gold_id for gold_id in q.gold_ids if gold_id not in position]
        missing.update(absent)
        if absent:
            continue
        if q.gold_ids != order_gold_ids(q.gold_ids, chunks):
            raise _fail(q.id, "gold_ids are not in corpus order")
        if q.gold_articles != gold_articles_for(q.gold_ids, chunks):
            raise _fail(q.id, "gold_articles do not match gold_ids")
    return sorted(missing)


def question_to_json(q: Question) -> str:
    """One JSON line, Vietnamese kept as is (no \\u escapes)."""
    return json.dumps(q.model_dump(), ensure_ascii=False)
