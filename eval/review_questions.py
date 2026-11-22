"""Console wizard that reviews the draft evaluation questions one unit at a time.

A unit is a direct/paraphrase pair or a single question. Accepted questions are appended to
eval/questions.jsonl and rejected ones (with a reason) to eval/questions.rejected.jsonl, so a
run can be stopped at any time and resumed later. No retrieval code is used here.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from eval.questions import (
    Group,
    Question,
    QuestionFileError,
    check_gold_ids,
    gold_articles_for,
    load_questions,
    order_gold_ids,
    question_to_json,
)
from ingestion.documents import REPO_ROOT
from ingestion.index import load_chunks
from ingestion.models import Chunk

ID_PREFIX = "45-2019-QH14:"
TARGETS: dict[Group, int] = {"direct": 40, "paraphrase": 40, "article_ref": 10, "out_of_scope": 20}
OUT_OF_SCOPE: Group = "out_of_scope"
DRAFT_NAME = "eval/questions.draft.jsonl"
EDIT_PROMPT = "gold ids (empty line cancels) > "
PROMPT = "[a]ccept [e]dit ids [c]ontext [r]eject [q]uit > "


@dataclass(frozen=True)
class ReviewUnit:
    key: str  # pair_id, or the question id when the question is not part of a pair
    questions: tuple[Question, ...]  # one question, or direct then paraphrase


def build_units(drafts: Sequence[Question]) -> list[ReviewUnit]:
    """Group the drafts into review units, keeping the order in which each unit first appears."""
    grouped: dict[str, list[Question]] = {}
    for question in drafts:
        grouped.setdefault(question.pair_id or question.id, []).append(question)
    return [
        ReviewUnit(key, tuple(sorted(members, key=lambda q: q.group != "direct")))
        for key, members in grouped.items()
    ]


def parse_gold_ids(text: str, chunks: Sequence[Chunk]) -> list[str]:
    """Parse ids typed by the user: separated by commas and/or spaces, prefix optional."""
    tokens = [token for token in re.split(r"[,\s]+", text) if token]
    if not tokens:
        raise ValueError("no chunk id given")
    known = {chunk.id for chunk in chunks}
    full_ids = [t if t.startswith(ID_PREFIX) else ID_PREFIX + t for t in tokens]
    for full_id in full_ids:
        if full_id not in known:
            raise ValueError(f"unknown chunk id: {full_id}")
    return order_gold_ids(full_ids, chunks)


def _read_rejected_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8-sig") as lines:
        for line_no, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            try:
                ids.append(json.loads(raw)["id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                raise QuestionFileError(f"{path}:{line_no}: invalid rejected record") from None
    return ids


class ReviewSession:
    """Review state backed by the accepted and rejected files (resumable)."""

    def __init__(
        self,
        drafts: Sequence[Question],
        chunks: Sequence[Chunk],
        accepted_path: Path,
        rejected_path: Path,
    ) -> None:
        self._units = build_units(drafts)
        self._chunk_ids = {chunk.id for chunk in chunks}
        self._chunks = chunks
        self._accepted_path = accepted_path
        self._rejected_path = rejected_path
        self._done: set[str] = set()
        self._accepted_by_group: dict[str, int] = dict.fromkeys(TARGETS, 0)
        self._rejected = 0

        if accepted_path.exists():
            for question in load_questions(accepted_path):
                self._done.add(question.id)
                self._accepted_by_group[question.group] += 1
        if rejected_path.exists():
            rejected_ids = _read_rejected_ids(rejected_path)
            self._done.update(rejected_ids)
            self._rejected = len(rejected_ids)
        self._refuse_partly_reviewed_units(accepted_path, rejected_path)

    def _refuse_partly_reviewed_units(self, accepted_path: Path, rejected_path: Path) -> None:
        """A unit is written whole, so a half-decided one means the draft was regrouped.

        Reviewing it again would write the decided questions twice (a duplicate id breaks
        the accepted file), so stop and let the user fix the files first.
        """
        for unit in self._units:
            decided = [q.id for q in unit.questions if q.id in self._done]
            if decided and len(decided) < len(unit.questions):
                raise QuestionFileError(
                    f"unit {unit.key} is only partly reviewed (already decided: "
                    f"{', '.join(decided)}); fix the draft or {accepted_path.name} / "
                    f"{rejected_path.name} so the unit is whole"
                )

    @property
    def total_units(self) -> int:
        return len(self._units)

    def pending(self) -> list[ReviewUnit]:
        """Units that still have a question neither accepted nor rejected."""
        return [u for u in self._units if not all(q.id in self._done for q in u.questions)]

    def accept(self, unit: ReviewUnit, gold_ids: list[str] | None = None) -> None:
        if gold_ids is not None and (not gold_ids or unit.questions[0].group == OUT_OF_SCOPE):
            raise ValueError("gold_ids must be non-empty and are not allowed for out_of_scope")
        questions = unit.questions
        if gold_ids is not None:
            for gold_id in gold_ids:
                if gold_id not in self._chunk_ids:
                    raise ValueError(f"unknown chunk id: {gold_id}")
            ordered = order_gold_ids(gold_ids, self._chunks)
            update = {
                "gold_ids": ordered,
                "gold_articles": gold_articles_for(ordered, self._chunks),
            }
            questions = tuple(q.model_copy(update=update) for q in questions)

        # One write for the whole unit, so a pair is never left half written.
        payload = "".join(question_to_json(q) + "\n" for q in questions)
        _append(self._accepted_path, payload)
        for question in questions:
            self._done.add(question.id)
            self._accepted_by_group[question.group] += 1

    def reject(self, unit: ReviewUnit, reason: str) -> None:
        reason = unicodedata.normalize("NFC", reason).strip()
        if not reason:
            raise ValueError("reason is empty")
        payload = "".join(
            json.dumps({"id": q.id, "reason": reason, "record": q.model_dump()}, ensure_ascii=False)
            + "\n"
            for q in unit.questions
        )
        _append(self._rejected_path, payload)
        for question in unit.questions:
            self._done.add(question.id)
            self._rejected += 1

    def counts(self) -> dict[str, int]:
        return {**self._accepted_by_group, "rejected": self._rejected}


def _append(path: Path, payload: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as out:
        out.write(payload)
        out.flush()


def _location(chunk: Chunk) -> str:
    if chunk.clause is None:
        return f"Điều {chunk.article}"
    return f"Điều {chunk.article}, khoản {chunk.clause}"


def _render_chunk(chunk: Chunk) -> str:
    return f"{_location(chunk)} — {chunk.article_title}\n{chunk.text}"


def render_unit(
    unit: ReviewUnit,
    gold_ids: list[str],
    chunks: Sequence[Chunk],
    position: int,
    total: int,
    counts: dict[str, int],
) -> str:
    lines = [f"{position}/{total}  " + ", ".join(f"{name} {n}" for name, n in counts.items()), ""]
    for question in unit.questions:
        lines.append(f"[{question.group}] {question.id}: {question.question}")
        if question.note:
            lines.append(f"  note: {question.note}")
    lines.append("")
    if unit.questions[0].group == OUT_OF_SCOPE:
        lines.append("không có đáp án")
    else:
        by_id = {chunk.id: chunk for chunk in chunks}
        for gold_id in gold_ids:
            lines += [_render_chunk(by_id[gold_id]), ""]
    lines.append(f"Sửa câu chữ trong {DRAFT_NAME}")
    return "\n".join(lines)


def render_articles(gold_ids: Sequence[str], chunks: Sequence[Chunk]) -> str:
    """Every clause of the articles that the given ids belong to, in corpus order."""
    articles = set(gold_articles_for(gold_ids, chunks))
    return "\n\n".join(_render_chunk(chunk) for chunk in chunks if chunk.article in articles)


def _summary(session: ReviewSession, exhausted: bool) -> str:
    counts = session.counts()
    parts = [f"{group} {counts[group]}/{target}" for group, target in TARGETS.items()]
    lines = [f"Đã duyệt: {', '.join(parts)}, bị loại {counts['rejected']}"]
    short = [group for group, target in TARGETS.items() if counts[group] < target]
    if exhausted and short:
        lines.append(
            f"Còn thiếu so với mục tiêu: {', '.join(short)}. "
            f"Sửa hoặc thêm câu trong {DRAFT_NAME} rồi chạy lại."
        )
    return "\n".join(lines)


def run(
    session: ReviewSession,
    chunks: Sequence[Chunk],
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    clear = write is print and sys.stdout.isatty()
    pending = session.pending()
    first_position = session.total_units - len(pending) + 1
    try:
        for offset, unit in enumerate(pending):
            outcome = _review_unit(
                session, unit, chunks, first_position + offset, read, write, clear
            )
            if outcome == "quit":
                write(_summary(session, exhausted=False))
                return 0
    except (EOFError, KeyboardInterrupt):
        # Nothing was written for the unit in progress, so a rerun starts from it again.
        write("")
        write(_summary(session, exhausted=False))
        return 0
    write(_summary(session, exhausted=True))
    return 0


def _review_unit(
    session: ReviewSession,
    unit: ReviewUnit,
    chunks: Sequence[Chunk],
    position: int,
    read: Callable[[str], str],
    write: Callable[[str], None],
    clear: bool,
) -> str:
    """Handle one unit. Returns "done" once it is written, or "quit" if the user stops."""
    is_out_of_scope = unit.questions[0].group == OUT_OF_SCOPE
    gold_ids = list(unit.questions[0].gold_ids)
    edited = False
    redraw = True
    while True:
        if redraw:
            if clear:
                os.system("cls" if os.name == "nt" else "clear")
            screen = render_unit(
                unit, gold_ids, chunks, position, session.total_units, session.counts()
            )
            write(screen)
            redraw = False
        choice = read(PROMPT).strip().lower()
        if choice == "a":
            session.accept(unit, gold_ids if edited else None)
            return "done"
        if choice == "q":
            return "quit"
        if choice == "r":
            try:
                session.reject(unit, read("reason > "))
            except ValueError as error:
                write(f"{error}; the unit was not rejected")
                continue
            return "done"
        if choice in ("e", "c") and is_out_of_scope:
            write("not available for out_of_scope")
        elif choice == "c":
            write(render_articles(gold_ids, chunks))
        elif choice == "e":
            while True:
                text = read(EDIT_PROMPT)
                if not text.strip():
                    write("edit cancelled, ids unchanged")
                    break
                try:
                    gold_ids = parse_gold_ids(text, chunks)
                except ValueError as error:
                    write(str(error))
                    continue
                edited = True
                redraw = True
                break
        else:
            write("unknown key")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not use UTF-8; Vietnamese output must not crash.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    # Piped input (a script or PowerShell) arrives as UTF-8, possibly with a BOM, not in the
    # console code page. An interactive console is read through the console API instead.
    reconfigure_stdin = getattr(sys.stdin, "reconfigure", None)
    if reconfigure_stdin is not None and not sys.stdin.isatty():
        reconfigure_stdin(encoding="utf-8-sig")
    parser = argparse.ArgumentParser(description="Review the draft evaluation questions.")
    parser.add_argument("--draft", type=Path, default=REPO_ROOT / "eval" / "questions.draft.jsonl")
    parser.add_argument("--accepted", type=Path, default=REPO_ROOT / "eval" / "questions.jsonl")
    parser.add_argument(
        "--rejected", type=Path, default=REPO_ROOT / "eval" / "questions.rejected.jsonl"
    )
    parser.add_argument(
        "--chunks", type=Path, default=REPO_ROOT / "data" / "processed" / "chunks.jsonl"
    )
    args = parser.parse_args(argv)

    try:
        chunks = load_chunks(args.chunks)
        drafts = load_questions(args.draft)
        missing = check_gold_ids(drafts, chunks)
        if missing:
            print("unknown gold ids: " + ", ".join(missing), file=sys.stderr)
            return 1
        session = ReviewSession(drafts, chunks, args.accepted, args.rejected)
    except ValidationError as error:
        print(f"invalid chunks file: {error}", file=sys.stderr)
        return 1
    except (QuestionFileError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return run(session, chunks)


if __name__ == "__main__":
    sys.exit(main())
