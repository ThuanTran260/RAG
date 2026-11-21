import json
import unicodedata

import pytest

from eval.questions import (
    Question,
    QuestionFileError,
    check_gold_ids,
    gold_articles_for,
    load_questions,
    order_gold_ids,
    question_to_json,
)
from ingestion.documents import REPO_ROOT

D35K1 = "45-2019-QH14:d35:k1"
D35K2 = "45-2019-QH14:d35:k2"
D24K1 = "45-2019-QH14:d24:k1"


def record(id_, group, pair_id, gold_ids, gold_articles, question="Người lao động nghỉ bao nhiêu?"):
    return {
        "id": id_,
        "group": group,
        "pair_id": pair_id,
        "question": question,
        "gold_ids": gold_ids,
        "gold_articles": gold_articles,
        "note": "",
    }


def valid_records():
    return [
        record("q001", "direct", "p01", [D35K1], [35]),
        record("q002", "paraphrase", "p01", [D35K1], [35], "Tôi muốn nghỉ việc thì sao?"),
        record("q003", "article_ref", None, [D24K1], [24], "Điều 24 quy định gì?"),
        record("q004", "out_of_scope", None, [], [], "Giá vàng hôm nay là bao nhiêu?"),
    ]


def write_lines(tmp_path, records):
    """Write one JSON object per line; a str record is written as is (for broken JSON)."""
    lines = [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in records]
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def test_load_questions_accepts_a_valid_file(tmp_path):
    questions = load_questions(write_lines(tmp_path, valid_records()))

    assert len(questions) == 4
    assert all(isinstance(q, Question) for q in questions)
    assert [q.id for q in questions] == ["q001", "q002", "q003", "q004"]
    assert questions[1].group == "paraphrase"
    assert questions[3].gold_ids == []


def test_load_questions_accepts_bom_crlf_and_blank_lines(tmp_path):
    records = valid_records()
    text = "\r\n\r\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\r\n"
    path = tmp_path / "questions.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))

    assert [q.id for q in load_questions(path)] == ["q001", "q002", "q003", "q004"]

    # Lines are: 1 record, 2 blank, 3 record. The error must name physical line 3.
    records[1]["id"] = "Q2"
    text = "\r\n".join(
        [json.dumps(records[0], ensure_ascii=False), ""]
        + [json.dumps(r, ensure_ascii=False) for r in records[1:]]
    )
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    with pytest.raises(QuestionFileError, match=r":3:"):
        load_questions(path)


def test_load_questions_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_bytes(b"")

    assert load_questions(path) == []


def _broken_json(records):
    return [records[0], "{not json"] + records[2:], 2, "invalid JSON"


def _extra_field(records):
    records[0]["score"] = 1
    return records, 1, "score (extra_forbidden)"


def _missing_gold_articles(records):
    del records[2]["gold_articles"]
    return records, 3, "gold_articles (missing)"


def _bad_id(records):
    records[1]["id"] = "Q1"
    return records, 2, "id (string_pattern_mismatch)"


def _duplicate_id(records):
    records[1]["id"] = "q001"
    return records, 2, "duplicate id"


def _bad_group(records):
    records[2]["group"] = "other"
    return records, 3, "group (literal_error)"


def _nfd_question(records):
    records[3]["question"] = unicodedata.normalize("NFD", "Người lao động")
    return records, 4, "not NFC-normalized"


def _blank_question(records):
    records[3]["question"] = "   "
    return records, 4, "question is empty"


def _out_of_scope_with_gold(records):
    records[3]["gold_ids"] = [D35K1]
    records[3]["gold_articles"] = [35]  # keep the iff rule satisfied so the group rule fires
    return records, 4, "out_of_scope needs"


def _direct_without_pair(records):
    records[0]["pair_id"] = None
    return records, 1, "direct needs gold_ids and a pair_id"


def _article_out_of_range(records):
    records[2]["gold_articles"] = [221]
    return records, 3, "within 1-220"


def _articles_not_increasing(records):
    records[2]["gold_ids"] = [D35K2, D24K1]
    records[2]["gold_articles"] = [36, 35]
    return records, 3, "strictly increasing"


def _two_directs_in_pair(records):
    records[1]["group"] = "direct"
    return records, 1, "exactly one direct and one paraphrase"


@pytest.mark.parametrize(
    "mutate",
    [
        _broken_json,
        _extra_field,
        _missing_gold_articles,
        _bad_id,
        _duplicate_id,
        _bad_group,
        _nfd_question,
        _blank_question,
        _out_of_scope_with_gold,
        _direct_without_pair,
        _article_out_of_range,
        _articles_not_increasing,
        _two_directs_in_pair,
    ],
)
def test_load_questions_rejects(tmp_path, mutate):
    records, line, reason = mutate(valid_records())
    path = write_lines(tmp_path, records)

    with pytest.raises(QuestionFileError) as info:
        load_questions(path)

    assert f":{line}:" in str(info.value)
    assert reason in str(info.value)


def test_load_questions_error_does_not_echo_line_content(tmp_path):
    records = valid_records()
    records[0]["id"] = "Q1"
    records[0]["question"] = "Câu hỏi bí mật không được lộ ra"

    with pytest.raises(QuestionFileError) as info:
        load_questions(write_lines(tmp_path, records))

    assert "bí mật" not in str(info.value)
    assert "id" in str(info.value)


def test_load_questions_rejects_pair_with_different_gold(tmp_path):
    records = valid_records()
    records[1]["gold_ids"] = [D35K2]

    with pytest.raises(QuestionFileError, match=r":1:"):
        load_questions(write_lines(tmp_path, records))


def test_load_questions_rejects_incomplete_pair(tmp_path):
    records = valid_records()[:1]

    with pytest.raises(QuestionFileError, match=r":1:"):
        load_questions(write_lines(tmp_path, records))


def _question(**changes):
    return Question.model_validate({**valid_records()[0], **changes})


def test_check_gold_ids_returns_missing_ids(all_chunks):
    missing_id = "45-2019-QH14:d35:k99"
    questions = [_question(gold_ids=[missing_id]), _question(id="q002", gold_ids=[D35K1])]

    assert check_gold_ids(questions, all_chunks) == [missing_id]


def test_check_gold_ids_accepts_consistent_questions(all_chunks):
    questions = [_question(gold_ids=[D35K1, D35K2], gold_articles=[35])]

    assert check_gold_ids(questions, all_chunks) == []


def test_check_gold_ids_rejects_wrong_articles(all_chunks):
    question = _question(id="q007", gold_ids=[D35K1], gold_articles=[36])

    with pytest.raises(QuestionFileError, match="q007"):
        check_gold_ids([question], all_chunks)


def test_check_gold_ids_rejects_out_of_corpus_order(all_chunks):
    question = _question(id="q008", gold_ids=[D35K2, D35K1], gold_articles=[35])

    with pytest.raises(QuestionFileError, match="q008"):
        check_gold_ids([question], all_chunks)


def test_gold_articles_for_is_sorted_and_unique(all_chunks):
    assert gold_articles_for([D35K2, D24K1, D35K1], all_chunks) == [24, 35]


def test_order_gold_ids_dedupes_and_sorts_by_corpus(all_chunks):
    assert order_gold_ids([D35K2, D35K1, D35K2], all_chunks) == [D35K1, D35K2]


def test_question_to_json_round_trips_vietnamese():
    question = Question.model_validate(valid_records()[0])

    line = question_to_json(question)

    assert "Người" in line
    assert r"\u" not in line
    assert "\n" not in line
    assert Question.model_validate_json(line) == question


def test_reviewed_questions_file_is_valid(all_chunks):
    questions = load_questions(REPO_ROOT / "eval/questions.jsonl")

    assert check_gold_ids(questions, all_chunks) == []
    pair_counts: dict[str, int] = {}
    for q in questions:
        if q.group in ("direct", "paraphrase"):
            pair_counts[q.pair_id] = pair_counts.get(q.pair_id, 0) + 1
    assert all(count == 2 for count in pair_counts.values())
