import codecs
import io
import json
import sys
import unicodedata

import pytest

from eval import review_questions
from eval.questions import QuestionFileError, check_gold_ids, load_questions
from eval.review_questions import (
    ReviewSession,
    build_units,
    main,
    parse_gold_ids,
    render_articles,
    render_unit,
    run,
)
from ingestion.documents import REPO_ROOT

D35K1 = "45-2019-QH14:d35:k1"
D35K2 = "45-2019-QH14:d35:k2"
D24K1 = "45-2019-QH14:d24:k1"
OUT_OF_SCOPE_NOTE = "Not about labour law."


def record(id_, group, pair_id, gold_ids, gold_articles, question, note=""):
    return {
        "id": id_,
        "group": group,
        "pair_id": pair_id,
        "question": question,
        "gold_ids": gold_ids,
        "gold_articles": gold_articles,
        "note": note,
    }


def draft_records():
    """Pair p01 (two questions), one article_ref and one out_of_scope: three review units."""
    return [
        record("q001", "direct", "p01", [D35K1], [35], "Báo trước bao lâu khi nghỉ việc?"),
        record("q002", "paraphrase", "p01", [D35K1], [35], "Tôi muốn nghỉ việc thì sao?"),
        record("q003", "article_ref", None, [D24K1], [24], "Điều 24 quy định gì?"),
        record("q004", "out_of_scope", None, [], [], "Giá vàng hôm nay?", OUT_OF_SCOPE_NOTE),
    ]


def write_drafts(tmp_path, records, name="questions.draft.jsonl"):
    path = tmp_path / name
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def paths(tmp_path):
    return {
        "draft": write_drafts(tmp_path, draft_records()),
        "accepted": tmp_path / "questions.jsonl",
        "rejected": tmp_path / "questions.rejected.jsonl",
    }


@pytest.fixture
def make_session(paths, all_chunks):
    def _make(records=None):
        if records is not None:
            write_drafts(paths["draft"].parent, records)
        drafts = load_questions(paths["draft"])
        return ReviewSession(drafts, all_chunks, paths["accepted"], paths["rejected"])

    return _make


def script(*answers, end=EOFError):
    """A fake `read`: returns the answers one by one, then raises `end` (EOF by default)."""
    remaining = list(answers)

    def read(prompt=""):
        if not remaining:
            raise end
        return remaining.pop(0)

    return read


def collector():
    lines: list[str] = []
    return lines, lines.append


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def chunk_text(all_chunks, chunk_id):
    return next(chunk.text for chunk in all_chunks if chunk.id == chunk_id)


# ---- build_units ----


def test_build_units_groups_pairs_in_first_seen_order(paths):
    # The draft lists the paraphrase before its direct question and another unit in between.
    records = draft_records()
    records[0], records[1] = records[1], records[0]
    records.insert(1, records.pop(2))  # q003 now sits between the two pair members
    drafts = load_questions(write_drafts(paths["draft"].parent, records))

    units = build_units(drafts)

    assert [unit.key for unit in units] == ["p01", "q003", "q004"]
    assert [q.group for q in units[0].questions] == ["direct", "paraphrase"]
    assert [len(unit.questions) for unit in units] == [2, 1, 1]


# ---- parse_gold_ids ----


def test_parse_gold_ids_accepts_short_and_separated_forms(all_chunks):
    text = "d35:k2, d35:k1  45-2019-QH14:d35:k2"
    assert parse_gold_ids(text, all_chunks) == [D35K1, D35K2]


def test_parse_gold_ids_rejects_unknown_and_empty(all_chunks):
    with pytest.raises(ValueError, match="unknown chunk id: 45-2019-QH14:d35:k99"):
        parse_gold_ids("d35:k1, d35:k99", all_chunks)
    with pytest.raises(ValueError, match="unknown chunk id: 45-2019-QH14:abc"):
        parse_gold_ids("abc", all_chunks)
    for blank in ("", "   ", " , "):
        with pytest.raises(ValueError, match="no chunk id given"):
            parse_gold_ids(blank, all_chunks)


# ---- ReviewSession ----


def test_accept_writes_pair_and_resume_skips_it(make_session, paths, all_chunks):
    session = make_session()
    first = session.pending()[0]

    session.accept(first)

    written = read_lines(paths["accepted"])
    assert [line["id"] for line in written] == ["q001", "q002"]
    assert "Báo trước" in paths["accepted"].read_text(encoding="utf-8")  # no \u escapes
    resumed = make_session()
    assert [unit.key for unit in resumed.pending()] == ["q003", "q004"]
    accepted = load_questions(paths["accepted"])
    assert check_gold_ids(accepted, all_chunks) == []


def test_accept_with_new_gold_updates_both_questions_and_articles(make_session, paths):
    session = make_session()

    session.accept(session.pending()[0], [D35K2])

    written = read_lines(paths["accepted"])
    assert [line["gold_ids"] for line in written] == [[D35K2], [D35K2]]
    assert [line["gold_articles"] for line in written] == [[35], [35]]


def test_accept_orders_new_gold_ids_by_corpus_order(make_session, paths):
    session = make_session()

    session.accept(session.pending()[0], [D35K2, D35K1, D35K2])

    assert read_lines(paths["accepted"])[0]["gold_ids"] == [D35K1, D35K2]


def test_accept_rules_for_empty_gold_ids_and_out_of_scope(make_session, paths):
    session = make_session()
    pair, _article_ref, out_of_scope = session.pending()

    with pytest.raises(ValueError):
        session.accept(pair, [])
    with pytest.raises(ValueError):
        session.accept(out_of_scope, [D35K1])
    with pytest.raises(ValueError, match="unknown chunk id"):
        session.accept(pair, ["45-2019-QH14:d35:k99"])
    assert not paths["accepted"].exists()

    session.accept(out_of_scope)  # a plain accept of out_of_scope is valid

    line = read_lines(paths["accepted"])[0]
    assert (line["id"], line["gold_ids"], line["gold_articles"]) == ("q004", [], [])


def test_reject_writes_reason_for_every_question_in_unit(make_session, paths):
    session = make_session()
    pair = session.pending()[0]

    session.reject(pair, "  Câu hỏi mơ hồ  ")

    written = read_lines(paths["rejected"])
    assert [set(line) for line in written] == [{"id", "reason", "record"}] * 2
    assert [line["id"] for line in written] == ["q001", "q002"]
    assert all(line["reason"] == "Câu hỏi mơ hồ" for line in written)
    assert written[0]["record"]["question"] == "Báo trước bao lâu khi nghỉ việc?"
    assert [unit.key for unit in make_session().pending()] == ["q003", "q004"]
    assert session.counts()["rejected"] == 2


def test_reject_normalizes_reason_to_nfc(make_session, paths):
    session = make_session()

    session.reject(session.pending()[0], unicodedata.normalize("NFD", "Thiếu ngữ cảnh"))

    assert read_lines(paths["rejected"])[0]["reason"] == "Thiếu ngữ cảnh"


def test_reject_requires_reason(make_session, paths):
    session = make_session()
    pair = session.pending()[0]

    for reason in ("", "   \t"):
        with pytest.raises(ValueError):
            session.reject(pair, reason)
    assert not paths["rejected"].exists()
    assert len(session.pending()) == 3


@pytest.mark.parametrize("decided_in", ["accepted", "rejected"])
def test_partly_reviewed_pair_is_refused_at_session_start(make_session, paths, decided_in):
    # Only q001 of pair p01 was decided, e.g. because the draft was regrouped afterwards.
    first = draft_records()[0] | {"pair_id": None, "group": "article_ref"}
    if decided_in == "accepted":
        paths["accepted"].write_text(json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        row = {"id": "q001", "reason": "x", "record": first}
        paths["rejected"].write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(QuestionFileError, match=r"unit p01 .*already decided: q001"):
        make_session()


def test_main_refuses_a_partly_reviewed_unit_with_a_one_line_message(paths, monkeypatch, capsys):
    first = draft_records()[0] | {"pair_id": None, "group": "article_ref"}
    paths["accepted"].write_text(json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(review_questions, "run", pytest.fail)

    code = main(
        [
            "--draft",
            str(paths["draft"]),
            "--accepted",
            str(paths["accepted"]),
            "--rejected",
            str(paths["rejected"]),
        ]
    )

    err = capsys.readouterr().err
    assert code == 1
    assert len(err.strip().splitlines()) == 1
    assert "unit p01" in err and "q001" in err
    assert paths["accepted"].read_text(encoding="utf-8").count("\n") == 1  # not rewritten


def test_counts_group_accepted_questions_and_include_rejected(make_session):
    session = make_session()
    pair, article_ref, out_of_scope = session.pending()
    session.accept(pair)
    session.accept(article_ref)
    session.reject(out_of_scope, "Quá chung chung")

    assert session.counts() == {
        "direct": 1,
        "paraphrase": 1,
        "article_ref": 1,
        "out_of_scope": 0,
        "rejected": 1,
    }
    assert make_session().counts() == session.counts()  # restored from the files


# ---- rendering ----


def test_render_unit_shows_questions_and_gold_text(make_session, all_chunks):
    session = make_session()
    pair, _article_ref, _out = session.pending()
    counts = {"direct": 4, "paraphrase": 4, "article_ref": 0, "out_of_scope": 0, "rejected": 1}

    text = render_unit(pair, [D35K1], all_chunks, 1, 3, counts)

    assert "1/3" in text
    assert "direct 4" in text and "rejected 1" in text
    for question in pair.questions:
        assert f"{question.id}" in text and question.question in text
    assert "Điều 35, khoản 1" in text
    assert "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động" in text
    assert chunk_text(all_chunks, D35K1) in text
    assert "Sửa câu chữ trong eval/questions.draft.jsonl" in text


def test_render_unit_uses_the_given_gold_ids_not_the_draft_ones(make_session, all_chunks):
    pair = make_session().pending()[0]

    text = render_unit(pair, [D35K2], all_chunks, 1, 3, {})

    assert chunk_text(all_chunks, D35K2) in text
    assert chunk_text(all_chunks, D35K1) not in text


def test_render_unit_out_of_scope_has_no_answer_and_shows_note(make_session, all_chunks):
    out_of_scope = make_session().pending()[2]

    text = render_unit(out_of_scope, [], all_chunks, 3, 3, {})

    assert "không có đáp án" in text
    assert OUT_OF_SCOPE_NOTE in text
    assert "Điều" not in text


def test_render_unit_omits_clause_for_an_article_without_clauses(tmp_path, all_chunks):
    chunk = next(c for c in all_chunks if c.clause is None)
    draft = write_drafts(
        tmp_path,
        [record("q001", "article_ref", None, [chunk.id], [chunk.article], "Điều này quy định gì?")],
    )
    unit = build_units(load_questions(draft))[0]

    text = render_unit(unit, [chunk.id], all_chunks, 1, 1, {})

    assert f"Điều {chunk.article} — {chunk.article_title}" in text
    assert ", khoản" not in text


def test_render_articles_shows_every_clause_of_the_answer_articles(all_chunks):
    text = render_articles([D35K1], all_chunks)

    for chunk in all_chunks:
        if chunk.article == 35:
            assert chunk.text in text
        elif chunk.article in (34, 36):
            assert chunk.text not in text


# ---- run ----


def test_run_edit_invalid_then_valid_then_accept(make_session, paths, all_chunks):
    session = make_session()
    lines, write = collector()

    code = run(session, all_chunks, script("e", "d35:k99", "d35:k2", "a", "q"), write)

    assert code == 0
    assert "unknown chunk id: 45-2019-QH14:d35:k99" in "\n".join(lines)
    written = read_lines(paths["accepted"])
    assert [(line["id"], line["gold_ids"]) for line in written] == [
        ("q001", [D35K2]),
        ("q002", [D35K2]),
    ]


def test_run_edit_is_kept_in_memory_until_accept(make_session, paths, all_chunks):
    session = make_session()
    lines, write = collector()

    run(session, all_chunks, script("e", "d35:k2", "q"), write)

    assert not paths["accepted"].exists()
    redrawn = "\n".join(lines)
    assert chunk_text(all_chunks, D35K2) in redrawn


def test_run_context_shows_whole_article(make_session, all_chunks):
    session = make_session()
    article_35 = [chunk for chunk in all_chunks if chunk.article == 35]
    second_clause = chunk_text(all_chunks, D35K2)

    without_context, write = collector()
    run(session, all_chunks, script("q"), write)
    with_context, write = collector()
    run(session, all_chunks, script("c", "q"), write)

    assert second_clause not in "\n".join(without_context)
    for chunk in article_35:
        assert chunk.text in "\n".join(with_context)


def test_run_out_of_scope_disables_edit_and_context(make_session, paths, all_chunks):
    session = make_session([draft_records()[3]])
    lines, write = collector()

    code = run(session, all_chunks, script("e", "c", "a"), write)

    output = "\n".join(lines)
    assert code == 0
    assert output.count("not available for out_of_scope") == 2
    assert "không có đáp án" in output
    assert read_lines(paths["accepted"])[0]["gold_ids"] == []


def test_run_edit_empty_line_cancels_and_keeps_ids(make_session, paths, all_chunks):
    session = make_session()
    lines, write = collector()

    # "e" + empty line cancels; a whitespace-only line cancels too; "a" keeps the draft ids.
    run(session, all_chunks, script("e", "", "e", "  ", "a", "q"), write)

    output = "\n".join(lines)
    assert output.count("edit cancelled, ids unchanged") == 2
    assert "unknown chunk id" not in output
    written = read_lines(paths["accepted"])
    assert [(line["id"], line["gold_ids"]) for line in written] == [
        ("q001", [D35K1]),
        ("q002", [D35K1]),
    ]


def test_run_edit_invalid_ids_still_reprompt_after_a_cancel(make_session, paths, all_chunks):
    session = make_session()
    lines, write = collector()

    run(session, all_chunks, script("e", "", "e", "d35:k99", "d35:k2", "a", "q"), write)

    assert "unknown chunk id: 45-2019-QH14:d35:k99" in "\n".join(lines)
    assert read_lines(paths["accepted"])[0]["gold_ids"] == [D35K2]


def test_run_reject_asks_for_a_reason_and_skips_an_empty_one(make_session, paths, all_chunks):
    session = make_session()
    lines, write = collector()

    run(session, all_chunks, script("r", "   ", "r", "Trùng ý với câu khác", "q"), write)

    assert "reason is empty" in "\n".join(lines)
    written = read_lines(paths["rejected"])
    assert [line["id"] for line in written] == ["q001", "q002"]
    assert written[0]["reason"] == "Trùng ý với câu khác"


def test_run_unknown_key_asks_again(make_session, paths, all_chunks):
    session = make_session()
    _lines, write = collector()

    code = run(session, all_chunks, script("x", "", "a", "q"), write)

    assert code == 0
    assert [line["id"] for line in read_lines(paths["accepted"])] == ["q001", "q002"]


@pytest.mark.parametrize("end", [EOFError, KeyboardInterrupt])
def test_eof_mid_unit_writes_nothing_and_resume_continues(make_session, paths, all_chunks, end):
    session = make_session()
    lines, write = collector()

    # "e" is answered, then the read for the ids raises: the unit is half done.
    code = run(session, all_chunks, script("e", end=end), write)

    assert code == 0
    assert not paths["accepted"].exists() or paths["accepted"].read_text(encoding="utf-8") == ""
    assert not paths["rejected"].exists()
    assert "direct 0/40" in "\n".join(lines)  # the summary is still printed

    resumed = make_session()
    assert resumed.pending()[0].key == "p01"
    run(resumed, all_chunks, script("a", "q"), write)
    assert [line["id"] for line in read_lines(paths["accepted"])] == ["q001", "q002"]


def test_run_summary_reports_counts_against_targets(make_session, all_chunks):
    session = make_session()
    lines, write = collector()

    code = run(session, all_chunks, script("a", "a", "r", "Lệch chủ đề"), write)

    output = "\n".join(lines)
    assert code == 0
    for expected in ("direct 1/40", "paraphrase 1/40", "article_ref 1/10", "out_of_scope 0/20"):
        assert expected in output
    assert "bị loại 1" in output
    assert "Còn thiếu" in output  # all units are done but the targets are not met


def test_run_quit_early_prints_summary_without_the_missing_reminder(make_session, all_chunks):
    session = make_session()
    lines, write = collector()

    code = run(session, all_chunks, script("a", "q"), write)

    output = "\n".join(lines)
    assert code == 0
    assert "direct 1/40" in output
    assert "Còn thiếu" not in output


def test_run_with_nothing_pending_returns_zero(make_session, all_chunks):
    session = make_session()
    for unit in session.pending():
        session.accept(unit)
    lines, write = collector()

    assert run(make_session(), all_chunks, script(), write) == 0
    assert "direct 1/40" in "\n".join(lines)


# ---- main ----


def test_main_rejects_invalid_draft_before_review(paths, monkeypatch, capsys):
    records = draft_records()
    records[0]["question"] = unicodedata.normalize("NFD", records[0]["question"])
    write_drafts(paths["draft"].parent, records)
    monkeypatch.setattr(review_questions, "run", pytest.fail)

    code = main(
        [
            "--draft",
            str(paths["draft"]),
            "--accepted",
            str(paths["accepted"]),
            "--rejected",
            str(paths["rejected"]),
        ]
    )

    assert code == 1
    assert "questions.draft.jsonl:1:" in capsys.readouterr().err


def test_main_rejects_unknown_gold_ids_before_review(paths, monkeypatch, capsys):
    records = draft_records()
    records[2]["gold_ids"] = ["45-2019-QH14:d24:k99"]
    write_drafts(paths["draft"].parent, records)
    monkeypatch.setattr(review_questions, "run", pytest.fail)

    code = main(["--draft", str(paths["draft"]), "--accepted", str(paths["accepted"])])

    assert code == 1
    assert "unknown gold ids: 45-2019-QH14:d24:k99" in capsys.readouterr().err


def test_main_reports_a_broken_accepted_file(paths, monkeypatch, capsys):
    paths["accepted"].write_text("not json\n", encoding="utf-8")
    monkeypatch.setattr(review_questions, "run", pytest.fail)

    code = main(["--draft", str(paths["draft"]), "--accepted", str(paths["accepted"])])

    assert code == 1
    assert "questions.jsonl:1:" in capsys.readouterr().err


def test_main_reports_a_missing_draft_file(paths, monkeypatch, capsys):
    monkeypatch.setattr(review_questions, "run", pytest.fail)

    code = main(["--draft", str(paths["draft"].with_name("missing.jsonl"))])

    assert code == 1
    assert "missing.jsonl" in capsys.readouterr().err


def test_main_runs_the_wizard_on_a_valid_draft(paths, monkeypatch):
    seen = {}

    def fake_run(session, chunks):
        seen["pending"] = [unit.key for unit in session.pending()]
        seen["chunks"] = len(chunks)
        return 0

    monkeypatch.setattr(review_questions, "run", fake_run)

    code = main(
        [
            "--draft",
            str(paths["draft"]),
            "--accepted",
            str(paths["accepted"]),
            "--rejected",
            str(paths["rejected"]),
        ]
    )

    assert code == 0
    assert seen["pending"] == ["p01", "q003", "q004"]
    assert seen["chunks"] > 200


def test_main_reads_piped_stdin_as_utf8_with_bom(paths, monkeypatch):
    raw = codecs.BOM_UTF8 + "q\nKhông rõ ràng\n".encode()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="cp1252"))
    seen = []

    def fake_run(session, chunks):
        seen.extend([input(), input()])
        return 0

    monkeypatch.setattr(review_questions, "run", fake_run)

    code = main(
        [
            "--draft",
            str(paths["draft"]),
            "--accepted",
            str(paths["accepted"]),
            "--rejected",
            str(paths["rejected"]),
        ]
    )

    assert code == 0
    assert seen == ["q", "Không rõ ràng"]


def test_review_module_does_not_import_retrieval():
    source = (REPO_ROOT / "eval" / "review_questions.py").read_text(encoding="utf-8")

    for forbidden in ("app.rag.retriever", "app.rag.search", "app.rag.ask"):
        assert forbidden not in source
