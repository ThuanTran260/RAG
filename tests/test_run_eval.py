import hashlib
import json
from datetime import date

import pytest

from app.rag.retriever import RetrievedChunk
from app.rag.store import QdrantUnavailableError
from eval import run_eval
from eval.questions import Question, load_questions, question_to_json
from ingestion.documents import REPO_ROOT

PREFIX = "45-2019-QH14:"
DIGEST = "abc123def456"


def full_id(short: str) -> str:
    return PREFIX + short


def make_question(qid, group, text, gold, pair_id=None) -> Question:
    """Build a question from short gold ids such as 'd35:k1' (articles are derived)."""
    gold_ids = [full_id(g) for g in gold]
    articles = sorted({int(g.split(":")[0][1:]) for g in gold})
    return Question(
        id=qid,
        group=group,
        pair_id=pair_id,
        question=text,
        gold_ids=gold_ids,
        gold_articles=articles,
    )


class FakeRetriever:
    """Returns real chunks for the ids mapped to each query, with scores 0.9, 0.89, ..."""

    def __init__(self, mapping, chunks, fail_after=None):
        self.mapping = mapping
        self.by_id = {chunk.id: chunk for chunk in chunks}
        self.fail_after = fail_after
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query, k=5):
        self.calls.append((query, k))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise QdrantUnavailableError("Qdrant is not reachable at http://127.0.0.1:6333")
        ids = self.mapping[query]
        return [
            RetrievedChunk(chunk=self.by_id[full_id(short)], score=round(0.9 - 0.01 * rank, 2))
            for rank, short in enumerate(ids)
        ]


# Two pairs: p01 hits only for the direct question, p02 hits for both.
Q_DIRECT_1 = make_question("q001", "direct", "câu direct một", ["d35:k1"], "p01")
Q_PARA_1 = make_question("q002", "paraphrase", "câu paraphrase một", ["d35:k1"], "p01")
Q_REF = make_question("q003", "article_ref", "Điều 25 nói gì?", ["d25:k1"])
Q_OOS = make_question("q004", "out_of_scope", "Cách nấu phở bò ngon", [])
Q_DIRECT_2 = make_question("q005", "direct", "câu direct hai", ["d36:k2"], "p02")
Q_PARA_2 = make_question("q006", "paraphrase", "câu paraphrase hai", ["d36:k2"], "p02")
QUESTIONS = [Q_DIRECT_1, Q_PARA_1, Q_REF, Q_OOS, Q_DIRECT_2, Q_PARA_2]

MAPPING = {
    Q_DIRECT_1.question: ["d36:k2", "d36:k1", "d35:k2", "d35:k1"],
    Q_PARA_1.question: ["d36:k2", "d36:k1", "d37:k1"],
    Q_REF.question: ["d25:k1", "d25:k2"],
    Q_OOS.question: ["d37:k1"],
    Q_DIRECT_2.question: ["d36:k2", "d36:k1"],
    Q_PARA_2.question: ["d37:k1", "d36:k1", "d36:k2"],
}


@pytest.fixture
def fake(all_chunks):
    return FakeRetriever(MAPPING, all_chunks)


@pytest.fixture
def results(fake):
    return run_eval.evaluate(QUESTIONS, fake)


def write_questions(path, questions):
    path.write_text("".join(question_to_json(q) + "\n" for q in questions), encoding="utf-8")
    return path


def test_evaluate_ranks_article_and_clause_levels(results):
    first = results[0]
    assert first.question is Q_DIRECT_1
    assert first.rank_article == 3
    assert first.rank_clause == 4
    assert first.top[0] == (full_id("d36:k2"), 36, 0.9)


def test_evaluate_requests_k10_once_per_question(fake):
    run_eval.evaluate(QUESTIONS, fake)
    assert fake.calls == [(q.question, 10) for q in QUESTIONS]


def test_evaluate_out_of_scope_has_no_ranks(results):
    oos = results[3]
    assert oos.question is Q_OOS
    assert oos.rank_article is None
    assert oos.rank_clause is None
    assert oos.top  # the scores are still recorded


def test_render_report_lists_misses_and_pairs(results):
    report = run_eval.render_report(
        results, clause_level=True, collection="coll", model="mdl", digest=DIGEST
    )
    assert "q002" in report  # the paraphrase that misses the top 5
    assert "d36:k2" in report  # its first retrieved chunk
    assert "q001" not in report.split("## 3.")[1].split("## 4.")[0]  # a hit is not listed
    for label in ("cả hai trúng", "chỉ direct trúng", "chỉ paraphrase trúng", "cả hai trượt"):
        assert label in report
    assert "p01" in report  # the direct-only pair
    assert "coll" in report and "mdl" in report and DIGEST in report


def test_render_report_prints_scores_with_decimal_comma(results):
    report = run_eval.render_report(
        results, clause_level=True, collection="c", model="m", digest=DIGEST
    )
    assert "0,750" in report  # main article-level recall@5: 3 of 4 hit
    assert "0,458" in report  # main article-level MRR
    assert "0,396" in report  # main clause-level MRR


def _section(report, number):
    """Lines of the numbered report section (without its heading)."""
    body = report.split(f"## {number}.")[1]
    return body.split("\n\n## ")[0].splitlines()[1:]


def _row(lines, name):
    return next(line.split() for line in lines if line.startswith(name))


def test_render_report_table_has_recall_at_1_for_both_levels(results):
    report = run_eval.render_report(
        results, clause_level=True, collection="c", model="m", digest=DIGEST
    )
    table = _section(report, 2)
    # name, n, then Điều R@1 R@5 R@10 MRR@10, then Khoản R@1 R@5 R@10 MRR@10
    assert _row(table, "direct") == [
        *("direct", "2"),
        *("0,500", "1,000", "1,000", "0,667"),
        *("0,500", "1,000", "1,000", "0,625"),
    ]
    assert _row(table, "paraphrase")[2:4] == ["0,000", "0,500"]  # Điều R@1, R@5
    assert _row(table, "số chính")[3:6] == ["0,250", "0,750", "0,750"]  # after "số", "chính", n


def test_render_report_recall_at_1_is_na_at_clause_level_without_gold_ids(results):
    report = run_eval.render_report(
        results, clause_level=False, collection="c", model="m", digest=DIGEST
    )
    row = _row(_section(report, 2), "direct")
    assert row[2:6] == ["0,500", "1,000", "1,000", "0,667"]
    assert row[6:] == ["N/A", "N/A", "N/A", "N/A"]


def test_render_report_lists_article_rank_histograms_for_direct_and_paraphrase(results):
    report = run_eval.render_report(
        results, clause_level=True, collection="c", model="m", digest=DIGEST
    )
    histogram = _section(report, 6)
    assert report.index("## 5.") < report.index("## 6.")  # existing sections keep their numbers
    # ranks 1..10 then "miss" (not in the top 10); direct ranks are [3, 1]
    assert _row(histogram, "direct") == ["direct", "1", "0", "1", *["0"] * 7, "0"]
    # paraphrase ranks are [None, 2]
    assert _row(histogram, "paraphrase") == ["paraphrase", "0", "1", *["0"] * 8, "1"]
    assert "article_ref" not in "\n".join(histogram)


def test_render_report_histogram_handles_a_group_without_questions(fake):
    only_direct = run_eval.evaluate([Q_DIRECT_1], fake)
    report = run_eval.render_report(
        only_direct, clause_level=True, collection="c", model="m", digest=DIGEST
    )
    assert _row(_section(report, 6), "paraphrase") == ["paraphrase", *["0"] * 11]


def test_render_report_without_clause_level_prints_na(results):
    report = run_eval.render_report(
        results, clause_level=False, collection="c", model="m", digest=DIGEST
    )
    assert "N/A" in report
    assert "0,396" not in report  # clause-level MRR must not be printed


def test_render_report_handles_groups_without_questions(fake):
    only_direct = run_eval.evaluate([Q_DIRECT_1], fake)
    report = run_eval.render_report(
        only_direct, clause_level=True, collection="c", model="m", digest=DIGEST
    )
    assert "N/A" in report  # paraphrase, article_ref and out_of_scope are empty
    row = run_eval.results_row(
        only_direct, clause_level=True, digest=DIGEST, label="x", day=date(2026, 11, 21)
    )
    assert "direct 0/1, paraphrase 0/0;" in row
    assert "n=1 (+0 article_ref, 0 ngoài phạm vi)" in row


def test_results_row_format(results):
    row = run_eval.results_row(
        results, clause_level=True, digest=DIGEST, label="dense baseline", day=date(2026, 11, 21)
    )
    # Main questions are q001, q002 (paraphrase), q005, q006 (paraphrase).
    # Article ranks [3, None, 1, 2]: R@1 1/4, R@5 and R@10 3/4, MRR (1/3 + 1 + 1/2) / 4 = 0.4583
    # Clause ranks [4, None, 1, 3]: R@1 1/4, R@5 and R@10 3/4, MRR (1/4 + 1 + 1/3) / 4 = 0.3958
    # Rank-1 articles by group: direct [3, 1] -> 1/2, paraphrase [None, 2] -> 0/2
    assert row == (
        "| 2026-11-21 | dense baseline | 0,250 | 0,750 | 0,750 | 0,458 | "
        "Mức Khoản: 0,250 / 0,750 / 0,750 / 0,396; "
        "R@1 theo nhóm (mức Điều): direct 1/2, paraphrase 0/2; "
        f"bộ câu {DIGEST}; n=4 (+1 article_ref, 1 ngoài phạm vi) |"
    )


def test_results_row_without_clause_level(results):
    row = run_eval.results_row(
        results, clause_level=False, digest=DIGEST, label="dense baseline", day=date(2026, 11, 21)
    )
    # Same article ranks [3, None, 1, 2] as above; the four clause-level scores are unavailable.
    assert row == (
        "| 2026-11-21 | dense baseline | 0,250 | 0,750 | 0,750 | 0,458 | "
        "Mức Khoản: N/A / N/A / N/A / N/A; "
        "R@1 theo nhóm (mức Điều): direct 1/2, paraphrase 0/2; "
        f"bộ câu {DIGEST}; n=4 (+1 article_ref, 1 ngoài phạm vi) |"
    )


def test_baseline_row_matches_results_md():
    """The committed baseline row is what results_row prints for the saved run."""
    questions_path = REPO_ROOT / "eval" / "questions.jsonl"
    run_path = REPO_ROOT / "eval" / "runs" / "2026-11-22-dense-baseline.jsonl"
    saved = {
        row["id"]: row for row in map(json.loads, run_path.read_text(encoding="utf-8").splitlines())
    }
    results = [
        run_eval.QuestionResult(
            question,
            top=[],
            rank_article=saved[question.id]["rank_article"],
            rank_clause=saved[question.id]["rank_clause"],
        )
        for question in load_questions(questions_path)
    ]
    row = run_eval.results_row(
        results,
        clause_level=True,
        digest="577b8392afeb",
        label="dense baseline (bge-m3)",
        day=date(2026, 11, 22),
    )
    results_md = (REPO_ROOT / "eval" / "results.md").read_text(encoding="utf-8")
    assert run_eval.questions_digest(questions_path) == "577b8392afeb"
    lines = results_md.splitlines()
    assert row in lines
    # A stale baseline row left beside the new one must not pass.
    assert sum(line.startswith("| 2026-11-22 |") for line in lines) == 1


def test_questions_digest_is_12_hex_chars(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_bytes(b"abc\n")
    digest = run_eval.questions_digest(path)
    assert digest == hashlib.sha256(b"abc\n").hexdigest()[:12]
    assert len(digest) == 12


def test_main_prints_report_and_row(tmp_path, monkeypatch, capsys, fake):
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    path = write_questions(tmp_path / "q.jsonl", QUESTIONS)
    assert run_eval.main(["--questions", str(path)]) == 0
    out = capsys.readouterr().out
    assert "| " in out
    assert "dense baseline" in out
    assert "Mức Khoản: 0,250 / 0,750 / 0,750 / 0,396" in out


def test_main_missing_gold_ids_warns_and_uses_article_level(
    tmp_path, monkeypatch, capsys, fake, all_chunks
):
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    path = write_questions(tmp_path / "q.jsonl", QUESTIONS)
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in all_chunks[:12]), encoding="utf-8"
    )
    assert run_eval.main(["--questions", str(path), "--chunks", str(partial)]) == 0
    captured = capsys.readouterr()
    assert "clause-level metrics are N/A" in captured.err
    assert "Mức Khoản: N/A / N/A / N/A / N/A" in captured.out


def test_main_empty_questions_returns_1(tmp_path, monkeypatch, capsys, fake):
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    path = tmp_path / "q.jsonl"
    path.write_text("", encoding="utf-8")
    assert run_eval.main(["--questions", str(path)]) == 1
    assert f"no questions in {path}" in capsys.readouterr().err
    assert fake.calls == []


def test_main_invalid_questions_returns_1_with_line(tmp_path, monkeypatch, capsys, fake):
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    path = tmp_path / "q.jsonl"
    path.write_text(question_to_json(Q_DIRECT_1) + "\nnot json\n", encoding="utf-8")
    assert run_eval.main(["--questions", str(path)]) == 1
    assert f"{path}:2:" in capsys.readouterr().err
    assert fake.calls == []


def test_main_failure_mid_run_leaves_no_out_file(tmp_path, monkeypatch, capsys, all_chunks):
    failing = FakeRetriever(MAPPING, all_chunks, fail_after=1)
    monkeypatch.setattr(run_eval, "default_retriever", lambda: failing)
    path = write_questions(tmp_path / "q.jsonl", QUESTIONS)
    out = tmp_path / "out.jsonl"
    code = run_eval.main(["--questions", str(path), "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Qdrant is not reachable" in captured.err
    assert not out.exists()
    assert not (tmp_path / "out.jsonl.tmp").exists()
    assert "| 20" not in captured.out


def test_main_writes_out_jsonl(tmp_path, monkeypatch, fake):
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    path = write_questions(tmp_path / "q.jsonl", QUESTIONS)
    out = tmp_path / "out.jsonl"
    assert run_eval.main(["--questions", str(path), "--out", str(out)]) == 0
    assert not (tmp_path / "out.jsonl.tmp").exists()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(QUESTIONS)
    for row in rows:
        assert set(row) == {"id", "group", "pair_id", "rank_article", "rank_clause", "top"}
        assert all(set(item) == {"id", "score"} for item in row["top"])
    assert rows[0]["id"] == "q001"
    assert rows[0]["rank_article"] == 3
    assert rows[0]["rank_clause"] == 4
    assert rows[0]["top"][0] == {"id": full_id("d36:k2"), "score": 0.9}


@pytest.mark.integration
@pytest.mark.server
@pytest.mark.model
def test_run_eval_against_live_qdrant(tmp_path, capsys):
    """Needs Qdrant with the indexed collection and the embedding model (not a unit test)."""
    questions = [
        make_question(
            "q001",
            "direct",
            "Người lao động muốn nghỉ việc phải báo trước bao nhiêu ngày?",
            ["d35:k1"],
            "p01",
        ),
        make_question(
            "q002",
            "paraphrase",
            "Tôi xin thôi việc thì phải báo cho công ty trước mấy ngày?",
            ["d35:k1"],
            "p01",
        ),
        make_question("q003", "out_of_scope", "Cách nấu phở bò ngon", []),
    ]
    path = write_questions(tmp_path / "q.jsonl", questions)
    assert run_eval.main(["--questions", str(path)]) == 0
    out = capsys.readouterr().out
    for heading in (
        "## 1. Tổng quan",
        "## 2. Bảng kết quả",
        "## 3. Câu trượt ở top 5",
        "## 4. Điểm top-1",
        "## 5. Bảng cặp",
    ):
        assert heading in out
    assert "dense baseline" in out
