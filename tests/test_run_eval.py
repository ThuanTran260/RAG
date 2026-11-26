import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.rag import retriever as retriever_module
from app.rag.retriever import LazyEmbedder, RetrievedChunk, Retriever, default_retriever
from app.rag.store import QdrantUnavailableError
from eval import run_eval
from eval.questions import Question, load_questions, question_to_json
from eval.stats import clopper_pearson, mcnemar
from ingestion.documents import REPO_ROOT
from ingestion.index import index_chunks, index_hybrid_chunks

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
def promotion_off(monkeypatch):
    """A run without flags then measures default_retriever() alone and needs no dense probe."""
    monkeypatch.setenv("ARTICLE_PROMOTION", "false")


@pytest.fixture
def fake(all_chunks, promotion_off):
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
    misses = _section(report, 3)
    # Article ranks: q001 3, q002 None, q003 1, q005 1, q006 2. Every rank but 1 is a miss.
    assert [line.split()[1] for line in misses if line.startswith("- ")] == ["q001", "q002", "q006"]
    assert "## 3. Câu trượt ở hạng 1 (mức Điều)" in report
    assert any("d36:k2" in line for line in misses)  # the first retrieved chunk of q001 and q002
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


def test_main_failure_mid_run_leaves_no_out_file(
    tmp_path, monkeypatch, capsys, all_chunks, promotion_off
):
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
        "## 3. Câu trượt ở hạng 1",
        "## 4. Điểm top-1",
        "## 5. Bảng cặp",
    ):
        assert heading in out
    assert "dense + promotion" in out


# --- retrieval configurations ------------------------------------------------------

RUN_KEYS = {"id", "group", "pair_id", "rank_article", "rank_clause", "top"}
EXTRA_KEYS = {"fusion_score", "dense_rank", "sparse_rank"}
META_KEYS = {
    "questions_digest",
    "collection",
    "collection_metadata",
    "mode",
    "k",
    "rrf_k",
    "promotion",
    "device",
    "git_sha",
    "date",
}
STORED_METADATA = {"tokenizer_version": "vi-syl-bigram-v1", "points": 676}
# The cosine of the dense rank-1 chunk, unlike any score a fake main retriever returns.
PROBE_SCORES = {
    Q_DIRECT_1.question: 0.81,
    Q_PARA_1.question: 0.72,
    Q_REF.question: 0.93,
    Q_OOS.question: 0.85,
    Q_DIRECT_2.question: 0.77,
    Q_PARA_2.question: 0.79,
}


class ScriptedRetriever:
    """Answers each question with (short id, score, fusion score, dense rank, sparse rank)."""

    def __init__(self, answers, chunks):
        self.answers = answers
        self.by_id = {chunk.id: chunk for chunk in chunks}
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query, k=5):
        self.calls.append((query, k))
        return [
            RetrievedChunk(
                chunk=self.by_id[full_id(short)],
                score=score,
                fusion_score=fusion,
                dense_rank=dense,
                sparse_rank=sparse,
            )
            for short, score, fusion, dense, sparse in self.answers[query]
        ]


def probe_answers():
    return {
        question: [("d37:k1", score, None, None, None)] for question, score in PROBE_SCORES.items()
    }


def low_answers(fused=False):
    """One answer per question whose first chunk is not the dense top-1 (its cosine is 0.11)."""
    extra = (0.03, 2, 1) if fused else (None, None, None)
    return {q.question: [("d36:k2", 0.11, *extra)] for q in QUESTIONS}


class Wiring:
    """Stands in for default_retriever() and Retriever and records what run_eval builds."""

    def __init__(self, chunks, main_answers):
        self.chunks = chunks
        self.main_answers = main_answers
        self.metadata_reads: list[str] = []
        self.base_calls = 0
        self.built: list[SimpleNamespace] = []
        # The default retriever: only its client and embedder are used once a flag is given.
        self.base = SimpleNamespace(client=self, embedder=SimpleNamespace(device="cuda:0"))

    def get_collection(self, name):
        self.metadata_reads.append(name)
        return SimpleNamespace(config=SimpleNamespace(metadata=dict(STORED_METADATA)))

    def default_retriever(self):
        self.base_calls += 1
        return self.base

    def retriever(self, client, embedder, collection, mode="dense", *, promote=False):
        # The second, plain dense retriever of a run is the one that scores the dense top-1.
        is_probe = bool(self.built) and mode == "dense" and not promote
        answers = probe_answers() if is_probe else self.main_answers
        fake = ScriptedRetriever(answers, self.chunks)
        built = SimpleNamespace(
            client=client,
            embedder=embedder,
            collection=collection,
            mode=mode,
            promote=promote,
            fake=fake,
            retrieve=fake.retrieve,
        )
        self.built.append(built)
        return built


@pytest.fixture
def wire(monkeypatch, all_chunks):
    def _wire(main_answers):
        wiring = Wiring(all_chunks, main_answers)
        monkeypatch.setattr(run_eval, "default_retriever", wiring.default_retriever)
        monkeypatch.setattr(run_eval, "Retriever", wiring.retriever)
        return wiring

    return _wire


@pytest.fixture
def questions_path(tmp_path):
    return write_questions(tmp_path / "q.jsonl", QUESTIONS)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_meta(path):
    return json.loads(path.read_text(encoding="utf-8"))


def label_of(output):
    """The label cell of the results.md row, the last line of the output."""
    return output.strip().splitlines()[-1].split("|")[2].strip()


def test_without_promotion_a_run_without_flags_uses_only_the_default_retriever(
    monkeypatch, capsys, fake, questions_path
):
    built = []
    monkeypatch.setattr(run_eval, "default_retriever", lambda: built.append(1) or fake)
    monkeypatch.setattr(run_eval, "Retriever", lambda *a, **k: pytest.fail("a Retriever was built"))

    assert run_eval.main(["--questions", str(questions_path)]) == 0

    assert built == [1]
    assert fake.calls == [(q.question, 10) for q in QUESTIONS]
    out = capsys.readouterr().out
    assert label_of(out) == "dense baseline"
    overview = _section(out, 1)
    assert f"Collection: {Settings().collection_name}" in overview
    assert "Chế độ truy xuất: dense" in overview
    assert "Promotion số Điều: tắt" in overview
    # Every rank-1 score of the default run is 0.9, and no run is compared.
    assert "Trong phạm vi: n=5, thấp nhất 0,900" in out
    assert "## 7." not in out


# flags, mode, promotion, hybrid collection, dense probe, label
CONFIGURATIONS = [
    (["--retriever", "dense"], "dense", False, False, False, "dense baseline"),
    (["--collection", "default"], "dense", False, False, False, "dense baseline"),
    (["--collection", "hybrid"], "dense", False, True, False, "dense, hybrid collection"),
    (["--retriever", "sparse"], "sparse", False, True, True, "sparse"),
    (["--retriever", "hybrid"], "hybrid", False, True, True, "hybrid"),
    (["--retriever", "hybrid", "--collection", "hybrid"], "hybrid", False, True, True, "hybrid"),
    (["--promote"], "dense", True, False, True, "dense + promotion"),
    (
        ["--collection", "hybrid", "--promote"],
        "dense",
        True,
        True,
        True,
        "dense + promotion, hybrid collection",
    ),
    (["--retriever", "hybrid", "--promote"], "hybrid", True, True, True, "hybrid + promotion"),
]


@pytest.mark.parametrize(
    ("flags", "mode", "promotion", "hybrid", "probed", "label"), CONFIGURATIONS
)
def test_flags_build_the_configured_retrievers(
    wire, capsys, questions_path, flags, mode, promotion, hybrid, probed, label
):
    wiring = wire(low_answers())
    settings = Settings()
    collection = settings.hybrid_collection_name if hybrid else settings.collection_name

    assert run_eval.main(["--questions", str(questions_path), *flags]) == 0

    out = capsys.readouterr().out
    assert wiring.base_calls == 1
    assert label_of(out) == label
    overview = _section(out, 1)
    assert f"Collection: {collection}" in overview
    assert f"Chế độ truy xuất: {mode}" in overview
    assert f"Promotion số Điều: {'bật' if promotion else 'tắt'}" in overview
    assert wiring.metadata_reads == []  # only the meta file needs it
    expected = [(collection, mode, promotion)]
    if probed:
        expected.append((collection, "dense", False))
    assert [(b.collection, b.mode, b.promote) for b in wiring.built] == expected
    for built in wiring.built:
        assert built.client is wiring.base.client
        assert built.embedder is wiring.base.embedder  # so the model loads once
        assert built.fake.calls == [(q.question, 10) for q in QUESTIONS]


def test_a_label_given_by_hand_wins(wire, capsys, questions_path):
    wire(low_answers())
    code = run_eval.main(
        ["--questions", str(questions_path), "--retriever", "hybrid", "--label", "thử"]
    )
    assert code == 0
    assert label_of(capsys.readouterr().out) == "thử"


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (
            ["--retriever", "sparse", "--collection", "default"],
            "--retriever sparse needs --collection hybrid",
        ),
        (
            ["--retriever", "hybrid", "--collection", "default"],
            "--retriever hybrid needs --collection hybrid",
        ),
        (["--retriever", "sparse", "--promote"], "--promote has no effect with --retriever sparse"),
    ],
)
def test_impossible_flag_combinations_are_rejected_before_any_retrieval(
    wire, capsys, questions_path, flags, message
):
    wiring = wire(low_answers())

    with pytest.raises(SystemExit) as stopped:
        run_eval.main(["--questions", str(questions_path), *flags])

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    assert (wiring.base_calls, wiring.built) == (0, [])


@pytest.mark.parametrize(
    ("flags", "fused"),
    [(["--retriever", "sparse"], False), (["--retriever", "hybrid"], True), (["--promote"], False)],
    ids=["sparse", "hybrid", "promoted"],
)
def test_top_1_scores_are_the_dense_top_1_cosines(wire, capsys, questions_path, flags, fused):
    wire(low_answers(fused))

    assert run_eval.main(["--questions", str(questions_path), *flags]) == 0

    # PROBE_SCORES in scope: 0.81, 0.72, 0.93, 0.77, 0.79; out of scope: 0.85.
    assert _section(capsys.readouterr().out, 4) == [
        "Trong phạm vi: n=5, thấp nhất 0,720, trung vị 0,790, cao nhất 0,930",
        "Ngoài phạm vi: n=1, thấp nhất 0,850, trung vị 0,850, cao nhất 0,850",
        "Câu ngoài phạm vi có điểm cao hơn điểm thấp nhất trong phạm vi: 1/1",
    ]


def test_a_question_without_a_hit_still_has_a_dense_top_1_score(wire, capsys, questions_path):
    answers = low_answers()
    answers[Q_PARA_1.question] = []  # BM25 found nothing
    wire(answers)

    assert run_eval.main(["--questions", str(questions_path), "--retriever", "sparse"]) == 0

    out = capsys.readouterr().out
    assert "Trong phạm vi: n=5, thấp nhất 0,720" in out
    misses = _section(out, 3)
    after_question = misses[misses.index(f"- q002 [paraphrase] {Q_PARA_1.question}") + 1]
    assert after_question == "  Điều đúng: 35; top 3: (không có)"


def test_a_dense_run_file_keeps_todays_keys(wire, tmp_path, questions_path):
    wire(low_answers())
    out = tmp_path / "dense.jsonl"

    code = run_eval.main(
        ["--questions", str(questions_path), "--collection", "hybrid", "--out", str(out)]
    )

    assert code == 0
    for row in read_rows(out):
        assert set(row) == RUN_KEYS
        assert all(set(item) == {"id", "score"} for item in row["top"])


def test_a_failure_while_rendering_leaves_no_output(wire, tmp_path, questions_path, monkeypatch):
    wire(low_answers())
    out = tmp_path / "dense.jsonl"

    def broken(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(run_eval, "render_report", broken)

    with pytest.raises(RuntimeError, match="render failed"):
        run_eval.main(
            ["--questions", str(questions_path), "--collection", "hybrid", "--out", str(out)]
        )

    assert not out.exists()
    assert not run_eval.meta_path(out).exists()


def test_a_hybrid_run_file_adds_the_keys_that_are_set(wire, tmp_path, questions_path):
    answers = low_answers(fused=True)
    # A chunk without a BM25 hit, and a question that fell back to the dense ranking.
    answers[Q_DIRECT_1.question] = [("d36:k2", 0.5, 0.03, 2, 1), ("d35:k1", 0.6, 0.016, 1, None)]
    answers[Q_PARA_1.question] = [("d35:k1", 0.6, None, 1, None)]
    wire(answers)
    out = tmp_path / "hybrid.jsonl"

    code = run_eval.main(
        ["--questions", str(questions_path), "--retriever", "hybrid", "--out", str(out)]
    )

    assert code == 0
    rows = read_rows(out)
    assert all(set(row) == RUN_KEYS for row in rows)
    assert rows[0]["top"] == [
        {
            "id": full_id("d36:k2"),
            "score": 0.5,
            "fusion_score": 0.03,
            "dense_rank": 2,
            "sparse_rank": 1,
        },
        {"id": full_id("d35:k1"), "score": 0.6, "fusion_score": 0.016, "dense_rank": 1},
    ]
    assert rows[1]["top"] == [{"id": full_id("d35:k1"), "score": 0.6, "dense_rank": 1}]
    assert set(rows[2]["top"][0]) == {"id", "score"} | EXTRA_KEYS


def head_commit():
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def test_the_meta_file_of_a_run_without_promotion(monkeypatch, tmp_path, fake, questions_path):
    fake.embedder = SimpleNamespace(device="cpu")
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    out = tmp_path / "run.jsonl"

    assert run_eval.main(["--questions", str(questions_path), "--out", str(out)]) == 0

    meta = read_meta(tmp_path / "run.meta.json")
    assert set(meta) == META_KEYS
    assert meta == {
        "questions_digest": run_eval.questions_digest(questions_path),
        "collection": Settings().collection_name,
        "collection_metadata": None,
        "mode": "dense",
        "k": 10,
        "rrf_k": None,
        "promotion": False,
        "device": "cpu",
        "git_sha": head_commit(),
        "date": date.today().isoformat(),
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "q.jsonl",
        "run.jsonl",
        "run.meta.json",
    ]


@pytest.mark.parametrize(
    ("flags", "mode", "rrf_k", "promotion"),
    [
        (["--collection", "hybrid"], "dense", None, False),
        (["--retriever", "sparse"], "sparse", None, False),
        (["--retriever", "hybrid"], "hybrid", 60, False),
        (["--retriever", "hybrid", "--promote"], "hybrid", 60, True),
    ],
)
def test_the_meta_file_of_a_run_on_the_hybrid_collection(
    wire, tmp_path, questions_path, flags, mode, rrf_k, promotion
):
    wiring = wire(low_answers())
    out = tmp_path / "run.jsonl"

    code = run_eval.main(["--questions", str(questions_path), "--out", str(out), *flags])

    assert code == 0
    collection = Settings().hybrid_collection_name
    meta = read_meta(tmp_path / "run.meta.json")
    assert set(meta) == META_KEYS
    assert meta["collection"] == collection
    assert meta["collection_metadata"] == STORED_METADATA
    assert wiring.metadata_reads == [collection]
    assert (meta["mode"], meta["rrf_k"], meta["promotion"]) == (mode, rrf_k, promotion)
    assert meta["device"] == "cuda:0"


def test_the_meta_file_never_holds_a_secret(monkeypatch, tmp_path, fake, questions_path):
    monkeypatch.setenv("LLM_API_KEY", "SECRET-MARKER")
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    out = tmp_path / "run.jsonl"

    assert run_eval.main(["--questions", str(questions_path), "--out", str(out)]) == 0

    assert "SECRET-MARKER" not in (tmp_path / "run.meta.json").read_text(encoding="utf-8")


def test_a_meta_file_that_cannot_be_written_leaves_no_output(
    monkeypatch, tmp_path, capsys, fake, questions_path
):
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    (tmp_path / "run.meta.json").mkdir()  # the meta path is taken by a directory
    out = tmp_path / "run.jsonl"

    assert run_eval.main(["--questions", str(questions_path), "--out", str(out)]) == 1

    assert "cannot read or write a file" in capsys.readouterr().err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["q.jsonl", "run.meta.json"]


# Local mode searches exactly by brute force and warns that the exact=True flag changes nothing.
@pytest.mark.filterwarnings("ignore:Local mode performs exact:UserWarning")
@pytest.mark.parametrize("promote", [False, True])
def test_a_hybrid_run_through_the_real_retriever(
    monkeypatch, tmp_path, capsys, all_chunks, fake_embedder, questions_path, promote
):
    wanted = {full_id(short) for ids in MAPPING.values() for short in ids}
    chunks = [chunk for chunk in all_chunks if chunk.id in wanted]
    client = QdrantClient(":memory:")
    settings = Settings()
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in chunks])
    metadata = index_hybrid_chunks(
        chunks,
        vectors,
        client,
        settings.hybrid_collection_name,
        model_name=fake_embedder.model_name,
        chunks_sha256="0" * 64,
    )
    fake_embedder.device = "cpu"
    base = Retriever(client, LazyEmbedder(lambda: fake_embedder), settings.collection_name)
    monkeypatch.setattr(run_eval, "default_retriever", lambda: base)
    out = tmp_path / "run.jsonl"
    flags = ["--retriever", "hybrid", *(["--promote"] if promote else [])]

    code = run_eval.main(["--questions", str(questions_path), "--out", str(out), *flags])

    captured = capsys.readouterr()
    assert code == 0, captured.err
    meta = read_meta(tmp_path / "run.meta.json")
    assert meta["collection_metadata"] == metadata
    assert meta["device"] == "cpu"
    rows = read_rows(out)
    assert any("fusion_score" in item and "sparse_rank" in item for item in rows[2]["top"])
    # q003 names Điều 25, so promotion puts a chunk of it first.
    if promote:
        assert rows[2]["rank_article"] == 1
    dense = Retriever(client, fake_embedder, settings.hybrid_collection_name)
    best = [dense.retrieve(q.question, 10)[0].score for q in QUESTIONS if q.group != "out_of_scope"]
    assert f"Trong phạm vi: n=5, thấp nhất {run_eval.format_score(min(best))}" in captured.out
    client.close()


# --- the shipped configuration: a run without flags -------------------------------


@pytest.fixture
def wire_shipped(wire):
    """Like ``wire``, for a run without flags: the default retriever itself is measured."""

    def _wire(answers):
        wiring = wire(probe_answers())  # the only retriever such a run builds is the probe
        measured = ScriptedRetriever(answers, wiring.chunks)
        wiring.base.retrieve = measured.retrieve
        return wiring, measured

    return _wire


# environment, mode, promotion, hybrid collection, dense probe, rrf_k, label
SHIPPED = [
    ({}, "dense", True, False, True, None, "dense + promotion"),
    ({"ARTICLE_PROMOTION": "false"}, "dense", False, False, False, None, "dense baseline"),
    ({"RETRIEVAL_MODE": "hybrid"}, "hybrid", True, True, True, 60, "hybrid + promotion"),
    (
        {"RETRIEVAL_MODE": "hybrid", "ARTICLE_PROMOTION": "false"},
        *("hybrid", False, True, True, 60, "hybrid"),
    ),
    (
        {"RETRIEVAL_MODE": "sparse", "ARTICLE_PROMOTION": "false"},
        *("sparse", False, True, True, None, "sparse"),
    ),
]


@pytest.mark.parametrize(
    ("environment", "mode", "promotion", "hybrid", "probed", "rrf_k", "label"),
    SHIPPED,
    ids=["defaults", "promotion off", "hybrid", "hybrid without promotion", "sparse"],
)
def test_a_run_without_flags_measures_the_configuration_of_the_settings(
    wire_shipped,
    monkeypatch,
    tmp_path,
    capsys,
    questions_path,
    environment,
    mode,
    promotion,
    hybrid,
    probed,
    rrf_k,
    label,
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    wiring, measured = wire_shipped(low_answers(fused=mode == "hybrid"))
    settings = Settings()
    collection = settings.hybrid_collection_name if hybrid else settings.collection_name
    out = tmp_path / "run.jsonl"

    assert run_eval.main(["--questions", str(questions_path), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert wiring.base_calls == 1
    assert measured.calls == [(q.question, 10) for q in QUESTIONS]
    expected = [(collection, "dense", False)] if probed else []
    assert [(b.collection, b.mode, b.promote) for b in wiring.built] == expected
    for built in wiring.built:
        assert built.client is wiring.base.client
        assert built.embedder is wiring.base.embedder  # so the model loads once
        assert built.fake.calls == [(q.question, 10) for q in QUESTIONS]
    overview = _section(printed, 1)
    assert f"Collection: {collection}" in overview
    assert f"Chế độ truy xuất: {mode}" in overview
    assert f"Promotion số Điều: {'bật' if promotion else 'tắt'}" in overview
    assert label_of(printed) == label
    # Section 4 shows the dense top-1 cosines of the probe; without one, the run's own (0.11).
    lowest = "0,720" if probed else "0,110"
    assert _section(printed, 4)[0].startswith(f"Trong phạm vi: n=5, thấp nhất {lowest}")
    meta = read_meta(tmp_path / "run.meta.json")
    assert (meta["mode"], meta["promotion"], meta["rrf_k"]) == (mode, promotion, rrf_k)
    assert meta["collection"] == collection
    assert meta["collection_metadata"] == (STORED_METADATA if hybrid else None)


def test_a_run_without_flags_describes_the_shipped_run_in_section_7(
    wire_shipped, tmp_path, capsys, questions_path
):
    wire_shipped(low_answers())
    reference = write_run(tmp_path / "a.jsonl", [run_row(q) for q in QUESTIONS])

    code = run_eval.main(["--questions", str(questions_path), "--reference", str(reference)])

    assert code == 0
    assert "B (run này): mode dense, promotion bật" in capsys.readouterr().out


# Local mode searches exactly by brute force and warns that the exact=True flag changes nothing.
@pytest.mark.filterwarnings("ignore:Local mode performs exact:UserWarning")
def test_a_run_without_flags_through_the_real_default_retriever(
    monkeypatch, tmp_path, capsys, all_chunks, fake_embedder, questions_path
):
    wanted = {full_id(short) for ids in MAPPING.values() for short in ids}
    chunks = [chunk for chunk in all_chunks if chunk.id in wanted]
    client = QdrantClient(":memory:")
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in chunks])
    index_chunks(chunks, vectors, client, Settings().collection_name)
    fake_embedder.device = "cpu"
    monkeypatch.setattr(retriever_module, "connect", lambda _settings: client)
    monkeypatch.setattr(retriever_module, "build_embedder", lambda _settings: fake_embedder)
    out = tmp_path / "run.jsonl"

    default_retriever.cache_clear()
    try:
        code = run_eval.main(["--questions", str(questions_path), "--out", str(out)])
    finally:
        default_retriever.cache_clear()

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert label_of(captured.out) == "dense + promotion"
    meta = read_meta(tmp_path / "run.meta.json")
    assert (meta["mode"], meta["promotion"], meta["device"]) == ("dense", True, "cpu")
    assert meta["collection"] == Settings().collection_name
    # q003 names Điều 25, so promotion puts a chunk of it first.
    assert read_rows(out)[2]["rank_article"] == 1
    client.close()


def test_a_run_without_flags_rejects_sparse_with_promotion(
    wire, monkeypatch, capsys, questions_path
):
    monkeypatch.setenv("RETRIEVAL_MODE", "sparse")
    wiring = wire(low_answers())

    assert run_eval.main(["--questions", str(questions_path)]) == 1

    captured = capsys.readouterr()
    assert "RETRIEVAL_MODE=sparse" in captured.err
    assert "ARTICLE_PROMOTION=false" in captured.err
    assert (captured.out, wiring.base_calls, wiring.built) == ("", 0, [])


# The retrieval flags of the five commands that T6 ran, by the run each one saved.
T6_COMMANDS = {
    "r1-dense": ["--collection", "hybrid"],
    "r2-sparse": ["--retriever", "sparse"],
    "r3-hybrid": ["--retriever", "hybrid"],
    "r4-dense-promo": ["--collection", "hybrid", "--promote"],
    "r5-hybrid-promo": ["--retriever", "hybrid", "--promote"],
}


@pytest.mark.parametrize(
    "environment",
    [{}, {"RETRIEVAL_MODE": "hybrid"}, {"ARTICLE_PROMOTION": "false"}],
    ids=["default settings", "hybrid settings", "promotion off"],
)
@pytest.mark.parametrize("run", T6_COMMANDS)
def test_the_t6_commands_still_configure_their_runs(
    wire, monkeypatch, tmp_path, questions_path, run, environment
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    wire(low_answers())
    out = tmp_path / "run.jsonl"

    code = run_eval.main(["--questions", str(questions_path), "--out", str(out), *T6_COMMANDS[run]])

    assert code == 0
    recorded = read_meta(REPO_ROOT / "eval" / "runs" / f"2026-11-27-{run}.meta.json")
    meta = read_meta(tmp_path / "run.meta.json")
    for key in ("collection", "mode", "rrf_k", "promotion"):
        assert meta[key] == recorded[key], key


# --- comparison with a reference run ----------------------------------------------


def numbered(group, number, text=None, gold="d35:k1"):
    """A synthetic question; direct and paraphrase ones need a pair id."""
    pair = f"p{number % 100:02d}" if group in ("direct", "paraphrase") else None
    return make_question(
        f"q{number:03d}",
        group,
        text or f"câu thử số {number}",
        [] if group == "out_of_scope" else [gold],
        pair,
    )


def item(short, score=0.5, fusion=None):
    entry = {"id": full_id(short), "score": score}
    if fusion is not None:
        entry["fusion_score"] = fusion
    return entry


def run_row(question, rank=1, clause=None, top=()):
    return {
        "id": question.id,
        "group": question.group,
        "pair_id": question.pair_id,
        "rank_article": rank,
        "rank_clause": rank if clause is None else clause,
        "top": list(top),
    }


def write_run(path, rows, **meta):
    """Write a run file, and its meta file when any meta value is given."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    if meta:
        run_eval.meta_path(path).write_text(json.dumps(meta), encoding="utf-8")
    return path


def load_pair(tmp_path, rows_a, rows_b):
    """Two run files, written and read back."""
    return [
        run_eval.load_reference(write_run(tmp_path / name, rows))
        for name, rows in (("a.jsonl", rows_a), ("b.jsonl", rows_b))
    ]


def runs(tmp_path, questions, ranks_a, ranks_b):
    """Two run files with the given article ranks, read back."""
    return load_pair(
        tmp_path,
        *[[run_row(q, rank) for q, rank in zip(questions, ranks)] for ranks in (ranks_a, ranks_b)],
    )


def compare(questions, reference, run_b, **options):
    options = {"mode_b": "dense", "promotion_b": False, "clause_level": True} | options
    return run_eval.render_comparison(questions, reference, run_b.rows, **options)


def line_with(text, start):
    return next(line for line in text.splitlines() if line.startswith(start))


def test_meta_path_sits_beside_the_run_file(tmp_path):
    assert (
        run_eval.meta_path(tmp_path / "2026-11-26-x.jsonl") == tmp_path / "2026-11-26-x.meta.json"
    )


def test_a_reference_without_a_meta_file_counts_as_dense(tmp_path):
    question = numbered("direct", 1)
    path = write_run(tmp_path / "a.jsonl", [run_row(question, 2, 3, [item("d35:k1", 0.7)])])

    reference = run_eval.load_reference(path)

    assert (reference.mode, reference.digest) == ("dense", None)
    row = reference.rows["q001"]
    assert (row.group, row.rank_article, row.rank_clause) == ("direct", 2, 3)
    assert (row.top[0].id, row.top[0].score) == (full_id("d35:k1"), 0.7)
    assert row.top[0].fusion_score is None


def test_a_reference_reads_the_mode_and_digest_of_its_meta_file(tmp_path):
    path = write_run(
        tmp_path / "a.jsonl",
        [run_row(numbered("direct", 1))],
        mode="hybrid",
        questions_digest=DIGEST,
    )
    reference = run_eval.load_reference(path)
    assert (reference.mode, reference.digest) == ("hybrid", DIGEST)


GOOD_LINE = json.dumps(run_row(numbered("direct", 1))) + "\n"


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("not json\n", "a.jsonl:1:"),
        ('{"id": "q001", "group": "direct"}\n', "a.jsonl:1:"),
        (GOOD_LINE + "[]\n", "a.jsonl:2:"),
        (GOOD_LINE * 2, "duplicate question id q001"),
    ],
)
def test_a_broken_reference_is_reported_with_its_line(tmp_path, content, reason):
    path = tmp_path / "a.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(run_eval.ReferenceRunError, match=reason):
        run_eval.load_reference(path)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"questions_digest": "abc"}',
        '{"mode": "bm25", "questions_digest": "abc"}',
        '{"mode": "dense"}',
    ],
)
def test_a_broken_reference_meta_file_is_reported(tmp_path, content):
    path = write_run(tmp_path / "a.jsonl", [run_row(numbered("direct", 1))])
    run_eval.meta_path(path).write_text(content, encoding="utf-8")
    with pytest.raises(run_eval.ReferenceRunError, match="a.meta.json"):
        run_eval.load_reference(path)


def test_comparison_prints_r1_per_group_and_the_article_ref_ranks(tmp_path):
    questions = [
        numbered("direct", 1),
        numbered("direct", 2),
        numbered("paraphrase", 3),
        numbered("article_ref", 4, "Điều 35 nói gì?"),
        numbered("out_of_scope", 5),
    ]
    reference, run_b = runs(tmp_path, questions, [1, 2, 3, None, None], [1, 1, 1, 2, None])

    text = compare(questions, reference, run_b)

    assert "## 7. So với run tham chiếu" in text
    assert line_with(text, "direct:") == "direct: A 1/2, B 2/2"
    assert line_with(text, "paraphrase:") == "paraphrase: A 0/1, B 1/1"
    assert line_with(text, "article_ref:") == "article_ref: A 0/1, B 0/1"
    assert line_with(text, "  q004:") == (
        "  q004: hạng Điều A ngoài top 10, B 2; hạng Khoản A ngoài top 10, B 2"
    )


def test_comparison_clause_ranks_are_na_without_gold_ids(tmp_path):
    questions = [numbered("article_ref", 4, "Điều 35 nói gì?")]
    reference, run_b = runs(tmp_path, questions, [3], [1])
    text = compare(questions, reference, run_b, clause_level=False)
    assert line_with(text, "  q004:") == "  q004: hạng Điều A 3, B 1; hạng Khoản N/A"
    assert line_with(text, "direct:") == "direct: A 0/0, B 0/0"


@pytest.mark.parametrize(
    ("b", "c", "c1", "alternative"),
    [
        (3, 1, "không", "không"),  # exact p = 10/16, mid-p = 6/16
        (4, 1, "không", "có"),  # exact p = 12/32, mid-p = 7/32
        (6, 0, "có", "có"),  # exact p = 2/64, mid-p = 1/64
        (7, 1, "có", "có"),  # exact p = 18/256 > 0.05, mid-p = 10/256 <= 0.05: C1 reads the mid-p
        (5, 2, "không", "không"),  # b - c = 3, but c = 2 > 1
        (0, 6, "không", "không"),  # the same p values, but B lost
    ],
)
def test_comparison_mcnemar_matches_eval_stats(tmp_path, b, c, c1, alternative):
    # b gains, c losses, two questions right in both and one wrong in both.
    ranks_a = [2] * b + [1] * c + [1, 1, 4]
    ranks_b = [1] * b + [2] * c + [1, 1, None]
    questions = [numbered("paraphrase", n) for n in range(1, len(ranks_a) + 1)]
    reference, run_b = runs(tmp_path, questions, ranks_a, ranks_b)

    text = compare(questions, reference, run_b)

    expected = mcnemar(b, c)
    assert line_with(text, "n = ") == (
        f"n = {len(questions)}; b = {b} (B đúng hạng 1, A sai); c = {c} (A đúng hạng 1, B sai)"
    )
    assert line_with(text, "p chính xác") == (
        f"p chính xác = {expected.exact_p:.6f}; mid-p = {expected.mid_p:.6f}".replace(".", ",")
    )
    assert line_with(text, "C1 ") == f"C1 (mid-p ≤ 0,05 và b > c): {c1}"
    assert line_with(text, "b − c ≥ 3") == f"b − c ≥ 3 và c ≤ 1: {alternative}"
    assert "chốt" not in text  # no adoption verdict: T6 decides


def test_comparison_mcnemar_hand_checked_p_values(tmp_path):
    questions = [numbered("paraphrase", n) for n in range(1, 5)]
    reference, run_b = runs(tmp_path, questions, [2, 2, 2, 1], [1, 1, 1, 2])
    text = compare(questions, reference, run_b)
    assert "p chính xác = 0,625000; mid-p = 0,375000" in text


def test_comparison_without_paraphrase_questions_says_so(tmp_path):
    questions = [numbered("direct", 1)]
    reference, run_b = runs(tmp_path, questions, [1], [1])
    assert "Không có câu paraphrase" in compare(questions, reference, run_b)


def guard_of(tmp_path, ranks_a, ranks_b):
    questions = [numbered("direct", n) for n in range(1, len(ranks_a) + 1)]
    reference, run_b = runs(tmp_path, questions, ranks_a, ranks_b)
    return run_eval.guard_check(questions, reference.rows, run_b.rows)


def test_guard_with_nothing_lost_out_of_44(tmp_path):
    # q045 was not at rank 1 in A, so it is not in D.
    guard = guard_of(tmp_path, [1] * 44 + [2], [1] * 45)
    assert (guard.size, guard.lost, guard.left_top5, guard.holds) == (44, [], [], True)
    assert guard.upper == pytest.approx(0.0804, abs=5e-5)


def test_guard_tolerates_one_lost_question(tmp_path):
    guard = guard_of(tmp_path, [1, 1, 1], [5, 1, 1])
    assert (guard.size, guard.lost, guard.left_top5, guard.holds) == (3, ["q001"], [], True)
    assert guard.upper == clopper_pearson(1, 3, 0.95)[1]


def test_guard_fails_on_two_lost_questions(tmp_path):
    guard = guard_of(tmp_path, [1, 1, 1], [2, 2, 1])
    assert (guard.lost, guard.left_top5, guard.holds) == (["q001", "q002"], [], False)


def test_guard_fails_when_a_question_leaves_the_top_5(tmp_path):
    # q001: A 3 -> B 6 leaves; q002: A 5 -> B None leaves; q003: A 6 -> B None was never inside;
    # q004: A None -> B 7 was never inside; q005: A 2 -> B 5 stays.
    guard = guard_of(tmp_path, [3, 5, 6, None, 2], [6, None, None, 7, 5])
    assert (guard.size, guard.lost, guard.holds) == (0, [], False)
    assert guard.left_top5 == ["q001", "q002"]
    assert guard.upper is None  # D is empty


def test_guard_ignores_other_groups(tmp_path):
    questions = [numbered("paraphrase", 1), numbered("article_ref", 2, "Điều 35 nói gì?")]
    reference, run_b = runs(tmp_path, questions, [1, 1], [None, None])
    guard = run_eval.guard_check(questions, reference.rows, run_b.rows)
    assert (guard.size, guard.lost, guard.left_top5, guard.holds) == (0, [], [], True)


def test_comparison_prints_the_guard(tmp_path):
    questions = [numbered("direct", n) for n in range(1, 4)]
    reference, run_b = runs(tmp_path, questions, [1, 1, 4], [3, 1, None])

    text = compare(questions, reference, run_b)

    assert line_with(text, "|D| = ") == "|D| = 2 (câu direct có Điều đúng ở hạng 1 trong A)"
    assert line_with(text, "Mất hạng 1") == "Mất hạng 1 ở B: 1/2: q001 (1 → 3)"
    upper = f"{clopper_pearson(1, 2, 0.95)[1]:.4f}".replace(".", ",")
    assert line_with(text, "Cận trên") == (
        f"Cận trên Clopper–Pearson hai phía 95 % của tỷ lệ mất: {upper}"
    )
    assert line_with(text, "Rời top 5") == (
        "Rời top 5 (direct, A trong top 5, B ngoài): 1: q003 (4 → ngoài top 10)"
    )
    assert line_with(text, "Guard đạt") == (
        "Guard đạt (mất tối đa 1 và không câu nào rời top 5): không"
    )


def test_comparison_prints_the_guard_with_an_empty_d(tmp_path):
    questions = [numbered("direct", 1)]
    reference, run_b = runs(tmp_path, questions, [2], [2])
    text = compare(questions, reference, run_b)
    assert line_with(text, "Mất hạng 1") == "Mất hạng 1 ở B: 0/0: (không có)"
    assert line_with(text, "Cận trên").endswith("tỷ lệ mất: N/A")


def test_rank_regressions_count_a_missing_article_as_rank_11(tmp_path):
    questions = [numbered("direct", n) for n in range(1, 6)]
    reference, run_b = runs(tmp_path, questions, [10, None, 1, 3, None], [None, None, 2, 3, 10])

    found = run_eval.rank_regressions(
        questions, reference.rows, run_b.rows, mode_b="dense", promotion_b=False
    )

    assert [(r.question_id, r.rank_a, r.rank_b, r.cause) for r in found] == [
        ("q001", 10, None, "khác"),
        ("q003", 1, 2, "khác"),
    ]


def regression_of(tmp_path, top_b, *, rank=2, text=None, mode="hybrid", promotion=False):
    """The one regression of a question (gold Điều 35) that was at rank 1 in run A."""
    question = numbered("direct", 1, text)
    reference, run_b = load_pair(
        tmp_path, [run_row(question, 1)], [run_row(question, rank, top=top_b)]
    )
    found = run_eval.rank_regressions(
        [question], reference.rows, run_b.rows, mode_b=mode, promotion_b=promotion
    )
    assert len(found) == 1
    return found[0]


NEAR = [item("d36:k2", fusion=0.03005), item("d35:k1", fusion=0.03)]  # gap 5e-5
FAR = [item("d36:k2", fusion=0.0302), item("d35:k1", fusion=0.03)]  # gap 2e-4


def test_cause_parser_when_the_parser_names_other_articles(tmp_path):
    found = regression_of(tmp_path, NEAR, text="Điều 36 nói gì?", promotion=True)
    assert (found.cause, found.detail) == ("parser", "article_refs = {36}, gold = {35}")


def test_cause_parser_when_the_parser_finds_nothing(tmp_path):
    found = regression_of(tmp_path, NEAR, mode="dense", promotion=True)
    assert (found.cause, found.detail) == ("parser", "article_refs = ∅, gold = {35}")


def test_a_correct_parser_moves_on_to_the_next_cause(tmp_path):
    hybrid = regression_of(tmp_path, NEAR, text="Điều 35 nói gì?", promotion=True)
    dense = regression_of(tmp_path, NEAR, text="Điều 35 nói gì?", mode="dense", promotion=True)
    assert (hybrid.cause, dense.cause) == ("gần hòa điểm", "khác")


def test_cause_parser_needs_a_run_with_promotion(tmp_path):
    assert regression_of(tmp_path, FAR, text="Điều 36 nói gì?").cause == "chunk khác trùng từ khóa"


def test_cause_near_tie_prints_the_gap_to_the_chunk_above(tmp_path):
    # The chunk at rank 1 is far ahead; only the one directly above rank 3 counts.
    top = [item("d37:k1", fusion=0.032), item("d36:k2", fusion=0.03005)]
    found = regression_of(tmp_path, [*top, item("d35:k1", fusion=0.03)], rank=3)
    assert (found.cause, found.detail) == ("gần hòa điểm", "gap 5.00e-05")


def test_an_exact_tie_is_a_near_tie(tmp_path):
    top = [item("d36:k2", fusion=0.03), item("d35:k1", fusion=0.03)]
    assert regression_of(tmp_path, top).cause == "gần hòa điểm"


@pytest.mark.parametrize(
    ("top", "rank"),
    [
        (FAR, 2),
        (NEAR, None),  # the gold article is not in the top 10
        ([item("d36:k2"), item("d35:k1", fusion=0.03)], 2),  # no fusion score above
        ([item("d36:k2", fusion=0.03), item("d35:k1")], 2),  # none on the gold chunk
        # A promoted chunk above with a lower score: the gold chunk is not behind it.
        ([item("d36:k2", fusion=0.02), item("d35:k1", fusion=0.03)], 2),
    ],
)
def test_cause_shared_keywords_for_the_other_hybrid_regressions(tmp_path, top, rank):
    found = regression_of(tmp_path, top, rank=rank)
    assert (found.cause, found.detail) == ("chunk khác trùng từ khóa", "")


def test_a_sparse_run_lists_regressions_without_a_cause(tmp_path):
    found = regression_of(tmp_path, NEAR, mode="sparse")
    assert (found.cause, found.rank_a, found.rank_b) == (None, 1, 2)


def test_comparison_lists_regressions_and_causes(tmp_path):
    question = numbered("direct", 1)
    reference, run_b = load_pair(tmp_path, [run_row(question, 1)], [run_row(question, 2, top=NEAR)])

    hybrid = compare([question], reference, run_b, mode_b="hybrid")
    promoted = compare([question], reference, run_b, mode_b="hybrid", promotion_b=True)
    sparse = compare([question], reference, run_b, mode_b="sparse")

    assert "Số câu tụt hạng: 1" in hybrid
    assert "- q001 [direct] 1 → 2; nguyên nhân: gần hòa điểm (gap 5.00e-05)" in hybrid
    assert "- q001 [direct] 1 → 2; nguyên nhân: parser (article_refs = ∅, gold = {35})" in promoted
    assert "- q001 [direct] 1 → 2" in sparse.splitlines()


def test_comparison_without_regressions_says_so(tmp_path):
    questions = [numbered("direct", 1)]
    reference, run_b = runs(tmp_path, questions, [2], [1])
    assert "Số câu tụt hạng: 0" in compare(questions, reference, run_b)


# --- reproduction tolerance (§5.7) -------------------------------------------------

SHORTS = [f"d{article}:k1" for article in (35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)]


def top_ten():
    """Ten chunks with cosines 0.90, 0.89, ... 0.81."""
    return [item(short, round(0.90 - 0.01 * place, 2)) for place, short in enumerate(SHORTS[:10])]


def tied(top, place, gap):
    """Give the chunk at ``place`` (0-based) the score of the one above it, less ``gap``."""
    top = list(top)
    top[place] = top[place] | {"score": top[place - 1]["score"] - gap}
    return top


def swapped(top, place):
    """Swap the chunks at ``place - 1`` and ``place``."""
    top = list(top)
    top[place - 1], top[place] = top[place], top[place - 1]
    return top


def tolerance_of(tmp_path, top_a, top_b, mode="dense", group="direct", ranks=((1, 1), (1, 1))):
    question = numbered(group, 1)
    (rank_a, clause_a), (rank_b, clause_b) = ranks
    reference, run_b = load_pair(
        tmp_path,
        [run_row(question, rank_a, clause_a, top_a) | {"rank_clause": clause_a}],
        [run_row(question, rank_b, clause_b, top_b) | {"rank_clause": clause_b}],
    )
    return run_eval.tolerance_check([question], reference.rows, run_b.rows, mode)


def test_tolerance_of_identical_runs(tmp_path):
    check = tolerance_of(tmp_path, top_ten(), top_ten())
    assert (check.in_scope, check.rank_violations) == (1, [])
    everything = check.everything
    assert (everything.questions, everything.identical, everything.swapped) == (1, 1, 0)
    assert (everything.violated, everything.swaps, everything.violations) == (0, [], [])
    assert (everything.scores_checked, everything.score_violations) == (10, [])
    assert check.without_refs == everything


@pytest.mark.parametrize("ranks", [((1, 1), (2, 1)), ((1, 1), (1, 3)), ((None, None), (10, 10))])
def test_tolerance_a_needs_identical_ranks(tmp_path, ranks):
    check = tolerance_of(tmp_path, top_ten(), top_ten(), ranks=ranks)
    assert len(check.rank_violations) == 1
    assert check.rank_violations[0].startswith("q001")


def test_tolerance_a_skips_out_of_scope_questions(tmp_path):
    check = tolerance_of(
        tmp_path, top_ten(), top_ten(), group="out_of_scope", ranks=((None, None), (None, None))
    )
    assert (check.in_scope, check.rank_violations, check.everything.questions) == (0, [], 1)


def test_tolerance_b_accepts_a_swap_of_near_ties(tmp_path):
    old = tied(top_ten(), 4, 5e-5)

    check = tolerance_of(tmp_path, old, swapped(old, 4)).everything

    assert (check.identical, check.swapped, check.violated, check.violations) == (0, 1, 0, [])
    assert check.swaps == [
        f"q001: đổi chỗ {full_id('d38:k1')} <-> {full_id('d39:k1')} (gap A 5.00e-05, B 5.00e-05)"
    ]
    assert check.score_violations == []


@pytest.mark.parametrize("wide_run", ["a", "b"])
def test_tolerance_b_needs_the_tie_in_each_run(tmp_path, wide_run):
    old = tied(top_ten(), 4, 5e-4 if wide_run == "a" else 5e-5)
    new = tied(top_ten(), 4, 5e-4 if wide_run == "b" else 5e-5)

    check = tolerance_of(tmp_path, old, swapped(new, 4)).everything

    assert (check.swapped, check.violated, check.swaps) == (0, 1, [])
    assert len(check.violations) == 1
    assert "đổi chỗ" in check.violations[0]


def test_tolerance_b_accepts_a_near_tie_at_the_cut_edge(tmp_path):
    # Each run has one chunk the other lacks, within 1e-4 of the other run's 10th score.
    new = [*top_ten()[:9], item(SHORTS[10], 0.81 + 5e-5)]

    check = tolerance_of(tmp_path, top_ten(), new).everything

    assert (check.swapped, check.violated, check.violations) == (1, 0, [])
    assert check.swaps == [
        f"q001: biên top 10, {full_id(SHORTS[9])} chỉ có ở A (gap 5.00e-05 tới hạng 10 của B)",
        f"q001: biên top 10, {full_id(SHORTS[10])} chỉ có ở B (gap 5.00e-05 tới hạng 10 của A)",
    ]


@pytest.mark.parametrize("side", ["a", "b"])
def test_tolerance_b_rejects_a_chunk_far_from_the_other_cut(tmp_path, side):
    stranger = [*top_ten()[:9], item(SHORTS[10], 0.8102)]  # 2e-4 above the other 10th score
    old, new = (stranger, top_ten()) if side == "a" else (top_ten(), stranger)

    check = tolerance_of(tmp_path, old, new).everything

    assert (check.swapped, check.violated, check.swaps) == (0, 1, [])
    assert len(check.violations) == 2  # one chunk only in each run


def test_tolerance_b_has_no_cut_edge_against_a_shorter_list(tmp_path):
    check = tolerance_of(tmp_path, top_ten(), top_ten()[:9]).everything
    assert (check.violated, check.swaps) == (1, [])
    assert check.violations == [
        f"q001: {full_id(SHORTS[9])} chỉ có ở A, B chỉ có 9 chunk nên không có biên top 10"
    ]


def test_tolerance_b_counts_a_question_with_both_kinds_as_a_violation(tmp_path):
    old = tied(top_ten(), 4, 5e-5)
    new = swapped(swapped(old, 4), 1)  # also swaps a pair 0.01 apart

    check = tolerance_of(tmp_path, old, new).everything

    assert (check.identical, check.swapped, check.violated) == (0, 0, 1)
    assert (len(check.swaps), len(check.violations)) == (1, 1)


def test_tolerance_c_checks_the_score_of_every_shared_chunk(tmp_path):
    new = top_ten()
    new[0] = new[0] | {"score": 0.9 + 2e-4}
    new[1] = new[1] | {"score": 0.89 + 5e-5}

    check = tolerance_of(tmp_path, top_ten(), new).everything

    assert (check.identical, check.scores_checked) == (1, 10)
    assert check.score_violations == [f"q001: {full_id(SHORTS[0])} |Δ score| = 2.00e-04"]


def fused_top(fusions):
    """Chunks with cosines 0.01 apart and the given fusion scores."""
    return [
        item(short, 0.9 - 0.01 * place, fusion)
        for place, (short, fusion) in enumerate(zip(SHORTS, fusions))
    ]


def test_tolerance_c_checks_fusion_scores_when_both_rows_have_one(tmp_path):
    old = fused_top([0.032, 0.031, 0.030])
    new = fused_top([0.0323, 0.031, None])

    check = tolerance_of(tmp_path, old, new, mode="hybrid").everything

    assert check.score_violations == [f"q001: {full_id(SHORTS[0])} |Δ fusion_score| = 3.00e-04"]


def test_tolerance_ranks_hybrid_rows_by_fusion_score(tmp_path):
    # The cosines differ by 0.01, the fusion scores by 5e-5.
    old = fused_top([0.03005, 0.03])

    hybrid = tolerance_of(tmp_path, old, swapped(old, 1), mode="hybrid").everything
    dense = tolerance_of(tmp_path, old, swapped(old, 1), mode="dense").everything

    assert (hybrid.swapped, hybrid.violated) == (1, 0)
    assert (dense.swapped, dense.violated) == (0, 1)


def test_tolerance_uses_the_cosine_of_a_hybrid_row_without_fusion_score(tmp_path):
    # The dense fallback of hybrid mode: no fusion score, so the near-tie cosines decide.
    old = tied(top_ten(), 1, 5e-5)
    check = tolerance_of(tmp_path, old, swapped(old, 1), mode="hybrid").everything
    assert (check.swapped, check.violated) == (1, 0)


def test_tolerance_d_leaves_out_the_article_ref_questions(tmp_path):
    questions = [
        numbered("direct", 1),
        numbered("article_ref", 2, "Điều 35 nói gì?"),
        numbered("out_of_scope", 3),
    ]
    moved = swapped(top_ten(), 1)
    moved[5] = moved[5] | {"score": 0.99}
    reference, run_b = load_pair(
        tmp_path,
        [run_row(q, 1, top=top_ten()) for q in questions],
        [run_row(q, 1, top=moved if q.id == "q002" else top_ten()) for q in questions],
    )

    check = run_eval.tolerance_check(questions, reference.rows, run_b.rows, "dense")

    assert check.in_scope == 2
    everything, rest = check.everything, check.without_refs
    assert (everything.questions, everything.identical, everything.violated) == (3, 2, 1)
    assert (everything.scores_checked, len(everything.score_violations)) == (30, 1)
    assert (rest.questions, rest.identical, rest.violated, rest.violations) == (2, 2, 0, [])
    assert (rest.scores_checked, rest.score_violations) == (20, [])


@pytest.mark.parametrize(
    ("mode_a", "mode_b", "checked"),
    [
        (None, "dense", True),  # no meta file: dense
        ("dense", "dense", True),
        ("hybrid", "hybrid", True),
        (None, "hybrid", False),
        ("dense", "hybrid", False),
        ("hybrid", "dense", False),
        ("dense", "sparse", False),
        ("sparse", "sparse", False),
    ],
)
def test_comparison_runs_the_tolerance_check_only_for_equal_dense_or_hybrid_modes(
    tmp_path, mode_a, mode_b, checked
):
    question = numbered("direct", 1)
    rows = [run_row(question, 1, top=top_ten())]
    meta = {} if mode_a is None else {"mode": mode_a, "questions_digest": DIGEST}
    reference = run_eval.load_reference(write_run(tmp_path / "a.jsonl", rows, **meta))
    run_b = run_eval.load_reference(write_run(tmp_path / "b.jsonl", rows))

    text = compare([question], reference, run_b, mode_b=mode_b)

    assert ("(a) rank_article" in text) is checked
    assert ("Bỏ qua kiểm tra dung sai" in text) is not checked


def test_comparison_prints_the_tolerance_counts_swaps_and_violations(tmp_path):
    questions = [numbered("direct", 1), numbered("article_ref", 2, "Điều 35 nói gì?")]
    near = tied(top_ten(), 4, 5e-5)
    reference, run_b = load_pair(
        tmp_path,
        [run_row(questions[0], 1, top=near), run_row(questions[1], 1, top=top_ten())],
        [
            run_row(questions[0], 1, top=swapped(near, 4)),
            run_row(questions[1], 2, 1, top=swapped(top_ten(), 1)),
        ],
    )

    lines = compare(questions, reference, run_b).splitlines()

    assert "(a) rank_article và rank_clause giống hệt: 1/2 câu trong phạm vi; đạt: không" in lines
    assert "  - q002: rank_article 1 → 2, rank_clause 1 → 1" in lines
    assert (
        "(b) top 10 (2 câu): giống hệt 0, chỉ đổi chỗ trong dung sai 1, có vi phạm 1; đạt: không"
        in lines
    )
    assert "(c) |Δ điểm| < 1e-4: 20 chunk chung, vi phạm 0; đạt: có" in lines
    assert "(d) (b) và (c) trên 1 câu không phải article_ref:" in lines
    assert (
        "(b) top 10 (1 câu): giống hệt 0, chỉ đổi chỗ trong dung sai 1, có vi phạm 0; đạt: có"
        in lines
    )
    assert "(c) |Δ điểm| < 1e-4: 10 chunk chung, vi phạm 0; đạt: có" in lines
    # Every swap and violation is listed: the swap in (b) and (d), the violation in (b) only.
    assert sum(line.startswith("  - q001: đổi chỗ") for line in lines) == 2
    assert sum(line.startswith("  - q002: đổi chỗ") for line in lines) == 1


# --- the reference on the command line --------------------------------------------


def make_reference(tmp_path, questions_path, monkeypatch, fake):
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    out = tmp_path / "a.jsonl"
    assert run_eval.main(["--questions", str(questions_path), "--out", str(out)]) == 0
    return out


def test_main_compares_a_new_run_with_a_reference(
    monkeypatch, tmp_path, capsys, fake, all_chunks, questions_path
):
    reference = make_reference(tmp_path, questions_path, monkeypatch, fake)
    capsys.readouterr()
    worse = MAPPING | {Q_DIRECT_2.question: ["d37:k1", "d36:k2"]}
    monkeypatch.setattr(run_eval, "default_retriever", lambda: FakeRetriever(worse, all_chunks))
    out = tmp_path / "b.jsonl"

    code = run_eval.main(
        ["--questions", str(questions_path), "--out", str(out), "--reference", str(reference)]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "direct: A 1/2, B 0/2" in printed
    assert "- q005 [direct] 1 → 2; nguyên nhân: khác" in printed
    assert "(a) rank_article và rank_clause giống hệt: 4/5 câu trong phạm vi" in printed
    assert printed.index("## 6.") < printed.index("## 7.")
    assert label_of(printed) == "dense baseline"  # the results row stays last
    assert read_rows(out)[4]["rank_article"] == 2


def regrouped(questions):
    """Turn the pair p01 into two article_ref questions with the same ids."""
    return [
        numbered("article_ref", int(q.id[1:]), q.question) if q.id in ("q001", "q002") else q
        for q in questions
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda questions: questions[:-1], "q006"),
        (lambda questions: [*questions, numbered("direct", 7)], "q007"),
        (regrouped, "q001, q002"),
    ],
    ids=["missing", "extra", "regrouped"],
)
def test_main_rejects_a_reference_with_other_questions(
    monkeypatch, tmp_path, capsys, fake, questions_path, change, message
):
    path = write_run(tmp_path / "a.jsonl", [run_row(q) for q in change(QUESTIONS)])
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)

    code = run_eval.main(["--questions", str(questions_path), "--reference", str(path)])

    captured = capsys.readouterr()
    assert code == 1
    assert "does not have the same question ids and groups" in captured.err
    assert message in captured.err
    assert (captured.out, fake.calls) == ("", [])


def test_main_rejects_a_reference_run_on_another_question_set(
    monkeypatch, tmp_path, capsys, fake, questions_path
):
    path = write_run(
        tmp_path / "a.jsonl",
        [run_row(q) for q in QUESTIONS],
        mode="dense",
        questions_digest=DIGEST,
    )
    monkeypatch.setattr(run_eval, "default_retriever", lambda: fake)
    out = tmp_path / "b.jsonl"

    code = run_eval.main(
        ["--questions", str(questions_path), "--reference", str(path), "--out", str(out)]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert DIGEST in captured.err
    assert run_eval.questions_digest(questions_path) in captured.err
    assert (captured.out, fake.calls) == ("", [])
    assert not out.exists()


@pytest.mark.parametrize(
    "target",
    [
        "a.jsonl",  # the reference itself
        "sub/../a.jsonl",  # the same file, spelled differently
        "a.meta.json",  # its meta file
        "a.json",  # its meta file would be written over
    ],
)
def test_main_refuses_to_overwrite_the_reference(
    monkeypatch, tmp_path, capsys, fake, questions_path, target
):
    reference = make_reference(tmp_path, questions_path, monkeypatch, fake)
    (tmp_path / "sub").mkdir()
    capsys.readouterr()
    fake.calls.clear()
    before = (reference.read_bytes(), run_eval.meta_path(reference).read_bytes())

    code = run_eval.main(
        [
            *("--questions", str(questions_path)),
            *("--reference", str(reference)),
            *("--out", str(tmp_path / target)),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "would overwrite the reference" in captured.err
    assert fake.calls == []
    assert (reference.read_bytes(), run_eval.meta_path(reference).read_bytes()) == before


# The child process gets no fixtures, so it wires a fake retriever itself, and it must not read
# the developer's .env, reach Qdrant or load the model.
CONSOLE_SCRIPT = """
import sys
from app.core.config import Settings
from app.rag.retriever import RetrievedChunk
from eval import run_eval
from ingestion.documents import REPO_ROOT
from ingestion.index import load_chunks

chunks = load_chunks(REPO_ROOT / "data" / "processed" / "chunks.jsonl")


class Fake:
    def retrieve(self, query, k=5):
        return [RetrievedChunk(chunk=chunks[0], score=0.5)]


def refuse(*args, **kwargs):
    raise AssertionError("a real retriever was built")


Settings.model_config["env_file"] = None
run_eval.default_retriever = lambda: Fake()
run_eval.Retriever = refuse
raise SystemExit(run_eval.main(sys.argv[1:]))
"""


def run_on_a_cp1252_console(arguments):
    env = os.environ | {
        "PYTHONIOENCODING": "cp1252",
        "QDRANT_COLLECTION": "luật_lao_động",
        "ARTICLE_PROMOTION": "false",  # the fake retriever has no client for a dense probe
    }
    return subprocess.run(
        [sys.executable, "-c", CONSOLE_SCRIPT, *arguments],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        timeout=120,
    )


def test_the_comparison_survives_a_legacy_console_encoding(tmp_path, questions_path):
    reference = write_run(tmp_path / "a.jsonl", [run_row(q) for q in QUESTIONS])
    arguments = ["--questions", str(questions_path), "--reference"]

    result = run_on_a_cp1252_console([*arguments, str(reference)])

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    printed = result.stdout.decode("utf-8")
    assert "Collection: luật_lao_động__" in printed
    assert "## 7. So với run tham chiếu" in printed
    assert "Số câu tụt hạng" in printed

    result = run_on_a_cp1252_console([*arguments, str(tmp_path / "thiếu-điều.jsonl")])

    assert result.returncode == 1
    assert "thiếu-điều.jsonl" in result.stderr.decode("utf-8")
