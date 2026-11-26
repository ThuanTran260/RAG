import time
import unicodedata

import pytest

from app.rag.article_refs import QUANTITY_WORDS, article_refs
from eval.questions import load_questions
from ingestion.documents import REPO_ROOT

PREFIX = "45-2019-QH14:"

# Spec §7.4. A chunk id means the text is quoted verbatim from that chunk.
MUST_TRIGGER = [
    ("d39", "các điều 35, 36 và 37 của Bộ luật này", {35, 36, 37}),
    ("d63:k2", "các điều 42, 44, 93, 104, 118", {42, 44, 93, 104, 118}),
    ("d34:k11", "Điều 42 và Điều 43 của Bộ luật này", {42, 43}),
    ("d37:k1", "khoản 1 Điều 36 của Bộ luật này", {36}),
    (None, "Điều 24 quy định gì", {24}),
]
MUST_NOT_TRIGGER = [
    ("d219:k2", "Điều 32 của Bộ luật Tố tụng dân sự"),
    ("d34:k4", "Điều 328 của Bộ luật Tố tụng hình sự"),
    ("d219:k1", "Điều 54 của Luật này"),
    ("d220:k1", "Bộ luật Lao động số 10/2012/QH13"),
    ("d219:k1", "Điều kiện"),
    (None, "Tôi muốn hỏi điều 1 chút"),
    (None, "Điều 5a"),
]


@pytest.fixture(scope="session")
def texts(all_chunks):
    return {chunk.id.removeprefix(PREFIX): chunk.text for chunk in all_chunks}


@pytest.mark.parametrize(("source", "excerpt", "articles"), MUST_TRIGGER)
def test_a_fixture_gives_its_articles(texts, source, excerpt, articles):
    if source is not None:
        assert excerpt in texts[source]
    assert article_refs(excerpt) == articles


def test_the_article_of_the_question_fixture_is_in_the_corpus(all_chunks):
    assert any(chunk.article == 24 for chunk in all_chunks)


@pytest.mark.parametrize(("source", "excerpt"), MUST_NOT_TRIGGER)
def test_a_must_not_fixture_gives_nothing(texts, source, excerpt):
    if source is not None:
        assert excerpt in texts[source]
    assert article_refs(excerpt) == set()


def test_each_eval_question_gives_its_gold_articles_or_nothing():
    questions = load_questions(REPO_ROOT / "eval" / "questions.jsonl")
    named = [question for question in questions if question.group == "article_ref"]
    others = [question for question in questions if question.group != "article_ref"]

    assert (len(named), len(others)) == (12, 115)
    assert {q.id: article_refs(q.question) for q in named} == {
        q.id: set(q.gold_articles) for q in named
    }
    assert {q.id: found for q in others if (found := article_refs(q.question))} == {}


@pytest.mark.parametrize(
    ("text", "articles"),
    [
        ("Điều 35?", {35}),
        ("Điều 35.", {35}),
        ("Điều 5a", set()),
        ("Điều 5a, Điều 6", {6}),
        ("Điều 35, 36a", {35}),
        ("Điều 35đ nói gì", set()),
    ],
)
def test_a_number_must_end_at_a_boundary(text, articles):
    assert article_refs(text) == articles


@pytest.mark.parametrize(
    ("text", "articles"),
    [
        ("Điều 0", set()),
        ("Điều 1", {1}),
        ("Điều 220", {220}),
        ("Điều 221", set()),
        ("các điều 219, 220 và 221", {219, 220}),
    ],
)
def test_only_articles_of_the_code_count(text, articles):
    assert article_refs(text) == articles


@pytest.mark.parametrize("text", ["điều 35 nói gì", "ĐIỀU 35 NÓI GÌ"])
def test_a_mention_counts_in_any_case(text):
    assert article_refs(text) == {35}


def test_a_decomposed_mention_counts():
    text = unicodedata.normalize("NFD", "Điều 35 của Bộ luật này")
    assert text != unicodedata.normalize("NFC", text)

    assert article_refs(text) == {35}


@pytest.mark.parametrize("word", QUANTITY_WORDS)
def test_a_quantity_is_not_an_article(word):
    assert article_refs(f"Cho mình hỏi điều 1 {word}") == set()
    assert article_refs(f"Cho mình hỏi điều 1 {word.upper()}?") == set()


@pytest.mark.parametrize(
    ("text", "articles"),
    [
        ("Cho mình hỏi điều 1 chút xíu", set()),
        ("Điều 24, 1 chút thôi", {24}),  # only the number the word follows is a quantity
        ("Điều 24 tính thế nào", {24}),  # "tính" only starts like "tí"
    ],
)
def test_the_quantity_rule_takes_whole_words_after_one_number(text, articles):
    assert article_refs(text) == articles


@pytest.mark.parametrize(
    "name",
    [
        "của Bộ luật này",
        "Bộ luật Lao động",
        "của Bộ Luật Lao Động Việt Nam",
        "Bộ luật Lao động 2019",
        "Bộ luật Lao động năm 2019",
        "Bộ luật Lao động số 45/2019/QH14",
        "Bộ luật số 45/2019/QH14",
        "Luật số 45/2019/QH14",
        "Luật Lao động",
        "Luật Lao động 2019",
        "luật lao động quy định gì",
        "BLLĐ",
        "BLLĐ 2019",
        "(Bộ luật Lao động)",
        ", Bộ luật Lao động",
        "luật quy định sao",
        "luật nói gì",
        "luật ghi thế nào",
        "của luật hiện hành",
        "của Bộ luật?",
        "của luật",
    ],
)
def test_names_of_this_code_keep_the_article(name):
    assert article_refs(f"Điều 35 {name}") == {35}


@pytest.mark.parametrize(
    "name",
    [
        "của Bộ luật Tố tụng dân sự",
        "Bộ luật Tố tụng dân sự",
        ", Bộ luật Tố tụng dân sự",
        ": Bộ luật Dân sự",
        "(Bộ luật Hình sự)",
        "của Luật này",
        "của Luật Bảo hiểm xã hội",
        "của luật bảo hiểm xã hội",
        "Luật Việc làm",
        "luật định",
        "của bộ luật đó",
        "khoản 2 của Luật Công đoàn",
        ", khoản 2, điểm a Luật Công đoàn",
        "Bộ luật Lao động 2012",
        "Bộ luật Lao động năm 2012",
        "Bộ luật Lao động số 10/2012/QH13",
        "của Bộ luật Lao động (số 10/2012/QH13)",
        "Bộ luật Lao động 10/2012/QH13",
        "Luật Lao động 2012",
        "BLLĐ 2012",
        "Bộ luật số 10/2012/QH13",
        "Nghị định 145",
        "theo Nghị định 145/2020/NĐ-CP",
        "quy định tại Thông tư 10",
        "trong Nghị quyết của Quốc hội",
        "của Pháp lệnh Thủ tục",
        "của Hiến pháp",
        "của Công ước 98",
        "Quyết định số 1989",
        "Quyết định 1989",
        "Chỉ thị số 5",
        "NĐ 145",
        "TT 10",
        "BLDS",
        "BLHS",
        "BLTTDS",
        "BLTTHS",
    ],
)
def test_names_of_other_documents_drop_the_article(name):
    assert article_refs(f"Điều 35 {name}") == set()


@pytest.mark.parametrize(
    ("text", "articles"),
    [
        ("Điều 3 trong hợp đồng của tôi", set()),
        ("Điều 3 của hợp đồng", set()),
        ("Điều 5 của nội quy công ty", set()),
        ("Điều 5 trong quy chế lương", set()),
        ("khoản 2 Điều 5 trong bản thỏa ước", set()),
        ("Điều 5 trong thoả ước", set()),
        ("Điều 5 của cái điều lệ", set()),
        ("Điều 5 khoản 2 của hợp đồng", set()),
        # Without "của" or "trong" the words are the topic of the question.
        ("Điều 21 hợp đồng lao động gồm những gì", {21}),
        ("Điều 21 quy định hợp đồng lao động gồm những gì", {21}),
    ],
)
def test_a_contract_or_company_rules_drop_the_article(text, articles):
    assert article_refs(text) == articles


@pytest.mark.parametrize(
    "text",
    [
        "Điều 98 quy định gì",
        "Điều 98 nói về vấn đề gì",
        "Nội dung Điều 98 là gì",
        "Điều 98 quyết định thế nào",
        "Điều 98 quy định thông tin gì",
        "Điều 98 được hướng dẫn ở Nghị định 145",  # the name does not follow the number
        "Điều 98 của Bộ luật này và Nghị định 145",
    ],
)
def test_words_that_are_not_a_following_name_keep_the_article(text):
    assert article_refs(text) == {98}


def test_a_chain_of_mentions_of_this_code_is_kept(texts):
    excerpt = "khoản 2 Điều 18, điểm a khoản 1 Điều 145 và khoản 1 Điều 162 của Bộ luật này"
    assert excerpt in texts["d14:k2"]

    assert article_refs(excerpt) == {18, 145, 162}


def test_a_whole_clause_gives_every_article_it_cites(texts):
    assert article_refs(texts["d63:k2"]) == {36, 42, 44, 93, 104, 118, 128}


@pytest.mark.parametrize(
    ("text", "articles"),
    [
        ("Điều 32 và Điều 33 của Bộ luật Tố tụng dân sự", set()),
        ("khoản 1 Điều 32, các điều 35 và 36 và điểm b khoản 2 Điều 40 của Luật này", set()),
        ("Điều 32 hoặc khoản 1 và khoản 2 Điều 33 của Luật này", set()),
        ("Điều 32; Điều 33 hay Điều 34 của Luật này", set()),
        # Other words between two mentions end the chain, so each has its own name.
        ("Điều 35 khác gì Điều 32 của Bộ luật Tố tụng dân sự", {35}),
    ],
)
def test_a_chain_of_mentions_shares_one_document_name(text, articles):
    assert article_refs(text) == articles


@pytest.mark.parametrize("gap", [" ", "\n", "\t "])
def test_a_long_run_of_whitespace_is_read_quickly(gap):
    run = gap * 2000
    text = f"Điều 1{run}x{run}Điều 2{run}của{run}Luật này"
    start = time.perf_counter()

    assert article_refs(text) == {1}

    # Regexes that retry every split of a run need minutes here.
    assert time.perf_counter() - start < 1


# A question has no length limit. Each shape repeats sub-references that no document name follows,
# so a pattern that can split one space two ways retries every split.
@pytest.mark.parametrize(
    ("head", "unit", "tail", "articles"),
    [
        ("Điều 1", " khoản 1", " x", {1}),
        ("Điều 1 ", "khoản 1 điểm a ", "", {1}),
        ("Điều 1", " các khoản 1, 2 và 3", "", {1}),
        ("Điều 1", " khoản 1", " của", {1}),
        ("Điều 1", " khoản 1", " x Điều 2", {1, 2}),
        ("Điều 1", " khoản 1 và 2", " x", {1}),
        ("Điều 1", " khoản 1 điểm a", " x Điều 2", {1, 2}),
    ],
)
def test_a_long_run_of_sub_references_is_read_quickly(head, unit, tail, articles):
    text = head + unit * (10_000 // len(unit) + 1) + tail
    assert len(text) >= 10_000
    start = time.perf_counter()

    assert article_refs(text) == articles

    assert time.perf_counter() - start < 1


@pytest.mark.parametrize(
    ("text", "articles"),
    [
        ("Điều " + "1" * 5000, set()),
        ("Điều 35, " + "1" * 5000, {35}),
        ("Điều " + "0" * 5000 + "35", {35}),
        ("Điều 0035", {35}),
        ("Điều 35, " + "1" * 5000 + "a", {35}),
        ("Điều 35, " + "0" * 5000 + "35a", set()),  # "35a" drops 35, however it is written
    ],
    ids=["too-long", "list-with-too-long", "long-zeros", "zeros", "long-with-letter", "same-value"],
)
def test_a_long_digit_run_is_only_a_large_number(text, articles):
    assert article_refs(text) == articles


def test_text_without_a_mention_gives_nothing():
    assert article_refs("") == set()
    assert article_refs("Người lao động được nghỉ mấy ngày?") == set()
