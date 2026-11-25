import unicodedata
from collections import Counter

import pytest

from app.rag import sparse
from app.rag.sparse import (
    SparseVector,
    average_length,
    clean_query,
    document_terms,
    document_vector,
    fold_tone,
    normalize,
    query_terms,
    query_vector,
    term_id,
)


def test_tokenizer_constants():
    assert (sparse.TOKENIZER_VERSION, sparse.K1, sparse.B) == ("vi-syl-bigram-v1", 1.2, 0.75)


def test_normalize_composes_lowercases_and_repairs_eth():
    decomposed = unicodedata.normalize("NFD", "Điều")
    assert normalize(decomposed) == "điều"
    # U+00D0 and U+00F0 look like the Vietnamese letter but are the Icelandic eth.
    assert normalize("\u00d0IỀU \u00f0ủ") == "điều đủ"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("hoà", "hòa"),
        ("thuỷ", "thủy"),
        ("khoẻ", "khỏe"),
        ("toà", "tòa"),
        ("thoả", "thỏa"),
        ("huỷ", "hủy"),
        ("tuỳ", "tùy"),
        ("uỷ", "ủy"),
        ("loè", "lòe"),
        ("tuý", "túy"),
        # The remaining endings of rule 4.
        ("hoá", "hóa"),
        ("doã", "dõa"),
        ("hoạ", "họa"),
        ("khoé", "khóe"),
        ("hoẽ", "hõe"),
        ("hoẹ", "họe"),
        ("luỹ", "lũy"),
        ("tuỵ", "tụy"),
    ],
)
def test_fold_tone_moves_the_tone_to_the_first_vowel(token, expected):
    assert fold_tone(token) == expected


@pytest.mark.parametrize(
    "token",
    [
        "hòa",
        "thủy",
        "hoàn",
        "khoản",
        "khoảng",
        "toàn",
        "đoàn",
        "ngoài",
        "loại",
        "hoạt",
        "hoạch",
        "hoặc",
        "xoắn",
        "xoáy",
        "khoèo",
        "huỳnh",
        "khuỷu",
        "tuýt",
        "tuyển",
        "thuyền",
        "quý",
        "quỳnh",
        "quỵ",
        "quy",
        "giá",
        "giàu",
    ],
)
def test_fold_tone_leaves_other_syllables_unchanged(token):
    assert fold_tone(token) == token


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Điều 35. Quyền đơn phương",
            ["điều", "35", "điều 35", "quyền", "đơn", "phương", "quyền đơn", "đơn phương"],
        ),
        (
            "Thời hạn 01 năm; 10/2012/QH13, 61% và 0,7",
            [
                "thời",
                "hạn",
                "1",
                "năm",
                "thời hạn",
                "hạn 1",
                "1 năm",
                "10",
                "2012",
                "qh13",
                "61",
                "và",
                "0",
                "và 0",
                "7",
            ],
        ),
        (
            "\u00d0IỀU 5a, 1a và 8a",
            ["điều", "5a", "điều 5a", "1a", "và", "8a", "1a và", "và 8a"],
        ),
        (
            "Hoà giải viên, thuỷ thủ, khoẻ, quý, hoàn",
            [
                "hòa",
                "giải",
                "viên",
                "hòa giải",
                "giải viên",
                "thủy",
                "thủ",
                "thủy thủ",
                "khỏe",
                "quý",
                "hoàn",
            ],
        ),
        (unicodedata.normalize("NFD", "hòa giải"), ["hòa", "giải", "hòa giải"]),
        (
            "Bộ luật Lao động\nChương I - Những quy định chung",
            [
                "bộ",
                "luật",
                "lao",
                "động",
                "bộ luật",
                "luật lao",
                "lao động",
                "chương",
                "i",
                "chương i",
                "những",
                "quy",
                "định",
                "chung",
                "những quy",
                "quy định",
                "định chung",
            ],
        ),
    ],
)
def test_document_terms(text, expected):
    assert Counter(document_terms(text)) == Counter(expected)


def test_document_terms_ignore_the_tone_style():
    assert document_terms("thoả thuận") == document_terms("thỏa thuận")


@pytest.mark.parametrize(("token", "expected"), [("01", "1"), ("00", "0"), ("0", "0")])
def test_digit_tokens_lose_leading_zeros(token, expected):
    assert document_terms(token) == [expected]


@pytest.mark.parametrize("separator", [" ", "\t", "\u00a0"])
def test_space_tab_and_nbsp_stay_inside_a_segment(separator):
    assert "lao động" in document_terms(f"lao{separator}động")


@pytest.mark.parametrize("separator", ["\n", "-", "/", "%", ".", "|"])
def test_other_non_word_characters_end_a_segment(separator):
    assert Counter(document_terms(f"lao{separator}động")) == Counter(["lao", "động"])


@pytest.mark.parametrize(
    ("question", "cleaned"),
    [
        ("Thử việc tối đa bao lâu?", "thử việc tối đa"),
        ("Lương tối thiểu là bao nhiêu vậy?", "lương tối thiểu là"),
        ("Có được nghỉ không ạ?", "có được nghỉ"),
        ("Hợp đồng không xác định thời hạn là gì?", "hợp đồng không xác định thời hạn là"),
        ("Người chưa thành niên được làm việc gì?", "người chưa thành niên được làm việc"),
        (
            "Hợp đồng lao động bao gồm những nội dung nào?",
            "hợp đồng lao động bao gồm những nội dung",
        ),
        ("Đã đủ tuổi nghỉ hưu chưa?", "đã đủ tuổi nghỉ hưu"),
        ("Thay thế người lao động thế nào?", "thay thế người lao động"),
        ("Ai?", ""),
        ("Mấy giờ?", "giờ"),
        ("Ai là ai?", "là"),
        ("\n Mấy giờ?", "giờ"),
    ],
)
def test_clean_query(question, cleaned):
    assert clean_query(question) == cleaned


@pytest.mark.parametrize(("question", "terms"), [("Mấy giờ?", ["giờ"]), ("Ai là ai?", ["là"])])
def test_query_terms_of_a_question_that_starts_with_an_interrogative(question, terms):
    assert query_terms(question) == terms


@pytest.mark.parametrize(
    "phrase",
    [
        "như thế nào",
        "thế nào",
        "bao nhiêu",
        "bao lâu",
        "bao giờ",
        "tại sao",
        "vì sao",
        "gì",
        "nào",
        "ai",
        "đâu",
        "mấy",
    ],
)
def test_every_interrogative_becomes_a_segment_boundary(phrase):
    spaced = phrase.replace(" ", " \n ")
    assert Counter(query_terms(f"nghỉ {spaced} lương")) == Counter(["nghỉ", "lương"])


@pytest.mark.parametrize("particle", ["không", "chưa", "à", "ư", "nhỉ", "nhé", "hả", "ạ", "vậy"])
def test_every_trailing_particle_is_removed(particle):
    assert clean_query(f"được nghỉ {particle}?!") == "được nghỉ"


def test_interrogatives_and_particles_match_whole_words_only():
    # "mai" and "tai" contain "ai", "khà" ends with "à".
    assert clean_query("ngày mai bị tai nạn khà") == "ngày mai bị tai nạn khà"


def test_query_terms_are_unique_in_first_occurrence_order():
    assert query_terms("Thử việc tối đa bao lâu?") == [
        "thử",
        "việc",
        "tối",
        "đa",
        "thử việc",
        "việc tối",
        "tối đa",
    ]
    assert query_terms("lao lao động") == ["lao", "động", "lao lao", "lao động"]


def test_query_terms_keep_words_that_the_law_uses():
    assert "bao gồm" in query_terms("Hợp đồng lao động bao gồm những nội dung nào?")
    terms = query_terms("Thay thế người lao động thế nào?")
    assert "thay thế" in terms
    assert terms.count("thế") == 1  # only the one inside "thay thế"
    assert "động thế" not in terms


def test_removed_middle_span_breaks_bigrams():
    terms = query_terms("Nghỉ bao lâu thì được trả lương?")
    assert "nghỉ thì" not in terms
    assert "thì được" in terms


def test_query_that_cleans_to_nothing():
    assert query_terms("Ai?") == []
    assert query_vector("Ai?") == SparseVector((), ())


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("lao", 2780255392),
        ("động", 1759712262),
        ("lao động", 2505683276),
        ("1", 4237569031),
        ("hòa", 1031448236),
        ("khoản", 809149146),
        ("điều 35", 720491457),
    ],
)
def test_term_id_golden_values(term, expected):
    assert term_id(term) == expected


@pytest.mark.parametrize(("avgdl", "lao", "other"), [(5, 1.375, 1.0), (10, 1.6, 2.2 / 1.75)])
def test_document_vector_bm25_weights(avgdl, lao, other):
    text = "lao lao động"
    assert Counter(document_terms(text)) == Counter(["lao", "lao", "động", "lao lao", "lao động"])
    vector = document_vector(text, avgdl)
    weights = dict(zip(vector.indices, vector.values, strict=True))
    assert weights == pytest.approx(
        {
            term_id("lao"): lao,
            term_id("động"): other,
            term_id("lao lao"): other,
            term_id("lao động"): other,
        }
    )


def test_vector_ids_are_strictly_increasing_and_unique():
    text = "Người lao động được nghỉ hằng năm, người sử dụng lao động trả lương."
    for vector in (document_vector(text, 10), query_vector(text)):
        assert len(vector.indices) == len(vector.values) > 1
        assert list(vector.indices) == sorted(set(vector.indices))


def test_query_vector_values_are_all_one():
    vector = query_vector("lao lao động")
    assert vector.values == (1.0,) * 4
    assert set(vector.indices) == {term_id(t) for t in ["lao", "động", "lao lao", "lao động"]}


def test_colliding_ids_are_merged(monkeypatch):
    monkeypatch.setattr(sparse, "term_id", lambda term: 7)
    # Weights at avgdl 5: 1.375 for "lao" plus 1.0 for each of the three other terms.
    document = document_vector("lao lao động", 5)
    assert document.indices == (7,)
    assert document.values == pytest.approx((4.375,))
    assert query_vector("lao lao động") == SparseVector((7,), (1.0,))


def test_empty_text_gives_empty_document_vector():
    assert document_vector("", 5) == SparseVector((), ())


def test_average_length():
    assert average_length(["lao động"]) == 3.0
    assert average_length(text for text in ["lao động", "lao"]) == 2.0


def test_average_length_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        average_length([])


@pytest.mark.parametrize("avgdl", [0, -1.0])
def test_document_vector_rejects_non_positive_average_length(avgdl):
    with pytest.raises(ValueError):
        document_vector("x", avgdl)


def test_corpus_terms_have_no_id_collision(all_chunks):
    assert len(all_chunks) == 676
    terms = {term for chunk in all_chunks for term in document_terms(chunk.context_text)}
    assert len({term_id(term) for term in terms}) == len(terms)
