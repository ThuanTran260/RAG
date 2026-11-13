from ingestion.parser import parse_articles

PREAMBLE = ["BỘ LUẬT", "LAO ĐỘNG", "1. Luật Công nghiệp công nghệ số số 71/2025/QH15;"]
CH1 = ["Chương I", "NHỮNG QUY ĐỊNH CHUNG"]


def test_preamble_is_skipped_and_article_inherits_chapter():
    arts = parse_articles(PREAMBLE + CH1 + ["Điều 1. Phạm vi điều chỉnh", "Bộ luật Lao động quy định."])
    assert len(arts) == 1
    a = arts[0]
    assert (a.number, a.title) == (1, "Phạm vi điều chỉnh")
    assert (a.chapter, a.chapter_title) == ("I", "NHỮNG QUY ĐỊNH CHUNG")
    assert (a.section, a.section_title) == (None, None)


def test_article_without_clauses_keeps_body_as_intro():
    arts = parse_articles(CH1 + ["Điều 1. Phạm vi điều chỉnh", "Bộ luật Lao động quy định."])
    assert arts[0].intro == "Bộ luật Lao động quy định."
    assert arts[0].clauses == []


def test_clauses_points_and_continuation_paragraphs():
    arts = parse_articles(CH1 + [
        "Điều 1. Giải thích từ ngữ",
        "Trong Bộ luật này, các từ ngữ dưới đây được hiểu như sau:",
        "1. Người lao động là người làm việc.",
        "Độ tuổi lao động tối thiểu là đủ 15 tuổi.",
        "2. Người lao động có các quyền sau đây:",
        "a) Làm việc;",
        "đ) Đơn phương chấm dứt hợp đồng lao động;",
    ])
    a = arts[0]
    assert a.intro == "Trong Bộ luật này, các từ ngữ dưới đây được hiểu như sau:"
    assert [c.label for c in a.clauses] == ["1", "2"]
    assert a.clauses[0].text == "1. Người lao động là người làm việc.\nĐộ tuổi lao động tối thiểu là đủ 15 tuổi."
    assert a.clauses[1].text.endswith("a) Làm việc;\nđ) Đơn phương chấm dứt hợp đồng lao động;")


def test_lettered_clause_label():
    arts = parse_articles(CH1 + ["Điều 1. Thử", "8. Tám.", "8a. Tám a.", "9. Chín."])
    assert [c.label for c in arts[0].clauses] == ["8", "8a", "9"]


def test_multi_paragraph_headings_are_joined():
    arts = parse_articles([
        "Chương I", "BẢO HIỂM XÃ HỘI, BẢO HIỂM Y TẾ,", "BẢO HIỂM THẤT NGHIỆP",
        "Mục 1", "NGƯỜI LAO ĐỘNG VIỆT NAM ĐI LÀM VIỆC Ở NƯỚC NGOÀI,", "LAO ĐỘNG CHO CÁC TỔ CHỨC",
        "Điều 1. Thử", "Nội dung.",
    ])
    a = arts[0]
    assert a.chapter_title == "BẢO HIỂM XÃ HỘI, BẢO HIỂM Y TẾ, BẢO HIỂM THẤT NGHIỆP"
    assert (a.section, a.section_title) == (
        1, "NGƯỜI LAO ĐỘNG VIỆT NAM ĐI LÀM VIỆC Ở NƯỚC NGOÀI, LAO ĐỘNG CHO CÁC TỔ CHỨC")


def test_section_is_reset_by_a_new_chapter():
    arts = parse_articles([
        "Chương I", "MỘT", "Mục 1", "GIAO KẾT", "Điều 1. A", "Nội dung.",
        "Chương II", "HAI", "Điều 2. B", "Nội dung.",
    ])
    assert (arts[0].section, arts[0].section_title) == (1, "GIAO KẾT")
    assert (arts[1].chapter, arts[1].section, arts[1].section_title) == ("II", None, None)


import pytest

from ingestion.parser import ParseError


def test_quoted_article_inside_an_amendment_is_not_a_new_article():
    arts = parse_articles(CH1 + [
        "Điều 1. Sửa đổi, bổ sung một số điều của các luật có liên quan",
        "1. Sửa đổi, bổ sung một số điều của Luật Bảo hiểm xã hội:",
        "a) Sửa đổi, bổ sung Điều 54 như sau:",
        "“Điều 54. Điều kiện hưởng lương hưu",
        "1. Người lao động được hưởng lương hưu.",
        "2. Điều kiện khác theo quy định của Chính phủ.”;",
        "2. Sửa đổi, bổ sung Điều 32 của Bộ luật Tố tụng dân sự.",
        "Điều 2. Hiệu lực thi hành",
        "Bộ luật này có hiệu lực.",
    ])
    assert [a.number for a in arts] == [1, 2]
    assert [c.label for c in arts[0].clauses] == ["1", "2"]
    assert "“Điều 54. Điều kiện hưởng lương hưu" in arts[0].clauses[0].text
    assert arts[0].clauses[0].text.endswith("2. Điều kiện khác theo quy định của Chính phủ.”;")


def test_quote_closed_in_the_same_paragraph_does_not_open_a_block():
    arts = parse_articles(CH1 + [
        "Điều 1. Thử",
        "1. Sửa đổi khoản 1 như sau:",
        "“1. Người lao động hưởng lương hưu.”.",
        "2. Khoản hai.",
    ])
    assert [c.label for c in arts[0].clauses] == ["1", "2"]


def test_gap_in_article_numbers_raises():
    with pytest.raises(ParseError, match="expected Điều 2, found Điều 3"):
        parse_articles(CH1 + ["Điều 1. A", "Nội dung.", "Điều 3. C", "Nội dung."])


def test_first_article_must_be_number_one():
    with pytest.raises(ParseError, match="expected Điều 1, found Điều 5"):
        parse_articles(CH1 + ["Điều 5. A", "Nội dung."])


def test_duplicate_clause_label_raises():
    with pytest.raises(ParseError, match="duplicate clause 1 in Điều 1"):
        parse_articles(CH1 + ["Điều 1. A", "1. Một.", "1. Lại một."])


def test_unterminated_quote_raises():
    with pytest.raises(ParseError, match="unterminated quote in Điều 1"):
        parse_articles(CH1 + ["Điều 1. A", "1. Sửa như sau:", "“Điều 9. Trích dở"])


def test_text_without_any_article_raises():
    with pytest.raises(ParseError, match="no articles found"):
        parse_articles(["LUẬT CÔNG AN NHÂN DÂN", "Nội dung không có cấu trúc."])


def test_clause_number_jump_raises():
    # A paragraph that merely starts with a quoted term opens a block and swallows clauses 2-3.
    with pytest.raises(ParseError, match="clause 4 out of order in Điều 1"):
        parse_articles(CH1 + [
            "Điều 1. Giải thích",
            "1. Khoản một.",
            "“Người lao động” là người làm việc.x",
            "2. Khoản hai.",
            "3. Gọi là “bên”.",
            "4. Khoản bốn.",
        ])


def test_clause_number_going_backwards_raises():
    with pytest.raises(ParseError, match="clause 2a out of order in Điều 1"):
        parse_articles(CH1 + ["Điều 1. A", "1. Một.", "2. Hai.", "3. Ba.", "2a. Hai a."])


def test_lettered_clauses_after_their_number_are_in_order():
    arts = parse_articles(CH1 + ["Điều 1. A", "1. Một.", "1a. Một a.", "1b. Một b.", "2. Hai."])
    assert [c.label for c in arts[0].clauses] == ["1", "1a", "1b", "2"]
