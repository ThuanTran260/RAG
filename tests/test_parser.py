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
