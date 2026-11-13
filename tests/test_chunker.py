from ingestion.chunker import build_article_records, build_chunks
from ingestion.documents import LABOUR_CODE
from ingestion.models import Article, Clause


def article(**overrides):
    base = {
        "number": 35,
        "title": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
        "chapter": "III",
        "chapter_title": "HỢP ĐỒNG LAO ĐỘNG",
        "section": 3,
        "section_title": "CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG",
        "intro": "",
        "clauses": [Clause(label="1", text="1. Người lao động có quyền.\na) Ít nhất 45 ngày;")],
    }
    return Article(**(base | overrides))


def test_clause_chunk_id_and_metadata():
    chunk = build_chunks([article()], LABOUR_CODE)[0]
    assert chunk.id == "45-2019-QH14:d35:k1"
    assert (chunk.doc_id, chunk.doc_no, chunk.doc_title) == (
        "45-2019-QH14",
        "45/2019/QH14",
        "Bộ luật Lao động",
    )
    assert chunk.source_no == "18/VBHN-VPQH"
    assert (chunk.chapter, chunk.section, chunk.article, chunk.clause) == ("III", 3, 35, "1")
    assert chunk.text == "1. Người lao động có quyền.\na) Ít nhất 45 ngày;"
    assert chunk.char_len == len(chunk.text)
    assert chunk.checked_at == LABOUR_CODE.checked_at


def test_context_text_layout_with_section():
    chunk = build_chunks([article()], LABOUR_CODE)[0]
    assert chunk.context_text == (
        "Bộ luật Lao động\n"
        "Chương III - HỢP ĐỒNG LAO ĐỘNG\n"
        "Mục 3 - CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG\n"
        "Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động\n"
        "1. Người lao động có quyền.\na) Ít nhất 45 ngày;"
    )


def test_context_text_includes_intro_and_omits_missing_section():
    a = article(
        number=25,
        title="Thời gian thử việc",
        section=None,
        section_title=None,
        intro="Thời gian thử việc do hai bên thỏa thuận:",
        clauses=[Clause(label="1", text="1. Không quá 180 ngày;")],
    )
    chunk = build_chunks([a], LABOUR_CODE)[0]
    assert chunk.context_text == (
        "Bộ luật Lao động\n"
        "Chương III - HỢP ĐỒNG LAO ĐỘNG\n"
        "Điều 25. Thời gian thử việc\n"
        "Thời gian thử việc do hai bên thỏa thuận:\n"
        "1. Không quá 180 ngày;"
    )


def test_article_without_clauses_is_one_chunk():
    a = article(
        number=1,
        title="Phạm vi điều chỉnh",
        chapter="I",
        chapter_title="NHỮNG QUY ĐỊNH CHUNG",
        section=None,
        section_title=None,
        intro="Bộ luật Lao động quy định.",
        clauses=[],
    )
    chunks = build_chunks([a], LABOUR_CODE)
    assert len(chunks) == 1
    assert (chunks[0].id, chunks[0].clause, chunks[0].text) == (
        "45-2019-QH14:d1",
        None,
        "Bộ luật Lao động quy định.",
    )
    assert chunks[0].context_text.endswith("Điều 1. Phạm vi điều chỉnh\nBộ luật Lao động quy định.")
    assert chunks[0].context_text.count("Bộ luật Lao động quy định.") == 1


def test_lettered_clause_id():
    a = article(
        number=154, clauses=[Clause(label="8a", text="8a. Nhân lực công nghiệp công nghệ số.")]
    )
    assert build_chunks([a], LABOUR_CODE)[0].id == "45-2019-QH14:d154:k8a"


def test_article_record_holds_the_full_text():
    a = article(
        intro="Lời dẫn:",
        clauses=[Clause(label="1", text="1. Một."), Clause(label="2", text="2. Hai.")],
    )
    record = build_article_records([a], LABOUR_CODE)[0]
    assert record.id == "45-2019-QH14:d35"
    assert record.text == (
        "Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động\n"
        "Lời dẫn:\n1. Một.\n2. Hai."
    )
