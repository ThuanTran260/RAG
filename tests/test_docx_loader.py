import unicodedata

import pytest

from ingestion.documents import LABOUR_CODE
from ingestion.docx_loader import load_paragraphs


def test_drops_superscript_footnote_digits(make_docx):
    path = make_docx([[("3.", False), ("3", True), (" Người sử dụng lao động", False)]])
    assert load_paragraphs(path) == ["3. Người sử dụng lao động"]


def test_normalises_decomposed_text_to_nfc(make_docx):
    decomposed = unicodedata.normalize("NFD", "Điều 1. Phạm vi điều chỉnh")
    assert load_paragraphs(make_docx([decomposed])) == ["Điều 1. Phạm vi điều chỉnh"]


def test_replaces_nbsp_and_collapses_whitespace(make_docx):
    assert load_paragraphs(make_docx(["a)\u00a0Làm  việc;\t "])) == ["a) Làm việc;"]


def test_skips_empty_paragraphs_and_table_content(make_docx):
    path = make_docx(["Điều 220. Hiệu lực thi hành", "", "   "], table_cells=("VĂN PHÒNG QUỐC HỘI",))
    assert load_paragraphs(path) == ["Điều 220. Hiệu lực thi hành"]


def test_rejects_file_that_is_not_docx(tmp_path):
    legacy = tmp_path / "old.doc"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    with pytest.raises(ValueError, match="not a .docx"):
        load_paragraphs(legacy)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_paragraphs(tmp_path / "missing.docx")


def test_real_labour_code_paragraphs():
    paragraphs = load_paragraphs(LABOUR_CODE.path)
    assert len(paragraphs) == 1377
    assert paragraphs[0] == "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
    assert paragraphs[10:13] == ["Chương I", "NHỮNG QUY ĐỊNH CHUNG", "Điều 1. Phạm vi điều chỉnh"]
    assert paragraphs[-1].startswith("3. Chế độ lao động đối với cán bộ, công chức")
    assert all(unicodedata.is_normalized("NFC", p) for p in paragraphs)
    assert not any("\u00a0" in p for p in paragraphs)
    assert not any("CÔNG BÁO" in p for p in paragraphs)  # page header must not leak into the body
