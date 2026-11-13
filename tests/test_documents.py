import hashlib

from ingestion.documents import LABOUR_CODE


def test_labour_code_file_matches_recorded_checksum():
    digest = hashlib.sha256(LABOUR_CODE.path.read_bytes()).hexdigest()
    assert digest == LABOUR_CODE.sha256


def test_labour_code_metadata():
    assert LABOUR_CODE.doc_id == "45-2019-QH14"
    assert LABOUR_CODE.doc_no == "45/2019/QH14"
    assert LABOUR_CODE.title == "Bộ luật Lao động"
    assert LABOUR_CODE.source_no == "18/VBHN-VPQH"
    assert LABOUR_CODE.expected_articles == 220
    assert LABOUR_CODE.path.is_absolute()
