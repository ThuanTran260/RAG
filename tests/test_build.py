import json
import os
import subprocess
import sys

import pytest

from ingestion.build import build
from ingestion.documents import LABOUR_CODE, REPO_ROOT


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    stats = build(LABOUR_CODE, out)
    chunks = [
        json.loads(line) for line in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    articles = [
        json.loads(line)
        for line in (out / "articles.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return stats, {c["id"]: c for c in chunks}, chunks, articles, out


def test_counts_match_the_reference_numbers(built):
    stats, by_id, chunks, articles, _ = built
    assert (stats.articles, stats.chunks) == (220, 676)
    assert (stats.char_len_min, stats.char_len_median, stats.char_len_max) == (11, 189.5, 4512)
    assert stats.char_len_total == 172017
    assert len(articles) == 220
    assert len(by_id) == len(chunks) == 676
    assert sum(1 for c in chunks if c["clause"] is None) == 37
    assert len({c["chapter"] for c in chunks}) == 17
    assert len({(c["chapter"], c["section"]) for c in chunks if c["section"] is not None}) == 24


def test_footnote_digits_are_gone(built):
    by_id = built[1]
    assert by_id["45-2019-QH14:d61:k3"]["text"].startswith(
        "3. Người sử dụng lao động tuyển người vào học nghề"
    )
    assert by_id["45-2019-QH14:d139:k1"]["text"].startswith("1. Lao động nữ được nghỉ thai sản")
    assert by_id["45-2019-QH14:d220:k1"]["chapter_title"] == "ĐIỀU KHOẢN THI HÀNH"
    assert "\na) Thành lập cơ sở giáo dục nghề nghiệp" in by_id["45-2019-QH14:d59:k2"]["text"]


def test_added_clause_8a(built):
    chunks = built[2]
    labels = [c["clause"] for c in chunks if c["article"] == 154]
    assert labels == ["1", "2", "3", "4", "5", "6", "7", "8", "8a", "9"]


def test_quoted_amendments_stay_inside_article_219(built):
    by_id, chunks = built[1], built[2]
    assert [c["id"] for c in chunks if c["article"] == 219] == [
        "45-2019-QH14:d219:k1",
        "45-2019-QH14:d219:k2",
    ]
    assert "“Điều 54. Điều kiện hưởng lương hưu" in by_id["45-2019-QH14:d219:k1"]["text"]
    assert by_id["45-2019-QH14:d219:k1"]["char_len"] == 4512
    assert by_id["45-2019-QH14:d219:k2"]["text"].endswith("b) Bãi bỏ khoản 2 Điều 32.")


def test_multi_paragraph_headings(built):
    by_id = built[1]
    assert (
        by_id["45-2019-QH14:d214:k1"]["chapter_title"]
        == "THANH TRA LAO ĐỘNG, XỬ LÝ VI PHẠM PHÁP LUẬT VỀ LAO ĐỘNG"
    )
    assert (
        by_id["45-2019-QH14:d168:k1"]["chapter_title"]
        == "BẢO HIỂM XÃ HỘI, BẢO HIỂM Y TẾ, BẢO HIỂM THẤT NGHIỆP"
    )
    d154 = by_id["45-2019-QH14:d154:k1"]
    assert d154["section"] == 3
    assert d154["section_title"].endswith("LAO ĐỘNG LÀ NGƯỜI NƯỚC NGOÀI LÀM VIỆC TẠI VIỆT NAM")


def test_known_articles(built):
    by_id = built[1]
    assert by_id["45-2019-QH14:d1"]["text"].startswith(
        "Bộ luật Lao động quy định tiêu chuẩn lao động"
    )
    assert by_id["45-2019-QH14:d124:k4"]["text"] == "4. Sa thải."
    d35 = by_id["45-2019-QH14:d35:k1"]
    assert d35["article_title"] == "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động"
    assert (d35["chapter"], d35["section"], d35["section_title"]) == (
        "III",
        3,
        "CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG",
    )
    assert "a) Ít nhất 45 ngày" in d35["text"]


def test_output_is_utf8_jsonl_with_lf_newlines(built):
    raw = (built[4] / "chunks.jsonl").read_bytes()
    assert b"\r\n" not in raw
    assert "Điều".encode() in raw
    assert b"\\u" not in raw


def test_build_is_deterministic(built, tmp_path):
    build(LABOUR_CODE, tmp_path)
    for name in ("articles.jsonl", "chunks.jsonl"):
        assert (tmp_path / name).read_bytes() == (built[4] / name).read_bytes()


def test_checksum_mismatch_stops_before_writing(tmp_path):
    wrong = LABOUR_CODE.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="checksum mismatch"):
        build(wrong, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cli_survives_a_legacy_console_encoding(tmp_path):
    env = os.environ | {"PYTHONIOENCODING": "cp1252"}
    result = subprocess.run(
        [sys.executable, "-m", "ingestion.build", "--out", str(tmp_path), "--sample", "3"],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    out = result.stdout.decode("utf-8")
    assert "articles: 220" in out
    assert "chunks: 676" in out
    assert "char_len min/median/max: 11 / 189.5 / 4512" in out
    assert out.count("--- 45-2019-QH14:d") == 3
