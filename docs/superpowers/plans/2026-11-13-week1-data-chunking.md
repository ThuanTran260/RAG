# Tuần 1 — Dữ liệu và chunking Bộ luật Lao động: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Từ file DOCX Bộ luật Lao động (bản hợp nhất 2026) sinh ra `data/processed/articles.jsonl` và `data/processed/chunks.jsonl` bằng một lệnh, có test bảo vệ.

**Architecture:** Pipeline 3 bước thuần Python, mỗi bước một module: `docx_loader` đọc DOCX thành danh sách đoạn văn đã làm sạch; `parser` là máy trạng thái biến danh sách đoạn thành cấu trúc Chương → Mục → Điều → Khoản; `chunker` biến cấu trúc đó thành bản ghi có metadata. `build` nối ba bước, kiểm tra checksum của file nguồn và ghi JSONL.

**Tech Stack:** Python 3.13 (tối thiểu 3.11), uv, python-docx 1.2.0, pydantic 2, pytest, ruff.

**Spec:** `KE_HOACH_RAG.md` (mục 4.1, mục 5 "Tuần 1", mục 6, mục 7 "Về dữ liệu và chunking").

## Global Constraints

- Python 3.11+; quản lý môi trường bằng `uv`; ghim phiên bản bằng `uv.lock` được commit.
- Không dùng LangChain/LlamaIndex.
- Không cắt chunk theo số ký tự cố định. Chunk theo Khoản; Điều không có khoản là một chunk.
- Mọi text phải ở dạng Unicode NFC.
- Mỗi chunk có đủ metadata: tên văn bản, số hiệu, chương, số điều, tên điều, số khoản.
- Đọc và ghi file luôn với `encoding="utf-8"`; file JSONL dùng xuống dòng `\n` và `ensure_ascii=False`.
- Không commit `.env` hay API key.
- Comment trong code, commit message, tên branch viết bằng tiếng Anh.
- Làm trên branch `week1-data-chunking`, mỗi task một commit, không push.
- Lệnh chạy test của mọi task: `uv run pytest`.

## Review Focus

Năm tình huống thực tế dễ gây lỗi nhất, mỗi tình huống đã có test ở task nêu trong ngoặc:

1. **Sai file nguồn** (số văn bản hợp nhất trùng giữa các năm, file tải lỗi): build phải dừng với lỗi checksum và không ghi output. (Task 1, Task 6)
2. **Văn bản thiếu hoặc nhảy số Điều**: parser phải báo lỗi nêu rõ số Điều mong đợi, không âm thầm cho ra kết quả thiếu. (Task 4)
3. **Điều luật của luật khác được trích trong ngoặc kép** (Điều 219): không được tạo Điều mới hay Khoản mới, không được sinh ID trùng. (Task 4, Task 6)
4. **Chạy lệnh build trên terminal Windows không dùng UTF-8**: không được crash khi in tiếng Việt. (Task 6)
5. **Chạy build hai lần**: output phải giống nhau từng byte. (Task 6)

## Cấu trúc file

| File | Trách nhiệm |
|---|---|
| `pyproject.toml`, `.python-version`, `uv.lock` | Môi trường và phụ thuộc |
| `data/raw/SOURCES.md` | Nguồn gốc, ngày tải, checksum của từng file gốc |
| `ingestion/models.py` | Các kiểu dữ liệu: `DocumentMeta`, `Clause`, `Article`, `Chunk`, `ArticleRecord` |
| `ingestion/documents.py` | Hằng `LABOUR_CODE` mô tả văn bản đang nạp |
| `ingestion/docx_loader.py` | DOCX → danh sách đoạn văn sạch |
| `ingestion/parser.py` | Danh sách đoạn văn → `list[Article]` |
| `ingestion/chunker.py` | `list[Article]` → `list[Chunk]` và `list[ArticleRecord]` |
| `ingestion/build.py` | Nối pipeline, ghi JSONL, in thống kê và mẫu |
| `tests/conftest.py` | Fixture tạo file DOCX nhỏ để test |
| `tests/test_*.py` | Một file test cho mỗi module |

## Số liệu chuẩn

Đo ngày 13/11/2026 bằng parser nháp trên `data/raw/2026_131_18_VBHN-VPQH.docx`. Các test tích hợp dùng đúng những số này.

- 1.377 đoạn văn không rỗng sau khi làm sạch; 10 đoạn đầu là phần mở đầu trước `Chương I`.
- 17 Chương, 24 Mục, 220 Điều đánh số liên tục 1–220.
- 639 Khoản; 37 Điều không có khoản; tổng 676 chunk.
- 12 Điều có lời dẫn trước khoản 1: 3, 25, 56, 58, 67, 99, 108, 120, 125, 187, 199, 205.
- Nhãn khoản có chữ cái duy nhất: `8a` ở Điều 154.
- Độ dài `text` của chunk: nhỏ nhất 11 (`4. Sa thải.`, khoản 4 Điều 124), trung vị 189,5, lớn nhất 4.512 (khoản 1 Điều 219), tổng 172.017 ký tự.
- Ba tiêu đề trải trên nhiều đoạn văn: Chương XII, Chương XVI, Mục 3 Chương XI.

## Ngoài phạm vi plan này

- **Nghị định 145/2020/NĐ-CP**: cần plan riêng. Còn chờ quyết định có loại phần cho thuê lại lao động hay không, cần bản từ nguồn chính thức, và file hiện có là `.doc` phải chuyển đổi.
- Trường `refs` (dẫn chiếu giữa các Điều) và `amended_by` (thông tin sửa đổi từ chú thích).
- Tách chunk tới mức Điểm, hoặc gộp các khoản quá ngắn như Điều 124. Đây là thí nghiệm của Tuần 5, dùng `articles.jsonl`.
- Embedding và Qdrant (Tuần 2).

---

### Task 1: Môi trường và khóa nguồn dữ liệu

**Files:**
- Create: `pyproject.toml`, `.python-version`, `ingestion/__init__.py` (rỗng), `ingestion/models.py`, `ingestion/documents.py`, `data/raw/SOURCES.md`, `tests/test_documents.py`
- Modify: `.gitignore` (thêm vào cuối file)

**Interfaces:**
- Produces: `ingestion.models.DocumentMeta`; `ingestion.documents.REPO_ROOT: Path`; `ingestion.documents.LABOUR_CODE: DocumentMeta`

- [ ] **Step 1: Tạo branch**

Run: `git switch -c week1-data-chunking`

- [ ] **Step 2: Tạo `pyproject.toml` và `.python-version`**

`pyproject.toml`:

```toml
[project]
name = "labor-law-rag"
version = "0.1.0"
description = "RAG assistant for Vietnamese labour law"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2,<3",
    "python-docx==1.2.0",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
```

`.python-version` chứa một dòng: `3.13`

Run: `uv sync`
Expected: tạo `.venv/` và `uv.lock`, không có lỗi.

- [ ] **Step 3: Thêm vào cuối `.gitignore`**

```gitignore

# Decree 145/2020 raw files: third-party copy, waiting for its own ingestion plan
data/raw/nghi-dinh-145-2020-*
```

- [ ] **Step 4: Viết test thất bại trong `tests/test_documents.py`**

```python
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
```

- [ ] **Step 5: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_documents.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'ingestion.documents'`

- [ ] **Step 6: Viết `DocumentMeta` trong `ingestion/models.py`**

Model pydantic với `model_config = ConfigDict(frozen=True)` và các trường:

```python
doc_id: str            # used as the prefix of chunk ids
doc_no: str
title: str
source_no: str
source_date: date
source_url: str
path: Path             # absolute path to the raw file
sha256: str
checked_at: date       # day the legal status was last checked
expected_articles: int
```

- [ ] **Step 7: Viết `ingestion/documents.py`**

`REPO_ROOT = Path(__file__).resolve().parents[1]` và hằng `LABOUR_CODE` với các giá trị:

| Trường | Giá trị |
|---|---|
| `doc_id` | `"45-2019-QH14"` |
| `doc_no` | `"45/2019/QH14"` |
| `title` | `"Bộ luật Lao động"` |
| `source_no` | `"18/VBHN-VPQH"` |
| `source_date` | `date(2026, 2, 12)` |
| `source_url` | `"https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-18-vbhn-vpqh-468971.htm"` |
| `path` | `REPO_ROOT / "data" / "raw" / "2026_131_18_VBHN-VPQH.docx"` |
| `sha256` | `"1386441b1f513defdd55186d7e65b8432dcac87c2e0d78676de25facb9c5e6ff"` |
| `checked_at` | `date(2026, 11, 13)` |
| `expected_articles` | `220` |

- [ ] **Step 8: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_documents.py -v`
Expected: 2 passed

- [ ] **Step 9: Tạo `data/raw/SOURCES.md`**

```markdown
# Nguồn dữ liệu gốc

Mỗi file trong thư mục này phải có một mục ở đây. Số văn bản hợp nhất đánh lại từ đầu mỗi năm (ví dụ "18/VBHN-VPQH" có ở cả 2024 và 2026), nên luôn đối chiếu ngày ký và tên văn bản, không chỉ số hiệu.

## 2026_131_18_VBHN-VPQH.docx

- Văn bản: Văn bản hợp nhất số 18/VBHN-VPQH ngày 12/02/2026 của Văn phòng Quốc hội, hợp nhất Bộ luật Lao động số 45/2019/QH14
- Đăng: Công báo số 131 ngày 28/02/2026
- Trang tải: https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-18-vbhn-vpqh-468971.htm
- Ngày tải và kiểm tra hiệu lực: 13/11/2026
- SHA256: `1386441b1f513defdd55186d7e65b8432dcac87c2e0d78676de25facb9c5e6ff`
- Cấu trúc đã kiểm: 17 Chương, 24 Mục, 220 Điều
- Đã hợp nhất sửa đổi từ: Luật Công nghiệp công nghệ số 71/2025/QH15 (hiệu lực 01/01/2026), Luật Dân số 113/2025/QH15 (hiệu lực 01/7/2026), Luật Giáo dục nghề nghiệp 124/2025/QH15 (hiệu lực 01/01/2026)
- Các chỗ đã sửa đổi theo chú thích của văn bản: điểm a khoản 2 Điều 59; khoản 3 Điều 61; khoản 1 Điều 139; khoản 8a Điều 154 (bổ sung)
- Tình trạng ngày 13/11/2026: còn hiệu lực. Dự án luật sửa đổi Bộ luật Lao động được Quốc hội thảo luận ngày 09/11/2026, chưa thấy thông qua (theo báo chí, chưa xác nhận trên quochoi.vn). Kiểm tra lại trước khi deploy.

## nghi-dinh-145-2020-*.doc và .md (chưa dùng, không commit)

- Nghị định 145/2020/NĐ-CP, nguyên văn năm 2020, tải từ luatvietnam.vn ngày 13/11/2026.
- Chưa nạp: chờ plan riêng. Bản chính thức: https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-145-2020-nd-cp-32732.htm
- Bản `.md` có lỗi chuyển đổi (mất 2 công thức ở Điều 55, dính đoạn ở 21 Điều), không dùng làm đầu vào.
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .python-version uv.lock .gitignore ingestion/__init__.py ingestion/models.py ingestion/documents.py data/raw/SOURCES.md data/raw/2026_131_18_VBHN-VPQH.docx tests/test_documents.py
git commit -m "chore: set up uv project and pin the Labour Code source file"
```

Kiểm tra: `git status --short` không còn liệt kê file `nghi-dinh-145-2020-*`.

---

### Task 2: Đọc DOCX thành danh sách đoạn văn sạch

**Files:**
- Create: `ingestion/docx_loader.py`, `tests/conftest.py`, `tests/test_docx_loader.py`

**Interfaces:**
- Consumes: `ingestion.documents.LABOUR_CODE`
- Produces: `ingestion.docx_loader.load_paragraphs(path: Path) -> list[str]`; fixture pytest `make_docx(paragraphs, table_cells=()) -> Path`

Quy tắc làm sạch, theo đúng thứ tự:

1. Chỉ lấy đoạn văn cấp thân văn bản (`Document.paragraphs`); nội dung trong bảng bị bỏ. Khối chữ ký cuối file nằm trong một bảng.
2. Bỏ mọi run có `run.font.superscript` là true. File thật có đúng 5 run như vậy, đều là số chú thích dính sau dấu khoản/điểm (`3.` + `3` + ` Người…`).
3. Nối text các run còn lại, chuẩn hóa `unicodedata.normalize("NFC", …)`.
4. Đổi `\u00a0` thành dấu cách, gộp mọi chuỗi khoảng trắng (`\s+`) thành một dấu cách, rồi `strip()`.
5. Nếu đoạn bắt đầu bằng nhãn điểm (`a)`, `đ)`) dính liền chữ phía sau (số chú thích ở giữa đã bị bỏ ở quy tắc 2), chèn một dấu cách sau nhãn: `a)Thành…` thành `a) Thành…`. Chỉ có đúng một đoạn như vậy trong file thật (Điều 59 khoản 2).
6. Bỏ đoạn rỗng.

- [ ] **Step 1: Viết fixture trong `tests/conftest.py`**

```python
import pytest
from docx import Document


@pytest.fixture
def make_docx(tmp_path):
    """Build a small .docx. Each paragraph is a str, or a list of (text, superscript) runs."""

    def _make(paragraphs, table_cells=()):
        doc = Document()
        for item in paragraphs:
            paragraph = doc.add_paragraph()
            runs = [(item, False)] if isinstance(item, str) else item
            for text, superscript in runs:
                paragraph.add_run(text).font.superscript = superscript
        if table_cells:
            table = doc.add_table(rows=1, cols=len(table_cells))
            for cell, text in zip(table.rows[0].cells, table_cells):
                cell.text = text
        path = tmp_path / "sample.docx"
        doc.save(path)
        return path

    return _make
```

- [ ] **Step 2: Viết test thất bại trong `tests/test_docx_loader.py`**

```python
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
    assert not any("C\u00d4NG B\u00c1O" in p for p in paragraphs)  # page header must not leak into the body
```

- [ ] **Step 3: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_docx_loader.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'ingestion.docx_loader'`

- [ ] **Step 4: Viết `load_paragraphs(path: Path) -> list[str]` trong `ingestion/docx_loader.py`**

Dùng `docx.Document(path)` và `paragraph.runs`. Kiểm tra `path.exists()` trước để ném `FileNotFoundError`; bắt `docx.opc.exceptions.PackageNotFoundError` và ném lại `ValueError(f"{path} is not a .docx file")`.

- [ ] **Step 5: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_docx_loader.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add ingestion/docx_loader.py tests/conftest.py tests/test_docx_loader.py
git commit -m "feat: load DOCX paragraphs with footnote digits and whitespace cleaned"
```

---

### Task 3: Parser — cấu trúc Chương, Mục, Điều, Khoản

**Files:**
- Create: `ingestion/parser.py`, `tests/test_parser.py`
- Modify: `ingestion/models.py` (thêm `Clause`, `Article`)

**Interfaces:**
- Consumes: danh sách đoạn văn từ `load_paragraphs`
- Produces: `ingestion.parser.parse_articles(paragraphs: list[str]) -> list[Article]`; `ingestion.models.Clause`, `ingestion.models.Article`

```python
class Clause(BaseModel):
    label: str   # "1", "8a"
    text: str    # every paragraph of the clause joined with "\n", label included


class Article(BaseModel):
    number: int
    title: str
    chapter: str               # roman numeral, e.g. "III"
    chapter_title: str
    section: int | None
    section_title: str | None
    intro: str                 # paragraphs before the first clause joined with "\n"; "" if none
    clauses: list[Clause]
```

Các regex cố định, đặt làm hằng ở đầu `ingestion/parser.py`:

```python
RE_CHAPTER = re.compile(r"^Chương ([IVXLC]+)$")
RE_SECTION = re.compile(r"^Mục (\d+)$")
RE_ARTICLE = re.compile(r"^Điều (\d+)\. (.+)$")
RE_CLAUSE = re.compile(r"^(\d+[a-z]?)\. ")
```

Quy tắc của máy trạng thái (đọc tuần tự từng đoạn):

1. Bỏ qua mọi đoạn trước dòng `Chương` đầu tiên. Phần mở đầu có các dòng `1. Luật…` trông giống khoản.
2. Sau một dòng `Chương` hoặc `Mục`, mọi đoạn cho tới dòng `Chương`/`Mục`/`Điều` kế tiếp đều là tiêu đề, nối với nhau bằng một dấu cách.
3. Dòng `Chương` mới đặt lại `section` và `section_title` về `None`.
4. Dòng `Điều` mở một Điều mới, mang theo Chương và Mục hiện tại.
5. Trong một Điều: đoạn khớp `RE_CLAUSE` mở khoản mới. Đoạn khác nối vào khoản đang mở, hoặc vào `intro` nếu chưa có khoản nào. Dòng điểm (`a)`, `đ)`) là đoạn nối tiếp của khoản.

- [ ] **Step 1: Viết test thất bại trong `tests/test_parser.py`**

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_parser.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'ingestion.parser'`

- [ ] **Step 3: Thêm `Clause`, `Article` vào `ingestion/models.py` và viết `parse_articles` trong `ingestion/parser.py`**

Viết thành một vòng lặp qua các đoạn với biến trạng thái tường minh (Chương hiện tại, Mục hiện tại, Điều đang mở, đang đọc tiêu đề hay không), theo đúng 5 quy tắc ở trên. Không dùng một regex lớn chạy trên toàn văn.

- [ ] **Step 4: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_parser.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/models.py ingestion/parser.py tests/test_parser.py
git commit -m "feat: parse chapters, sections, articles and clauses"
```

---

### Task 4: Parser — đoạn trích trong ngoặc kép và kiểm tra tính toàn vẹn

**Files:**
- Modify: `ingestion/parser.py`, `tests/test_parser.py`

**Interfaces:**
- Consumes: `parse_articles` từ Task 3
- Produces: `ingestion.parser.ParseError` (kế thừa `ValueError`); `parse_articles` giữ nguyên chữ ký, thêm hành vi

Quy tắc bổ sung:

6. **Khối trích dẫn.** Thêm hằng `RE_QUOTE_CLOSE = re.compile(r"”[.;]?$")`. Một đoạn bắt đầu bằng `“` mà không khớp `RE_QUOTE_CLOSE` thì mở khối trích dẫn. Khi khối đang mở, mọi đoạn (kể cả đoạn trông như `Điều`, `Chương`, khoản) đều nối vào khoản đang mở của Điều hiện tại, hoặc vào `intro` nếu Điều đó chưa có khoản. Đoạn khớp `RE_QUOTE_CLOSE` đóng khối. Quy tắc này được kiểm tra trước các quy tắc 2–5.
7. **Kiểm tra sau khi đọc xong**, ném `ParseError` khi:
   - không có Điều nào: thông báo `no articles found`;
   - số Điều không liên tục từ 1: thông báo `expected Điều {n}, found Điều {m}`;
   - một Điều có hai khoản trùng nhãn: thông báo `duplicate clause {label} in Điều {n}`;
   - số khoản trong một Điều không tăng dần (nhảy số hoặc lùi số): thông báo `clause {label} out of order in Điều {n}`;
   - hết văn bản mà khối trích dẫn chưa đóng: thông báo `unterminated quote in Điều {n}`.

- [ ] **Step 1: Thêm test thất bại vào cuối `tests/test_parser.py`**

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_parser.py -v`
Expected: lỗi khi thu thập test, `ImportError: cannot import name 'ParseError' from 'ingestion.parser'`

- [ ] **Step 3: Thêm `ParseError`, quy tắc 6 và quy tắc 7 vào `ingestion/parser.py`**

- [ ] **Step 4: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_parser.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/parser.py tests/test_parser.py
git commit -m "feat: keep quoted amendments inside their clause and validate numbering"
```

---

### Task 5: Chunker — từ Điều/Khoản ra bản ghi có metadata

**Files:**
- Create: `ingestion/chunker.py`, `tests/test_chunker.py`
- Modify: `ingestion/models.py` (thêm `Chunk`, `ArticleRecord`)

**Interfaces:**
- Consumes: `Article`, `Clause`, `DocumentMeta`, `LABOUR_CODE`
- Produces:
  - `ingestion.chunker.build_chunks(articles: list[Article], doc: DocumentMeta) -> list[Chunk]`
  - `ingestion.chunker.build_article_records(articles: list[Article], doc: DocumentMeta) -> list[ArticleRecord]`

```python
class Chunk(BaseModel):
    id: str                    # "45-2019-QH14:d35:k1", or "45-2019-QH14:d1" for an article with no clause
    doc_id: str
    doc_no: str
    doc_title: str
    source_no: str
    chapter: str
    chapter_title: str
    section: int | None
    section_title: str | None
    article: int
    article_title: str
    clause: str | None
    text: str                  # verbatim text to display
    context_text: str          # text to embed
    char_len: int              # len(text)
    checked_at: date


class ArticleRecord(BaseModel):
    id: str                    # "45-2019-QH14:d35"
    doc_id: str
    doc_no: str
    doc_title: str
    source_no: str
    chapter: str
    chapter_title: str
    section: int | None
    section_title: str | None
    article: int
    article_title: str
    text: str                  # full article: heading line, intro, then every clause
    checked_at: date
```

Quy tắc:

- Mỗi khoản là một chunk, `text` là `Clause.text`. Điều không có khoản là một chunk với `clause=None` và `text` là `Article.intro`.
- `context_text` gồm các dòng sau, nối bằng `\n`, bỏ dòng nào không có:
  1. `doc.title`
  2. `Chương {chapter} - {chapter_title}`
  3. `Mục {section} - {section_title}` (chỉ khi có Mục)
  4. `Điều {number}. {title}`
  5. `intro` (chỉ khi Điều có khoản và `intro` khác rỗng)
  6. `text`
- `ArticleRecord.text` là `Điều {number}. {title}`, rồi `intro` nếu có, rồi `text` của từng khoản, nối bằng `\n`.

- [ ] **Step 1: Viết test thất bại trong `tests/test_chunker.py`**

```python
from ingestion.chunker import build_article_records, build_chunks
from ingestion.documents import LABOUR_CODE
from ingestion.models import Article, Clause


def article(**overrides):
    base = dict(
        number=35, title="Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
        chapter="III", chapter_title="HỢP ĐỒNG LAO ĐỘNG",
        section=3, section_title="CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG",
        intro="", clauses=[Clause(label="1", text="1. Người lao động có quyền.\na) Ít nhất 45 ngày;")],
    )
    return Article(**(base | overrides))


def test_clause_chunk_id_and_metadata():
    chunk = build_chunks([article()], LABOUR_CODE)[0]
    assert chunk.id == "45-2019-QH14:d35:k1"
    assert (chunk.doc_id, chunk.doc_no, chunk.doc_title) == ("45-2019-QH14", "45/2019/QH14", "Bộ luật Lao động")
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
    a = article(number=25, title="Thời gian thử việc", section=None, section_title=None,
                intro="Thời gian thử việc do hai bên thỏa thuận:",
                clauses=[Clause(label="1", text="1. Không quá 180 ngày;")])
    chunk = build_chunks([a], LABOUR_CODE)[0]
    assert chunk.context_text == (
        "Bộ luật Lao động\n"
        "Chương III - HỢP ĐỒNG LAO ĐỘNG\n"
        "Điều 25. Thời gian thử việc\n"
        "Thời gian thử việc do hai bên thỏa thuận:\n"
        "1. Không quá 180 ngày;"
    )


def test_article_without_clauses_is_one_chunk():
    a = article(number=1, title="Phạm vi điều chỉnh", chapter="I", chapter_title="NHỮNG QUY ĐỊNH CHUNG",
                section=None, section_title=None, intro="Bộ luật Lao động quy định.", clauses=[])
    chunks = build_chunks([a], LABOUR_CODE)
    assert len(chunks) == 1
    assert (chunks[0].id, chunks[0].clause, chunks[0].text) == ("45-2019-QH14:d1", None, "Bộ luật Lao động quy định.")
    assert chunks[0].context_text.endswith("Điều 1. Phạm vi điều chỉnh\nBộ luật Lao động quy định.")
    assert chunks[0].context_text.count("Bộ luật Lao động quy định.") == 1


def test_lettered_clause_id():
    a = article(number=154, clauses=[Clause(label="8a", text="8a. Nhân lực công nghiệp công nghệ số.")])
    assert build_chunks([a], LABOUR_CODE)[0].id == "45-2019-QH14:d154:k8a"


def test_article_record_holds_the_full_text():
    a = article(intro="Lời dẫn:", clauses=[Clause(label="1", text="1. Một."), Clause(label="2", text="2. Hai.")])
    record = build_article_records([a], LABOUR_CODE)[0]
    assert record.id == "45-2019-QH14:d35"
    assert record.text == (
        "Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động\n"
        "Lời dẫn:\n1. Một.\n2. Hai."
    )
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'ingestion.chunker'`

- [ ] **Step 3: Thêm `Chunk`, `ArticleRecord` vào `ingestion/models.py` và viết hai hàm trong `ingestion/chunker.py`**

- [ ] **Step 4: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/models.py ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: build clause-level chunks with article context"
```

---

### Task 6: Lệnh build, kiểm tra trên văn bản thật, và tài liệu

**Files:**
- Create: `ingestion/build.py`, `tests/test_build.py`, `data/processed/articles.jsonl`, `data/processed/chunks.jsonl` (hai file sau do lệnh build sinh ra)
- Modify: `README.md`, `KE_HOACH_RAG.md` (dòng 67, 73, 95–98)

**Interfaces:**
- Consumes: `load_paragraphs`, `parse_articles`, `ParseError`, `build_chunks`, `build_article_records`, `LABOUR_CODE`
- Produces:
  - `ingestion.build.BuildStats` (pydantic): `articles: int`, `chunks: int`, `char_len_min: int`, `char_len_median: float`, `char_len_max: int`, `char_len_total: int`
  - `ingestion.build.build(doc: DocumentMeta, out_dir: Path) -> BuildStats`
  - `ingestion.build.main(argv: list[str] | None = None) -> int`
  - Lệnh: `uv run python -m ingestion.build [--out DIR] [--sample N] [--seed N]`

Hành vi của `build`:

1. Tính SHA256 của `doc.path`. Nếu khác `doc.sha256` thì ném `ValueError` có chữ `checksum mismatch` và tên file, trước khi tạo bất kỳ file output nào.
2. `load_paragraphs` → `parse_articles`. Nếu số Điều khác `doc.expected_articles` thì ném `ParseError` với thông báo `expected {n} articles, found {m}`.
3. Ghi `out_dir/articles.jsonl` và `out_dir/chunks.jsonl`: mỗi dòng là `model_dump_json()` của một bản ghi, theo thứ tự trong văn bản. Mở file với `encoding="utf-8", newline="\n"`. Tạo `out_dir` nếu chưa có.
4. Trả về `BuildStats` tính từ `char_len` của các chunk.

Hành vi của `main`:

- Việc đầu tiên là `sys.stdout.reconfigure(encoding="utf-8")`.
- Mặc định `--out` là `REPO_ROOT / "data" / "processed"`, `--sample` là `0`, `--seed` là `0`.
- In đúng ba dòng thống kê:

```
articles: 220
chunks: 676
char_len min/median/max: 11 / 189.5 / 4512
```

- Nếu `--sample N` lớn hơn 0: chọn `random.Random(seed).sample(chunks, N)` và với mỗi chunk in một dòng `--- {id}` rồi `context_text`.
- Trả về `0`.

- [ ] **Step 1: Viết test thất bại trong `tests/test_build.py`**

```python
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
    chunks = [json.loads(line) for line in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    articles = [json.loads(line) for line in (out / "articles.jsonl").read_text(encoding="utf-8").splitlines()]
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
    assert by_id["45-2019-QH14:d61:k3"]["text"].startswith("3. Người sử dụng lao động tuyển người vào học nghề")
    assert by_id["45-2019-QH14:d139:k1"]["text"].startswith("1. Lao động nữ được nghỉ thai sản")
    assert by_id["45-2019-QH14:d220:k1"]["chapter_title"] == "ĐIỀU KHOẢN THI HÀNH"


def test_added_clause_8a(built):
    chunks = built[2]
    labels = [c["clause"] for c in chunks if c["article"] == 154]
    assert labels == ["1", "2", "3", "4", "5", "6", "7", "8", "8a", "9"]


def test_quoted_amendments_stay_inside_article_219(built):
    by_id, chunks = built[1], built[2]
    assert [c["id"] for c in chunks if c["article"] == 219] == ["45-2019-QH14:d219:k1", "45-2019-QH14:d219:k2"]
    assert "“Điều 54. Điều kiện hưởng lương hưu" in by_id["45-2019-QH14:d219:k1"]["text"]
    assert by_id["45-2019-QH14:d219:k1"]["char_len"] == 4512
    assert by_id["45-2019-QH14:d219:k2"]["text"].endswith("b) Bãi bỏ khoản 2 Điều 32.")


def test_multi_paragraph_headings(built):
    by_id = built[1]
    assert by_id["45-2019-QH14:d214:k1"]["chapter_title"] == "THANH TRA LAO ĐỘNG, XỬ LÝ VI PHẠM PHÁP LUẬT VỀ LAO ĐỘNG"
    assert by_id["45-2019-QH14:d168:k1"]["chapter_title"] == "BẢO HIỂM XÃ HỘI, BẢO HIỂM Y TẾ, BẢO HIỂM THẤT NGHIỆP"
    d154 = by_id["45-2019-QH14:d154:k1"]
    assert d154["section"] == 3
    assert d154["section_title"].endswith("LAO ĐỘNG LÀ NGƯỜI NƯỚC NGOÀI LÀM VIỆC TẠI VIỆT NAM")


def test_known_articles(built):
    by_id = built[1]
    assert by_id["45-2019-QH14:d1"]["text"].startswith("Bộ luật Lao động quy định tiêu chuẩn lao động")
    assert by_id["45-2019-QH14:d124:k4"]["text"] == "4. Sa thải."
    d35 = by_id["45-2019-QH14:d35:k1"]
    assert d35["article_title"] == "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động"
    assert (d35["chapter"], d35["section"], d35["section_title"]) == ("III", 3, "CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG")
    assert "a) Ít nhất 45 ngày" in d35["text"]


def test_output_is_utf8_jsonl_with_lf_newlines(built):
    raw = (built[4] / "chunks.jsonl").read_bytes()
    assert b"\r\n" not in raw
    assert "Điều".encode("utf-8") in raw
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
        cwd=REPO_ROOT, env=env, capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    out = result.stdout.decode("utf-8")
    assert "articles: 220" in out
    assert "chunks: 676" in out
    assert "char_len min/median/max: 11 / 189.5 / 4512" in out
    assert out.count("--- 45-2019-QH14:d") == 3
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_build.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'ingestion.build'`

- [ ] **Step 3: Viết `BuildStats`, `build` và `main` trong `ingestion/build.py`**

Kết thúc file bằng `if __name__ == "__main__": raise SystemExit(main())`. Dùng `statistics.median` cho trung vị.

- [ ] **Step 4: Chạy toàn bộ test**

Run: `uv run pytest -v`
Expected: 41 passed, 0 failed (Task 1: 2, Task 2: 7, Task 3–4: 13, Task 5: 6, Task 6: 10)

Nếu một số chuẩn trong `test_counts_match_the_reference_numbers` lệch, đừng sửa số trong test. Hãy tìm quy tắc nào của loader hoặc parser đang khác với mô tả ở Task 2–4.

- [ ] **Step 5: Kiểm tra lint**

Run: `uv run ruff check ingestion tests` rồi `uv run ruff format --check ingestion tests`
Expected: `All checks passed!` và không file nào cần format lại. Nếu có, chạy `uv run ruff format ingestion tests`.

- [ ] **Step 6: Sinh output thật và soát 20 chunk bằng mắt**

Run: `uv run python -m ingestion.build --sample 20`
Expected: ba dòng thống kê như trên, rồi 20 chunk.

Đọc từng chunk và kiểm tra: dòng Chương/Mục/Điều đúng với nội dung; không có số chú thích dính vào chữ; khoản không bị cắt giữa chừng; không lẫn tiêu đề viết hoa của Chương sau vào cuối khoản.

- [ ] **Step 7: Cập nhật `README.md`**

Thay dòng `**Trạng thái:** …` và thêm các mục sau, viết bằng tiếng Việt:

- Trạng thái: xong Tuần 1 (dữ liệu và chunking).
- Cách chạy: `uv sync`, `uv run pytest`, `uv run python -m ingestion.build --sample 20`.
- Kết quả: 220 Điều, 676 chunk, độ dài nhỏ nhất/trung vị/lớn nhất 11 / 189,5 / 4.512 ký tự.
- Dữ liệu: bản hợp nhất 18/VBHN-VPQH ngày 12/02/2026, kiểm tra hiệu lực ngày 13/11/2026, chi tiết ở `data/raw/SOURCES.md`.
- Hạn chế đã biết:
  1. Chưa nạp Nghị định 145/2020/NĐ-CP.
  2. Khoản quá ngắn như các khoản của Điều 124 thiếu nghĩa khi đứng riêng; sẽ so sánh chunk theo Khoản với theo Điều ở Tuần 5.
  3. Khoản 1 và 2 Điều 219 chứa nguyên văn sửa đổi của luật khác, có thể gây nhiễu khi truy xuất.
  4. Chưa trích dẫn chiếu giữa các Điều và chưa lưu thông tin sửa đổi vào metadata.

- [ ] **Step 8: Sửa `KE_HOACH_RAG.md` cho khớp thực tế đã kiểm chứng**

Dòng 67, thay cả hàng `UTS_VLC` bằng:

```markdown
| **`undertheseanlp/UTS_VLC`** (Hugging Face) | Bộ dữ liệu Hiến pháp, Bộ luật, Luật Việt Nam | **Không dùng cho Bộ luật Lao động**: kiểm tra ngày 12/11/2026, bản ghi `45/2019/QH14` (split `2026`) chứa nhầm nội dung văn bản khác |
```

Dòng 73, thay bằng:

```markdown
1. Bộ luật Lao động 2019 (45/2019/QH14), dùng bản hợp nhất 18/VBHN-VPQH ngày 12/02/2026 tải từ Công báo (xem `data/raw/SOURCES.md`)
```

Dòng 95–98: đổi bốn ô `- [ ]` của Tuần 1 thành `- [x]`.

- [ ] **Step 9: Commit và gắn tag**

```bash
git add ingestion/build.py tests/test_build.py data/processed/articles.jsonl data/processed/chunks.jsonl README.md KE_HOACH_RAG.md
git commit -m "feat: add build command that writes articles.jsonl and chunks.jsonl"
git tag v0.1-week1
```

Kiểm tra: `git status --short` không còn file nào ngoài hai file Nghị định 145 đã bị ignore, và `git log --oneline` có 6 commit mới trên branch `week1-data-chunking`.
