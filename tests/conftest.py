import pytest
from docx import Document

from app.core.config import get_settings


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


SETTINGS_ENV_NAMES = (
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "EMBEDDING_BATCH_SIZE",
)


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch):
    """Keep unit tests independent of the developer's shell environment."""
    for name in SETTINGS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
