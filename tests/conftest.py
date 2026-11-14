import hashlib

import numpy as np
import pytest
from docx import Document

from app.core.config import get_settings
from ingestion.documents import REPO_ROOT
from ingestion.models import Chunk


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


class FakeEmbedder:
    """Deterministic stand-in for a model: equal texts give equal unit vectors."""

    def __init__(self, model_name: str = "fake/model", dimension: int = 8) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.calls: list[list[str]] = []  # one entry per embed_documents call

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vector = np.random.default_rng(seed).standard_normal(self.dimension).astype(np.float32)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def make_fake_embedder():
    return FakeEmbedder


@pytest.fixture(scope="session")
def all_chunks() -> list[Chunk]:
    """Every chunk of the committed data/processed/chunks.jsonl."""
    path = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
    with path.open(encoding="utf-8") as lines:
        return [Chunk.model_validate_json(line) for line in lines]


@pytest.fixture
def sample_chunks(all_chunks) -> list[Chunk]:
    """The first 12 real chunks (Điều 1 to part of Điều 3, which ends mid-article)."""
    return all_chunks[:12]
