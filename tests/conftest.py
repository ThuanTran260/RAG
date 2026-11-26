import hashlib

import numpy as np
import pytest
from docx import Document

from app.core.config import Settings, get_settings
from app.rag.llm import LLMRequest, LLMResult
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
    "RETRIEVAL_MODE",
    "ARTICLE_PROMOTION",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
)


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch, request):
    """Keep unit tests independent of the developer's shell environment."""
    live_llm = request.node.get_closest_marker("llm") is not None
    for name in SETTINGS_ENV_NAMES:
        if live_llm and name.startswith("LLM_"):
            continue
        monkeypatch.delenv(name, raising=False)
    if not live_llm:
        monkeypatch.setitem(Settings.model_config, "env_file", None)
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


@pytest.fixture
def contexts_for(all_chunks):
    """Build RetrievedChunk values from real chunk ids, with fake scores that decrease."""
    from app.rag.retriever import RetrievedChunk

    by_id = {chunk.id: chunk for chunk in all_chunks}

    def _build(ids):
        return [
            RetrievedChunk(chunk=by_id[chunk_id], score=round(0.9 - 0.01 * rank, 2))
            for rank, chunk_id in enumerate(ids)
        ]

    return _build


class FakeLLMClient:
    """Counts calls and records requests; replies with a canned text or raises a canned error."""

    def __init__(
        self,
        reply: str | Exception,
        requested_model: str = "fake-model",
        response_model: str | None = "fake-model-resp",
    ) -> None:
        self.reply = reply
        self.requested_model = requested_model
        self.response_model = response_model
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResult:
        self.calls += 1
        self.requests.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResult(self.reply, self.requested_model, self.response_model)


@pytest.fixture
def make_fake_llm_client():
    return FakeLLMClient
