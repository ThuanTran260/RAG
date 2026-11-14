"""Needs the real embedding model. Run with: uv run pytest -m model"""

import numpy as np
import pytest

from app.rag.embedder import SentenceTransformerEmbedder

pytestmark = [pytest.mark.integration, pytest.mark.model]


def test_model_dimension_and_normalisation():
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    vectors = embedder.embed_documents(["Thời gian thử việc", "Tiền lương làm thêm giờ"])
    assert embedder.model_name == "BAAI/bge-m3"
    assert embedder.dimension == 1024
    assert vectors.shape == (2, 1024)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    query = embedder.embed_query("Thời gian thử việc")
    assert query.shape == (1024,)
    assert float(query @ vectors[0]) > 0.99
    assert embedder.embed_documents([]).shape == (0, 1024)
