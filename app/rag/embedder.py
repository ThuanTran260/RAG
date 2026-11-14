"""Embedding interface and the sentence-transformers implementation.

Torch and sentence-transformers are imported only when a real model is loaded, so importing
this module (and every unit test that uses a fake embedder) stays fast.
"""

from typing import Protocol

import numpy as np

from app.core.config import Settings


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension), L2-normalised per row."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return one L2-normalised float32 vector of shape (dimension,)."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16) -> None:
        # Deferred import: loading torch takes seconds and is not needed to import this module.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        dimension = self._model.get_embedding_dimension()
        # bool is an int subclass, so reject it explicitly.
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"cannot determine the embedding dimension of {model_name}")
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
