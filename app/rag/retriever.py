"""Dense retrieval: embed a question and return the nearest chunks with their scores."""

import unicodedata
from collections.abc import Callable
from functools import lru_cache
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.rag.embedder import Embedder, build_embedder
from app.rag.store import connect, qdrant_errors
from ingestion.models import Chunk


class InvalidQueryError(ValueError):
    """The question or k cannot be used for a search."""


class CollectionNotIndexedError(RuntimeError):
    """The Qdrant collection for the configured model does not exist or holds no points."""


class InvalidCollectionDataError(RuntimeError):
    """A stored payload does not match the current Chunk schema."""


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity; higher is closer


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


def prepare_query(query: str, k: int) -> str:
    """Normalise the question to NFC and strip it; reject an empty question or k < 1."""
    query = unicodedata.normalize("NFC", query).strip()
    if not query:
        raise InvalidQueryError("query is empty")
    if k < 1:
        raise InvalidQueryError("k must be at least 1")
    return query


class LazyEmbedder:
    """Builds the real embedder on first use, so checks that fail early never load the model."""

    def __init__(self, factory: Callable[[], Embedder]) -> None:
        self._factory = factory
        self._embedder: Embedder | None = None

    def embed_query(self, text: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = self._factory()
        return self._embedder.embed_query(text)


class Retriever:
    def __init__(self, client: QdrantClient, embedder: QueryEmbedder, collection: str) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        query = prepare_query(query, k)
        # One boundary around the collection check and the search: only Qdrant failures are
        # translated; embedding errors and CollectionNotIndexedError pass through unchanged.
        with qdrant_errors():
            if not self.client.collection_exists(self.collection):
                raise CollectionNotIndexedError(
                    f"collection {self.collection} does not exist. "
                    "Create it with: uv run python -m ingestion.index"
                )
            # The collection check comes first so that a missing index never loads the model.
            vector = self.embedder.embed_query(query)
            hits = self.client.query_points(
                self.collection, query=vector.tolist(), limit=k, with_payload=True
            ).points
        # There are no filters or thresholds, so no hit at all means the collection has no points.
        if not hits:
            raise CollectionNotIndexedError(
                f"collection {self.collection} is empty. "
                "Fill it with: uv run python -m ingestion.index"
            )
        return [RetrievedChunk(chunk=self._to_chunk(hit.payload), score=hit.score) for hit in hits]

    def _to_chunk(self, payload: dict | None) -> Chunk:
        try:
            return Chunk.model_validate(payload)
        except ValidationError as error:
            # Name no field value: the payload is stored data, not something to echo.
            raise InvalidCollectionDataError(
                f"a payload in collection {self.collection} is incompatible with the current "
                "chunk schema. Rebuild it with: uv run python -m ingestion.index"
            ) from error


@lru_cache
def default_retriever() -> Retriever:
    settings = get_settings()
    return Retriever(
        connect(settings),
        LazyEmbedder(lambda: build_embedder(settings)),
        settings.collection_name,
    )


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    # Validate first, so a bad question or k never opens a connection or loads the model.
    prepare_query(query, k)
    return default_retriever().retrieve(query, k)
