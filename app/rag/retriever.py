"""Retrieval over one Qdrant collection: dense, sparse (BM25) or hybrid (RRF of the two).

``score`` is the dense cosine similarity in every mode. Sparse and hybrid modes need a hybrid
collection and query it at full depth, so that every chunk has a cosine and a dense rank.

With ``promote`` on, dense and hybrid modes move chunks of the articles the question names to the
top of the results, so the results are then no longer sorted by ``score``.
"""

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol, get_args

import numpy as np
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    ScoredPoint,
    SearchParams,
    SparseVector,
)

from app.core.config import get_settings
from app.rag.embedder import Embedder, build_embedder
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.sparse import query_vector
from app.rag.store import (
    HYBRID_INDEX_COMMAND,
    INDEX_COMMAND,
    SPARSE_VECTOR_NAME,
    connect,
    hybrid_collection_problem,
    qdrant_errors,
)
from ingestion.models import Chunk

RetrievalMode = Literal["dense", "sparse", "hybrid"]
RETRIEVAL_MODES: tuple[RetrievalMode, ...] = get_args(RetrievalMode)
MAX_PROMOTED = 3  # chunks moved up per question, across all the articles it names


class InvalidQueryError(ValueError):
    """The question or k cannot be used for a search."""


class CollectionNotIndexedError(RuntimeError):
    """The Qdrant collection to search does not exist or holds no points."""


class InvalidCollectionDataError(RuntimeError):
    """The collection cannot be searched: its layout, metadata or a payload does not fit."""


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # dense cosine similarity in every mode; higher is closer
    fusion_score: float | None = None  # RRF score; None unless two rankings were fused
    dense_rank: int | None = None  # 1-based, in the full dense ranking; None in dense mode
    sparse_rank: int | None = None  # 1-based, among the BM25 hits; None without a hit


@dataclass(frozen=True)
class Ranking:
    """One search before the cut to k."""

    # Best first. Dense mode asks for k chunks only; sparse mode holds every BM25 hit and
    # hybrid mode every chunk of the collection.
    results: list[RetrievedChunk]
    dense_vector: list[float]  # the query embedding, for further dense queries


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
    def __init__(
        self,
        client: QdrantClient,
        embedder: QueryEmbedder,
        collection: str,
        mode: RetrievalMode = "dense",
        *,
        promote: bool = False,
    ) -> None:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode: {mode!r}")
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.mode = mode
        self.promote = promote  # ignored in sparse mode, which shows BM25 hits only
        # Named in error messages: sparse and hybrid modes need the hybrid build.
        self._index_command = INDEX_COMMAND if mode == "dense" else HYBRID_INDEX_COMMAND

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """The best ``k`` chunks; with ``promote`` on, those of the named articles come first.

        At most max(0, min(3, k - 2)) chunks are promoted, taken in the order of the full
        ranking. The top ``k`` of that ranking fills the other places.
        """
        ranking = self.rank(query, k)
        top = ranking.results[:k]
        slots = max(0, min(MAX_PROMOTED, k - 2))
        if not self.promote or self.mode == "sparse" or slots == 0:
            return top
        # Imported here because of a cycle: article_refs imports generate, which imports prompts,
        # which imports this module.
        from app.rag.article_refs import article_refs

        articles = article_refs(prepare_query(query, k))
        if not articles:
            return top
        if self.mode == "dense":  # its ranking stops at k, so ask for the named articles
            promoted = self._article_hits(ranking.dense_vector, articles, slots)
        else:
            promoted = [r for r in ranking.results if r.chunk.article in articles][:slots]
        taken = {result.chunk.id for result in promoted}
        return (promoted + [r for r in top if r.chunk.id not in taken])[:k]

    def rank(self, query: str, k: int = 5) -> Ranking:
        """Search for ``query``; return the ranking before promotion and before the cut to ``k``."""
        query = prepare_query(query, k)
        full_depth = self.mode != "dense"
        # One boundary around every Qdrant call: only Qdrant failures are translated; embedding
        # errors and the collection errors raised here pass through unchanged.
        with qdrant_errors():
            if not self.client.collection_exists(self.collection):
                raise CollectionNotIndexedError(
                    f"collection {self.collection} does not exist. "
                    f"Create it with: {self._index_command}"
                )
            # The collection checks come first, so an unusable index never loads the model.
            depth = self._point_count() if full_depth else k
            vector = self.embedder.embed_query(query).tolist()
            dense_hits = self._dense_hits(vector, depth, exact=full_depth)
            # There are no filters or thresholds, so no hit at all means there are no points.
            if not dense_hits:
                raise CollectionNotIndexedError(
                    f"collection {self.collection} is empty. Fill it with: {self._index_command}"
                )
            sparse_hits = self._sparse_hits(query, depth) if full_depth else []
        if full_depth:
            return Ranking(self._rank_with_bm25(dense_hits, sparse_hits), vector)
        results = [
            RetrievedChunk(chunk=self._to_chunk(hit.payload), score=hit.score) for hit in dense_hits
        ]
        return Ranking(results, vector)

    def _point_count(self) -> int:
        """Check that the collection can serve BM25 and return the depth of a full ranking."""
        info = self.client.get_collection(self.collection)
        problem = hybrid_collection_problem(info, self.collection)
        if problem is not None:
            raise InvalidCollectionDataError(problem)
        return info.config.metadata["points"]

    def _dense_hits(self, vector: list[float], limit: int, *, exact: bool) -> list[ScoredPoint]:
        # A full ranking must hold every point, so it must not depend on the HNSW index.
        extra = {"search_params": SearchParams(exact=True)} if exact else {}
        return self.client.query_points(
            self.collection, query=vector, limit=limit, with_payload=True, **extra
        ).points

    def _article_hits(
        self, vector: list[float], articles: set[int], limit: int
    ) -> list[RetrievedChunk]:
        """The ``limit`` chunks of ``articles`` closest to the query vector."""
        only_named = Filter(
            must=[FieldCondition(key="article", match=MatchAny(any=sorted(articles)))]
        )
        with qdrant_errors():
            hits = self.client.query_points(
                self.collection,
                query=vector,
                query_filter=only_named,
                limit=limit,
                with_payload=True,
                # Exact, so the order is that of the full ranking whatever the HNSW index does.
                search_params=SearchParams(exact=True),
            ).points
        results = [
            RetrievedChunk(chunk=self._to_chunk(hit.payload), score=hit.score) for hit in hits
        ]
        # Equal cosines among the hits are ordered by chunk id, as in the BM25 modes.
        return sorted(results, key=lambda result: (-result.score, result.chunk.id))

    def _sparse_hits(self, query: str, depth: int) -> list[ScoredPoint]:
        """Every chunk with a positive BM25 score, ids and scores only."""
        terms = query_vector(query)
        if not terms.indices:  # Qdrant is never sent an empty sparse vector
            return []
        hits = self.client.query_points(
            self.collection,
            query=SparseVector(indices=list(terms.indices), values=list(terms.values)),
            using=SPARSE_VECTOR_NAME,
            limit=depth,
            with_payload=False,  # the dense hits carry the payloads
        ).points
        return [hit for hit in hits if hit.score > 0]

    def _rank_with_bm25(
        self, dense_hits: list[ScoredPoint], sparse_hits: list[ScoredPoint]
    ) -> list[RetrievedChunk]:
        """The BM25 hits in sparse mode; the fused ranking of every chunk in hybrid mode."""
        scored = [(hit.id, self._to_chunk(hit.payload), hit.score) for hit in dense_hits]
        # Equal cosines are ordered by chunk id, so no rank depends on the server's order.
        scored.sort(key=lambda item: (-item[2], item[1].id))
        by_point = {
            point: RetrievedChunk(chunk=chunk, score=score, dense_rank=rank)
            for rank, (point, chunk, score) in enumerate(scored, start=1)
        }
        if any(hit.id not in by_point for hit in sparse_hits):
            raise InvalidCollectionDataError(
                f"collection {self.collection} holds more points than its metadata says. "
                f"Rebuild it with: {self._index_command}"
            )
        # Equal BM25 scores are ordered by dense rank, for the same reason.
        ordered = sorted(sparse_hits, key=lambda hit: (-hit.score, by_point[hit.id].dense_rank))
        matched = [
            by_point[hit.id].model_copy(update={"sparse_rank": rank})
            for rank, hit in enumerate(ordered, start=1)
        ]
        if self.mode == "sparse":
            return matched  # no dense fallback: this mode shows BM25 hits only
        dense = list(by_point.values())
        if not matched:
            return dense  # nothing to fuse, so the dense ranking stands
        by_id = {result.chunk.id: result for result in dense}
        fused = reciprocal_rank_fusion(list(by_id), [result.chunk.id for result in matched])
        return [
            by_id[item.key].model_copy(
                update={"fusion_score": item.score, "sparse_rank": item.sparse_rank}
            )
            for item in fused
        ]

    def _to_chunk(self, payload: dict | None) -> Chunk:
        try:
            return Chunk.model_validate(payload)
        except ValidationError as error:
            # Name no field value: the payload is stored data, not something to echo.
            raise InvalidCollectionDataError(
                f"a payload in collection {self.collection} is incompatible with the current "
                f"chunk schema. Rebuild it with: {self._index_command}"
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
