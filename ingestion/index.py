"""Index command: embed every chunk and rebuild the Qdrant collection of the configured model.

With ``--hybrid`` it rebuilds the hybrid collection (the same dense vectors plus BM25 sparse
vectors) and leaves the dense-only collection alone.
"""

import argparse
import hashlib
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import HYBRID_SCHEMA_VERSION, REPO_ROOT, get_settings, model_key
from app.rag.embedder import EmbeddingModelUnavailableError, build_embedder
from app.rag.sparse import K1, TOKENIZER_VERSION, B, average_length, document_vector
from app.rag.store import (
    HYBRID_INDEX_COMMAND,
    INDEX_COMMAND,
    SPARSE_VECTOR_NAME,
    QdrantUnavailableError,
    connect,
    qdrant_errors,
    rejection_message,
)
from ingestion.embed_cache import embed_with_cache
from ingestion.models import Chunk

POINT_NAMESPACE = uuid.UUID("b7e2c1a4-5d3f-4e8a-9c6b-2f1a0d4e7c35")
UPSERT_BATCH_SIZE = 256


def point_id(chunk_id: str) -> str:
    """Qdrant accepts only integers and UUIDs as point ids, so derive a stable UUID."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as lines:
        return [Chunk.model_validate_json(line) for line in lines if line.strip()]


def _check_input(chunks: list[Chunk], vectors: np.ndarray) -> None:
    """Validate everything before the collection is touched, so bad input keeps the old one."""
    if not chunks:
        raise ValueError("no chunks to index")
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be a 2-D array (chunks x dimension), got {vectors.ndim}-D")
    if len(vectors) != len(chunks):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if vectors.shape[1] < 1:
        raise ValueError("vectors must have at least one column")
    if not np.issubdtype(vectors.dtype, np.floating):
        raise ValueError("vectors must be a floating-point array")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or infinite values")
    counts = Counter(chunk.id for chunk in chunks)
    duplicates = sorted(chunk_id for chunk_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate chunk ids: {', '.join(duplicates[:5])}")


def index_chunks(
    chunks: list[Chunk], vectors: np.ndarray, client: QdrantClient, collection: str
) -> int:
    """Rebuild ``collection`` from the chunks and return the number of stored points.

    A rebuild interrupted midway can leave the collection incomplete; run the command again.
    """
    _check_input(chunks, vectors)

    # One boundary around the whole write, so a transport failure at any step is reported.
    with qdrant_errors():
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=point_id(chunk.id),
                vector=vector.tolist(),
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            client.upsert(collection, points[start : start + UPSERT_BATCH_SIZE], wait=True)
        return client.count(collection, exact=True).count


def index_hybrid_chunks(
    chunks: list[Chunk],
    vectors: np.ndarray,
    client: QdrantClient,
    collection: str,
    *,
    model_name: str,
    chunks_sha256: str,
) -> dict[str, str | int | float]:
    """Rebuild ``collection`` with dense and BM25 vectors; return the metadata stored on it.

    Dense vectors and payloads are stored as ``index_chunks`` stores them. The metadata is written
    after the points, so a collection from an interrupted rebuild has none; run the command again.
    """
    _check_input(chunks, vectors)
    # Build every sparse vector before the collection is touched, as the input check is.
    avgdl = average_length(chunk.context_text for chunk in chunks)
    sparse = [document_vector(chunk.context_text, avgdl) for chunk in chunks]
    points = [
        PointStruct(
            id=point_id(chunk.id),
            vector={
                "": dense.tolist(),  # the unnamed dense vector
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=list(weights.indices), values=list(weights.values)
                ),
            },
            payload=chunk.model_dump(mode="json"),
        )
        for chunk, dense, weights in zip(chunks, vectors, sparse, strict=True)
    ]

    with qdrant_errors():
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
            # Qdrant applies the IDF part of BM25 to the query weights.
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
        )
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            client.upsert(collection, points[start : start + UPSERT_BATCH_SIZE], wait=True)
        metadata = {
            "schema_version": HYBRID_SCHEMA_VERSION,
            "model_key": model_key(model_name),
            "chunks_sha256": chunks_sha256,
            "points": client.count(collection, exact=True).count,
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": K1,
            "b": B,
            "avgdl": avgdl,
        }
        client.update_collection(collection, metadata=metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not use UTF-8; Vietnamese output must not crash.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Embed the chunks and index them into Qdrant.")
    parser.add_argument(
        "--chunks", type=Path, default=REPO_ROOT / "data" / "processed" / "chunks.jsonl"
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".cache" / "embeddings")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="build the hybrid (dense + BM25) collection and keep the dense-only one",
    )
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ValidationError as error:
        print(f"invalid settings: {error}", file=sys.stderr)
        return 1

    if not args.chunks.is_file():
        print(
            f"chunks file not found: {args.chunks}. Create it with: uv run python -m ingestion.build",
            file=sys.stderr,
        )
        return 1

    # Probe the server first: loading the model takes much longer than a refused connection.
    try:
        client = connect(settings)
    except QdrantUnavailableError as error:
        print(str(error), file=sys.stderr)
        return 1
    except UnexpectedResponse as error:
        print(rejection_message(error, searching=False), file=sys.stderr)
        return 1

    try:
        # Hashed next to the load, so the stored digest names the bytes that were indexed.
        chunks_sha256 = hashlib.sha256(args.chunks.read_bytes()).hexdigest()
        chunks = load_chunks(args.chunks)
        try:
            embedder = build_embedder(settings)
        except EmbeddingModelUnavailableError as error:
            print(str(error), file=sys.stderr)
            return 1
        vectors = embed_with_cache(
            [chunk.context_text for chunk in chunks], embedder, args.cache_dir
        )
        collection = settings.hybrid_collection_name if args.hybrid else settings.collection_name
        rerun = HYBRID_INDEX_COMMAND if args.hybrid else INDEX_COMMAND
        try:
            if args.hybrid:
                metadata = index_hybrid_chunks(
                    chunks,
                    vectors,
                    client,
                    collection,
                    model_name=embedder.model_name,
                    chunks_sha256=chunks_sha256,
                )
                count = metadata["points"]
            else:
                count = index_chunks(chunks, vectors, client, collection)
        except QdrantUnavailableError as error:
            print(str(error), file=sys.stderr)
            print(f"The collection may be incomplete; rerun: {rerun}", file=sys.stderr)
            return 1
        except UnexpectedResponse as error:
            print(rejection_message(error, searching=False), file=sys.stderr)
            print("The collection may be incomplete.", file=sys.stderr)
            return 1
    finally:
        client.close()

    print(f"collection: {collection}")
    print(f"points: {count}")
    print(f"dimension: {vectors.shape[1]}")
    if args.hybrid:
        print(f"sparse vector: {SPARSE_VECTOR_NAME} (modifier idf)")
        print(f"metadata: {json.dumps(metadata, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
