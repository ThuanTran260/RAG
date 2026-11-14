"""Index command: embed every chunk and rebuild the Qdrant collection of the configured model."""

import argparse
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import REPO_ROOT, get_settings
from app.rag.embedder import build_embedder
from app.rag.store import QdrantUnavailableError, connect, qdrant_errors
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


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not use UTF-8; Vietnamese output must not crash.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Embed the chunks and index them into Qdrant.")
    parser.add_argument(
        "--chunks", type=Path, default=REPO_ROOT / "data" / "processed" / "chunks.jsonl"
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".cache" / "embeddings")
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

    chunks = load_chunks(args.chunks)
    embedder = build_embedder(settings)
    vectors = embed_with_cache([chunk.context_text for chunk in chunks], embedder, args.cache_dir)
    try:
        count = index_chunks(chunks, vectors, client, settings.collection_name)
    except QdrantUnavailableError as error:
        print(str(error), file=sys.stderr)
        print(
            "The collection may be incomplete; rerun: uv run python -m ingestion.index",
            file=sys.stderr,
        )
        return 1

    print(f"collection: {settings.collection_name}")
    print(f"points: {count}")
    print(f"dimension: {vectors.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
