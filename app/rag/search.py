"""Search command: uv run python -m app.rag.search "question" [-k N]"""

import argparse
import sys

from pydantic import ValidationError
from qdrant_client.http.exceptions import UnexpectedResponse

from app.rag.retriever import (
    CollectionNotIndexedError,
    InvalidCollectionDataError,
    InvalidQueryError,
    retrieve,
)
from app.rag.store import QdrantUnavailableError, rejection_message

PREVIEW_CHARS = 160


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not use UTF-8; Vietnamese output must not crash.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search the labour law chunks in Qdrant.")
    parser.add_argument("query", help="the question to search for")
    parser.add_argument("-k", type=int, default=5, help="number of results (default: 5)")
    args = parser.parse_args(argv)

    try:
        results = retrieve(args.query, args.k)
    except (
        QdrantUnavailableError,
        CollectionNotIndexedError,
        InvalidCollectionDataError,
        InvalidQueryError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    except UnexpectedResponse as error:
        print(rejection_message(error, searching=True), file=sys.stderr)
        return 1
    except ValidationError as error:
        print(f"invalid settings: {error}", file=sys.stderr)
        return 1

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview = " ".join(chunk.text.split())[:PREVIEW_CHARS]
        print(
            f"{rank}. {result.score:.4f}  {chunk.id}  Điều {chunk.article}. {chunk.article_title}"
        )
        print(f"   {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
