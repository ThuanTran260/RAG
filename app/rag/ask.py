"""Ask command: uv run python -m app.rag.ask "question" [-k N]"""

import argparse
import sys

from pydantic import ValidationError
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.rag.embedder import EmbeddingModelUnavailableError
from app.rag.generate import AnswerValidationError, generate, render_result
from app.rag.llm import LLMError, build_llm_client
from app.rag.prompts import MAX_CONTEXTS, GenerationInputError
from app.rag.retriever import (
    CollectionNotIndexedError,
    InvalidCollectionDataError,
    InvalidQueryError,
    prepare_query,
    retrieve,
)
from app.rag.store import QdrantUnavailableError, rejection_message


def settings_error_line(error: ValidationError) -> str:
    """One line naming the offending variables only; never the values (they may be secrets)."""
    names = sorted({str(item["loc"][0]).upper() for item in error.errors() if item["loc"]})
    return (
        f"invalid settings: {', '.join(names) or 'unknown'}. "
        "Check these variables in the environment or .env."
    )


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not use UTF-8; Vietnamese output must not crash.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Answer a labour law question from the retrieved chunks, with citations."
    )
    parser.add_argument("query", help="the question to answer")
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help=f"number of chunks to retrieve, 1 to {MAX_CONTEXTS} (default: 5)",
    )
    args = parser.parse_args(argv)

    try:
        # Cheap checks first, so a bad question or k never reads settings or opens a connection.
        query = prepare_query(args.query, args.k)
        if args.k > MAX_CONTEXTS:
            raise InvalidQueryError(f"k must be at most {MAX_CONTEXTS}")
        try:
            settings = get_settings()
        except ValidationError as error:
            print(settings_error_line(error), file=sys.stderr)
            return 1
        # A missing key stops here, before the embedding model is loaded or Qdrant is called.
        client = build_llm_client(settings)
        contexts = retrieve(query, args.k)
        result = generate(query, contexts, client)
    except (
        InvalidQueryError,
        QdrantUnavailableError,
        CollectionNotIndexedError,
        InvalidCollectionDataError,
        EmbeddingModelUnavailableError,
        GenerationInputError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    except UnexpectedResponse as error:
        print(rejection_message(error, searching=True), file=sys.stderr)
        return 1
    except LLMError as error:
        print(f"LLM: {error}", file=sys.stderr)
        return 1
    except AnswerValidationError as error:
        print(f"invalid model answer: {error.code}", file=sys.stderr)
        return 1

    print(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
