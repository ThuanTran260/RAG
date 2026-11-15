"""Download command: uv run python -m ingestion.download_model

The one command that contacts the Hugging Face Hub. It puts the configured embedding model in
the local cache; `ingestion.index` and `app.rag.search` then load it from there, offline.
"""

import argparse
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import ValidationError

from app.core.config import get_settings
from app.rag.embedder import (
    DOWNLOAD_COMMAND,
    EmbeddingModelUnavailableError,
    has_regular_vocabulary,
    resolve_local_model,
)

# transformers reads this at load time: when the weights come as pytorch_model.bin it would
# otherwise start a second download of the same weights as safetensors in a background thread.
CONVERSION_SWITCH = "DISABLE_SAFETENSORS_CONVERSION"


@contextmanager
def _safetensors_conversion_disabled() -> Iterator[None]:
    """Turn the conversion off for this process only and put the previous value back."""
    previous = os.environ.get(CONVERSION_SWITCH)
    os.environ[CONVERSION_SWITCH] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CONVERSION_SWITCH, None)
        else:
            os.environ[CONVERSION_SWITCH] = previous


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not use UTF-8; output must not crash.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(
        description="Download the configured embedding model into the local cache (needs internet)."
    ).parse_args(argv)

    try:
        model_name = get_settings().embedding_model
    except ValidationError as error:
        print(f"invalid settings: {error}", file=sys.stderr)
        return 1

    # Deferred import: loading torch takes seconds.
    from httpx2 import HTTPError
    from sentence_transformers import SentenceTransformer

    print(f"Downloading {model_name} if needed (first run can take a while)...", flush=True)
    try:
        # Online on purpose. The CPU keeps the GPU free; building the model on it is how the
        # library fetches and checks every file it needs. Nothing is embedded or indexed.
        with _safetensors_conversion_disabled():
            model = SentenceTransformer(model_name, device="cpu", local_files_only=False)
    except (OSError, HTTPError) as error:
        # Name the failure class only: the library's message can echo URLs, paths and values.
        # An OSError is as likely to be a full disk or a damaged cached file as a lost network.
        print(
            f"could not download {model_name} ({type(error).__name__}). Check the internet "
            "connection, the free disk space, the files in the Hugging Face cache and the "
            "model name, then run this command again.",
            file=sys.stderr,
        )
        return 1

    # When the network drops midway the library falls back to the files already cached and
    # raises nothing, even if the tokenizer's vocabulary never arrived.
    if not has_regular_vocabulary(getattr(model, "tokenizer", None)):
        print(
            f"{model_name} is incomplete: its tokenizer has no vocabulary, probably because the "
            f"download was interrupted. Restore the connection and run again: {DOWNLOAD_COMMAND}",
            file=sys.stderr,
        )
        return 1

    try:
        resolve_local_model(model_name)
    except EmbeddingModelUnavailableError:
        print(
            f"{model_name} was loaded but is not in the local Hugging Face cache, so the regular "
            "commands cannot find it. Set EMBEDDING_MODEL to a repository id like org/name "
            "or to a local directory.",
            file=sys.stderr,
        )
        return 1

    print(f"{model_name} is in the local cache. Regular commands use it without the Hub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
