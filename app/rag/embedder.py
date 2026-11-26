"""Embedding interface and the sentence-transformers implementation.

Torch and sentence-transformers are imported only when a real model is loaded, so importing
this module (and every unit test that uses a fake embedder) stays fast.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import Settings

DOWNLOAD_COMMAND = "uv run python -m ingestion.download_model"


class EmbeddingModelUnavailableError(RuntimeError):
    """The embedding model is not (completely) in the local cache."""


def _unavailable(model_name: str) -> EmbeddingModelUnavailableError:
    return EmbeddingModelUnavailableError(
        f"embedding model {model_name} is not in the local cache. "
        f"Download it once with: {DOWNLOAD_COMMAND}"
    )


def resolve_local_model(model_name: str) -> str:
    """Return where the model sits on this machine, without any request to the Hugging Face Hub.

    A directory is used as it is. Any other name is a Hub repository id, looked up in the local
    cache only; EmbeddingModelUnavailableError says so when it is not there.
    """
    if Path(model_name).is_dir():
        return model_name
    # Deferred like the other heavy imports.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except LocalEntryNotFoundError as error:
        raise _unavailable(model_name) from error


def has_regular_vocabulary(tokenizer: object | None) -> bool:
    """Tell whether a tokenizer knows any token besides its special ones.

    transformers builds a tokenizer whose vocabulary file is missing without raising: only the
    special tokens (<s>, <pad>, ...) are left, so every text turns into the same few ids and
    unrelated texts get almost the same embedding. A complete tokenizer holds far more than its
    special tokens, whatever its format. No tokenizer at all (a model without text input) has no
    vocabulary to check.
    """
    if tokenizer is None:
        return True
    special_ids = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
    return len(tokenizer) > len(special_ids)


def _is_missing_file_error(error: OSError) -> bool:
    """Recognise missing files, including transformers' untyped local-file errors.

    Transformers uses plain OSError for missing weights/tokenizer files. Match only its
    explicit missing-file messages; corrupt checkpoints and other I/O failures must propagate.
    """
    if isinstance(error, FileNotFoundError):
        return True
    if type(error) is not OSError or error.errno is not None:
        return False
    message = str(error)
    return (
        ("no file named " in message and "found in directory " in message)
        or " does not appear to have a file named " in message
        or (" does not appear to have files named " in message)
    )


@contextmanager
def _model_loading_progress_hidden() -> Iterator[None]:
    """Hide the "Loading weights" bar while the model is built, then put both switches back.

    Only the progress switches are touched: warnings and loggers stay as they were. The
    transformers switch also moves the Hugging Face Hub one, so each is saved and restored
    on its own.
    """
    from huggingface_hub import utils as hub_utils
    from transformers.utils import logging as transformers_logging

    transformers_enabled = transformers_logging.is_progress_bar_enabled()
    hub_disabled = hub_utils.are_progress_bars_disabled()
    transformers_logging.disable_progress_bar()
    try:
        yield
    finally:
        if transformers_enabled:
            transformers_logging.enable_progress_bar()
        else:
            transformers_logging.disable_progress_bar()
        # Last, because the transformers switch above has just moved this one as well.
        if hub_disabled:
            hub_utils.disable_progress_bars()
        else:
            hub_utils.enable_progress_bars()


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
        # The regular commands never go online: the model must already be in the local cache,
        # put there by `ingestion.download_model`. model_name stays the configured identity.
        local_path = resolve_local_model(model_name)
        try:
            with _model_loading_progress_hidden():
                self._model = SentenceTransformer(local_path, device=device, local_files_only=True)
        except OSError as error:
            if _is_missing_file_error(error):
                raise _unavailable(model_name) from error
            raise
        # Weights without the tokenizer's vocabulary files raise nothing but embed garbage.
        if not has_regular_vocabulary(getattr(self._model, "tokenizer", None)):
            raise _unavailable(model_name)
        dimension = self._model.get_embedding_dimension()
        # bool is an int subclass, so reject it explicitly.
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"cannot determine the embedding dimension of {model_name}")
        self.dimension = dimension

    @property
    def device(self) -> str:
        """Where the loaded model runs, such as cuda:0."""
        return str(self._model.device)

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
