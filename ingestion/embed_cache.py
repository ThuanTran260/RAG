"""On-disk embedding cache: embed only the texts that this model has not embedded before.

One file per model, named ``<model_key(model_name)>.npz`` inside the cache directory, holding the
SHA256 of every cached text and its vector. A cache that is missing, unreadable, written for
another model or dimension, or holds invalid numbers is treated as empty (it is only an
optimisation), and a failed write never damages the previous cache file.

Limits, by design:
- Only one process may write at a time; there is no cross-process lock.
- Entries that are no longer used stay in the file. Delete the file to clean it up.
- The key is the model name only. If the weights, the revision or the tokenisation change under
  the same model name, delete that model's cache file and index again.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from app.core.config import model_key
from app.rag.embedder import Embedder

CACHE_FORMAT_VERSION = 1


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scalar_int(array: np.ndarray) -> int | None:
    if array.ndim != 0 or array.dtype.kind not in "iu":
        return None
    return int(array)


def _load_cache(path: Path, embedder: Embedder) -> dict[str, np.ndarray]:
    """Return ``hash -> vector`` from a valid cache file, or ``{}`` for any problem.

    The broad ``except`` is intentional and limited to this function: a bad cache must never
    stop indexing, it only costs a recomputation.
    """
    try:
        # Read every array into memory inside the block so the file handle is closed before
        # the file is replaced (required on Windows).
        with np.load(path, allow_pickle=False) as data:
            format_version = data["format_version"]
            model_name = data["model_name"]
            dimension = data["dimension"]
            hashes = data["hashes"]
            vectors = data["vectors"]

        if _scalar_int(format_version) != CACHE_FORMAT_VERSION:
            return {}
        if model_name.ndim != 0 or model_name.dtype.kind != "U":
            return {}
        if str(model_name) != embedder.model_name:
            return {}
        if _scalar_int(dimension) != embedder.dimension:
            return {}
        if hashes.ndim != 1 or hashes.dtype.kind != "U":
            return {}
        hash_list = [str(item) for item in hashes]
        if len(set(hash_list)) != len(hash_list):
            return {}
        if vectors.dtype != np.float32 or vectors.shape != (len(hash_list), embedder.dimension):
            return {}
        if not np.isfinite(vectors).all():
            return {}
        return dict(zip(hash_list, vectors, strict=True))
    except Exception:  # noqa: BLE001 - any read failure just means a cache miss
        return {}


def _write_atomically(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        # Pass a file handle: given a name, np.savez would append its own ".npz" suffix.
        with os.fdopen(handle, "wb") as temp_file:
            np.savez(temp_file, **arrays)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _validated_vectors(raw: object, rows: int, embedder: Embedder) -> np.ndarray:
    """Check the encoder output and return it as float32, or raise ValueError."""
    array = np.asarray(raw)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"embedder returned invalid embeddings: non-numeric dtype {array.dtype}")
    expected = (rows, embedder.dimension)
    if array.shape != expected:
        raise ValueError(
            f"embedder returned invalid embeddings: expected shape {expected}, got {array.shape}"
        )
    with np.errstate(over="ignore"):
        vectors = array.astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("embedder returned invalid embeddings: contains NaN or Inf")
    return vectors


def embed_with_cache(texts: list[str], embedder: Embedder, cache_dir: Path) -> np.ndarray:
    """Return a float32 array with one row per text, in input order.

    Only texts missing from this model's cache are passed to ``embedder.embed_documents``,
    once, in order of first appearance and without duplicates.
    """
    if not texts:
        return np.zeros((0, embedder.dimension), dtype=np.float32)

    path = cache_dir / f"{model_key(embedder.model_name)}.npz"
    entries = _load_cache(path, embedder)

    hashes = [_text_hash(text) for text in texts]
    missing: dict[str, str] = {}  # hash -> text, in order of first appearance
    for text, text_hash in zip(texts, hashes, strict=True):
        if text_hash not in entries and text_hash not in missing:
            missing[text_hash] = text

    if missing:
        fresh = _validated_vectors(
            embedder.embed_documents(list(missing.values())), len(missing), embedder
        )
        entries.update(zip(missing, fresh, strict=True))
        _write_atomically(
            path,
            {
                "format_version": np.array(CACHE_FORMAT_VERSION),
                "model_name": np.array(embedder.model_name),
                "dimension": np.array(embedder.dimension),
                "hashes": np.array(list(entries), dtype="U64"),
                "vectors": np.stack(list(entries.values())).astype(np.float32),
            },
        )

    return np.stack([entries[text_hash] for text_hash in hashes]).astype(np.float32)
