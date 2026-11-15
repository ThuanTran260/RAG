import subprocess
import sys
import types

import huggingface_hub
import huggingface_hub.utils as hub_utils
import numpy as np
import pytest
import transformers.utils.logging as transformers_logging
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

from app.core.config import REPO_ROOT, Settings
from app.rag import embedder as embedder_module
from app.rag.embedder import (
    EmbeddingModelUnavailableError,
    SentenceTransformerEmbedder,
    build_embedder,
)

SNAPSHOT = "/cache/models--some--model/snapshots/abc123"


class StubTokenizer:
    """The three things of a transformers tokenizer that the vocabulary check looks at."""

    def __init__(self, size=250002, special_ids=(0, 1, 2, 3, 250001)):
        self.size = size
        self.all_special_ids = list(special_ids)
        self.added_tokens_decoder = {token_id: object() for token_id in special_ids}

    def __len__(self):
        return self.size


# What transformers builds when the vocabulary file is missing: only the special tokens remain.
VOCABULARY_LESS = StubTokenizer(size=5, special_ids=(0, 1, 2, 3, 4))


class StubModel:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dimension=4, tokenizer=None):
        self.dimension = dimension
        self.tokenizer = StubTokenizer() if tokenizer is None else tokenizer
        self.encode_calls = []
        self.loads = []
        self.snapshot_calls = []
        self.progress_seen = []

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, sentences, **kwargs):
        self.encode_calls.append((list(sentences), kwargs))
        # float64 on purpose: the adapter must hand back float32
        return np.full((len(sentences), self.dimension), 0.5, dtype=np.float64)


def current_progress_state():
    """(transformers progress enabled, Hub progress disabled) as the real libraries report them."""
    return (transformers_logging.is_progress_bar_enabled(), hub_utils.are_progress_bars_disabled())


def set_progress_state(transformers_enabled, hub_disabled):
    if transformers_enabled:
        transformers_logging.enable_progress_bar()
    else:
        transformers_logging.disable_progress_bar()
    # The transformers switch also moves the Hub one, so the Hub one is set last.
    if hub_disabled:
        hub_utils.disable_progress_bars()
    else:
        hub_utils.enable_progress_bars()


@pytest.fixture
def stub_model(monkeypatch):
    """Fake the third-party loaders: no torch import, download or network access."""
    model = StubModel()

    def factory(model_name, device=None, local_files_only=False):
        model.loads.append((model_name, device, local_files_only))
        # What the loader sees of the progress switches while it runs.
        model.progress_seen.append(current_progress_state())
        return model

    def snapshot_download(repo_id, local_files_only=False, **kwargs):
        model.snapshot_calls.append((repo_id, local_files_only, kwargs))
        return SNAPSHOT

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return model


@pytest.fixture
def progress_state():
    """Set both progress switches for a test and put the real ones back afterwards."""
    original = current_progress_state()
    yield set_progress_state
    set_progress_state(*original)


def break_snapshot_download(monkeypatch, error):
    def broken(repo_id, local_files_only=False, **kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", broken)


def break_loader(monkeypatch, error):
    def broken(model_name, device=None, local_files_only=False):
        raise error

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = broken
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_fake_embedder_is_deterministic_and_normalised(fake_embedder):
    first = fake_embedder.embed_documents(["Thời gian thử việc", "Tiền lương làm thêm giờ"])
    second = fake_embedder.embed_documents(["Thời gian thử việc", "Tiền lương làm thêm giờ"])
    assert first.shape == (2, 8)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(fake_embedder.embed_query("Thời gian thử việc"), first[0])
    assert fake_embedder.calls == [["Thời gian thử việc", "Tiền lương làm thêm giờ"]] * 2


def test_adapter_loads_the_model_and_passes_the_encode_flags(stub_model):
    embedder = SentenceTransformerEmbedder("some/model", device="cpu", batch_size=8)
    vectors = embedder.embed_documents(["a", "b"])
    assert (embedder.model_name, embedder.dimension) == ("some/model", 4)
    assert stub_model.encode_calls == [
        (
            ["a", "b"],
            {
                "batch_size": 8,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            },
        )
    ]
    assert isinstance(vectors, np.ndarray)
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 4)


def test_adapter_loads_the_local_snapshot_and_keeps_the_configured_identity(stub_model):
    embedder = SentenceTransformerEmbedder("some/model", device="cpu")
    # The Hub is asked about the local cache only, and the loader gets the snapshot, not the repo.
    assert stub_model.snapshot_calls == [("some/model", True, {})]
    assert stub_model.loads == [(SNAPSHOT, "cpu", True)]
    assert embedder.model_name == "some/model"


def test_a_local_directory_is_used_as_it_is(stub_model, tmp_path):
    embedder = SentenceTransformerEmbedder(str(tmp_path))
    assert stub_model.snapshot_calls == []
    assert stub_model.loads == [(str(tmp_path), None, True)]
    assert embedder.model_name == str(tmp_path)


def test_a_missing_cache_names_the_download_command_and_never_goes_online(stub_model, monkeypatch):
    break_snapshot_download(
        monkeypatch, LocalEntryNotFoundError("Cannot find an appropriate cached snapshot folder")
    )
    with pytest.raises(EmbeddingModelUnavailableError) as raised:
        SentenceTransformerEmbedder("some/model")
    assert "some/model" in str(raised.value)
    assert "uv run python -m ingestion.download_model" in str(raised.value)
    assert stub_model.loads == []  # no fallback to a loader that may go online


@pytest.mark.parametrize(
    "error",
    [
        OSError("no file named model.safetensors, or pytorch_model.bin, found in directory x"),
        OSError("x does not appear to have a file named tokenizer.json. Checkout available files."),
        OSError("x does not appear to have files named ('a', 'b'). Checkout available files."),
        FileNotFoundError("model.safetensors"),
    ],
)
def test_an_incomplete_cache_is_reported_as_unavailable(monkeypatch, stub_model, error):
    break_loader(monkeypatch, error)
    with pytest.raises(EmbeddingModelUnavailableError, match="ingestion.download_model"):
        SentenceTransformerEmbedder("some/model")


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Invalid device string: 'nope'"),
        ValueError("bad architecture"),
        KeyError("unexpected key"),
        PermissionError(13, "Permission denied"),
        OSError(28, "No space left on device"),
        OSError("Unable to load weights from checkpoint: invalid header"),
        OSError("unrelated loader failure"),
        RuntimeError("CUDA out of memory"),
    ],
)
def test_unrelated_loader_failures_are_not_reported_as_a_missing_cache(
    monkeypatch, stub_model, error
):
    break_loader(monkeypatch, error)
    with pytest.raises(type(error)) as raised:
        SentenceTransformerEmbedder("some/model")
    assert raised.value is error


def test_an_invalid_repository_name_is_not_reported_as_a_missing_cache(stub_model, monkeypatch):
    error = HFValidationError("Repo id must be in the form 'name'")
    break_snapshot_download(monkeypatch, error)
    with pytest.raises(HFValidationError) as raised:
        SentenceTransformerEmbedder("not a repo id")
    assert raised.value is error


def test_a_snapshot_without_a_tokenizer_vocabulary_is_reported_as_unavailable(stub_model):
    # Weights are there but the vocabulary files are not: the library still builds a tokenizer
    # out of the special tokens alone and every text would be embedded as the same few ids.
    stub_model.tokenizer = VOCABULARY_LESS
    with pytest.raises(EmbeddingModelUnavailableError) as raised:
        SentenceTransformerEmbedder("some/model")
    assert "some/model" in str(raised.value)
    assert "uv run python -m ingestion.download_model" in str(raised.value)
    assert stub_model.encode_calls == []  # nothing was embedded


@pytest.mark.parametrize(
    "tokenizer",
    [
        StubTokenizer(size=6, special_ids=(0, 1, 2, 3, 4)),  # one regular token is enough
        StubTokenizer(size=30000, special_ids=(100, 101, 102, 103)),  # BERT-like vocabulary
        None,  # a model without a tokenizer has no vocabulary to check
    ],
)
def test_a_usable_or_absent_tokenizer_passes_the_vocabulary_check(stub_model, tokenizer):
    stub_model.tokenizer = tokenizer
    assert SentenceTransformerEmbedder("some/model").dimension == 4


def test_the_vocabulary_check_does_not_hide_a_tokenizer_failure(stub_model):
    class Broken:
        all_special_ids = ()
        added_tokens_decoder = ()

        def __len__(self):
            raise RuntimeError("tokenizer bug")

    stub_model.tokenizer = Broken()
    with pytest.raises(RuntimeError, match="tokenizer bug"):
        SentenceTransformerEmbedder("some/model")


@pytest.mark.parametrize("transformers_enabled", [True, False])
@pytest.mark.parametrize("hub_disabled", [True, False])
def test_model_loading_progress_is_hidden_and_the_prior_state_restored(
    stub_model, progress_state, transformers_enabled, hub_disabled
):
    progress_state(transformers_enabled, hub_disabled)
    SentenceTransformerEmbedder("some/model")
    # Hidden while the model loads, whatever the starting state was ...
    assert stub_model.progress_seen == [(False, True)]
    # ... and each switch back where it was, independently of the other.
    assert current_progress_state() == (transformers_enabled, hub_disabled)


@pytest.mark.parametrize("transformers_enabled", [True, False])
@pytest.mark.parametrize("hub_disabled", [True, False])
def test_progress_state_is_restored_when_loading_fails(
    monkeypatch, stub_model, progress_state, transformers_enabled, hub_disabled
):
    break_loader(monkeypatch, RuntimeError("boom"))
    progress_state(transformers_enabled, hub_disabled)
    with pytest.raises(RuntimeError, match="boom"):
        SentenceTransformerEmbedder("some/model")
    assert current_progress_state() == (transformers_enabled, hub_disabled)


def test_progress_state_is_restored_when_the_cache_is_missing(
    monkeypatch, stub_model, progress_state
):
    break_snapshot_download(monkeypatch, LocalEntryNotFoundError("missing"))
    progress_state(True, False)
    with pytest.raises(EmbeddingModelUnavailableError):
        SentenceTransformerEmbedder("some/model")
    assert current_progress_state() == (True, False)


def test_query_is_one_float32_vector_and_empty_input_skips_the_model(stub_model):
    embedder = SentenceTransformerEmbedder("some/model")
    query = embedder.embed_query("q")
    assert stub_model.encode_calls[0][0] == ["q"]
    assert query.shape == (4,)
    assert query.dtype == np.float32
    empty = embedder.embed_documents([])
    assert empty.shape == (0, 4)
    assert empty.dtype == np.float32
    assert len(stub_model.encode_calls) == 1


@pytest.mark.parametrize("dimension", [None, 0, -5])
def test_an_unusable_dimension_is_rejected(stub_model, dimension):
    stub_model.dimension = dimension
    with pytest.raises(ValueError, match="embedding dimension"):
        SentenceTransformerEmbedder("some/model")


def test_a_boolean_dimension_is_rejected(stub_model):
    # bool is an int subclass: True must not pass as a dimension of 1
    stub_model.dimension = True
    with pytest.raises(ValueError, match="embedding dimension"):
        SentenceTransformerEmbedder("some/model")


def test_importing_the_module_does_not_import_torch():
    code = (
        "import sys, app.rag.embedder; "
        "assert not {'torch', 'sentence_transformers'} & set(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_build_embedder_uses_the_settings(monkeypatch):
    created = {}

    class Recorder:
        def __init__(self, model_name, device=None, batch_size=16):
            created.update(model_name=model_name, device=device, batch_size=batch_size)

    monkeypatch.setattr(embedder_module, "SentenceTransformerEmbedder", Recorder)
    settings = Settings(
        _env_file=None,
        embedding_model="AITeamVN/Vietnamese_Embedding",
        embedding_device="cpu",
        embedding_batch_size=4,
    )
    assert isinstance(build_embedder(settings), Recorder)
    assert created == {
        "model_name": "AITeamVN/Vietnamese_Embedding",
        "device": "cpu",
        "batch_size": 4,
    }
