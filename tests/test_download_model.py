import os
import sys
import types

import huggingface_hub
import pytest
from httpx2 import ConnectError, ReadTimeout
from huggingface_hub.errors import LocalEntryNotFoundError

from app.core.config import Settings
from ingestion import download_model

CONVERSION_SWITCH = "DISABLE_SAFETENSORS_CONVERSION"


class Tokenizer:
    """The parts of a transformers tokenizer that the vocabulary check looks at."""

    def __init__(self, size=250002, special_ids=(0, 1, 2, 3, 250001)):
        self.size = size
        self.all_special_ids = list(special_ids)
        self.added_tokens_decoder = {token_id: object() for token_id in special_ids}

    def __len__(self):
        return self.size


class Model:
    """No encode and no dimension: the command must not embed anything."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


@pytest.fixture
def loader(monkeypatch):
    """Fake sentence_transformers and the local cache lookup: no torch, download or network."""
    state = types.SimpleNamespace(
        loads=[], resolved=[], error=None, tokenizer=Tokenizer(), conversion_seen=[]
    )

    def factory(model_name, **kwargs):
        state.loads.append((model_name, kwargs))
        # What the loader sees of the transformers switch for the safetensors conversion.
        state.conversion_seen.append(os.environ.get(CONVERSION_SWITCH))
        if state.error is not None:
            raise state.error
        return Model(state.tokenizer)

    def snapshot_download(repo_id, local_files_only=False, **kwargs):
        state.resolved.append((repo_id, local_files_only))
        return "/cache/snapshot"

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(download_model, "get_settings", lambda: Settings(_env_file=None))
    return state


def test_it_loads_the_configured_model_online_on_the_cpu(loader, capsys):
    assert download_model.main([]) == 0
    # Online on purpose, on the CPU so that the shared GPU is not touched, no remote code.
    assert loader.loads == [("BAAI/bge-m3", {"device": "cpu", "local_files_only": False})]
    # It then checks that the regular commands will find the model in the local cache.
    assert loader.resolved == [("BAAI/bge-m3", True)]
    captured = capsys.readouterr()
    assert "BAAI/bge-m3" in captured.out
    assert captured.err == ""


def test_it_follows_the_embedding_model_setting(loader, monkeypatch):
    settings = Settings(_env_file=None, embedding_model="AITeamVN/Vietnamese_Embedding")
    monkeypatch.setattr(download_model, "get_settings", lambda: settings)
    assert download_model.main([]) == 0
    assert loader.loads[0][0] == "AITeamVN/Vietnamese_Embedding"


def test_it_needs_no_qdrant(loader, monkeypatch):
    import qdrant_client

    def unexpected_client(*args, **kwargs):
        pytest.fail("download must not create a Qdrant client")

    monkeypatch.setattr(qdrant_client, "QdrantClient", unexpected_client)
    assert download_model.main([]) == 0


@pytest.mark.parametrize("error_type", [OSError, ConnectError, ReadTimeout])
def test_a_download_failure_is_reported_briefly_without_its_details(loader, capsys, error_type):
    loader.error = error_type("could not reach https://example.invalid/?token=SECRET-MARKER")
    assert download_model.main([]) == 1
    captured = capsys.readouterr()
    assert "is in the local cache" not in captured.out  # no success claim
    assert "BAAI/bge-m3" in captured.err
    assert error_type.__name__ in captured.err
    assert "MARKER" not in captured.err
    assert "Traceback" not in captured.err
    assert len(captured.err.strip().splitlines()) == 1


@pytest.mark.parametrize(
    "error",
    [
        OSError(28, "No space left on device: C:/secret/path"),
        OSError("It looks like the config file at 'C:/secret/path' is not a valid JSON file."),
    ],
)
def test_a_storage_or_file_failure_is_not_blamed_on_the_network_alone(loader, capsys, error):
    loader.error = error
    assert download_model.main([]) == 1
    captured = capsys.readouterr()
    # The guidance covers the disk and the cached files as well as the connection ...
    assert "disk" in captured.err
    assert "cache" in captured.err
    assert "connection" in captured.err
    # ... and never repeats what the library said, which can hold paths, URLs and values.
    assert "secret" not in captured.err
    assert "Traceback" not in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_a_cache_without_a_tokenizer_vocabulary_is_not_reported_as_downloaded(loader, capsys):
    # The online build can fall back to the cache when the network drops midway: the library
    # then builds a tokenizer out of the special tokens alone and raises nothing.
    loader.tokenizer = Tokenizer(size=5, special_ids=(0, 1, 2, 3, 4))
    assert download_model.main([]) == 1
    captured = capsys.readouterr()
    assert "is in the local cache" not in captured.out  # no success claim
    assert "BAAI/bge-m3" in captured.err
    assert "vocabulary" in captured.err
    assert "ingestion.download_model" in captured.err  # run it again, with the network back
    assert "Traceback" not in captured.err


def test_the_safetensors_conversion_is_off_while_downloading_and_the_switch_restored(
    loader, monkeypatch
):
    # transformers would otherwise start a second download of the weights in another format.
    monkeypatch.delenv(CONVERSION_SWITCH, raising=False)
    assert download_model.main([]) == 0
    assert loader.conversion_seen == ["1"]
    assert CONVERSION_SWITCH not in os.environ


@pytest.mark.parametrize("previous", ["0", "false", "1", ""])
def test_a_previous_value_of_the_switch_comes_back(loader, monkeypatch, previous):
    monkeypatch.setenv(CONVERSION_SWITCH, previous)
    assert download_model.main([]) == 0
    assert loader.conversion_seen == ["1"]
    assert os.environ[CONVERSION_SWITCH] == previous


@pytest.mark.parametrize(
    "error", [OSError("offline"), ConnectError("offline"), RuntimeError("bug")]
)
@pytest.mark.parametrize("previous", [None, "false"])
def test_the_switch_comes_back_when_the_download_fails(loader, monkeypatch, error, previous):
    if previous is None:
        monkeypatch.delenv(CONVERSION_SWITCH, raising=False)
    else:
        monkeypatch.setenv(CONVERSION_SWITCH, previous)
    loader.error = error
    try:
        download_model.main([])
    except RuntimeError:
        pass  # programming errors are not hidden, see below
    assert os.environ.get(CONVERSION_SWITCH) == previous


def test_a_model_that_regular_commands_cannot_find_afterwards_is_reported(
    loader, monkeypatch, capsys
):
    def miss(repo_id, local_files_only=False, **kwargs):
        raise LocalEntryNotFoundError("missing")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", miss)
    assert download_model.main([]) == 1
    captured = capsys.readouterr()
    assert "is in the local cache" not in captured.out  # no success claim
    assert "not in the local Hugging Face cache" in captured.err
    # The remedy of the regular commands would loop back here, so it must not be suggested.
    assert "ingestion.download_model" not in captured.err


def test_invalid_settings_are_reported(monkeypatch, capsys):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    monkeypatch.setattr(download_model, "get_settings", Settings)
    assert download_model.main([]) == 1
    captured = capsys.readouterr()
    assert "embedding_batch_size" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("error", [RuntimeError("bug"), ValueError("bug")])
def test_programming_errors_are_not_hidden(loader, error):
    loader.error = error
    with pytest.raises(type(error), match="bug"):
        download_model.main([])
