import subprocess
import sys
import types

import numpy as np
import pytest

from app.core.config import REPO_ROOT, Settings
from app.rag import embedder as embedder_module
from app.rag.embedder import SentenceTransformerEmbedder, build_embedder


class StubModel:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dimension=4):
        self.dimension = dimension
        self.encode_calls = []
        self.loads = []

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, sentences, **kwargs):
        self.encode_calls.append((list(sentences), kwargs))
        # float64 on purpose: the adapter must hand back float32
        return np.full((len(sentences), self.dimension), 0.5, dtype=np.float64)


@pytest.fixture
def stub_model(monkeypatch):
    """Fake the sentence_transformers module: no torch import, download or network access."""
    model = StubModel()

    def factory(model_name, device=None):
        model.loads.append((model_name, device))
        return model

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return model


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
    assert stub_model.loads == [("some/model", "cpu")]
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
