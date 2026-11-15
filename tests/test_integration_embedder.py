"""Needs the real embedding model. Run with: uv run pytest -m model"""

import json
import socket

import httpx
import httpx2
import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.rag import retriever as retriever_module
from app.rag import search
from app.rag.embedder import (
    EmbeddingModelUnavailableError,
    SentenceTransformerEmbedder,
    build_embedder,
)
from app.rag.retriever import default_retriever
from ingestion import download_model
from ingestion.index import index_chunks

pytestmark = [pytest.mark.integration, pytest.mark.model]

LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _host_of(address):
    host = address[0] if isinstance(address, tuple) else address
    return host.decode() if isinstance(host, bytes) else str(host)


@pytest.fixture
def outbound_attempts(monkeypatch):
    """Record, and refuse, every attempt to resolve or reach a host that is not this machine.

    Setting HF_HUB_OFFLINE would only assume that nothing goes out; this notices when it does,
    even if the library swallows the error and falls back to the cache.
    """
    attempts = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        # AF_UNIX addresses are paths, not hosts.
        if self.family in (socket.AF_INET, socket.AF_INET6) and _host_of(address) not in LOOPBACK:
            attempts.append(_host_of(address))
            raise ConnectionRefusedError("outbound connection blocked by the test")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and _host_of(address) not in LOOPBACK:
            attempts.append(_host_of(address))
            return 111
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host is not None and _host_of(host) not in LOOPBACK:
            attempts.append(_host_of(host))
            raise socket.gaierror("name resolution blocked by the test")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    # Guard the HTTP boundary as well: it names the host a Hub request was meant for, and it
    # covers a client that does not go through the socket methods above. Qdrant's loopback
    # requests are allowed in either client library.
    for client_type in (httpx.Client, httpx2.Client):
        real_send = client_type.send

        def guarded_send(self, request, *args, _send=real_send, **kwargs):
            host = request.url.host
            if host not in LOOPBACK:
                attempts.append(host)
                raise ConnectionRefusedError("outbound HTTP blocked by the test")
            return _send(self, request, *args, **kwargs)

        monkeypatch.setattr(client_type, "send", guarded_send)
    return attempts


def test_the_outbound_guard_notices_an_attempt(outbound_attempts):
    with pytest.raises(OSError):
        socket.create_connection(("huggingface.co", 443), timeout=1)
    assert outbound_attempts == ["huggingface.co"]


@pytest.mark.parametrize("client_type", [httpx.Client, httpx2.Client])
def test_the_outbound_guard_notices_http_requests(outbound_attempts, client_type):
    with client_type() as client, pytest.raises(ConnectionRefusedError):
        client.get("https://huggingface.co")
    assert outbound_attempts == ["huggingface.co"]


def test_model_dimension_and_normalisation():
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    vectors = embedder.embed_documents(["Thời gian thử việc", "Tiền lương làm thêm giờ"])
    assert embedder.model_name == "BAAI/bge-m3"
    assert embedder.dimension == 1024
    assert vectors.shape == (2, 1024)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    query = embedder.embed_query("Thời gian thử việc")
    assert query.shape == (1024,)
    assert float(query @ vectors[0]) > 0.99
    assert embedder.embed_documents([]).shape == (0, 1024)


def test_loading_the_cached_model_makes_no_outbound_call_and_shows_no_progress(
    outbound_attempts, capfd, caplog
):
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    embedder.embed_query("Thời gian thử việc")

    assert outbound_attempts == []
    captured = capfd.readouterr()
    for output in (captured.out, captured.err, caplog.text):
        assert "Loading weights" not in output
        assert "unauthenticated" not in output
        assert "HF Hub" not in output


def test_a_search_makes_no_outbound_call_and_shows_no_progress(
    monkeypatch, outbound_attempts, capfd, caplog, sample_chunks
):
    settings = Settings(_env_file=None)
    client = QdrantClient(":memory:")
    default_retriever.cache_clear()
    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(retriever_module, "connect", lambda _settings: client)
    try:
        embedder = build_embedder(settings)
        vectors = embedder.embed_documents([chunk.context_text for chunk in sample_chunks])
        index_chunks(sample_chunks, vectors, client, settings.collection_name)
        capfd.readouterr()  # only the search below is under test

        target = sample_chunks[5]
        assert search.main([target.context_text, "-k", "2"]) == 0

        assert outbound_attempts == []
        captured = capfd.readouterr()
        assert captured.out.splitlines()[0].split()[2] == target.id
        for output in (captured.out, captured.err, caplog.text):
            assert "Loading weights" not in output
            assert "unauthenticated" not in output
            assert "HF Hub" not in output
    finally:
        default_retriever.cache_clear()
        client.close()


TOKENIZER_CONFIG = {
    "bos_token": "<s>",
    "cls_token": "<s>",
    "eos_token": "</s>",
    "mask_token": "<mask>",
    "pad_token": "<pad>",
    "sep_token": "</s>",
    "unk_token": "<unk>",
    "tokenizer_class": "XLMRobertaTokenizer",
    "model_max_length": 64,
}


@pytest.fixture
def snapshot_without_vocabulary(tmp_path):
    """A tiny random model in the layout of bge-m3 whose vocabulary files never arrived.

    The weights and the config are there, and so are tokenizer_config.json and
    special_tokens_map.json; tokenizer.json and sentencepiece.bpe.model are not. This is what
    a download interrupted after the weights leaves behind. The folder is local, so nothing
    needs the network, and the weights are random, so nothing is downloaded or compared.
    """
    from transformers import XLMRobertaConfig, XLMRobertaModel

    config = XLMRobertaConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=70,
    )
    XLMRobertaModel(config).save_pretrained(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG), encoding="utf-8")
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text(
        json.dumps({"word_embedding_dimension": 8, "pooling_mode_cls_token": True}),
        encoding="utf-8",
    )
    return tmp_path


def test_a_snapshot_without_vocabulary_files_is_rejected_before_anything_is_embedded(
    snapshot_without_vocabulary, outbound_attempts
):
    with pytest.raises(EmbeddingModelUnavailableError, match="ingestion.download_model"):
        SentenceTransformerEmbedder(str(snapshot_without_vocabulary), device="cpu")
    assert outbound_attempts == []


def test_the_download_command_does_not_report_success_for_such_a_snapshot(
    snapshot_without_vocabulary, outbound_attempts, monkeypatch, capsys
):
    settings = Settings(_env_file=None, embedding_model=str(snapshot_without_vocabulary))
    monkeypatch.setattr(download_model, "get_settings", lambda: settings)
    assert download_model.main([]) == 1
    captured = capsys.readouterr()
    assert "is in the local cache" not in captured.out
    assert "vocabulary" in captured.err
    assert outbound_attempts == []
