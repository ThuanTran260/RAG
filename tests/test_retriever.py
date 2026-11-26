import unicodedata
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Modifier,
    SearchParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import HYBRID_SCHEMA_VERSION, Settings
from app.rag import retriever as retriever_module
from app.rag import search
from app.rag.embedder import EmbeddingModelUnavailableError
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.retriever import (
    RETRIEVAL_MODES,
    CollectionNotIndexedError,
    InvalidCollectionDataError,
    InvalidQueryError,
    RetrievedChunk,
    Retriever,
    default_retriever,
    retrieve,
)
from app.rag.sparse import K1, TOKENIZER_VERSION, B, query_vector
from app.rag.store import HYBRID_INDEX_COMMAND, QdrantUnavailableError
from eval.questions import load_questions
from ingestion.documents import REPO_ROOT
from ingestion.index import index_chunks, index_hybrid_chunks, point_id


class NoEmbedder:
    """An embedder that must never be reached."""

    def embed_query(self, text):
        pytest.fail("the query was embedded")


def fail_transport(*args, **kwargs):
    raise ResponseHandlingException(OSError("connection reset"))


def reject_with(status):
    def reject(*args, **kwargs):
        raise UnexpectedResponse(
            status, "Rejected", b"RESPONSE-BODY-MARKER", {"authorization": "HEADER-MARKER"}
        )

    return reject


def make_payload_incompatible(client, collection, chunk):
    """Replace one stored payload with a neutral one that is not a Chunk."""
    client.overwrite_payload(
        collection, {"legacy_field": "PAYLOAD-MARKER"}, points=[point_id(chunk.id)]
    )


def create_empty_collection(client, collection):
    client.create_collection(
        collection, vectors_config=VectorParams(size=8, distance=Distance.COSINE)
    )


@pytest.fixture(autouse=True)
def fresh_default_retriever():
    default_retriever.cache_clear()
    yield
    default_retriever.cache_clear()


@pytest.fixture
def client():
    qdrant = QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


@pytest.fixture
def retriever(client, sample_chunks, fake_embedder):
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in sample_chunks])
    index_chunks(sample_chunks, vectors, client, "test")
    return Retriever(client, fake_embedder, "test")


@pytest.fixture
def wired(monkeypatch, client, make_fake_embedder):
    """Point default_retriever() at an in-memory Qdrant and a fake embedder, counting use."""
    settings = Settings(_env_file=None)
    state = SimpleNamespace(
        settings=settings,
        client=client,
        embedder=make_fake_embedder(model_name=settings.embedding_model),
        connects=0,
        model_loads=0,
    )

    def fake_connect(_settings):
        state.connects += 1
        return state.client

    def fake_build_embedder(_settings):
        state.model_loads += 1
        return state.embedder

    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(retriever_module, "connect", fake_connect)
    monkeypatch.setattr(retriever_module, "build_embedder", fake_build_embedder)
    return state


@pytest.fixture
def indexed(wired, sample_chunks):
    texts = [chunk.context_text for chunk in sample_chunks]
    vectors = wired.embedder.embed_documents(texts)
    index_chunks(sample_chunks, vectors, wired.client, wired.settings.collection_name)
    return wired


# --- Retriever class -------------------------------------------------------------


def test_the_identical_text_is_the_top_hit(retriever, sample_chunks):
    target = sample_chunks[7]
    results = retriever.retrieve(target.context_text, k=3)
    assert len(results) == 3
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk == target
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_results_are_sorted_best_first(retriever, sample_chunks):
    scores = [result.score for result in retriever.retrieve(sample_chunks[0].context_text, k=12)]
    assert scores == sorted(scores, reverse=True)


def test_default_k_is_five(retriever, sample_chunks):
    assert len(retriever.retrieve(sample_chunks[0].context_text)) == 5


def test_k_larger_than_the_collection_returns_every_chunk(retriever, sample_chunks):
    results = retriever.retrieve(sample_chunks[0].context_text, k=100)
    assert {result.chunk.id for result in results} == {chunk.id for chunk in sample_chunks}


def test_query_is_stripped_and_normalised_to_nfc(retriever, sample_chunks):
    target = sample_chunks[2]
    decomposed = unicodedata.normalize("NFD", target.context_text)
    assert decomposed != target.context_text
    results = retriever.retrieve(f"  {decomposed}\n", k=1)
    assert results[0].chunk.id == target.id
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_raises(retriever, query):
    with pytest.raises(InvalidQueryError, match="query is empty"):
        retriever.retrieve(query)


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(retriever, k):
    with pytest.raises(InvalidQueryError, match="k must be at least 1"):
        retriever.retrieve("thử việc", k=k)


def test_missing_collection_raises_before_the_query_is_embedded(client):
    retriever = Retriever(client, NoEmbedder(), "labor_law_vn__missing")
    with pytest.raises(CollectionNotIndexedError, match="uv run python -m ingestion.index"):
        retriever.retrieve("thử việc")


@pytest.mark.parametrize("method", ["collection_exists", "query_points"])
def test_a_server_failure_becomes_qdrant_unavailable(retriever, monkeypatch, method):
    monkeypatch.setattr(retriever.client, method, fail_transport)
    with pytest.raises(QdrantUnavailableError):
        retriever.retrieve("thử việc")


def test_a_payload_that_is_not_a_chunk_is_reported_without_its_content(retriever, sample_chunks):
    target = sample_chunks[3]
    make_payload_incompatible(retriever.client, "test", target)

    with pytest.raises(retriever_module.InvalidCollectionDataError) as info:
        retriever.retrieve(target.context_text, k=1)

    message = str(info.value)
    assert "collection test" in message
    assert "incompatible" in message
    assert "uv run python -m ingestion.index" in message
    assert "PAYLOAD-MARKER" not in message


def test_an_empty_collection_is_reported_as_not_indexed(client, fake_embedder):
    create_empty_collection(client, "test")
    retriever = Retriever(client, fake_embedder, "test")

    with pytest.raises(CollectionNotIndexedError, match="empty") as info:
        retriever.retrieve("thử việc")

    assert "uv run python -m ingestion.index" in str(info.value)


# --- public retrieve() and default_retriever() -----------------------------------


def test_retrieve_returns_results_and_reuses_the_cached_retriever(indexed, sample_chunks):
    target = sample_chunks[7]
    first = retrieve(target.context_text, k=2)
    second = retrieve(target.context_text, k=2)
    assert first[0].chunk == target
    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]
    assert indexed.connects == 1
    assert indexed.model_loads == 1


def test_retrieve_normalises_a_decomposed_query(indexed, sample_chunks):
    target = sample_chunks[2]
    decomposed = unicodedata.normalize("NFD", target.context_text)
    assert retrieve(decomposed, k=1)[0].chunk.id == target.id


@pytest.mark.parametrize("query", ["", "   "])
def test_retrieve_rejects_an_empty_query_before_connecting(wired, query):
    with pytest.raises(InvalidQueryError, match="query is empty"):
        retrieve(query)
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_retrieve_rejects_a_small_k_before_connecting(wired):
    with pytest.raises(InvalidQueryError, match="k must be at least 1"):
        retrieve("thử việc", k=0)
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_retrieve_reports_a_missing_collection_without_loading_the_model(wired):
    with pytest.raises(CollectionNotIndexedError, match="uv run python -m ingestion.index"):
        retrieve("thử việc")
    assert wired.model_loads == 0


def test_retrieve_reports_an_unreachable_server_without_loading_the_model(monkeypatch):
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        retriever_module, "build_embedder", lambda _settings: pytest.fail("model was loaded")
    )
    with pytest.raises(QdrantUnavailableError, match="docker compose up -d qdrant"):
        retrieve("thử việc")


# --- search command ---------------------------------------------------------------


def test_search_cli_prints_ranked_results(indexed, sample_chunks, capsys):
    target = sample_chunks[7]

    assert search.main([target.context_text, "-k", "2"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[0] == f"1. 1.0000  {target.id}  Điều {target.article}. {target.article_title}"
    assert lines[1] == "   " + " ".join(target.text.split())[:160]
    assert lines[2].startswith("2. ")


def test_search_cli_reports_errors_without_a_traceback(monkeypatch, capsys):
    def not_indexed(_query, _k):
        raise CollectionNotIndexedError("collection x does not exist. Create it with: ...")

    monkeypatch.setattr(search, "retrieve", not_indexed)
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "collection x does not exist. Create it with: ..."


@pytest.mark.parametrize(
    ("args", "message"), [(["   "], "query is empty"), (["x", "-k", "0"], "k must be at least 1")]
)
def test_search_cli_rejects_a_bad_query_before_connecting(wired, capsys, args, message):
    assert search.main(args) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert captured.out == ""
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_search_cli_reports_a_missing_collection(wired, capsys):
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "uv run python -m ingestion.index" in captured.err
    assert captured.out == ""
    assert wired.model_loads == 0


def test_search_cli_reports_an_unreachable_server(monkeypatch, capsys):
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        retriever_module, "build_embedder", lambda _settings: pytest.fail("model was loaded")
    )
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "docker compose up -d qdrant" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_search_cli_reports_a_server_that_fails_after_the_retriever_was_cached(
    indexed, sample_chunks, monkeypatch, capsys
):
    text = sample_chunks[0].context_text
    assert search.main([text]) == 0
    capsys.readouterr()

    monkeypatch.setattr(indexed.client, "query_points", fail_transport)
    assert search.main([text]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_search_cli_reports_invalid_settings(monkeypatch, capsys):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "embedding_batch_size" in captured.err
    assert captured.out == ""


def test_search_cli_labels_an_incompatible_payload_as_collection_data(
    indexed, sample_chunks, capsys
):
    target = sample_chunks[3]
    make_payload_incompatible(indexed.client, indexed.settings.collection_name, target)

    assert search.main([target.context_text, "-k", "1"]) == 1

    captured = capsys.readouterr()
    assert "incompatible" in captured.err
    assert "uv run python -m ingestion.index" in captured.err
    assert "invalid settings" not in captured.err
    assert "PAYLOAD-MARKER" not in captured.err
    assert captured.out == ""


def test_search_cli_reports_an_empty_collection(wired, capsys):
    create_empty_collection(wired.client, wired.settings.collection_name)

    assert search.main(["thử việc"]) == 1

    captured = capsys.readouterr()
    assert "empty" in captured.err
    assert "uv run python -m ingestion.index" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("phase", ["connect", "collection_exists", "query_points"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 418, 422, 429])
def test_search_cli_reports_an_http_rejection(indexed, monkeypatch, capsys, phase, status):
    if phase == "connect":
        monkeypatch.setattr(retriever_module, "connect", reject_with(status))
    else:
        monkeypatch.setattr(indexed.client, phase, reject_with(status))

    assert search.main(["thử việc"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"HTTP {status}" in captured.err
    assert len(captured.err.strip().splitlines()) == 1
    assert "MARKER" not in captured.err
    if status in (401, 403):
        assert "ingestion.index" not in captured.err


@pytest.mark.parametrize("error", [RuntimeError("bug"), ValueError("bug")])
def test_search_cli_does_not_hide_programming_errors(indexed, monkeypatch, error):
    def bug(*args, **kwargs):
        raise error

    monkeypatch.setattr(indexed.client, "query_points", bug)
    with pytest.raises(type(error), match="bug"):
        search.main(["thử việc"])


def test_search_cli_reports_a_missing_model_cache_without_a_traceback(indexed, monkeypatch, capsys):
    def unavailable(_settings):
        raise EmbeddingModelUnavailableError(
            "embedding model x is not in the local cache. "
            "Download it once with: uv run python -m ingestion.download_model"
        )

    monkeypatch.setattr(retriever_module, "build_embedder", unavailable)
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "uv run python -m ingestion.download_model" in captured.err
    assert "Traceback" not in captured.err


# --- sparse and hybrid modes ------------------------------------------------------

# Questions outside the evaluation set (spec §5.8): no question there has Điều 77, 92, 100 or
# 205 as gold, which a test below checks.
CONTRACTOR_QUESTION = "Người cai thầu không trả lương thì ai chịu trách nhiệm?"  # Điều 100
COUNCIL_QUESTION = "Hội đồng tiền lương quốc gia tư vấn cho ai?"  # Điều 92
LOCKOUT_QUESTION = (  # Điều 205
    "Quyết định đóng cửa tạm thời nơi làm việc phải thông báo cho những ai?"
)
OUTSIDE_QUESTIONS = (CONTRACTOR_QUESTION, COUNCIL_QUESTION, LOCKOUT_QUESTION)
OUTSIDE_ARTICLES = (77, 92, 100, 205)
# Nothing is left of these once the interrogatives and final particles are removed.
QUESTIONS_WITHOUT_TERMS = ("gì?", "Vì sao vậy?", "Như thế nào ạ?", "?!")
BM25_MODES = ("sparse", "hybrid")

D77 = "45-2019-QH14:d77"
D92_K1 = "45-2019-QH14:d92:k1"
D100_K1 = "45-2019-QH14:d100:k1"
D100_K2 = "45-2019-QH14:d100:k2"
D205_K1 = "45-2019-QH14:d205:k1"
D205_K2 = "45-2019-QH14:d205:k2"
D205_K3 = "45-2019-QH14:d205:k3"

# Server order, best first, with made-up scores.
DENSE_HITS = ((D92_K1, 0.9), (D100_K1, 0.8), (D100_K2, 0.7), (D205_K1, 0.6), (D77, 0.5))
SPARSE_HITS = ((D100_K2, 9.0), (D92_K1, 5.0), (D77, 2.0))
# DENSE_HITS as hybrid mode returns them when nothing is fused.
DENSE_ONLY_SUMMARY = [
    (chunk_id, score, None, rank, None)
    for rank, (chunk_id, score) in enumerate(DENSE_HITS, start=1)
]


def hybrid_info(points, sparse_vectors=None, **metadata):
    """The parts of get_collection() that the retriever reads."""
    if sparse_vectors is None:
        sparse_vectors = {"bm25_ub": SparseVectorParams(modifier=Modifier.IDF)}
    stored = {
        "schema_version": HYBRID_SCHEMA_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "k1": K1,
        "b": B,
        "points": points,
    }
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(sparse_vectors=sparse_vectors),
            metadata=stored | metadata,
        )
    )


class ScriptedClient:
    """Answers the retriever from fixed hit lists and records every call."""

    def __init__(self, chunks, dense=DENSE_HITS, sparse=SPARSE_HITS, *, exists=True, info=None):
        self.payloads = {chunk.id: chunk.model_dump(mode="json") for chunk in chunks}
        self.dense = list(dense)
        self.sparse = list(sparse)
        self.exists = exists
        self.info = info if info is not None else hybrid_info(len(self.dense))
        self.calls = []  # (method, arguments) in call order

    def collection_exists(self, collection):
        self.calls.append(("collection_exists", {"collection": collection}))
        return self.exists

    def get_collection(self, collection):
        self.calls.append(("get_collection", {"collection": collection}))
        return self.info

    def query_points(self, collection, **arguments):
        self.calls.append(("query_points", {"collection": collection, **arguments}))
        is_sparse = isinstance(arguments["query"], SparseVector)
        points = [
            SimpleNamespace(
                id=point_id(chunk_id),
                score=score,
                payload=self.payloads[chunk_id] if arguments["with_payload"] else None,
            )
            for chunk_id, score in (self.sparse if is_sparse else self.dense)
        ]
        return SimpleNamespace(points=points[: arguments["limit"]])

    def queries(self):
        return [arguments for method, arguments in self.calls if method == "query_points"]


@pytest.fixture
def outside_chunks(all_chunks):
    return [chunk for chunk in all_chunks if chunk.article in OUTSIDE_ARTICLES]


@pytest.fixture
def scripted(outside_chunks, fake_embedder):
    """Build a retriever over a scripted client; returns (retriever, client)."""

    def _build(mode, embedder=fake_embedder, **script):
        client = ScriptedClient(outside_chunks, **script)
        return Retriever(client, embedder, "scripted", mode=mode), client

    return _build


def summary(results):
    return [(r.chunk.id, r.score, r.fusion_score, r.dense_rank, r.sparse_rank) for r in results]


def test_the_hybrid_test_questions_stay_outside_the_evaluation_set():
    questions = load_questions(REPO_ROOT / "eval" / "questions.jsonl")
    asked = {question.question.casefold() for question in questions}

    assert not asked & {text.casefold() for text in OUTSIDE_QUESTIONS + QUESTIONS_WITHOUT_TERMS}
    gold = {article for question in questions for article in question.gold_articles}
    assert not gold & set(OUTSIDE_ARTICLES)


def test_dense_mode_makes_the_same_single_query_as_before(scripted, fake_embedder):
    retriever, client = scripted("dense")

    results = retriever.retrieve(CONTRACTOR_QUESTION, k=3)

    vector = fake_embedder.embed_query(CONTRACTOR_QUESTION).tolist()
    # No metadata read, no "using", no search parameters.
    assert client.calls == [
        ("collection_exists", {"collection": "scripted"}),
        (
            "query_points",
            {"collection": "scripted", "query": vector, "limit": 3, "with_payload": True},
        ),
    ]
    assert summary(results) == [
        (chunk_id, score, None, None, None) for chunk_id, score in DENSE_HITS[:3]
    ]


def test_the_default_mode_is_dense(client, fake_embedder):
    assert Retriever(client, fake_embedder, "any").mode == "dense"
    assert RETRIEVAL_MODES == ("dense", "sparse", "hybrid")


def test_an_unknown_mode_is_rejected(client, fake_embedder):
    with pytest.raises(ValueError, match="unknown retrieval mode: 'bm25'"):
        Retriever(client, fake_embedder, "any", mode="bm25")


def test_default_retriever_stays_dense_on_the_old_collection(wired):
    retriever = default_retriever()

    assert retriever.mode == "dense"
    assert retriever.collection == wired.settings.collection_name


@pytest.mark.parametrize("mode", BM25_MODES)
def test_bm25_modes_query_both_branches_at_the_metadata_depth(scripted, fake_embedder, mode):
    retriever, client = scripted(mode)

    retriever.retrieve(CONTRACTOR_QUESTION, k=2)

    terms = query_vector(CONTRACTOR_QUESTION)
    assert client.calls == [
        ("collection_exists", {"collection": "scripted"}),
        ("get_collection", {"collection": "scripted"}),
        (
            "query_points",
            {
                "collection": "scripted",
                "query": fake_embedder.embed_query(CONTRACTOR_QUESTION).tolist(),
                "limit": 5,
                "with_payload": True,
                "search_params": SearchParams(exact=True),
            },
        ),
        (
            "query_points",
            {
                "collection": "scripted",
                "query": SparseVector(indices=list(terms.indices), values=list(terms.values)),
                "using": "bm25_ub",
                "limit": 5,
                "with_payload": False,
            },
        ),
    ]


@pytest.mark.parametrize("question", OUTSIDE_QUESTIONS + QUESTIONS_WITHOUT_TERMS)
@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_no_mode_ever_asks_the_server_to_fuse(scripted, mode, question):
    retriever, client = scripted(mode)

    retriever.retrieve(question)

    assert client.queries()
    for arguments in client.queries():
        # Only a plain dense vector or a plain sparse vector: no prefetch, no fusion query.
        assert "prefetch" not in arguments
        assert isinstance(arguments["query"], list | SparseVector)


@pytest.mark.parametrize("question", QUESTIONS_WITHOUT_TERMS)
@pytest.mark.parametrize("mode", BM25_MODES)
def test_an_empty_sparse_vector_is_never_sent(scripted, mode, question):
    retriever, client = scripted(mode)

    retriever.retrieve(question)

    assert len(client.queries()) == 1
    assert isinstance(client.queries()[0]["query"], list)


def test_hybrid_returns_the_fused_order_with_cosine_scores_and_ranks(scripted):
    retriever, _ = scripted("hybrid")

    results = retriever.retrieve(CONTRACTOR_QUESTION, k=5)

    assert summary(results) == [
        (D92_K1, 0.9, 123 / 3782, 1, 2),  # 1/61 + 1/62
        (D100_K2, 0.7, 124 / 3843, 3, 1),  # 1/63 + 1/61
        (D77, 0.5, 128 / 4095, 5, 3),  # 1/65 + 1/63
        (D100_K1, 0.8, 1 / 62, 2, None),
        (D205_K1, 0.6, 1 / 64, 4, None),
    ]


def test_sparse_returns_only_its_hits_with_dense_cosines_and_both_ranks(scripted):
    retriever, _ = scripted("sparse")

    results = retriever.retrieve(CONTRACTOR_QUESTION, k=5)

    assert summary(results) == [
        (D100_K2, 0.7, None, 3, 1),
        (D92_K1, 0.9, None, 1, 2),
        (D77, 0.5, None, 5, 3),
    ]


def test_equal_cosines_are_ranked_by_chunk_id(scripted):
    # The server lists khoản 3 first.
    retriever, _ = scripted("hybrid", dense=[(D205_K3, 0.5), (D205_K2, 0.5)], sparse=[])

    results = retriever.retrieve(LOCKOUT_QUESTION)

    assert [(r.chunk.id, r.dense_rank) for r in results] == [(D205_K2, 1), (D205_K3, 2)]


def test_equal_bm25_scores_are_ranked_by_dense_rank(scripted):
    # Khoản 3 is the better dense match; neither the server order nor the chunk id says so.
    retriever, _ = scripted(
        "sparse",
        dense=[(D205_K3, 0.8), (D205_K2, 0.7)],
        sparse=[(D205_K2, 5.0), (D205_K3, 5.0)],
    )

    results = retriever.retrieve(LOCKOUT_QUESTION)

    assert [(r.chunk.id, r.dense_rank, r.sparse_rank) for r in results] == [
        (D205_K3, 1, 1),
        (D205_K2, 2, 2),
    ]


@pytest.mark.parametrize("question", QUESTIONS_WITHOUT_TERMS)
def test_hybrid_without_sparse_terms_returns_the_dense_ranking(scripted, question):
    retriever, _ = scripted("hybrid")

    assert summary(retriever.retrieve(question, k=5)) == DENSE_ONLY_SUMMARY


def test_hybrid_without_sparse_hits_returns_the_dense_ranking(scripted):
    retriever, client = scripted("hybrid", sparse=[])

    assert summary(retriever.retrieve(CONTRACTOR_QUESTION, k=5)) == DENSE_ONLY_SUMMARY
    assert len(client.queries()) == 2


@pytest.mark.parametrize("question", QUESTIONS_WITHOUT_TERMS)
def test_sparse_without_terms_returns_nothing(scripted, question):
    retriever, _ = scripted("sparse")

    assert retriever.retrieve(question) == []


def test_sparse_without_hits_returns_nothing(scripted):
    retriever, _ = scripted("sparse", sparse=[])

    assert retriever.retrieve(CONTRACTOR_QUESTION) == []


@pytest.mark.parametrize("mode", BM25_MODES)
def test_non_positive_sparse_scores_are_not_hits(scripted, mode):
    retriever, _ = scripted(mode, sparse=[(D77, 0.0), (D205_K1, -1.0)])

    results = retriever.retrieve(CONTRACTOR_QUESTION, k=5)

    assert summary(results) == ([] if mode == "sparse" else DENSE_ONLY_SUMMARY)


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_an_empty_dense_query_means_an_empty_collection(scripted, mode):
    retriever, client = scripted(mode, dense=[], info=hybrid_info(5))

    with pytest.raises(CollectionNotIndexedError, match="collection scripted is empty") as info:
        retriever.retrieve(CONTRACTOR_QUESTION)

    assert (HYBRID_INDEX_COMMAND in str(info.value)) == (mode != "dense")
    assert len(client.queries()) == 1


@pytest.mark.parametrize("mode", BM25_MODES)
def test_a_missing_collection_names_the_hybrid_command(scripted, mode):
    retriever, client = scripted(mode, embedder=NoEmbedder(), exists=False)

    with pytest.raises(CollectionNotIndexedError, match="does not exist") as info:
        retriever.retrieve(CONTRACTOR_QUESTION)

    assert HYBRID_INDEX_COMMAND in str(info.value)
    assert [method for method, _ in client.calls] == ["collection_exists"]


@pytest.mark.parametrize(
    ("info", "complaint"),
    [
        (hybrid_info(5, sparse_vectors={}), "bm25_ub"),
        (hybrid_info(5, tokenizer_version="vi-syl-bigram-v0"), "tokenizer_version"),
    ],
)
@pytest.mark.parametrize("mode", BM25_MODES)
def test_a_collection_that_cannot_serve_bm25_is_rejected_before_embedding(
    scripted, mode, info, complaint
):
    retriever, client = scripted(mode, embedder=NoEmbedder(), info=info)

    with pytest.raises(InvalidCollectionDataError, match=complaint) as error:
        retriever.retrieve(CONTRACTOR_QUESTION)

    assert HYBRID_INDEX_COMMAND in str(error.value)
    assert client.queries() == []


@pytest.mark.parametrize("mode", BM25_MODES)
def test_a_sparse_hit_without_a_dense_score_is_an_error(scripted, mode):
    # The collection holds more points than its metadata says, so the dense query stops short.
    retriever, _ = scripted(mode, info=hybrid_info(2))

    with pytest.raises(InvalidCollectionDataError, match="scripted") as info:
        retriever.retrieve(CONTRACTOR_QUESTION)

    assert HYBRID_INDEX_COMMAND in str(info.value)


@pytest.mark.parametrize(("mode", "length"), [("sparse", 3), ("hybrid", 5)])
def test_rank_keeps_the_full_ranking_and_retrieve_cuts_it(scripted, fake_embedder, mode, length):
    retriever, client = scripted(mode)

    ranking = retriever.rank(CONTRACTOR_QUESTION, k=2)

    assert len(ranking.results) == length
    assert ranking.dense_vector == fake_embedder.embed_query(CONTRACTOR_QUESTION).tolist()
    assert len(client.queries()) == 2
    assert retriever.retrieve(CONTRACTOR_QUESTION, k=2) == ranking.results[:2]


def test_the_dense_ranking_stops_at_k(scripted, fake_embedder):
    retriever, _ = scripted("dense")

    ranking = retriever.rank(CONTRACTOR_QUESTION, k=2)

    assert [r.chunk.id for r in ranking.results] == [D92_K1, D100_K1]
    assert ranking.dense_vector == fake_embedder.embed_query(CONTRACTOR_QUESTION).tolist()


@pytest.mark.parametrize("mode", BM25_MODES)
def test_bm25_modes_reject_a_bad_request_before_any_call(scripted, mode):
    retriever, client = scripted(mode, embedder=NoEmbedder())

    with pytest.raises(InvalidQueryError, match="query is empty"):
        retriever.retrieve("   ")
    with pytest.raises(InvalidQueryError, match="k must be at least 1"):
        retriever.retrieve(CONTRACTOR_QUESTION, k=0)

    assert client.calls == []


@pytest.mark.parametrize("mode", BM25_MODES)
def test_a_bad_payload_below_the_cut_names_the_hybrid_command(scripted, mode):
    retriever, client = scripted(mode)
    client.payloads[D205_K1] = {"legacy_field": "PAYLOAD-MARKER"}

    with pytest.raises(InvalidCollectionDataError, match="incompatible") as info:
        retriever.retrieve(CONTRACTOR_QUESTION, k=1)

    assert HYBRID_INDEX_COMMAND in str(info.value)
    assert "PAYLOAD-MARKER" not in str(info.value)


@pytest.mark.parametrize("failing", ["get_collection", "dense query", "sparse query"])
def test_a_server_failure_in_a_bm25_mode_becomes_qdrant_unavailable(scripted, monkeypatch, failing):
    retriever, client = scripted("hybrid")
    answer = client.query_points

    def query_points(collection, **arguments):
        asked = "sparse query" if isinstance(arguments["query"], SparseVector) else "dense query"
        if asked == failing:
            fail_transport()
        return answer(collection, **arguments)

    monkeypatch.setattr(client, "query_points", query_points)
    if failing == "get_collection":
        monkeypatch.setattr(client, "get_collection", fail_transport)

    with pytest.raises(QdrantUnavailableError):
        retriever.retrieve(CONTRACTOR_QUESTION)


@pytest.fixture
def hybrid_build(client, all_chunks, outside_chunks, fake_embedder):
    """A collection made by the index code in local mode, which applies the IDF modifier."""
    chunks = all_chunks[:12] + outside_chunks
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in chunks])
    index_hybrid_chunks(
        chunks,
        vectors,
        client,
        "hybrid",
        model_name=fake_embedder.model_name,
        chunks_sha256="0" * 64,
    )
    return SimpleNamespace(
        size=len(chunks),
        retrievers={
            mode: Retriever(client, fake_embedder, "hybrid", mode=mode) for mode in RETRIEVAL_MODES
        },
    )


# Local mode searches exactly by brute force and warns that the exact=True flag changes nothing.
@pytest.mark.filterwarnings("ignore:Local mode performs exact:UserWarning")
@pytest.mark.parametrize(
    ("question", "article"),
    [(CONTRACTOR_QUESTION, 100), (COUNCIL_QUESTION, 92), (LOCKOUT_QUESTION, 205)],
)
def test_bm25_modes_work_on_a_real_hybrid_build(hybrid_build, question, article):
    retrievers, size = hybrid_build.retrievers, hybrid_build.size
    dense = retrievers["dense"].retrieve(question, k=size)
    sparse = retrievers["sparse"].retrieve(question, k=size)
    hybrid = retrievers["hybrid"].retrieve(question, k=size)

    # The fake dense vectors are random: the right article comes first because of BM25.
    assert sparse[0].chunk.article == article
    assert [r.sparse_rank for r in sparse] == list(range(1, len(sparse) + 1))
    cosines = {r.chunk.id: r.score for r in dense}
    for result in sparse + hybrid:
        assert result.score == pytest.approx(cosines[result.chunk.id])
        assert dense[result.dense_rank - 1].chunk.id == result.chunk.id
    expected = reciprocal_rank_fusion([r.chunk.id for r in dense], [r.chunk.id for r in sparse])
    assert len(hybrid) == size
    assert [(r.chunk.id, r.fusion_score, r.dense_rank, r.sparse_rank) for r in hybrid] == [
        (item.key, item.score, item.dense_rank, item.sparse_rank) for item in expected
    ]
