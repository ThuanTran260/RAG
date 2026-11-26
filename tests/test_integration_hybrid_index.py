"""The hybrid collection against the real Qdrant server and embedding model.

Build the collection first with: uv run python -m ingestion.index --hybrid
Then run: uv run pytest -m integration tests/test_integration_hybrid_index.py
The test only reads both collections and the embedding cache. Every retrieval here is dense: the
spec (§5.8) forbids any sparse or hybrid retrieval on the evaluation questions at this stage.
"""

import hashlib
import json
import unicodedata
from itertools import combinations

import numpy as np
import pytest

from app.core.config import REPO_ROOT, Settings, model_key
from app.rag.embedder import SentenceTransformerEmbedder
from app.rag.retriever import Retriever
from app.rag.sparse import average_length, document_vector
from app.rag.store import connect, hybrid_collection_problem
from eval.questions import load_questions
from eval.run_eval import evaluate, questions_digest
from ingestion.embed_cache import embed_with_cache
from ingestion.index import point_id

QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
BASELINE_PATH = REPO_ROOT / "eval" / "runs" / "2026-11-22-dense-baseline.jsonl"
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
CACHE_DIR = REPO_ROOT / ".cache" / "embeddings"
QUESTIONS_DIGEST = "577b8392afeb"
TOLERANCE = 1e-4  # reproduction tolerance of the spec, §5.7

# Not in the evaluation set: its answer is Điều 133 khoản 1, and no question has Điều 133 as gold.
OUTSIDE_QUESTION = "Chương trình quốc gia về an toàn, vệ sinh lao động do cơ quan nào quyết định?"
OUTSIDE_ARTICLE = 133

integration = pytest.mark.integration


def unexplained_swaps(
    old: list[tuple[str, float]], new: list[tuple[str, float]], depth: int = 10
) -> tuple[list[str], list[str]]:
    """Compare the top ``depth`` of two rankings and return (tolerated swaps, violations).

    ``old`` holds the stored top ``depth``; ``new`` may be deeper, so that the new score of a
    chunk that left the top ``depth`` is known. A swap is a pair that the two rankings order
    differently; it is tolerated when its score gap is below the tolerance in every ranking that
    scores both chunks.
    """
    old_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(old[:depth])}
    new_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(new)}
    old_score, new_score = dict(old[:depth]), dict(new)
    union = list(dict.fromkeys([*old_rank, *(chunk_id for chunk_id, _ in new[:depth])]))
    unranked = [chunk_id for chunk_id in union if chunk_id not in new_rank]
    if unranked:
        return [], [f"missing from the new ranking: {', '.join(unranked)}"]
    swaps: list[str] = []
    violations: list[str] = []
    for first, second in combinations(union, 2):
        before, after = old_rank.get(first, depth), old_rank.get(second, depth)
        # Two chunks that were both below the old cut have no old order to compare with.
        if before == after or (before < after) == (new_rank[first] < new_rank[second]):
            continue
        gaps = [abs(new_score[first] - new_score[second])]
        if first in old_score and second in old_score:
            gaps.append(abs(old_score[first] - old_score[second]))
        line = f"{first} <-> {second} (gap {max(gaps):.2e})"
        (swaps if max(gaps) < TOLERANCE else violations).append(line)
    return swaps, violations


def ranking(*pairs):
    return list(pairs)


def test_equal_rankings_have_no_swap():
    old = ranking(("a", 0.9), ("b", 0.8))
    assert unexplained_swaps(old, old, depth=2) == ([], [])


def test_a_swap_of_near_ties_is_tolerated_also_at_the_cut():
    old = ranking(("a", 0.9), ("b", 0.50000))
    new = ranking(("a", 0.9), ("c", 0.50001), ("b", 0.50000))
    swaps, violations = unexplained_swaps(old, new, depth=2)
    assert violations == []
    assert len(swaps) == 1 and swaps[0].startswith("b <-> c")


@pytest.mark.parametrize(
    "new",
    [
        ranking(("b", 0.8), ("a", 0.7)),  # reordered with a wide gap
        ranking(("a", 0.9), ("c", 0.85), ("b", 0.8)),  # a stranger pushed b below the cut
        ranking(("a", 0.9), ("c", 0.85)),  # b is gone altogether
    ],
)
def test_a_real_reordering_is_a_violation(new):
    old = ranking(("a", 0.9), ("b", 0.8))
    assert unexplained_swaps(old, new, depth=2)[1]


@pytest.fixture(scope="module")
def settings():
    # No .env and the local server only, as in the other integration tests.
    return Settings(_env_file=None, qdrant_url="http://127.0.0.1:6333")


@pytest.fixture(scope="module")
def client(settings):
    qdrant = connect(settings)
    yield qdrant
    qdrant.close()


@pytest.fixture(scope="module")
def embedder(settings):
    return SentenceTransformerEmbedder(settings.embedding_model)


def stored_points(client, collection):
    points, next_page = client.scroll(collection, limit=1000, with_payload=True, with_vectors=True)
    assert next_page is None
    return {point.id: point for point in points}


class CacheOnlyEmbedder:
    """The identity of the real model; any text that is not cached fails the test."""

    def __init__(self, embedder):
        self.model_name = embedder.model_name
        self.dimension = embedder.dimension

    def embed_documents(self, texts):
        pytest.fail(f"embedding cache miss for {len(texts)} chunks")


@integration
def test_every_chunk_is_an_embedding_cache_hit(all_chunks, embedder, capsys):
    texts = [chunk.context_text for chunk in all_chunks]
    assert len(set(texts)) == 676

    vectors = embed_with_cache(texts, CacheOnlyEmbedder(embedder), CACHE_DIR)

    assert vectors.shape == (676, 1024)
    with capsys.disabled():
        print(f"\nembedding cache hits: {len(vectors)}/{len(texts)}")


@integration
def test_the_hybrid_collection_carries_the_expected_metadata(client, settings, all_chunks):
    name = settings.hybrid_collection_name
    assert client.collection_exists(name), (
        "build it with: uv run python -m ingestion.index --hybrid"
    )
    info = client.get_collection(name)

    assert hybrid_collection_problem(info, name) is None
    assert info.config.metadata == {
        "schema_version": 1,
        "model_key": model_key(settings.embedding_model),
        "chunks_sha256": hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest(),
        "points": 676,
        "tokenizer_version": "vi-syl-bigram-v1",
        "k1": 1.2,
        "b": 0.75,
        "avgdl": pytest.approx(average_length(chunk.context_text for chunk in all_chunks)),
    }
    assert client.count(name, exact=True).count == 676
    # The dense-only collection keeps its layout.
    dense = client.get_collection(settings.collection_name).config
    assert not dense.params.sparse_vectors
    assert not dense.metadata


@integration
def test_dense_vectors_and_payloads_are_bit_identical(client, settings, all_chunks, capsys):
    old = stored_points(client, settings.collection_name)
    new = stored_points(client, settings.hybrid_collection_name)
    assert len(old) == len(new) == 676
    assert set(old) == set(new) == {point_id(chunk.id) for chunk in all_chunks}

    for key, point in old.items():
        before = np.asarray(point.vector, dtype=np.float64)
        after = np.asarray(new[key].vector[""], dtype=np.float64)
        assert before.shape == (1024,)
        assert before.tobytes() == after.tobytes(), key
        assert new[key].payload == point.payload, key

    with capsys.disabled():
        print(f"\nbit-identical dense vectors: {len(old)}/676")


@integration
def test_the_sparse_vectors_are_the_bm25_weights_of_the_context_text(client, settings, all_chunks):
    new = stored_points(client, settings.hybrid_collection_name)
    avgdl = average_length(chunk.context_text for chunk in all_chunks)

    for chunk in all_chunks:
        stored = new[point_id(chunk.id)].vector["bm25_ub"]
        expected = document_vector(chunk.context_text, avgdl)
        # The server keeps float32 and does not promise the order of the indices.
        actual = dict(zip(stored.indices, np.float32(stored.values), strict=True))
        wanted = dict(zip(expected.indices, np.float32(expected.values), strict=True))
        assert actual == wanted, chunk.id


@integration
def test_one_query_vector_scores_the_same_on_both_collections(client, settings, embedder, capsys):
    questions = load_questions(QUESTIONS_PATH)
    asked = unicodedata.normalize("NFC", OUTSIDE_QUESTION).casefold()
    assert asked not in {question.question.casefold() for question in questions}
    assert all(OUTSIDE_ARTICLE not in question.gold_articles for question in questions)
    vector = embedder.embed_query(OUTSIDE_QUESTION).tolist()

    def search(collection, limit):
        hits = client.query_points(collection, query=vector, limit=limit, with_payload=True)
        return [(hit.payload["id"], hit.score) for hit in hits.points]

    old = search(settings.collection_name, 10)
    new = search(settings.hybrid_collection_name, 10)

    assert len(old) == 10
    assert new == old
    full = dict(search(settings.collection_name, 676))
    assert len(full) == 676
    assert dict(search(settings.hybrid_collection_name, 676)) == full
    with capsys.disabled():
        print(f"\nsame query on both collections: top 10 equal, top 1 {new[0][0]} {new[0][1]:.6f}")


@integration
def test_dense_retrieval_reproduces_the_baseline_run(client, settings, embedder, capsys):
    assert questions_digest(QUESTIONS_PATH) == QUESTIONS_DIGEST
    questions = load_questions(QUESTIONS_PATH)
    with BASELINE_PATH.open(encoding="utf-8") as lines:
        baseline = {row["id"]: row for row in map(json.loads, lines)}
    assert len(questions) == len(baseline) == 127
    assert {question.id for question in questions} == set(baseline)
    retriever = Retriever(client, embedder, settings.hybrid_collection_name)

    results = evaluate(questions, retriever)

    in_scope = identical = shared = 0
    largest_delta = 0.0
    swaps: list[str] = []
    failures: list[str] = []
    for result in results:
        question = result.question
        row = baseline[question.id]
        old = [(item["id"], item["score"]) for item in row["top"]]
        new = [(chunk_id, score) for chunk_id, _article, score in result.top]
        assert len(old) == len(new) == 10
        if question.group != "out_of_scope":
            in_scope += 1
            ranks = (result.rank_article, result.rank_clause)
            expected = (row["rank_article"], row["rank_clause"])
            if ranks != expected:
                failures.append(f"{question.id}: ranks {ranks}, baseline {expected}")
        new_scores = dict(new)
        for chunk_id, score in old:
            if chunk_id in new_scores:
                shared += 1
                delta = abs(new_scores[chunk_id] - score)
                largest_delta = max(largest_delta, delta)
                if delta >= TOLERANCE:
                    failures.append(f"{question.id}: {chunk_id} score moved by {delta:.2e}")
        if [chunk_id for chunk_id, _ in old] == [chunk_id for chunk_id, _ in new]:
            identical += 1
            continue
        # Search the whole collection, to score the chunks that fell out of the top 10.
        deep = retriever.retrieve(question.question, 676)
        tolerated, violations = unexplained_swaps(old, [(r.chunk.id, r.score) for r in deep])
        swaps += [f"{question.id}: {line}" for line in tolerated]
        failures += [f"{question.id}: swap {line}" for line in violations]

    device = embedder._model.device  # no public accessor on the embedder
    with capsys.disabled():
        print(f"\ndevice: {device}")
        if device.type == "cuda":
            import torch

            print(f"gpu: {torch.cuda.get_device_name(device)}")
        print(f"questions: {len(results)}, in scope: {in_scope}")
        print(f"identical top-10 order: {identical}/{len(results)}")
        print(f"largest |score delta|: {largest_delta:.2e} over {shared} shared chunks")
        print(f"tolerated swaps: {len(swaps)}")
        for line in swaps:
            print(f"  {line}")
    assert in_scope == 102
    assert failures == []
