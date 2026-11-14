"""Needs the real embedding model, not Docker. Run with: uv run pytest -m model"""

import pytest
from qdrant_client import QdrantClient

from app.rag.embedder import SentenceTransformerEmbedder
from app.rag.retriever import Retriever
from ingestion.embed_cache import embed_with_cache
from ingestion.index import index_chunks

pytestmark = [pytest.mark.integration, pytest.mark.model]


@pytest.fixture(scope="module")
def real_retriever(all_chunks, tmp_path_factory):
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    cache_dir = tmp_path_factory.mktemp("embeddings")
    vectors = embed_with_cache([chunk.context_text for chunk in all_chunks], embedder, cache_dir)
    client = QdrantClient(":memory:")
    index_chunks(all_chunks, vectors, client, "integration")
    yield Retriever(client, embedder, "integration")
    client.close()


@pytest.mark.parametrize(
    ("question", "article"),
    [
        ("Thử việc tối đa bao lâu?", 25),
        ("Người lao động muốn nghỉ việc phải báo trước bao nhiêu ngày?", 35),
        ("Nghỉ thai sản được mấy tháng?", 139),
        ("Một năm được nghỉ phép bao nhiêu ngày?", 113),
        ("Làm thêm giờ được trả lương như thế nào?", 98),
        ("Tuổi nghỉ hưu là bao nhiêu?", 169),
    ],
)
def test_everyday_questions_find_the_right_article(real_retriever, question, article):
    results = real_retriever.retrieve(question, k=5)
    assert article in {result.chunk.article for result in results}


def test_an_unrelated_question_scores_lower_than_a_relevant_one(real_retriever):
    # One negative query is an observation, not a basis for a refusal threshold (Week 3).
    on_topic = real_retriever.retrieve("Thử việc tối đa bao lâu?", k=1)[0].score
    off_topic = real_retriever.retrieve("Cách nấu phở bò ngon", k=1)[0].score
    assert off_topic < on_topic
