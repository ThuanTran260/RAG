import numpy as np
import pytest

from app.core.config import model_key
from ingestion import embed_cache
from ingestion.embed_cache import embed_with_cache

TEXTS = ["Văn bản thử nghiệm số 1", "Văn bản thử nghiệm số 2", "Văn bản thử nghiệm số 3"]


def cache_file(embedder, cache_dir):
    return cache_dir / f"{model_key(embedder.model_name)}.npz"


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def without(name):
    return lambda arrays: {key: value for key, value in arrays.items() if key != name}


def poisoned(vectors, row, column, value):
    changed = vectors.copy()
    changed[row, column] = value
    return changed


TAMPERING = {
    **{
        f"missing_{name}": without(name)
        for name in ("format_version", "model_name", "dimension", "hashes", "vectors")
    },
    "other_format_version": lambda a: {**a, "format_version": np.array(99)},
    "other_model_name": lambda a: {**a, "model_name": np.array("other/model")},
    "other_dimension": lambda a: {**a, "dimension": np.array(99)},
    "float64_vectors": lambda a: {**a, "vectors": a["vectors"].astype(np.float64)},
    "flat_vectors": lambda a: {**a, "vectors": a["vectors"].reshape(-1)},
    "missing_row": lambda a: {**a, "vectors": a["vectors"][:-1]},
    "nan_value": lambda a: {**a, "vectors": poisoned(a["vectors"], 0, 0, np.nan)},
    "inf_value": lambda a: {**a, "vectors": poisoned(a["vectors"], 1, 1, np.inf)},
    "duplicate_hashes": lambda a: {**a, "hashes": np.repeat(a["hashes"][:1], len(a["hashes"]))},
    "pickled_hashes": lambda a: {**a, "hashes": a["hashes"].astype(object)},
}

CORRUPTIONS = {
    "garbage": lambda data: b"not a zip archive",
    "truncated": lambda data: data[: len(data) // 2],
    "empty": lambda data: b"",
}

BAD_OUTPUTS = {
    "nan": lambda rows, width: np.full((rows, width), np.nan, dtype=np.float32),
    "inf": lambda rows, width: np.full((rows, width), np.inf, dtype=np.float32),
    "too_few_rows": lambda rows, width: np.ones((rows - 1, width), dtype=np.float32),
    "wrong_width": lambda rows, width: np.ones((rows, width + 1), dtype=np.float32),
    "flat": lambda rows, width: np.ones(rows * width, dtype=np.float32),
}


def test_returns_one_row_per_text_in_input_order(fake_embedder, tmp_path):
    vectors = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert vectors.shape == (3, fake_embedder.dimension)
    assert vectors.dtype == np.float32
    for row, text in zip(vectors, TEXTS, strict=True):
        assert np.array_equal(row, fake_embedder.embed_query(text))


def test_second_call_does_not_embed_again(fake_embedder, tmp_path):
    first = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    second = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert len(fake_embedder.calls) == 1
    assert np.array_equal(first, second)


def test_only_new_or_changed_texts_are_embedded(fake_embedder, tmp_path):
    embed_with_cache(TEXTS, fake_embedder, tmp_path)
    changed = [TEXTS[0], "Văn bản thử nghiệm số 2 (đã đổi)", TEXTS[2], "Văn bản thử nghiệm số 4"]
    vectors = embed_with_cache(changed, fake_embedder, tmp_path)
    assert fake_embedder.calls[1] == [changed[1], changed[3]]
    assert np.array_equal(vectors[1], fake_embedder.embed_query(changed[1]))
    assert np.array_equal(vectors[0], fake_embedder.embed_query(changed[0]))


def test_duplicate_texts_are_embedded_once(fake_embedder, tmp_path):
    vectors = embed_with_cache([TEXTS[0], TEXTS[1], TEXTS[0]], fake_embedder, tmp_path)
    assert fake_embedder.calls == [[TEXTS[0], TEXTS[1]]]
    assert np.array_equal(vectors[0], vectors[2])


def test_empty_input_embeds_nothing_and_writes_nothing(fake_embedder, tmp_path):
    vectors = embed_with_cache([], fake_embedder, tmp_path)
    assert vectors.shape == (0, fake_embedder.dimension)
    assert fake_embedder.calls == []
    assert list(tmp_path.iterdir()) == []


def test_cache_directory_is_created(fake_embedder, tmp_path):
    cache_dir = tmp_path / "nested" / "embeddings"
    embed_with_cache(TEXTS, fake_embedder, cache_dir)
    assert cache_file(fake_embedder, cache_dir).is_file()


def test_each_model_has_its_own_cache_file(make_fake_embedder, tmp_path):
    first = make_fake_embedder("BAAI/bge-m3")
    second = make_fake_embedder("Qwen/Qwen3-Embedding-0.6B")
    embed_with_cache(TEXTS, first, tmp_path)
    embed_with_cache(TEXTS, second, tmp_path)
    assert len(second.calls) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"{model_key('BAAI/bge-m3')}.npz", f"{model_key('Qwen/Qwen3-Embedding-0.6B')}.npz"]
    )


def test_model_names_with_the_same_slug_do_not_share_a_cache(make_fake_embedder, tmp_path):
    dotted = make_fake_embedder("org/a.b")
    dashed = make_fake_embedder("org/a-b")
    embed_with_cache(TEXTS, dotted, tmp_path)
    embed_with_cache(TEXTS, dashed, tmp_path)
    assert len(dashed.calls) == 1
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_cache_of_another_dimension_is_recomputed(make_fake_embedder, tmp_path):
    embed_with_cache(TEXTS, make_fake_embedder(dimension=8), tmp_path)
    wider = make_fake_embedder(dimension=16)
    vectors = embed_with_cache(TEXTS, wider, tmp_path)
    assert len(wider.calls) == 1
    assert vectors.shape == (3, 16)


@pytest.mark.parametrize("corrupt", list(CORRUPTIONS.values()), ids=list(CORRUPTIONS))
def test_unreadable_cache_file_is_a_miss(fake_embedder, tmp_path, corrupt):
    expected = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    path = cache_file(fake_embedder, tmp_path)
    path.write_bytes(corrupt(path.read_bytes()))
    vectors = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert fake_embedder.calls[1] == TEXTS
    assert np.array_equal(vectors, expected)


@pytest.mark.parametrize("tamper", list(TAMPERING.values()), ids=list(TAMPERING))
def test_invalid_cache_content_is_a_miss_and_is_rewritten(fake_embedder, tmp_path, tamper):
    expected = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    path = cache_file(fake_embedder, tmp_path)
    np.savez(path, **tamper(read_arrays(path)))
    vectors = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert fake_embedder.calls[1] == TEXTS
    assert np.array_equal(vectors, expected)
    embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert len(fake_embedder.calls) == 2


@pytest.mark.parametrize("make_bad", list(BAD_OUTPUTS.values()), ids=list(BAD_OUTPUTS))
def test_invalid_encoder_output_is_rejected_and_the_old_cache_is_kept(
    fake_embedder, tmp_path, monkeypatch, make_bad
):
    embed_with_cache(TEXTS[:2], fake_embedder, tmp_path)
    path = cache_file(fake_embedder, tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(
        fake_embedder,
        "embed_documents",
        lambda texts: make_bad(len(texts), fake_embedder.dimension),
    )
    with pytest.raises(ValueError, match="invalid embeddings"):
        embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_failed_replacement_keeps_the_old_cache_and_leaves_no_temp_file(
    fake_embedder, tmp_path, monkeypatch
):
    embed_with_cache(TEXTS[:2], fake_embedder, tmp_path)
    path = cache_file(fake_embedder, tmp_path)
    before = path.read_bytes()

    def broken_replace(source, destination):
        raise OSError("replace failed")

    with monkeypatch.context() as patch:
        patch.setattr(embed_cache.os, "replace", broken_replace)
        with pytest.raises(OSError, match="replace failed"):
            embed_with_cache(TEXTS, fake_embedder, tmp_path)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    embed_with_cache(TEXTS[:2], fake_embedder, tmp_path)
    assert len(fake_embedder.calls) == 2  # the old entries still hit
