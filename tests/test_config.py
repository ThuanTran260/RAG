import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import REPO_ROOT, Settings, get_settings, model_key, model_slug
from app.rag import search


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "labor_law_vn"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_device is None
    assert settings.embedding_batch_size == 16
    assert settings.collection_name == "labor_law_vn__baai-bge-m3--d790e737453c"


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
    settings = Settings(_env_file=None)
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.collection_name == ("labor_law_vn__aiteamvn-vietnamese-embedding--5d5026633ba6")


def test_hybrid_collection_name_extends_the_dense_one():
    settings = Settings(_env_file=None)
    assert settings.hybrid_collection_name == ("labor_law_vn__baai-bge-m3--d790e737453c__hybrid-v1")


def test_hybrid_collection_name_follows_the_collection_and_the_model():
    settings = Settings(_env_file=None, qdrant_collection="other", embedding_model="fake/model")
    assert settings.hybrid_collection_name == "other__fake-model--da372fba630a__hybrid-v1"


def test_env_file_is_read_and_unknown_keys_are_ignored(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("UNKNOWN_SETTING=ignored\nQDRANT_COLLECTION=other\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.qdrant_collection == "other"


@pytest.mark.parametrize("field", ["qdrant_url", "qdrant_collection", "embedding_model"])
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_settings_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("size", [0, -3])
def test_batch_size_below_one_is_rejected(size):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_batch_size=size)


def test_batch_size_from_the_environment_is_validated(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    with pytest.raises(ValidationError, match="embedding_batch_size"):
        Settings(_env_file=None)


def test_surrounding_whitespace_is_stripped_and_blank_device_means_unset():
    settings = Settings(_env_file=None, embedding_model=" BAAI/bge-m3 ", embedding_device="  ")
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_device is None


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("BAAI/bge-m3", "baai-bge-m3"),
        ("Qwen/Qwen3-Embedding-0.6B", "qwen-qwen3-embedding-0-6b"),
        (
            "ibm-granite/granite-embedding-311m-multilingual-r2",
            "ibm-granite-granite-embedding-311m-multilingual-r2",
        ),
    ],
)
def test_model_slug(name, slug):
    assert model_slug(name) == slug


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("BAAI/bge-m3", "baai-bge-m3--d790e737453c"),
        ("Qwen/Qwen3-Embedding-0.6B", "qwen-qwen3-embedding-0-6b--caa978c0b354"),
        ("fake/model", "fake-model--da372fba630a"),
    ],
)
def test_model_key_is_the_slug_plus_a_hash_of_the_exact_name(name, key):
    assert model_key(name) == key


def test_model_names_with_the_same_slug_get_different_keys_and_collections():
    names = ["org/a.b", "org/a-b", "org/a_b"]
    assert {model_slug(name) for name in names} == {"org-a-b"}
    assert len({model_key(name) for name in names}) == 3
    collections = {Settings(_env_file=None, embedding_model=name).collection_name for name in names}
    assert len(collections) == 3


def test_model_key_is_case_sensitive():
    assert model_slug("BAAI/bge-m3") == model_slug("baai/bge-m3")
    assert model_key("BAAI/bge-m3") != model_key("baai/bge-m3")


@pytest.mark.parametrize("name", ["", "   ", "\n"])
def test_model_key_rejects_blank_names(name):
    with pytest.raises(ValueError, match="model_name"):
        model_key(name)


def test_model_key_of_a_name_without_letters_still_has_a_slug():
    assert model_key("///").startswith("model--")


def test_repo_root_is_the_project_directory():
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_llm_key_is_read_as_a_secret_and_never_shown(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=fake-key-do-not-leak\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert isinstance(settings.llm_api_key, SecretStr)
    assert settings.llm_api_key.get_secret_value() == "fake-key-do-not-leak"
    assert "fake-key-do-not-leak" not in repr(settings)
    assert "fake-key-do-not-leak" not in str(settings)
    assert "fake-key-do-not-leak" not in settings.model_dump_json()


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_llm_key_means_unset(monkeypatch, value):
    monkeypatch.setenv("LLM_API_KEY", value)
    assert Settings(_env_file=None).llm_api_key is None


def test_llm_defaults():
    settings = Settings(_env_file=None)
    assert (settings.llm_provider, settings.llm_model) == ("gemini", "gemini-3.5-flash-lite")
    assert settings.llm_api_key is None


@pytest.mark.parametrize("field", ["llm_provider", "llm_model"])
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_llm_provider_and_model_are_rejected(field, value):
    with pytest.raises(ValidationError, match=field):
        Settings(_env_file=None, **{field: value})


def test_llm_provider_and_model_are_stripped():
    settings = Settings(_env_file=None, llm_provider=" gemini ", llm_model=" m-1 ")
    assert (settings.llm_provider, settings.llm_model) == ("gemini", "m-1")


def test_retrieval_defaults_to_dense_with_article_promotion():
    settings = Settings(_env_file=None)
    assert (settings.retrieval_mode, settings.article_promotion) == ("dense", True)
    assert settings.retrieval_collection_name == settings.collection_name


@pytest.mark.parametrize(
    ("mode", "promotion", "hybrid_collection"),
    [
        ("dense", "false", False),
        ("hybrid", "true", True),
        ("hybrid", "false", True),
        ("sparse", "false", True),
    ],
)
def test_retrieval_settings_come_from_the_environment(
    monkeypatch, mode, promotion, hybrid_collection
):
    monkeypatch.setenv("RETRIEVAL_MODE", mode)
    monkeypatch.setenv("ARTICLE_PROMOTION", promotion)

    settings = Settings(_env_file=None)

    assert settings.retrieval_mode == mode
    assert settings.article_promotion is (promotion == "true")
    expected = settings.hybrid_collection_name if hybrid_collection else settings.collection_name
    assert settings.retrieval_collection_name == expected


# None leaves promotion at its default, which is on.
@pytest.mark.parametrize("promotion", [None, "true"])
def test_sparse_with_promotion_is_rejected_naming_both_variables(monkeypatch, promotion):
    monkeypatch.setenv("RETRIEVAL_MODE", "sparse")
    if promotion is not None:
        monkeypatch.setenv("ARTICLE_PROMOTION", promotion)

    with pytest.raises(ValidationError) as info:
        Settings(_env_file=None)

    message = str(info.value)
    for text in ("RETRIEVAL_MODE=sparse", "ARTICLE_PROMOTION=false"):
        assert text in message
    # app.rag.ask prints only the field an error sits on, so it must be the one to change.
    assert [error["loc"] for error in info.value.errors()] == [("article_promotion",)]


@pytest.mark.parametrize("mode", ["bm25", "", "Dense"])
def test_an_unknown_retrieval_mode_names_the_variable(monkeypatch, mode):
    monkeypatch.setenv("RETRIEVAL_MODE", mode)

    with pytest.raises(ValidationError) as info:
        Settings(_env_file=None)

    # One error only: the promotion check says nothing about a mode that was rejected.
    assert [error["loc"] for error in info.value.errors()] == [("retrieval_mode",)]
    assert "retrieval_mode" in str(info.value)


@pytest.mark.parametrize("mode", ["sparse", "bm25"], ids=["sparse with promotion", "unknown"])
def test_a_retrieval_settings_error_never_shows_the_llm_key(monkeypatch, capsys, mode):
    monkeypatch.setenv("LLM_API_KEY", "SECRET-MARKER")
    monkeypatch.setenv("RETRIEVAL_MODE", mode)

    with pytest.raises(ValidationError) as info:
        Settings(_env_file=None)
    # Loading the settings fails first, so the command opens no connection and loads no model.
    assert search.main(["x"]) == 1

    captured = capsys.readouterr()
    assert "invalid settings" in captured.err
    assert "SECRET-MARKER" not in str(info.value)
    assert "SECRET-MARKER" not in captured.err
    assert captured.out == ""


def test_unit_tests_do_not_read_the_repo_env_file():
    assert Settings.model_config["env_file"] is None
    assert get_settings().llm_api_key is None
