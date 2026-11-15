import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import REPO_ROOT, Settings, get_settings, model_key, model_slug


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


def test_unit_tests_do_not_read_the_repo_env_file():
    assert Settings.model_config["env_file"] is None
    assert get_settings().llm_api_key is None
