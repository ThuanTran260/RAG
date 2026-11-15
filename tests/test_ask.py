import json

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import ValidationError
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import Settings
from app.rag import ask, embedder, retriever
from app.rag.ask import settings_error_line
from app.rag.embedder import EmbeddingModelUnavailableError
from app.rag.generate import (
    DISCLAIMER,
    INSUFFICIENT_TEXT,
    AnswerValidationError,
    generate,
    render_result,
)
from app.rag.llm import (
    LLMAuthError,
    LLMBadResponseError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.rag.retriever import CollectionNotIndexedError, InvalidCollectionDataError
from app.rag.store import QdrantUnavailableError, rejection_message

DOC = "45-2019-QH14"
QUESTION = "Câu hỏi giả lập?"
FAKE_KEY = "fake-key-do-not-leak"
BODY_MARKER = "BODY-MARKER"


def cid(suffix: str) -> str:
    return f"{DOC}:{suffix}"


def answered_reply(*sources: str) -> str:
    claim = {"text": "Khẳng định giả lập.", "sources": list(sources)}
    return json.dumps({"status": "answered", "claims": [claim]}, ensure_ascii=False)


def insufficient_reply() -> str:
    return json.dumps({"status": "insufficient", "claims": []})


@pytest.fixture
def settings():
    # _env_file=None keeps the developer's real .env out of the tests.
    return Settings(_env_file=None, llm_api_key=FAKE_KEY)


@pytest.fixture
def wired(monkeypatch, settings, make_fake_llm_client, contexts_for):
    """Replace settings, the LLM client and retrieval; record how retrieve is called."""

    class Wiring:
        def __init__(self):
            self.retrieve_calls: list[tuple[str, int]] = []
            self.contexts = contexts_for([cid("d34:k1"), cid("d1")])
            self.client = make_fake_llm_client(answered_reply("S1", "S2"))

        def retrieve(self, query, k):
            self.retrieve_calls.append((query, k))
            return self.contexts

    wiring = Wiring()
    monkeypatch.setattr(ask, "get_settings", lambda: settings)
    monkeypatch.setattr(ask, "build_llm_client", lambda _settings: wiring.client)
    monkeypatch.setattr(ask, "retrieve", wiring.retrieve)
    return wiring


def run(capsys, *args: str) -> tuple[int, str, str]:
    code = ask.main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def assert_one_line_failure(code: int, out: str, err: str) -> str:
    assert code == 1
    assert out == ""
    assert "Traceback" not in err
    assert len(err.splitlines()) == 1
    return err.strip()


# --- checks that run before anything expensive -----------------------------------------------


def test_missing_key_fails_before_retrieval_or_loading_the_model(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("must not be reached without a key")

    monkeypatch.setattr(ask, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(ask, "retrieve", forbidden)
    monkeypatch.setattr(retriever, "build_embedder", forbidden)

    code, out, err = run(capsys, QUESTION)

    message = assert_one_line_failure(code, out, err)
    assert "LLM_API_KEY" in message


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([""], "query is empty"),
        (["   "], "query is empty"),
        (["x", "-k", "0"], "k must be at least 1"),
        (["x", "-k", "9"], "k must be at most 8"),
    ],
)
def test_blank_question_and_out_of_range_k_fail_before_settings(monkeypatch, capsys, args, message):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("settings were read before the input checks")

    monkeypatch.setattr(ask, "get_settings", forbidden)

    code, out, err = run(capsys, *args)

    assert assert_one_line_failure(code, out, err) == message


def test_invalid_settings_is_one_line_without_values(monkeypatch, capsys):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "abc-SETTINGS-MARKER")

    code, out, err = run(capsys, QUESTION)

    message = assert_one_line_failure(code, out, err)
    assert "EMBEDDING_BATCH_SIZE" in message
    assert "SETTINGS-MARKER" not in message


def test_settings_error_line_names_variables_only():
    with pytest.raises(ValidationError) as raised:
        Settings(_env_file=None, embedding_batch_size="abc-SETTINGS-MARKER", qdrant_url="  ")

    line = settings_error_line(raised.value)

    assert line == (
        "invalid settings: EMBEDDING_BATCH_SIZE, QDRANT_URL. "
        "Check these variables in the environment or .env."
    )


# --- successful results ----------------------------------------------------------------------


def test_answer_is_printed_with_labels_and_verbatim_text(wired, capsys):
    code, out, err = run(capsys, QUESTION)

    assert code == 0
    assert err == ""
    assert wired.client.calls == 1
    assert out == render_result(generate(QUESTION, wired.contexts, wired.client)) + "\n"
    assert "Điều 34, khoản 1" in out
    assert "Điều 1" in out
    for retrieved in wired.contexts:
        assert retrieved.chunk.text in out
    assert out.rstrip("\n").endswith(DISCLAIMER)


def test_insufficient_is_a_successful_result(wired, make_fake_llm_client, monkeypatch, capsys):
    client = make_fake_llm_client(insufficient_reply())
    monkeypatch.setattr(ask, "build_llm_client", lambda _settings: client)

    code, out, err = run(capsys, QUESTION)

    assert code == 0
    assert err == ""
    assert client.calls == 1
    assert out == f"{INSUFFICIENT_TEXT}\n\n{DISCLAIMER}\n"


def test_empty_retrieval_result_is_insufficient_without_an_llm_call(wired, monkeypatch, capsys):
    monkeypatch.setattr(ask, "retrieve", lambda _query, _k: [])

    code, out, err = run(capsys, QUESTION)

    assert code == 0
    assert err == ""
    assert wired.client.calls == 0
    assert out == f"{INSUFFICIENT_TEXT}\n\n{DISCLAIMER}\n"


def test_k_is_passed_to_retrieve_and_defaults_to_five(wired, capsys):
    assert run(capsys, "  " + QUESTION + "  ")[0] == 0
    assert run(capsys, QUESTION, "-k", "8")[0] == 0
    assert run(capsys, QUESTION, "-k", "1")[0] == 0

    # The question reaches retrieve already stripped by prepare_query.
    assert wired.retrieve_calls == [(QUESTION, 5), (QUESTION, 8), (QUESTION, 1)]


# --- operational errors ----------------------------------------------------------------------


def test_unindexed_or_empty_collection_exits_1_and_never_calls_the_llm(wired, monkeypatch, capsys):
    def not_indexed(_query, _k):
        raise CollectionNotIndexedError("collection x is empty. Fill it with: ...")

    monkeypatch.setattr(ask, "retrieve", not_indexed)

    code, out, err = run(capsys, QUESTION)

    assert assert_one_line_failure(code, out, err) == "collection x is empty. Fill it with: ..."
    assert wired.client.calls == 0


def rejection(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status, "Not Found", b"", {})


@pytest.mark.parametrize(
    "error",
    [
        QdrantUnavailableError("Qdrant is not reachable at http://x. Start it with: ..."),
        CollectionNotIndexedError("collection x does not exist. Create it with: ..."),
        InvalidCollectionDataError("a payload in collection x is incompatible. Rebuild it."),
        rejection(404),
    ],
    ids=["unavailable", "not_indexed", "invalid_data", "rejection_404"],
)
def test_qdrant_errors_reuse_the_search_messages(wired, monkeypatch, capsys, error):
    def failing(_query, _k):
        raise error

    monkeypatch.setattr(ask, "retrieve", failing)
    expected = (
        rejection_message(error, searching=True)
        if isinstance(error, UnexpectedResponse)
        else str(error)
    )

    code, out, err = run(capsys, QUESTION)

    assert assert_one_line_failure(code, out, err) == expected
    assert wired.client.calls == 0


def test_missing_embedding_model_cache_exits_1(wired, monkeypatch, tmp_path, capsys):
    def no_local_snapshot(*, repo_id, local_files_only):
        assert local_files_only is True
        raise LocalEntryNotFoundError("missing")

    # The import inside resolve_local_model reads this attribute, so no network is possible.
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_local_snapshot)
    with pytest.raises(EmbeddingModelUnavailableError) as raised:
        embedder.resolve_local_model(str(tmp_path / "absent-model"))
    real_error = raised.value

    def failing(_query, _k):
        raise real_error

    monkeypatch.setattr(ask, "retrieve", failing)

    code, out, err = run(capsys, QUESTION)

    message = assert_one_line_failure(code, out, err)
    assert message == str(real_error)
    assert "uv run python -m ingestion.download_model" in message
    assert wired.client.calls == 0


@pytest.mark.parametrize(
    "error",
    [
        LLMAuthError("LLM authentication failed (HTTP 401)", 401),
        LLMRateLimitError("LLM rate limit reached (HTTP 429)", 429),
        LLMUnavailableError("LLM connection failed"),
        LLMBadResponseError("LLM response has no text"),
        AnswerValidationError("unknown_source"),
    ],
    ids=["auth", "rate_limit", "unavailable", "bad_response", "validation"],
)
def test_llm_and_validation_errors_exit_1_with_one_stderr_line(
    wired, make_fake_llm_client, monkeypatch, capsys, error
):
    client = make_fake_llm_client(error)
    monkeypatch.setattr(ask, "build_llm_client", lambda _settings: client)
    expected = (
        f"invalid model answer: {error.code}"
        if isinstance(error, AnswerValidationError)
        else f"LLM: {error}"
    )

    code, out, err = run(capsys, QUESTION)

    assert assert_one_line_failure(code, out, err) == expected
    assert client.calls == 1


def test_generation_input_error_exits_1_with_one_line(wired, monkeypatch, contexts_for, capsys):
    duplicated = contexts_for([cid("d1")]) * 2
    monkeypatch.setattr(ask, "retrieve", lambda _query, _k: duplicated)

    code, out, err = run(capsys, QUESTION)

    message = assert_one_line_failure(code, out, err)
    assert message == "contexts contain duplicate chunk ids"
    assert wired.client.calls == 0


def test_secrets_and_response_bodies_never_reach_the_output(
    wired, make_fake_llm_client, monkeypatch, settings, capsys
):
    assert settings.llm_api_key.get_secret_value() == FAKE_KEY
    leaking_cause = RuntimeError(f"{BODY_MARKER} {FAKE_KEY}")
    rejected = UnexpectedResponse(400, "Bad Request", f"{BODY_MARKER} {FAKE_KEY}".encode(), {})

    def rejecting(_query, _k):
        raise rejected

    def failing_llm() -> LLMUnavailableError:
        try:
            raise leaking_cause
        except RuntimeError as cause:
            error = LLMUnavailableError("LLM provider unavailable (HTTP 503)", 503)
            error.__cause__ = cause
            return error

    # A Qdrant rejection carrying the markers in its body.
    monkeypatch.setattr(ask, "retrieve", rejecting)
    outputs = [run(capsys, QUESTION)]
    # An LLM failure whose cause carries the markers.
    monkeypatch.setattr(ask, "retrieve", wired.retrieve)
    client = make_fake_llm_client(failing_llm())
    monkeypatch.setattr(ask, "build_llm_client", lambda _settings: client)
    outputs.append(run(capsys, QUESTION))

    for code, out, err in outputs:
        assert code == 1
        for marker in (FAKE_KEY, BODY_MARKER):
            assert marker not in out
            assert marker not in err


def test_unexpected_exceptions_propagate(wired, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(ask, "generate", broken)

    with pytest.raises(RuntimeError, match="bug"):
        ask.main([QUESTION])
