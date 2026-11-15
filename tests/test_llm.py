import json
import logging
import socket
import threading
import time
import tomllib
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from google.genai import _interactions as sdk_errors
from google.genai.interactions import Interaction

from app.core.config import Settings
from app.rag.llm import (
    GeminiClient,
    LLMAuthError,
    LLMBadResponseError,
    LLMConfigError,
    LLMRateLimitError,
    LLMRequest,
    LLMRequestError,
    LLMResult,
    LLMUnavailableError,
    build_http_options,
    build_interaction_kwargs,
    build_llm_client,
    extract_text,
    map_sdk_error,
)
from ingestion.documents import REPO_ROOT

PYPROJECT = REPO_ROOT / "pyproject.toml"
TINY_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
REQUEST = LLMRequest(system="Return JSON.", user="ping", schema=TINY_SCHEMA)


def interaction_body(status="completed", text='{"ok": true}', **extra):
    return {
        "id": "stub-1",
        "status": status,
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        **extra,
    }


def fake_sdk_client(create):
    """A stand-in for genai.Client exposing only what GeminiClient touches."""
    interactions = SimpleNamespace(
        create=create, sdk_configuration=SimpleNamespace(debug_logger=None)
    )
    return SimpleNamespace(interactions=interactions)


EMPTY_TEXT = Interaction(status="completed")
TRUNCATED = Interaction(**interaction_body("incomplete", '{"status": "ans'))
CONTINUATION = Interaction(status="requires_continuation", continuation_token="t")
FAILED = Interaction(status="failed")


@pytest.fixture(autouse=True)
def isolated_google_environment(monkeypatch):
    for name in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENAI_API_VERSION",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_GENAI_HTTP_CLIENT",
        "GOOGLE_GENAI_DEBUG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def stub_server():
    class StubServer(ThreadingHTTPServer):
        request_count = 0
        last_path = None
        last_headers = None
        last_body = None
        paths = None
        redirect_status = None
        response_status = 200
        response_body = "{}"
        delay_seconds = 0.0

        @property
        def url(self):
            return f"http://127.0.0.1:{self.server_port}/"

        def respond(self, status, *, body, delay_seconds=0.0):
            self.response_status = status
            self.response_body = body
            self.delay_seconds = delay_seconds

        def redirect(self, status):
            """Answer the first path with a redirect to /redirect-target."""
            self.redirect_status = status

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            server = self.server
            server.request_count += 1
            server.paths.append(self.path)
            server.last_path = self.path
            server.last_headers = self.headers
            server.last_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(server.delay_seconds)
            if server.redirect_status and self.path != "/redirect-target":
                self.send_response(server.redirect_status)
                self.send_header("Location", "/redirect-target")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            encoded = server.response_body.encode("utf-8")
            try:
                self.send_response(server.response_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # A timeout test intentionally closes its connection before the reply.
                pass

        def log_message(self, *_):
            pass

    server = StubServer(("127.0.0.1", 0), Handler)
    server.paths = []
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_default_pytest_options_exclude_the_llm_marker():
    options = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    assert "not llm" in options["addopts"]
    assert any(marker.startswith("llm:") for marker in options["markers"])


def test_interaction_kwargs_are_stateless_single_shot_and_carry_the_schema():
    kwargs = build_interaction_kwargs(REQUEST, "m-1")
    assert kwargs["model"] == "m-1" and kwargs["store"] is False
    assert kwargs["input"] == "ping" and kwargs["system_instruction"] == "Return JSON."
    assert kwargs["automatic_continuation"] is False
    for banned in ("tools", "previous_interaction_id", "background", "temperature"):
        assert banned not in kwargs
    assert kwargs["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": TINY_SCHEMA,
    }
    assert kwargs["generation_config"] == {"max_output_tokens": 4096, "thinking_level": "minimal"}


def test_http_options_pin_version_milliseconds_and_a_single_attempt():
    options = build_http_options("http://127.0.0.1:1/", 30.0)
    assert options.base_url == "http://127.0.0.1:1/"
    assert (options.api_version, options.timeout) == ("v1", 30000)
    assert options.retry_options.attempts == 1


@pytest.mark.parametrize("bad", [EMPTY_TEXT, TRUNCATED, CONTINUATION, FAILED])
def test_extract_text_rejects_everything_but_completed_with_text(bad):
    with pytest.raises(LLMBadResponseError):
        extract_text(bad)


def test_extract_text_accepts_a_completed_interaction_with_text():
    interaction = Interaction(**interaction_body(text=' {"ok": true} '))
    assert extract_text(interaction) == ' {"ok": true} '


@pytest.mark.parametrize("bad", [None, {}, SimpleNamespace(status="completed", output_text=12)])
def test_extract_text_rejects_invalid_response_objects(bad):
    with pytest.raises(LLMBadResponseError):
        extract_text(bad)


def test_unknown_interaction_status_is_not_echoed():
    interaction = Interaction(**interaction_body("STATUS-MARKER", "BODY-MARKER"))
    with pytest.raises(LLMBadResponseError) as caught:
        extract_text(interaction)
    assert "STATUS-MARKER" not in str(caught.value)
    assert "BODY-MARKER" not in str(caught.value)


def test_successful_http_response_and_wire_body(stub_server):
    stub_server.respond(200, body=json.dumps(interaction_body(model="server-model")))
    result = GeminiClient("fake-key-do-not-leak", "m-1", base_url=stub_server.url).complete(REQUEST)
    assert result == LLMResult('{"ok": true}', "m-1", "server-model")
    body = stub_server.last_body
    assert body["model"] == "m-1" and body["input"] == "ping"
    assert body["store"] is False and body["system_instruction"] == "Return JSON."
    assert body["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": TINY_SCHEMA,
    }
    assert body["generation_config"] == {"max_output_tokens": 4096, "thinking_level": "minimal"}
    for banned in ("automatic_continuation", "tools", "previous_interaction_id", "background"):
        assert banned not in body
    assert stub_server.last_path == "/v1/interactions" and stub_server.request_count == 1
    assert stub_server.last_headers["x-goog-api-key"] == "fake-key-do-not-leak"


@pytest.mark.parametrize("model", [None, ""])
def test_missing_response_model_does_not_replace_the_requested_model(stub_server, model):
    stub_server.respond(200, body=json.dumps(interaction_body(model=model)))
    result = GeminiClient("fake-key-do-not-leak", "m-1", base_url=stub_server.url).complete(REQUEST)
    assert result == LLMResult('{"ok": true}', "m-1", None)


@pytest.mark.parametrize(
    "body",
    [
        interaction_body(text=""),
        interaction_body("incomplete", "BODY-MARKER"),
        interaction_body("failed", "BODY-MARKER"),
        interaction_body("STATUS-MARKER", "BODY-MARKER"),
    ],
)
def test_http_unusable_responses_are_rejected_without_echoing_content(stub_server, body):
    stub_server.respond(200, body=json.dumps(body))
    client = GeminiClient("fake-key-do-not-leak", "m-1", base_url=stub_server.url)
    with pytest.raises(LLMBadResponseError) as caught:
        client.complete(REQUEST)
    assert stub_server.request_count == 1
    text = f"{caught.value!s} {caught.value!r}"
    assert "BODY-MARKER" not in text and "STATUS-MARKER" not in text


def test_http_invalid_json_is_a_sanitized_bad_response(stub_server):
    stub_server.respond(200, body="BODY-MARKER-invalid-json")
    client = GeminiClient("fake-key-do-not-leak", "m-1", base_url=stub_server.url)
    with pytest.raises(LLMBadResponseError) as caught:
        client.complete(REQUEST)
    assert stub_server.request_count == 1
    text = "".join(traceback.format_exception(caught.value))
    assert "BODY-MARKER" not in text and "fake-key-do-not-leak" not in text


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, LLMRequestError),
        (401, LLMAuthError),
        (403, LLMAuthError),
        (404, LLMRequestError),
        (408, LLMRequestError),
        (409, LLMRequestError),
        (422, LLMRequestError),
        (429, LLMRateLimitError),
        (500, LLMUnavailableError),
        (503, LLMUnavailableError),
    ],
)
def test_http_status_maps_to_the_error_family_with_one_request(stub_server, status, error_type):
    stub_server.respond(status, body="BODY-MARKER-do-not-print")
    client = GeminiClient("fake-key-do-not-leak", "m-1", base_url=stub_server.url)
    with pytest.raises(error_type) as caught:
        client.complete(REQUEST)
    text = f"{caught.value!s} {caught.value!r}"
    formatted = "".join(traceback.format_exception(caught.value))
    for marker in ("BODY-MARKER", "fake-key-do-not-leak"):
        assert marker not in text and marker not in formatted
    assert caught.value.__suppress_context__ is True
    assert caught.value.status_code == status
    assert stub_server.request_count == 1
    assert stub_server.last_path == "/v1/interactions"
    if status == 400:
        assert "LLM_API_KEY" in text and "LLM_MODEL" in text


def test_requires_continuation_is_a_bad_response_without_extra_hops_or_warnings(
    stub_server, recwarn
):
    stub_server.respond(200, body=CONTINUATION.model_dump_json())
    client = GeminiClient("fake-key-do-not-leak", "m-1", base_url=stub_server.url)
    with pytest.raises(LLMBadResponseError):
        client.complete(REQUEST)
    assert stub_server.request_count == 1
    assert not recwarn.list


@pytest.mark.parametrize("status", [307, 308])
def test_a_redirect_is_never_followed(stub_server, status):
    stub_server.respond(200, body=json.dumps(interaction_body()))
    stub_server.redirect(status)
    client = GeminiClient("fake-key-do-not-leak", "m-1", base_url=stub_server.url)
    with pytest.raises(LLMUnavailableError) as caught:
        client.complete(REQUEST)
    assert caught.value.status_code == status
    assert stub_server.request_count == 1
    assert stub_server.paths == ["/v1/interactions"]
    text = f"{caught.value!s} {caught.value!r}"
    formatted = "".join(traceback.format_exception(caught.value))
    for marker in ("fake-key-do-not-leak", "redirect-target"):
        assert marker not in text and marker not in formatted


def test_sdk_debug_environment_never_logs_the_key_or_the_response_body(
    stub_server, monkeypatch, caplog
):
    monkeypatch.setenv("GOOGLE_GENAI_DEBUG", "1")
    caplog.set_level(logging.DEBUG)
    stub_server.respond(400, body="BODY-MARKER-do-not-print")
    client = GeminiClient("fake-key-do-not-leak", "m-1", base_url=stub_server.url)
    with pytest.raises(LLMRequestError) as caught:
        client.complete(REQUEST)
    assert stub_server.request_count == 1
    logged = caplog.text + "".join(record.getMessage() for record in caplog.records)
    formatted = "".join(traceback.format_exception(caught.value))
    for marker in ("BODY-MARKER", "fake-key-do-not-leak"):
        assert marker not in logged and marker not in formatted


def test_the_interactions_client_gets_a_silent_logger_scoped_to_itself(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_DEBUG", "1")
    client = GeminiClient("fake-key-do-not-leak", "m-1")
    sdk_logger = client._client.interactions.sdk_configuration.debug_logger
    assert not isinstance(sdk_logger, logging.Logger)
    assert logging.getLogger("google.genai").disabled is False


def test_the_client_ignores_ambient_google_environment(stub_server, monkeypatch):
    for name, value in {
        "GOOGLE_GENAI_API_VERSION": "v1beta",
        "GOOGLE_API_KEY": "ambient",
        "GEMINI_API_KEY": "ambient",
        "GOOGLE_GEMINI_BASE_URL": "http://127.0.0.1:1/",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_GENAI_USE_ENTERPRISE": "true",
    }.items():
        monkeypatch.setenv(name, value)
    stub_server.respond(200, body=json.dumps(interaction_body()))
    result = GeminiClient("explicit", "m-1", base_url=stub_server.url).complete(REQUEST)
    assert result.text == '{"ok": true}'
    assert stub_server.last_path == "/v1/interactions"
    assert stub_server.request_count == 1
    assert stub_server.last_headers["x-goog-api-key"] == "explicit"


def test_a_slow_server_becomes_unavailable(stub_server):
    stub_server.respond(200, body=json.dumps(interaction_body()), delay_seconds=0.2)
    client = GeminiClient(
        "fake-key-do-not-leak", "m-1", base_url=stub_server.url, timeout_seconds=0.05
    )
    with pytest.raises(LLMUnavailableError) as caught:
        client.complete(REQUEST)
    assert caught.value.status_code is None
    assert stub_server.request_count == 1


def test_a_refused_connection_becomes_unavailable():
    # A bound, non-listening socket reserves a local port without a reuse race.
    with socket.socket() as closed_port:
        closed_port.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{closed_port.getsockname()[1]}/"
        client = GeminiClient("fake-key-do-not-leak", "m-1", base_url=url, timeout_seconds=0.05)
        with pytest.raises(LLMUnavailableError) as caught:
            client.complete(REQUEST)
    assert caught.value.status_code is None


def test_sdk_response_validation_error_is_safely_mapped():
    response = httpx.Response(200, request=httpx.Request("POST", "http://127.0.0.1/"))
    error = sdk_errors.APIResponseValidationError(
        response, body="BODY-MARKER", message="fake-key-do-not-leak"
    )
    mapped = map_sdk_error(error)
    assert isinstance(mapped, LLMBadResponseError)
    assert mapped.status_code == 200
    assert "BODY-MARKER" not in f"{mapped!s} {mapped!r}"
    assert "fake-key-do-not-leak" not in f"{mapped!s} {mapped!r}"


def test_other_sdk_errors_are_safely_unavailable():
    error = sdk_errors.APIError(
        "fake-key-do-not-leak", httpx.Request("POST", "http://127.0.0.1/"), body="BODY-MARKER"
    )
    mapped = map_sdk_error(error)
    assert isinstance(mapped, LLMUnavailableError)
    assert "BODY-MARKER" not in f"{mapped!s} {mapped!r}"
    assert "fake-key-do-not-leak" not in f"{mapped!s} {mapped!r}"


def test_a_programming_error_is_not_translated():
    assert map_sdk_error(TypeError("bug")) is None


def test_complete_reraises_programming_errors(monkeypatch):
    error = TypeError("bug")

    def fail(**_):
        raise error

    monkeypatch.setattr(
        "app.rag.llm.genai.Client",
        lambda **_: fake_sdk_client(fail),
    )
    with pytest.raises(TypeError) as caught:
        GeminiClient("fake", "m-1").complete(REQUEST)
    assert caught.value is error


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_or_blank_key_names_llm_api_key(monkeypatch, key):
    monkeypatch.setattr(
        "app.rag.llm.genai.Client", lambda **_: pytest.fail("SDK client must not be built")
    )
    with pytest.raises(LLMConfigError) as caught:
        build_llm_client(Settings(_env_file=None, llm_api_key=key))
    assert str(caught.value) == "LLM_API_KEY is not configured"


def test_unsupported_provider_raises_config_error_without_building_the_sdk(monkeypatch):
    monkeypatch.setattr(
        "app.rag.llm.genai.Client", lambda **_: pytest.fail("SDK client must not be built")
    )
    with pytest.raises(LLMConfigError) as caught:
        build_llm_client(Settings(_env_file=None, llm_provider="other", llm_api_key="fake"))
    assert "LLM_PROVIDER" in str(caught.value)


def test_build_client_passes_explicit_settings_and_safe_transport_options(monkeypatch):
    captured = {}

    def construct(**kwargs):
        captured.update(kwargs)
        return fake_sdk_client(lambda **_: Interaction(**interaction_body()))

    monkeypatch.setattr("app.rag.llm.genai.Client", construct)
    client = build_llm_client(Settings(_env_file=None, llm_api_key="fake", llm_model="m-1"))
    assert isinstance(client, GeminiClient)
    assert captured["api_key"] == "fake" and captured["vertexai"] is False
    options = captured["http_options"]
    assert options.base_url == "https://generativelanguage.googleapis.com/"
    assert options.api_version == "v1" and options.timeout == 30000
    assert options.retry_options.attempts == 1
    assert client.complete(REQUEST).requested_model == "m-1"
