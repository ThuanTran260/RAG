"""A single-shot LLM boundary and the Gemini Interactions adapter."""

import warnings
from dataclasses import dataclass
from typing import Any, Protocol

from google import genai
from google.genai import _interactions as sdk_errors
from google.genai import types

from app.core.config import Settings

GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/"
GEMINI_API_VERSION = "v1"
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_OUTPUT_TOKENS = 4096  # Includes thought tokens.
THINKING_LEVEL = "minimal"


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class LLMResult:
    text: str
    requested_model: str
    response_model: str | None


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResult: ...


class LLMError(RuntimeError):
    """An application error with a sanitized message and optional HTTP status."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMConfigError(LLMError):
    """The provider or API key is not configured."""


class LLMAuthError(LLMError):
    """The provider rejected authentication or authorization."""


class LLMRateLimitError(LLMError):
    """The provider rate limit was reached."""


class LLMRequestError(LLMError):
    """The provider rejected the request."""


class LLMUnavailableError(LLMError):
    """The provider or connection is unavailable."""


class LLMBadResponseError(LLMError):
    """The provider did not return a completed response containing text."""


def build_http_options(base_url: str, timeout_seconds: float) -> types.HttpOptions:
    return types.HttpOptions(
        base_url=base_url,
        api_version=GEMINI_API_VERSION,
        timeout=int(timeout_seconds * 1000),
        retry_options=types.HttpRetryOptions(attempts=1),
        # The SDK follows redirects by default, which would send a second POST.
        client_args={"follow_redirects": False},
    )


class _SilentLogger:
    """Drops the SDK's raw request and response dumps, which include the API key."""

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        pass


def build_interaction_kwargs(request: LLMRequest, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": request.user,
        "system_instruction": request.system,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": request.schema,
        },
        "generation_config": {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_level": THINKING_LEVEL,
        },
        "store": False,
        "automatic_continuation": False,
    }


def extract_text(interaction: Any) -> str:
    """Accept completed text only; never echo arbitrary status or response data."""
    status = getattr(interaction, "status", None)
    if status != "completed":
        documented_statuses = (
            "in_progress",
            "requires_action",
            "failed",
            "cancelled",
            "incomplete",
            "budget_exceeded",
            "queued",
            "requires_continuation",
        )
        detail = f" ({status})" if status in documented_statuses else ""
        raise LLMBadResponseError(f"LLM response is not completed{detail}")
    text = getattr(interaction, "output_text", None)
    if not isinstance(text, str) or not text:
        raise LLMBadResponseError("LLM response has no text")
    return text


def map_sdk_error(error: Exception) -> LLMError | None:
    """Translate Interactions errors without copying their message, body, or request."""
    if isinstance(error, sdk_errors.APITimeoutError):
        return LLMUnavailableError("LLM request timed out")
    if isinstance(error, sdk_errors.APIConnectionError):
        return LLMUnavailableError("LLM connection failed")
    if isinstance(error, sdk_errors.APIResponseValidationError):
        return LLMBadResponseError("LLM response is invalid", status_code=error.status_code)
    if isinstance(error, sdk_errors.APIStatusError):
        status = error.status_code
        if status in (401, 403):
            return LLMAuthError(f"LLM authentication failed (HTTP {status})", status)
        if status == 429:
            return LLMRateLimitError("LLM rate limit reached (HTTP 429)", status)
        if 400 <= status < 500:
            hint = "; check LLM_API_KEY and LLM_MODEL" if status == 400 else ""
            return LLMRequestError(f"LLM request rejected (HTTP {status}){hint}", status)
        return LLMUnavailableError(f"LLM provider unavailable (HTTP {status})", status)
    if isinstance(error, sdk_errors.APIError):
        return LLMUnavailableError("LLM SDK request failed", status_code=error.status_code)
    return None


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = GEMINI_BASE_URL,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._client = genai.Client(
            api_key=api_key,
            vertexai=False,
            http_options=build_http_options(base_url, timeout_seconds),
        )
        # GOOGLE_GENAI_DEBUG would make this client log request headers (the API key)
        # and response bodies, so replace its logger before the first request.
        # Only this client is affected; the process-wide logging setup is untouched.
        self._client.interactions.sdk_configuration.debug_logger = _SilentLogger()
        self._model = model

    def complete(self, request: LLMRequest) -> LLMResult:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Interaction stopped before completion", category=UserWarning
                )
                interaction = self._client.interactions.create(
                    **build_interaction_kwargs(request, self._model)
                )
        except Exception as error:
            mapped = map_sdk_error(error)
            if mapped is None:
                raise
            raise mapped from None
        model = getattr(interaction, "model", None)
        return LLMResult(
            text=extract_text(interaction),
            requested_model=self._model,
            response_model=model if isinstance(model, str) and model else None,
        )


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider != "gemini":
        raise LLMConfigError("LLM_PROVIDER is unsupported; expected gemini")
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key is not None else None
    if api_key is None or not api_key.strip():
        raise LLMConfigError("LLM_API_KEY is not configured")
    return GeminiClient(api_key=api_key, model=settings.llm_model)
