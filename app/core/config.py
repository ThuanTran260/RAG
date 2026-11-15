"""Application settings and the identifier shared by the Qdrant collection and the cache."""

import hashlib
import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def model_slug(model_name: str) -> str:
    """Lowercase the name and collapse every run of non-alphanumerics into one dash.

    Only for human readability: different names can share a slug, so it is not an identifier.
    """
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def model_key(model_name: str) -> str:
    """Unique identifier for a model name: a readable slug plus a hash of the exact name."""
    if not model_name.strip():
        raise ValueError("model_name must not be blank")
    slug = model_slug(model_name) or "model"
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    return f"{slug}--{digest}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "labor_law_vn"
    embedding_model: str = "BAAI/bge-m3"
    # None lets sentence-transformers pick the GPU when one is available.
    embedding_device: str | None = None
    embedding_batch_size: int = Field(default=16, ge=1)
    llm_provider: str = "gemini"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gemini-3.5-flash-lite"

    @field_validator(
        "qdrant_url", "qdrant_collection", "embedding_model", "llm_provider", "llm_model"
    )
    @classmethod
    def _strip_and_require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("embedding_device")
    @classmethod
    def _blank_device_means_unset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _blank_key_means_unset(cls, value: str | SecretStr | None) -> str | SecretStr | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def collection_name(self) -> str:
        return f"{self.qdrant_collection}__{model_key(self.embedding_model)}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
