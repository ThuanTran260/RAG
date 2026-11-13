from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class DocumentMeta(BaseModel):
    """Metadata describing one legal source file under data/raw/."""

    model_config = ConfigDict(frozen=True)

    doc_id: str  # used as the prefix of chunk ids
    doc_no: str
    title: str
    source_no: str
    source_date: date
    source_url: str
    path: Path  # absolute path to the raw file
    sha256: str
    checked_at: date  # day the legal status was last checked
    expected_articles: int
