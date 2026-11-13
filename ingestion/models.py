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


class Clause(BaseModel):
    label: str  # "1", "8a"
    text: str  # every paragraph of the clause joined with "\n", label included


class Article(BaseModel):
    number: int
    title: str
    chapter: str  # roman numeral, e.g. "III"
    chapter_title: str
    section: int | None
    section_title: str | None
    intro: str  # paragraphs before the first clause joined with "\n"; "" if none
    clauses: list[Clause]
