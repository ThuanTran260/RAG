"""Build command: DOCX -> articles.jsonl and chunks.jsonl, with a checksum guard on the source."""

import argparse
import hashlib
import random
import statistics
import sys
from pathlib import Path

from pydantic import BaseModel

from ingestion.chunker import build_article_records, build_chunks
from ingestion.documents import LABOUR_CODE, REPO_ROOT
from ingestion.docx_loader import load_paragraphs
from ingestion.models import Chunk, DocumentMeta
from ingestion.parser import ParseError, parse_articles


class BuildStats(BaseModel):
    articles: int
    chunks: int
    char_len_min: int
    char_len_median: float
    char_len_max: int
    char_len_total: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records: list[BaseModel]) -> None:
    # Fixed LF newlines keep the output byte-identical across platforms.
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def build(doc: DocumentMeta, out_dir: Path) -> BuildStats:
    """Parse the source file of `doc` and write the JSONL files into `out_dir`."""
    actual = _sha256(doc.path)
    if actual != doc.sha256:
        raise ValueError(
            f"checksum mismatch for {doc.path.name}: expected {doc.sha256}, got {actual}"
        )

    articles = parse_articles(load_paragraphs(doc.path))
    if len(articles) != doc.expected_articles:
        raise ParseError(f"expected {doc.expected_articles} articles, found {len(articles)}")

    chunks = build_chunks(articles, doc)
    records = build_article_records(articles, doc)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "articles.jsonl", records)
    _write_jsonl(out_dir / "chunks.jsonl", chunks)

    lengths = [c.char_len for c in chunks]
    return BuildStats(
        articles=len(articles),
        chunks=len(chunks),
        char_len_min=min(lengths),
        char_len_median=statistics.median(lengths),
        char_len_max=max(lengths),
        char_len_total=sum(lengths),
    )


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not use UTF-8; Vietnamese output must not crash.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Build articles.jsonl and chunks.jsonl from the Labour Code DOCX."
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument(
        "--sample", type=int, default=0, help="print N random chunks for a visual check"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    stats = build(LABOUR_CODE, args.out)
    print(f"articles: {stats.articles}")
    print(f"chunks: {stats.chunks}")
    print(
        f"char_len min/median/max: {stats.char_len_min} / {stats.char_len_median} / {stats.char_len_max}"
    )

    if args.sample > 0:
        lines = (args.out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines]
        for chunk in random.Random(args.seed).sample(chunks, args.sample):
            print(f"--- {chunk.id}")
            print(chunk.context_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
