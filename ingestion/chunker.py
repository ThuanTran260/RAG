from ingestion.models import Article, ArticleRecord, Chunk, DocumentMeta


def _heading(article: Article) -> str:
    return f"Điều {article.number}. {article.title}"


def _context_text(article: Article, doc: DocumentMeta, text: str, with_intro: bool) -> str:
    lines = [doc.title, f"Chương {article.chapter} - {article.chapter_title}"]
    if article.section is not None:
        lines.append(f"Mục {article.section} - {article.section_title}")
    lines.append(_heading(article))
    if with_intro and article.intro:
        lines.append(article.intro)
    lines.append(text)
    return "\n".join(lines)


def build_chunks(articles: list[Article], doc: DocumentMeta) -> list[Chunk]:
    """One chunk per clause; an article without clauses becomes a single chunk."""
    chunks: list[Chunk] = []
    for article in articles:
        base_id = f"{doc.doc_id}:d{article.number}"
        # (clause label, verbatim text) pairs; the intro stands alone when there are no clauses
        parts = (
            [(clause.label, clause.text) for clause in article.clauses]
            if article.clauses
            else [(None, article.intro)]
        )
        for label, text in parts:
            chunks.append(
                Chunk(
                    id=base_id if label is None else f"{base_id}:k{label}",
                    doc_id=doc.doc_id,
                    doc_no=doc.doc_no,
                    doc_title=doc.title,
                    source_no=doc.source_no,
                    chapter=article.chapter,
                    chapter_title=article.chapter_title,
                    section=article.section,
                    section_title=article.section_title,
                    article=article.number,
                    article_title=article.title,
                    clause=label,
                    text=text,
                    context_text=_context_text(article, doc, text, with_intro=label is not None),
                    char_len=len(text),
                    checked_at=doc.checked_at,
                )
            )
    return chunks


def build_article_records(articles: list[Article], doc: DocumentMeta) -> list[ArticleRecord]:
    """One record per article holding its full text, for article-level experiments."""
    records: list[ArticleRecord] = []
    for article in articles:
        lines = [_heading(article)]
        if article.intro:
            lines.append(article.intro)
        lines.extend(clause.text for clause in article.clauses)
        records.append(
            ArticleRecord(
                id=f"{doc.doc_id}:d{article.number}",
                doc_id=doc.doc_id,
                doc_no=doc.doc_no,
                doc_title=doc.title,
                source_no=doc.source_no,
                chapter=article.chapter,
                chapter_title=article.chapter_title,
                section=article.section,
                section_title=article.section_title,
                article=article.number,
                article_title=article.title,
                text="\n".join(lines),
                checked_at=doc.checked_at,
            )
        )
    return records
