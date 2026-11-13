"""Turn the cleaned paragraph list into Chapter -> Section -> Article -> Clause structure."""

import re

from ingestion.models import Article, Clause

RE_CHAPTER = re.compile(r"^Chương ([IVXLC]+)$")
RE_SECTION = re.compile(r"^Mục (\d+)$")
RE_ARTICLE = re.compile(r"^Điều (\d+)\. (.+)$")
RE_CLAUSE = re.compile(r"^(\d+[a-z]?)\. ")


def parse_articles(paragraphs: list[str]) -> list[Article]:
    """Read paragraphs in order and build the articles with an explicit state machine."""
    articles: list[Article] = []

    # Heading state: the Chương/Mục line just seen, whose title paragraphs are still being read.
    chapter: str | None = None  # None until the first Chương line (rule 1)
    chapter_title = ""
    section: int | None = None
    section_title: str | None = None
    heading_target: str | None = None  # "chapter", "section", or None when not reading a title
    heading_parts: list[str] = []

    # Article state: the Điều currently open, if any.
    art_number = 0
    art_title = ""
    intro_parts: list[str] = []
    clauses: list[tuple[str, list[str]]] = []  # (label, paragraphs) of finished and open clauses
    in_article = False

    def flush_heading() -> None:
        nonlocal chapter_title, section_title, heading_target, heading_parts
        title = " ".join(heading_parts)
        if heading_target == "chapter":
            chapter_title = title
        elif heading_target == "section":
            section_title = title
        heading_target = None
        heading_parts = []

    def flush_article() -> None:
        nonlocal in_article, intro_parts, clauses
        if in_article:
            assert chapter is not None
            articles.append(Article(
                number=art_number,
                title=art_title,
                chapter=chapter,
                chapter_title=chapter_title,
                section=section,
                section_title=section_title,
                intro="\n".join(intro_parts),
                clauses=[Clause(label=label, text="\n".join(parts)) for label, parts in clauses],
            ))
        in_article = False
        intro_parts = []
        clauses = []

    for para in paragraphs:
        m_chapter = RE_CHAPTER.match(para)
        m_section = RE_SECTION.match(para)
        m_article = RE_ARTICLE.match(para)

        if chapter is None and not m_chapter:
            continue  # rule 1: skip the preamble before the first Chương

        if m_chapter:  # rule 3: a new chapter resets the section
            flush_heading()
            flush_article()
            chapter = m_chapter.group(1)
            chapter_title = ""
            section = None
            section_title = None
            heading_target = "chapter"
        elif m_section:
            flush_heading()
            flush_article()
            section = int(m_section.group(1))
            section_title = ""
            heading_target = "section"
        elif m_article:  # rule 4: open a new article carrying the current chapter and section
            flush_heading()
            flush_article()
            art_number = int(m_article.group(1))
            art_title = m_article.group(2)
            in_article = True
        elif heading_target is not None:  # rule 2: title paragraphs, joined later with a space
            heading_parts.append(para)
        elif in_article:  # rule 5: clause start, or continuation of the open clause or the intro
            m_clause = RE_CLAUSE.match(para)
            if m_clause:
                clauses.append((m_clause.group(1), [para]))
            elif clauses:
                clauses[-1][1].append(para)
            else:
                intro_parts.append(para)

    flush_heading()
    flush_article()
    return articles
