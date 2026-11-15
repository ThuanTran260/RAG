"""Validate the model's JSON reply, build citation labels from chunk metadata and render it.

No unvalidated model text is ever printed or put in an error message. Source labels, article
titles and verbatim text come from ``Chunk`` metadata, never from the model.
"""

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.rag.llm import LLMClient
from app.rag.prompts import (
    MAX_CLAIMS,
    MAX_SOURCES_PER_CLAIM,
    Source,
    assign_sources,
    build_request,
)
from app.rag.retriever import RetrievedChunk, prepare_query
from ingestion.models import Chunk

ARTICLE_MENTION = re.compile(r"\bđiều\s+(\d+(?:\s*(?:,|và|hoặc)\s*\d+)*)", re.IGNORECASE)
MAX_CLAIM_CHARS = 800
INSUFFICIENT_TEXT = "Không tìm thấy căn cứ trong dữ liệu đã truy xuất để trả lời câu hỏi này."
VERIFICATION_NOTICE = (
    "Lưu ý: mã nguồn [S#], nhãn Điều/Khoản và nguyên văn do hệ thống lấy từ dữ liệu đã truy xuất; "
    "hệ thống không kiểm tra câu trả lời có được nguyên văn ủng hộ hay không. "
    "Hãy đối chiếu với nguyên văn ở trên."
)
DISCLAIMER = "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý."


class AnswerValidationError(RuntimeError):
    """The model reply broke the contract; ``code`` is a fixed identifier, never model text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    sources: list[str]


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "insufficient"]
    claims: list[Claim]


class CitedSource(BaseModel):
    source_id: str  # "S1"
    chunk_id: str
    label: str  # "Điều 154, khoản 8a"
    article_title: str
    text: str  # chunk.text, verbatim
    score: float


class GenerationResult(BaseModel):
    status: Literal["answered", "insufficient"]
    reason: Literal["no_contexts", "model_insufficient"] | None  # None when answered
    claims: list[Claim]  # empty when insufficient
    sources: list[CitedSource]  # only the sources used by claims, in order of first use
    unsupported_articles: list[int]  # sorted; see unsupported_article_mentions
    requested_model: str | None  # None when no LLM call was made
    response_model: str | None  # as returned by the server, may be absent
    retrieved: list[RetrievedChunk]  # all contexts with scores, for Week 3 calibration


def source_label(chunk: Chunk) -> str:
    label = f"Điều {chunk.article}"
    if chunk.clause is not None:
        label += f", khoản {chunk.clause}"
    return label


def _normalise_claim_text(text: str) -> str:
    """NFC, then collapse every whitespace run (line breaks included) so a claim is one line."""
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not text:
        raise AnswerValidationError("empty_claim")
    if any(unicodedata.category(char) == "Cc" for char in text):
        raise AnswerValidationError("schema")
    if len(text) > MAX_CLAIM_CHARS:
        raise AnswerValidationError("claim_too_long")
    return text


def _structural_error_code(error: ValidationError) -> str:
    """Pydantic reports unparsable JSON as ``json_invalid``; any other error is a schema break."""
    if all(item["type"] == "json_invalid" for item in error.errors(include_input=False)):
        return "not_json"
    return "schema"


def validate_answer(raw: str, source_ids: Sequence[str]) -> ModelAnswer:
    """Parse and check the model reply; raise AnswerValidationError(code) on any violation."""
    code: str | None = None
    answer: ModelAnswer | None = None
    try:
        answer = ModelAnswer.model_validate_json(raw, strict=True)
    except ValidationError as error:
        code = _structural_error_code(error)
    if answer is None:
        # Raised outside the except block so no pydantic error (and no input) stays attached.
        raise AnswerValidationError(code or "schema")
    if answer.status == "insufficient":
        return ModelAnswer(status="insufficient", claims=[])
    if not answer.claims:
        raise AnswerValidationError("no_claims")
    if len(answer.claims) > MAX_CLAIMS:
        raise AnswerValidationError("too_many_claims")
    allowed = set(source_ids)
    claims: list[Claim] = []
    for claim in answer.claims:
        text = _normalise_claim_text(claim.text)
        if not claim.sources:
            raise AnswerValidationError("no_sources")
        if len(claim.sources) > MAX_SOURCES_PER_CLAIM:  # counted before de-duplication
            raise AnswerValidationError("too_many_sources")
        if any(source not in allowed for source in claim.sources):
            raise AnswerValidationError("unknown_source")
        claims.append(Claim(text=text, sources=list(dict.fromkeys(claim.sources))))
    return ModelAnswer(status="answered", claims=claims)


def article_mentions(text: str) -> set[int]:
    """Every article number that follows "Điều" in the text, lists such as "35, 36 và 37" included."""
    numbers: set[int] = set()
    for match in ARTICLE_MENTION.finditer(unicodedata.normalize("NFC", text)):
        numbers.update(int(number) for number in re.findall(r"\d+", match.group(1)))
    return numbers


def unsupported_article_mentions(
    claims: Sequence[Claim], contexts: Sequence[RetrievedChunk]
) -> list[int]:
    """Article numbers the claims mention that no retrieved chunk is, or mentions itself.

    Advisory only: it neither blocks an answer nor proves that a source entails a claim.
    """
    mentioned: set[int] = set()
    for claim in claims:
        mentioned |= article_mentions(claim.text)
    known: set[int] = set()
    for context in contexts:
        known.add(context.chunk.article)
        known |= article_mentions(context.chunk.context_text)
    return sorted(mentioned - known)


def _result(
    answer: ModelAnswer,
    contexts: Sequence[RetrievedChunk],
    sources: Sequence[Source],
    requested_model: str,
    response_model: str | None,
) -> GenerationResult:
    retrieved = list(contexts)
    if answer.status == "insufficient":
        return GenerationResult(
            status="insufficient",
            reason="model_insufficient",
            claims=[],
            sources=[],
            unsupported_articles=[],
            requested_model=requested_model,
            response_model=response_model,
            retrieved=retrieved,
        )
    by_id = {source.id: source.retrieved for source in sources}
    used = list(dict.fromkeys(source for claim in answer.claims for source in claim.sources))
    cited = [
        CitedSource(
            source_id=source_id,
            chunk_id=by_id[source_id].chunk.id,
            label=source_label(by_id[source_id].chunk),
            article_title=by_id[source_id].chunk.article_title,
            text=by_id[source_id].chunk.text,
            score=by_id[source_id].score,
        )
        for source_id in used
    ]
    return GenerationResult(
        status="answered",
        reason=None,
        claims=answer.claims,
        sources=cited,
        unsupported_articles=unsupported_article_mentions(answer.claims, contexts),
        requested_model=requested_model,
        response_model=response_model,
        retrieved=retrieved,
    )


def generate(query: str, contexts: Sequence[RetrievedChunk], client: LLMClient) -> GenerationResult:
    """Answer from the contexts with at most one LLM call; the caller handles errors."""
    query = prepare_query(query, 1)
    if not contexts:
        return GenerationResult(
            status="insufficient",
            reason="no_contexts",
            claims=[],
            sources=[],
            unsupported_articles=[],
            requested_model=None,
            response_model=None,
            retrieved=[],
        )
    sources = assign_sources(contexts)
    source_ids = [source.id for source in sources]
    response = client.complete(build_request(query, sources))
    answer = validate_answer(response.text, source_ids)
    return _result(answer, contexts, sources, response.requested_model, response.response_model)


def render_result(result: GenerationResult) -> str:
    if result.status == "insufficient":
        return f"{INSUFFICIENT_TEXT}\n\n{DISCLAIMER}"
    lines = [
        f"{number}. {claim.text} [{', '.join(claim.sources)}]"
        for number, claim in enumerate(result.claims, start=1)
    ]
    lines += ["", "Căn cứ:"]
    for index, source in enumerate(result.sources):
        if index:
            lines.append("")
        lines.append(
            f"[{source.source_id}] {source.label} — {source.article_title} "
            f"({source.chunk_id}, điểm {source.score:.4f})"
        )
        lines.append(source.text)
    lines.append("")
    lines += [
        f"Cảnh báo: câu trả lời nhắc Điều {number} không có trong các nguồn đã truy xuất; "
        "hãy đối chiếu."
        for number in result.unsupported_articles
    ]
    lines += [VERIFICATION_NOTICE, DISCLAIMER]
    return "\n".join(lines)
