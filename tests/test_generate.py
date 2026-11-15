import ast
import json
import traceback
import unicodedata
from pathlib import Path

import pytest

from app.rag import generate as generate_module
from app.rag.generate import (
    DISCLAIMER,
    INSUFFICIENT_TEXT,
    VERIFICATION_NOTICE,
    AnswerValidationError,
    Claim,
    article_mentions,
    generate,
    render_result,
    source_label,
    unsupported_article_mentions,
    validate_answer,
)
from app.rag.llm import LLMBadResponseError, LLMRateLimitError
from app.rag.prompts import GenerationInputError, assign_sources, build_request
from app.rag.retriever import InvalidQueryError

DOC = "45-2019-QH14"
QUERY = "Câu hỏi giả lập?"
MARKER = "REPLY-MARKER"


def cid(suffix: str) -> str:
    return f"{DOC}:{suffix}"


def reply(*claims: tuple[str, list[str]], status: str = "answered") -> str:
    return json.dumps(
        {"status": status, "claims": [{"text": text, "sources": src} for text, src in claims]},
        ensure_ascii=False,
    )


def header_of(retrieved, source_id: str) -> str:
    chunk = retrieved.chunk
    return (
        f"[{source_id}] {source_label(chunk)} — {chunk.article_title} "
        f"({chunk.id}, điểm {retrieved.score:.4f})"
    )


def run(contexts, body, make_fake_llm_client, query=QUERY):
    client = make_fake_llm_client(body)
    return generate(query, contexts, client), client


# --- labels and rendering -------------------------------------------------------------------


def test_answered_reply_renders_labels_and_verbatim_text_from_metadata(
    contexts_for, make_fake_llm_client
):
    contexts = contexts_for([cid("d34:k1"), cid("d1")])
    body = reply(("Khẳng định giả lập thứ nhất.", ["S1", "S2"]))
    result, client = run(contexts, body, make_fake_llm_client)
    output = render_result(result)

    assert client.calls == 1
    assert result.status == "answered"
    assert result.reason is None
    assert [s.label for s in result.sources] == ["Điều 34, khoản 1", "Điều 1"]
    lines = output.split("\n")
    assert lines[0] == "1. Khẳng định giả lập thứ nhất. [S1, S2]"
    assert lines[1] == ""
    assert lines[2] == "Căn cứ:"
    for source_id, retrieved in zip(("S1", "S2"), contexts):
        assert header_of(retrieved, source_id) + "\n" + retrieved.chunk.text in output
    assert output.endswith(VERIFICATION_NOTICE + "\n" + DISCLAIMER)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("d1", "Điều 1"),
        ("d154:k8a", "Điều 154, khoản 8a"),
        ("d34:k1", "Điều 34, khoản 1"),
    ],
)
def test_source_label_edge_cases(all_chunks, suffix, expected):
    chunk = next(c for c in all_chunks if c.id == cid(suffix))
    assert source_label(chunk) == expected
    assert "None" not in source_label(chunk)


def test_rendered_source_text_is_the_full_verbatim_chunk(contexts_for, make_fake_llm_client):
    contexts = contexts_for([cid("d219:k1")])
    result, _ = run(contexts, reply(("Khẳng định giả lập.", ["S1"])), make_fake_llm_client)
    chunk = contexts[0].chunk
    output = render_result(result)

    assert len(chunk.text) == 4512
    assert result.sources[0].text == chunk.text
    assert chunk.text in output
    assert header_of(contexts[0], "S1") + "\n" + chunk.text in output


def test_source_text_with_line_breaks_is_not_collapsed(contexts_for, make_fake_llm_client):
    contexts = contexts_for([cid("d83")])
    chunk = contexts[0].chunk
    assert "\n" in chunk.text
    result, _ = run(contexts, reply(("Khẳng định giả lập.", ["S1"])), make_fake_llm_client)

    assert result.sources[0].text == chunk.text
    assert header_of(contexts[0], "S1") + "\n" + chunk.text in render_result(result)


def test_sources_are_ordered_by_first_use_and_deduplicated(contexts_for, make_fake_llm_client):
    contexts = contexts_for([cid("d1"), cid("d2:k1"), cid("d9:k1")])
    body = reply(
        ("Khẳng định giả lập A.", ["S3", "S1", "S3"]),
        ("Khẳng định giả lập B.", ["S1", "S2", "S3"]),
    )
    result, _ = run(contexts, body, make_fake_llm_client)

    assert [s.source_id for s in result.sources] == ["S3", "S1", "S2"]
    assert [s.chunk_id for s in result.sources] == [
        contexts[2].chunk.id,
        contexts[0].chunk.id,
        contexts[1].chunk.id,
    ]
    assert [c.sources for c in result.claims] == [["S3", "S1"], ["S1", "S2", "S3"]]
    lines = render_result(result).split("\n")
    assert lines[0] == "1. Khẳng định giả lập A. [S3, S1]"
    assert lines[1] == "2. Khẳng định giả lập B. [S1, S2, S3]"
    headings = [line for line in lines if line.startswith("[S")]
    assert [h[:4] for h in headings] == ["[S3]", "[S1]", "[S2]"]


def test_each_cited_label_maps_to_its_own_chunk_label_and_verbatim_text(
    contexts_for, make_fake_llm_client
):
    contexts = contexts_for([cid("d34:k1"), cid("d1"), cid("d154:k8a")])
    body = reply(("Khẳng định giả lập A.", ["S3"]), ("Khẳng định giả lập B.", ["S1", "S2"]))
    result, _ = run(contexts, body, make_fake_llm_client)
    output = render_result(result)

    by_label = {"S1": contexts[0], "S2": contexts[1], "S3": contexts[2]}
    assert [s.source_id for s in result.sources] == ["S3", "S1", "S2"]
    for cited in result.sources:
        expected = by_label[cited.source_id]
        assert cited.chunk_id == expected.chunk.id
        assert cited.label == source_label(expected.chunk)
        assert cited.article_title == expected.chunk.article_title
        assert cited.text == expected.chunk.text
        assert cited.score == expected.score
        assert header_of(expected, cited.source_id) + "\n" + expected.chunk.text in output
    assert [s.label for s in result.sources] == [
        "Điều 154, khoản 8a",
        "Điều 34, khoản 1",
        "Điều 1",
    ]


def test_result_keeps_every_retrieved_chunk_with_its_score_and_both_model_ids(
    contexts_for, make_fake_llm_client
):
    contexts = contexts_for([cid("d1"), cid("d2:k1"), cid("d9:k1")])
    result, _ = run(contexts, reply(("Khẳng định giả lập.", ["S2"])), make_fake_llm_client)

    assert result.retrieved == contexts
    assert [r.score for r in result.retrieved] == [r.score for r in contexts]
    assert [s.source_id for s in result.sources] == ["S2"]
    assert result.sources[0].score == contexts[1].score
    assert result.sources[0].article_title == contexts[1].chunk.article_title
    assert result.requested_model == "fake-model"
    assert result.response_model == "fake-model-resp"


def test_response_model_may_be_absent(contexts_for, make_fake_llm_client):
    contexts = contexts_for([cid("d1")])
    client = make_fake_llm_client(reply(("Khẳng định giả lập.", ["S1"])), response_model=None)
    result = generate(QUERY, contexts, client)

    assert result.requested_model == "fake-model"
    assert result.response_model is None


def test_request_matches_build_request_and_is_sent_once(contexts_for, make_fake_llm_client):
    contexts = contexts_for([cid("d1"), cid("d2:k1")])
    result, client = run(contexts, reply(("Khẳng định giả lập.", ["S1"])), make_fake_llm_client)

    assert client.calls == 1
    assert client.requests[0] == build_request(QUERY, assign_sources(contexts))
    assert result.status == "answered"


# --- inputs ---------------------------------------------------------------------------------


def test_empty_contexts_return_insufficient_without_calling_the_client(make_fake_llm_client):
    client = make_fake_llm_client(reply(("Không dùng.", ["S1"])))
    result = generate(QUERY, [], client)

    assert client.calls == 0
    assert result.status == "insufficient"
    assert result.reason == "no_contexts"
    assert result.claims == []
    assert result.sources == []
    assert result.unsupported_articles == []
    assert result.retrieved == []
    assert result.requested_model is None
    assert result.response_model is None


@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
@pytest.mark.parametrize("with_contexts", [True, False])
def test_blank_query_is_rejected_before_the_client_is_called(
    query, with_contexts, contexts_for, make_fake_llm_client
):
    contexts = contexts_for([cid("d1")]) if with_contexts else []
    client = make_fake_llm_client(reply(("Không dùng.", ["S1"])))

    with pytest.raises(InvalidQueryError):
        generate(query, contexts, client)
    assert client.calls == 0


def test_query_is_nfc_normalised_in_the_request(contexts_for, make_fake_llm_client):
    nfd = unicodedata.normalize("NFD", "  Thời giờ làm việc?  ")
    assert unicodedata.normalize("NFC", nfd) != nfd
    _, client = run(
        contexts_for([cid("d1")]),
        reply(("Khẳng định giả lập.", ["S1"])),
        make_fake_llm_client,
        query=nfd,
    )

    question = json.loads(client.requests[0].user)["question"]
    assert question == "Thời giờ làm việc?"
    assert unicodedata.normalize("NFC", question) == question


def test_more_than_eight_or_duplicate_contexts_are_rejected_before_the_client(
    all_chunks, contexts_for, make_fake_llm_client
):
    ids = [chunk.id for chunk in all_chunks[:9]]
    client = make_fake_llm_client(reply(("Không dùng.", ["S1"])))

    with pytest.raises(GenerationInputError):
        generate(QUERY, contexts_for(ids), client)
    with pytest.raises(GenerationInputError):
        generate(QUERY, contexts_for([ids[0], ids[1], ids[0]]), client)
    assert client.calls == 0


def test_exactly_eight_contexts_are_accepted(all_chunks, contexts_for, make_fake_llm_client):
    contexts = contexts_for([chunk.id for chunk in all_chunks[:8]])
    result, client = run(contexts, reply(("Khẳng định giả lập.", ["S8"])), make_fake_llm_client)

    assert client.calls == 1
    assert [s.source_id for s in result.sources] == ["S8"]
    assert len(result.retrieved) == 8


# --- model reply: insufficient and errors ---------------------------------------------------


def test_insufficient_reply_gives_the_fixed_text_and_ignores_claims(
    contexts_for, make_fake_llm_client
):
    contexts = contexts_for([cid("d1")])
    # The leftover claims break every business rule; insufficient still wins.
    leftovers = [("Văn bản\x1b thừa.", ["S9"]), ("   ", []), ("x" * 900, ["S1"] * 5)]
    result, _ = run(contexts, reply(*leftovers, status="insufficient"), make_fake_llm_client)

    assert result.status == "insufficient"
    assert result.reason == "model_insufficient"
    assert result.claims == []
    assert result.sources == []
    assert result.unsupported_articles == []
    assert result.retrieved == contexts
    assert result.requested_model == "fake-model"
    assert result.response_model == "fake-model-resp"
    assert render_result(result) == INSUFFICIENT_TEXT + "\n\n" + DISCLAIMER


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"status": "insufficient", "claims": [], "extra": 1}),
        json.dumps({"status": "insufficient", "claims": [{"text": 1, "sources": ["S1"]}]}),
        json.dumps({"status": "insufficient", "claims": [{"text": "a", "sources": "S1"}]}),
        json.dumps(
            {"status": "insufficient", "claims": [{"text": "a", "sources": ["S1"], "x": 0}]}
        ),
        json.dumps({"status": "insufficient"}),
        json.dumps({"claims": []}),
        json.dumps({"status": "insufficient", "claims": None}),
    ],
)
def test_insufficient_still_needs_a_valid_structure(body):
    with pytest.raises(AnswerValidationError) as info:
        validate_answer(body, ["S1"])
    assert info.value.code == "schema"


INVALID_CLAIM = ("Khẳng định giả lập.", ["S1"])
LONG_UNDER_LIMIT = "a" * 800
INVALID_REPLIES = [
    pytest.param(f"{MARKER} not json", "not_json", id="not-json"),
    pytest.param("", "not_json", id="empty-body"),
    pytest.param(json.dumps([MARKER]), "schema", id="json-array"),
    pytest.param(
        json.dumps({"claims": [{"text": MARKER, "sources": ["S1"]}]}), "schema", id="no-status"
    ),
    pytest.param(json.dumps({"status": "answered"}), "schema", id="no-claims-field"),
    pytest.param(json.dumps({"status": MARKER, "claims": []}), "schema", id="unknown-status"),
    pytest.param(json.dumps({"status": None, "claims": []}), "schema", id="null-status"),
    pytest.param(
        json.dumps({"status": "answered", "claims": MARKER}), "schema", id="claims-not-list"
    ),
    pytest.param(
        json.dumps({"status": "insufficient", "claims": [], "note": MARKER}),
        "schema",
        id="extra-field",
    ),
    pytest.param(
        json.dumps(
            {"status": "answered", "claims": [{"text": MARKER, "sources": ["S1"], "quote": "x"}]}
        ),
        "schema",
        id="extra-claim-field",
    ),
    pytest.param(
        json.dumps({"status": "answered", "claims": [{"text": 5, "sources": [MARKER]}]}),
        "schema",
        id="text-not-string",
    ),
    pytest.param(
        json.dumps({"status": "answered", "claims": [{"text": MARKER, "sources": "S1"}]}),
        "schema",
        id="sources-not-list",
    ),
    pytest.param(
        json.dumps({"status": "answered", "claims": [{"text": MARKER, "sources": [1]}]}),
        "schema",
        id="source-not-string",
    ),
    pytest.param(reply(), "no_claims", id="answered-without-claims"),
    pytest.param(reply((MARKER, [])), "no_sources", id="claim-without-sources"),
    pytest.param(reply((MARKER, ["S9"])), "unknown_source", id="S9"),
    pytest.param(reply((MARKER, ["s1"])), "unknown_source", id="lowercase-label"),
    pytest.param(reply((MARKER, [" S1"])), "unknown_source", id="leading-space-label"),
    pytest.param(reply((MARKER, ["S1 "])), "unknown_source", id="trailing-space-label"),
    pytest.param(reply((MARKER, ["S1\n"])), "unknown_source", id="trailing-newline-label"),
    pytest.param(reply((MARKER, ["S１"])), "unknown_source", id="full-width-digit-label"),
    pytest.param(reply((MARKER, ["Ｓ1"])), "unknown_source", id="full-width-letter-label"),
    pytest.param(reply((MARKER, ["S3"])), "unknown_source", id="label-beyond-request-size"),
    pytest.param(reply(("   ", ["S1"])), "empty_claim", id="blank-claim"),
    pytest.param(
        reply((MARKER + "x" * (801 - len(MARKER)), ["S1"])), "claim_too_long", id="801-chars"
    ),
    pytest.param(reply(*([INVALID_CLAIM] * 7)), "too_many_claims", id="seven-claims"),
    pytest.param(reply((MARKER, ["S1", "S2", "S1", "S2"])), "too_many_sources", id="four-sources"),
    pytest.param(reply((MARKER + "\x1b", ["S1"])), "schema", id="escape-in-claim"),
    pytest.param(f"```json\n{reply(INVALID_CLAIM)}\n```  {MARKER}", "not_json", id="code-fence"),
    pytest.param(reply(INVALID_CLAIM) + f" {MARKER}", "not_json", id="trailing-text"),
    pytest.param(reply(INVALID_CLAIM)[:-1] + MARKER, "not_json", id="truncated"),
    pytest.param(
        '{"status":"answered","claims":[{"text":"\\ud800 ' + MARKER + '","sources":["S1"]}]}',
        "not_json",
        id="lone-escaped-surrogate",
    ),
    pytest.param("[" * 100000 + MARKER, "not_json", id="deeply-nested"),
]


@pytest.mark.parametrize(("body", "code"), INVALID_REPLIES)
def test_invalid_replies_raise_validation_error_and_never_echo_the_reply(
    body, code, contexts_for, make_fake_llm_client
):
    contexts = contexts_for([cid("d1"), cid("d2:k1")])
    client = make_fake_llm_client(body)

    with pytest.raises(AnswerValidationError) as info:
        generate(QUERY, contexts, client)

    error = info.value
    assert error.code == code
    assert str(error) == code
    assert error.args == (code,)
    assert MARKER not in str(error)
    assert MARKER not in repr(error)
    # No raw pydantic error or reply body hangs off the exception chain.
    assert error.__cause__ is None
    assert error.__context__ is None
    formatted = "".join(traceback.format_exception(error))
    assert MARKER not in formatted
    assert "pydantic" not in formatted


def test_llm_errors_propagate_unchanged(contexts_for, make_fake_llm_client):
    for error in (LLMRateLimitError("limit", 429), LLMBadResponseError("bad")):
        client = make_fake_llm_client(error)
        with pytest.raises(type(error)) as info:
            generate(QUERY, contexts_for([cid("d1")]), client)
        assert info.value is error
        assert client.calls == 1


# --- claim normalisation --------------------------------------------------------------------


def test_claim_text_800_vs_801_characters_is_measured_after_nfc():
    nfc_ok = "ế" * 800
    nfd_ok = unicodedata.normalize("NFD", nfc_ok)
    assert len(nfd_ok) > 800

    answer = validate_answer(reply((nfd_ok, ["S1"])), ["S1"])
    assert answer.claims[0].text == nfc_ok

    nfd_long = unicodedata.normalize("NFD", "ế" * 801)
    with pytest.raises(AnswerValidationError) as info:
        validate_answer(reply((nfd_long, ["S1"])), ["S1"])
    assert info.value.code == "claim_too_long"


def test_claim_length_is_measured_after_whitespace_collapse():
    padded = "  " + "a" * 400 + "     " * 50 + "b" * 399 + "  "
    assert len(padded) > 800
    answer = validate_answer(reply((padded, ["S1"])), ["S1"])
    assert answer.claims[0].text == "a" * 400 + " " + "b" * 399
    assert len(answer.claims[0].text) == 800


@pytest.mark.parametrize(
    "separator",
    ["\n", "\r", "\r\n", "\t", "\u2028", "\u2029", "\u0085", "\x0b", "\x0c", "\x1c", "\x1d"],
    ids=["LF", "CR", "CRLF", "TAB", "U+2028", "U+2029", "NEL", "VT", "FF", "FS", "GS"],
)
def test_line_breaks_in_a_claim_cannot_create_a_second_line_or_source_heading(
    separator, contexts_for, make_fake_llm_client
):
    contexts = contexts_for([cid("d1"), cid("d2:k1")])
    fake_heading = "[S2] Nhãn giả lập — tiêu đề giả lập (id-giả, điểm 1.0000)"
    claim_text = f"alpha{separator}{fake_heading}{separator}beta"
    result, _ = run(contexts, reply((claim_text, ["S1"])), make_fake_llm_client)
    output = render_result(result)

    assert result.claims[0].text == f"alpha {fake_heading} beta"
    claim_lines = [line for line in output.splitlines() if line.startswith("1. ")]
    assert claim_lines == [f"1. alpha {fake_heading} beta [S1]"]
    headings = [line for line in output.splitlines() if line.startswith("[S")]
    assert headings == [header_of(contexts[0], "S1")]
    assert [s.source_id for s in result.sources] == ["S1"]


@pytest.mark.parametrize(
    "control", ["\x1b", "\x00", "\x7f", "\x9b"], ids=["ESC", "NUL", "DEL", "CSI"]
)
def test_residual_control_characters_in_a_claim_give_schema(control):
    body = reply((f"alpha{control}beta", ["S1"]))
    with pytest.raises(AnswerValidationError) as info:
        validate_answer(body, ["S1"])
    assert info.value.code == "schema"
    assert info.value.__context__ is None


@pytest.mark.parametrize(
    "blank", ["", " ", "\n\t\r", "\u2028", "\u00a0", "\u3000", "\u2003 \n"], ids=repr
)
def test_whitespace_only_claim_gives_empty_claim(blank):
    with pytest.raises(AnswerValidationError) as info:
        validate_answer(reply((blank, ["S1"])), ["S1"])
    assert info.value.code == "empty_claim"


def test_claims_are_nfc_before_matching_and_rendering(contexts_for, make_fake_llm_client):
    nfd = unicodedata.normalize("NFD", "Điều 36")
    assert nfd != unicodedata.normalize("NFC", nfd)
    contexts = contexts_for([cid("d39")])
    answer = validate_answer(reply((nfd, ["S1"])), ["S1"])
    result, _ = run(contexts, reply((nfd, ["S1"])), make_fake_llm_client)

    assert answer.claims[0].text == unicodedata.normalize("NFC", nfd)
    assert result.claims[0].text == unicodedata.normalize("NFC", nfd)
    assert result.unsupported_articles == []
    assert render_result(result).split("\n")[0] == "1. Điều 36 [S1]"


def test_validate_answer_returns_normalised_claims_and_deduplicated_sources():
    answer = validate_answer(reply(("  một\n\nhai  ", ["S2", "S1", "S2"])), ["S1", "S2"])

    assert answer.status == "answered"
    assert answer.claims == [Claim(text="một hai", sources=["S2", "S1"])]


# --- article mentions -----------------------------------------------------------------------


def chunk_by_id(all_chunks, suffix):
    return next(c for c in all_chunks if c.id == cid(suffix))


def test_article_mentions_lists_case_and_non_numeric_text(all_chunks):
    d39 = chunk_by_id(all_chunks, "d39")
    d196 = chunk_by_id(all_chunks, "d196:k3")

    assert article_mentions(d39.context_text) == {35, 36, 37, 39}
    assert {200, 201, 202} <= article_mentions(d196.context_text)
    assert article_mentions("Điều 5") == article_mentions("ĐIỀU 5") == {5}
    assert article_mentions("điều 5 và điều 6") == {5, 6}
    assert article_mentions("điều 5, 6 và 7 hoặc 8") == {5, 6, 7, 8}
    assert article_mentions("điều kiện lao động; một điều khoản") == set()
    assert article_mentions("Điều") == set()
    assert article_mentions(unicodedata.normalize("NFD", "Điều 36")) == {36}


UNSUPPORTED_CASES = [
    pytest.param(["d34:k1"], ["Căn cứ khoản 4 Điều 177"], [], id="cross-reference-in-context"),
    pytest.param(["d34:k1"], ["Điều 34"], [], id="own-article"),
    pytest.param(["d34:k1"], ["Điều 1"], [1], id="foreign-article"),
    pytest.param(["d34:k1"], ["Điều 9", "Điều 2"], [2, 9], id="deduplicated-sorted"),
    pytest.param(["d34:k1"], ["Điều 9", "điều 9", "Điều 9, 2"], [2, 9], id="repeated-mentions"),
    pytest.param(["d39"], ["Điều 36"], [], id="listed-in-context-text"),
    pytest.param(["d34:k1"], ["điều 200"], [200], id="lowercase-foreign"),
    pytest.param(["d34:k1"], ["điều kiện nêu trên"], [], id="not-an-article-mention"),
    pytest.param(["d34:k1"], ["Điều 2 và 177 hoặc 9"], [2, 9], id="list-forms"),
    pytest.param(["d1", "d2:k1", "d9:k1"], ["Điều 9", "Điều 2", "Điều 1"], [], id="all-contexts"),
    pytest.param(["d34:k1"], ["Khẳng định không nhắc số"], [], id="no-mention"),
]


@pytest.mark.parametrize(("context_ids", "texts", "expected"), UNSUPPORTED_CASES)
def test_unsupported_article_mentions(context_ids, texts, expected, contexts_for):
    contexts = contexts_for([cid(i) for i in context_ids])
    claims = [Claim(text=text, sources=["S1"]) for text in texts]

    assert unsupported_article_mentions(claims, contexts) == expected


def test_unsupported_articles_flow_into_the_result(contexts_for, make_fake_llm_client):
    contexts = contexts_for([cid("d34:k1")])
    result, _ = run(
        contexts, reply(("Điều 9 và Điều 2", ["S1"]), ("Điều 34", ["S1"])), make_fake_llm_client
    )

    assert result.unsupported_articles == [2, 9]


def test_render_prints_one_warning_line_per_unsupported_article(contexts_for, make_fake_llm_client):
    contexts = contexts_for([cid("d34:k1")])
    result, _ = run(contexts, reply(("Điều 9 và Điều 2", ["S1"])), make_fake_llm_client)
    lines = render_result(result).split("\n")

    warnings = [line for line in lines if line.startswith("Cảnh báo:")]
    assert warnings == [
        "Cảnh báo: câu trả lời nhắc Điều 2 không có trong các nguồn đã truy xuất; hãy đối chiếu.",
        "Cảnh báo: câu trả lời nhắc Điều 9 không có trong các nguồn đã truy xuất; hãy đối chiếu.",
    ]
    assert lines[-2:] == [VERIFICATION_NOTICE, DISCLAIMER]
    assert lines.index(warnings[1]) < lines.index(VERIFICATION_NOTICE)


def test_no_warning_line_when_every_mention_is_supported(contexts_for, make_fake_llm_client):
    result, _ = run(
        contexts_for([cid("d34:k1")]), reply(("Điều 177", ["S1"])), make_fake_llm_client
    )

    assert "Cảnh báo" not in render_result(result)


# --- fixed texts ----------------------------------------------------------------------------


def test_render_insufficient_is_the_fixed_text_plus_disclaimer(make_fake_llm_client):
    result = generate(QUERY, [], make_fake_llm_client("unused"))

    assert render_result(result) == INSUFFICIENT_TEXT + "\n\n" + DISCLAIMER


def test_notice_does_not_claim_verification(contexts_for, make_fake_llm_client):
    answered, _ = run(
        contexts_for([cid("d1")]), reply(("Khẳng định giả lập.", ["S1"])), make_fake_llm_client
    )
    insufficient = generate(QUERY, [], make_fake_llm_client("unused"))

    for result in (answered, insufficient):
        assert "đã xác minh" not in render_result(result).lower()
    assert VERIFICATION_NOTICE in render_result(answered)
    assert "đã xác minh" not in (VERIFICATION_NOTICE + DISCLAIMER + INSUFFICIENT_TEXT).lower()


def test_generate_module_has_no_sdk_and_does_not_call_retrieve():
    tree = ast.parse(Path(generate_module.__file__).read_text(encoding="utf-8"))
    modules, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(alias.name for alias in node.names)

    assert not any(m.split(".")[0] in {"google", "genai", "openai"} for m in modules)
    assert "retrieve" not in names
    assert "app.rag.store" not in modules
