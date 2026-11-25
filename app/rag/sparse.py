"""BM25 sparse vectors over Vietnamese syllables and syllable bigrams.

This is the tokenizer of the BM25 side only. Text shown to users or passed to the
article-reference parser is never folded. TOKENIZER_VERSION must change with any rule change,
because indexed vectors depend on these rules.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

TOKENIZER_VERSION = "vi-syl-bigram-v1"  # bump whenever any rule below changes
K1 = 1.2
B = 0.75

# Everything but word characters, space, tab and NBSP ends a segment, line breaks included.
_SEGMENT_BREAK = re.compile(r"[^\w \t\u00a0]+")
_TOKEN = re.compile(r"\w+")

# Open syllables only: the old spelling (left) moves its tone to the first vowel (right).
_TONE_FOLDS = {
    "oà": "òa",
    "oá": "óa",
    "oả": "ỏa",
    "oã": "õa",
    "oạ": "ọa",
    "oè": "òe",
    "oé": "óe",
    "oẻ": "ỏe",
    "oẽ": "õe",
    "oẹ": "ọe",
    "uỳ": "ùy",
    "uý": "úy",
    "uỷ": "ủy",
    "uỹ": "ũy",
    "uỵ": "ụy",
}


def _whole_word(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w)(?:{pattern})(?!\w)")


# Applied in this order: every phrase, longest first, before any single word. "bao", "thế" and
# "sao" are not listed on their own, because the law uses "bao gồm", "thay thế" and "bản sao".
_INTERROGATIVES = tuple(
    _whole_word(r"\s+".join(phrase.split()))
    for phrase in (
        "như thế nào",
        "bao nhiêu",
        "thế nào",
        "bao lâu",
        "bao giờ",
        "tại sao",
        "vì sao",
        "gì",
        "nào",
        "ai",
        "đâu",
        "mấy",
    )
)
# Removed only from the end of a question: "không" and "chưa" carry meaning anywhere else.
_TRAILING_PARTICLE = re.compile(r"(?<!\w)(?:không|chưa|à|ư|nhỉ|nhé|hả|ạ|vậy)\Z")


@dataclass(frozen=True)
class SparseVector:
    indices: tuple[int, ...]  # strictly increasing, unique
    values: tuple[float, ...]


def normalize(text: str) -> str:
    """NFC, lowercase, and the Icelandic eth (U+00F0) replaced by the Vietnamese "đ"."""
    return unicodedata.normalize("NFC", text).lower().replace("\u00f0", "\u0111")


def fold_tone(token: str) -> str:
    """Spell one lowercase NFC syllable the way the corpus does: "hoà" becomes "hòa"."""
    stem, ending = token[:-2], token[-2:]
    # In "quý" the "u" belongs to the initial consonant, so the tone is already in place.
    if ending not in _TONE_FOLDS or (ending.startswith("u") and stem.endswith("q")):
        return token
    return stem + _TONE_FOLDS[ending]


def _term_token(token: str) -> str:
    if token.isascii() and token.isdigit():
        return token.lstrip("0") or "0"
    return fold_tone(token)


def document_terms(text: str) -> list[str]:
    """Unigrams and the bigrams of adjacent tokens within each segment, repeats kept."""
    terms: list[str] = []
    for segment in _SEGMENT_BREAK.split(normalize(text)):
        tokens = [_term_token(token) for token in _TOKEN.findall(segment)]
        terms.extend(tokens)
        terms.extend(f"{first} {second}" for first, second in pairwise(tokens))
    return terms


def _drop_trailing_non_word(text: str) -> str:
    end = len(text)
    while end and not _TOKEN.match(text, end - 1, end):
        end -= 1
    return text[:end]


def clean_query(text: str) -> str:
    """Normalise a question and remove its interrogatives and final particles.

    A removed interrogative leaves "|", a segment break, so no bigram joins its neighbours.
    """
    text = normalize(text)
    for interrogative in _INTERROGATIVES:
        text = interrogative.sub(" | ", text)
    while True:
        text = _drop_trailing_non_word(text)
        particle = _TRAILING_PARTICLE.search(text)
        if particle is None:
            return text
        text = text[: particle.start()]


def query_terms(text: str) -> list[str]:
    """The distinct terms of the cleaned question, in order of first occurrence."""
    return list(dict.fromkeys(document_terms(clean_query(text))))


def term_id(term: str) -> int:
    """A stable 32-bit id; the built-in hash() differs between processes."""
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")


def average_length(texts: Iterable[str]) -> float:
    """Mean number of terms per document; ValueError when there is no document."""
    lengths = [len(document_terms(text)) for text in texts]
    if not lengths:
        raise ValueError("cannot average the length of no documents")
    return sum(lengths) / len(lengths)


def document_vector(text: str, avgdl: float, k1: float = K1, b: float = B) -> SparseVector:
    """BM25 term-frequency weights of one document, without IDF.

    Terms whose ids collide share one entry that holds the sum of their weights.
    """
    if avgdl <= 0:
        raise ValueError("avgdl must be positive")
    terms = document_terms(text)
    length = len(terms)
    weights: dict[int, float] = {}
    for term, tf in Counter(terms).items():
        weight = tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))
        index = term_id(term)
        weights[index] = weights.get(index, 0.0) + weight
    indices = tuple(sorted(weights))
    return SparseVector(indices, tuple(weights[index] for index in indices))


def query_vector(text: str) -> SparseVector:
    """Weight 1.0 for every distinct term id; empty when the question cleans to nothing."""
    indices = tuple(sorted({term_id(term) for term in query_terms(text)}))
    return SparseVector(indices, (1.0,) * len(indices))
