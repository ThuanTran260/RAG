"""Find the articles of this Labour Code that a text mentions (Week 4 spec §7.4).

A wrong trigger would push chunks of an unrelated article up, while a missed one only leaves the
ranking as it was. So a mention counts only when nothing says it refers to something else.
"""

import re
import unicodedata

from app.rag.generate import ARTICLE_MENTION, article_mentions
from ingestion.documents import LABOUR_CODE

# A closed list: a number followed by one of these words counts something ("điều 1 chút").
QUANTITY_WORDS = ("chút", "tí", "tý", "tị", "xíu", "tẹo", "lát")
_QUANTITY = re.compile(rf"\s+(?:{'|'.join(QUANTITY_WORDS)})(?!\w)", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")

# "khoản 1", "điểm a, b và c", "các khoản 2 và 3"
_ITEM = r"(?:\d+[a-zđ]?|[a-zđ])(?!\w)"
_SUBREF = rf"(?:các\s+)?(?:điểm|khoản)\s+{_ITEM}(?:\s*(?:,|và|hoặc)\s*{_ITEM})*"

# All that may stand between two mentions that share one document name, as in
# "Điều 42 và Điều 43 của Bộ luật này".
_JOINER = re.compile(
    rf"\s*[,;]?\s*(?:(?:và|hoặc|hay)(?!\w))?"
    rf"(?:\s*{_SUBREF}(?:\s*(?:,|và|hoặc))?)*"
    r"\s*(?:các\s+)?",
    re.IGNORECASE,
)
# All that may stand between a number and the name of its document: a name any further away does
# not belong to the number.
_BEFORE_NAME = re.compile(
    rf"(?:\s*,?\s*{_SUBREF})*"
    r"(?:\s*[,:(]|\s+(?:của|trong|tại|thuộc|theo|ở|bên|được|quy\s+định)(?!\w))*"
    r"\s*",
    re.IGNORECASE,
)
# Names of this Code. Beyond "Bộ luật này" and "Bộ luật Lao động", people also write
# "Luật Lao động", "BLLĐ", the law number, or a bare "luật". A year or number of 2012 is the
# repealed Code.
_THIS_CODE = re.compile(
    r"bộ\s+luật\s+này(?!\w)"
    r"|(?:(?:bộ\s+)?luật\s+lao\s+động|bllđ)(?!\w)"
    r"(?!\s*\(?\s*(?:(?:số|năm)\s*)?(?:10\s*/\s*)?2012)"
    r"|(?:bộ\s+)?luật\s+(?:số\s*)?45\s*/\s*2019\s*/\s*qh14"
    # A bare "luật" names no other document: it ends the clause or a verb follows it.
    r"|(?:bộ\s+)?luật(?=\s*(?:$|[^\w\s]))"
    r"|(?:bộ\s+)?luật\s+(?:quy\s+định|nói|ghi|hiện\s+hành)(?!\w)",
    re.IGNORECASE,
)
# Any legal document. After "luật" or "bộ luật", every wording that _THIS_CODE does not list
# is another law. "quyết định" and "chỉ thị" are common words, so they need a number.
_LEGAL_DOCUMENT = re.compile(
    r"(?:(?:bộ\s+)?luật|bllđ|blds|blhs|blttds|bltths"
    r"|nghị\s+định|nghị\s+quyết|thông\s+tư|pháp\s+lệnh|hiến\s+pháp|công\s+ước)(?!\w)"
    r"|(?:quyết\s+định|chỉ\s+thị)\s+(?:số(?!\w)|\d)"
    r"|(?:nđ|tt)[\s.\-]*\d",
    re.IGNORECASE,
)
# A contract or a company's own rules. Only "của" or "trong" ties them to the number: without
# one, "Điều 21 hợp đồng lao động gồm những gì" asks about the topic of the article.
_PRIVATE_DOCUMENT = re.compile(
    rf"(?:\s*,?\s*{_SUBREF})*\s+(?:của|trong)\s+(?:(?:cái|bản)\s+)?"
    r"(?:hợp\s+đồng|nội\s+quy|quy\s+chế|thỏa\s+ước|thoả\s+ước|điều\s+lệ)(?!\w)",
    re.IGNORECASE,
)


def article_refs(text: str) -> set[int]:
    """The articles of this Code that ``text`` names, from 1 to the last article of the Code."""
    # One space per run of whitespace: the patterns below would retry every split of a long run.
    text = " ".join(unicodedata.normalize("NFC", text).split())
    matches = list(ARTICLE_MENTION.finditer(text))
    found: set[int] = set()
    chain: set[int] = set()  # numbers of consecutive mentions that share one document name
    for match, following in zip(matches, [*matches[1:], None]):
        chain |= _numbers(text, match)
        if following and _JOINER.fullmatch(text, match.end(), following.start()):
            continue  # the name, if any, follows the last mention of the chain
        if not _names_another_document(text, match.end()):
            found |= chain
        chain = set()
    return found


def _numbers(text: str, match: re.Match[str]) -> set[int]:
    """The article numbers of one mention, without those that cannot be an article of the Code."""
    numbers = article_mentions(match.group(0))
    for digits in _NUMBER.finditer(text, match.start(1), match.end(1)):
        end = digits.end()
        # "5a" is not article 5, and "1 chút" is a quantity.
        if text[end : end + 1].isalpha() or _QUANTITY.match(text, end):
            numbers.discard(int(digits.group()))
    return {number for number in numbers if 1 <= number <= LABOUR_CODE.expected_articles}


def _names_another_document(text: str, position: int) -> bool:
    """Whether the words at ``position``, right after a mention, tie it to another document."""
    if _PRIVATE_DOCUMENT.match(text, position):
        return True
    start = _BEFORE_NAME.match(text, position).end()
    return bool(_LEGAL_DOCUMENT.match(text, start)) and not _THIS_CODE.match(text, start)
