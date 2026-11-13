import re
import unicodedata
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError


def load_paragraphs(path: Path) -> list[str]:
    """Read a .docx into a list of cleaned, non-empty body paragraphs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        doc = Document(path)
    except (PackageNotFoundError, BadZipFile) as exc:
        # python-docx raises BadZipFile when the bytes are not a zip container at all.
        raise ValueError(f"{path} is not a .docx file") from exc

    paragraphs = []
    # Body-level paragraphs only: table content (e.g. the signature block) is skipped.
    for paragraph in doc.paragraphs:
        # Superscript runs are footnote numbers glued after clause/point labels.
        text = "".join(run.text for run in paragraph.runs if not run.font.superscript)
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()
        if text:
            paragraphs.append(text)
    return paragraphs
