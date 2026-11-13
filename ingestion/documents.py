from datetime import date
from pathlib import Path

from ingestion.models import DocumentMeta

REPO_ROOT = Path(__file__).resolve().parents[1]

LABOUR_CODE = DocumentMeta(
    doc_id="45-2019-QH14",
    doc_no="45/2019/QH14",
    title="Bộ luật Lao động",
    source_no="18/VBHN-VPQH",
    source_date=date(2026, 2, 12),
    source_url="https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-18-vbhn-vpqh-468971.htm",
    path=REPO_ROOT / "data" / "raw" / "2026_131_18_VBHN-VPQH.docx",
    sha256="1386441b1f513defdd55186d7e65b8432dcac87c2e0d78676de25facb9c5e6ff",
    checked_at=date(2026, 11, 13),
    expected_articles=220,
)
