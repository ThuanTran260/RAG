# Trợ lý tra cứu pháp luật lao động (RAG)

Hỏi đáp tiếng Việt về Bộ luật Lao động 2019 (45/2019/QH14), trả lời kèm trích dẫn Điều/Khoản gốc.

> Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.

**Trạng thái:** xong Tuần 1 (dữ liệu và chunking). Kế hoạch chi tiết xem tại [KE_HOACH_RAG.md](KE_HOACH_RAG.md).

## Cách chạy

```bash
uv sync
uv run pytest
uv run python -m ingestion.build --sample 20
```

Lệnh build ghi `data/processed/articles.jsonl` và `data/processed/chunks.jsonl`, in thống kê và 20 chunk mẫu để kiểm tra bằng mắt.

## Kết quả

- 220 Điều, 676 chunk.
- Độ dài `text` của chunk (nhỏ nhất / trung vị / lớn nhất): 11 / 189,5 / 4.512 ký tự.

## Dữ liệu

Bản hợp nhất 18/VBHN-VPQH ngày 12/02/2026 của Bộ luật Lao động 45/2019/QH14, kiểm tra hiệu lực ngày 13/11/2026. Nguồn gốc và checksum xem ở `data/raw/SOURCES.md`.

## Hạn chế đã biết

1. Chưa nạp Nghị định 145/2020/NĐ-CP.
2. Khoản quá ngắn như các khoản của Điều 124 thiếu nghĩa khi đứng riêng; sẽ so sánh chunk theo Khoản với theo Điều ở Tuần 5.
3. Khoản 1 và 2 Điều 219 chứa nguyên văn sửa đổi của luật khác, có thể gây nhiễu khi truy xuất.
4. Chưa trích dẫn chiếu giữa các Điều và chưa lưu thông tin sửa đổi vào metadata.

