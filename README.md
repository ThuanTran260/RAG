# Trợ lý tra cứu pháp luật lao động (RAG)

Hỏi đáp tiếng Việt về Bộ luật Lao động 2019 (45/2019/QH14), trả lời kèm trích dẫn Điều/Khoản gốc.

> Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.

**Trạng thái:** xong Tuần 1 (dữ liệu và chunking) và nửa đầu Tuần 2 (Qdrant và truy xuất dense), chưa có phần sinh câu trả lời. Kế hoạch chi tiết xem tại [KE_HOACH_RAG.md](KE_HOACH_RAG.md).

## Cách chạy

```bash
uv sync --locked
uv run python -m ingestion.download_model
docker compose up -d qdrant
uv run python -m ingestion.index
uv run python -m app.rag.search "Thử việc tối đa bao lâu?"
uv run pytest
uv run pytest -m model
uv run pytest -m server
```

- `uv run python -m ingestion.build --sample 20` ghi `data/processed/articles.jsonl` và `data/processed/chunks.jsonl`, in thống kê và 20 chunk mẫu để kiểm tra bằng mắt.
- `uv run python -m ingestion.download_model` là bước tải model một lần trước khi index: dùng `EMBEDDING_MODEL` (mặc định `BAAI/bge-m3`), cần internet, không cần Qdrant và không nhúng hay ghi vector. Lệnh này khởi tạo model trên CPU để tải các file cần thiết, tắt việc transformers tự tải thêm bản safetensors chạy nền (chỉ trong tiến trình lệnh), rồi kiểm tra tokenizer có từ vựng; nếu mạng đứt giữa chừng và cache thiếu file từ vựng, lệnh báo lỗi và thoát mã 1 thay vì báo thành công. Có thể hiện tiến trình và mất vài giây khởi tạo, cộng thêm thời gian tải tùy đường truyền.
- `index` và `search` chỉ nạp snapshot model đã có trên máy, không liên hệ Hugging Face Hub và không hiện thanh tiến trình nạp trọng số; chúng vẫn cần kết nối Qdrant và vẫn mất thời gian khởi tạo CPU/GPU. Có thể đặt `EMBEDDING_MODEL` thành đường dẫn tới thư mục model cục bộ. Nếu cache thiếu trọng số hoặc tokenizer không có từ vựng (chỉ còn token đặc biệt, thường do lần tải bị ngắt), lệnh báo lỗi và hướng dẫn chạy `uv run python -m ingestion.download_model`, không tự tải ngầm. Kiểm tra từ vựng chỉ xác nhận tokenizer có token ngoài token đặc biệt; nó không phát hiện file từ vựng sai hay hỏng.
- `uv run python -m ingestion.index` nhúng 676 chunk bằng `BAAI/bge-m3` và nạp vào Qdrant; lần chạy thật in `points: 676` và `dimension: 1024`. Chạy lại lệnh này cho cùng kết quả (embedding được cache ở `.cache/embeddings/<model_key>.npz`).
- `uv run python -m app.rag.search "<câu hỏi>"` in 5 chunk gần nhất kèm điểm cosine.
- Với các lỗi vận hành được xử lý, `index` và `search` in lý do lên stderr, trả mã thoát 1 và không in gì ra stdout; lỗi cấu hình hoặc lần index bị ngắt có thể có thêm dòng chi tiết. Các lỗi gặp thường: Qdrant chưa chạy, collection chưa có hoặc đang rỗng (chạy `uv run python -m ingestion.index`), dữ liệu trong collection lệch schema chunk hiện tại (index lại), Qdrant trả HTTP 4xx (kiểm tra `QDRANT_URL`, quyền truy cập, `QDRANT_COLLECTION` và `EMBEDDING_MODEL`). Lỗi lập trình không bị che và vẫn có traceback.
- `uv run pytest` (mặc định) bỏ qua test `integration`. `uv run pytest -m model` cần model, không cần Docker. `uv run pytest -m server` cần Qdrant đang chạy, không cần model.
- Dữ liệu Qdrant nằm trong named volume `qdrant_storage`: sau `docker compose down` (không `-v`) rồi `docker compose up -d qdrant`, collection vẫn còn 676 điểm. Không dùng `docker compose down -v` nếu muốn giữ dữ liệu.

Muốn chủ động cập nhật model, chạy lại `uv run python -m ingestion.download_model`: lệnh liên hệ Hub và có thể cập nhật snapshot cache sang trọng số mới dưới cùng tên model. Sau khi đổi trọng số hoặc reset cache model rồi tải lại, xóa file vector `.cache/embeddings/<model_key>.npz` tương ứng và chạy lại `uv run python -m ingestion.index` trước khi tìm kiếm. Hệ thống chưa tự phát hiện thay đổi phiên bản trọng số; tên collection và cache vector vẫn dựa trên tên model (xem hạn chế 8).

## Kết quả

- 220 Điều, 676 chunk.
- Độ dài `text` của chunk (nhỏ nhất / trung vị / lớn nhất): 11 / 189,5 / 4.512 ký tự.

## Truy xuất thử

Kết quả của `uv run python -m app.rag.search` trên Qdrant thật (collection `labor_law_vn__baai-bge-m3--d790e737453c`, 676 điểm, cosine). Đây là kiểm tra bằng mắt trên 7 câu chọn tay, chưa đo recall.

| Câu hỏi | Kết quả đứng đầu | Điểm | Điều mong đợi trong top 5 |
|---|---|---|---|
| Thử việc tối đa bao lâu? | `45-2019-QH14:d25:k1` | 0,7258 | Điều 25, hạng 1 |
| Người lao động muốn nghỉ việc phải báo trước bao nhiêu ngày? | `45-2019-QH14:d36:k2` | 0,7071 | Điều 35, hạng 3 |
| Nghỉ thai sản được mấy tháng? | `45-2019-QH14:d139:k1` | 0,7405 | Điều 139, hạng 1 |
| Một năm được nghỉ phép bao nhiêu ngày? | `45-2019-QH14:d113:k1` | 0,7166 | Điều 113, hạng 1 |
| Làm thêm giờ được trả lương như thế nào? | `45-2019-QH14:d98:k1` | 0,7740 | Điều 98, hạng 1 |
| Tuổi nghỉ hưu là bao nhiêu? | `45-2019-QH14:d169:k2` | 0,7171 | Điều 169, hạng 1 |
| Cách nấu phở bò ngon (ngoài phạm vi) | `45-2019-QH14:d93:k1` | 0,3439 | không có |

## Dữ liệu

Bản hợp nhất 18/VBHN-VPQH ngày 12/02/2026 của Bộ luật Lao động 45/2019/QH14, kiểm tra hiệu lực ngày 13/11/2026. Nguồn gốc và checksum xem ở `data/raw/SOURCES.md`.

## Hạn chế đã biết

1. Chưa nạp Nghị định 145/2020/NĐ-CP.
2. Khoản quá ngắn như các khoản của Điều 124 thiếu nghĩa khi đứng riêng; sẽ so sánh chunk theo Khoản với theo Điều ở Tuần 5.
3. Khoản 1 và 2 Điều 219 chứa nguyên văn sửa đổi của luật khác, có thể gây nhiễu khi truy xuất.
4. Chưa trích dẫn chiếu giữa các Điều và chưa lưu thông tin sửa đổi vào metadata.
5. Truy xuất mới chỉ dùng dense vector. Câu hỏi về báo trước khi nghỉ việc xếp Điều 36 (quyền của người sử dụng lao động, điểm 0,7071) trên Điều 35 (quyền của người lao động, hạng 3, điểm 0,6800).
6. Chưa có bước sinh câu trả lời và chưa có ngưỡng từ chối. Câu hỏi ngoài phạm vi vẫn trả về 5 kết quả, với điểm thấp hơn (0,3439 so với 0,7071–0,7740 của 6 câu trong phạm vi). Một câu ngoài phạm vi chưa đủ để đặt ngưỡng.
7. Máy không có GPU NVIDIA vẫn chạy được trên CPU (`EMBEDDING_DEVICE=cpu`) nhưng chậm hơn; `pyproject.toml` đang lấy PyTorch từ index `cu126`.
8. Cache embedding chỉ đi theo tên model: đổi trọng số, revision hay cách mã hóa dưới cùng tên model thì phải xóa `.cache/embeddings/<model_key>.npz` và index lại. Chỉ chạy một tiến trình index tại một thời điểm; lần index bị ngắt giữa chừng có thể để collection thiếu điểm, chạy lại lệnh index là đủ.
9. Cổng Qdrant chỉ mở trên `127.0.0.1`; dữ liệu nằm trong named volume `qdrant_storage`.
