# Kết quả đánh giá

Ghi lại số liệu sau mỗi thay đổi (recall@5, recall@10, MRR), kể cả thử nghiệm không hiệu quả.

_Chưa có số liệu. Baseline sẽ được đo ở Tuần 3._

| Ngày | Thay đổi | recall@5 | recall@10 | MRR | Ghi chú |
|---|---|---|---|---|---|
| 2026-11-15 | Task 2: interface embedding và bge-m3 | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Task 3: cache embedding theo model_key | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Task 4: nạp chunk vào Qdrant | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Task 5: retrieve dense và lệnh search | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Task 6: chạy thật trên Qdrant (676 điểm, kiểm tra bằng mắt 7 câu, xem README) | N/A | N/A | N/A | chưa đo recall; không phải số liệu recall |
| 2026-11-15 | F1: báo lỗi CLI cho HTTP 4xx, payload lệch schema và collection rỗng (không đổi xếp hạng) | N/A | N/A | N/A | chưa đo recall; run_eval.py còn là placeholder, không in số liệu |
| 2026-11-15 | F2: nạp model chỉ từ snapshot cục bộ, ẩn tiến trình nạp trọng số và tách lệnh tải model | N/A | N/A | N/A | chưa đo recall; run_eval.py trước/sau là placeholder; smoke 2 lần giữ nguyên top 2 và điểm, collection thật vẫn 676 điểm |
| 2026-11-15 | F2 vòng sửa 1: từ chối cache thiếu từ vựng tokenizer ở index/search/download, tắt safetensors conversion khi tải, thông báo lỗi tải trung thực hơn (không đổi xếp hạng) | N/A | N/A | N/A | chưa đo recall; run_eval.py trước/sau là placeholder, không in số liệu |
| 2026-11-15 | Stage 2 Task 2: prompt và schema trích dẫn | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Stage 2 Task 3: generate, kiểm trích dẫn | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Stage 2 Task 4: lệnh ask | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Stage 2 Task 5: smoke LLM có cổng và README | N/A | N/A | N/A | chưa đo recall; smoke thật chưa chạy |
| 2026-11-15 | Stage 2: smoke LLM thật (`RAG_LIVE_LLM=1`, 2 test, 2 request, pass; requested_model và response_model đều là `gemini-3.5-flash-lite`) và 4 lần chạy `ask` bằng tay: 2 câu trong phạm vi trả lời có trích dẫn, 1 câu ngoài phạm vi bị từ chối, 1 lần câu "Thử việc tối đa bao lâu?" ra văn bản lặp vô nghĩa nhưng vẫn qua kiểm tra (chạy lại thì bình thường) | N/A | N/A | N/A | chưa đo recall; không phải số liệu recall; lỗi văn bản lặp chưa xử lý |
