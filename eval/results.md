# Kết quả đánh giá

Ghi lại số liệu sau mỗi thay đổi (recall@1, recall@5, recall@10, MRR), kể cả thử nghiệm không hiệu quả.

Baseline dense đo ngày 22/11/2026 trên bộ câu `577b8392afeb` (127 câu, xem mục "Đánh giá truy xuất" trong `README.md`). Bốn cột số là mức Điều trên 90 câu `direct` + `paraphrase`; mức Khoản ghi ở cột Ghi chú. Cột recall@1 và R@1 theo nhóm của dòng baseline được tính lại ngày 26/11/2026 từ `eval/runs/2026-11-22-dense-baseline.jsonl` qua `results_row`, không chạy lại truy xuất. recall@5 mức Điều đã chạm trần 1,000, nên các thay đổi sau cần so thêm MRR@10 và hạng 1 (xem README). Sau duyệt nghiệm thu, Điều 129 là đáp án của hai cặp (p25 và p31, vì `d129:k1` được thêm vào p31 cho đủ đáp án) nên quy tắc "mỗi Điều một cặp" của spec §5.2 không còn đúng cho bộ đã duyệt; đáp án p38 cũng gồm thêm `d193:k3` và `d197:k3`.

| Ngày | Thay đổi | recall@1 | recall@5 | recall@10 | MRR | Ghi chú |
|---|---|---|---|---|---|---|
| 2026-11-15 | Task 2: interface embedding và bge-m3 | N/A | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Task 3: cache embedding theo model_key | N/A | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Task 4: nạp chunk vào Qdrant | N/A | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Task 5: retrieve dense và lệnh search | N/A | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Task 6: chạy thật trên Qdrant (676 điểm, kiểm tra bằng mắt 7 câu, xem README) | N/A | N/A | N/A | N/A | chưa đo recall; không phải số liệu recall |
| 2026-11-15 | F1: báo lỗi CLI cho HTTP 4xx, payload lệch schema và collection rỗng (không đổi xếp hạng) | N/A | N/A | N/A | N/A | chưa đo recall; run_eval.py còn là placeholder, không in số liệu |
| 2026-11-15 | F2: nạp model chỉ từ snapshot cục bộ, ẩn tiến trình nạp trọng số và tách lệnh tải model | N/A | N/A | N/A | N/A | chưa đo recall; run_eval.py trước/sau là placeholder; smoke 2 lần giữ nguyên top 2 và điểm, collection thật vẫn 676 điểm |
| 2026-11-15 | F2 vòng sửa 1: từ chối cache thiếu từ vựng tokenizer ở index/search/download, tắt safetensors conversion khi tải, thông báo lỗi tải trung thực hơn (không đổi xếp hạng) | N/A | N/A | N/A | N/A | chưa đo recall; run_eval.py trước/sau là placeholder, không in số liệu |
| 2026-11-15 | Stage 2 Task 2: prompt và schema trích dẫn | N/A | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Stage 2 Task 3: generate, kiểm trích dẫn | N/A | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Stage 2 Task 4: lệnh ask | N/A | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Stage 2 Task 5: smoke LLM có cổng và README | N/A | N/A | N/A | N/A | chưa đo recall; smoke thật chưa chạy |
| 2026-11-15 | Stage 2: smoke LLM thật (`RAG_LIVE_LLM=1`, 2 test, 2 request, pass; requested_model và response_model đều là `gemini-3.5-flash-lite`) và 4 lần chạy `ask` bằng tay: 2 câu trong phạm vi trả lời có trích dẫn, 1 câu ngoài phạm vi bị từ chối, 1 lần câu "Thử việc tối đa bao lâu?" ra văn bản lặp vô nghĩa nhưng vẫn qua kiểm tra (chạy lại thì bình thường) | N/A | N/A | N/A | N/A | chưa đo recall; không phải số liệu recall; lỗi văn bản lặp chưa xử lý |
| 2026-11-15 | Stage 2 Task 6: từ chối khẳng định lặp vô nghĩa (degenerate_claim) | N/A | N/A | N/A | N/A | chưa đo recall |
| 2026-11-15 | Stage 2 Task 6, chạy thật sau khi sửa: 10 lần gửi lại cùng request của câu "Thử việc tối đa bao lâu?" (0 lần ra văn bản lặp) và 4 câu hỏi chạy tay qua `ask` (thử việc, thai sản, làm thêm giờ, tuổi nghỉ hưu) đều trả lời có trích dẫn, mã thoát 0; tổng 21 request trong ngày | N/A | N/A | N/A | N/A | chưa đo recall; kiểm tra bằng mắt, chưa đối chiếu từng khẳng định với nguyên văn |
| 2026-11-22 | dense baseline (bge-m3) | 0,844 | 1,000 | 1,000 | 0,907 | Mức Khoản: 0,778 / 0,989 / 1,000 / 0,864; R@1 theo nhóm (mức Điều): direct 44/45, paraphrase 32/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi) |
| 2026-11-26 | S0: thêm recall@1 và R@1 theo nhóm vào dòng results.md, tính lại cho baseline từ file run (không đổi xếp hạng) | N/A | N/A | N/A | N/A | không chạy lại truy xuất |
| 2026-11-26 | T3: retriever dense/sparse/hybrid và RRF (mặc định vẫn dense) | 0,844 | 1,000 | 1,000 | 0,907 | Mức Khoản: 0,778 / 0,989 / 1,000 / 0,864; R@1 theo nhóm (mức Điều): direct 44/45, paraphrase 32/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi); chưa chạy hybrid (pre-registration) |
| 2026-11-26 | T4: parser số Điều và promotion (mặc định tắt) | 0,844 | 1,000 | 1,000 | 0,907 | Mức Khoản: 0,778 / 0,989 / 1,000 / 0,864; R@1 theo nhóm (mức Điều): direct 44/45, paraphrase 32/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi); chưa chạy hybrid (pre-registration) |
