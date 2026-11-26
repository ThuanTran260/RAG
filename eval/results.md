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
| 2026-11-27 | T6 (1): dense trên collection hybrid | 0,844 | 1,000 | 1,000 | 0,907 | Mức Khoản: 0,778 / 0,989 / 1,000 / 0,864; R@1 theo nhóm (mức Điều): direct 44/45, paraphrase 32/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi); article_ref R@1 9/12; xem mục Tuần 4 |
| 2026-11-27 | T6 (2): sparse BM25 | 0,611 | 0,744 | 0,811 | 0,668 | Mức Khoản: 0,556 / 0,711 / 0,778 / 0,625; R@1 theo nhóm (mức Điều): direct 42/45, paraphrase 13/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi); article_ref R@1 9/12; xem mục Tuần 4 |
| 2026-11-27 | T6 (3): hybrid RRF | 0,744 | 0,922 | 0,944 | 0,815 | Mức Khoản: 0,700 / 0,889 / 0,911 / 0,773; R@1 theo nhóm (mức Điều): direct 44/45, paraphrase 23/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi); article_ref R@1 11/12; xem mục Tuần 4 |
| 2026-11-27 | T6 (4): dense + promotion số Điều | 0,844 | 1,000 | 1,000 | 0,907 | Mức Khoản: 0,778 / 0,989 / 1,000 / 0,864; R@1 theo nhóm (mức Điều): direct 44/45, paraphrase 32/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi); article_ref R@1 12/12; xem mục Tuần 4 |
| 2026-11-27 | T6 (5): hybrid + promotion số Điều | 0,744 | 0,922 | 0,944 | 0,815 | Mức Khoản: 0,700 / 0,889 / 0,911 / 0,773; R@1 theo nhóm (mức Điều): direct 44/45, paraphrase 23/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi); article_ref R@1 12/12; xem mục Tuần 4 |
| 2026-11-27 | T7: search/ask dùng dense + promotion số Điều (mặc định mới) | 0,844 | 1,000 | 1,000 | 0,907 | Mức Khoản: 0,778 / 0,989 / 1,000 / 0,864; R@1 theo nhóm (mức Điều): direct 44/45, paraphrase 32/45; bộ câu 577b8392afeb; n=90 (+12 article_ref, 25 ngoài phạm vi); article_ref R@1 12/12; so với run (4) của T6: dung sai (a)–(d) đạt, 0 đổi chỗ |

## Tuần 4: hybrid và promotion số Điều (pre-registration)

Luật so sánh và luật chốt: mục 5 của `docs/superpowers/specs/2026-11-26-week4-hybrid-design.md`. Năm run chạy ngày 27/11/2026 trong một phiên, trên commit `8a25e98`; file run, file meta và báo cáo ở `eval/runs/2026-11-27-r*`. Trong mỗi cặp, A là run tham chiếu và B là run đang xét.

### 1. Tái lập: run (1) so với run 22/11/2026

- (a) `rank_article` và `rank_clause`: giống hệt ở 102/102 câu trong phạm vi; đạt.
- (b) top 10 của 127 câu: giống hệt 127, vi phạm 0; đạt.
- (c) |Δ điểm| < 1e-4: 1270 chunk chung, vi phạm 0; đạt.
- Đổi chỗ trong dung sai: không có.

### 2. Hybrid: run (3) so với run (1)

- R@1 mức Điều, 45 câu paraphrase: 32/45 → 23/45 (−9 câu; 0,711 → 0,511).
- McNemar: b = 3, c = 12; p chính xác = 0,035156; mid-p = 0,021271.
- C1: không (b < c). C1 cần mid-p ≤ 0,05 và b > c. Điều kiện thay thế (b − c ≥ 3 và c ≤ 1): không.
- Kiểm định hai phía có ý nghĩa theo chiều ngược: mid-p 0,021271 ≤ 0,05 với c > b.
- Guard cho câu direct: đạt. |D| = 44; mất hạng 1: 0/44 (không có câu nào); cận trên Clopper–Pearson hai phía 95 %: 0,0804; rời top 5: không có câu nào.
- **Quyết định (luật 5.4): giữ dense:** guard đạt nhưng C1 không đạt (b < c) và nhánh thay thế không đạt (b − c = −9).

### 3. Promotion số Điều

- So sánh dùng: run (4) với run (1), vì giữ dense.
- (d), 115 câu không phải article_ref: (b) top 10 giống hệt 115, vi phạm 0, đạt; (c) 1150 chunk chung, vi phạm 0, đạt.
- Đổi chỗ trong dung sai: không có.
- (a) và (b) tính trên mọi câu không đạt, đúng dự kiến vì promotion đổi kết quả của câu article_ref: (a) khác ở q093, q094, q095; (b) vi phạm ở q092–q095.
- R@1 mức Điều, article_ref: 9/12 → 12/12. Hạng Khoản: q097 1 → 1; q102 1 → 1.
- Số liệu article_ref là độ phủ của parser (parser được nghiệm thu trên chính văn bản 12 câu này, spec mục 7.4), không kiểm định thống kê.
- **Quyết định: bật promotion** (D3 và luật 5.5: (d) đạt, R@1 article_ref không giảm).

### 4. Câu tụt hạng (mức Điều, ngoài top 10 tính là 11)

Run (3) so với run (1), 21 câu:

- q001 [direct] 2 → 3; nguyên nhân: chunk khác trùng từ khóa
- q004 [paraphrase] 2 → 3; nguyên nhân: chunk khác trùng từ khóa
- q006 [paraphrase] 4 → ngoài top 10; nguyên nhân: chunk khác trùng từ khóa
- q010 [paraphrase] 1 → 5; nguyên nhân: chunk khác trùng từ khóa
- q012 [paraphrase] 1 → 3; nguyên nhân: chunk khác trùng từ khóa
- q014 [paraphrase] 1 → 2; nguyên nhân: chunk khác trùng từ khóa
- q016 [paraphrase] 2 → ngoài top 10; nguyên nhân: chunk khác trùng từ khóa
- q020 [paraphrase] 2 → 6; nguyên nhân: chunk khác trùng từ khóa
- q026 [paraphrase] 1 → 4; nguyên nhân: chunk khác trùng từ khóa
- q030 [paraphrase] 2 → ngoài top 10; nguyên nhân: chunk khác trùng từ khóa
- q032 [paraphrase] 1 → 4; nguyên nhân: chunk khác trùng từ khóa
- q040 [paraphrase] 3 → 5; nguyên nhân: chunk khác trùng từ khóa
- q042 [paraphrase] 1 → 2; nguyên nhân: chunk khác trùng từ khóa
- q052 [paraphrase] 1 → ngoài top 10; nguyên nhân: chunk khác trùng từ khóa
- q058 [paraphrase] 1 → 2; nguyên nhân: chunk khác trùng từ khóa
- q068 [paraphrase] 1 → 3; nguyên nhân: chunk khác trùng từ khóa
- q070 [paraphrase] 1 → 2; nguyên nhân: chunk khác trùng từ khóa
- q074 [paraphrase] 2 → 3; nguyên nhân: chunk khác trùng từ khóa
- q080 [paraphrase] 1 → ngoài top 10; nguyên nhân: chunk khác trùng từ khóa
- q082 [paraphrase] 3 → 7; nguyên nhân: chunk khác trùng từ khóa
- q090 [paraphrase] 1 → 2; nguyên nhân: chunk khác trùng từ khóa

Run (4) so với run (1): 0 câu.

Run (5) so với run (3): 0 câu.

Thăm dò, không dùng để quyết định — run (2) sparse so với run (1), 36 câu (không gán nguyên nhân):

- q001 [direct] 2 → 3
- q004 [paraphrase] 2 → ngoài top 10
- q006 [paraphrase] 4 → ngoài top 10
- q007 [direct] 1 → 3
- q010 [paraphrase] 1 → ngoài top 10
- q012 [paraphrase] 1 → 8
- q013 [direct] 1 → 2
- q014 [paraphrase] 1 → 7
- q016 [paraphrase] 2 → ngoài top 10
- q020 [paraphrase] 2 → 8
- q022 [paraphrase] 1 → 4
- q026 [paraphrase] 1 → 4
- q028 [paraphrase] 2 → ngoài top 10
- q030 [paraphrase] 2 → ngoài top 10
- q032 [paraphrase] 1 → ngoài top 10
- q034 [paraphrase] 1 → 2
- q040 [paraphrase] 3 → ngoài top 10
- q042 [paraphrase] 1 → 8
- q046 [paraphrase] 3 → ngoài top 10
- q048 [paraphrase] 3 → ngoài top 10
- q052 [paraphrase] 1 → ngoài top 10
- q054 [paraphrase] 1 → 2
- q056 [paraphrase] 1 → 3
- q058 [paraphrase] 1 → 9
- q062 [paraphrase] 1 → 5
- q064 [paraphrase] 1 → 3
- q068 [paraphrase] 1 → ngoài top 10
- q070 [paraphrase] 1 → ngoài top 10
- q074 [paraphrase] 2 → ngoài top 10
- q076 [paraphrase] 1 → 2
- q078 [paraphrase] 1 → 3
- q080 [paraphrase] 1 → ngoài top 10
- q082 [paraphrase] 3 → ngoài top 10
- q088 [paraphrase] 1 → 10
- q090 [paraphrase] 1 → ngoài top 10
- q091 [article_ref] 1 → 5

### 5. Luật lấp chỗ spec để ngỏ

- Cố định trước run (2). Hòa điểm trong nhánh (T3): dense theo (−cosine, chunk id), sparse theo (−BM25, hạng dense). Parser số Điều (T4): các cách gọi khác của Bộ luật này, gồm "luật" đứng chung chung; loại bản 2012; "của"/"trong" nối với hợp đồng, nội quy, quy chế, thỏa ước, điều lệ là văn bản khác; truy vấn promotion dense chạy exact, hòa cosine theo chunk id; danh sách từ chỉ lượng chút, tí, tý, tị, xíu, tẹo, lát. Nội dung ghi trong commit message của `f3379ab` và `8744637`.
- Luật tuyên bố §5.6 chỉ viết sẵn câu 'không có ý nghĩa thống kê' cho trường hợp C1 không đạt; kết quả rơi vào chiều ngược (mid-p ≤ 0,05 với c > b), nên câu tuyên bố ở README do người điều phối chọn sau lần nhìn đầu tiên. Câu này không đổi kiểm định (§5.2), α hay quyết định (§5.4), và không tuyên bố cải thiện nào.

### 6. Tiếp theo

- T7 đã xong trong commit `feat: ship dense retrieval with article promotion as the default (T7)`: mặc định là dense trên collection chỉ có dense, bật promotion số Điều.
