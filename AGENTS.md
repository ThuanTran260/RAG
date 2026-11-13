# Hướng dẫn cho agent: dự án RAG pháp luật lao động

Dự án trả lời câu hỏi về Bộ luật Lao động 2019 bằng RAG, có trích dẫn Điều/Khoản. File này là nguồn quy tắc chung cho mọi agent. `CLAUDE.md` nạp file này và chỉ bổ sung phần riêng của Claude Code.

- Phạm vi, stack, lộ trình 8 tuần: `KE_HOACH_RAG.md`. Khi nhận việc thuộc một tuần, đọc tuần đó ở mục 5 và các lưu ý ở mục 7.
- Plan chi tiết từng tuần: `docs/superpowers/plans/`. Plan của tuần có "Global Constraints" thì tuân theo phần đó.

## Ràng buộc cứng

Repo là **PUBLIC**: mọi thứ đã commit coi như đã công bố.

- **Nguồn sự thật pháp lý.** Mọi số Điều, Khoản và nội dung luật trong code, prompt, test hoặc đáp án eval phải trích từ `data/raw/2026_131_18_VBHN-VPQH.docx` (văn bản hợp nhất Bộ luật Lao động 45/2019/QH14) hoặc `data/processed/chunks.jsonl`. Mở file và trích, kể cả khi bạn nghĩ mình đã nhớ.
- **File Nghị định 145/2020** trong `data/raw/` chưa dùng: không làm đầu vào, không commit. Nguồn gốc và checksum của từng file gốc ghi ở `data/raw/SOURCES.md` (khi đã tạo).
- **Commit có xác nhận.** File trong `data/raw/` và `.env` chỉ được `git add` sau khi người dùng xác nhận trong phiên hiện tại, kể cả khi plan có sẵn lệnh `git add` cho chúng. Trước mỗi commit, chạy `git status --short` và đối chiếu từng file với danh sách được phép.
- **Khóa API** chỉ đọc từ biến môi trường. Tên biến khai báo trong `.env.example`. Giá trị khóa không xuất hiện trong log, output, commit hay brief.

## Quy ước

- Trả lời người dùng bằng tiếng Việt có dấu. Comment trong code, commit message, tên branch và mô tả PR viết bằng tiếng Anh.
- Pipeline RAG tự viết bằng Python và thư viện client, không dựa vào LangChain/LlamaIndex (lý do ở `KE_HOACH_RAG.md` mục 3).
- Text tiếng Việt chuẩn hóa NFC. Đọc và ghi file với `encoding="utf-8"`. Máy dev chạy Windows: lệnh in tiếng Việt ra terminal phải chạy được cả khi console không dùng UTF-8.

## Quy trình mỗi task

1. **Chia việc.** Mỗi task có đủ bốn trường: mục tiêu, file được phép sửa, tiêu chí nghiệm thu, lệnh kiểm tra.
2. **Phân tier** theo bảng dưới. Không chắc thì chọn tier cao hơn.
3. **Thiết kế** (chỉ tier Khó). Trước khi code, viết ra: hàm và kiểu dữ liệu, các ca biên lấy từ văn bản thật, danh sách test. Xong khi người điều phối hoặc người dùng duyệt. Nếu plan của tuần đã có thiết kế và code mẫu cho task đó, bước này coi như xong.
4. **Làm.** Mỗi lần chỉ thay đổi một thứ.
5. **Kiểm tra cơ học.** Chạy lại lệnh kiểm tra, sau đó chạy `git status --short` (để thấy cả file mới) và `git diff --stat`. Xong khi lệnh pass và mọi file đổi hoặc file mới đều nằm trong danh sách được phép.
6. **Review.** Tier Dễ: người điều phối đọc diff. Tier Trung bình và Khó: một reviewer có ngữ cảnh mới nhận diff, spec và tiêu chí, trả về lỗi kèm `file:line`, mỗi lỗi đánh dấu blocker hoặc nit. Xong khi có verdict.
7. **Sửa vòng.** Tối đa hai vòng cho mỗi tier. Hết hai vòng mà còn blocker: task Trung bình nâng lên Khó (thêm bước thiết kế, viết lại brief); task Khó thì dừng và báo người dùng kèm lỗi cụ thể.
8. **Eval gate**, nếu task đụng tới chunking, truy xuất, reranker hoặc prompt (mục dưới).

| Tier | Ví dụ trong dự án |
|---|---|
| Dễ | Tạo file hoặc thư mục, sửa config, `.gitignore`, docstring, sửa một file theo spec đã đủ giá trị |
| Trung bình | Hàm hoặc module 1–3 file theo spec, pytest cho hàm có sẵn, chunker với input mẫu, script recall@k và MRR |
| Khó | Parser cấu trúc luật có ngoại lệ, hybrid search và RRF, lọc metadata, prompt chống bịa, debug truy xuất sai |

## Eval gate

- Chạy `eval/run_eval.py` trước và sau thay đổi. Ghi một dòng vào `eval/results.md` (recall@5, recall@10, MRR), kể cả khi thử nghiệm không hiệu quả.
- Nếu recall@5 giảm, chỉ chốt thay đổi sau khi giải thích được nguyên nhân.
- Baseline đo ở Tuần 3. Trước đó `run_eval.py` là placeholder và `eval/questions.jsonl` rỗng, nên gate dựa vào test và số liệu chuẩn trong plan của tuần (nếu plan có); báo cáo ghi rõ "chưa đo recall".

## Điều phối sub-agent

"Người điều phối" là phiên đang chia việc. "Brief" là prompt giao việc cho sub-agent.

- **Brief tự đứng được**, vì sub-agent không thấy hội thoại: mục tiêu, file được phép sửa, tiêu chí, lệnh kiểm tra, phần ngoài phạm vi. Sub-agent xong khi trả về danh sách file đã đổi và output của lệnh kiểm tra.
- **Model:** implementer dùng model tầm trung; reviewer dùng model mạnh hơn implementer; model nhỏ nhất chỉ làm tier Dễ và không review.
- **Song song** chỉ khi tập file không giao nhau và mỗi sub-agent có worktree riêng. Worktree mới không chứa file chưa được track (hiện gồm `data/raw/` và `.env`), nên task cần đọc văn bản gốc chạy tuần tự trong checkout chính.
- **Sửa nhỏ** (dưới một dòng, không đổi hành vi): người điều phối tự làm.
