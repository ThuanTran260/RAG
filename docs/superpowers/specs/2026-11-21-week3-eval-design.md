# Tuần 3: Bộ đánh giá truy xuất (Design)

**Trạng thái:** người dùng duyệt ngày 21/11/2026, gồm cả lựa chọn ghép cặp và yêu cầu lập plan rồi thực thi. Đã qua một vòng builder check của reviewer Opus. Code đã xong; ngày 22/11/2026 người dùng giao việc duyệt bộ câu cho người điều phối (ba reviewer Claude Opus, theo tiêu chí từ tài liệu nghiên cứu), và baseline dense đã đo (xem `eval/results.md`).

## 1. Mục đích

Hiện chưa có số liệu nào cho biết truy xuất dense sai thường xuyên đến đâu. Trường hợp duy nhất đã biết là câu "Người lao động muốn nghỉ việc phải báo trước bao nhiêu ngày?". Với câu này, `45-2019-QH14:d36:k2` (báo trước của người sử dụng lao động) xếp hạng 1, còn Điều 35 xếp hạng 3 (`README.md:82`, `README.md:99`).

Tuần 3 xây bộ câu hỏi có đáp án và script đo recall@5, recall@10 và MRR. Kết quả có hai công dụng:

- Cho biết lỗi kiểu "Điều 36 trên Điều 35" phổ biến đến mức nào.
- Làm baseline để Tuần 4 (hybrid search, RRF) và Tuần 5 (reranker, đổi cách chunk) chứng minh mình hiệu quả hơn. Từ Tuần 3, mọi thay đổi truy xuất đều phải đo lại (`AGENTS.md`, mục Eval gate).

Thuật ngữ dùng trong tài liệu:

- **recall@k:** tỷ lệ câu hỏi có ít nhất một kết quả đúng trong k kết quả đầu.
- **MRR** (mean reciprocal rank): trung bình của 1/hạng của kết quả đúng đầu tiên; câu không có kết quả đúng tính là 0.

## 2. Mục tiêu, ngoài phạm vi, anti-goal

**Mục tiêu**

1. `eval/questions.jsonl` có ít nhất 110 câu đã được người dùng duyệt từng câu: ít nhất 40 câu trực tiếp, 40 câu đời thường, 20 câu ngoài phạm vi và 10 câu nêu số Điều (nhóm `article_ref`, phục vụ phần lọc metadata ở Tuần 4, `KE_HOACH_RAG.md:117`). Ngày 22/11/2026 người dùng giao việc duyệt này cho người điều phối, người này giao lại cho ba reviewer Claude Opus; bộ câu vì thế chưa được người duyệt từng câu.
2. Lệnh `uv run python -m eval.run_eval` in recall@5, recall@10 và MRR ở hai mức (Điều và Khoản), có tách theo nhóm câu hỏi, liệt kê các câu trượt và phân bố điểm top-1 của câu ngoài phạm vi.
3. Ghi dòng baseline dense vào `eval/results.md`.
4. Có công cụ duyệt từng câu theo kiểu wizard: `uv run python -m eval.review_questions`.

**Ngoài phạm vi**

- Đặt ngưỡng điểm để từ chối. Tuần 3 chỉ ghi lại phân bố điểm làm căn cứ.
- Đánh giá chất lượng câu trả lời của `generate`, gồm faithfulness và RAGAS (thuộc Tuần 5).
- Hybrid search, reranker, đổi cách chunk.
- Dùng câu hỏi từ `thangvip/vietnamese-legal-qa`, Zalo Legal hoặc cổng thông tin. Các nguồn này có license riêng, cần kiểm tra trước khi đưa vào repo public.
- Gọi LLM trong quá trình đánh giá.

**Anti-goal** (dù chạy được vẫn coi là thất bại)

- **Bộ câu hỏi bị chỉnh theo retriever hiện tại.** Khi soạn và khi duyệt, không ai được xem kết quả `retrieve` của câu hỏi. Nếu không, baseline sẽ đẹp giả và Tuần 4–5 không còn gì để chứng minh.
- **Đáp án sai luật.** Mỗi id đáp án phải tồn tại trong `data/processed/chunks.jsonl`, và người soạn phải đọc nguyên văn chunk trước khi gán (`AGENTS.md`, mục Nguồn sự thật pháp lý).
- **Số liệu không so được giữa các tuần.** Phải giữ ổn định cách tính, bộ câu và cách đếm hạng; xem §5 và §6.
- **Câu "đời thường" chép lại từ ngữ của luật.** Khi đó nhóm này đo cùng một thứ với nhóm trực tiếp.

## 3. Ràng buộc

- Repo public. Câu hỏi soạn mới, không chép từ nguồn bên ngoài. Không chứa thông tin cá nhân thật.
- Text chuẩn hóa NFC, đọc và ghi bằng `encoding="utf-8"`. CLI gọi `sys.stdout.reconfigure(encoding="utf-8")` (và stderr) giống `app/rag/search.py:23`.
- Đo truy xuất bằng `app.rag.retriever.default_retriever()`, giống hệt cách `search` và `ask` đang dùng. Cần Qdrant đang chạy với 676 point và model bge-m3 có trong cache cục bộ.
- Pytest mặc định không cần Qdrant hay model. Test cần chúng gắn marker `integration` (`pyproject.toml`).
- Không thêm dependency mới. Thống kê đơn giản dùng module `statistics` của thư viện chuẩn.

## 4. Các phương án đã cân nhắc (cách tính "trúng")

Người dùng giao quyền chọn cho người điều phối; phương án dưới đây là lựa chọn của người điều phối.

- **A. Chỉ mức Khoản:** trúng khi đúng id chunk. Khắt khe và sát với trích dẫn, nhưng sẽ hỏng khi Tuần 5 thử chunk theo Điều, vì id `d35:k1` khi đó không còn tồn tại.
- **B. Chỉ mức Điều:** bền và đơn giản, nhưng không thấy được lỗi chọn sai khoản trong cùng một Điều.
- **C. Đáp án ghi ở mức Khoản, tính cả hai mức; số chính là recall@5 mức Điều.** ← chọn
- **D. Đáp án chia "chính" và "chấp nhận được":** bỏ, vì phân loại chủ quan và làm tăng công duyệt.

Lý do chọn C:

- Dữ liệu thật có 220 Điều trong 676 chunk; phần lớn Điều có 1–4 chunk, chỉ 6 Điều có từ 8 chunk trở lên (3, 34, 67, 154, 156, 178). Vì vậy tính ở mức Điều không làm con số dễ đạt quá mức.
- Mức Điều so được xuyên suốt Tuần 3–5. Mức Khoản được giữ để thấy lỗi trích dẫn.

## 5. Bộ câu hỏi

### 5.1. Định dạng

Hai file có cùng schema: bản nháp `eval/questions.draft.jsonl` và bản đã duyệt `eval/questions.jsonl`. Mỗi dòng là một câu:

```json
{"id": "q001", "group": "direct", "pair_id": "p01", "question": "...", "gold_ids": ["45-2019-QH14:d35:k1"], "gold_articles": [35], "note": "..."}
```

- `group`: một trong `direct`, `paraphrase`, `article_ref`, `out_of_scope`. `article_ref` là câu nêu thẳng số Điều, ví dụ "Điều 24 quy định gì?".
- `gold_ids`: danh sách id chunk, theo thứ tự trong corpus và không trùng lặp. Câu `out_of_scope` có danh sách rỗng; các câu khác có ít nhất một id.
- `gold_articles`: tập số Điều của `gold_ids`, lấy từ metadata chunk và sắp tăng dần. Trường này được lưu riêng để mức Điều vẫn tính được khi Tuần 5 đổi cách chunk và id cũ không còn.
- `note`: tùy chọn, ghi lý do chọn đáp án hoặc vì sao câu nằm ngoài phạm vi.
- `id` không trùng lặp. Câu bị loại giữ lại id của mình, không đánh số lại.

`eval/questions.py` tách việc kiểm tra thành hai hàm. Cả hai báo lỗi rõ ràng, kèm số dòng.

- **`load_questions(path)`** kiểm tra cấu trúc, không cần corpus. Hàm báo lỗi khi:
  - JSON hỏng, thừa trường hoặc thiếu trường;
  - `group` không hợp lệ hoặc `id` trùng lặp;
  - text chưa ở dạng NFC;
  - số Điều nằm ngoài khoảng 1–220;
  - câu trong phạm vi mà không có đáp án, hoặc câu ngoài phạm vi mà có đáp án.
- **`check_gold_ids(questions, chunks)`** đối chiếu với corpus. Hàm báo lỗi khi một id không có trong `chunks`, hoặc khi `gold_articles` không khớp với Điều của các id. Wizard và unit test luôn gọi hàm này với `data/processed/chunks.jsonl`.

`run_eval` gọi `load_questions`, rồi gọi `check_gold_ids` với corpus hiện tại:

- Nếu khớp, tính cả hai mức.
- Nếu có id không còn tồn tại, ví dụ khi Tuần 5 chunk lại theo Điều, thì in cảnh báo và chỉ tính mức Điều; mức Khoản in là `N/A`.
- Nếu id vẫn còn nhưng `gold_articles` lệch, thì báo lỗi và thoát mã 1, vì như vậy là đáp án sai chứ không phải corpus đổi.

Nhờ vậy bộ câu đã đóng băng vẫn chạy được sau khi corpus đổi (lựa chọn của người điều phối).

Số câu mỗi nhóm thấp hơn mục tiêu ở §2 chỉ là cảnh báo, không chặn.

### 5.2. Soạn nháp (người điều phối hoặc agent)

- **Số lượng:** soạn 127 câu (45 trực tiếp, 45 đời thường, 12 nêu số Điều, 25 ngoài phạm vi), để sau khi loại bớt vẫn còn ít nhất 40/40/10/20.
- **Câu nêu số Điều:** chọn Điều trải đều các Chương. Câu có thể chỉ hỏi số Điều ("Điều 24 quy định gì?") hoặc kết hợp số Điều với nội dung ("Theo Điều 35, người lao động phải báo trước bao lâu?"). Đáp án là các khoản của Điều đó mà câu hỏi nhắm tới; câu chỉ hỏi số Điều thì đáp án là mọi khoản của Điều.
- **Ghép cặp câu trực tiếp và câu đời thường** (người dùng đồng ý ngày 21/11/2026). Mỗi câu đời thường diễn đạt lại một câu trực tiếp và có cùng `gold_ids`. Hai câu cùng cặp mang chung giá trị `pair_id` (ví dụ `p01`); các nhóm khác có `pair_id` là `null`. 45 cặp ứng với 45 Điều khác nhau.
  - **Lý do:** chênh lệch recall giữa hai câu cùng cặp chỉ còn do cách diễn đạt, nên Tuần 4–5 biết được hybrid search hay reranker thắng nhờ khớp từ khóa hay nhờ hiểu nghĩa.
  - **Cái giá:** phủ ít Điều hơn, và hai câu cùng cặp thường cùng trúng hoặc cùng trượt. Với giả định tương quan 0,5, 90 câu có giá trị thống kê cỡ 60 câu độc lập; đây là ước lượng chưa đo.
  - **Giảm nhẹ:** 12 câu `article_ref` chọn các Điều không trùng với 45 cặp.
- **Quy tắc kiểm tra cặp** (thêm vào `load_questions`): mỗi `pair_id` có đúng một câu `direct` và một câu `paraphrase`, và hai câu có cùng `gold_ids`. Khi wizard loại hoặc sửa một câu, nó hiện câu cùng cặp và đề nghị xử lý luôn câu đó: sửa id thì áp dụng cho cả hai câu; loại một câu thì câu kia bị loại theo. Nhờ vậy `questions.jsonl` chỉ chứa cặp đủ hai câu.
- **Phân bố:** 45 Điều của các cặp chia theo số Điều của từng Chương (Chương III có 46 Điều, Chương XIV có 33 Điều...). Mỗi Chương có ít nhất một Điều. Mỗi Điều chỉ thuộc một cặp.
- **Ca khó cố định:** bộ câu phải có câu "Người lao động muốn nghỉ việc phải báo trước bao nhiêu ngày?" với đáp án `d35:k1`. Thêm vài cặp Điều dễ nhầm có chủ thể khác nhau, chẳng hạn quyền của người lao động và của người sử dụng lao động. Các cặp lấy từ văn bản thật khi soạn.
- **Câu trực tiếp:** dùng thuật ngữ trong luật.
- **Câu đời thường:** viết như người dân hỏi, không dùng cụm từ trong tiêu đề Điều hay thuật ngữ chính của khoản đáp án. Ví dụ "bị đuổi việc" thay cho "đơn phương chấm dứt hợp đồng lao động".
- **Ưu tiên câu chỉ hỏi một ý.** Chỉ ghi nhiều id khi câu trả lời thật sự cần nhiều khoản.
- **Câu ngoài phạm vi:** một câu chỉ thuộc nhóm này khi không Điều nào của Bộ luật Lao động trả lời được nó, kể cả trả lời một phần. Câu "lưng chừng", tức có Điều liên quan nhưng không có con số cụ thể, thì không soạn. Ví dụ "mức đóng bảo hiểm xã hội là bao nhiêu phần trăm?" chạm tới `d168:k1` (nghĩa vụ tham gia bảo hiểm xã hội) nên bị loại. Lý do: một câu lưng chừng bị xếp vào nhóm nào cũng làm sai lệch, hoặc recall, hoặc phân bố điểm dùng để đặt ngưỡng sau này (lựa chọn của người điều phối). Nhóm gồm ba loại:
  - lĩnh vực luật khác, như giao thông, hình sự, đất đai;
  - gần với lao động nhưng Bộ luật không nói tới, như cách tính thuế thu nhập cá nhân trên tiền lương hay thủ tục đăng ký doanh nghiệp. Mỗi câu loại này phải được kiểm tra bằng cách tìm từ khóa trong `chunks.jsonl`, và ghi kết quả vào `note`;
  - không phải câu hỏi pháp luật.

  Loại thứ hai chiếm khoảng một nửa nhóm.
- Wizard không đổi được `group`. Câu bị xếp sai nhóm thì loại rồi soạn lại trong file nháp.
- **Quy tắc bắt buộc:** người soạn đọc nguyên văn `text` của chunk trước khi gán id, và không chạy `search`, `ask` hay `retrieve` lên câu hỏi mình soạn.

### 5.3. Duyệt bằng wizard (người dùng)

Lệnh `uv run python -m eval.review_questions`, file `eval/review_questions.py`. Đây là bước duy nhất cần người dùng làm tay.

Mỗi câu hiện trên một màn hình riêng (xóa màn hình trước khi hiện câu mới), gồm:

- tiến độ dạng `12/115`, kèm số câu đã duyệt và đã loại;
- nhóm của câu, câu hỏi và ghi chú;
- với mỗi id đáp án: nhãn `Điều X, khoản Y`, tiêu đề Điều và nguyên văn `chunk.text`;
- với câu ngoài phạm vi: dòng "không có đáp án", kèm lý do trong `note`.

Các phím thao tác:

- `a`: duyệt.
- `e`: sửa danh sách id. Id mới được kiểm tra ngay; id sai thì hỏi lại.
- `c`: xem toàn bộ các khoản của Điều đáp án, để quyết định có thêm khoản lân cận hay không.
- `r`: loại câu, kèm một dòng lý do.
- `q`: lưu tiến độ và thoát.

Quy tắc của công cụ:

- **Lưu sau mỗi quyết định.** Câu được duyệt thêm vào `eval/questions.jsonl`. Câu bị loại ghi vào `eval/questions.rejected.jsonl`, gồm id, lý do và bản ghi gốc. Chạy lại lệnh thì bỏ qua các id đã có ở một trong hai file, nên dừng giữa chừng rồi làm tiếp được.
- **Thứ tự ghi trong `questions.jsonl`** theo thứ tự duyệt. `run_eval` không phụ thuộc vào thứ tự này.
- **Không cho sửa nội dung câu hỏi trong console**, vì gõ tiếng Việt qua `input()` trên console Windows không đáng tin. Muốn sửa câu chữ thì sửa trong `questions.draft.jsonl` trước khi duyệt; công cụ in đường dẫn file khi người dùng chọn loại câu.
- **Tuyệt đối không hiện kết quả truy xuất** (xem anti-goal).
- **Kết thúc** bằng bản tóm tắt: số câu đã duyệt theo nhóm so với mục tiêu 40/40/20, số câu bị loại.

Phần lõi của công cụ tách khỏi `input`/`print`, để test được bằng một chuỗi thao tác giả.

### 5.4. Đóng băng bộ câu

Sau khi đo baseline, `eval/questions.jsonl` chỉ được thay đổi kèm một dòng giải thích trong `eval/results.md`. `run_eval` in mã sha256 (12 ký tự đầu) của file câu hỏi. Mỗi dòng trong `results.md` ghi mã này để biết các con số có được đo trên cùng bộ câu hay không.

## 6. Cách tính (`eval/metrics.py`, `eval/run_eval.py`)

### 6.1. Truy xuất và chấm điểm

- Mỗi câu gọi `retrieve` một lần với `k = 10`. recall@5 tính trên 5 kết quả đầu của chính danh sách này.
- Số chính tính trên câu `direct` và `paraphrase`. Câu `article_ref` có dòng riêng và không gộp vào số chính, để Tuần 4 thấy rõ phần lọc metadata thay đổi được gì mà số chính không bị lẫn.
- Hai mức trúng:
  - **Khoản:** `chunk.id ∈ gold_ids`.
  - **Điều:** `chunk.article ∈ gold_articles`.
- `hit@k`: có ít nhất một kết quả trúng trong top k. `recall@k` là trung bình của `hit@k`.
- `MRR@10` tính theo hạng (bắt đầu từ 1) của kết quả trúng đầu tiên trong danh sách chunk gốc; không có thì tính 0. Không gộp các chunk cùng Điều trước khi xếp hạng.
- **Số chính:** recall@5 mức Điều.

### 6.2. Hàm thuần trong `eval/metrics.py`

Các hàm không gọi Qdrant, để unit test được:

- `hit_rank(retrieved, is_hit) -> int | None`
- `recall_at(ranks, k) -> float`
- `mrr(ranks) -> float`

Câu ngoài phạm vi không tính vào recall. Với nhóm này chỉ ghi điểm top-1, rồi in min, trung vị và max của nó cạnh điểm top-1 của nhóm trong phạm vi, cộng thêm số câu ngoài phạm vi có điểm top-1 cao hơn điểm top-1 thấp nhất của nhóm trong phạm vi.

### 6.3. Đầu ra của `run_eval`

1. Thông tin lần chạy: collection, model, số câu theo nhóm, mã sha256 của bộ câu.
2. Bảng số liệu gồm các cột recall@5, recall@10, MRR@10, mỗi cột ở hai mức. Có bốn dòng: số chính (`direct` cộng `paraphrase`), `direct`, `paraphrase`, `article_ref`.
3. Danh sách câu trượt ở mức Điều trong top 5: id câu, câu hỏi, Điều đáp án, 3 kết quả đầu (id và điểm). Danh sách này dùng để tìm các lỗi kiểu Điều 35/36.
4. Phân bố điểm top-1 như ở §6.2.
4b. Bảng cặp, tính ở mức Điều trong top 5. Có bốn ô: cả hai câu trúng, chỉ câu trực tiếp trúng, chỉ câu đời thường trúng, cả hai trượt. Kèm danh sách `pair_id` của ô "chỉ câu trực tiếp trúng".
5. Một dòng markdown sẵn sàng dán vào `eval/results.md`. Ba cột số của bảng hiện có ghi số mức Điều; cột Ghi chú ghi số mức Khoản và mã sha256 của bộ câu.

Tùy chọn `--out PATH` ghi kết quả từng câu ra JSONL: id, nhóm, `rank_article`, `rank_clause`, top 10 (id và điểm). Lần chạy baseline lưu vào `eval/runs/2026-xx-xx-dense-baseline.jsonl` và commit, để Tuần 4–5 so sánh với từng câu.

### 6.4. Lỗi vận hành

Qdrant không chạy, collection rỗng, payload lệch schema, thiếu model hoặc cấu hình sai: bắt đúng các lỗi mà `main` của `app/rag/search.py:30-46` đang bắt, in một dòng lên stderr theo cùng cách (`UnexpectedResponse` dùng `app.rag.store.rejection_message(error, searching=True)`), rồi thoát mã 1. Lỗi đầu tiên dừng cả lần chạy; không bỏ qua câu lỗi rồi đi tiếp, vì như vậy con số sẽ được tính trên một bộ câu thiếu. File câu hỏi không hợp lệ: in lỗi kèm số dòng, rồi thoát mã 1. Bộ câu rỗng là lỗi, vì `run_eval` không còn là placeholder.

## 7. File và test

**File mới**

- `eval/__init__.py`
- `eval/questions.py`
- `eval/metrics.py`
- `eval/review_questions.py`
- `eval/questions.draft.jsonl`
- `eval/questions.rejected.jsonl`
- `eval/runs/…`
- `tests/test_eval_questions.py`
- `tests/test_eval_metrics.py`
- `tests/test_review_questions.py`

**File sửa**

- `eval/run_eval.py`
- `eval/questions.jsonl`
- `eval/results.md`
- `README.md` (mục đánh giá: cách chạy và cách đọc số)
- `KE_HOACH_RAG.md` (tick mục Tuần 3)

**Test**

- **Unit:**
  - kiểm tra dữ liệu trong `load_questions` với từng loại lỗi ở §5.1;
  - các hàm metric với danh sách giả, gồm câu có nhiều id đáp án, câu không trúng và câu trúng ở hạng 1, 5, 6, 10;
  - lõi của wizard với chuỗi thao tác giả, gồm duyệt, sửa id sai rồi sửa đúng, loại câu, thoát rồi chạy lại.
- **Integration** (marker `integration`): `run_eval` chạy trên Qdrant thật với 2–3 câu mẫu và in ra đủ các mục.
- Không có test nào được gọi LLM.

## 8. Phần để người xây dựng tự quyết

- Bố cục chi tiết của bảng in ra và màn hình wizard, miễn chứa đủ các mục ở §5.3 và §6.3.
- Cách xóa màn hình trên Windows và POSIX.
- Chia task và thứ tự commit. Gợi ý thứ tự: `questions.py` → `metrics.py` → `run_eval.py` → wizard → soạn nháp → người dùng duyệt → đo baseline.
- Ai soạn nháp (người điều phối hay sub-agent) và tier tương ứng. Việc soạn phải chạy trong checkout chính, vì cần đọc dữ liệu thật.
