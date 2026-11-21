# Trợ lý tra cứu pháp luật lao động (RAG)

Hỏi đáp tiếng Việt về Bộ luật Lao động 2019 (45/2019/QH14), trả lời kèm trích dẫn Điều/Khoản gốc.

> Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.

**Trạng thái:** xong Tuần 1 (dữ liệu và chunking) và nửa đầu Tuần 2 (Qdrant và truy xuất dense). Phần sinh câu trả lời có trích dẫn (lệnh `ask`) đã có mã và test đơn vị nhưng chưa chạy với nhà cung cấp LLM thật, chưa đo chất lượng hay độ trễ; Tuần 2 chưa được xác nhận là xong. Kế hoạch chi tiết xem tại [KE_HOACH_RAG.md](KE_HOACH_RAG.md).

## Cách chạy

```bash
uv sync --locked
uv run python -m ingestion.download_model
docker compose up -d qdrant
uv run python -m ingestion.index
uv run python -m app.rag.search "Thử việc tối đa bao lâu?"
uv run python -m app.rag.ask "Thử việc tối đa bao lâu?"
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

## Sinh câu trả lời (`ask`)

Cấu hình LLM nằm trong `.env` (sao chép từ `.env.example`, không commit) hoặc biến môi trường:

- `LLM_PROVIDER` (mặc định `gemini`, hiện chỉ hỗ trợ `gemini`).
- `LLM_API_KEY` (không có mặc định; không commit, không ghi vào log hay brief).
- `LLM_MODEL` (mặc định `gemini-3.5-flash-lite`).

`uv run python -m app.rag.ask "<câu hỏi>" [-k N]` chạy theo thứ tự: kiểm tra câu hỏi và `k` (mặc định 5, tối đa 8), đọc cấu hình, dựng client LLM, truy xuất `k` chunk dense như `search`, rồi gọi LLM tối đa một lần (không gọi khi truy xuất không có chunk nào). Thiếu khóa thì dừng trước khi nạp model embedding hay gọi Qdrant. Vì có bước truy xuất, `ask` vẫn cần model embedding trong cache cục bộ (`uv run python -m ingestion.download_model`) và Qdrant đang chạy (`docker compose up -d qdrant`) với collection đã index.

- Client: Gemini `gemini-3.5-flash-lite` qua Interactions API với `store=False`, timeout 30 giây, tối đa 4096 token đầu ra (gồm token suy nghĩ), `thinking_level` `minimal`, một request cho mỗi câu hỏi, không retry, không fallback. Chi tiết SDK chỉ nằm ở `app/rag/llm.py`.
- Mô hình chỉ trả JSON gồm `status` (`answered` hoặc `insufficient`) và `claims` (tối đa 6 khẳng định, mỗi khẳng định tối đa 3 mã nguồn `S1`, `S2`...). Mã nguồn, nhãn Điều/Khoản, tiêu đề Điều và nguyên văn đều do hệ thống lấy từ metadata của chunk, không lấy từ văn bản mô hình viết.
- Đầu ra khi `answered`: danh sách khẳng định kèm `[S#]`, mục `Căn cứ:` (nhãn, tiêu đề Điều, id chunk, điểm cosine, nguyên văn chunk), dòng `Cảnh báo` cho mỗi số Điều mà câu trả lời nhắc nhưng không có trong các nguồn đã truy xuất, rồi lưu ý đối chiếu và tuyên bố tham khảo. Khi `insufficient` hoặc truy xuất không có chunk, in một câu từ chối cố định.
- Lỗi vận hành in lên stderr và trả mã thoát 1: câu hỏi hoặc `k` không hợp lệ; Qdrant hoặc model embedding không dùng được; cấu hình sai (`invalid settings: <TÊN BIẾN>...`, chỉ tên biến, không in giá trị); lỗi LLM (`LLM: ...`, thông báo đã làm sạch); phản hồi vi phạm hợp đồng (`invalid model answer: <mã>`, không in văn bản mô hình).
- Khi chưa có khóa, lệnh `uv run python -m app.rag.ask "..."` in đúng một dòng `LLM: LLM_API_KEY is not configured` lên stderr và thoát mã 1, trước khi truy xuất (điều phối viên chạy thực tế ngày 15/11/2026).

### Chính sách dữ liệu

- `ask` gửi câu hỏi và nội dung các chunk đã truy xuất tới Google Gemini API.
- Theo trang pricing ngày 15/11/2026 mà kế hoạch dựa vào, free tier có thể dùng nội dung để cải thiện sản phẩm của Google. `store=False` là một thiết lập khác và không chứng minh việc loại trừ đó.
- Không nhập thông tin cá nhân vào câu hỏi.

### Smoke test LLM thật

`uv run pytest` mặc định không gọi nhà cung cấp LLM: `addopts` loại marker `llm` khỏi lần chạy không có `-m` tường minh (với `-m` tường minh, hai test vẫn được chọn nhưng bị bỏ qua nếu thiếu `RAG_LIVE_LLM=1`). Chỉ khi đặt `RAG_LIVE_LLM=1` thì `tests/test_llm_live.py` mới được phép gửi request; ngoài ra cần `LLM_API_KEY`. Mỗi lần chạy gửi tối đa 2 request (mỗi test một request, không retry) tới nhà cung cấp trả phí theo mô tả marker `llm`. PowerShell:

```powershell
$env:RAG_LIVE_LLM = "1"; uv run --locked pytest -m llm tests/test_llm_live.py -q -rs; Remove-Item Env:RAG_LIVE_LLM
```

```bash
RAG_LIVE_LLM=1 uv run --locked pytest -m llm tests/test_llm_live.py -q -rs
```

Smoke chỉ kiểm tra hợp đồng kỹ thuật (JSON đúng schema nhỏ; kết quả `generate` trên chunk `45-2019-QH14:d1` thỏa hợp đồng nhãn và nguyên văn), không kiểm tra ý nghĩa pháp lý hay chất lượng. Lần chạy thật đầu tiên là ngày 15/11/2026 với `gemini-3.5-flash-lite` (free tier): smoke 2 test đều qua; kết quả ghi ở `eval/results.md`.

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

## Đánh giá truy xuất

```powershell
uv run python -m eval.run_eval [--out eval/runs/<ngày>-<nhãn>.jsonl] [--label "<thay đổi>"]
```

Lệnh cần Qdrant đang chạy và model embedding trong cache cục bộ. Mỗi câu gọi `retrieve` một lần với `k = 10`; không gọi LLM.

- **Bộ câu** `eval/questions.jsonl` (127 câu): 45 cặp câu trực tiếp / câu đời thường có cùng đáp án, 12 câu nêu số Điều (`article_ref`), 25 câu ngoài phạm vi. Đáp án là danh sách id chunk mức Khoản; `gold_articles` lưu riêng để mức Điều vẫn tính được khi đổi cách chunk.
- **Hai mức đo:** mức Khoản trúng khi đúng id chunk; mức Điều trúng khi chunk thuộc Điều đáp án. Số chính là recall@5 mức Điều trên 90 câu `direct` + `paraphrase`. Câu ngoài phạm vi không tính recall, chỉ ghi điểm top-1.
- **Cách tạo bộ câu:** bản nháp do các model Claude soạn và sửa (chủ yếu Claude Sonnet) từ nguyên văn `chunks.jsonl`, không chạy truy xuất khi soạn hay duyệt. Hai vòng review nội dung và một vòng duyệt nghiệm thu do Claude Opus thực hiện theo các tiêu chí: đáp án đủ (Buckley & Voorhees, SIGIR 2004), câu trả lời được và không mơ hồ, tránh trùng từ khóa với chunk đáp án (Thakur và cộng sự, BEIR, NeurIPS 2021), câu ngoài phạm vi trông liên quan nhưng không Điều nào trả lời (Rajpurkar và cộng sự, ACL 2018), giữ đủ số câu (Voorhees & Buckley, SIGIR 2002). **Bộ câu chưa được người duyệt**; đáp án do một họ model soạn và duyệt, nên có rủi ro thiên lệch (Faggioli và cộng sự, ICTIR 2023; Panickssery và cộng sự, NeurIPS 2024).
- **Đáp án sau duyệt nghiệm thu:** Điều 129 là đáp án của hai cặp (p25 và p31), vì `d129:k1` được thêm vào p31 cho đủ đáp án; p38 có thêm `d193:k3` và `d197:k3`. Quy tắc "mỗi Điều chỉ thuộc một cặp" của spec (§5.2) vì vậy không còn đúng cho bộ đã duyệt, và p31 dễ trúng hơn một chút ở mức Điều (chunk của Điều 129 hoặc 164 đều tính). Các chỉnh sửa đáp án của p31 và p38 chỉ nằm trong `eval/questions.jsonl`; `eval/questions.draft.jsonl` vẫn giữ đáp án gốc, nên chạy lại wizard từ file duyệt rỗng sẽ khôi phục đáp án cũ.
- **Đóng băng:** sau baseline, `eval/questions.jsonl` chỉ đổi kèm một dòng giải thích trong `eval/results.md`. `run_eval` in mã sha256 (12 ký tự đầu) của bộ câu; mỗi dòng kết quả ghi mã này.

### Baseline dense (22/11/2026, bge-m3, bộ câu `577b8392afeb`)

| Nhóm | n | Điều R@5 | Điều R@10 | Điều MRR@10 | Khoản R@5 | Khoản R@10 | Khoản MRR@10 |
|---|---|---|---|---|---|---|---|
| Số chính (direct + paraphrase) | 90 | 1,000 | 1,000 | 0,907 | 0,989 | 1,000 | 0,864 |
| direct | 45 | 1,000 | 1,000 | 0,989 | 1,000 | 1,000 | 0,956 |
| paraphrase | 45 | 1,000 | 1,000 | 0,826 | 0,978 | 1,000 | 0,773 |
| article_ref | 12 | 0,833 | 0,917 | 0,775 | 0,833 | 0,917 | 0,775 |

- **Chạm trần ở top 5.** Cả 90 câu có Điều đúng trong top 5, nên recall@5 không còn chỗ để Tuần 4–5 chứng minh cải thiện. Chênh lệch nằm ở hạng: Điều đúng đứng hạng 1 ở 44/45 câu trực tiếp nhưng chỉ 32/45 câu đời thường (hạng 2: 6, hạng 3: 5, hạng 4: 2; tính từ `eval/runs/2026-11-22-dense-baseline.jsonl`). `run_eval` in sẵn R@1 ở cả hai mức (mục 2 của báo cáo) và phân bố hạng mức Điều cho direct và paraphrase (mục 6), nên không cần tính tay.
- **Bảng cặp (mức Điều, top 5):** cả hai trúng 45, chỉ câu trực tiếp trúng 0, chỉ câu đời thường trúng 0, cả hai trượt 0.
- **Ca Điều 35/36 vẫn còn:** q001 xếp `d36:k2` trên `d35:k1` (hạng 2), q002 ("Người lao động muốn nghỉ việc phải báo trước bao nhiêu ngày?") để Điều 35 ở hạng 3.
- **Câu nêu số Điều:** 2/12 trượt top 5 (q094 "Nội dung của Điều 140 là gì?", q095 "Điều 198 nói về vấn đề gì?"), vì vector không mang số Điều. Đây là việc của lọc metadata ở Tuần 4.
- **Điểm top-1:** trong phạm vi (102 câu) thấp nhất 0,478, trung vị 0,704, cao nhất 0,811; ngoài phạm vi (25 câu) thấp nhất 0,305, trung vị 0,475, cao nhất 0,623. 12/25 câu ngoài phạm vi có điểm cao hơn câu trong phạm vi thấp nhất, nên một ngưỡng điểm đơn lẻ không tách được hai nhóm.

## Dữ liệu

Bản hợp nhất 18/VBHN-VPQH ngày 12/02/2026 của Bộ luật Lao động 45/2019/QH14, kiểm tra hiệu lực ngày 13/11/2026. Nguồn gốc và checksum xem ở `data/raw/SOURCES.md`.

## Hạn chế đã biết

1. Chưa nạp Nghị định 145/2020/NĐ-CP.
2. Khoản quá ngắn như các khoản của Điều 124 thiếu nghĩa khi đứng riêng; sẽ so sánh chunk theo Khoản với theo Điều ở Tuần 5.
3. Khoản 1 và 2 Điều 219 chứa nguyên văn sửa đổi của luật khác, có thể gây nhiễu khi truy xuất.
4. Chưa trích dẫn chiếu giữa các Điều và chưa lưu thông tin sửa đổi vào metadata.
5. Truy xuất mới chỉ dùng dense vector. Câu hỏi về báo trước khi nghỉ việc xếp Điều 36 (quyền của người sử dụng lao động, điểm 0,7071) trên Điều 35 (quyền của người lao động, hạng 3, điểm 0,6800).
6. Chưa có ngưỡng từ chối theo điểm: `ask` vẫn đưa `k` chunk vào LLM kể cả khi điểm thấp, và việc từ chối (`insufficient`) hoàn toàn dựa vào mô hình, chưa được hiệu chỉnh. Câu hỏi ngoài phạm vi vẫn trả về 5 kết quả khi truy xuất, với điểm thấp hơn (0,3439 so với 0,7071–0,7740 của 6 câu trong phạm vi). Một câu ngoài phạm vi chưa đủ để đặt ngưỡng.
7. Máy không có GPU NVIDIA vẫn chạy được trên CPU (`EMBEDDING_DEVICE=cpu`) nhưng chậm hơn; `pyproject.toml` đang lấy PyTorch từ index `cu126`.
8. Cache embedding chỉ đi theo tên model: đổi trọng số, revision hay cách mã hóa dưới cùng tên model thì phải xóa `.cache/embeddings/<model_key>.npz` và index lại. Chỉ chạy một tiến trình index tại một thời điểm; lần index bị ngắt giữa chừng có thể để collection thiếu điểm, chạy lại lệnh index là đủ.
9. Cổng Qdrant chỉ mở trên `127.0.0.1`; dữ liệu nằm trong named volume `qdrant_storage`.
10. Trích dẫn hợp lệ (mã `[S#]` có thật, nhãn lấy từ metadata) không chứng minh câu trả lời đúng hay được nguồn ủng hộ; hệ thống không kiểm tra điều đó.
11. Cảnh báo số Điều chỉ là gợi ý đối chiếu, không phải kiểm chứng: nó so số Điều trong câu trả lời với Điều của các chunk đã truy xuất và với mọi số Điều được nhắc trong `context_text` của chúng, nên một số được nhắc trong nguồn sẽ không bị cảnh báo. Biểu thức nhận số có thể gom nhầm một số không phải số Điều vào danh sách, làm yếu cảnh báo.
12. Nguyên văn hiển thị của chunk khoản (`text`) không gồm đoạn dẫn của Điều (đoạn nằm trước khoản đầu tiên); đoạn này chỉ nằm trong `context_text` gửi cho mô hình. Hãy mở Điều đầy đủ để đọc đủ ngữ cảnh.
13. Chưa đo chất lượng câu trả lời hay độ trễ, chưa so sánh nhà cung cấp. Các lần chạy thật ngày 15/11/2026 (8 câu hỏi chạy tay qua `ask`, xem `eval/results.md`) chỉ là kiểm tra bằng mắt, không phải phép đo.
14. Một khẳng định chỉ gồm các ký tự định dạng Unicode vô hình (loại Cf) vẫn được coi là có nội dung, và các ký tự này không bị lọc khỏi khẳng định; chúng không thể thêm dòng hay tiêu đề nguồn, nhưng các ký tự điều khiển hướng văn bản (bidi, ví dụ U+202E) thuộc loại này có thể làm dòng khẳng định hiển thị sai thứ tự.
15. Khi chưa cấu hình `LLM_API_KEY`, lệnh `ask` dừng sớm với một dòng lỗi (xem phần Sinh câu trả lời); `ask` không có chế độ không cần khóa (riêng `search` vẫn chạy được mà không cần khóa).
16. Chống chèn chỉ thị: cách đóng gói JSON và prompt "dữ liệu, không phải chỉ thị" (`app/rag/prompts.py`) chỉ giảm rủi ro, không loại bỏ. Các test chỉ kiểm cấu trúc (nội dung độc không làm vỡ JSON và được giữ nguyên từng ký tự), không khẳng định mô hình không bị lừa. Chưa đo độ bền thật trước chỉ thị nằm trong câu hỏi hoặc trong văn bản truy xuất; việc đó chỉ đo được bằng eval với mô hình thật.
17. Văn bản lặp vô nghĩa: mô hình từng trả các khẳng định chỉ gồm một từ lặp lại (lần chạy thật ngày 15/11/2026). Hệ thống từ chối khẳng định có đơn vị 1–4 từ lặp liên tiếp từ 4 lần trở lên (không phân biệt hoa thường, bỏ dấu câu và ký hiệu ở hai đầu mỗi từ); câu trả lời như vậy kết thúc bằng `invalid model answer: degenerate_claim` và mã thoát 1. Lệnh không tự thử lại, nên hãy chạy lại lệnh. Ngưỡng 4 dựa trên toàn bộ 676 chunk (`text` và `context_text`), nơi đơn vị lặp liên tiếp dài nhất là 2 lần. Ba giới hạn: đơn vị lặp dài hơn 4 từ không bị phát hiện; lặp trải qua nhiều khẳng định khác nhau không bị phát hiện; đây chỉ là kiểm tra cấu trúc, không đánh giá khẳng định có đúng hay có được nguồn ủng hộ hay không.
