# Tuần 2 (nửa sau) — Sinh câu trả lời có trích dẫn: Design

**Trạng thái: thiết kế và plan Stage 2 đã được review và coordinator chấp nhận để triển khai trong tương lai.** Bản này đã sửa theo review vòng 1–2 (reviewer Opus, đối chiếu wheel `google-genai==3.3.0` chính thức). Đây không phải sign-off của người dùng hay chỉ thị triển khai: mã sản phẩm Stage 2 và việc gọi LLM thật chưa được thực hiện hoặc cho phép. Chưa gọi API, chưa dùng khóa, chưa chạy test sản phẩm khi viết tài liệu.

Plan thực thi đi kèm: `docs/superpowers/plans/2026-11-15-week2-generation.md`.

## 1. Mục tiêu, tiền điều kiện, ngoài phạm vi

**Mục tiêu.** Hoàn tất mục còn lại của Tuần 2: `generate(query, contexts, client)` và `uv run python -m app.rag.ask "câu hỏi"`. Trích dẫn Điều/Khoản do code dựng từ metadata; từ chối khi thiếu căn cứ; lỗi nhà cung cấp báo gọn.

**Tiền điều kiện (hiện trạng).** Stage 1: F1 (ranh giới lỗi CLI) đã review và commit `21c94a6` trên `week2-qdrant-retrieve` (255 unit + 9 integration pass theo coordinator); F2 (tải embedder chỉ từ cache cục bộ, lệnh tải riêng `uv run python -m ingestion.download_model`) đã được Opus review độc lập PASS, 0 blocker, và coordinator commit `6fae037`. Kiểm tra cuối của coordinator cho Stage 1: 311 unit pass (16 deselected), 16 integration pass (311 deselected); ruff check và format sạch (34 file). Search smoke giữ nguyên hai kết quả đầu và điểm, 676 point, không gọi Hugging Face Hub; chưa đo recall. Báo cáo review F2 cục bộ: `.cache/week2-final-20261115/f2-review/review-f2-closure.md`. Giao diện là `app.rag.embedder.EmbeddingModelUnavailableError`. Loader thường không gọi Hugging Face Hub; Qdrant vẫn dùng mạng cục bộ. Stage 1 đã chốt tại hai commit trên; chưa có PR hay merge được báo cáo. Chưa có mã Stage 2; khi có chỉ thị triển khai, Stage 2 bắt đầu từ base đã duyệt có F1 và F2. Giao diện F1 đã ổn định: `InvalidCollectionDataError`, `CollectionNotIndexedError` (cả collection chưa tạo lẫn rỗng), `rejection_message(error, *, searching)`, `prepare_query(query, k)`.

**Ngoài phạm vi.** SSE (Tuần 6), FastAPI, web, hybrid, reranker, ngưỡng điểm đã hiệu chỉnh (Tuần 3), bộ câu hỏi eval mới, retry tự động, fallback trả phí, nhiều adapter hay registry provider, hội thoại nhiều lượt, tool calling, metric recall.

## 2. Provider: khuyến nghị của coordinator

Người dùng giao quyền chọn provider cho coordinator và Fable 5.2. Đây là **khuyến nghị**; triển khai cần chỉ thị mới.

- **Gemini `gemini-3.5-flash-lite`, một adapter `GeminiClient` sau Protocol `LLMClient`.** Model ghi `Stable`, hỗ trợ structured output ([model](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite), cập nhật 07/2026).
- **Groq `openai/gpt-oss-120b`: chỉ là phương án thay thế ghi trong tài liệu**, không adapter, không fallback, không registry. Lý do Fable nêu: structured output của Groq không hỗ trợ streaming (Tuần 6 cần SSE) và 8K TPM chật cho eval; chưa kiểm lại độc lập.
- **Chưa khẳng định** chất lượng trả lời luật tiếng Việt hay độ trễ của provider nào; so sánh bằng eval Tuần 3–5.
- Giá chỉ để tham khảo ngày 15/11/2026 ([pricing](https://aistudio.google.com/docs/pricing)): free tier có sẵn, trả phí 0,30 / 2,50 USD mỗi 1M token vào / ra; chưa kiểm hạn mức tài khoản.
- Model ID ghi hai chỗ: `requested_model` (giá trị `LLM_MODEL`) và `response_model` (`Interaction.model` do server trả, có thể vắng); hai giá trị khác nhau về nghĩa.

## 3. Hợp đồng SDK (đã xác minh từ wheel, ngày 15/11/2026)

Nguồn: wheel `google_genai-3.3.0-py3-none-any.whl`, sha256 `86bd1830fec4b12e474237414fa52383aa0f49458910f6c9c938c207715559b5` (khớp [PyPI JSON](https://pypi.org/pypi/google-genai/json); 3.3.0 phát hành 12/11/2026, Python >= 3.11); [changelog](https://raw.githubusercontent.com/googleapis/python-genai/v3.3.0/CHANGELOG.md); [api-versions](https://aistudio.google.com/docs/api-versions); [structured-output](https://aistudio.google.com/docs/structured-output); [interactions](https://aistudio.google.com/docs/interactions). Đường dẫn dưới đây tính từ `google/genai/` trong wheel.

- **Ghim `google-genai==3.3.0`**: bản mới nhất, không có thay đổi Interactions phá vỡ sau 3.0.0, và 3.0.0 sửa "Retries = attempts - 1" (lý do không dùng 2.x). Phụ thuộc khai báo đã thỏa bởi `uv.lock` hiện có (pydantic, httpx2, anyio, typing-extensions); gói mới: `google-auth`, `requests`, `tenacity`, `websockets`, `distro`, `sniffio`. `uv lock` thật chạy ở Task 1.
- **Gọi:** `client.interactions.create(**body)` (`_gaos/google_genai.py:269`). Khóa body không hợp lệ ném `TypeError` (`:1775`). Thân gồm `model`, `input`, `system_instruction: str`, `response_format` (dict `{"type": "text", "mime_type": "application/json", "schema": ...}`), `generation_config` (`max_output_tokens`, `thinking_level` gồm `minimal`..`xhigh`; `temperature` ghi DEPRECATED nên **không đặt**), `store`. `minLength`/`maxLength` bị bỏ qua, `minItems`/`maxItems` được ép (`_gaos/types/interactions/textresponseformat.py:60`).
- **Một request duy nhất (liên quan hóa đơn):** mặc định SDK có tổng cộng 5 lần thử (1 request đầu + tối đa 4 retry) cho mã 408/409/429/5XX (`_api_client.py:674`, `_gaos/google_genai.py:173`, `_gaos/interactions.py:538`) **và** `automatic_continuation=True` có thể gửi thêm các bước tính tiền (`_gaos/google_genai.py:297-360`). Đặt `HttpRetryOptions(attempts=1)` (SDK test `tests/interactions/test_retry_options.py:93`) và `automatic_continuation=False`; khi đó kết quả `requires_continuation` chỉ phát một `UserWarning` ("Interaction stopped before completion"), adapter lọc cảnh báo này.
- **Timeout:** `HttpOptions.timeout` là **mili giây** (`types.py:2778`); tham số `timeout=` mỗi lần gọi là giây.
- **Phiên bản API và endpoint:** `v1` là mặc định của SDK (`_api_client.py:151`) và Interactions GA ở `v1` theo trang api-versions; nhưng SDK đọc các biến `GOOGLE_GENAI_API_VERSION`, `GOOGLE_GEMINI_BASE_URL`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`. Vì vậy adapter đặt tường minh `api_version="v1"`, `base_url`, `vertexai=False`, `api_key=`. Đường dẫn là `/v1/interactions`, khóa gửi ở header `x-goog-api-key`. Việc `v1` chấp nhận model này trên dịch vụ thật chỉ được chứng minh bằng smoke.
- **Phản hồi** (`_gaos/types/interactions/interaction.py:269,335,344`): `Interaction.status` (`completed`, `incomplete`, `failed`, `requires_continuation`, ...), `model` (tùy chọn), `output_text` (chỉ có khi không rỗng), `usage`. **Không có** trường `finish_reason` hay lý do chặn. Quy tắc dùng được: `status == "completed"` **và** `output_text` không rỗng; còn lại là `LLMBadResponseError`.
- **Lỗi của Interactions** không kế thừa `google.genai.errors.APIError`: gốc `GeminiNextGenAPIClientError`, gồm `APIStatusError` (có `status_code`), `APIConnectionError`, `APITimeoutError`, `APIResponseValidationError`, bề mặt công khai `google.genai._interactions` (`_gaos/lib/compat_errors.py`). `str(error)` chứa thân phản hồi (`Error code: {status} - {body}`), nên **không bao giờ in** và dịch bằng `raise ... from None`.

## 4. Kiến trúc

Chi tiết SDK chỉ nằm trong `app/rag/llm.py`.

- `app/core/config.py`: `llm_provider`, `llm_api_key: SecretStr | None`, `llm_model` (`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`).
- `app/rag/llm.py`: `LLMRequest`, `LLMResult`, `LLMClient`, họ lỗi `LLMError`, `GeminiClient`, `build_llm_client`.
- `app/rag/prompts.py`: nhãn `S1..Sn`, schema phản hồi, `SYSTEM_PROMPT`, thông điệp JSON.
- `app/rag/generate.py`: kiểm phản hồi, nhãn Điều/Khoản, cảnh báo số Điều, `generate`, `render_result`.
- `app/rag/ask.py`: CLI; import lỗi và `rejection_message` từ `app.rag.retriever` / `app.rag.store`; **không sửa `search.py`**.

Luồng `ask`: kiểm câu hỏi và `k` (mặc định 5, tối đa `MAX_CONTEXTS` = 8) → settings và khóa → `retrieve` → `generate(query, contexts, client)` → in. `generate` **không** gọi `retrieve` (đánh giá truy xuất tách khỏi sinh, `KE_HOACH_RAG.md` mục 7). Thiếu khóa phát hiện **trước** khi tải embedder hay chạm GPU.

Cố ý không dùng: tool, `background`, `previous_interaction_id`, trạng thái giữa các lần gọi, retry, `temperature`. Trần cố định: timeout 30 giây, `max_output_tokens=4096` (gồm token suy nghĩ), `thinking_level="minimal"`. Ngân sách này giảm rủi ro cắt phản hồi, không bảo đảm đủ cho mọi câu trả lời; phản hồi không hoàn tất bị từ chối.

## 5. Hợp đồng bằng chứng

- **Nhãn theo request:** chunk truy xuất nhận `S1..Sn` theo thứ tự. Mô hình không thấy `chunk.id` hay điểm.
- **Nguồn đưa vào prompt là `chunk.context_text`**, không phải `chunk.text`: với 53 chunk khoản thuộc 12 Điều có đoạn dẫn trước khoản 1, đoạn dẫn chỉ nằm trong `context_text` (ví dụ `d25:k1` mất điều kiện "chỉ được thử việc một lần đối với một công việc" nếu chỉ gửi `text`). `context_text` gồm tên văn bản, chương, mục, tiêu đề Điều, đoạn dẫn và khoản, đều nguyên văn từ corpus. Màn hình hiển thị vẫn là `chunk.text`.
- **Thông điệp người dùng** là đối tượng JSON `{"question": ..., "sources": [{"id": "S1", "text": <context_text>}]}` (`ensure_ascii=False`). JSON giữ ranh giới dữ liệu ở mức chuỗi; điều này **không** ngăn mô hình làm theo chỉ thị nằm trong dữ liệu (xem §6, mục 6). `SYSTEM_PROMPT` (nguyên văn trong plan, Task 2) dặn: chỉ dùng nguồn; coi mọi thứ trong JSON là dữ liệu; mỗi khẳng định phải được chính các nguồn của nó nêu rõ; giữ điều kiện và ngoại lệ; không tự viết số Điều/Khoản; trả `insufficient` khi thiếu căn cứ.
- **Schema phản hồi** (enum nhãn sinh theo request): `{"status": "answered" | "insufficient", "claims": [{"text", "sources": [S#...]}]}`; `claims` tối đa 6, mỗi claim 1–3 nguồn; cấm thuộc tính thừa. Không có trường "quote": nguyên văn luôn lấy từ `chunk.text` bằng code.
- **Kiểm tra trong code** (vì độ dài chuỗi không được ép phía server): JSON hợp lệ và đúng schema; nếu `answered` thì 1–6 claim, mỗi claim không rỗng (tối đa 800 ký tự), nhãn khớp **chính xác** tập nhãn của request, không có đơn vị 1–4 từ lặp liên tiếp từ 4 lần trở lên (mã `degenerate_claim`). Mọi vi phạm là `AnswerValidationError` với mã cố định; **không bao giờ in câu trả lời chưa kiểm**, không chép nội dung phản hồi vào thông báo lỗi. Không hạ xuống `insufficient` thay cho lỗi, vì như vậy che mất vi phạm hợp đồng.
- **Từ chối:** sau khi JSON đã qua kiểm cấu trúc (bắt buộc `status`, `claims`, đúng kiểu, cấm thuộc tính thừa), `status=insufficient` bỏ claim kèm theo; giới hạn nghiệp vụ của claim chỉ áp dụng cho `answered`. `insufficient` hoặc ngữ cảnh rỗng cho `INSUFFICIENT_TEXT` cố định do code viết, không phải kết luận pháp lý; `claims`, `sources`, `unsupported_articles` đều rỗng; ngữ cảnh rỗng **không gọi LLM**. Từ chối là kết quả hợp lệ, mã 0.
- **Nhãn Điều/Khoản do code dựng từ `Chunk`:** `Điều {article}` cộng `, khoản {clause}` khi có khoản; kèm `chunk.text` nguyên văn.
- **Cảnh báo số Điều (theo `KE_HOACH_RAG.md` mục 7, bắt buộc):** dùng chung `article_mentions(text) -> set[int]` cho lời mô hình và `context_text`: không phân biệt hoa/thường, nhận danh sách số sau “Điều/điều” phân cách bằng dấu phẩy, “và”, “hoặc”; lấy mọi số trong nhóm danh sách. Chuẩn hóa claim về NFC trước kiểm tra và hiển thị. Chỉ cảnh báo các số không thuộc `{article của các chunk truy xuất} ∪ {article_mentions(context_text) của chúng}`. Vì vậy `d34:k1` cho phép 177; `d39` có “điều 35, 36 và 37” nên claim “Điều 36” không cảnh báo; claim “điều 200” với chỉ `d34:k1` thì cảnh báo. Regex có thể gom nhầm một số không phải số Điều vào danh sách, làm yếu cảnh báo; đây vẫn chỉ là cảnh báo, không chặn và **không phải kiểm chứng entailment**.

## 6. Mức bảo đảm và giới hạn (ghi vào README)

1. **Tồn tại (code kiểm):** nhãn nguồn thuộc tập đã truy xuất; sai thì không in.
2. **Nguyên văn (cấu trúc):** đoạn hiển thị là `chunk.text`, không phải lời mô hình. `chunk.text` của chunk khoản **không** gồm đoạn dẫn của Điều; người đọc nên mở Điều đầy đủ.
3. **Suy ra được (entailment) và diễn giải pháp lý đúng: code không kiểm được.** Trích dẫn hợp lệ không chứng minh nguồn ủng hộ khẳng định. Cần người đánh giá (chấm tay 30 câu hoặc RAGAS, Tuần 5). Output và README không dùng cụm "đã xác minh".
4. **Dẫn chiếu chéo:** Điều 34 khoản 1 dẫn khoản 4 Điều 177; Điều 219 chứa nguyên văn luật khác. Không dùng regex cấm số Điều; chỉ có cảnh báo ở §5 và việc đánh giá bởi người.
5. **Từ chối dựa vào mô hình, chưa hiệu chỉnh.** Không đặt ngưỡng điểm từ một câu ngoài phạm vi. `GenerationResult` giữ toàn bộ chunk truy xuất kèm điểm cho Tuần 3.
6. **Chống chèn chỉ thị:** prompt và cách đóng gói JSON giảm rủi ro, không loại bỏ. Test chỉ kiểm cấu trúc (nội dung độc không làm vỡ JSON và được giữ nguyên từng ký tự); **không** khẳng định mô hình không bị lừa. Độ bền thật chỉ đo bằng eval có mô hình thật.

## 7. Lỗi và CLI

- Thứ tự: câu hỏi và `k` → settings có khóa → truy xuất → LLM. Thiếu hoặc rỗng khóa: `LLMConfigError("LLM_API_KEY is not configured")`; provider không hỗ trợ cũng là `LLMConfigError`.
- Mọi lỗi vận hành của `ask`: **một dòng stderr, stdout trống, mã 1**; từ chối mã 0; lỗi lập trình vẫn lan ra. Danh sách bắt: `InvalidQueryError`; `ValidationError` (chỉ quanh `get_settings()`, qua `settings_error_line`: chỉ tên biến, không in giá trị vì có thể là bí mật); `LLMError` (`LLM: {message}`); `QdrantUnavailableError`, `CollectionNotIndexedError`, `InvalidCollectionDataError` (`str(error)`, giống `search`); `UnexpectedResponse` (`rejection_message(error, searching=True)`, giống `search`); `EmbeddingModelUnavailableError` (`str(error)`, từ `app.rag.embedder`); `GenerationInputError`; `AnswerValidationError` (`invalid model answer: {code}`). Dòng cài đặt sai của `ask` là dòng **mới**; `search` giữ nguyên (việc dọn sau, ngoài Stage 2).
- Lỗi LLM chỉ in loại lỗi và mã HTTP: 401/403 `LLMAuthError`; 429 `LLMRateLimitError` ("thử lại sau", không retry); 400, 404, 4xx khác `LLMRequestError` (400 gợi ý kiểm `LLM_API_KEY` và `LLM_MODEL`, vì khóa sai có thể trả 400); timeout, mất kết nối, 5xx, lỗi SDK khác `LLMUnavailableError`; phản hồi không hợp lệ hoặc không `completed` `LLMBadResponseError`.
- **Chỉ mục:** `CollectionNotIndexedError` (chưa tạo hoặc rỗng) là lỗi vận hành mã 1, không gọi LLM. `retrieve` trả `[]` (chỉ còn là ca thư viện) thì `generate` từ chối cố định không gọi LLM, mã 0. Hai ca khác nhau có chủ đích: chưa có dữ liệu là lỗi vận hành, không có căn cứ cho một câu hỏi là kết quả hợp lệ.
- Khóa: `SecretStr`, chỉ `get_secret_value()` bên trong `build_llm_client`. Không in vào log, output, test, commit hay brief; không xin khóa trong chat.

## 8. Kiểm thử: ba tầng

1. **Đơn vị tất định** (mặc định `uv run pytest`): `FakeLLMClient`, hàm thuần dựng request và đọc response, chunk lấy từ `chunks.jsonl`, câu trả lời giả không chứa nội dung luật, ngoại trừ dẫn chiếu lấy từ corpus trong test cảnh báo. Fixture `Interaction` dựng trực tiếp từ kiểu SDK, không mạng.
2. **Contract adapter:** server `127.0.0.1` do test dựng (có `request_count`, `last_path`, `last_headers`, `last_body` JSON); một ca 200 với `steps=[{type: model_output, content: [{type: text, text: ...}]}]` kiểm text/model và thân request; mã 400/401/403/404/429/500/503, timeout, kết nối bị từ chối, `requires_continuation`; khẳng định `request_count == 1` (retry và continuation tắt), đường dẫn `/v1/interactions`, thân lỗi và khóa giả không xuất hiện trong lỗi. Fixture xóa biến môi trường của Google và proxy.
3. **Smoke thật, chỉ khi người dùng cho phép trong phiên.** Hai cổng: marker `llm` bị loại bởi `addopts` (`-m 'not integration and not llm'`), và `skipif(RAG_LIVE_LLM != "1")` ở cấp module; khóa đọc từ `Settings()` (môi trường hoặc `.env`), thiếu thì `pytest.skip`. Chạy mặc định cho "2 deselected" và mã 5 (pytest không có test), `-m llm` khi chưa đặt `RAG_LIVE_LLM` cho "2 skipped" mã 0; `-m integration` không chọn test `llm`. Cận trên: 2 test × đúng 1 HTTP request.
   - **Điều kiện trước khi dùng thật** (cổng sử dụng, không chặn việc viết plan hay mã): người dùng cho phép gọi trả phí; hạn mức free tier của tài khoản (chỉ xem được trong AI Studio); chấp nhận chính sách dữ liệu: trang pricing ghi "Used to improve our products": free **Yes**, paid **No**. `store=False` chỉ quyết định Google có giữ interaction để truy vấn lại hay không (free 1 ngày, paid 55 ngày), **không** chứng minh việc loại trừ khỏi cải thiện sản phẩm. Smoke chỉ dùng văn bản luật công khai không sửa và câu hỏi tổng hợp; README cảnh báo không nhập thông tin cá nhân.

Test đơn vị luôn xóa biến `LLM_*` và đặt `Settings.model_config["env_file"]` thành `None`, trừ test `llm`. Khẳng định hiện có ở `tests/test_config.py` rằng settings không có `llm_api_key` phải được sửa có chủ đích.

## 9. Ca biên lấy từ `chunks.jsonl` (đã mở file; 676 chunk, đều NFC)

- Chunk không có khoản: `45-2019-QH14:d1` (Điều 1, 317 ký tự) và 36 chunk khác (tổng 37). Nhãn `Điều 1`, không "khoản None".
- Nhãn khoản chữ-số: `45-2019-QH14:d154:k8a` (duy nhất). Nhãn `Điều 154, khoản 8a`.
- Dẫn chiếu chéo: `45-2019-QH14:d34:k1` ("khoản 4 Điều 177 của Bộ luật này"); nhãn là `Điều 34, khoản 1`, không phải Điều 177; số 177 hợp lệ ở cảnh báo §5.
- Đoạn dẫn của Điều: `45-2019-QH14:d25:k1`, `context_text` có câu dẫn, `text` thì không.
- Chunk chứa luật khác và dài nhất: `45-2019-QH14:d219:k1` (4.512 ký tự); nguyên văn hiển thị đủ.
- Tám chunk dài nhất: 14.762 ký tự `text`, 15.905 ký tự `context_text`; ngân sách ngữ cảnh không đổi đáng kể.

## 10. Eval gate và ranh giới PR

- Task 2–5 chạy `uv run python eval/run_eval.py` trước và sau, thêm một dòng đủ 6 cột `Ngày | Thay đổi | recall@5 | recall@10 | MRR | Ghi chú` vào `eval/results.md`, ngày là ngày thực thi, ba metric = `N/A`, ghi chú `chưa đo recall`. Không cài metric, không bịa số, không ghi kết quả smoke là recall.
- Theo chỉ thị routing mới nhất của người dùng cho công việc tương lai: implementer mặc định Codex `gpt-6.1-sol`, effort `high` hoặc cao hơn; reviewer Codex độc lập, mạnh hơn (`gpt-6-astra`, effort `xhigh`); coordinator/root tổng hợp bằng chứng và giao sửa tiếp khi cần. Giữ Fable 5.2 cho task Khó/phức tạp và thiết kế, phân tích sâu, gồm bước thiết kế Task 2–3. Đây là chỉ thị của phiên, không thay đổi quy tắc chung trong `AGENTS.md`; các review Opus ở trên là bằng chứng lịch sử. Mỗi task Trung bình/Khó tối đa hai vòng sửa; hết hai vòng còn blocker: Trung bình nâng lên Khó (thêm thiết kế và brief mới), Khó dừng và báo người dùng theo `AGENTS.md`. Task chạy tuần tự theo phụ thuộc, không sang task sau khi còn blocker. Không ấn định số commit. Push và mở PR cần chỉ thị rõ.
- Với quyết định thật sự thuộc người dùng, coordinator gửi đồng thời cùng một bản tóm tắt ngắn và 2–3 phương án cho người dùng và Fable 5.2; thao tác thường ngày đã được ủy quyền không thành cổng duyệt mới. Giữ tư vấn provider trước đây của Fable; không mở lại thiết kế provider, SDK hay prompt.
- Không còn câu hỏi API hay tên giao diện chờ quyết định. F2 đã hoàn tất tại `6fae037`; khi được chỉ thị triển khai, dùng base có cả F1 và F2; `uv lock` thật và smoke thật chỉ có bằng chứng khi thực thi, không phải kết quả của tài liệu này.
