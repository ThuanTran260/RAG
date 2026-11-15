# Tuần 2 (nửa đầu) — Qdrant và `retrieve`: Implementation Plan

> **For agentic workers:** Plan này được thực thi qua các worker do Orca quản lý, từng task tuần tự (xem mục "Quyết định thực thi"); ưu tiên điều phối của người dùng thắng hướng dẫn tạo sub-agent native của bất kỳ skill nào. `superpowers:executing-plans` chỉ dùng để hướng dẫn cách làm việc của một worker trong task của nó. Các bước dùng checkbox (`- [ ]`) để theo dõi.

**Goal:** Nạp 676 chunk của Bộ luật Lao động vào Qdrant bằng embedding bge-m3 và có hàm `retrieve(query, k)` trả về các Điều/Khoản gần nghĩa nhất, chạy được từ terminal.

**Architecture:** Embedding nằm sau một interface nhỏ (`Embedder`) để đổi model và thay bằng bản giả trong test. Lệnh `ingestion.index` embedding `context_text` của từng chunk (có cache ra đĩa) rồi dựng lại collection Qdrant; mỗi model có collection và file cache riêng, đặt tên bằng `model_key(model_name)`. `Retriever` embedding câu hỏi và hỏi Qdrant; test đơn vị dùng Qdrant chế độ trong bộ nhớ và model giả nên không cần Docker, GPU hay mạng.

**Tech Stack:** Python 3.13, uv, PyTorch 2.14.1 (bản CUDA `cu126`), sentence-transformers 6.3.0, qdrant-client 1.20.0, numpy 2.5.4, Qdrant server v1.20.1 (Docker), pydantic-settings 2, pytest, ruff.

**Spec:** `KE_HOACH_RAG.md` (mục 3, mục 5 "Tuần 2", mục 6, mục 7 "Về truy xuất" và "Về kỹ thuật và chi phí") cùng `AGENTS.md`.

## Trạng thái của plan này

- Bản chỉnh ngày 14/11/2026. Người dùng đã chọn "chỉnh plan và chốt phương án trước", tức là **chỉ** duyệt việc chỉnh plan và việc đưa ra khuyến nghị thực thi. Ngày 15/11/2026, sau khi người điều phối đối chiếu lại hiện trạng, người dùng ra lệnh thực thi phương án (a): worker Orca, model theo độ phức tạp của task (bảng tier ở `CLAUDE.md`). Branch `week2-qdrant-retrieve` đã được người điều phối tạo từ `6b23662` trước khi giao Task 1.
- Chưa có mã hoặc test nào của Tuần 2. Hiện trạng đã kiểm tra: `uv run --no-sync pytest -q` ra 44 passed (toàn bộ là test của Tuần 1).
- Các khối test trong plan **chưa được chạy với mã sản phẩm**. Chúng mới chỉ được kiểm tra cú pháp tĩnh (`ast.parse`) và định dạng; việc đó không chứng minh test đúng hay mã đúng.

## Quyết định thực thi

Ba cách thực thi đã được so sánh:

- **(a) Tuần tự từng task trên Orca, mỗi task một reviewer Opus mới.** Chuỗi phụ thuộc thật là 1 → 2 → 3 → 4 → 5 → 6: Task 2 dùng `Settings` và `model_key` của Task 1; Task 3 dùng `Embedder` và fixture của Task 2; Task 4 dùng cache của Task 3; Task 5 dùng `store` và `index_chunks` của Task 4; Task 6 chạy mọi thứ trên Qdrant thật. Lỗi ở task trước bị bắt trước khi task sau xây lên nó.
- **(b) Song song bằng worktree.** Không hợp ở quy mô này: các task dùng chung `pyproject.toml`, `tests/conftest.py`, `eval/results.md`, cache embedding, GPU và một Qdrant; chuỗi phụ thuộc không có nhánh độc lập; thêm chi phí gộp xung đột. Plan còn là file chưa được track cho tới khi Task 1 commit, nên worktree mới sẽ không có nó.
- **(c) Một implementer native làm cả sáu task rồi một review cuối.** Rẻ nhất, nhưng lỗi sớm (ví dụ trùng khóa cache, validate thiếu) chỉ lộ ở review cuối, khi năm task đã xây trên nó.

**Khuyến nghị: (a).** Quy tắc khi thực thi (chỉ áp dụng sau khi người dùng ra lệnh):

1. Chạy tuần tự trong cùng một workspace và cùng một venv, branch `week2-qdrant-retrieve`, với `--worktree current`.
2. Mỗi task: một implementer Sonnet (effort high), rồi một reviewer Opus có ngữ cảnh mới. Reviewer kiểm cả đúng spec lẫn đúng chuẩn dự án (`AGENTS.md`, ràng buộc cứng, ruff), trả lỗi kèm `file:line`, mỗi lỗi đánh dấu blocker hoặc nit, và một verdict cho task. Reviewer của Task 6 còn kiểm tính tương thích của các interface cộng dồn từ Task 1–5. Không thêm một agent review toàn branch riêng.
3. Không sang task sau khi task hiện tại còn blocker. Tối đa hai vòng sửa; hết hai vòng mà còn blocker thì nâng tier theo `AGENTS.md` (viết lại brief có bước thiết kế), nếu đã ở tier Khó thì dừng và báo người dùng kèm lỗi cụ thể. Mỗi task có ít nhất một commit gắn kết; commit sửa theo review được phép, không cần viết lại lịch sử (commit cục bộ, chưa push). Về nit, theo `AGENTS.md`, người điều phối chỉ tự sửa thay đổi dưới một dòng và không đổi hành vi; sửa hành vi vẫn do implementer làm.
4. Brief của mỗi task gồm năm phần, lấy từ chính phần task trong plan: mục tiêu, file được sửa, tiêu chí nghiệm thu, lệnh kiểm tra, ngoài phạm vi.
5. Orca `worker-start` tạo và quản lý workspace cho worker; nó không thay thế sub-agent native của Claude Code. Chọn agent, model, effort theo bảng ở `CLAUDE.md` khi tạo worker; nếu Orca không cho chọn đúng thì báo người điều phối, không tự đổi.
6. **Giao thức quyết định.** Chỉ khi gặp một lựa chọn thực sự chưa có trong plan và cần người dùng quyết (không phải chi tiết kỹ thuật implementer tự chọn được), người điều phối gửi **một** bản tóm tắt quyết định gồm dữ kiện và 2–3 phương án, đồng thời cho người dùng và cho một advisor Fable 5.2 chỉ đọc, rồi trình cả hai ý kiến cho người dùng. Người dùng giữ quyền quyết định; advisor chỉ tư vấn, không phê duyệt thay, không thay review Opus của từng task và không cấp quyền thực thi sản phẩm. Các thao tác thường ngày có thể hoàn tác hoặc đã được ủy quyền trong plan không trở thành cổng duyệt mới.

## Global Constraints

- Repo công khai. Không commit `.env` và `.cache/` (cả hai đã có trong `.gitignore`). Không `git add` file trong `data/raw/` và `.env` khi chưa có xác nhận của người dùng trong phiên hiện tại. Trước mỗi commit chạy `git status --short` và đối chiếu từng file với danh sách file của task.
- Khóa API và mọi cấu hình chỉ đọc từ biến môi trường hoặc `.env`; tên biến khai báo trong `.env.example`. Giá trị khóa không xuất hiện trong log, output, commit hay brief.
- Không dùng LangChain/LlamaIndex.
- Mọi số Điều, Khoản và nội dung luật trong test lấy từ `data/processed/chunks.jsonl` (mở file và trích, không viết theo trí nhớ). Test về cache và các cơ chế không liên quan nội dung luật dùng chuỗi tổng hợp trung tính, không bịa nội dung sửa đổi luật.
- Text tiếng Việt chuẩn hóa NFC; đọc ghi file với `encoding="utf-8"`; lệnh in tiếng Việt phải chạy được khi console không dùng UTF-8.
- Cache embedding ra đĩa để không tính lại mỗi lần chạy.
- Comment trong code, commit message, tên branch viết bằng tiếng Anh.
- Làm trên branch `week2-qdrant-retrieve`, mỗi task có ít nhất một commit gắn kết (commit sửa theo review được phép), không push.
- Lệnh kiểm tra chung của mọi task: `uv run pytest`, `uv run ruff check app ingestion tests`, `uv run ruff format --check app ingestion tests`. Khối test chép từ plan có thể cần `uv run ruff format app ingestion tests` để đúng định dạng; chỉ định dạng, không đổi assertion.
- **Test trong plan là điểm xuất phát, không phải bất biến.** Giữ các assertion có cơ sở. Được sửa test khi chạy thật cho thấy test sai, khi ruff yêu cầu, hoặc khi reviewer ghi nhận blocker; mỗi thay đổi phải nêu trong báo cáo. Được thêm test có ý nghĩa. Không xóa hay làm yếu assertion chỉ để test qua.
- **Số test là cận dưới suy ra từ plan, không phải con số chính xác hay trần.** Mỗi task nêu số test tối thiểu của file test mới (đếm cả các ca `parametrize`), kiểm bằng `uv run pytest <file> --collect-only -q`; điều kiện qua là toàn bộ test pass. Con số cũ 85 unit + 9 integration của bản plan đầu chỉ là kế hoạch, không phải kết quả đã chạy.
- **Eval gate** (`AGENTS.md`): `eval/run_eval.py` hiện là placeholder không in số liệu, và `eval/results.md` ghi chưa có baseline. Task 2–5 (chạm embedding, cache, index, truy xuất) chạy `uv run python eval/run_eval.py` trước khi sửa và sau khi xong (đều phải thoát mã 0 và không in số liệu), rồi thêm một dòng vào bảng trong `eval/results.md` với recall@5 = recall@10 = MRR = `N/A` và ghi chú `chưa đo recall` (có dấu). Bảng do Task 2 tạo, các task sau thêm dòng. Task 1 không chạm truy xuất nên không thêm dòng. Task 6 ghi kết quả chạy thật, không ghi là recall. Không viết số liệu recall, không sửa `eval/questions.jsonl`, không cài đặt metric (việc của Tuần 3).
- Dev đơn lẻ trên một máy: chỉ một tiến trình `ingestion.index` ghi tại một thời điểm (không có khóa liên tiến trình).

## Review Focus

Năm tình huống dễ gây lỗi nhất khi dùng thật, mỗi tình huống có test ở task nêu trong ngoặc:

1. **Qdrant chưa chạy hoặc chết giữa chừng**: lệnh index và lệnh search phải thoát mã 1 với một thông báo ngắn, không in traceback, không in gì ra stdout, và không tải model khi chưa cần. (Task 4, Task 5)
2. **Chưa nạp dữ liệu, đổi model chưa nạp lại, hoặc hai tên model có cùng slug**: `retrieve` báo lỗi nêu rõ lệnh cần chạy, không trả kết quả của model khác; tên collection và file cache không đụng nhau. (Task 1, Task 5)
3. **Câu hỏi bất thường**: rỗng, toàn khoảng trắng, Unicode tổ hợp (NFD), `k` nhỏ hơn 1, `k` lớn hơn số chunk; kiểm tra trước khi kết nối hay tải model. (Task 5)
4. **Cache cũ, hỏng hoặc bị cắt cụt, chứa NaN/Inf, sai kiểu, sai model, sai số chiều, hoặc ghi dở**: phải tính lại, không nạp vector sai, và lần ghi lỗi không phá cache cũ. (Task 3)
5. **Chạy lại index, hoặc nạp đầu vào hỏng** (vector sai hình dạng, NaN, trùng ID): không nhân đôi dữ liệu và không xóa collection đang dùng được. (Task 4)

## Cấu trúc file

| File | Trách nhiệm |
|---|---|
| `pyproject.toml`, `uv.lock` | Phụ thuộc, index PyTorch bản CUDA, marker pytest |
| `docker-compose.yml` | Service `qdrant`, chỉ mở cổng trên `127.0.0.1` |
| `.env.example` | Thêm hai biến tùy chọn cho embedding |
| `app/core/config.py` | `Settings`, `get_settings`, `model_slug`, `model_key`, `REPO_ROOT` |
| `app/rag/embedder.py` | Interface `Embedder`, `SentenceTransformerEmbedder`, `build_embedder` |
| `app/rag/store.py` | `connect` tới Qdrant, `qdrant_errors`, `QdrantUnavailableError` |
| `app/rag/retriever.py` | `Retriever`, `RetrievedChunk`, hàm `retrieve(query, k)` |
| `app/rag/search.py` | Lệnh tìm kiếm trên terminal |
| `ingestion/embed_cache.py` | Cache embedding theo `model_key` và theo nội dung |
| `ingestion/index.py` | Lệnh nạp chunk vào Qdrant |
| `tests/conftest.py` | Cô lập môi trường, `FakeEmbedder`, fixture đọc chunk thật |
| `tests/test_*.py` | Test đơn vị, không cần Docker, model hay mạng |
| `tests/test_integration_*.py` | Test cần model thật (`model`) hoặc Qdrant thật (`server`), đều có marker `integration` |
| `eval/results.md` | Mỗi task 2–6 thêm một dòng `N/A` (chưa đo recall) |

## Kiểm tra và quan sát

**Đã kiểm tra ngày 14/11/2026 (người điều phối):**

- `uv run --no-sync pytest -q`: 44 passed trong 1,96 giây (test của Tuần 1).
- `data/processed/chunks.jsonl` có 676 chunk. Văn bản gốc và `data/raw/SOURCES.md` đã được track.
- Docker đang chạy (`docker info --format '{{.ServerVersion}}'` ra `29.9.3`); GPU RTX 4070, driver 566.36; uv 0.12.23; `.python-version` là 3.13.
- Tài liệu chính thức PyTorch có bản torch 2.14.1 cu126: https://pytorch.org/get-started/previous-versions/. PyPI có sentence-transformers 6.3.0 và qdrant-client 1.20.0; Qdrant có bản v1.20.1.
- `SentenceTransformer.get_embedding_dimension()` có tồn tại và có thể trả về `None`: https://sbert.net/docs/package_reference/sentence_transformer/model.html
- Qdrant nói rõ trên Windows có thể cần named volume thay vì thư mục cục bộ: https://qdrant.tech/documentation/quickstart/
- Docker publish cổng không chỉ định IP sẽ mở trên mọi giao diện của máy: https://docs.docker.com/engine/network/port-publishing/
- `qdrant-client` 1.20.0 (đọc từ mã nguồn của wheel): lỗi truyền tải qua REST bị bọc thành `qdrant_client.http.exceptions.ResponseHandlingException`; phản hồi HTTP lỗi thành `qdrant_client.http.exceptions.UnexpectedResponse` (có `status_code`). Cả hai kế thừa `ApiException`. Plan dùng REST (không bật `prefer_grpc`). Sau Task 1, implementer xác nhận bằng `uv run python -c "from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse"`; nếu import lỗi thì dừng và báo người điều phối, không tự đặt tên khác.
- `QdrantClient(":memory:")` sau khi `close()` sẽ ném `RuntimeError`; vì vậy `main` của các lệnh không đóng client được tiêm trong test.
- `eval/run_eval.py` chạy thoát mã 0 và không in số liệu; `eval/results.md` ghi chưa có baseline.
- `model_slug` hiện nay ánh xạ `org/a.b`, `org/a-b`, `org/a_b` về cùng `org-a-b`; vì vậy slug không đủ làm định danh collection và cache (xem Task 1).

**Quan sát lịch sử, bản chạy thử ngày 13/11/2026** (do bản chạy thử báo cáo, chưa đo lại; `data/processed/chunks.jsonl` của commit `6b23662`). Dùng làm tham chiếu để so sánh, không dùng làm điều kiện nghiệm thu chính xác:

- Phiên bản chạy được cùng nhau: torch `2.14.1+cu126`, sentence-transformers `6.3.0`, qdrant-client `1.20.0`, numpy `2.5.4`, Python 3.13; `torch.cuda.is_available()` là `True`.
- bge-m3: vector 1.024 chiều, độ dài tối đa 8.192 token. `context_text` dài nhất là 1.259 token (khoản 1 Điều 219), nên không chunk nào bị cắt.
- Embedding 676 chunk mất khoảng 6 giây trên GPU, đỉnh VRAM khoảng 3 GB; tải model lần đầu khoảng 2,3 GB. Tốc độ và CUDA chưa được đo lại ngày 14/11.
- Hai lần embedding cùng một text lệch nhau cỡ 1e-6. Vì vậy test với model thật chỉ so bằng dung sai.
- Qdrant từ chối ID điểm dạng `45-2019-QH14:d1`; ID phải là số nguyên không dấu hoặc UUID.
- Kết quả truy xuất top 5 (cosine) của 7 câu hỏi thử:

| Câu hỏi | Kết quả đứng đầu | Điểm | Điều mong đợi trong top 5 |
|---|---|---|---|
| Thử việc tối đa bao lâu? | `45-2019-QH14:d25:k1` | 0,726 | Điều 25, hạng 1 |
| Người lao động muốn nghỉ việc phải báo trước bao nhiêu ngày? | `45-2019-QH14:d36:k2` | 0,707 | Điều 35, hạng 3 |
| Nghỉ thai sản được mấy tháng? | `45-2019-QH14:d139:k1` | 0,741 | Điều 139, hạng 1 |
| Một năm được nghỉ phép bao nhiêu ngày? | `45-2019-QH14:d113:k1` | 0,717 | Điều 113, hạng 1 |
| Làm thêm giờ được trả lương như thế nào? | `45-2019-QH14:d98:k1` | 0,774 | Điều 98, hạng 1 |
| Tuổi nghỉ hưu là bao nhiêu? | `45-2019-QH14:d169:k2` | 0,717 | Điều 169, hạng 1 |
| Cách nấu phở bò ngon (ngoài phạm vi) | `45-2019-QH14:d93:k1` | 0,344 | không có |

Hai điều rút ra từ quan sát đó: câu ngoài phạm vi có điểm thấp hơn hẳn trong lần thử này (0,34 so với 0,71–0,77), nhưng **một câu ngoài phạm vi không đủ để chọn ngưỡng từ chối cho sản phẩm** (việc đó cần bộ đánh giá của Tuần 3); và câu hỏi về báo trước khi nghỉ việc xếp Điều 36 (quyền của người sử dụng lao động) trên Điều 35 (quyền của người lao động), là điểm yếu của truy xuất dense thuần cần ghi nhận.

## Ngoài phạm vi plan này

- Hàm `generate`, prompt, và mọi lời gọi LLM (nửa sau của Tuần 2).
- Sparse vector, hybrid search, RRF (Tuần 4) và reranker (Tuần 5).
- Đo recall@k, MRR (Tuần 3).
- So sánh các model embedding khác. Thiết kế đã sẵn chỗ: đổi biến `EMBEDDING_MODEL` rồi chạy lại lệnh index sẽ tạo collection mới. Các model cần câu hướng dẫn cho câu hỏi (Qwen3, harrier) chưa được hỗ trợ.
- Quản lý revision hoặc prompt của model. Đổi trọng số, revision hay cách mã hóa của model **dưới cùng một tên** thì phải xóa file cache của nó (`.cache/embeddings/<model_key>.npz`) và index lại; việc này nằm ngoài luồng Tuần 2 thông thường.
- Collection tạm, alias hay nâng cấp bằng upsert khi index lại; khóa liên tiến trình; cache phân tán.
- Service `api` và `web` trong Docker Compose (Tuần 8).

---

### Task 1: Phụ thuộc, cấu hình và service Qdrant

**Tier:** Trung bình.

**Mục tiêu:** Dự án cài được PyTorch bản CUDA cùng các thư viện mới với phiên bản chính được ghim, có `Settings` đọc và kiểm tra cấu hình, có `model_key` làm định danh duy nhất cho collection và cache, và có `docker-compose.yml` cho Qdrant chỉ mở cổng cục bộ.

**Files:**
- Create: `app/__init__.py` (rỗng), `app/core/__init__.py` (rỗng), `app/core/config.py`, `tests/test_config.py`
- Modify: `pyproject.toml`, `uv.lock` (do `uv lock` sinh), `docker-compose.yml`, `.env.example`, `tests/conftest.py` (thêm fixture cô lập môi trường), `docs/superpowers/plans/2026-11-13-week2-qdrant-retrieve.md` (file plan này, được commit cùng task)

**Interfaces:**
- Produces:
  - `app.core.config.REPO_ROOT: Path`
  - `app.core.config.model_slug(model_name: str) -> str` (chỉ để đọc cho người; không phải định danh)
  - `app.core.config.model_key(model_name: str) -> str`: `f"{slug}--{sha256(model_name.encode('utf-8')).hexdigest()[:12]}"`
  - `app.core.config.Settings` với các trường `qdrant_url`, `qdrant_collection`, `embedding_model`, `embedding_device`, `embedding_batch_size` và property `collection_name`
  - `app.core.config.get_settings() -> Settings` (có `lru_cache`)
  - Marker pytest `integration`, `model`, `server`; mặc định `uv run pytest` loại các test `integration`
  - Fixture autouse `isolated_settings_environment` trong `tests/conftest.py`

**Tiêu chí nghiệm thu:** `uv run pytest` pass toàn bộ (tối thiểu 44 test cũ + 26 test của `tests/test_config.py`); lệnh kiểm tra phiên bản in đúng bốn phiên bản đã ghim và `True` cho CUDA trên máy dev này; `docker compose config` cho thấy hai cổng gắn `127.0.0.1`; sau commit `uv sync --locked` thành công; hai lệnh ruff sạch; commit của task.

**Lệnh kiểm tra:** `uv run pytest`, `uv run pytest tests/test_config.py --collect-only -q`, `uv sync --locked`, `docker compose config`, `uv run ruff check app ingestion tests`, `uv run ruff format --check app ingestion tests`.

**Ngoài phạm vi:** mọi mã ở `app/rag/` và `ingestion/`; khởi động container; thiết kế lại index PyTorch; không chạy eval (task này không chạm truy xuất).

- [ ] **Step 1: Kiểm tra trạng thái rồi tạo branch**

Run: `git branch --show-current`, `git status --short`, `git rev-parse HEAD`

Expected: đang ở `main`, HEAD là `6b23662…` (hoặc commit hậu duệ mà người dùng xác nhận), và `git status --short` chỉ có file plan này chưa track. Nếu khác (branch khác, file đã sửa hoặc đã staged, thêm file chưa track), **dừng và báo người điều phối**; không `git pull`, không `git switch` sang branch khác, không stash. Nếu đúng, chạy:

`git switch -c week2-qdrant-retrieve`

(Tạo branch từ HEAD hiện tại, giữ nguyên file plan chưa track. Không chạy `git pull`. Branch `main` không bị chạm.)

- [ ] **Step 2: Thay toàn bộ `pyproject.toml`**

```toml
[project]
name = "labor-law-rag"
version = "0.1.0"
description = "RAG assistant for Vietnamese labour law"
requires-python = ">=3.12"
dependencies = [
    "numpy==2.5.4",
    "pydantic>=2,<3",
    "pydantic-settings>=2,<3",
    "python-docx==1.2.0",
    "qdrant-client==1.20.0",
    "sentence-transformers==6.3.0",
    "torch==2.14.1",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu126", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "integration: needs the real embedding model and/or a running Qdrant server",
    "model: needs the real embedding model on disk (GPU or CPU); no Docker",
    "server: needs a Qdrant server at http://127.0.0.1:6333; no embedding model",
]
addopts = "-m 'not integration' --strict-markers"

[tool.ruff]
line-length = 100
```

Phiên bản chính được ghim chính xác; các phụ thuộc còn lại để `uv.lock` chốt. Mọi test integration mang `integration` cộng đúng một trong `model` hoặc `server`.

Run: `uv lock`, rồi `uv sync`
Expected: khóa và cài xong không lỗi; `uv.lock` đổi. Lần đầu tải PyTorch khoảng vài GB. Nếu `uv lock` báo không giải được phiên bản (ví dụ `requires-python`), dừng và báo người điều phối, không tự nới ghim.

- [ ] **Step 3: Kiểm tra phiên bản và PyTorch nhận GPU**

Run: `uv run python -c "import torch, sentence_transformers, numpy; from importlib.metadata import version; print(torch.__version__, torch.cuda.is_available(), sentence_transformers.__version__, version('qdrant-client'), numpy.__version__)"`

(Sửa ngày 15/11/2026 khi thực thi Task 1: `requires-python` là `>=3.12` vì `numpy==2.5.4` cần Python 3.12 trở lên; `qdrant_client` không có `__version__` nên lấy phiên bản qua `importlib.metadata`.)

Expected trên máy dev này (RTX 4070): `2.14.1+cu126 True 6.3.0 1.20.0 2.5.4`. Nếu in `False` hoặc thiếu đuôi `+cu126`, dừng và báo người điều phối, không đổi index. Máy không có GPU NVIDIA vẫn chạy được trên CPU (đặt `EMBEDDING_DEVICE=cpu`); đó không phải lỗi, chỉ chậm hơn.

- [ ] **Step 4: Thêm fixture cô lập môi trường vào `tests/conftest.py`**

Giữ nguyên fixture `make_docx`. Thêm `import pytest` nếu chưa có, dòng `from app.core.config import get_settings` và đoạn sau vào cuối file:

```python
SETTINGS_ENV_NAMES = (
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "EMBEDDING_BATCH_SIZE",
)


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch):
    """Keep unit tests independent of the developer's shell environment."""
    for name in SETTINGS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
```

Quy ước cho mọi test về sau: dựng `Settings(_env_file=None, ...)` để không đọc `.env` thật của máy dev.

- [ ] **Step 5: Viết test thất bại trong `tests/test_config.py`**

```python
import pytest
from pydantic import ValidationError

from app.core.config import REPO_ROOT, Settings, model_key, model_slug


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "labor_law_vn"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_device is None
    assert settings.embedding_batch_size == 16
    assert settings.collection_name == "labor_law_vn__baai-bge-m3--d790e737453c"


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
    settings = Settings(_env_file=None)
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.collection_name == ("labor_law_vn__aiteamvn-vietnamese-embedding--5d5026633ba6")


def test_env_file_is_read_and_unknown_keys_are_ignored(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=not-a-real-key\nQDRANT_COLLECTION=other\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.qdrant_collection == "other"
    assert not hasattr(settings, "llm_api_key")


@pytest.mark.parametrize("field", ["qdrant_url", "qdrant_collection", "embedding_model"])
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_settings_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("size", [0, -3])
def test_batch_size_below_one_is_rejected(size):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_batch_size=size)


def test_batch_size_from_the_environment_is_validated(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    with pytest.raises(ValidationError, match="embedding_batch_size"):
        Settings(_env_file=None)


def test_surrounding_whitespace_is_stripped_and_blank_device_means_unset():
    settings = Settings(_env_file=None, embedding_model=" BAAI/bge-m3 ", embedding_device="  ")
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_device is None


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("BAAI/bge-m3", "baai-bge-m3"),
        ("Qwen/Qwen3-Embedding-0.6B", "qwen-qwen3-embedding-0-6b"),
        (
            "ibm-granite/granite-embedding-311m-multilingual-r2",
            "ibm-granite-granite-embedding-311m-multilingual-r2",
        ),
    ],
)
def test_model_slug(name, slug):
    assert model_slug(name) == slug


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("BAAI/bge-m3", "baai-bge-m3--d790e737453c"),
        ("Qwen/Qwen3-Embedding-0.6B", "qwen-qwen3-embedding-0-6b--caa978c0b354"),
        ("fake/model", "fake-model--da372fba630a"),
    ],
)
def test_model_key_is_the_slug_plus_a_hash_of_the_exact_name(name, key):
    assert model_key(name) == key


def test_model_names_with_the_same_slug_get_different_keys_and_collections():
    names = ["org/a.b", "org/a-b", "org/a_b"]
    assert {model_slug(name) for name in names} == {"org-a-b"}
    assert len({model_key(name) for name in names}) == 3
    collections = {Settings(_env_file=None, embedding_model=name).collection_name for name in names}
    assert len(collections) == 3


def test_model_key_is_case_sensitive():
    assert model_slug("BAAI/bge-m3") == model_slug("baai/bge-m3")
    assert model_key("BAAI/bge-m3") != model_key("baai/bge-m3")


@pytest.mark.parametrize("name", ["", "   ", "\n"])
def test_model_key_rejects_blank_names(name):
    with pytest.raises(ValueError, match="model_name"):
        model_key(name)


def test_model_key_of_a_name_without_letters_still_has_a_slug():
    assert model_key("///").startswith("model--")


def test_repo_root_is_the_project_directory():
    assert (REPO_ROOT / "pyproject.toml").is_file()
```

Các hash trong test tính từ SHA256 của tên model (UTF-8), 12 ký tự hex đầu; đây là đặc tả chứ không phải kết quả chạy mã.

- [ ] **Step 6: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_config.py -v`
Expected: lỗi khi thu thập test (`ModuleNotFoundError`), vì `app.core.config` chưa tồn tại.

- [ ] **Step 7: Tạo `app/__init__.py`, `app/core/__init__.py` (rỗng) và viết `app/core/config.py`**

- `REPO_ROOT = Path(__file__).resolve().parents[2]`.
- `model_slug`: chuyển về chữ thường, thay mọi chuỗi ký tự ngoài `a-z0-9` bằng một dấu `-`, bỏ `-` ở hai đầu. Giữ nguyên hành vi này (chỉ để dễ đọc).
- `model_key(model_name)`: nếu `model_name.strip()` rỗng thì ném `ValueError("model_name must not be blank")`. Phần slug là `model_slug(model_name)`, hoặc `"model"` nếu slug rỗng. Phần hash là 12 ký tự hex đầu của SHA256 trên **chính xác** `model_name.encode("utf-8")` (không strip, không đổi chữ hoa). Kết quả: `f"{slug}--{hash}"`.
- `Settings(BaseSettings)` với `SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")` và các giá trị mặc định:

| Trường | Kiểu | Mặc định | Kiểm tra |
|---|---|---|---|
| `qdrant_url` | `str` | `"http://127.0.0.1:6333"` | strip, không được rỗng |
| `qdrant_collection` | `str` | `"labor_law_vn"` | strip, không được rỗng |
| `embedding_model` | `str` | `"BAAI/bge-m3"` | strip, không được rỗng |
| `embedding_device` | `str \| None` | `None` (để thư viện tự chọn GPU khi có) | chuỗi rỗng hay toàn khoảng trắng thành `None` |
| `embedding_batch_size` | `int` | `16` | `>= 1` (`Field(ge=1)`) |

  Dùng `field_validator` của pydantic cho việc strip, từ chối rỗng và đổi thiết bị rỗng thành `None`. Lỗi là `pydantic.ValidationError`.
- Property `collection_name` trả về `f"{qdrant_collection}__{model_key(embedding_model)}"`. Đây là định danh duy nhất của collection; file cache dùng cùng `model_key` (Task 3).
- `get_settings()` bọc `lru_cache`, trả về `Settings()`.

Mặc định `qdrant_url` là `127.0.0.1` thay vì `localhost` để khớp với cổng chỉ gắn `127.0.0.1` trong Compose (tránh phụ thuộc vào việc `localhost` phân giải sang IPv4 hay IPv6). `.env.example` cập nhật tương ứng ở Step 9.

- [ ] **Step 8: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_config.py -v`
Expected: toàn bộ pass; `uv run pytest tests/test_config.py --collect-only -q` liệt kê ít nhất 26 test.

- [ ] **Step 9: Thay toàn bộ `docker-compose.yml` và cập nhật `.env.example`**

```yaml
# Services: qdrant now; api and web are added in Week 8.
services:
  qdrant:
    image: qdrant/qdrant:v1.20.1
    ports:
      # Bound to loopback only: Docker publishes to every host interface when no IP is given.
      # https://docs.docker.com/engine/network/port-publishing/
      - "127.0.0.1:6333:6333" # REST API and dashboard (http://127.0.0.1:6333/dashboard)
      - "127.0.0.1:6334:6334" # gRPC
    volumes:
      # Named volume: Qdrant's quickstart notes that Windows may need one instead of a local folder.
      # https://qdrant.tech/documentation/quickstart/
      - qdrant_storage:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_storage:
```

Dữ liệu Qdrant nằm trong một named volume của Docker, không phải thư mục trong repo, và luôn dựng lại được bằng lệnh index. Không dùng `docker compose down -v` (xóa volume) trừ khi chủ ý.

Trong `.env.example`, đổi dòng `QDRANT_URL=http://localhost:6333` thành `QDRANT_URL=http://127.0.0.1:6333` và thêm vào cuối file:

```dotenv

# Optional embedding settings. Leave EMBEDDING_DEVICE unset to use the GPU when one is available.
# EMBEDDING_DEVICE=cpu
# EMBEDDING_BATCH_SIZE=16
```

Run: `docker compose config` (chỉ in cấu hình, không khởi động container)
Expected: cả hai cổng có `host_ip: 127.0.0.1`.

- [ ] **Step 10: Chạy lệnh kiểm tra chung, commit, rồi kiểm tra tái lập**

Run: `uv run pytest` (toàn bộ pass) và hai lệnh ruff; rồi `git status --short` và `git diff --stat`, đối chiếu từng file với danh sách ở mục **Files**.

```bash
git add pyproject.toml uv.lock docker-compose.yml .env.example app/__init__.py app/core/__init__.py app/core/config.py tests/conftest.py tests/test_config.py docs/superpowers/plans/2026-11-13-week2-qdrant-retrieve.md
git commit -m "chore: add the Week 2 dependencies, settings and the Qdrant service"
```

File plan này được commit cùng Task 1 (chỉ sau khi người dùng đã rà soát bản chỉnh). Sau commit, máy khác dựng lại môi trường bằng `uv sync --locked`; chạy lệnh đó ở đây và xác nhận không lỗi.

---

### Task 2: Interface embedding và model bge-m3

**Tier:** Trung bình.

**Mục tiêu:** Có interface `Embedder`, bản cài đặt thật bằng sentence-transformers kèm test đơn vị nhanh qua một model giả (không tải torch, model hay mạng), và bản `FakeEmbedder` dùng cho mọi test về sau.

**Files:**
- Create: `app/rag/__init__.py` (rỗng), `app/rag/embedder.py`, `tests/test_embedder.py`, `tests/test_integration_embedder.py`
- Modify: `tests/conftest.py` (thêm `FakeEmbedder` và fixture), `eval/results.md`

**Interfaces:**
- Consumes: `app.core.config.Settings`, fixture autouse `isolated_settings_environment`
- Produces:
  - `app.rag.embedder.Embedder` (Protocol): thuộc tính `model_name: str`, `dimension: int`; `embed_documents(texts: list[str]) -> np.ndarray` trả mảng `float32` kích thước `(len(texts), dimension)` đã chuẩn hóa L2; `embed_query(text: str) -> np.ndarray` trả mảng `(dimension,)`
  - `app.rag.embedder.SentenceTransformerEmbedder(model_name: str, device: str | None = None, batch_size: int = 16)`
  - `SentenceTransformerEmbedder.__init__` import `SentenceTransformer` bên trong hàm; test thay thế bằng một module giả đặt vào `sys.modules["sentence_transformers"]` (không thêm phụ thuộc, không cần seam riêng)
  - `app.rag.embedder.build_embedder(settings: Settings) -> Embedder`
  - Fixture pytest: `fake_embedder`, `make_fake_embedder`, `all_chunks`, `sample_chunks`

**Tiêu chí nghiệm thu:** `uv run pytest` pass toàn bộ (tối thiểu 44 + 26 + 8 = 78 test, với 8 là số test tối thiểu của `tests/test_embedder.py`); `uv run pytest -m model` pass (tối thiểu 1 test) trên máy có model; không test đơn vị nào tải torch hay model; hai lệnh ruff sạch; `eval/results.md` có dòng của Task 2; commit của task.

**Lệnh kiểm tra:** `uv run pytest`, `uv run pytest tests/test_embedder.py --collect-only -q`, `uv run pytest -m model`, `uv run python eval/run_eval.py`, hai lệnh ruff.

**Ngoài phạm vi:** cache (Task 3), Qdrant, hỗ trợ model cần câu hướng dẫn, thêm phụ thuộc mới.

- [ ] **Step 1: Chạy eval trước thay đổi**

Run: `uv run python eval/run_eval.py`
Expected: thoát mã 0, không in số liệu (placeholder).

- [ ] **Step 2: Thêm `FakeEmbedder` và fixture vào `tests/conftest.py`**

Giữ nguyên `make_docx` và fixture cô lập môi trường của Task 1. Thêm các import `hashlib`, `numpy as np`, `from ingestion.documents import REPO_ROOT`, `from ingestion.models import Chunk` (nếu chưa có) và các định nghĩa sau vào cuối file:

```python
class FakeEmbedder:
    """Deterministic stand-in for a model: equal texts give equal unit vectors."""

    def __init__(self, model_name: str = "fake/model", dimension: int = 8) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.calls: list[list[str]] = []  # one entry per embed_documents call

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vector = np.random.default_rng(seed).standard_normal(self.dimension).astype(np.float32)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def make_fake_embedder():
    return FakeEmbedder


@pytest.fixture(scope="session")
def all_chunks() -> list[Chunk]:
    """Every chunk of the committed data/processed/chunks.jsonl."""
    path = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
    with path.open(encoding="utf-8") as lines:
        return [Chunk.model_validate_json(line) for line in lines]


@pytest.fixture
def sample_chunks(all_chunks) -> list[Chunk]:
    """The first 12 real chunks (Điều 1 to Điều 3)."""
    return all_chunks[:12]
```

Trong `conftest.py`, `REPO_ROOT` lấy từ `ingestion.documents` (đã có từ Tuần 1); `app.core.config.REPO_ROOT` trỏ cùng thư mục.

- [ ] **Step 3: Viết test thất bại trong `tests/test_embedder.py`**

```python
import subprocess
import sys
import types

import numpy as np
import pytest

from app.core.config import REPO_ROOT, Settings
from app.rag import embedder as embedder_module
from app.rag.embedder import SentenceTransformerEmbedder, build_embedder


class StubModel:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dimension=4):
        self.dimension = dimension
        self.encode_calls = []
        self.loads = []

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, sentences, **kwargs):
        self.encode_calls.append((list(sentences), kwargs))
        # float64 on purpose: the adapter must hand back float32
        return np.full((len(sentences), self.dimension), 0.5, dtype=np.float64)


@pytest.fixture
def stub_model(monkeypatch):
    """Fake the sentence_transformers module: no torch import, download or network access."""
    model = StubModel()

    def factory(model_name, device=None):
        model.loads.append((model_name, device))
        return model

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return model


def test_fake_embedder_is_deterministic_and_normalised(fake_embedder):
    first = fake_embedder.embed_documents(["Thời gian thử việc", "Tiền lương làm thêm giờ"])
    second = fake_embedder.embed_documents(["Thời gian thử việc", "Tiền lương làm thêm giờ"])
    assert first.shape == (2, 8)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(fake_embedder.embed_query("Thời gian thử việc"), first[0])
    assert fake_embedder.calls == [["Thời gian thử việc", "Tiền lương làm thêm giờ"]] * 2


def test_adapter_loads_the_model_and_passes_the_encode_flags(stub_model):
    embedder = SentenceTransformerEmbedder("some/model", device="cpu", batch_size=8)
    vectors = embedder.embed_documents(["a", "b"])
    assert stub_model.loads == [("some/model", "cpu")]
    assert (embedder.model_name, embedder.dimension) == ("some/model", 4)
    assert stub_model.encode_calls == [
        (
            ["a", "b"],
            {
                "batch_size": 8,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            },
        )
    ]
    assert isinstance(vectors, np.ndarray)
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 4)


def test_query_is_one_float32_vector_and_empty_input_skips_the_model(stub_model):
    embedder = SentenceTransformerEmbedder("some/model")
    query = embedder.embed_query("q")
    assert stub_model.encode_calls[0][0] == ["q"]
    assert query.shape == (4,)
    assert query.dtype == np.float32
    empty = embedder.embed_documents([])
    assert empty.shape == (0, 4)
    assert empty.dtype == np.float32
    assert len(stub_model.encode_calls) == 1


@pytest.mark.parametrize("dimension", [None, 0, -5])
def test_an_unusable_dimension_is_rejected(stub_model, dimension):
    stub_model.dimension = dimension
    with pytest.raises(ValueError, match="embedding dimension"):
        SentenceTransformerEmbedder("some/model")


def test_importing_the_module_does_not_import_torch():
    code = (
        "import sys, app.rag.embedder; "
        "assert not {'torch', 'sentence_transformers'} & set(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_build_embedder_uses_the_settings(monkeypatch):
    created = {}

    class Recorder:
        def __init__(self, model_name, device=None, batch_size=16):
            created.update(model_name=model_name, device=device, batch_size=batch_size)

    monkeypatch.setattr(embedder_module, "SentenceTransformerEmbedder", Recorder)
    settings = Settings(
        _env_file=None,
        embedding_model="AITeamVN/Vietnamese_Embedding",
        embedding_device="cpu",
        embedding_batch_size=4,
    )
    assert isinstance(build_embedder(settings), Recorder)
    assert created == {
        "model_name": "AITeamVN/Vietnamese_Embedding",
        "device": "cpu",
        "batch_size": 4,
    }
```

- [ ] **Step 4: Viết test tích hợp trong `tests/test_integration_embedder.py`**

Test này tách riêng: dùng model thật, cần model trên đĩa, không cần Docker.

```python
"""Needs the real embedding model. Run with: uv run pytest -m model"""

import numpy as np
import pytest

from app.rag.embedder import SentenceTransformerEmbedder

pytestmark = [pytest.mark.integration, pytest.mark.model]


def test_model_dimension_and_normalisation():
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    vectors = embedder.embed_documents(["Thời gian thử việc", "Tiền lương làm thêm giờ"])
    assert embedder.model_name == "BAAI/bge-m3"
    assert embedder.dimension == 1024
    assert vectors.shape == (2, 1024)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    query = embedder.embed_query("Thời gian thử việc")
    assert query.shape == (1024,)
    assert float(query @ vectors[0]) > 0.99
    assert embedder.embed_documents([]).shape == (0, 1024)
```

- [ ] **Step 5: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: lỗi khi thu thập test (`ImportError`), vì `app.rag.embedder` chưa tồn tại.

- [ ] **Step 6: Tạo `app/rag/__init__.py` (rỗng) và viết `app/rag/embedder.py`**

- `Embedder` là `typing.Protocol` như mô tả ở Interfaces.
- Đầu module chỉ import numpy và `Settings`, không import torch hay sentence-transformers.
- `SentenceTransformerEmbedder.__init__` import `SentenceTransformer` **bên trong hàm**, rồi gọi `SentenceTransformer(model_name, device=device)`. Lấy số chiều bằng `model.get_embedding_dimension()` (tên cũ `get_sentence_embedding_dimension` đã bị đánh dấu lỗi thời ở bản 6). Hàm này có thể trả `None`: nếu kết quả không phải `int` dương (loại cả `bool`) thì `ValueError(f"cannot determine the embedding dimension of {model_name}")`. `batch_size` đã được `Settings` kiểm tra (>= 1), adapter không kiểm tra lại.
- `embed_documents`: danh sách rỗng trả `np.zeros((0, dimension), dtype=np.float32)` mà không gọi model. Ngược lại gọi `model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)` rồi ép về `float32` bằng `np.asarray(..., dtype=np.float32)`. Việc kiểm tra kích thước và giá trị hữu hạn của kết quả nằm ở Task 3 (cache) và Task 4 (index), không lặp ở đây.
- `embed_query(text)` là dòng đầu của `embed_documents([text])`.
- `build_embedder` truyền `embedding_model`, `embedding_device`, `embedding_batch_size` từ `settings` vào `SentenceTransformerEmbedder`.

- [ ] **Step 7: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: toàn bộ pass; `--collect-only -q` liệt kê ít nhất 8 test.

Run: `uv run pytest -m model`
Expected: pass (tối thiểu 1 test). Lần đầu có thể mất khoảng một phút để nạp model. Chạy được mà không cần Docker.

- [ ] **Step 8: Chạy eval sau thay đổi và ghi `eval/results.md`**

Run: `uv run python eval/run_eval.py` (thoát mã 0, không in số liệu).

Trong `eval/results.md`, giữ nguyên các dòng hiện có và thêm bảng sau vào cuối file (các task sau thêm dòng vào bảng này; `<ngày chạy>` là ngày chạy thật, dạng `YYYY-MM-DD`):

```markdown

| Ngày | Thay đổi | recall@5 | recall@10 | MRR | Ghi chú |
|---|---|---|---|---|---|
| <ngày chạy> | Task 2: interface embedding và bge-m3 | N/A | N/A | N/A | chưa đo recall |
```

- [ ] **Step 9: Chạy lệnh kiểm tra chung rồi commit**

Run: `uv run pytest` và hai lệnh ruff; rồi `git status --short` và `git diff --stat`, đối chiếu với mục **Files**.

```bash
git add app/rag/__init__.py app/rag/embedder.py tests/conftest.py tests/test_embedder.py tests/test_integration_embedder.py eval/results.md
git commit -m "feat: add the embedder interface with a sentence-transformers implementation"
```

---

### Task 3: Cache embedding ra đĩa

**Tier:** Trung bình.

**Mục tiêu:** Embedding một danh sách text mà chỉ tính những text chưa có trong cache của đúng model đó; cache hỏng, sai model, sai số chiều hay chứa số không hữu hạn chỉ bị coi là rỗng, và lần ghi lỗi không phá cache cũ.

**Files:**
- Create: `ingestion/embed_cache.py`, `tests/test_embed_cache.py`
- Modify: `eval/results.md`

**Interfaces:**
- Consumes: `Embedder`, `model_key`, fixture `fake_embedder`, `make_fake_embedder`
- Produces: `ingestion.embed_cache.embed_with_cache(texts: list[str], embedder: Embedder, cache_dir: Path) -> np.ndarray`; hằng `CACHE_FORMAT_VERSION = 1`

**Tiêu chí nghiệm thu:** `uv run pytest` pass toàn bộ (tối thiểu 44 + 26 + 8 + 33 = 111 test, với 33 là số test tối thiểu của `tests/test_embed_cache.py`); hai lệnh ruff sạch; `eval/results.md` có dòng của Task 3; commit của task.

**Lệnh kiểm tra:** `uv run pytest`, `uv run pytest tests/test_embed_cache.py --collect-only -q`, `uv run python eval/run_eval.py`, hai lệnh ruff.

**Ngoài phạm vi:** khóa liên tiến trình, cache phân tán, dọn mục cũ trong cache, theo dõi revision của model. Chỉ một tiến trình ghi tại một thời điểm. Đổi trọng số, revision hay cách mã hóa dưới cùng tên model thì xóa file cache của model đó rồi index lại; ghi điều này vào docstring đầu module.

Quy tắc:

1. File cache là `cache_dir / f"{model_key(embedder.model_name)}.npz"`, lưu bằng `np.savez` (không nén) với các mảng: `format_version` (số nguyên 0-d, bằng `CACHE_FORMAT_VERSION`), `model_name` (chuỗi 0-d, tên model chính xác), `dimension` (số nguyên 0-d), `hashes` (mảng chuỗi unicode 1-D dtype `U64`, tức `np.array(hashes, dtype="U64")`, SHA256 dạng hex của từng text mã hóa UTF-8; không dùng dtype object) và `vectors` (`float32`, kích thước `(len(hashes), dimension)`).
2. Khi đọc: `np.load(path, allow_pickle=False)` trong một hàm riêng trả về dict `hash -> vector` hoặc `{}`. Bất kỳ lỗi đọc nào (file thiếu, không phải zip, bị cắt cụt, thiếu mảng, mảng object cần pickle) và bất kỳ vi phạm nào sau đây đều là cache miss, không ném lỗi: `format_version` khác `CACHE_FORMAT_VERSION`; `model_name` khác `embedder.model_name`; `dimension` khác `embedder.dimension`; mảng metadata không phải 0-d; `hashes` không phải mảng 1-D dtype chuỗi unicode (`dtype.kind == "U"`) hoặc có hash trùng; `vectors` không phải `float32` 2-D kích thước `(len(hashes), embedder.dimension)`; `vectors` chứa NaN hoặc Inf. Việc bắt `Exception` rộng chỉ nằm trong hàm đọc này (cache chỉ là tối ưu); mọi chỗ khác không nuốt lỗi. Mở file bằng `with np.load(...) as data:` và đọc hết mảng cần dùng vào bộ nhớ trong khối đó, để handle đã đóng trước khi ghi đè (bắt buộc trên Windows).
3. Chỉ gọi `embedder.embed_documents` một lần, với các text chưa có trong cache, theo thứ tự xuất hiện, mỗi text trùng nhau chỉ một lần. Không thiếu text nào thì không gọi.
4. Kiểm tra kết quả mới **trước khi ghi**: phải là mảng số dạng `(số text thiếu, embedder.dimension)` và hữu hạn; ép về `float32`. Sai thì `ValueError("embedder returned invalid embeddings: ...")` và không ghi gì.
5. Ghi cache gồm cả mục cũ lẫn mục mới (mục cũ không còn dùng vẫn ở lại; xóa file cache để dọn). Tạo `cache_dir` nếu chưa có. Ghi qua file tạm duy nhất **cùng thư mục** rồi `os.replace`. Truyền **file handle** cho `np.savez` (truyền tên file thì nó tự thêm đuôi `.npz` ngoài ý muốn), và đóng mọi handle (file tạm, file `np.load`) trước khi `os.replace` (cần cho Windows):

```python
def _write_atomically(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as temp_file:
            np.savez(temp_file, **arrays)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
```

   Module phải dùng `import os` và gọi `os.replace` (test thay hàm này). Ghi lỗi thì ném lại `OSError`, cache cũ nguyên vẹn và không để lại file `.tmp`.
6. Trả về mảng `float32` có một dòng cho mỗi text, đúng thứ tự đầu vào. Đầu vào rỗng trả mảng `(0, dimension)` và không ghi file.

- [ ] **Step 1: Chạy eval trước thay đổi**

Run: `uv run python eval/run_eval.py`
Expected: thoát mã 0, không in số liệu.

- [ ] **Step 2: Viết test thất bại trong `tests/test_embed_cache.py`**

Các chuỗi dưới đây là văn bản tổng hợp trung tính, không phải nội dung luật.

```python
import numpy as np
import pytest

from app.core.config import model_key
from ingestion import embed_cache
from ingestion.embed_cache import embed_with_cache

TEXTS = ["Văn bản thử nghiệm số 1", "Văn bản thử nghiệm số 2", "Văn bản thử nghiệm số 3"]


def cache_file(embedder, cache_dir):
    return cache_dir / f"{model_key(embedder.model_name)}.npz"


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def without(name):
    return lambda arrays: {key: value for key, value in arrays.items() if key != name}


def poisoned(vectors, row, column, value):
    changed = vectors.copy()
    changed[row, column] = value
    return changed


TAMPERING = {
    **{
        f"missing_{name}": without(name)
        for name in ("format_version", "model_name", "dimension", "hashes", "vectors")
    },
    "other_format_version": lambda a: {**a, "format_version": np.array(99)},
    "other_model_name": lambda a: {**a, "model_name": np.array("other/model")},
    "other_dimension": lambda a: {**a, "dimension": np.array(99)},
    "float64_vectors": lambda a: {**a, "vectors": a["vectors"].astype(np.float64)},
    "flat_vectors": lambda a: {**a, "vectors": a["vectors"].reshape(-1)},
    "missing_row": lambda a: {**a, "vectors": a["vectors"][:-1]},
    "nan_value": lambda a: {**a, "vectors": poisoned(a["vectors"], 0, 0, np.nan)},
    "inf_value": lambda a: {**a, "vectors": poisoned(a["vectors"], 1, 1, np.inf)},
    "duplicate_hashes": lambda a: {**a, "hashes": np.repeat(a["hashes"][:1], len(a["hashes"]))},
    "pickled_hashes": lambda a: {**a, "hashes": a["hashes"].astype(object)},
}

CORRUPTIONS = {
    "garbage": lambda data: b"not a zip archive",
    "truncated": lambda data: data[: len(data) // 2],
    "empty": lambda data: b"",
}

BAD_OUTPUTS = {
    "nan": lambda rows, width: np.full((rows, width), np.nan, dtype=np.float32),
    "inf": lambda rows, width: np.full((rows, width), np.inf, dtype=np.float32),
    "too_few_rows": lambda rows, width: np.ones((rows - 1, width), dtype=np.float32),
    "wrong_width": lambda rows, width: np.ones((rows, width + 1), dtype=np.float32),
    "flat": lambda rows, width: np.ones(rows * width, dtype=np.float32),
}


def test_returns_one_row_per_text_in_input_order(fake_embedder, tmp_path):
    vectors = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert vectors.shape == (3, fake_embedder.dimension)
    assert vectors.dtype == np.float32
    for row, text in zip(vectors, TEXTS, strict=True):
        assert np.array_equal(row, fake_embedder.embed_query(text))


def test_second_call_does_not_embed_again(fake_embedder, tmp_path):
    first = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    second = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert len(fake_embedder.calls) == 1
    assert np.array_equal(first, second)


def test_only_new_or_changed_texts_are_embedded(fake_embedder, tmp_path):
    embed_with_cache(TEXTS, fake_embedder, tmp_path)
    changed = [TEXTS[0], "Văn bản thử nghiệm số 2 (đã đổi)", TEXTS[2], "Văn bản thử nghiệm số 4"]
    vectors = embed_with_cache(changed, fake_embedder, tmp_path)
    assert fake_embedder.calls[1] == [changed[1], changed[3]]
    assert np.array_equal(vectors[1], fake_embedder.embed_query(changed[1]))
    assert np.array_equal(vectors[0], fake_embedder.embed_query(changed[0]))


def test_duplicate_texts_are_embedded_once(fake_embedder, tmp_path):
    vectors = embed_with_cache([TEXTS[0], TEXTS[1], TEXTS[0]], fake_embedder, tmp_path)
    assert fake_embedder.calls == [[TEXTS[0], TEXTS[1]]]
    assert np.array_equal(vectors[0], vectors[2])


def test_empty_input_embeds_nothing_and_writes_nothing(fake_embedder, tmp_path):
    vectors = embed_with_cache([], fake_embedder, tmp_path)
    assert vectors.shape == (0, fake_embedder.dimension)
    assert fake_embedder.calls == []
    assert list(tmp_path.iterdir()) == []


def test_cache_directory_is_created(fake_embedder, tmp_path):
    cache_dir = tmp_path / "nested" / "embeddings"
    embed_with_cache(TEXTS, fake_embedder, cache_dir)
    assert cache_file(fake_embedder, cache_dir).is_file()


def test_each_model_has_its_own_cache_file(make_fake_embedder, tmp_path):
    first = make_fake_embedder("BAAI/bge-m3")
    second = make_fake_embedder("Qwen/Qwen3-Embedding-0.6B")
    embed_with_cache(TEXTS, first, tmp_path)
    embed_with_cache(TEXTS, second, tmp_path)
    assert len(second.calls) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"{model_key('BAAI/bge-m3')}.npz", f"{model_key('Qwen/Qwen3-Embedding-0.6B')}.npz"]
    )


def test_model_names_with_the_same_slug_do_not_share_a_cache(make_fake_embedder, tmp_path):
    dotted = make_fake_embedder("org/a.b")
    dashed = make_fake_embedder("org/a-b")
    embed_with_cache(TEXTS, dotted, tmp_path)
    embed_with_cache(TEXTS, dashed, tmp_path)
    assert len(dashed.calls) == 1
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_cache_of_another_dimension_is_recomputed(make_fake_embedder, tmp_path):
    embed_with_cache(TEXTS, make_fake_embedder(dimension=8), tmp_path)
    wider = make_fake_embedder(dimension=16)
    vectors = embed_with_cache(TEXTS, wider, tmp_path)
    assert len(wider.calls) == 1
    assert vectors.shape == (3, 16)


@pytest.mark.parametrize("corrupt", list(CORRUPTIONS.values()), ids=list(CORRUPTIONS))
def test_unreadable_cache_file_is_a_miss(fake_embedder, tmp_path, corrupt):
    expected = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    path = cache_file(fake_embedder, tmp_path)
    path.write_bytes(corrupt(path.read_bytes()))
    vectors = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert fake_embedder.calls[1] == TEXTS
    assert np.array_equal(vectors, expected)


@pytest.mark.parametrize("tamper", list(TAMPERING.values()), ids=list(TAMPERING))
def test_invalid_cache_content_is_a_miss_and_is_rewritten(fake_embedder, tmp_path, tamper):
    expected = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    path = cache_file(fake_embedder, tmp_path)
    np.savez(path, **tamper(read_arrays(path)))
    vectors = embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert fake_embedder.calls[1] == TEXTS
    assert np.array_equal(vectors, expected)
    embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert len(fake_embedder.calls) == 2


@pytest.mark.parametrize("make_bad", list(BAD_OUTPUTS.values()), ids=list(BAD_OUTPUTS))
def test_invalid_encoder_output_is_rejected_and_the_old_cache_is_kept(
    fake_embedder, tmp_path, monkeypatch, make_bad
):
    embed_with_cache(TEXTS[:2], fake_embedder, tmp_path)
    path = cache_file(fake_embedder, tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(
        fake_embedder,
        "embed_documents",
        lambda texts: make_bad(len(texts), fake_embedder.dimension),
    )
    with pytest.raises(ValueError, match="invalid embeddings"):
        embed_with_cache(TEXTS, fake_embedder, tmp_path)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_failed_replacement_keeps_the_old_cache_and_leaves_no_temp_file(
    fake_embedder, tmp_path, monkeypatch
):
    embed_with_cache(TEXTS[:2], fake_embedder, tmp_path)
    path = cache_file(fake_embedder, tmp_path)
    before = path.read_bytes()

    def broken_replace(source, destination):
        raise OSError("replace failed")

    with monkeypatch.context() as patch:
        patch.setattr(embed_cache.os, "replace", broken_replace)
        with pytest.raises(OSError, match="replace failed"):
            embed_with_cache(TEXTS, fake_embedder, tmp_path)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    embed_with_cache(TEXTS[:2], fake_embedder, tmp_path)
    assert len(fake_embedder.calls) == 2  # the old entries still hit
```

- [ ] **Step 3: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_embed_cache.py -v`
Expected: lỗi khi thu thập test (`ModuleNotFoundError`), vì `ingestion.embed_cache` chưa tồn tại.

- [ ] **Step 4: Viết `embed_with_cache` trong `ingestion/embed_cache.py` theo 6 quy tắc trên**

Docstring đầu module ghi: khóa là `model_key`; chỉ một tiến trình ghi; đổi trọng số/revision/cách mã hóa dưới cùng tên model thì xóa file cache và index lại.

- [ ] **Step 5: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_embed_cache.py -v`
Expected: toàn bộ pass; `--collect-only -q` liệt kê ít nhất 33 test.

- [ ] **Step 6: Chạy eval sau thay đổi và ghi `eval/results.md`**

Run: `uv run python eval/run_eval.py` (thoát mã 0, không in số liệu). Thêm vào bảng ở `eval/results.md`:

```markdown
| <ngày chạy> | Task 3: cache embedding theo model_key | N/A | N/A | N/A | chưa đo recall |
```

- [ ] **Step 7: Chạy lệnh kiểm tra chung rồi commit**

Run: `uv run pytest` và hai lệnh ruff; rồi `git status --short` và `git diff --stat`, đối chiếu với mục **Files**.

```bash
git add ingestion/embed_cache.py tests/test_embed_cache.py eval/results.md
git commit -m "feat: cache embeddings on disk per model and per text"
```

---

### Task 4: Nạp chunk vào Qdrant

**Tier:** Trung bình.

**Mục tiêu:** Lệnh `uv run python -m ingestion.index` embedding toàn bộ chunk và dựng lại collection Qdrant của model đang cấu hình; đầu vào hỏng không phá collection cũ; lỗi Qdrant (kể cả khi đang ghi) cho một thông báo ngắn và mã thoát 1.

**Files:**
- Create: `app/rag/store.py`, `ingestion/index.py`, `tests/test_store.py`, `tests/test_index.py`
- Modify: `eval/results.md`

**Interfaces:**
- Consumes: `Settings`, `get_settings`, `REPO_ROOT`, `model_key`, `build_embedder`, `embed_with_cache`, `ingestion.models.Chunk`, các fixture ở Task 2
- Produces:
  - `app.rag.store.QdrantUnavailableError` (kế thừa `RuntimeError`)
  - `app.rag.store.qdrant_errors(url: str | None = None)` (context manager, xem dưới)
  - `app.rag.store.connect(settings: Settings) -> QdrantClient`
  - `ingestion.index.point_id(chunk_id: str) -> str`
  - `ingestion.index.load_chunks(path: Path) -> list[Chunk]`
  - `ingestion.index.index_chunks(chunks: list[Chunk], vectors: np.ndarray, client: QdrantClient, collection: str) -> int`
  - `ingestion.index.main(argv: list[str] | None = None) -> int`
  - Lệnh: `uv run python -m ingestion.index [--chunks PATH] [--cache-dir PATH]`

**Tiêu chí nghiệm thu:** `uv run pytest` pass toàn bộ (tối thiểu 44 + 26 + 8 + 33 + 7 + 31 = 149 test, với 7 và 31 là số test tối thiểu của `tests/test_store.py` và `tests/test_index.py`); hai lệnh ruff sạch; `eval/results.md` có dòng của Task 4; commit của task.

**Lệnh kiểm tra:** `uv run pytest`, `uv run pytest tests/test_store.py tests/test_index.py --collect-only -q`, `uv run python eval/run_eval.py`, hai lệnh ruff.

**Ngoài phạm vi:** collection tạm, alias, nâng cấp bằng upsert (không đưa vào kiến trúc). Dựng lại collection là chấp nhận được cho baseline 676 chunk chạy cục bộ; nếu lần dựng bị ngắt giữa chừng, collection có thể thiếu điểm và chỉ cần chạy lại lệnh index. Không đụng `Retriever` (Task 5).

Quy tắc của `qdrant_errors` và `connect` (API lấy từ qdrant-client 1.20.0 qua REST, xem mục "Kiểm tra và quan sát"). Dùng đúng một hàm này làm biên xử lý lỗi cho mọi nơi gọi Qdrant:

```python
@contextmanager
def qdrant_errors(url: str | None = None) -> Iterator[None]:
    """The single place that turns Qdrant transport and 5xx failures into QdrantUnavailableError."""
    try:
        yield
    except ResponseHandlingException as error:
        # Also raised when a response cannot be parsed, so only the connection probe
        # (which passes the url) claims "not reachable"; other calls use the generic wording.
        if url:
            message = f"Qdrant is not reachable at {url}. Start it with: docker compose up -d qdrant"
        else:
            message = (
                f"Qdrant request failed ({type(error.source).__name__}). "
                "Check it with: docker compose ps qdrant"
            )
        raise QdrantUnavailableError(message) from error
    except UnexpectedResponse as error:
        if error.status_code is not None and error.status_code >= 500:
            raise QdrantUnavailableError(
                f"Qdrant answered HTTP {error.status_code}. "
                "Check it with: docker compose logs qdrant"
            ) from error
        raise
```

Hai lớp lỗi import từ `qdrant_client.http.exceptions`. Mọi lỗi khác (`ValueError`, `RuntimeError`, `UnexpectedResponse` mã dưới 500, lỗi lập trình) **không** bị bắt và phải bay lên nguyên vẹn.

- `connect(settings)`: tạo `QdrantClient(url=settings.qdrant_url, timeout=10)`, rồi gọi `client.get_collections()` trong `with qdrant_errors(settings.qdrant_url):`. Nếu bước thử thất bại thì `client.close()` rồi ném lại lỗi. Thông báo khi không kết nối được là đúng `Qdrant is not reachable at {url}. Start it with: docker compose up -d qdrant`.

Quy tắc của `index_chunks`:

1. **Kiểm tra toàn bộ đầu vào trước khi đụng tới collection**, theo thứ tự: không có chunk thì `ValueError("no chunks to index")`; `vectors` không phải mảng 2-D thì `ValueError("vectors must be a 2-D array ...")`; số dòng khác số chunk thì `ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")`; số cột nhỏ hơn 1 thì `ValueError("vectors must have at least one column")`; không phải mảng dấu phẩy động thì `ValueError("vectors must be a floating-point array")`; có NaN hoặc Inf thì `ValueError("vectors contain NaN or infinite values")`; trùng `chunk.id` thì `ValueError(f"duplicate chunk ids: ...")`.
2. Toàn bộ phần ghi (từ kiểm tra tồn tại tới `count`) nằm trong **một** `with qdrant_errors():`, không bọc riêng từng lời gọi; nhờ đó lỗi truyền tải khi đang ghi (không chỉ lúc kết nối) thành `QdrantUnavailableError`.
3. Nếu collection đã tồn tại thì xóa, rồi tạo mới với `VectorParams(size=vectors.shape[1], distance=Distance.COSINE)`.
4. Mỗi chunk là một `PointStruct`: `id=point_id(chunk.id)`, `vector=vector.tolist()`, `payload=chunk.model_dump(mode="json")`.
5. `point_id` trả về `str(uuid.uuid5(POINT_NAMESPACE, chunk_id))` với hằng `POINT_NAMESPACE = uuid.UUID("b7e2c1a4-5d3f-4e8a-9c6b-2f1a0d4e7c35")`.
6. Upsert theo lô 256 điểm với `wait=True`. Trả về `client.count(collection, exact=True).count`.

Quy tắc của `main`, theo đúng thứ tự:

1. `sys.stdout.reconfigure(encoding="utf-8")` và tương tự cho `sys.stderr`.
2. Mặc định `--chunks` là `REPO_ROOT / "data" / "processed" / "chunks.jsonl"`, `--cache-dir` là `REPO_ROOT / ".cache" / "embeddings"`.
3. `get_settings()`; gặp `pydantic.ValidationError` thì in `invalid settings: ...` ra stderr, trả về `1`.
4. File chunk không tồn tại: in ra stderr `chunks file not found: {path}. Create it with: uv run python -m ingestion.build`, trả về `1`.
5. Gọi `connect(settings)` **trước khi** nạp model. Gặp `QdrantUnavailableError` thì in thông báo ra stderr, trả về `1`.
6. Nạp chunk, `build_embedder(settings)`, `embed_with_cache` trên `context_text` của từng chunk, rồi `index_chunks` vào `settings.collection_name`. Chỉ bắt `QdrantUnavailableError` quanh `index_chunks`: in thông báo cộng thêm dòng `The collection may be incomplete; rerun: uv run python -m ingestion.index` ra stderr, trả về `1`. Không bắt lỗi khác.
7. In đúng ba dòng rồi trả về `0` (số chiều lấy từ `vectors.shape[1]`; ví dụ dưới là bge-m3):

```
collection: labor_law_vn__baai-bge-m3--d790e737453c
points: 676
dimension: 1024
```

`main` không đóng client (tiến trình kết thúc ngay, và test tiêm client trong bộ nhớ đã đóng thì mọi thao tác sau đó ném `RuntimeError`).

- [ ] **Step 1: Chạy eval trước thay đổi**

Run: `uv run python eval/run_eval.py`
Expected: thoát mã 0, không in số liệu.

- [ ] **Step 2: Viết test thất bại trong `tests/test_store.py`**

```python
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import Settings
from app.rag.store import QdrantUnavailableError, connect, qdrant_errors


def test_the_connection_probe_keeps_the_specific_hint():
    with pytest.raises(QdrantUnavailableError) as info, qdrant_errors("http://127.0.0.1:6333"):
        raise ResponseHandlingException(OSError("connection reset"))
    assert str(info.value) == (
        "Qdrant is not reachable at http://127.0.0.1:6333. "
        "Start it with: docker compose up -d qdrant"
    )


def test_other_transport_failures_use_the_generic_wording():
    expected = r"Qdrant request failed \(OSError\)"
    with pytest.raises(QdrantUnavailableError, match=expected), qdrant_errors():
        raise ResponseHandlingException(OSError("connection reset"))


def test_server_errors_become_qdrant_unavailable():
    failure = UnexpectedResponse(503, "Service Unavailable", b"", {})
    with pytest.raises(QdrantUnavailableError, match="HTTP 503"), qdrant_errors():
        raise failure


def test_client_errors_are_not_swallowed():
    failure = UnexpectedResponse(400, "Bad Request", b"", {})
    with pytest.raises(UnexpectedResponse), qdrant_errors():
        raise failure


@pytest.mark.parametrize("error", [ValueError("bad input"), RuntimeError("bug")])
def test_other_errors_are_not_swallowed(error):
    with pytest.raises(type(error)), qdrant_errors():
        raise error


@pytest.mark.filterwarnings("ignore:Failed to obtain server version")
def test_connect_reports_an_unreachable_server():
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    with pytest.raises(QdrantUnavailableError, match="docker compose up -d qdrant"):
        connect(settings)
```

- [ ] **Step 3: Viết test thất bại trong `tests/test_index.py`**

```python
import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import REPO_ROOT, Settings, model_key
from app.rag.store import QdrantUnavailableError
from ingestion import index
from ingestion.index import index_chunks, load_chunks, point_id

BAD_INPUTS = [
    "no_chunks",
    "flat_vectors",
    "too_few_vectors",
    "zero_columns",
    "integer_vectors",
    "nan_vector",
    "infinite_vector",
    "duplicate_chunk_ids",
]


@pytest.fixture
def client():
    qdrant = QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


def vectors_for(chunks, embedder):
    return embedder.embed_documents([chunk.context_text for chunk in chunks])


def bad_input(case, chunks, good):
    """Return (chunks, vectors, expected error message) for one invalid input."""
    poisoned = good.copy()
    if case == "no_chunks":
        return [], good[:0], "no chunks to index"
    if case == "flat_vectors":
        return chunks, good.reshape(-1), "2-D"
    if case == "too_few_vectors":
        return chunks, good[:3], "12 chunks but 3 vectors"
    if case == "zero_columns":
        return chunks, good[:, :0], "at least one column"
    if case == "integer_vectors":
        return chunks, good.astype(np.int64), "floating-point"
    if case == "nan_vector":
        poisoned[3, 2] = np.nan
        return chunks, poisoned, "NaN or infinite"
    if case == "infinite_vector":
        poisoned[4, 1] = np.inf
        return chunks, poisoned, "NaN or infinite"
    if case == "duplicate_chunk_ids":
        return [chunks[0], *chunks[:-1]], good, "duplicate chunk ids"
    raise AssertionError(case)


def test_point_id_is_a_stable_uuid():
    first = point_id("45-2019-QH14:d35:k1")
    assert first == point_id("45-2019-QH14:d35:k1")
    assert first != point_id("45-2019-QH14:d35:k2")
    assert str(uuid.UUID(first)) == first


def test_load_chunks_reads_the_committed_file():
    chunks = load_chunks(REPO_ROOT / "data" / "processed" / "chunks.jsonl")
    assert len(chunks) == 676
    assert chunks[0].id == "45-2019-QH14:d1"


def test_index_stores_every_chunk_with_its_payload(client, sample_chunks, fake_embedder):
    count = index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "test")
    assert count == 12
    chunk = sample_chunks[5]
    stored = client.retrieve("test", ids=[point_id(chunk.id)], with_payload=True)[0]
    assert stored.payload == chunk.model_dump(mode="json")


def test_indexing_again_replaces_the_collection(client, sample_chunks, fake_embedder):
    index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "test")
    fewer = sample_chunks[:5]
    assert index_chunks(fewer, vectors_for(fewer, fake_embedder), client, "test") == 5
    assert index_chunks(fewer, vectors_for(fewer, fake_embedder), client, "test") == 5


def test_collection_dimension_follows_the_vectors(client, sample_chunks, make_fake_embedder):
    index_chunks(
        sample_chunks, vectors_for(sample_chunks, make_fake_embedder(dimension=8)), client, "test"
    )
    wider = vectors_for(sample_chunks, make_fake_embedder(dimension=16))
    assert index_chunks(sample_chunks, wider, client, "test") == 12
    assert client.get_collection("test").config.params.vectors.size == 16


@pytest.mark.parametrize("case", BAD_INPUTS)
def test_invalid_input_raises_and_keeps_the_existing_collection(
    client, sample_chunks, fake_embedder, case
):
    good = vectors_for(sample_chunks, fake_embedder)
    assert index_chunks(sample_chunks, good, client, "test") == 12
    chunks, vectors, message = bad_input(case, sample_chunks, good)

    with pytest.raises(ValueError, match=message):
        index_chunks(chunks, vectors, client, "test")

    assert client.count("test", exact=True).count == 12
    assert client.get_collection("test").config.params.vectors.size == 8
    stored = client.retrieve("test", ids=[point_id(sample_chunks[0].id)], with_payload=True)
    assert stored[0].payload == sample_chunks[0].model_dump(mode="json")


@pytest.mark.parametrize("case", BAD_INPUTS)
def test_invalid_input_does_not_create_a_collection(client, sample_chunks, fake_embedder, case):
    good = vectors_for(sample_chunks, fake_embedder)
    chunks, vectors, message = bad_input(case, sample_chunks, good)
    with pytest.raises(ValueError, match=message):
        index_chunks(chunks, vectors, client, "test")
    assert not client.collection_exists("test")


@pytest.mark.parametrize("method", ["collection_exists", "create_collection", "upsert", "count"])
def test_transport_failures_while_writing_become_qdrant_unavailable(
    client, sample_chunks, fake_embedder, monkeypatch, method
):
    def fail(*args, **kwargs):
        raise ResponseHandlingException(OSError("connection reset"))

    monkeypatch.setattr(client, method, fail)
    with pytest.raises(QdrantUnavailableError, match="Qdrant request failed"):
        index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "test")


def test_programming_errors_while_writing_are_not_swallowed(
    client, sample_chunks, fake_embedder, monkeypatch
):
    def bug(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(client, "upsert", bug)
    with pytest.raises(RuntimeError, match="bug"):
        index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "test")


def test_main_indexes_all_chunks(monkeypatch, client, make_fake_embedder, tmp_path, capsys):
    settings = Settings(_env_file=None)
    embedder = make_fake_embedder(model_name=settings.embedding_model)
    monkeypatch.setattr(index, "get_settings", lambda: settings)
    monkeypatch.setattr(index, "connect", lambda _settings: client)
    monkeypatch.setattr(index, "build_embedder", lambda _settings: embedder)

    assert index.main(["--cache-dir", str(tmp_path)]) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"collection: {settings.collection_name}",
        "points: 676",
        "dimension: 8",
    ]
    assert client.count(settings.collection_name, exact=True).count == 676
    assert (tmp_path / f"{model_key(settings.embedding_model)}.npz").is_file()


def test_main_reports_a_missing_chunks_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(index, "get_settings", lambda: Settings(_env_file=None))
    assert index.main(["--chunks", str(tmp_path / "missing.jsonl")]) == 1
    assert "uv run python -m ingestion.build" in capsys.readouterr().err


def test_main_reports_invalid_settings(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    monkeypatch.setattr(index, "connect", lambda _settings: pytest.fail("connected"))
    assert index.main(["--cache-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "embedding_batch_size" in captured.err
    assert captured.out == ""


@pytest.mark.filterwarnings("ignore:Failed to obtain server version")
def test_main_reports_an_unreachable_server(monkeypatch, tmp_path, capsys):
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    monkeypatch.setattr(index, "get_settings", lambda: settings)
    monkeypatch.setattr(index, "build_embedder", lambda _settings: pytest.fail("model was loaded"))
    assert index.main(["--cache-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "Qdrant is not reachable at http://127.0.0.1:9" in captured.err
    assert captured.out == ""


def test_main_reports_a_server_failure_during_the_rebuild(
    monkeypatch, client, make_fake_embedder, tmp_path, capsys
):
    settings = Settings(_env_file=None)

    def fail(*args, **kwargs):
        raise ResponseHandlingException(OSError("connection reset"))

    monkeypatch.setattr(client, "upsert", fail)
    monkeypatch.setattr(index, "get_settings", lambda: settings)
    monkeypatch.setattr(index, "connect", lambda _settings: client)
    monkeypatch.setattr(
        index, "build_embedder", lambda _settings: make_fake_embedder(settings.embedding_model)
    )

    assert index.main(["--cache-dir", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "rerun: uv run python -m ingestion.index" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
```

- [ ] **Step 4: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_store.py tests/test_index.py -v`
Expected: lỗi khi thu thập test (`ModuleNotFoundError`), vì `app.rag.store` chưa tồn tại.

- [ ] **Step 5: Viết `app/rag/store.py` và `ingestion/index.py` theo các quy tắc trên**

Trong `ingestion/index.py`, import `get_settings`, `connect` và `build_embedder` ở đầu module dưới đúng các tên đó, vì test thay chúng bằng `monkeypatch.setattr(index, ...)`. Kết thúc file bằng `if __name__ == "__main__": raise SystemExit(main())`.

- [ ] **Step 6: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_store.py tests/test_index.py -v`
Expected: toàn bộ pass; `--collect-only -q` liệt kê ít nhất 7 test cho `tests/test_store.py` và 31 test cho `tests/test_index.py`. Các test về server không kết nối được mất khoảng vài giây mỗi test.

- [ ] **Step 7: Chạy eval sau thay đổi và ghi `eval/results.md`**

Run: `uv run python eval/run_eval.py` (thoát mã 0, không in số liệu). Thêm vào bảng ở `eval/results.md`:

```markdown
| <ngày chạy> | Task 4: nạp chunk vào Qdrant | N/A | N/A | N/A | chưa đo recall |
```

- [ ] **Step 8: Chạy lệnh kiểm tra chung rồi commit**

Run: `uv run pytest` và hai lệnh ruff; rồi `git status --short` và `git diff --stat`, đối chiếu với mục **Files**.

```bash
git add app/rag/store.py ingestion/index.py tests/test_store.py tests/test_index.py eval/results.md
git commit -m "feat: index chunk embeddings into a Qdrant collection"
```

---

### Task 5: Hàm `retrieve` và lệnh tìm kiếm

**Tier:** Trung bình.

**Mục tiêu:** `retrieve(query, k)` trả về các chunk gần nghĩa nhất kèm điểm, kiểm tra đầu vào và collection trước khi tải model; lệnh `uv run python -m app.rag.search "câu hỏi"` in kết quả ra terminal và báo lỗi Qdrant bằng một thông báo ngắn.

**Files:**
- Create: `app/rag/retriever.py`, `app/rag/search.py`, `tests/test_retriever.py`, `tests/test_integration_retrieval.py`
- Modify: `eval/results.md`

**Interfaces:**
- Consumes: `Embedder`, `build_embedder`, `connect`, `qdrant_errors`, `QdrantUnavailableError`, `get_settings`, `index_chunks`, `embed_with_cache`, `Chunk`
- Produces:
  - `app.rag.retriever.RetrievedChunk` (pydantic): `chunk: Chunk`, `score: float`
  - `app.rag.retriever.InvalidQueryError` (kế thừa `ValueError`) và `CollectionNotIndexedError` (kế thừa `RuntimeError`)
  - `app.rag.retriever.prepare_query(query: str, k: int) -> str`: chuẩn hóa NFC, strip, kiểm tra; trả câu hỏi đã chuẩn hóa
  - `app.rag.retriever.LazyEmbedder(factory: Callable[[], Embedder])` với `embed_query(text) -> np.ndarray`: dựng embedder thật ở lần dùng đầu tiên
  - `app.rag.retriever.Retriever(client: QdrantClient, embedder, collection: str)` với `retrieve(query: str, k: int = 5) -> list[RetrievedChunk]`; `embedder` chỉ cần có `embed_query`; lưu `client`, `embedder`, `collection` thành thuộc tính công khai (test thay phương thức của `retriever.client`)
  - `app.rag.retriever.default_retriever() -> Retriever` (có `lru_cache`)
  - `app.rag.retriever.retrieve(query: str, k: int = 5) -> list[RetrievedChunk]`
  - `app.rag.search.main(argv: list[str] | None = None) -> int`
  - Lệnh: `uv run python -m app.rag.search "câu hỏi" [-k N]`

**Tiêu chí nghiệm thu:** `uv run pytest` pass toàn bộ (tối thiểu 44 + 26 + 8 + 33 + 7 + 31 + 28 = 177 test, với 28 là số test tối thiểu của `tests/test_retriever.py`); `uv run pytest -m model` pass (tối thiểu 8 test model gồm Task 2) trên máy có model; test không tải model khi câu hỏi không hợp lệ, Qdrant không với tới được, hoặc collection chưa có; hai lệnh ruff sạch; `eval/results.md` có dòng của Task 5; commit của task.

**Lệnh kiểm tra:** `uv run pytest`, `uv run pytest tests/test_retriever.py --collect-only -q`, `uv run pytest -m model`, `uv run python eval/run_eval.py`, hai lệnh ruff.

**Ngoài phạm vi:** `generate`, ngưỡng từ chối, hybrid, reranker, tự kết nối lại khi retriever đã cache mất kết nối (lệnh một lần thoát ngay nên không cần), đổi hành vi NFC và top-k đã mô tả.

Quy tắc của `prepare_query(query, k)`:

1. Chuẩn hóa câu hỏi: `unicodedata.normalize("NFC", query).strip()`.
2. Câu hỏi rỗng sau chuẩn hóa: `InvalidQueryError("query is empty")`. `k < 1`: `InvalidQueryError("k must be at least 1")`.

Quy tắc của `Retriever.retrieve`, theo đúng thứ tự:

1. `prepare_query(query, k)`.
2. Bước 2 và 3 nằm trong **một** `with qdrant_errors():` (không bọc từng lời gọi; lỗi không phải của Qdrant, như lỗi embedding hay `CollectionNotIndexedError`, bay qua nguyên vẹn). Kiểm tra collection: chưa tồn tại thì `CollectionNotIndexedError(f"collection {collection} does not exist. Create it with: uv run python -m ingestion.index")`. Việc này diễn ra **trước** khi embedding.
3. Embedding câu hỏi bằng `embedder.embed_query`, rồi gọi `client.query_points(collection, query=vector.tolist(), limit=k, with_payload=True).points`.
4. Mỗi kết quả thành `RetrievedChunk(chunk=Chunk.model_validate(hit.payload), score=hit.score)`, giữ thứ tự Qdrant trả về (điểm cao trước).

`default_retriever()` lấy `get_settings()` rồi dựng `Retriever(connect(settings), LazyEmbedder(lambda: build_embedder(settings)), settings.collection_name)`, nên model chỉ được nạp khi một câu hỏi hợp lệ đã qua bước kiểm tra collection. Hàm `retrieve` cấp module gọi `prepare_query(query, k)` **trước**, rồi `default_retriever().retrieve(query, k)`, để câu hỏi hay `k` không hợp lệ không chạm tới kết nối hay model. Trong `app/rag/retriever.py`, import `get_settings`, `connect` và `build_embedder` ở đầu module dưới đúng các tên đó, vì test thay chúng bằng `monkeypatch`.

Quy tắc của `search.main`:

- Reconfigure stdout và stderr sang UTF-8 như ở Task 4. Tham số: `query` (bắt buộc) và `-k` (mặc định 5, kiểu `int`).
- Gọi hàm `retrieve` cấp module, import ở đầu file dưới đúng tên đó vì một test thay nó bằng `monkeypatch`.
- Gặp `QdrantUnavailableError`, `CollectionNotIndexedError` hoặc `InvalidQueryError`: in thông báo ra stderr, trả về `1`, không in gì ra stdout. Gặp `pydantic.ValidationError` (cấu hình sai): in `invalid settings: ...` ra stderr, trả về `1`. Không bắt lỗi khác.
- Mỗi kết quả in hai dòng, rồi trả về `0`:

```
{hạng}. {điểm:.4f}  {chunk.id}  Điều {chunk.article}. {chunk.article_title}
   {160 ký tự đầu của chunk.text sau khi gộp mọi khoảng trắng thành một dấu cách}
```

- [ ] **Step 1: Chạy eval trước thay đổi**

Run: `uv run python eval/run_eval.py`
Expected: thoát mã 0, không in số liệu.

- [ ] **Step 2: Viết test thất bại trong `tests/test_retriever.py`**

```python
import unicodedata
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import Settings
from app.rag import retriever as retriever_module
from app.rag import search
from app.rag.retriever import (
    CollectionNotIndexedError,
    InvalidQueryError,
    RetrievedChunk,
    Retriever,
    default_retriever,
    retrieve,
)
from app.rag.store import QdrantUnavailableError
from ingestion.index import index_chunks


class NoEmbedder:
    """An embedder that must never be reached."""

    def embed_query(self, text):
        pytest.fail("the query was embedded")


def fail_transport(*args, **kwargs):
    raise ResponseHandlingException(OSError("connection reset"))


@pytest.fixture(autouse=True)
def fresh_default_retriever():
    default_retriever.cache_clear()
    yield
    default_retriever.cache_clear()


@pytest.fixture
def client():
    qdrant = QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


@pytest.fixture
def retriever(client, sample_chunks, fake_embedder):
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in sample_chunks])
    index_chunks(sample_chunks, vectors, client, "test")
    return Retriever(client, fake_embedder, "test")


@pytest.fixture
def wired(monkeypatch, client, make_fake_embedder):
    """Point default_retriever() at an in-memory Qdrant and a fake embedder, counting use."""
    settings = Settings(_env_file=None)
    state = SimpleNamespace(
        settings=settings,
        client=client,
        embedder=make_fake_embedder(model_name=settings.embedding_model),
        connects=0,
        model_loads=0,
    )

    def fake_connect(_settings):
        state.connects += 1
        return state.client

    def fake_build_embedder(_settings):
        state.model_loads += 1
        return state.embedder

    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(retriever_module, "connect", fake_connect)
    monkeypatch.setattr(retriever_module, "build_embedder", fake_build_embedder)
    return state


@pytest.fixture
def indexed(wired, sample_chunks):
    texts = [chunk.context_text for chunk in sample_chunks]
    vectors = wired.embedder.embed_documents(texts)
    index_chunks(sample_chunks, vectors, wired.client, wired.settings.collection_name)
    return wired


# --- Retriever class -------------------------------------------------------------


def test_the_identical_text_is_the_top_hit(retriever, sample_chunks):
    target = sample_chunks[7]
    results = retriever.retrieve(target.context_text, k=3)
    assert len(results) == 3
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk == target
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_results_are_sorted_best_first(retriever, sample_chunks):
    scores = [result.score for result in retriever.retrieve(sample_chunks[0].context_text, k=12)]
    assert scores == sorted(scores, reverse=True)


def test_default_k_is_five(retriever, sample_chunks):
    assert len(retriever.retrieve(sample_chunks[0].context_text)) == 5


def test_k_larger_than_the_collection_returns_every_chunk(retriever, sample_chunks):
    results = retriever.retrieve(sample_chunks[0].context_text, k=100)
    assert {result.chunk.id for result in results} == {chunk.id for chunk in sample_chunks}


def test_query_is_stripped_and_normalised_to_nfc(retriever, sample_chunks):
    target = sample_chunks[2]
    decomposed = unicodedata.normalize("NFD", target.context_text)
    assert decomposed != target.context_text
    results = retriever.retrieve(f"  {decomposed}\n", k=1)
    assert results[0].chunk.id == target.id
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_raises(retriever, query):
    with pytest.raises(InvalidQueryError, match="query is empty"):
        retriever.retrieve(query)


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(retriever, k):
    with pytest.raises(InvalidQueryError, match="k must be at least 1"):
        retriever.retrieve("thử việc", k=k)


def test_missing_collection_raises_before_the_query_is_embedded(client):
    retriever = Retriever(client, NoEmbedder(), "labor_law_vn__missing")
    with pytest.raises(CollectionNotIndexedError, match="uv run python -m ingestion.index"):
        retriever.retrieve("thử việc")


@pytest.mark.parametrize("method", ["collection_exists", "query_points"])
def test_a_server_failure_becomes_qdrant_unavailable(retriever, monkeypatch, method):
    monkeypatch.setattr(retriever.client, method, fail_transport)
    with pytest.raises(QdrantUnavailableError):
        retriever.retrieve("thử việc")


# --- public retrieve() and default_retriever() -----------------------------------


def test_retrieve_returns_results_and_reuses_the_cached_retriever(indexed, sample_chunks):
    target = sample_chunks[7]
    first = retrieve(target.context_text, k=2)
    second = retrieve(target.context_text, k=2)
    assert first[0].chunk == target
    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]
    assert indexed.connects == 1
    assert indexed.model_loads == 1


def test_retrieve_normalises_a_decomposed_query(indexed, sample_chunks):
    target = sample_chunks[2]
    decomposed = unicodedata.normalize("NFD", target.context_text)
    assert retrieve(decomposed, k=1)[0].chunk.id == target.id


@pytest.mark.parametrize("query", ["", "   "])
def test_retrieve_rejects_an_empty_query_before_connecting(wired, query):
    with pytest.raises(InvalidQueryError, match="query is empty"):
        retrieve(query)
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_retrieve_rejects_a_small_k_before_connecting(wired):
    with pytest.raises(InvalidQueryError, match="k must be at least 1"):
        retrieve("thử việc", k=0)
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_retrieve_reports_a_missing_collection_without_loading_the_model(wired):
    with pytest.raises(CollectionNotIndexedError, match="uv run python -m ingestion.index"):
        retrieve("thử việc")
    assert wired.model_loads == 0


@pytest.mark.filterwarnings("ignore:Failed to obtain server version")
def test_retrieve_reports_an_unreachable_server_without_loading_the_model(monkeypatch):
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        retriever_module, "build_embedder", lambda _settings: pytest.fail("model was loaded")
    )
    with pytest.raises(QdrantUnavailableError, match="docker compose up -d qdrant"):
        retrieve("thử việc")


# --- search command ---------------------------------------------------------------


def test_search_cli_prints_ranked_results(indexed, sample_chunks, capsys):
    target = sample_chunks[7]

    assert search.main([target.context_text, "-k", "2"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[0] == f"1. 1.0000  {target.id}  Điều {target.article}. {target.article_title}"
    assert lines[1] == "   " + " ".join(target.text.split())[:160]
    assert lines[2].startswith("2. ")


def test_search_cli_reports_errors_without_a_traceback(monkeypatch, capsys):
    def not_indexed(_query, _k):
        raise CollectionNotIndexedError("collection x does not exist. Create it with: ...")

    monkeypatch.setattr(search, "retrieve", not_indexed)
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "collection x does not exist. Create it with: ..."


@pytest.mark.parametrize(
    ("args", "message"), [(["   "], "query is empty"), (["x", "-k", "0"], "k must be at least 1")]
)
def test_search_cli_rejects_a_bad_query_before_connecting(wired, capsys, args, message):
    assert search.main(args) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert captured.out == ""
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_search_cli_reports_a_missing_collection(wired, capsys):
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "uv run python -m ingestion.index" in captured.err
    assert captured.out == ""
    assert wired.model_loads == 0


@pytest.mark.filterwarnings("ignore:Failed to obtain server version")
def test_search_cli_reports_an_unreachable_server(monkeypatch, capsys):
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        retriever_module, "build_embedder", lambda _settings: pytest.fail("model was loaded")
    )
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "docker compose up -d qdrant" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_search_cli_reports_a_server_that_fails_after_the_retriever_was_cached(
    indexed, sample_chunks, monkeypatch, capsys
):
    text = sample_chunks[0].context_text
    assert search.main([text]) == 0
    capsys.readouterr()

    monkeypatch.setattr(indexed.client, "query_points", fail_transport)
    assert search.main([text]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_search_cli_reports_invalid_settings(monkeypatch, capsys):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "embedding_batch_size" in captured.err
    assert captured.out == ""
```

- [ ] **Step 3: Viết test tích hợp trong `tests/test_integration_retrieval.py`**

Các Điều mong đợi lấy từ bảng quan sát lịch sử. Test dùng Qdrant trong bộ nhớ nên không cần Docker; chỉ cần model.

```python
"""Needs the real embedding model, not Docker. Run with: uv run pytest -m model"""

import pytest
from qdrant_client import QdrantClient

from app.rag.embedder import SentenceTransformerEmbedder
from app.rag.retriever import Retriever
from ingestion.embed_cache import embed_with_cache
from ingestion.index import index_chunks

pytestmark = [pytest.mark.integration, pytest.mark.model]


@pytest.fixture(scope="module")
def real_retriever(all_chunks, tmp_path_factory):
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    cache_dir = tmp_path_factory.mktemp("embeddings")
    vectors = embed_with_cache([chunk.context_text for chunk in all_chunks], embedder, cache_dir)
    client = QdrantClient(":memory:")
    index_chunks(all_chunks, vectors, client, "integration")
    yield Retriever(client, embedder, "integration")
    client.close()


@pytest.mark.parametrize(
    ("question", "article"),
    [
        ("Thử việc tối đa bao lâu?", 25),
        ("Người lao động muốn nghỉ việc phải báo trước bao nhiêu ngày?", 35),
        ("Nghỉ thai sản được mấy tháng?", 139),
        ("Một năm được nghỉ phép bao nhiêu ngày?", 113),
        ("Làm thêm giờ được trả lương như thế nào?", 98),
        ("Tuổi nghỉ hưu là bao nhiêu?", 169),
    ],
)
def test_everyday_questions_find_the_right_article(real_retriever, question, article):
    results = real_retriever.retrieve(question, k=5)
    assert article in {result.chunk.article for result in results}


def test_an_unrelated_question_scores_lower_than_a_relevant_one(real_retriever):
    # One negative query is an observation, not a basis for a refusal threshold (Week 3).
    on_topic = real_retriever.retrieve("Thử việc tối đa bao lâu?", k=1)[0].score
    off_topic = real_retriever.retrieve("Cách nấu phở bò ngon", k=1)[0].score
    assert off_topic < on_topic
```

- [ ] **Step 4: Chạy test, xác nhận thất bại**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: lỗi khi thu thập test (`ImportError`), vì `app.rag.search` và `app.rag.retriever` chưa tồn tại.

- [ ] **Step 5: Viết `app/rag/retriever.py` và `app/rag/search.py` theo các quy tắc trên**

- [ ] **Step 6: Chạy test, xác nhận qua**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: toàn bộ pass; `--collect-only -q` liệt kê ít nhất 28 test.

Run: `uv run pytest -m model`
Expected: pass (tối thiểu 8 test: 1 của Task 2, 6 câu hỏi hằng ngày và 1 câu ngoài phạm vi), mất khoảng nửa phút trên máy có GPU. Không cần Docker.

Nếu một câu hỏi không tìm ra Điều mong đợi trong top 5, đừng sửa bảng câu hỏi. Hãy kiểm tra xem `context_text` có được dùng để embedding không và vector có được chuẩn hóa không, rồi báo người điều phối nếu vẫn lệch.

- [ ] **Step 7: Chạy eval sau thay đổi và ghi `eval/results.md`**

Run: `uv run python eval/run_eval.py` (thoát mã 0, không in số liệu). Thêm vào bảng ở `eval/results.md`:

```markdown
| <ngày chạy> | Task 5: retrieve dense và lệnh search | N/A | N/A | N/A | chưa đo recall |
```

- [ ] **Step 8: Chạy lệnh kiểm tra chung rồi commit**

Run: `uv run pytest` và hai lệnh ruff; rồi `git status --short` và `git diff --stat`, đối chiếu với mục **Files**.

```bash
git add app/rag/retriever.py app/rag/search.py tests/test_retriever.py tests/test_integration_retrieval.py eval/results.md
git commit -m "feat: add dense retrieval and a command line search"
```

---

### Task 6: Chạy thật với Qdrant trong Docker và cập nhật tài liệu

**Tier:** Trung bình.

**Mục tiêu:** Chứng minh cả chuỗi chạy được trên Qdrant thật, dữ liệu còn nguyên sau khi container bị xóa và tạo lại, và ghi lại cách chạy cùng kết quả thật vào tài liệu mà không khẳng định đã đo recall hay xong cả Tuần 2.

**Files:**
- Create: `tests/test_integration_server.py`
- Modify: `README.md`, `KE_HOACH_RAG.md` (chỉ ô đầu tiên của Tuần 2), `eval/results.md`

**Interfaces:**
- Consumes: mọi thứ ở Task 1–5.

**Tiêu chí nghiệm thu:**

- `uv run pytest -m server` pass (tối thiểu 1 test) với Qdrant đang chạy và không cần model; `uv run pytest -m model` pass không cần Docker; `uv run pytest -m integration` pass cả hai nhóm; `uv run pytest` (mặc định) không chạy test integration và pass.
- Lệnh index in `points: 676` và số chiều thực tế của model; số điểm vẫn là 676 sau `docker compose down` (không `-v`) rồi `docker compose up -d qdrant`.
- Lệnh search cho mỗi câu trong bảng 6 câu trong phạm vi có Điều mong đợi nằm trong top 5 (ghi hạng thực tế).
- README có bảng kết quả lấy từ lần chạy thật; `eval/results.md` có dòng của Task 6 ghi `N/A` và `chưa đo recall`; commit của task.

**Lệnh kiểm tra:** `uv run pytest`, `uv run pytest -m model`, `uv run pytest -m server`, `uv run pytest -m integration`, `uv run python eval/run_eval.py`, hai lệnh ruff.

**Ngoài phạm vi:** đo recall hay chọn ngưỡng từ chối (Tuần 3); `generate`; đánh dấu xong toàn bộ Tuần 2; `docker compose down -v` hay xóa volume.

- [ ] **Step 1: Kiểm tra Docker đang chạy**

Run: `docker info --format "{{.ServerVersion}}"`
Expected: in ra một số phiên bản. Nếu báo lỗi kết nối thì dừng lại và báo người điều phối nhờ người dùng mở Docker Desktop. Không tự cài hay khởi động Docker.

- [ ] **Step 2: Khởi động Qdrant**

Run: `docker compose up -d qdrant`, rồi kiểm tra kết nối bằng chính code của dự án:

`uv run python -c "from app.core.config import get_settings; from app.rag.store import connect; print(connect(get_settings()).get_collections())"`

Expected: in ra một dòng bắt đầu bằng `collections=`, không có lỗi. Lần đầu Docker phải tải image nên có thể cần chờ vài giây rồi chạy lại.

- [ ] **Step 3: Viết `tests/test_integration_server.py`**

Test dùng URL cục bộ cố định (không đọc `.env`), một tên collection dùng một lần cho mỗi test, và chỉ xóa collection do chính test tạo.

```python
"""Needs Qdrant running (docker compose up -d qdrant). Run with: uv run pytest -m server"""

import uuid
import warnings

import pytest
from qdrant_client.http.exceptions import ApiException

from app.core.config import Settings
from app.rag.retriever import Retriever
from app.rag.store import connect
from ingestion.index import index_chunks

pytestmark = [pytest.mark.integration, pytest.mark.server]

LOCAL_QDRANT_URL = "http://127.0.0.1:6333"


@pytest.fixture
def server_client():
    client = connect(Settings(_env_file=None, qdrant_url=LOCAL_QDRANT_URL))
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def collection(server_client):
    name = f"labor_law_vn__pytest_{uuid.uuid4().hex}"
    yield name
    try:  # cleanup must not hide the failure of the test itself
        server_client.delete_collection(name)
    except ApiException as error:
        warnings.warn(f"could not delete the test collection {name}: {error}", stacklevel=1)


def test_round_trip_through_the_qdrant_server(
    server_client, collection, sample_chunks, fake_embedder
):
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in sample_chunks])
    assert index_chunks(sample_chunks, vectors, server_client, collection) == 12
    results = Retriever(server_client, fake_embedder, collection).retrieve(
        sample_chunks[4].context_text, k=1
    )
    assert results[0].chunk == sample_chunks[4]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
```

- [ ] **Step 4: Chạy các nhóm test**

Run từng lệnh và ghi kết quả:

1. `uv run pytest`: mặc định, loại test `integration`; pass.
2. `uv run pytest -m model`: tối thiểu 8 test (1 ở `test_integration_embedder.py`, 7 ở `test_integration_retrieval.py`), không cần Docker.
3. `uv run pytest -m server`: tối thiểu 1 test, không tải model; cần Qdrant đang chạy.
4. `uv run pytest -m integration`: cả hai nhóm cùng lúc; số test bằng tổng hai nhóm trên.
5. `uv run pytest -m "integration and not model and not server" --collect-only -q`: phải không chọn test nào (pytest thoát mã 5, "no tests collected"); nếu chọn được test thì có test integration thiếu marker `model` hoặc `server`.

- [ ] **Step 5: Nạp dữ liệu thật, hai lần**

Run: `uv run python -m ingestion.index`, đo thời gian (ví dụ `time uv run python -m ingestion.index` trong bash).
Expected: ba dòng, trong đó `collection:` là `labor_law_vn__baai-bge-m3--d790e737453c`, `points: 676` và `dimension:` là số chiều thực tế của model (lịch sử báo 1024):

```
collection: labor_law_vn__baai-bge-m3--d790e737453c
points: 676
dimension: <số chiều thực tế>
```

Chạy lại đúng lệnh đó lần thứ hai. Output phải giống hệt. Ghi thời gian chạy của cả hai lần vào báo cáo như số liệu quan sát; **không** dùng "lần hai phải nhanh hơn" làm tiêu chí nghiệm thu (dễ sai lệch do máy). Việc cache không tính lại đã được test đơn vị của Task 3 chứng minh; file cache nằm ở `.cache/embeddings/<model_key>.npz`.

- [ ] **Step 6: Kiểm tra named volume giữ được dữ liệu**

Khởi động lại container không chứng minh được volume; phải xóa container rồi tạo lại. Run: `docker compose down` (**không** thêm `-v`, vì `-v` xóa volume), rồi `docker compose up -d qdrant`, đợi vài giây, rồi:

`uv run python -c "from app.core.config import get_settings; from app.rag.store import connect; s = get_settings(); print(connect(s).count(s.collection_name, exact=True).count)"`

Expected: in `676` (dữ liệu nằm trong named volume `qdrant_storage`, không phải trong container). Nếu lệnh kết nối lỗi, chờ thêm rồi chạy lại. Đây là bước kiểm tra thủ công, không thêm hạ tầng pytest.

- [ ] **Step 7: Tìm kiếm thử 7 câu hỏi**

Chạy `uv run python -m app.rag.search "<câu hỏi>"` cho từng câu trong bảng quan sát lịch sử.

Expected: với 6 câu trong phạm vi, Điều mong đợi có trong 5 dòng kết quả (ghi hạng thực tế, tính theo lần xuất hiện đầu tiên của Điều); dòng thứ nhất của câu "Thử việc tối đa bao lâu?" nên chứa `45-2019-QH14:d25:k1  Điều 25. Thời gian thử việc` (so khớp ý nghĩa, không đòi trùng tiền tố điểm). Với mỗi câu ghi lại kết quả đứng đầu, điểm thật của nó và hạng của Điều mong đợi; điểm là số quan sát thật, không phải số so khớp với bảng lịch sử. Nếu một câu không có Điều mong đợi trong top 5, đừng sửa bảng câu hỏi: kiểm tra `context_text` và chuẩn hóa vector rồi báo người điều phối. Câu ngoài phạm vi chỉ ghi điểm của kết quả đứng đầu.

- [ ] **Step 8: Cập nhật `README.md`**

Viết bằng tiếng Việt, mọi con số lấy từ lần chạy ở Step 5–7:

- Đổi dòng trạng thái thành: xong Tuần 1 và nửa đầu Tuần 2 (Qdrant và truy xuất dense), chưa có phần sinh câu trả lời.
- Bổ sung mục "Cách chạy": `uv sync --locked`, `docker compose up -d qdrant`, `uv run python -m ingestion.index`, `uv run python -m app.rag.search "Thử việc tối đa bao lâu?"`, `uv run pytest -m model` (cần model, không cần Docker) và `uv run pytest -m server` (cần Qdrant, không cần model).
- Thêm mục "Truy xuất thử" với bảng 7 câu hỏi (câu hỏi, kết quả đứng đầu, điểm, hạng của Điều mong đợi) lấy từ lần chạy thật, kèm một câu ghi rõ đây là kiểm tra bằng mắt trên 7 câu chọn tay, chưa đo recall.
- Thêm vào "Hạn chế đã biết":
  1. Truy xuất mới chỉ dùng dense vector; câu hỏi về báo trước khi nghỉ việc xếp Điều 36 trên Điều 35 (nếu lần chạy thật cũng cho kết quả đó).
  2. Chưa có bước sinh câu trả lời và chưa có ngưỡng từ chối; câu hỏi ngoài phạm vi vẫn trả về 5 kết quả với điểm thấp hơn (ghi điểm thật đã đo). Một câu ngoài phạm vi chưa đủ để đặt ngưỡng.
  3. Máy không có GPU NVIDIA vẫn chạy được trên CPU (`EMBEDDING_DEVICE=cpu`) nhưng chậm hơn; `pyproject.toml` đang lấy PyTorch từ index `cu126`.
  4. Cache embedding chỉ đi theo tên model: đổi trọng số, revision hay cách mã hóa dưới cùng tên model thì phải xóa `.cache/embeddings/<model_key>.npz` và index lại. Chỉ chạy một tiến trình index tại một thời điểm; lần index bị ngắt giữa chừng có thể để collection thiếu điểm, chạy lại lệnh index là đủ.
  5. Cổng Qdrant chỉ mở trên `127.0.0.1`; dữ liệu nằm trong named volume `qdrant_storage`.

- [ ] **Step 9: Cập nhật `KE_HOACH_RAG.md` và `eval/results.md`**

`KE_HOACH_RAG.md`: ở mục "Tuần 2", đổi riêng dòng `- [ ] Chạy Qdrant bằng Docker, embedding toàn bộ chunk bằng bge-m3` thành `- [x] ...`. Dòng về `retrieve` và `generate` giữ nguyên `- [ ]` vì `generate` chưa làm và Tuần 2 chưa xong. Không đổi dòng nào khác.

`eval/results.md`: chạy `uv run python eval/run_eval.py` (thoát mã 0, không in số liệu) rồi thêm vào bảng, với số điểm thật của lần chạy:

```markdown
| <ngày chạy> | Task 6: chạy thật trên Qdrant (676 điểm, kiểm tra bằng mắt 7 câu, xem README) | N/A | N/A | N/A | chưa đo recall; không phải số liệu recall |
```

- [ ] **Step 10: Chạy lệnh kiểm tra chung rồi commit**

Run: `uv run pytest`, `uv run pytest -m integration` và hai lệnh ruff; rồi `git status --short` và `git diff --stat`, đối chiếu với mục **Files**.

```bash
git add tests/test_integration_server.py README.md KE_HOACH_RAG.md eval/results.md
git commit -m "test: cover the Qdrant server round trip and document Week 2 retrieval"
```

Kiểm tra: `git status --short` không in gì (`.cache/` đã được ignore), và `git log --oneline main..HEAD` có ít nhất 6 commit (mỗi task một commit trở lên). Để container Qdrant tiếp tục chạy cho người dùng.
