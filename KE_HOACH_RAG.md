# Kế hoạch project RAG: Trợ lý tra cứu pháp luật Việt Nam

> Project cá nhân cho CV ngành CNTT. Thời lượng dự kiến: 8 tuần, khoảng 10–15 giờ/tuần.
> Đề tài mặc định: hỏi đáp trên **Bộ luật Lao động 2019** (số 45/2019/QH14) và các nghị định hướng dẫn chính. Có thể đổi sang luật khác mà không phải đổi kiến trúc.

---

## 1. Mục tiêu và phạm vi

**Sản phẩm:** một web app cho phép người dùng hỏi bằng tiếng Việt tự nhiên (ví dụ "Thử việc tối đa bao lâu?") và nhận câu trả lời kèm trích dẫn Điều/Khoản gốc.

**Tiêu chí hoàn thành:**

- [ ] Trả lời có trích dẫn điều khoản, bấm vào xem được nguyên văn
- [ ] Từ chối trả lời khi không tìm thấy căn cứ trong dữ liệu
- [ ] Có bộ câu hỏi đánh giá (tối thiểu 100 câu) và bảng số liệu trước/sau mỗi cải tiến
- [ ] Chạy được bằng một lệnh `docker compose up`
- [ ] Có link demo online, README, video demo 1–2 phút

**Cố ý KHÔNG làm (để giữ phạm vi):** đăng nhập/phân quyền, fine-tune mô hình, phủ toàn bộ hệ thống pháp luật, app mobile.

---

## 2. Kiến trúc tổng quan

```
                 ┌──────────── INGESTION (chạy offline) ────────────┐
 Văn bản luật →  Làm sạch → Chunk theo Điều/Khoản → Embedding → Qdrant
                 └──────────────────────────────────────────────────┘

                 ┌──────────── QUERY (mỗi câu hỏi) ─────────────────┐
 Câu hỏi → Hybrid search (dense + sparse) → Rerank → Ghép prompt
         → LLM (streaming) → Câu trả lời + trích dẫn
                 └──────────────────────────────────────────────────┘
```

---

## 3. Stack công nghệ

| Lớp | Lựa chọn chính | Phương án thay thế | Ghi chú |
|---|---|---|---|
| Ngôn ngữ | Python 3.11+ | | Dùng `uv` hoặc `venv` để quản lý môi trường |
| Backend API | FastAPI | | Hỗ trợ sẵn streaming (SSE) và tự sinh tài liệu API |
| Vector DB | Qdrant (Docker) | PostgreSQL + pgvector | Qdrant hỗ trợ cả dense và sparse vector trong một collection |
| Embedding | `BAAI/bge-m3` | `AITeamVN/Vietnamese_Embedding` (fine-tune từ bge-m3 cho tiếng Việt) | bge-m3 sinh được cả dense lẫn sparse vector; nên thử cả hai mô hình và so số liệu |
| Reranker | `BAAI/bge-reranker-v2-m3` | Bỏ qua nếu máy yếu | Chạy được trên CPU nhưng chậm; chỉ rerank top 20–30 |
| LLM | API có gói miễn phí (Gemini, Groq...) | Ollama chạy local | Viết một lớp `LLMClient` trừu tượng để đổi nhà cung cấp dễ dàng |
| Frontend | Next.js (React) | Streamlit | Chọn Streamlit nếu muốn dồn thời gian cho phần RAG |
| Đánh giá | Script tự viết (recall@k, MRR) | RAGAS | Tự viết phần retrieval, dùng RAGAS cho chất lượng câu trả lời |
| Đóng gói | Docker Compose | | 3 service: api, qdrant, web |
| Triển khai | Hugging Face Spaces / Render / VPS | | Kiểm tra giới hạn RAM của gói miễn phí (bge-m3 cần khoảng 2–3 GB) |

**Nguyên tắc:** tự viết pipeline RAG bằng Python thuần + thư viện client, không phụ thuộc hoàn toàn vào LangChain/LlamaIndex. Mục đích là khi phỏng vấn giải thích được từng bước.

**Lưu ý:** phiên bản thư viện và gói miễn phí của các dịch vụ thay đổi thường xuyên. Kiểm tra lại trang chính thức trước khi cài, và ghim phiên bản trong `requirements.txt`.

---

## 4. Nguồn dữ liệu

### 4.1. Văn bản luật (kho tri thức)

| Nguồn | Mô tả | Lưu ý |
|---|---|---|
| **vbpl.vn** | Cơ sở dữ liệu quốc gia về văn bản pháp luật (Bộ Tư pháp) | Nguồn chính thức, nên ưu tiên. Tải thủ công vài văn bản là đủ cho phạm vi project |
| **`undertheseanlp/UTS_VLC`** (Hugging Face) | Bộ dữ liệu Hiến pháp, Bộ luật, Luật Việt Nam | **Không dùng cho Bộ luật Lao động**: kiểm tra ngày 12/11/2026, bản ghi `45/2019/QH14` (split `2026`) chứa nhầm nội dung văn bản khác |
| **`th1nhng0/vietnamese-legal-documents`** (Hugging Face) | Hơn 500 nghìn văn bản (luật, nghị định, thông tư...) | Dùng khi muốn mở rộng sang nghị định, thông tư. Đọc kỹ license trên trang dataset |
| thuvienphapluat.vn | Kho văn bản lớn, có tình trạng hiệu lực | Trang thương mại: chỉ dùng để đối chiếu, không crawl hàng loạt |

**Văn bản đề xuất cho bản đầu tiên:**

1. Bộ luật Lao động 2019 (45/2019/QH14), dùng bản hợp nhất 18/VBHN-VPQH ngày 12/02/2026 tải từ Công báo (xem `data/raw/SOURCES.md`)
2. Nghị định 145/2020/NĐ-CP (hướng dẫn về điều kiện lao động và quan hệ lao động)
3. Mở rộng sau: Luật Bảo hiểm xã hội, Luật Việc làm

Trước khi nạp, **kiểm tra tình trạng hiệu lực** của từng văn bản (còn hiệu lực, đã sửa đổi hay bị thay thế) và ghi ngày kiểm tra vào metadata.

### 4.2. Dữ liệu đánh giá

| Nguồn | Dùng để |
|---|---|
| **Tự soạn 100 câu hỏi** | Bộ test chính. Mỗi câu ghi rõ Điều/Khoản là đáp án đúng |
| `thangvip/vietnamese-legal-qa` (Hugging Face) | Tham khảo cách đặt câu hỏi, lấy thêm câu hỏi thuộc phạm vi luật đã chọn |
| Zalo Legal Text Retrieval (Zalo AI Challenge 2021) | Benchmark truy xuất văn bản luật phổ biến, dùng nếu muốn so sánh với kết quả đã công bố |
| Mục hỏi đáp trên các cổng thông tin của bộ, ngành | Nguồn câu hỏi thực tế của người dân |

**Cách soạn bộ test:** chia 3 nhóm — câu hỏi trực tiếp (40%), câu hỏi diễn đạt đời thường khác từ ngữ trong luật (40%), câu hỏi ngoài phạm vi để kiểm tra khả năng từ chối (20%).

---

## 5. Lộ trình 8 tuần

### Tuần 1 — Dữ liệu và chunking
- [x] Tạo repo GitHub, cấu trúc thư mục, môi trường Python
- [x] Tải Bộ luật Lao động, làm sạch (bỏ header/footer, chuẩn hóa Unicode về NFC)
- [x] Viết parser tách theo cấu trúc Chương → Mục → Điều → Khoản
- [x] Mỗi chunk lưu metadata: tên văn bản, số hiệu, chương, số điều, tên điều, số khoản
- **Kết quả:** file `chunks.jsonl`, in thử 20 chunk để kiểm tra bằng mắt

### Tuần 2 — RAG cơ bản (baseline)
- [ ] Chạy Qdrant bằng Docker, embedding toàn bộ chunk bằng bge-m3
- [ ] Viết hàm `retrieve(query, k)` và `generate(query, contexts)`
- [ ] Prompt yêu cầu: chỉ trả lời dựa trên ngữ cảnh, ghi rõ Điều/Khoản, nói "không tìm thấy" nếu thiếu căn cứ
- **Kết quả:** hỏi đáp chạy được trong terminal hoặc notebook

### Tuần 3 — Bộ đánh giá
- [ ] Soạn 100 câu hỏi + đáp án (điều khoản đúng)
- [ ] Viết script tính recall@5, recall@10, MRR cho phần truy xuất
- [ ] Đo baseline và lưu kết quả vào `eval/results.md`
- **Kết quả:** con số baseline đầu tiên. Từ đây mọi thay đổi đều phải đo lại

### Tuần 4 — Hybrid search
- [ ] Thêm sparse vector (từ bge-m3 hoặc BM25) vào Qdrant
- [ ] Kết hợp kết quả dense + sparse bằng Reciprocal Rank Fusion (RRF)
- [ ] Xử lý truy vấn chứa số điều ("Điều 24 quy định gì?") bằng lọc metadata
- **Kết quả:** bảng so sánh dense / sparse / hybrid

### Tuần 5 — Reranking và cải thiện chunking
- [ ] Thêm reranker cho top 20–30 kết quả
- [ ] Thử nghiệm: chunk theo Khoản so với theo Điều; thêm tên Điều và tên Chương vào đầu mỗi chunk
- [ ] Đánh giá chất lượng câu trả lời (faithfulness, có trích dẫn đúng không) bằng RAGAS hoặc chấm tay 30 câu
- **Kết quả:** chọn cấu hình tốt nhất dựa trên số liệu

### Tuần 6 — Backend API
- [ ] FastAPI: `POST /chat` (streaming SSE), `GET /articles/{id}`, `GET /health`
- [ ] Lịch sử hội thoại: viết lại câu hỏi nối tiếp thành câu hỏi độc lập trước khi truy xuất
- [ ] Log mỗi request: câu hỏi, chunk truy xuất, thời gian từng bước
- [ ] Viết test cho parser và retrieval (pytest)

### Tuần 7 — Frontend
- [ ] Giao diện chat, hiển thị câu trả lời dạng streaming
- [ ] Thẻ trích dẫn: bấm vào mở nguyên văn điều luật
- [ ] Dòng cảnh báo cố định: "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý"
- [ ] Nút đánh giá câu trả lời (hữu ích / không hữu ích), lưu lại để phân tích

### Tuần 8 — Đóng gói và trình bày
- [ ] Dockerfile + `docker-compose.yml`
- [ ] Deploy bản demo
- [ ] README: mô tả, sơ đồ kiến trúc, cách chạy, bảng kết quả đánh giá, hạn chế đã biết
- [ ] Quay video demo, viết mô tả project cho CV

---

## 6. Cấu trúc thư mục đề xuất

```
RAG/
├── data/
│   ├── raw/              # văn bản gốc
│   └── processed/        # chunks.jsonl
├── ingestion/            # làm sạch, parser, embedding, nạp vào Qdrant
├── app/
│   ├── api/              # FastAPI routes
│   ├── rag/              # retriever, reranker, generator, prompts
│   └── core/             # config, logging
├── eval/
│   ├── questions.jsonl   # bộ câu hỏi test
│   ├── run_eval.py
│   └── results.md        # bảng số liệu qua từng phiên bản
├── web/                  # frontend
├── tests/
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 7. Các lưu ý cơ bản

### Về dữ liệu và chunking
- **Không cắt chunk theo số ký tự cố định.** Văn bản luật có cấu trúc sẵn; cắt giữa một Khoản sẽ làm mất nghĩa. Chunk theo Điều hoặc Khoản.
- **Giữ ngữ cảnh cho chunk.** Một Khoản đứng riêng thường thiếu chủ ngữ. Thêm tên Điều (và tên Chương) vào đầu nội dung chunk trước khi embedding.
- **Chuẩn hóa Unicode tiếng Việt (NFC).** Cùng một chữ có thể được mã hóa theo hai cách (dựng sẵn và tổ hợp); không chuẩn hóa thì tìm kiếm từ khóa sẽ trượt.
- **Điều khoản dẫn chiếu** ("theo quy định tại Điều 35") là điểm khó. Ghi nhận là hạn chế, hoặc xử lý bằng cách truy xuất thêm điều được dẫn chiếu.

### Về truy xuất
- **Đánh giá retrieval riêng với generation.** Nếu câu trả lời sai, cần biết do truy xuất sai hay do LLM diễn giải sai.
- **Tìm theo từ khóa vẫn quan trọng** với số điều, thuật ngữ pháp lý chính xác. Vì vậy cần hybrid search.
- **Mỗi lần chỉ thay đổi một thứ** rồi đo lại. Ghi mọi kết quả vào `results.md`, kể cả thử nghiệm không hiệu quả.

### Về sinh câu trả lời
- **Chống bịa:** prompt bắt buộc trích dẫn; đặt ngưỡng điểm truy xuất, dưới ngưỡng thì từ chối thay vì trả lời.
- **Kiểm tra trích dẫn bằng code:** số Điều mà LLM nêu phải nằm trong các chunk đã truy xuất, nếu không thì cảnh báo.
- **Giới hạn số chunk đưa vào prompt** (5–8 chunk). Nhiều hơn không chắc tốt hơn mà tốn token.

### Về kỹ thuật và chi phí
- **Không commit API key.** Dùng file `.env`, thêm vào `.gitignore` ngay từ commit đầu tiên, cung cấp `.env.example`.
- **Cache embedding** ra đĩa để không phải tính lại mỗi lần chạy.
- **Theo dõi giới hạn gói miễn phí** của LLM API; thêm rate limit ở backend khi deploy công khai.
- **Máy không có GPU:** bge-m3 vẫn chạy được trên CPU cho vài nghìn chunk; có thể embedding trên Google Colab rồi tải vector về.
- **Commit thường xuyên** với message rõ ràng. Lịch sử commit đều đặn trong 8 tuần cũng là một điểm cộng.

### Về pháp lý và đạo đức
- Văn bản quy phạm pháp luật là tài liệu công khai, nhưng **bộ dữ liệu và trang web tổng hợp có điều khoản sử dụng riêng**: đọc license trước khi dùng, ghi nguồn trong README.
- Luôn hiển thị cảnh báo sản phẩm không thay thế tư vấn pháp lý.
- Ghi rõ ngày cập nhật dữ liệu vì luật có thể được sửa đổi.

---

## 8. Cách trình bày trong CV

**Mẫu mô tả (điền số liệu thật sau khi đo):**

> **Trợ lý tra cứu pháp luật lao động (RAG)** — Python, FastAPI, Qdrant, bge-m3, Next.js, Docker
> - Xây dựng pipeline RAG cho văn bản luật tiếng Việt với chunking theo cấu trúc Điều/Khoản và trích dẫn nguồn
> - Cải thiện recall@5 từ X% lên Y% nhờ hybrid search (dense + sparse, RRF) và reranking, đánh giá trên bộ 100 câu hỏi tự xây dựng
> - Triển khai API streaming và giao diện chat; đóng gói bằng Docker Compose. Demo: [link] · Mã nguồn: [link]

**Câu hỏi phỏng vấn nên chuẩn bị:**

1. Vì sao chọn chunk theo Điều/Khoản? Đã thử cách nào khác, kết quả ra sao?
2. Hybrid search hoạt động thế nào? RRF là gì?
3. Làm sao biết hệ thống tốt lên? Recall@k và MRR nghĩa là gì?
4. Hệ thống xử lý thế nào khi câu hỏi nằm ngoài dữ liệu?
5. Nếu dữ liệu tăng lên 100 lần thì cần thay đổi gì?
6. Hạn chế lớn nhất hiện tại là gì và bạn sẽ cải thiện ra sao?

---

## 9. Hướng mở rộng (nếu còn thời gian)

- Mở rộng sang nghị định, thông tư và xử lý quan hệ "văn bản hướng dẫn / sửa đổi"
- Query rewriting: dùng LLM diễn đạt lại câu hỏi đời thường thành ngôn ngữ pháp lý trước khi tìm
- Fine-tune embedding trên cặp câu hỏi – điều luật
- Trang thống kê: câu hỏi phổ biến, tỉ lệ đánh giá hữu ích, độ trễ trung bình

---

## 10. Tài liệu tham khảo

- Qdrant – tài liệu hybrid search: https://qdrant.tech/documentation/
- BAAI/bge-m3: https://huggingface.co/BAAI/bge-m3
- AITeamVN/Vietnamese_Embedding: https://huggingface.co/AITeamVN/Vietnamese_Embedding
- UTS_VLC: https://huggingface.co/datasets/undertheseanlp/UTS_VLC
- vietnamese-legal-documents: https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents
- vietnamese-legal-qa: https://huggingface.co/datasets/thangvip/vietnamese-legal-qa
- Bài báo "Improving Vietnamese Legal Document Retrieval using Synthetic Data": https://arxiv.org/abs/2412.00657
- RAGAS: https://docs.ragas.io/
- FastAPI: https://fastapi.tiangolo.com/
