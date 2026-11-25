# Tuần 4: Hybrid search (dense + BM25) và câu hỏi nêu số Điều (Design)

**Trạng thái:** đã duyệt ngày 26/11/2026, sau hai vòng review (Opus, approve-with-fixes, đã sửa). Người dùng duyệt spec và mọi quyết định ở mục 10 theo đề xuất. Trước khi duyệt, người điều phối đã lấy các đề xuất đó làm giả định làm việc cho ba task không phụ thuộc kết quả đo (S0, T1, T5a). Chưa có lần chạy sparse hay hybrid nào. Bản này là mốc pre-registration, commit trong PR S0, trước khi bắt đầu T2. Mục 5 và 6 cố định từ commit đó: sửa trước run (2) của T6 cần người dùng duyệt lại và commit mới; sửa sau đó ghi là thăm dò (mục 5.10).

## 1. Mục đích

Baseline dense (bge-m3, 22/11/2026) đã chạm trần recall@5 mức Điều (1,000) nhưng còn sai ở hạng 1: 13/45 câu đời thường (paraphrase) và 3/12 câu nêu số Điều không có Điều đúng ở hạng 1. Tuần 4 (`KE_HOACH_RAG.md`, mục 5) thêm nhánh từ khóa BM25, gộp với dense bằng RRF và xử lý câu hỏi nêu thẳng số Điều. Tài liệu này khóa trước cách so sánh và luật chốt, để kết quả trên bộ câu đóng băng không bị chỉnh theo số liệu.

Thuật ngữ:

- **BM25:** xếp hạng theo từ khóa. Từ càng hiếm trong corpus (df, số chunk chứa từ, càng nhỏ) thì càng nặng điểm (IDF).
- **Sparse vector:** vector chỉ lưu các term có mặt, dạng id → trọng số.
- **RRF** (Reciprocal Rank Fusion): gộp hai bảng xếp hạng bằng tổng 1/(k + hạng).
- **R@1:** tỷ lệ câu có Điều đúng ở hạng 1.
- **McNemar mid-p:** kiểm định cặp, chỉ dùng các câu mà hai hệ thống cho kết quả khác nhau.
- **Pre-registration:** ghi luật so sánh và luật chốt trước khi nhìn kết quả.
- **Run:** một lần chạy `eval/run_eval.py` trên bộ câu, cho một dòng `eval/results.md` và một file trong `eval/runs/`.

## 2. Mục tiêu, ngoài phạm vi, anti-goal

**Mục tiêu**

1. Collection mới gồm vector dense (sao nguyên từ cache) và sparse BM25; collection cũ giữ nguyên.
2. Retriever có ba chế độ: dense, sparse, hybrid.
3. Câu hỏi nêu số Điều của Bộ luật Lao động được ưu tiên chunk của Điều đó: promote rồi lấp bằng xếp hạng gốc (mục 6.4).
4. Bảng dense / sparse / hybrid và hai dòng có promotion trong `eval/results.md` và README, kèm quyết định theo mục 5.
5. `search` và `ask` dùng cấu hình đã chốt (T7).

**Ngoài phạm vi:** reranker, đổi chunking (Tuần 5), ngưỡng từ chối theo điểm, sparse của bge-m3, truy vấn không dấu (backlog Tuần 6), biến thể i/y ("lí"/"lý").

**Anti-goal:** chỉnh tham số theo bộ câu đóng băng; dùng `Fusion.RRF` của Qdrant (k = 2); đổi `score` hiển thị cho người dùng thành điểm RRF; gộp promotion vào dòng "hybrid".

## 3. Ràng buộc

- `AGENTS.md`: nguồn sự thật pháp lý, repo public, khóa API chỉ qua biến môi trường, comment và commit tiếng Anh, NFC, UTF-8, console Windows không UTF-8.
- Pipeline tự viết, không LangChain/LlamaIndex. BM25 tính phía client.
- Không thêm dependency cho thống kê (scipy không có trong môi trường).
- Bộ câu `577b8392afeb` (127 câu) giữ nguyên.

## 4. Hiện trạng đã kiểm chứng (26/11/2026)

- **Baseline mức Điều:** R@5 1,000; R@1 44/45 direct, 32/45 paraphrase, 9/12 article_ref; MRR@10 0,907 trên 90 câu. **Mức Khoản:** R@1 70/90, R@5 0,989, MRR 0,864.
- **Ba câu article_ref trượt hạng 1 đều là câu "trơn":** q093 "Điều 131 quy định những gì?" (hạng 5), q094 "Nội dung của Điều 140 là gì?" (hạng 10), q095 "Điều 198 nói về vấn đề gì?" (ngoài top 10). `ARTICLE_MENTION` (`app/rag/generate.py:25`) bắt đúng 12 câu article_ref và 0 câu khác.
- **Độ trùng âm tiết câu hỏi – đáp án** (tỷ lệ âm tiết của câu hỏi có trong `context_text` của chunk đáp án): direct 0,87, paraphrase 0,36, article_ref 0,73. Câu trực tiếp chép gần nguyên văn luật nên BM25 có lợi thế sẵn ở nhóm này; nhóm paraphrase là thước đo chính.
- **`context_text`** = tiêu đề văn bản + "Chương …" + "Mục …" (nếu có) + "Điều N. tiêu đề" + phần dẫn của Điều (chỉ chunk theo Khoản) + nội dung (`ingestion/chunker.py:8-16`). Mỗi chunk chứa số Điều của chính nó.
- **Trích dẫn chéo:** có 131 lần nhắc Điều khác trong 87 chunk, ví dụ "theo quy định tại Điều 35" ở `d34:k9`. BM25 cho truy vấn "Điều 35" vì thế cũng chấm điểm các chunk trích dẫn Điều 35.
- **Kiểu đặt dấu:** corpus dùng kiểu A ("hòa", "thỏa", "ủy": 500 token trong `text`); kiểu B ("hoà") chỉ có 1 token (`tuý`). Quy định kèm Quyết định 1989/QĐ-BGDĐT (2018) dùng kiểu B cho sách giáo khoa (đọc qua bản chép trên luatvietnam.vn), nên người dùng có thể gõ kiểu B.
- **bge-m3:** chịu được khác kiểu dấu (cosine 0,970–0,995 giữa bản A và B, 4 câu ngoài bộ đánh giá) nhưng không chịu được câu không dấu (0,36–0,42).
- **Từ để hỏi:** hiếm trong luật nên BM25 coi như từ nội dung. "nào" có df 10 (IDF 4,17); "bao" có df 32 (IDF 3,04, do "bao gồm"). Các cụm "bao nhiêu", "bao lâu", "thế nào" có trong 0 chunk. Các số này chỉ dùng văn bản câu hỏi và corpus, không dùng kết quả truy xuất.
- **Tokenizer v1 (mục 6.1)** trên `context_text` của 676 chunk: 8.653 term khác nhau, avgdl 171,90, id 32-bit không va chạm.
- **Qdrant 1.20.1** (thử trên collection tạm, đã xóa):
  - Một collection chứa được vector dense không tên cùng một sparse có tên.
  - `Modifier.IDF` nhân trọng số truy vấn với ln((N − df + 0,5)/(df + 0,5) + 1).
  - `Fusion.RRF` dùng k = 2 với hạng tính từ 0.
  - Bộ lọc không khớp trả 0 điểm; `MatchAny` trên payload số nguyên chạy được khi chưa có index.
  - BM25 phía server (`TextQuery`) không có tokenizer tiếng Việt, và `ascii_folding` bỏ dấu.
- **Cỡ mẫu:** để đạt power 80 % cần lãi ròng khoảng +6 đến +7 câu paraphrase ở hạng 1 ([S] §9.2). Nguyen & Quan (EACL 2026) thấy bge-m3 hybrid ≈ dense trên ZaloLegalQA. Kết quả "không hơn" là khả năng thật.

## 5. Pre-registration

1. **So sánh:** hybrid (mục 6) với run (1) của T6. Run (1) là dense trên collection mới, cùng commit, cùng thiết bị và server Qdrant với các run (2)–(5), chạy liền trước chúng trong một phiên, không đổi code hay dữ liệu giữa các run. Đo ở mức Điều trên 90 câu direct + paraphrase của bộ `577b8392afeb`; mức Khoản là phụ.
2. **Chỉ số quyết định (C1):** R@1 mức Điều của 45 câu paraphrase, run (3) so với run (1). Kiểm định McNemar hai phía, mid-p, α = 0,05 (`eval/stats.py`).
   - b là số câu run (3) đúng ở hạng 1 mà run (1) sai; c là số câu ngược lại.
   - "C1 có ý nghĩa" nghĩa là mid-p ≤ 0,05 **và** b > c.
   - Báo kèm b, c và p chính xác. MRR@10 và mức Khoản là chỉ số phụ, chỉ để báo cáo.
3. **Guard cho câu trực tiếp:** gọi D là tập câu direct có Điều đúng ở hạng 1 trong run (1) (44 câu nếu tái lập đạt). Guard đạt khi có đủ hai điều kiện:
   - Run (3) làm mất hạng 1 ở tối đa 1 câu trong D. Câu tụt từ hạng 1 xuống hạng 2–5 tính là mất hạng 1.
   - Không câu direct nào có Điều đúng trong top 5 ở run (1) mà ra ngoài top 5 ở run (3).

   Báo cận trên Clopper–Pearson hai phía 95 % của tỷ lệ mất, mẫu số là |D| (0/44 → 0,0804).
4. **Luật chốt hybrid** (quyết định kỹ thuật): chốt khi đạt guard VÀ (C1 có ý nghĩa HOẶC (b − c ≥ 3 và c ≤ 1)), với b, c tính trên 45 câu paraphrase. Không đạt, kể cả khi b = c, thì giữ dense.
5. **Luật chốt promotion** (D3): so run có promotion của cấu hình sẽ ship với run gốc của nó: (4) với (1) nếu giữ dense, (5) với (3) nếu chốt hybrid.
   - Bật promotion khi D3 được duyệt và đạt cả hai điều kiện: 115 câu không phải article_ref có top 10 trùng nhau trong dung sai tái lập (5.7), mọi chỗ đổi chỗ đều được báo; R@1 mức Điều của 12 câu article_ref không thấp hơn.
   - Báo như độ phủ của parser: R@1 a/12 → b/12, kèm hạng Khoản của q097 và q102. Không kiểm định thống kê trên 12 câu.
6. **Luật tuyên bố** (README, CV): chỉ ghi "hybrid cải thiện R@1 câu đời thường" khi C1 có ý nghĩa. Nếu không, ghi b, c, chênh lệch tuyệt đối và "không có ý nghĩa thống kê". Chỉ ghi chênh lệch tuyệt đối, không ghi %.
7. **Thứ tự chạy cố định:** (1) dense trên collection mới, (2) sparse, (3) hybrid, (4) dense + promotion, (5) hybrid + promotion.
   - Mỗi run cho một dòng `eval/results.md`, một file run và file `.meta.json` của nó. Promotion luôn là dòng riêng.
   - **Dung sai tái lập** của run (1) so với run 22/11/2026 có ba điều kiện:
     - `rank_article` và `rank_clause` giống hệt ở 102 câu trong phạm vi.
     - Ở cả 127 câu, top 10 cùng id theo cùng thứ tự, trừ các cặp đổi chỗ (kể cả ở biên hạng 10) có điểm chênh < 1e-4.
     - |Δ điểm| < 1e-4 cho mọi chunk có mặt ở cả hai top 10.
   - Run (1) không đạt dung sai thì dừng trước run (2) và báo người dùng.
   - Lỗi phát hiện ở bất kỳ run nào, kể cả sau run (2), được sửa nếu bản sửa không đổi mục 6 và luật ở mục 5, tức chỉ đưa code về đúng spec. Run lỗi giữ dòng, ghi nhãn "lỗi" kèm mô tả và commit sửa, rồi chạy lại cả chuỗi (1)–(5) trên commit mới. Mọi thay đổi khác ghi là thăm dò.
   - Lần chạy nào cũng ghi dòng.
8. **Không nhìn trước:** trước run (2), không chạy truy xuất sparse, hybrid hay promotion trên câu nào trong 127 câu của bộ đánh giá.
   - Smoke test và test integration của sparse, hybrid và promotion chỉ dùng câu ngoài bộ, không chép hay diễn đạt lại câu của bộ.
   - Chạy dense trên câu của bộ thì được (kiểm tra tái lập của T2, dòng eval gate), vì kết quả dense đã biết từ baseline.
   - Dòng eval gate của T3 và T4 chạy `run_eval` mặc định (dense, không promotion). Dòng đó phải trùng số của dòng baseline, cột Ghi chú ghi "chưa chạy hybrid (pre-registration)".
   - Output của run (2) là lần nhìn đầu tiên.
9. **Câu tụt hạng** (để báo cáo, không phải điều kiện đạt): T5b liệt kê câu tụt hạng của run mới so với run tham chiếu (`--reference`). T6 báo các cặp (3)/(1), (4)/(1), (5)/(3).
   - Một câu tụt hạng khi hạng Điều ở run mới lớn hơn ở run tham chiếu. Hạng None (ngoài top 10) tính là 11.
   - Nguyên nhân gán theo cờ của run mới. Mỗi câu nhận nguyên nhân đầu tiên khớp:
     - parser: run có promotion và tập Điều mà `article_refs` trả về cho câu hỏi khác `gold_articles`;
     - gần hòa điểm: run hybrid, chunk đúng còn trong top 10 và `fusion_score` của nó kém chunk ngay trên dưới 1e-4;
     - chunk khác trùng từ khóa: các câu còn lại của run hybrid, tức nhánh sparse đã đẩy chunk khác lên;
     - khác: mọi câu còn lại.
   - Run sparse chỉ liệt kê câu tụt hạng, không gán nguyên nhân.
   - Ngưỡng 1e-4 chọn trước, không hiệu chỉnh. Để so sánh: phần đóng góp RRF của hai hạng liền nhau trong một nhánh chênh khoảng 1/(60 + r)², tức khoảng 2,6e-4 giữa hạng 1 và 2.
10. **Hậu kiểm:** mục 5 và 6 cố định từ commit bản đã duyệt; sửa trước run (2) cần người dùng duyệt lại và commit mới. Thay đổi sau lần nhìn đầu tiên (5.8), trừ bản sửa lỗi theo 5.7, được ghi "thăm dò" và không dùng để tuyên bố. Không có bộ holdout (phương án khác: D2).
11. **Nếu hybrid không hơn:** vẫn ghi bảng so sánh và nói rõ kết quả. Dòng CV "recall@5 từ X% lên Y%" (`KE_HOACH_RAG.md`, mục 8) không dùng được vì recall@5 đã là 1,000; dòng này viết lại ở Tuần 8.
12. **Thăm dò X1** (sau T6, không dùng để tuyên bố): bỏ dấu 45 câu paraphrase bằng máy, đo dense và hybrid.

## 6. Tham số cố định

### 6.1. Tokenizer BM25 (`app/rag/sparse.py`, `TOKENIZER_VERSION = "vi-syl-bigram-v1"`)

1. NFC → lowercase → "ð" (U+00F0) thành "đ".
2. Tách đoạn tại mọi ký tự khác chữ cái, chữ số, dấu cách, tab và NBSP (U+00A0), ví dụ dấu câu, "/", "%", "-", xuống dòng. Dấu cách, tab và NBSP chỉ tách âm tiết, không ngắt đoạn.
3. Âm tiết là `\w+`. Số chỉ gồm chữ số thì bỏ số 0 đầu ("01" → "1", vẫn giữ "0"). Âm tiết khác qua phép gộp dấu B → A:
   - Chỉ áp dụng cho âm tiết mở kết thúc bằng oa/oe/uy có dấu ở nguyên âm sau ("hoà" → "hòa", "thuỷ" → "thủy").
   - Trừ khi đứng sau "q" ("quý" giữ nguyên). Âm tiết đóng không đổi ("hoàn", "khoản").
4. Term gồm âm tiết và bigram của hai âm tiết liền nhau trong cùng đoạn; bigram không nối qua chỗ ngắt.
5. **Chỉ phía truy vấn:**
   - Xóa cụm hỏi trước, cụm dài trước: "như thế nào", rồi "bao nhiêu", "thế nào", "bao lâu", "bao giờ", "tại sao", "vì sao". Sau đó xóa các từ hỏi "gì", "nào", "ai", "đâu", "mấy". Chỉ khớp nguyên từ.
   - Ở cuối câu, bỏ dấu câu rồi bỏ trợ từ "không", "chưa", "à", "ư", "nhỉ", "nhé", "hả", "ạ", "vậy". Lặp lại đến khi cuối câu không còn trợ từ.
   - Mỗi chỗ xóa để lại một dấu ngắt đoạn, nên không bigram nào nối qua. Dấu ngắt thừa ở đầu và cuối câu bị bỏ.
   - Không bao giờ xóa: chữ số; "khoản", "điều"; "bao", "thế", "sao" đứng riêng (giữ "bao gồm", "thay thế", "bản sao"); "không", "chưa" ở giữa câu ("không xác định thời hạn", "chưa thành niên").
   - Term truy vấn bỏ trùng (Qdrant từ chối index trùng), mỗi term trọng số 1,0. Câu không còn term nào cho vector rỗng.
6. Id = `int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")`. Digest dài 4 byte do tham số `digest_size`, không phải 4 byte đầu của digest mặc định 64 byte; hai cách cho id khác nhau. Không có tiền tố namespace, vì collection chỉ có một nhánh sparse và phiên bản tokenizer được kiểm qua metadata. Không dùng `hash()` của Python.
7. Văn bản hiển thị, trích dẫn và đầu vào parser số Điều không bao giờ qua phép gộp dấu.

### 6.2. BM25

- Tham số: k1 = 1,2; b = 0,75.
- Trọng số tài liệu: tf·(k1 + 1) / (tf + k1·(1 − b + b·dl/avgdl)). dl là số term của `context_text`; avgdl tính bằng chính tokenizer.
- Truy vấn: mỗi term trọng số 1,0. IDF do Qdrant áp (`Modifier.IDF`).

### 6.3. RRF

- Tính trong Python, k = 60, hạng tính từ 1.
- Dùng toàn bộ xếp hạng (dense: mọi điểm của collection, số điểm lấy từ metadata, hiện là 676; sparse: mọi chunk có điểm > 0), nên không có điểm gãy ở độ sâu cắt.
- Trọng số hai nhánh bằng nhau. Hòa điểm thì xếp theo hạng dense, rồi theo chunk id.
- Không dùng `Fusion.RRF` của server.

### 6.4. Promotion số Điều

- Số chunk đưa lên, tính chung cho mọi Điều được nêu: m = max(0, min(3, k − 2)). Với k ≤ 2 thì không promote.
- Chunk của các Điều được nêu xếp theo thứ tự trong xếp hạng gốc đầy đủ, rồi lấy m chunk đầu:
  - Hybrid: xếp hạng hợp nhất của mọi chunk (mục 6.3), có sẵn trước khi cắt k.
  - Dense: thêm một truy vấn dense cùng vector truy vấn, lọc `MatchAny` trên trường `article`, limit m. Thứ tự cosine của truy vấn này trùng thứ tự trong xếp hạng dense đầy đủ.
- Phần còn lại lấp bằng top k của xếp hạng gốc, bỏ trùng, cắt còn đúng k.
- Parser không kích hoạt thì kết quả giống hệt khi tắt promotion.
- Áp dụng được cho cả dense và hybrid.

## 7. Kiến trúc

1. **Collection** `<collection hiện tại>__hybrid-v1`:
   - Vector dense không tên, lấy nguyên từ cache embedding và trùng từng bit với collection cũ.
   - Sparse có tên `bm25_ub`, modifier IDF.
   - Metadata: schema version, model key, sha256 của `chunks.jsonl`, số điểm, `TOKENIZER_VERSION`, k1, b, avgdl. Khi truy vấn, lệch `TOKENIZER_VERSION` thì báo lỗi kèm lệnh index lại.
   - Collection cũ không đổi và vẫn phục vụ `search`/`ask` cho tới T7.
2. **Điểm:** `RetrievedChunk.score` vẫn là cosine dense của chunk. Ở chế độ sparse và hybrid, cosine lấy từ một truy vấn dense toàn độ sâu (số điểm trong metadata collection), nên chunk nào cũng có cosine. Thêm các trường tùy chọn `fusion_score`, `dense_rank`, `sparse_rank`. Mục ngoài phạm vi trong báo cáo dùng cosine dense top-1, nên vẫn so được với baseline.
3. **Trường hợp rỗng:**
   - Không gửi sparse vector rỗng tới Qdrant.
   - Hybrid: truy vấn không còn term sparse, hoặc nhánh sparse không có kết quả, thì trả xếp hạng dense.
   - Sparse: chỉ trả các hit sparse, mỗi hit kèm cosine dense. Không có hit thì trả danh sách rỗng, không lấy dense bù vào, để dòng "sparse" không lẫn kết quả dense.
   - Ở mọi chế độ, chỉ khi truy vấn dense rỗng mới báo collection rỗng.
4. **Parser số Điều** (`app/rag/article_refs.py`):
   - Dựng trên `article_mentions` của `app/rag/generate.py`; không đổi hành vi của `generate.py`.
   - Chỉ nhận Điều từ 1 đến `LABOUR_CODE.expected_articles`.
   - Số phải kết thúc ở ranh giới: ký tự ngay sau không phải chữ cái. "Điều 5a" không kích hoạt, "Điều 35?" kích hoạt. `ARTICLE_MENTION` không có ranh giới này (nó trả {5} cho "Điều 5a"), nên parser tự kiểm.
   - Chỉ kích hoạt khi Điều thuộc Bộ luật này: không có tên văn bản khác theo sau, hoặc có "Bộ luật này" / "Bộ luật Lao động" mà không kèm "số 10/2012/QH13". "quy định", "nói về", "nội dung" không phải tên văn bản.
   - Chữ "điều" viết thường vẫn được nhận. Quy tắc loại "Tôi muốn hỏi điều 1 chút" (nghĩa "một chút", không phải Điều 1) do bước thiết kế của T4 chốt trước khi viết code, ví dụ một danh sách đóng các từ chỉ lượng đứng sau số.
   - Fixture phải kích hoạt, kèm tập Điều mong đợi (đoạn trích nguyên văn từ `chunks.jsonl`):
     - `d39` "các điều 35, 36 và 37 của Bộ luật này" → {35, 36, 37};
     - `d63:k2` "các điều 42, 44, 93, 104, 118" → {42, 44, 93, 104, 118};
     - `d34:k11` "Điều 42 và Điều 43 của Bộ luật này" → {42, 43};
     - `d37:k1` "khoản 1 Điều 36 của Bộ luật này" → {36};
     - câu hỏi "Điều 24 quy định gì" → {24} (Điều 24 có trong `chunks.jsonl`).
   - Fixture không được kích hoạt:
     - `d219:k2` "Điều 32 của Bộ luật Tố tụng dân sự";
     - `d34:k4` "Điều 328 của Bộ luật Tố tụng hình sự";
     - `d219:k1` "Điều 54 của Luật này";
     - `d220:k1` "Bộ luật Lao động số 10/2012/QH13";
     - các chuỗi "Điều kiện", "Tôi muốn hỏi điều 1 chút", "Điều 5a".
   - Nghiệm thu:
     - Với mỗi câu article_ref, tập Điều parser trả về bằng đúng `gold_articles` của câu. 115 câu còn lại cho tập rỗng.
     - Test đơn vị: tập rỗng thì promotion trả lại nguyên danh sách gốc; trần m và phần lấp đúng với k = 1, 2, 3, 5, 8, 10; client giả xác nhận truy vấn `MatchAny` chỉ chạy ở chế độ dense.
     - Không chạy promotion trên câu của bộ đánh giá trước T6 (mục 5.8).
5. **Metadata của run:** file `<run>.meta.json` đặt cạnh file run, chỉ gồm các khóa trong danh sách cho phép (digest bộ câu, tên và metadata collection, chế độ, k, rrf_k, promotion, device, git SHA, ngày); không dump Settings.

## 8. Task

Mọi task chạy `pytest -q`, `ruff check app ingestion eval tests` và `ruff format --check app ingestion eval tests`, rồi đối chiếu `git status --short` với danh sách file được sửa. Tier theo `AGENTS.md`; task Trung bình và Khó có reviewer Opus. CLI mới hoặc CLI được sửa gọi `sys.stdout.reconfigure(encoding="utf-8")` và làm tương tự với stderr, như `eval/run_eval.py:335-336`. Mỗi CLI đó có một test chạy với `PYTHONIOENCODING=cp1252`.

| # | Tier | Việc | File được sửa | Nghiệm thu |
|---|---|---|---|---|
| S0 | Trung bình | recall@1 và R@1 theo nhóm trong `results_row`, bổ sung cho dòng baseline; sau khi duyệt D1 thì sửa Eval gate của `AGENTS.md`; commit theo D7 | `eval/run_eval.py` (`results_row`), `tests/test_run_eval.py`, `eval/results.md`; sau D1: `AGENTS.md`; sau D7: các file ở D7 | test dựng lại dòng baseline từ file run khớp `results.md` |
| T1 | Trung bình | tokenizer và sparse vector (6.1, 6.2) | `app/rag/sparse.py`, `tests/test_sparse.py` | các ca ở 6.1; ca số "0", "0,7", "01", "10/2012/QH13", "61%", "1a", "8a"; NFD = NFC; Ð U+00D0; câu rỗng sau khi làm sạch cho vector rỗng; 0 va chạm id trên corpus |
| T5a | Trung bình | `eval/stats.py`: b/c, McNemar chính xác và mid-p, Clopper–Pearson | `eval/stats.py`, `tests/test_stats.py` | vector kiểm thử ở [S] §10 và run baseline |
| T2 | Trung bình | collection hybrid và metadata (7.1), lệnh index mới, tên collection hybrid trong Settings, gợi ý lỗi khi collection thiếu sparse | `ingestion/index.py` (hàm mới, chữ ký cũ giữ nguyên), `app/core/config.py`, `app/rag/store.py`, `tests/test_index.py`, `tests/test_config.py`, `tests/test_store.py`, `tests/test_integration_hybrid_index.py` (mới); `.env.example` và `tests/conftest.py` chỉ khi thêm biến môi trường | test đơn vị với client giả; `uv run pytest -m integration tests/test_integration_hybrid_index.py` qua: 676/676 cache hit, dense trùng từng bit giữa hai collection, cùng vector truy vấn (câu ngoài bộ) cho cùng id và điểm trên hai collection, dense trên collection mới đạt dung sai tái lập (5.7), ghi device |
| T3 | Khó | retriever dense/sparse/hybrid, RRF, điểm, trường hợp rỗng; giữ xếp hạng gốc đầy đủ trước khi cắt k cho promotion | `app/rag/retriever.py`, `app/rag/fusion.py`, `tests/test_retriever.py`, `tests/test_fusion.py`, `eval/results.md` | RRF tính tay; client giả xác nhận không dùng `Fusion.RRF` và không gửi sparse rỗng; các ca 7.3; test cũ vẫn qua; chạy lại test integration của T2; dòng eval gate theo 5.8 |
| T4 | Khó | parser số Điều và promotion (6.4, 7.4) | `app/rag/article_refs.py`, `app/rag/retriever.py`, `tests/test_article_refs.py`, `tests/test_retriever.py`, `eval/results.md` | mục 7.4; dòng eval gate theo 5.8 |
| T5b | Trung bình | `run_eval`: `--retriever`, `--promote`, chọn collection, `--reference <run>`, file meta, nhãn mặc định theo chế độ, ngoài phạm vi theo cosine dense, mục câu trượt theo R@1; mục so với run tham chiếu in R@1 theo nhóm, b, c, p chính xác, mid-p, số câu direct mất và cận Clopper–Pearson (gọi `eval/stats.py`), danh sách câu tụt hạng kèm nguyên nhân (5.9; nguyên nhân parser gọi `article_refs` trên câu hỏi); file run lưu thêm `fusion_score`, `dense_rank`, `sparse_rank` khi có | `eval/run_eval.py`, `tests/test_run_eval.py` | fake retriever cũ (`lambda: fake`) vẫn chạy; test mục so sánh với hai run giả cho b, c, p khớp `eval/stats.py`; test bộ khóa của file run được cập nhật; test console cp1252 |
| T6 | Trung bình | đo theo 5.7 (`--reference` là run (1) cho run (2)–(4), run (3) cho run (5)), ghi kết quả, quyết định theo 5.2–5.6 | `eval/runs/*`, `eval/results.md`, `README.md` | run (1) đạt dung sai tái lập; đủ 5 dòng; b, c, p và cận Clopper–Pearson lấy từ output của `run_eval`; danh sách câu tụt hạng; quyết định ghi rõ |
| T7 | Trung bình | nối cấu hình đã chốt vào `search`/`ask`: setting chế độ truy xuất và promotion, `default_retriever` đọc setting, `run_eval` không cờ đo cấu hình đang ship; chốt hybrid thì đọc collection hybrid, chỉ chốt promotion thì đọc collection cũ (dựa trên kết quả trùng từng bit và cùng kết quả truy vấn của T2); chỉ làm khi T6 chốt hybrid hoặc promotion | `app/core/config.py`, `app/rag/retriever.py`, `eval/run_eval.py`, `.env.example`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_retriever.py`, `tests/test_run_eval.py`, `eval/results.md`, `README.md` | client giả xác nhận `retrieve` dùng đúng chế độ, collection và promotion theo setting; giá trị mặc định bằng quyết định ở T6; dòng eval gate của `run_eval` không cờ trùng số dòng T6 của cấu hình đã chốt, sai khác chỉ được do đổi chỗ trong dung sai 5.7 và phải báo |
| X1 | thăm dò | mục 5.12 | `eval/runs/*`, `eval/results.md` | chỉ ghi số |

Thứ tự:

1. S0, T1, T5a chạy song song, đã giao ngày 26/11/2026.
2. Người dùng duyệt spec, rồi commit spec bản đã duyệt.
3. T2 → T3 → T4 → T5b → T6 → T7 (nếu có) → X1.

Từ T2 trở đi chạy tuần tự trong checkout chính, vì cần cache embedding, model và Qdrant.

## 9. Mục Eval gate trong `AGENTS.md` (D1, áp dụng trong PR S0)

Ba ý dưới đây thay toàn bộ danh sách cũ của mục Eval gate. Ý cũ về giai đoạn trước Tuần 3 (`run_eval.py` còn là placeholder) đã lỗi thời nên bỏ.

> - Chạy `eval/run_eval.py` trước và sau thay đổi. Ghi một dòng vào `eval/results.md` (recall@1, recall@5, recall@10, MRR ở mức Điều; mức Khoản và R@1 theo nhóm ở cột Ghi chú), kể cả khi thử nghiệm không hiệu quả.
> - recall@5 mức Điều đã chạm trần 1,000 trên bộ câu đóng băng, nên chỉ số chính là R@1 theo nhóm; MRR@10 là chỉ số phụ. Mọi câu tụt hạng phải được liệt kê kèm nguyên nhân.
> - Thay đổi đã có pre-registration trong spec của tuần thì chốt theo luật và chỉ số quyết định trong spec đó; số liệu ngoài pre-registration ghi là thăm dò.

## 10. Quyết định (người dùng duyệt theo đề xuất ngày 26/11/2026)

- **D1** luật chốt và chỉ số quyết định: như mục 5.2–5.6.
- **D2** holdout: không. Phương án khác: viết 15–20 cặp câu mới trước lần chạy đầu.
- **D3** promotion số Điều: có, theo luật 5.5; dùng cho `search` và `ask` (cùng đi qua `retrieve`) kể cả khi không chốt hybrid.
- **D4** collection mới, collection cũ giữ nguyên (mặc định).
- **D5** RRF trên toàn bộ xếp hạng: có.
- **D6** một nhánh BM25 (âm tiết + bigram): có.
- **D7** commit trong PR S0: `docs/research/2026-11-25-week4-*.md` (4 file); `eval/plot_baseline.py` cùng 2 ảnh trong `docs/images/` (script chạy bằng `uv run --with matplotlib`, matplotlib không thành dependency của dự án); spec này ở bản đã duyệt. `.claude/skills/labour-code-lookup/` không có đề xuất nên chưa commit; người dùng quyết sau.
- **D8** thống kê gọn (b/c, mid-p và p chính xác, Clopper–Pearson) (mặc định).
- **D9** gộp dấu và bỏ số 0 đầu ngay từ Tuần 4: có.
- **D10** báo cả kết quả "không hơn"; viết lại dòng CV ở Tuần 8 (mặc định).

## 11. Phần để người xây dựng tự quyết

- Cách tổ chức code bên trong module, tên hàm private, cấu trúc test.
- Câu chữ thông báo lỗi tiếng Việt, miễn là nói rõ lệnh cần chạy.
- Tên cờ và tên setting cụ thể ở T2, T5b, T7, miễn là đúng hành vi trong bảng mục 8.

## 12. Nguồn

- Ghi chú nghiên cứu: `docs/research/2026-11-25-week4-hybrid-retrieval.md` [R], `docs/research/2026-11-25-week4-gaps-qdrant.md` [Q], `docs/research/2026-11-25-week4-gaps-vietnamese.md` [V], `docs/research/2026-11-25-week4-gaps-stats.md` [S].
- Cormack, Clarke, Büttcher, SIGIR 2009: RRF, k = 60.
- Bruch, Gai, Ingber, ACM TOIS 42(1):1–35 (bản online 2023); đã đọc bản arXiv 2210.11934v2.
- Fagerland, Lydersen, Laake, BMC Medical Research Methodology 2013: McNemar mid-p.
- Fuhr, SIGIR Forum 51(3), 2017 và Sakai, SIGIR Forum 54(1), 2020: báo MRR, một giả thuyết khai báo trước, effect size.
- Nguyen & Quan, Findings of EACL 2026: so sánh phương pháp truy xuất cho tiếng Việt.
- COLIEE 2025 overview, Review of Socionetwork Strategies (2026), bản CC BY trên PubMed Central.
