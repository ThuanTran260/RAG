# Week 4 gap research: Vietnamese tokenisation, tone marks, explicit article references

Date: 2026-11-25. Follow-up to `docs/research/2026-11-25-week4-hybrid-retrieval.md` (sections 4 and 8).

Conventions in this file:

- "Read" means I opened the primary text (PDF, source file, README, spreadsheet) in this session. "Search summary only" means I did not open it and do not rely on it.
- `[inference]` marks my own reasoning. `[measured]` marks something I computed myself on `data/processed/chunks.jsonl` or by running a fetched function (I read the code first; I ran only pure string functions, never unpickled anything).
- GitHub sources are cited at the commit that was `main` on 2026-11-25. Quoted code is verbatim.
- Tone-mark naming: sources disagree on "old/new". To avoid the trap I use neutral labels throughout:
  - **Style A** = tone mark on the first letter of oa/oe/uy: hòa, thủy, khỏe, tòa, ủy. Vietnamese Wikipedia and the widely copied normaliser below call this "kiểu cũ" (old).
  - **Style B** = tone mark on the second letter: hoà, thuỷ, khoẻ, toà, uỷ. Called "kiểu mới" (new) by the same sources.
  - The earlier notes and the task brief call Style A "new". That naming is the opposite of the sources I read. Use A/B in code and tests.

---

## Summary: findings that change the plan

1. **The "85.8 R@5 / 70.3 MRR" repository is confirmed**, but it is `monster1909/vietnamese-legal-retrieval` (UIT Data Science Challenge 2026), not the first repo I found. The numbers are the **BM25-only row** on 1,000 validation questions over an 8.5k-document corpus. BM25 there uses accent stripping + syllable unigrams + syllable bigrams, `k1=0.9, b=0.75`. The README has no tokenizer ablation. It is document-level retrieval, so it is not comparable to our 676 clause chunks. (Section 3b.)
2. **The only tokenizer ablation with a paired test that I found is in another UIT 2026 repo** (`trnngcccuyn/vietnamese-legal-retrieval-DSC2026`): `\w+` syllable unigrams R@5 0.8116 vs syllable unigram+bigram 0.8555 on 1,000 dev questions (+0.0429 after removing a confound). The pyvi/underthesea arms were run, but their numbers are not in the repo. Unreviewed experiment log, different task. (Section 4.)
3. **The EACL 2026 paper is real and the citation is exact**, but two earlier claims need correcting: (a) the no-diacritics failure example is in an education dataset (EduCoQA), not a legal one; (b) alpha = 0.7 weights the dense score and was picked from a sweep on UIT-ViQuAD (open-domain Wikipedia QA), not on legal data. Every benchmark corpus is subsampled to 1,000 pairs. With bge-m3 on ZaloLegalQA, hybrid is no better than dense in the table (P@1 82.30 -> 82.10). (Section 3a.)
4. **Quyết định 07/2003/QĐ-BGDĐT is not about tone marks.** It is the interim rule on capitalising proper names in textbooks. The tone rule is Điều 8 of Quyết định 1989/QĐ-BGDĐT (25/05/2018), whose examples (hoà nhạc, thuỷ thủ, mạnh khoẻ) are **Style B**. Quyết định 240/QĐ (1984) full text was not retrievable. (Section 5.)
5. **Our corpus is Style A, the opposite of the ministry textbook rule** `[measured]`: open-syllable oa/oe/uy tokens (`qu-` excluded) are 500 Style A tokens of 13 types in the clause `text` (681 in `context_text`, which repeats headings: thỏa 386, hòa 126, ủy 58, tòa 31, khỏe 26) vs 1 Style B (`tuý`; `túy` never occurs). *Corrected 2026-11-25 by the coordinator: the first count (1,358 vs 2) summed every string field of each record (`text` 500 + `context_text` 681 + chapter/section/article titles 177), so clause tokens were counted twice and headings several times.* Users following the textbook rule (or typing "thoả thuận", "hoà giải") will miss "thỏa", "hòa" in BM25 unless tokens are folded. Dense bge-m3 is robust to the style (coordinator check on four non-eval sentences: cosine 0.970-0.995 between Style A and B) but not to missing diacritics (cosine 0.36-0.42).
6. **A naive fold regex corrupts closed syllables.** The fold in the DSC 2026 repo rewrites `khoản` -> `khỏan`, `hoàn` -> `hòan`, `toàn`, `ngoài`, `loại`, `hoạt` `[measured, I ran it]`. In `chunks.jsonl` that touches about 1,060 tokens (khoản 336, ngoài 344, hoạt 150, toàn 124, đoàn 42, loại 40, hoàn 22, khoảng 4). It is retrieval-neutral if applied on both sides, but it breaks any later regex that looks for "khoản". Use an end-anchored fold (open syllables only) inside the BM25 tokenizer only. I give a tested rule in Section 5.
7. **Direction of normalisation differs between tools, and BARTpho/PhoBERT go the other way from underthesea.** underthesea (`token_rules.xlsx`, 968 token rules, tokens of at most 6 chars): Style B -> A. BARTpho/PhoBERT pre-training script: Style A -> B (`"òa": "oà"`, `"úy": "uý"`). For BM25 the direction does not matter; consistency does. Choose A because the corpus is A and the text stays verbatim.
8. **Copying the Zalo 2021 winner's tokenizer would delete reference evidence.** `utils.py` in `CuongNN218/zalo_ltr_2021` removes the strings "1".."10", "khoản", "điểm", "số", "ngày", "năm" as stopwords. Do not use that list for article-reference questions.
9. **The best-known open reference regex does not handle the most natural question form.** In `hoadm-net/SA-RAG`, `find_references("Điều 24 quy định gì")` returns `[]` because the "other document" test treats "quy định" as a document type (it is in `_DOC_TYPES`). I ran it. Also `điều 1 chút` (lowercase, unrelated meaning) returns a reference. Both are test cases for our parser. (Section 6.)
10. **No literature measures query-side "Điều N" filtering.** COLIEE overviews I read say nothing about article numbers in queries; BSARD (labels parsed by regex from expert answers, questions anonymised) and IL-PCSR (masks explicit section numbers in queries) deliberately remove or avoid them; Fink et al. (COLIEE 2022 case retrieval) add a BM25 statute-section field and gain about +0.006 F1 on dev. So the article filter is an engineering choice to be justified by our own article_ref stratum, with the parser tested for false activations.
11. **Diacritic-less queries are a large, documented failure.** One README (`Messiac24/...khong-dau`, Zalo 2021 corpus, 788 test questions) reports hybrid Recall@10 0.9772 with diacritics, 0.1447 with diacritics stripped and left as is, 0.9670 after a bigram diacritic-restoration step (single run, README only, test queries synthetically stripped). `monster1909` sidesteps the problem by stripping accents inside BM25. Out of scope for the frozen set, but a cheap exploratory BM25 arm.
12. **Zalo 2021 official page could not be read** (JavaScript app; archive access blocked). Dataset size and metric come from the Miko team paper. See NEED DOWNLOAD.

---

## 1. Zalo AI Challenge 2021 - Legal Text Retrieval

### What I could and could not read

- **Official task page** `https://challenge.zalo.ai/portal/legal-text-retrieval`: the fetch returned only a title ("Zalo AI Challenge 2026"); the page is a JavaScript app. The Wayback Machine fetch was refused by the tool. Not read. See NEED DOWNLOAD.
- **Dataset facts via the Miko team paper** (read): "additional samples from last year's Zalo legal text retrieval competition (3196 questions and 61425 articles)", and "many data articles are over 2000 characters in length and concentrated in Zalo's dataset". arXiv 2211.02200, section V-B. Official metric (that paper states it for ALQAC's private test): macro F2.
- A 2025 NAACL paper lists "ZaloLegal2021" as a benchmark (2,556 train / 640 evaluation samples), from a search summary only; not opened.

### Systems (BM25 tokenisation, fusion, metrics)

| System | BM25 tokenisation | Combination with dense | Reported metrics |
|---|---|---|---|
| `CuongNN218/zalo_ltr_2021` (source code for a Zalo AI 2021 submission; commit `b63aa80`) | `utils.py`: `underthesea.word_tokenize` -> lowercase -> drop ASCII punctuation -> drop stopword list. `rank_bm25.BM25Plus(k1=0.4, b=0.6)` in `bm25_train.py`. | BM25 produces negatives for training; at inference "multiply cos-sim score with score from bm25", then keep articles with score in `[max-score - 2.6, max-score]`, "top 5 sentences for a question with multiple answers"; 4-model ensemble weights 0.1/0.3/0.4/0.2 | README gives no scores and no rank. That this is the 1st-place entry comes from the author's personal pages (search summary only; not opened). |
| `hieudx149/ZaloAI2021_LTR` (commit `093251d`) | Elasticsearch BM25; analyzer not stated in README | ES top-50 -> PhoBERT cross-encoder | "0.747 F2 score in public test"; "If using elasticsearch only, our F2 score is 0.54" |
| `Messiac24/vietnamese-legal-retrieval-khong-dau` (commit `cd9f612`, README read via fetch) | word segmentation with pyvi mentioned as the main data-prep cost | weighted fusion of BM25 and `AITeamVN/Vietnamese_Embedding` | Recall@10 on 788 test questions: BM25 0.8522, dense 0.9721, hybrid 0.9772 (diacritics present). Lower bound: 24 questions have conflicting labels. |
| Pham, Nguyen, Do, "Multi-stage Information Retrieval for Vietnamese Legal Texts" (PKAW 2022; arXiv 2209.14494) | `rank_bm25` BM25+ with VnCoreNLP segmentation + stopword removal | product scores `sqrt(BM25)*cos_sim` and `BM25*cos_sim` (they reject `(1-a)*lexical + a*semantic` as "will not work or take much time to find" a) | Their corpus is the Kien et al. COLING 2020 data (8,436 documents, 114,177 articles), **not** Zalo 2021. BM25+ Recall@20 0.557, F2 0.221. SPhoBERT-large(word): Recall@20 0.954 / F2 0.727 alone; 0.967 / 0.741 with `sqrt(BM25)*cos_sim`; 0.970 / 0.721 with `BM25*cos_sim`. |

Sources: https://github.com/CuongNN218/zalo_ltr_2021 (files `README.md`, `utils.py`, `bm25_train.py`), https://github.com/hieudx149/ZaloAI2021_LTR, https://github.com/Messiac24/vietnamese-legal-retrieval-khong-dau, https://arxiv.org/pdf/2209.14494 (Tables 3-4 and section 3).

Stopword list in `utils.py` (verbatim): `number = ["1", "2", ..., "10"]`, single letters `a..o`, then `"của", "và", "các", ... "khoản", ... "ngày", "năm", "số", ... "điểm", ...`.

### Implications for the Week 4 plan

- **Do not import this stopword list.** It removes the digit tokens 1-10, "khoản", "điểm", "số". For "khoản 2 Điều 35" the clause number would vanish from BM25. Use no stopwords in the primary arm (IDF handles frequent syllables), or a list that excludes digits and reference words. Test: tokenising `"khoản 2 Điều 35"` must keep `khoản`, `2`, `điều`, `35`.
- **Fusion by rank (RRF) is not what Zalo winners used.** The two systems that document it use score products or weighted sums on raw scores (cos-sim x BM25; ensemble weights). Our RRF k=60 plan is defensible but is not backed by Zalo evidence. `[inference]`
- **Cascade vs fusion.** Several systems (Miko, hieudx149) use BM25 only as a candidate generator for a cross-encoder. Not applicable to a 676-chunk corpus where dense scoring of everything is free.
- BM25 hyper-parameters in the Zalo winner are `k1=0.4, b=0.6` (low k1), in EACL 2026 `1.5/0.75`, in monster1909 `0.9/0.75`, in the plan `1.2/0.75`. Nothing here justifies tuning on the frozen set; keep the a-priori `1.2/0.75`.

---

## 2. ALQAC 2021-2024

### Overview papers

| Year | Paper | Access |
|---|---|---|
| 2021 | Nguyen Ha Thanh et al., "A Summary of the ALQAC 2021 Competition", KSE 2021, arXiv 2204.10717v2 (read, tables checked). https://arxiv.org/pdf/2204.10717 | open (arXiv) |
| 2022 | "ALQAC 2022: A Summary of the Competition", KSE 2022, pp. 1-5, DOI 10.1109/KSE56063.2022.9953764 | IEEE; not opened (NEED DOWNLOAD) |
| 2023 | "A Summary of the ALQAC 2023 Competition" (listed on the ISAILD 2023 page); DOI not found | not found (NEED DOWNLOAD) |
| 2024 | no summary paper found; competition site https://sites.google.com/view/ALQAC-2024 (4th ALQAC, KSE 2024) | n/a |
| 2025, 2026 | exist per https://alqac.github.io/ (2026 task is case outcome prediction) | not relevant to Task 1 |

ALQAC 2021 facts I read: Task 1 metric is macro-F2 (computed per query, averaged); baseline TF-IDF 0.6392. Best runs: Aleph 0.8807 (fine-tuned VNLawBERT binary ranker with hard negatives), AimeLaw `run2 bm25 bert cnn 09 threshold` 0.8061 and `run1 bm25` 0.7842 (BM25 alone, 0.14 above the TF-IDF baseline and 0.10 below the best run `[my arithmetic]`), Kodiak 0.7955, Dline 0.7936. AimeLaw: "combining scores of BM25 with Domain Invariant Supporting Model and Deep CNN Supporting Model using a weighted sum function". The summary does not describe tokenisation.

### Top systems

| Team / paper | BM25 tokenisation | Combination | Result |
|---|---|---|---|
| Miko Team, ALQAC 2022, arXiv 2211.02200 (open). https://arxiv.org/pdf/2211.02200 | "Word segmentation using pyvi", lowercase, remove punctuation and stopwords | cascade: BM25 top-150 -> fine-tuned XLM-R text-pair classifier | 1st in Task 1. Table III test F2 0.9667 (k1=150, k2=20 run) and 0.8667 (k=150 both). Preprocessing also "Standardize the typing of the accent (eg. oà to òa)" (I read the exact characters with pypdf: oà -> òa, i.e. Style B -> A). |
| NeCo, ALQAC 2023, arXiv 2309.05500 (open). https://arxiv.org/pdf/2309.05500 | not stated; the heading ("Điều 1 Luật Thanh niên") is prepended to article text before BM25 | min-max normalise BM25 and mBERT scores, `score = a*w_bm25 + (1-a)*w_bert`, a and threshold by grid search on validation; BM25 top-100 (training recall 100 at top-100, 94 at top-1) | Private test F2-macro 0.8661 (their run 2), stated as the highest among teams |
| NOWJ1, ALQAC 2023, arXiv 2309.09070 (open). https://arxiv.org/pdf/2309.09070 | RDRSegmenter (VnCoreNLP) "to form compound words" before TF-IDF/BM25 | LightGBM learning-to-rank over BM25, TF-IDF, BERT classifier features | Public F2 0.94 (top-1 only); private F2 0.8298 (run 3), second place |
| UIT-DarkCow, ALQAC 2024, IJALP 35(1), 2450010, DOI 10.1142/S2717554524500103 | "Vietnamese sentence parsing" with BM25 and sentence transformers (abstract only) | not read | third in retrieval, second in QA (abstract) |

### Implications for the Week 4 plan

- Every ALQAC team I could read uses word segmentation (pyvi or RDRSegmenter) for BM25 and **none ablates it**. Do not cite ALQAC as evidence for or against syllable-level BM25.
- Two of three documented fusion schemes are linear score combinations with a weight tuned on validation (NeCo, AimeLaw). Our "equal-weight RRF, no tuning on the frozen set" rule is stricter than the competition practice; keep a-priori CC alpha 0.7-0.8 as the pre-declared exploratory arm (EACL 2026 uses 0.7 weighting dense).
- NeCo prepends the article heading (with the number) to the BM25 text. Our chunks already carry "Điều N." in `context_text` per the earlier notes; keep the same field for BM25 and dense (already an acceptance check in the earlier notes).
- Miko normalises tone placement toward Style A; this is evidence that practitioners on legal data hit the same problem, but they report no effect size.

---

## 3. Verification of two claims from the earlier notes

### 3a. EACL 2026 paper

**Exact citation.** Long S. T. Nguyen and Tho T. Quan. "Which Works Best for Vietnamese? A Practical Study of Information Retrieval Methods across Domains." In *Findings of the Association for Computational Linguistics: EACL 2026*, pages 2098-2119, March 2026. DOI 10.18653/v1/2026.findings-eacl.110. https://aclanthology.org/2026.findings-eacl.110/ (PDF: https://aclanthology.org/2026.findings-eacl.110.pdf). Code and data: https://github.com/longstnguyen/ViRE (commit `60ea715`).

**Table 5 (Appendix A), verbatim:** BM25 k1 = 1.5; BM25 b = 0.75; weighted fusion coefficient (alpha) = 0.7; RRF constant (c) = 60. Confirmed.

**Table 3 (page 2104), cells I read from the rendered page** (P@1 / R@10 / MRR@10 / nDCG@10 / R@20):

| Row | ALQAC | ZaloLegalQA |
|---|---|---|
| BM25 | 89.25 / 97.92 / 92.20 / 93.59 / 99.25 | 71.40 / 92.18 / 79.39 / 82.49 / 94.42 |
| TF-IDF | 82.83 / 96.23 / 88.34 / 90.31 / 98.11 | 64.70 / 92.47 / 75.07 / 79.28 / 96.45 |
| bge-m3 Dense | 90.38 / 99.43 / 94.14 / 95.47 / 100.00 | 82.30 / 97.83 / 88.52 / 90.75 / 98.77 |
| bge-m3 + BM25 (alpha) | 94.72 / 99.62 / 96.66 / 97.40 / 99.81 | 82.10 / 98.02 / 88.62 / 90.93 / 98.62 |
| AITeamVN/Vietnamese_Embedding_v2 Dense | 90.38 / 99.06 / 93.96 / 95.24 / 99.81 | 86.20 / 98.32 / 91.04 / 92.74 / 98.97 |
| Vietnamese_Embedding_v2 + BM25 (alpha) | 93.77 / 99.25 / 95.95 / 96.78 / 100.00 | 87.40 / 98.17 / 91.67 / 93.18 / 98.67 |

For `Vietnamese_Embedding_v2 + BM25 (alpha)` the R@20 for ALQAC (100.00) comes from the text extraction line that PDF text scrambling split off; the bge-m3 rows I confirmed visually.

**What the paper says that matters here (quotes verbatim):**

- "Hybrid effectiveness critically depends on calibrated lexical weighting, as BM25-based linear interpolation consistently surpasses TF-IDF and rank-based fusion strategies; optimal alpha values typically lie between 0.6 and 0.8, indicating a dominant yet complementary contribution from dense similarity." (I3, page 2104). So **alpha weights the dense score** `[inference from the sentence and Fig. 4]`; the formula itself is not printed in the sections I read.
- Appendix F.2: the sweep was "on the full corpus of UIT-ViQuAD" with multilingual-e5-large + BM25, alpha 0 to 1 in steps of 0.05; "Performance peaks within alpha in [0.6, 0.8]"; "We therefore adopt alpha = 0.7 for all hybrid experiments". **Alpha was not tuned on legal data.**
- Section 4.2: "each dataset is standardized to 1,000 query-document pairs"; duplicate contexts removed, one query per unique context. So the legal numbers above come from 1,000-document corpora, not 61k articles.
- I9: "structured legal and encyclopedic corpora approaching saturation under strong hybrid configurations".
- BM25 tokenisation: "using standard inverted-index configurations" (section 3, retrieval paradigms). The Underthesea segmenter is stated only for dataset token statistics (Appendix C). ViRE `lexical.py` line 29 `TfidfVectorizer(analyzer=vi_segment)`, lines 63-64 `tokenized_docs = [vi_segment(d) for d in docs]; return BM25Okapi(tokenized_docs, k1=k1, b=b)`, with `vi_segment` = underthesea `word_tokenize(..., format="text").split()` if installed, else regex `[0-9A-Za-zÀ-ỹ_]+`, no lowercasing or normalisation in that module. So the benchmark's BM25 is **word-segmented (underthesea) with `rank_bm25` BM25Okapi**, falling back silently to syllable regex when underthesea is missing (`HAVE_UTS`). `[inference]` The paper does not say which path produced its numbers.

**Failure cases for queries without diacritics: partly confirmed.** Section 4.5, E1: "Spelling variation, typos, and missing diacritics further degrade retrieval quality, with inconsistent casing having similar effects." Appendix G.1 Table 10 gives one example, in **EduCoQA (education)**: query `khoa ktxd có những hb nào vậy?` ("informal orthography without diacritics"), with the note "Orthographic and register gaps reduce lexical overlap and weaken sparse-dense fusion." No legal-domain no-diacritics result and no quantified effect in the paper.

**Verdict:** the three numbers (k1 = 1.5, alpha = 0.7, no-diacritics failure case) exist, but "alpha around 0.7" is a dense-weighted open-domain choice, and the no-diacritics evidence is anecdotal and non-legal.

### 3b. The 2026 repository with syllable bigrams, 85.8 R@5 / 70.3 MRR

**Confirmed.** Repository: https://github.com/monster1909/vietnamese-legal-retrieval ("UIT 2026 Data Science Competition", created 2026-08-06, last push 2026-10-20, commit `6fa9647`). README, "Ablation" table, "Each stage on the 1,000 validation questions":

| Setup | Recall@5 | MRR |
|---|---|---|
| BM25 | 85.8 | 70.3 |
| Bi-encoder, round 1 (negatives from BM25) | 92.4 | 82.5 |
| Bi-encoder, round 2 | 93.4 | 84.8 |
| RRF fusion of the three | 94.7 | 85.6 |
| Fusion + reranker | 96.4 | 87.6 |

Also: "Fusion parameters were tuned on 700 of these questions. On the other 300, the full pipeline gets 95.5". Leaderboards: public 95.5 (rank 5/100), private 95.0 (rank 9/100).

Tokeniser, from `src/retrieval/bm25.py` (commit `6fa9647`) and `src/config.py`:

- `BM25Config`: `k1: float = 0.9`, `b: float = 0.75`, `min_df: int = 2`, `use_bigrams: bool = True  # syllable bigrams approximate Vietnamese word segmentation`, `add_docnum_token: bool = True  # lets queries citing a decree number match its body chunks`, `strip_accents: bool = True`.
- `_strip_accents`: NFD, drop all `Mn` marks, then `đ -> d`. This removes tone placement as a problem altogether, at the price of merging syllables that differ only by diacritics (e.g. "ma/má/mã/mạ").
- `tokenize`: `WORD_RE = [0-9a-zà-ỹđ]+`; stopwords `{"và","của","có","được","cho","các","trong","là","với","này","khi","theo","về","đã","những","một","tại","từ","để","đến","hoặc","như","thì","sẽ","bị","phải","còn","cũng","nếu","mà"}` are dropped from unigrams (bigrams are built from the unfiltered word list); document numbers matched by `\d{1,4}/(?:\d{4}/)?[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\-]{1,25}` are kept as one token (`§75/2017/ND-CP`).
- IDF: `np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))`, identical to the Qdrant `fancy_idf`.

**Caveats.** The task is document-level retrieval over 8.5k legal documents (chunks of about 1,000 tokens), top-5 recall; our task is clause-chunk, article-level. The README has **no unigram vs bigram vs word-segmented ablation**, so it supports "bigram BM25 is a workable baseline", not "bigrams beat alternatives". The first repo I found (`trnngcccuyn/...DSC2026`) reports a different BM25 number (dev R@5 0.8547, no MRR) and does not contain 85.8 or 70.3 anywhere (I searched README and `experiments.csv`).

### Implications for the Week 4 plan

- Cite `monster1909` as practitioner evidence for the arm "syllable unigram + bigram, no segmenter dependency", not as proof of superiority.
- Report BM25-only R@5 and MRR@10 in the same format so the numbers are comparable in kind (not in value).
- Keep alpha 0.7 / k1 1.5 only as documented literature values; the plan's a-priori `k1=1.2, b=0.75` stands. If a CC arm is run, pre-declare alpha 0.7 on the dense score and min-max normalisation, flag as exploratory (the paper does not print the normalisation rule in the sections I read).
- Expect hybrid gains on a clean legal corpus to be small and corpus-dependent: bge-m3 on ZaloLegalQA shows no gain in the table above.

---

## 4. Controlled comparisons of Vietnamese tokenisation for lexical retrieval

### 4.1 EVIA 2007 (exact citation and what it measured)

Ho Bao Quoc. "Vietnamese Text Retrieval: Test Collection and First Experimentations." *The First International Workshop on Evaluating Information Access (EVIA)*, May 15, 2007, Tokyo (NTCIR-6 online proceedings, paper 17). https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings6/EVIA/17.pdf (read).

- Setup: 14 topics (the paper says they plan 25), TREC-format topics; collection of two parts, VN1 (11,398 documents, about 30 MB) and VN2 (25,215 documents, about 69 MB, mined from a VOA English-Vietnamese site). Stopwords removed in all four methods. Weighting "ltc" in SMART (not BM25). The paper does not say which collection produced the numbers. It does not give statistical tests.
- 11-point average precision, verbatim: single "tiếng" (syllable unigram) 0.3636; bigram 0.3778 "but lost of precision for high recall"; unigram + dictionary-based compound words 0.4989; dictionary-based segmentation (dictionary of about 30,000 units, left-to-right) 0.5625.
- It also states two normalisation cases for retrieval (section 1): "the usage of vowels i and y is interchangeable ... 'bác sĩ' and 'bác sỹ'" and "the position of the tons may be variant, for example, 'hòa bình' and 'hoà bình' are acceptable". So the tone-placement problem was identified for Vietnamese IR in 2007.

### 4.2 Newer evidence (none is a clean controlled study)

- **`trnngcccuyn/vietnamese-legal-retrieval-DSC2026`** (UIT DSC 2026; commit `016bf68`; unreviewed experiment log). Tokenizer code `src/retrieval/tokenizers.py`: five backends `regex` (`\w+`, "mốc tham chiếu v0.1"), `whitespace`, `syllable_bigram` (syllables **plus** adjacent bigrams, bigrams not crossing punctuation), `pyvi`, `underthesea`. Config `configs/v0.2_bm25_tokenizer.yaml` states hypotheses H1 (word segmentation helps via higher IDF of compounds) and H1b (bigrams recover most of it); its header says "Kết quả: syllable_bigram + mean_top2 thắng". `experiments.csv`, row `v0.4_syl_notonefold`: "regex+fold_tone_false 0.8116 | syllable_bigram+fold_tone_false 0.8545 | syllable_bigram+fold_tone_true 0.8555. Hiệu số thứ nhất +0.0429 = công của tokenizer. Hiệu số thứ hai +0.0010 = công của fold_tone, KHÔNG phân biệt được: bootstrap KTC95 [+0.0000,+0.0030], McNemar 1-0". Row `v0.3_bm25_best_sylbigram`: McNemar 70-19, p=1.16e-07 for the combined change; the +0.0439 originally reported was corrected to +0.0429 after separating the confound. Setting: 1,000 dev questions, 524,422 chunks, BM25 `k1=1.5, b=0.75`, mean-of-top-2-chunks pooling, R@5. **The pyvi and underthesea arms are in the benchmark grid (`scripts/bench_retrieval.py`) but I found no numbers for them** in `experiments.csv` or the README. Between-set noise: "dao dong ~+-0.01 giua cac tap 1000 cau".
- **`monster1909/...`**: syllable unigrams + bigrams + accent stripping; no ablation (Section 3b).
- **Word segmentation vs syllable training in dense models**: Pham et al. 2022 (Section 1) compare syllable and word-level training for sentence-transformers, not BM25.
- **Search for a peer-reviewed comparison** (syllable unigram vs bigram vs segmented words, BM25, any domain, after 2007): **nothing found.** Queries run: effect of Vietnamese word segmentation on IR; syllable n-gram indexing comparison; Anserini/Pyserini Vietnamese ablation; TREC-style Vietnamese comparisons.

### Implications for the Week 4 plan

- Evidence ordering (weak): EVIA 2007 says dictionary segmentation >> bigram > unigram; DSC 2026 says unigram+bigram > unigram by +0.043 R@5; no source shows segmenter vs unigram+bigram head-to-head on modern data. `[inference]` unigram+bigram is better supported than pure unigram, and the segmenter arm remains an open question.
- **Pre-declare arms** (my recommendation, the literature does not decide): primary **U+B** = syllable unigrams plus adjacent syllable bigrams, bigrams never crossing punctuation or newline; secondary **U** = syllable unigrams only (the regression baseline); optional exploratory **W** = pyvi or underthesea word segmentation (extra dependency; segment before lowercasing as the DSC code comment advises: "Segment TRƯỚC khi lowercase"). Any other arm is a new hypothesis for the multiple-testing count.
- With 676 chunks and 45 clusters, a +0.04 difference in R@1 will not be significant (earlier notes, section 5). Report the discordant-pair counts, as the DSC log does.
- Tokenizer unit tests (no corpus needed):
  1. `"khoản 2 Điều 35"` -> keeps `khoản`, `2`, `điều`, `35` (and bigrams `khoản_2`, `2_điều`, `điều_35` in U+B).
  2. Bigram must not span a full stop: `"... quy định. Điều 5 ..."` does not produce `định_điều` (DSC comment: "Bigram không được bắc qua dấu câu").
  3. NFC input equals NFD input after tokenisation.
  4. Numbers: `"01 ngày"` vs `"1 ngày"` (142 leading-zero numbers in the corpus per the brief): no source treats this. `[inference]` canonicalise digit tokens by stripping leading zeros in both document and query tokens, and keep a test with `"01"`, `"1"`, `"10"`, `"100"`.
  5. Number words ("một", "ba mươi") are not mapped to digits; that is a lexical gap BM25 cannot close (dense branch's job). Record as a known limitation.

---

## 5. Vietnamese tone-mark placement

### 5.1 The two conventions and what each document prescribes

| Document | What it prescribes for oa/oe/uy (open syllables) | How I know |
|---|---|---|
| **Quyết định 1989/QĐ-BGDĐT**, Bộ Giáo dục và Đào tạo, signed 25/05/2018 by Deputy Minister Nguyễn Hữu Độ, in force from signing; "Quy định về chính tả trong Chương trình, sách giáo khoa giáo dục phổ thông" | **Style B.** Điều 8, nucleus written with one letter: "dấu thanh được đặt trên hoặc dưới chữ cái ghi âm chính của mỗi âm tiết"; examples "mái nhà, hoà nhạc, quý hoá, thuỷ thủ, mạnh khoẻ, trí tuệ". Two-letter nuclei: ia/ua/ưa -> first letter (bìa, lụa, lửa); iê/yê/uô/uơ -> second letter (biển, thuyền, nhuộm, được). | Transcription on https://luatvietnam.vn/giao-duc/quyet-dinh-1989-qd-bgddt-2018-quy-dinh-ve-chinh-ta-trong-chuong-trinh-sach-giao-khoa-gdpt-307547-d1.html (read). The official scan was not read; thuvienphapluat.vn returned HTTP 403 and I did not work around it. Whether it replaced an earlier decision is not stated on the page I read. |
| **Quyết định 240/QĐ**, Bộ trưởng Bộ Giáo dục, 5/3/1984, "Quy định về chính tả tiếng Việt và về thuật ngữ tiếng Việt trong các sách giáo khoa" | Not verified. Secondary summaries (search results) describe the same "tone on the letter of the main sound" rule with the example "hoà", i.e. Style B. | Title and date confirmed only via a citing article (https://vjol.info.vn/index.php/VJIAD/article/download/617/527, from a search summary). Full text not read. See NEED DOWNLOAD. |
| **Quyết định 07/2003/QĐ-BGDĐT**, Bộ Giáo dục và Đào tạo, 13/03/2003 | **Nothing on tone marks.** It is the interim rule on capitalising proper names in textbooks (and hyphenation of multi-syllable names). | https://vcci.com.vn/legal-document/quyet-dinh-072003qd-bgddt-cua-bo-giao-duc-va-dao-tao-ve-viec-ban-hanh-quy-dinh-tam-thoi-ve-viet-hoa-ten-rieng-trong-sach-giao-khoa (read: "The page says nothing about tone mark placement"). |

Neutral summary: the Ministry of Education textbook rule (2018) is Style B; the Labour Code consolidated text we index is Style A `[measured]`:

- `chunks.jsonl`, open syllables only: Style A 500 tokens in `text`, 681 in `context_text` (context_text counts: thỏa 386, hòa 126, ủy 58, tòa 31, khỏe 26, hỏa 9, tùy 8, dọa 7), Style B 1 (`tuý`). *Corrected 2026-11-25: the first figures (1,358 and 2) summed every string field of each record (`text`, `context_text` and the chapter/section/article titles).* Other `[measured]` counts, from the same all-fields pass and so inflated about twofold: `lý` 354 / `lí` 0, `kỹ` 122 / `kĩ` 0 (the zeros stand, so the i/y issue that EVIA mentions does not appear inside the corpus itself, but it can still appear in user queries).
- `[inference]` Legislative publishing (the consolidated text) follows a different habit from the education ministry's. I found no primary document that prescribes Style A for legal texts; do not claim one.

Vietnamese Wikipedia, "Quy tắc đặt dấu thanh trong chữ quốc ngữ" (secondary), labels Style A "kiểu cũ" and Style B "kiểu mới", and says "Đến năm 2022, các sách giáo khoa ở Việt Nam đặt dấu thanh theo 'kiểu mới'". https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_trong_ch%E1%BB%AF_qu%E1%BB%91c_ng%E1%BB%AF (read via fetch summary).

### 5.2 What open-source tools do (direction, scope, exceptions)

| Tool | Direction | Scope and exceptions | Source (quoted) |
|---|---|---|---|
| **underthesea `text_normalize`** | Style B -> A | **Dictionary lookup per token**, only for tokens of at most 6 characters; 968 rules in `token_rules.xlsx` (117 in group "oa", 44 "oe", 49 "uy"). Closed syllables are not rewritten by a general rule; the table instead **repairs** mis-toned closed syllables (`hòan -> hoàn`, `khỏan -> khoản`, `khỏang -> khoảng`, `hùynh -> huỳnh`, `chuỳ -> chùy`). `qu-` is kept on `y` (`qúy -> quý`, `qùy -> quỳ`, `qui -> quy`). Two rules go the other way: `lòa -> loà`, `nòa -> noà`. Character map in the shipped build is `ð -> đ`, `Ð -> Đ` (2 rules in the generated LaTeX table). | `underthesea/pipeline/text_normalize/token_normalize.py` (commit `22bdc17`): `if len(token) > 6: return token` ... `if token in token_map: return token_map[token]`. Rules read from `extensions/labs/text_normalize/token_rules.xlsx` (e.g. `hoà -> hòa`, `thuỷ -> thủy`, `khoẻ -> khỏe`). Caveat: the runtime artefact is the pickled `tn_rules_2023_07_14.bin`, built by `build_rules.py` from that xlsx; I did **not** unpickle it, so equality of the two is `[inference]`. |
| **BARTpho and PhoBERT pre-processing** (VinAI) | **Style A -> B** | Plain `str.replace` over a dict of 40 pairs (lower, Title, UPPER): `"òa": "oà"`, `"óa": "oá"`, `"ỏa": "oả"`, `"õa": "oã"`, `"ọa": "oạ"`, same for `e`, and `"ùy": "uỳ"`, `"úy": "uý"`, `"ủy": "uỷ"`, `"ũy": "uỹ"`, `"ụy": "uỵ"`. No `qu-` rule is needed (`quý` is `uý` not `úy`); closed syllables are untouched because their standard form never contains `òa`/`ùy` followed by a letter, `[inference]`. | https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md (file commit `f7e98cc`, 2021-09-24). BARTpho README: users "should perform Vietnamese tone normalization on the downstream task's data as this pre-process was also applied to the pre-training corpus". PhoBERT README: "PhoBERT employed the RDRSegmenter ... to pre-process the pre-training data (including Vietnamese tone normalization and word and sentence segmentation)". |
| **`chuan_hoa_dau_tu_tieng_viet`** (nguyenvanhieu.vn, copied in a Hugging Face Space) | -> Style A ("kiểu cũ") | Rule-based, works per word, lowercase only. Two vowels with the second at the word end -> tone on the first vowel (`hoà -> hòa`, `thuỷ -> thủy`); second vowel not at the end -> tone on the second (`hoàn`, `khoản` unchanged); `qu` and `gi` treat `u`/`i` as part of the onset; `ê`, `ơ` take the tone. Words failing `is_valid_vietnam_word` are returned unchanged. I ran it: `hoàn`, `khoản`, `huỳnh`, `khuỷu`, `ngoài`, `loại`, `hoạt` all unchanged. | https://huggingface.co/spaces/hoang1007/spelling-correction/raw/main/dataset/data_generation/normalize.py. Docstring: "Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ"; comment "dùng òa úy thay oà uý". Header: "Thằng code python này không giữ được lower/upper case". |
| **DSC 2026 `fold_tone_placement`** | Style B -> A (docstring says the reverse) | Regex `(?<!q)u[ỳýỷỹỵ]\|o[àáảãạèéẻẽẹ]` with **no end anchor**. Rewrites closed syllables. | `src/retrieval/tokenizers.py` lines 60-61: `# (?<!q): "quý", "quỷ" đã đúng chuẩn — đừng đụng vào chúng.` `_TONE_FOLD_RE = re.compile(r"(?<!q)u[ỳýỷỹỵ]\|o[àáảãạèéẻẽẹ]")`. Docstring line 71 says `("hòa" → "hoà")` but the map built at lines 53-55 sends `o + toned-a` to `toned-o + a`, i.e. `hoà -> hòa`. I ran it: `hoàn -> hòan`, `khoản -> khỏan`, `khoảng -> khỏang`, `ngoài -> ngòai`, `loại -> lọai`, `hoạt -> họat`, `toàn -> tòan`, `huỳnh -> hùynh`, `khuỷu -> khủyu`. Its own ablation found fold_tone worth +0.0010 R@5, indistinguishable from zero. |
| **monster1909 `_strip_accents`** | removes all diacritics | NFD + drop `Mn` + `đ -> d` | Section 3b. |
| vinorm, viet-text-tools, vistring | not established | `vistring` README documents only `remove_tones`. `viet-text-tools` is said to have a `new_style` flag (search summary); its PyPI page did not load. vinorm (a TTS number/date normaliser) was not checked. | https://github.com/pymivn/vistring/ |

### 5.3 Implications for the Week 4 plan (concrete tokenizer rules)

1. **Order of operations inside the BM25 tokenizer only**: NFC -> lowercase -> split into syllables -> per-token tone fold -> (bigrams) -> emit. Never fold the text that is shown to users, quoted in answers, or passed to the article-reference parser; folding `khoản` would break `\bkhoản\b`.
2. **Fold to Style A** (the corpus style, so `thỏa thuận` stays `thỏa thuận` and `hòa` stays `hòa`). Open syllables only, end-anchored. My tested implementation (run against the full corpus: it changes exactly 1 distinct token, `tuý -> túy`, 2 occurrences):

```python
_B2A = {}
for first, second, t1, t2 in (("o","a","òóỏõọ","àáảãạ"), ("o","e","òóỏõọ","èéẻẽẹ"), ("u","y","ùúủũụ","ỳýỷỹỵ")):
    for a, b in zip(t1, t2):
        _B2A[first + b] = a + second          # "oà" -> "òa", "uý" -> "úy"
_RE = re.compile(r"(?<![a-zà-ỹ])([a-zà-ỹ]*?)(oà|oá|oả|oã|oạ|oè|oé|oẻ|oẽ|oẹ|uỳ|uý|uỷ|uỹ|uỵ)(?![a-zà-ỹ])")
def fold_token(tok):                  # tok: NFC, lowercase, one syllable
    m = _RE.fullmatch(tok)
    if not m or (m.group(2)[0] == "u" and m.group(1).endswith("q")):
        return tok                    # not open oa/oe/uy, or qu- (quý, quỳ keep tone on y)
    return m.group(1) + _B2A[m.group(2)]
```

3. **Required test cases** (I ran each; expected outputs): `hoà -> hòa`, `hòa -> hòa`, `thuỷ -> thủy`, `khoẻ -> khỏe`, `toà -> tòa`, `thoả -> thỏa`, `huỷ -> hủy`, `tuỳ -> tùy`, `uỷ -> ủy`, `loè -> lòe`. Unchanged (closed or other nuclei): `hoàn`, `khoản`, `khoảng`, `toàn`, `đoàn`, `ngoài`, `loại`, `hoạt`, `hoạch`, `hoặc`, `xoắn`, `xoáy`, `khoèo`, `huỳnh`, `khuỷu`, `tuýt`, `tuyển`, `thuyền`, `quý`, `quỳnh`, `quỵ`, `quy`, `giá`, `giàu`. Cross-check `fold("thoả thuận") == fold("thỏa thuận")`.
4. **Not covered by this rule and not needed for the corpus**: closed syllables typed with the first-vowel tone (`hòan`, `khỏan`): underthesea handles them by table; we do not, `[inference]` because they are rare typos. If wanted, add them as an explicit small dict, not a regex.
5. **i/y** (EVIA): the corpus is consistent (`lý` 354 vs `lí` 0, `kỹ` 122 vs `kĩ` 0) `[measured]`. Users may type `lí do`, `quản lí`. Do not add a rule without a measured miss; log it as a known gap and add one query-side test later.
6. **Dense side**: BARTpho/PhoBERT normalise to Style B before training; I do not know what bge-m3's pre-training corpus did. A cheap check: embed the same sentence in Style A and Style B and compare cosine, but this was not run. `[inference]`
7. **Diacritics removed entirely** (queries typed without accents): README-level evidence (Messiac24: hybrid Recall@10 0.1447 vs 0.9772) suggests this is the larger failure; a diacritic-stripped BM25 field (monster1909 style) is a one-line exploratory arm, but it merges words, so do not make it primary. Out of scope for the frozen 90 questions.
8. Count the fold arm as one more hypothesis only if evaluated separately; DSC's log shows tone folding adds nothing measurable at 1,000 questions (+0.0010, CI [0.0000, 0.0030]); our value is correctness on real user queries, not recall on the frozen set. Say so in `eval/results.md`.

---

## 6. Explicit statute references in queries

### 6.1 COLIEE Task 3 (Japanese Civil Code) and other benchmarks

- **COLIEE 2021 overview** (Rabelo, Goebel, Kim, Kano, Yoshioka, Satoh), *Review of Socionetwork Strategies* 16:111-133 (2022), DOI 10.1007/s12626-022-00105-z, open access (CC BY 4.0). https://link.springer.com/content/pdf/10.1007/s12626-022-00105-z.pdf (read the Task 3 discussion). Queries are bar-exam statements; one example given is verbatim article text ("An underground space or airspace may be established as the subject matter of superficies for ownership of structures..."), the hard ones use anonymised symbols ("B obtained A's bicycle by fraud..."). **Nothing in the text I read says queries cite article numbers** or that participants exploited or removed them. HUKB's use of reference information between articles ("add detailed information from the referred articles"; "identify the relationships among articles based on the reference information with rank-based retrieval approach") is described as "not currently as good as expected". That is corpus-side cross-referencing, comparable to our 131 in-text references (in 87 chunks; an earlier "822" also counted each chunk's own heading), not query-side parsing.
- **COLIEE 2024 Task 3 summary slides** (Yoshioka, https://jurisinformaticscenter.github.io/jurisin2024/COLIEE2024_task3summary.pdf, read in full): 768 articles; 1,097 training and 109 test questions; macro F2; difficult questions mostly use anonymised symbols (9 of 15 questions with F2 below 0.2); no mention of article numbers in queries. I did **not** read individual participant papers. So "do queries cite article numbers" is **unverified**, not "no".
- **COLIEE 2025 overview** (*Review of Socionetwork Strategies* 2026, DOI 10.1007/s12626-026-00199-9; open copy on PubMed Central, https://pmc.ncbi.nlm.nih.gov/articles/PMC13095985/, CC BY 4.0). *Added 2026-11-25; the Task 3 section was read by a sub-agent and not re-checked by the coordinator.* "covering 768 articles"; "1,206 questions in total" for training; "73 new questions" from the 2024 bar exam (55 need one article, 17 two, one three); metric: macro-averaged F2. Best run JNLP_RUN1, F2 0.836 (bge-m3 pre-retrieval, then a reranker); CAPTAIN.H2 F2 0.830. BM25 is named only in two ensembles (IRNLPUI with a cross-encoder LLM, UA with a bi-encoder LLM); "All teams leveraged LLMs for computing similarity between questions and legal articles." No query is discussed as citing an article number; the only numbers appear in the results discussion ("The relevant article for this question is Article 472, but many teams incorrectly selected Article 472-2"). Still unverified for queries in general.
- **BSARD** (Louis and Spanakis, ACL 2022, arXiv 2108.11792, https://arxiv.org/pdf/2108.11792v2, read): the regex is applied to the **experts' answers**: "The questions collected are annotated with plain text references to relevant law articles (e.g., "Article 8 of the Civil Code"). We use regular expressions to parse these references and match them to the corresponding articles from our corpus." Questions are anonymised "as close as possible as if a layperson had asked it". So the benchmark's gold labels are built by reference parsing, and the questions do not carry them.
- **IL-PCSR** (Paul et al., arXiv 2511.00268, https://arxiv.org/pdf/2511.00268, read): "To prevent models from directly exploiting citation strings, we masked all portions of query texts where explicit references occurred (section numbers, Act names, or case titles)", replaced by `[SECTION]`, `[ACT]`. This is a deliberate neutralisation of the trick we want to use.
- **Fink, Recski, Kusa, Hanbury**, "Statute-enhanced lexical retrieval of court cases for COLIEE 2022", arXiv 2304.08188 (https://arxiv.org/pdf/2304.08188, read): statute titles and acronyms are matched in case text; "we use regular expressions to detect statute section numbers in the text" (the pattern itself is not printed); section numbers are mapped to the most frequently co-occurring statute; an Elasticsearch `statute-section` field scored by BM25 and added with a boost `sb`, plus a passage boost `Pb` when the query passage mentions at least one section. Table 2 (precision/recall/F1): dev passage-level LM 0.0993 / 0.2341 / 0.1395 -> with statute field 0.1073 / 0.2249 / 0.1453; training 0.1210 / 0.2218 / 0.1565 -> 0.1282 / 0.2090 / 0.1589. Gains are small and single-run. Query-side parsing for case retrieval, not statute retrieval.
- A search found no study reporting how often questions in a statutory-article benchmark contain article numbers, nor one that filters retrieval by a parsed query reference with a measured effect (searches: COLIEE article-number exploitation; statute-citation extraction from queries to filter or boost; legal QA article-number analysis).

### 6.2 Vietnamese reference detection (documents, not queries)

- **`hoadm-net/SA-RAG`** (commit `8d7e17e`, pushed 2026-11-24, no licence declared): `src/sarag/parsing/patterns.py` (article heading) and `src/sarag/expansion/structural.py` (in-text references). The repository declares no licence, so its code is described here, not copied (and must not be copied into this project):
  - The article-heading pattern accepts an optional letter suffix after the number (`Điều 5a`), which this law does not have (see the parser negatives below).
  - The in-text reference pattern is case-insensitive and optionally prefixed by point letters (`điểm a, b`) and clause numbers (`khoản 1, 2 và 3`), each list separated by commas, "và" or "hoặc"; the article part is either "này" or a list of numbers.
  - A list of document-type words decides whether a following document name makes the reference external. That list includes "Quy định", which is why the natural question form "Điều 24 quy định gì" is wrongly treated as external (see the table below).

  `find_references` skips a reference when the next 80 characters match `_OTHER_DOC_RE` and not `_THIS_DOC_RE` (docstring: "'Điều 53 Luật Dự trữ quốc gia', 'khoản 3 Điều 30 Nghị định số 29/2012/NĐ-CP' is external and skipped. A bare reference with no document named is taken as internal."). I ran `find_references` on 16 strings:

| Input | Result |
|---|---|
| `...quy định tại khoản 2 Điều 35 của Bộ luật này` | `Reference('35', (2,))` (correct) |
| `Điều 24 Bộ luật Dân sự quy định gì?` | `[]` (correct: other statute) |
| `Điều 24 Luật Lao động 2012` | `[]` (correct) |
| `khoản 3 Điều 30 Nghị định số 29/2012/NĐ-CP` | `[]` (correct) |
| `Điều 3 và Điều 5 nói gì` / `Điều 24, 25 và 26` | two / three references (correct) |
| `các khoản 1, 2 và 3 Điều 36` | `Reference('36', (1, 2, 3))` |
| `điểm a khoản 1 Điều 35` | `Reference('35', (1,))` (the point `a` is dropped) |
| `Điều 5a`, `Điều` + NBSP + `24` | matched |
| `Điều này không áp dụng` | `Reference('', ())` (self reference) |
| `Dieu 24 la gi` | `[]` (no diacritics: not detected) |
| **`điều 24 quy định gì`** | **`[]` (false negative: "quy định" is in `_DOC_TYPES`)** |
| **`Tôi muốn hỏi điều 1 chút`** | **`Reference('1', ())` (false positive: lowercase, unrelated meaning)** |
| `nghỉ sau 24 tháng thì sao` | `[]` (correct) |

  Not a published benchmark. It tests documents, not user questions.
- `monster1909` `DOCNUM_RE` (`\d{1,4}/(?:\d{4}/)?[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\-]{1,25}`) keeps decree numbers such as `75/2017/NĐ-CP` whole and adds the owning document's number as a token to each chunk "so queries citing a decree number match its body chunks". That is the closest in-the-wild example of exploiting an explicit identifier in a Vietnamese legal query; no ablation of it is reported.
- `trnngcccuyn/...DSC2026` `experiments.csv`: a "relation boost" using slug Jaccard and cited document numbers (`v0.5_relboost_dev`, dev R@5 0.8585 vs 0.8555, "+0.0030 đến từ câu đơn", "KẾT QUẢ ÂM"): small, negative-leaning evidence for citation-based boosts at document level.
- `Structure and Lineage Evidence for Vietnamese Statutory Article Retrieval on DRiLL` (Nguyen and Nguyen, arXiv 2611.09200, https://arxiv.org/pdf/2611.09200, abstract and feature table read): citation and lineage features computed on candidate articles improve official F2 from 71.77% to 72.91% over five seeds, with features that include "Article numbers use corpus positions or detected introduced aliases", "an explicitly named different instrument", and a letter-suffix check. Document-side. No query-side parsing.
- A search summary also mentioned a Vietnamese KG paper (arXiv 2309.09069) that uses regex to find sentences that mention laws and matches them to a list of 225 laws; I opened it and it gives no regex details.

### 6.3 Implications for the Week 4 plan (parser rules and test cases)

- **The sparse branch can already be fooled by citations.** `[inference]` With 131 in-text references in 87 chunks, like "theo quy định tại Điều 35" (`d34:k9`), BM25 over `điều 35` / `35` will score chunks that *cite* Điều 35 as well as chunks *of* Điều 35. For article_ref queries, measure sparse-only rank of the gold article against its citing chunks (a cheap diagnostic before the filter exists).
- **Parser positives (must activate; article set):** `Điều 35`, `điều 35` (lowercase, when followed by a clause-like or question context), `khoản 2 Điều 35` -> `{35}` (clause kept separately or ignored, decide), `điểm a khoản 1 Điều 35`, `Điều 3 và Điều 5` -> `{3, 5}`, `Điều 24, 25 và 26` -> `{24, 25, 26}`, `các khoản 1, 2 và 3 Điều 36`, NBSP and double spaces, trailing punctuation (`Điều 35?`), **`Điều 24 quy định gì`** and **`Điều 24 nói về cái gì`** (do not copy SA-RAG's `Quy định` in `_DOC_TYPES`; match a document type only when followed by a title or number, e.g. `Bộ luật Dân sự`, `Luật ... \d{4}`, `Nghị định số`).
- **Parser negatives (must not activate / must not filter):** other statute named after the number (`Điều 24 Bộ luật Dân sự`, `Điều 24 Luật Lao động 2012`, `khoản 3 Điều 30 Nghị định số 29/2012/NĐ-CP`, `Điều 24 của Luật Việc làm`); self reference `Điều này`; numbers without the word Điều (`sau 24 tháng`, `30 ngày`, `01 ngày`); `điều 1 chút`, `điều kiện 2`; article numbers outside 1..220 (fall back to unfiltered retrieval, no empty result); `Dieu 24` (decision needed: ignore, or accept with a flag; if accepted, add it to both lists as a pre-declared choice).
- **Our law has no letter-suffixed articles** `[measured]`: all 220 `article` values in `data/processed/articles.jsonl` are plain digits. Do not copy SA-RAG's `\d+[a-zđ]?`; `Điều 5a` should not map to 5 and should not activate the filter. Test it.
- **Zero false activations** on the 25 out-of-scope questions and about 20 number-bearing non-article questions (earlier notes), reported as a separate parser metric. The article_ref stratum (12 questions, dense R@1 9/12) measures the regex, not retrieval.
- **Soft alternative** stays the safer default: add a third RRF input from the article-filtered branch (the plan's exploratory arm), because the literature offers no evidence that a hard filter beats a boost on statute retrieval, and Fink's boost is the only published precedent (about +0.006 F1, single run).
- State in `eval/results.md` that the literature has no measured effect for query-side reference parsing; our result on 12 questions is descriptive only.

---

## NEED DOWNLOAD

1. **Zalo AI Challenge 2021, Legal Text Retrieval: official task page.**
   - Citation: Zalo AI Challenge 2021, "Legal Text Retrieval" (VNG/Zalo AI).
   - DOI: none.
   - Official URL: https://challenge.zalo.ai/portal/legal-text-retrieval
   - Why it failed: single-page JavaScript app; the fetch returned only the title (and a "2026" banner); an archive.org fetch was refused by the tool.
   - Needed: task description, dataset sizes (corpus articles, train/test questions), the official metric, timeline, winner list. I have the sizes only second-hand from arXiv 2211.02200 (3196 questions, 61425 articles).
2. **ALQAC 2022 overview.** Nguyen et al., "ALQAC 2022: A Summary of the Competition", KSE 2022, pp. 1-5, DOI 10.1109/KSE56063.2022.9953764 (IEEE Xplore; https://doi.org/10.1109/KSE56063.2022.9953764). Not opened: IEEE paywall, and I do not bypass it. Needed: the Task 1 results table (which teams, which F2) and the dataset statistics.
3. **ALQAC 2023 overview.** "A Summary of the ALQAC 2023 Competition" (KSE 2023, Hanoi, 18-20 Oct 2023, listed at https://sites.google.com/view/ISAILD-2023). DOI not found; two KSE 2023 Crossref DOIs (10.1109/KSE59128.2023.10299515, 10.1109/KSE59128.2023.10299426) were seen but not matched to the title. Needed: Task 1 leaderboard and data description. Also the ALQAC 2024 summary, if any (site https://sites.google.com/view/ALQAC-2024).
4. **Quyết định 240/QĐ, 5/3/1984** (Bộ Giáo dục), "Quy định về chính tả tiếng Việt và về thuật ngữ tiếng Việt trong các sách giáo khoa". No DOI. Search suggests the official database https://vbpl.vn (search "240/QĐ 1984"). Failed: no full text surfaced in search; thuvienphapluat.vn returned HTTP 403. Needed: the section on dấu thanh and the exact wording for oa/oe/uy.
5. **Quyết định 1989/QĐ-BGDĐT (25/05/2018), official text.** https://thuvienphapluat.vn/van-ban/Giao-duc/Quyet-dinh-1989-QD-BGDDT-2018-quy-dinh-chinh-ta-Chuong-trinh-sach-giao-khoa-giao-duc-pho-thong-445355.aspx returned HTTP 403 to the fetch tool (site refuses automated access; not circumvented). I read a transcription on luatvietnam.vn only. Needed: Điều 8 of the attached Quy định and any "replaces" clause, from the official Ministry source (moet.gov.vn or the Công báo).
6. **[Partly resolved 2026-11-25: COLIEE 2025 read via PubMed Central, see §6.1; the 2022-2024 overviews are still unread.]** **COLIEE overview papers 2022-2025 (Task 3 section)**, to settle whether bar-exam queries ever cite article numbers. Candidates: COLIEE 2025 overview, *Review of Socionetwork Strategies* 2026, DOI 10.1007/s12626-026-00199-9 (https://link.springer.com/article/10.1007/s12626-026-00199-9; the fetch was redirected to a login/cookie handshake). Also the COLIEE 2023 summary PDF (https://sites.ualberta.ca/~rabelo/COLIEE2023/COLIEE2023_summary.pdf returned 404). Needed: Task 3 query description and any discussion of explicit article references.
7. **`viet-text-tools` PyPI page** (https://pypi.org/project/viet-text-tools/0.1.5/): fetch failed to render. Needed only to confirm what its `new_style` flag does for oa/oe/uy; low priority.
8. **[Resolved 2026-11-25: `lexical.py` was read; see §3a, BM25 is underthesea-segmented with a silent regex fallback.]** **ViRE `docs/usage.md` and `src/vi_retrieval_eval/runner.py`** (https://github.com/longstnguyen/ViRE) would settle whether the EACL 2026 BM25 numbers used underthesea segmentation or the regex fallback (`HAVE_UTS`). Not fetched. Needed: the default flags and a run log, if published.
