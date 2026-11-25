# Week 4 hybrid retrieval: primary-source review

Date: 2026-11-25. Scope: Qdrant v1.20 sparse/fusion/filter behaviour, RRF, bge-m3 sparse head, BM25 for Vietnamese, significance testing, tuning discipline, lexical bias, explicit article references.

Conventions in this file:

- "Read" means I opened the primary text (PDF, source file, HF file) in this session.
- `[my arithmetic]` / `[my inference]` mark statements that are not in any source.
- Qdrant source links point at tag `v1.20.1` (checked out locally; HEAD commit dated 2026-11-13). Line numbers are for that tag.
- Paper numbers are copied from the text I read. Where the PDF text extraction was ambiguous I did not quote the number.

---

## Summary: the findings that matter most for the plan

1. **Qdrant's server-side RRF default is k = 2, not 60.** `DEFAULT_RRF_K: usize = 2`; `Fusion.RRF` in qdrant-client means "RRF with default parameters". k is only configurable via `RrfQuery(rrf=Rrf(k=60))` (docs: available as of v1.16.0). Qdrant also uses 0-based positions: score = `1/(pos + k)`. The plan says "RRF in Python, k=60": that is fine, but do not mix it up with `FusionQuery(fusion=Fusion.RRF)`.
2. **`Modifier.IDF` only multiplies the *query-side* sparse weights** by `ln((N - df + 0.5)/(df + 0.5) + 1)`, using collection-wide N and df by default. The client must still produce the document-side BM25 term weight (`tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))`) and send query weights of 1.0. The official fastembed BM25 does exactly this.
3. **For 676 x 1024-d float vectors (about 2.7 MB) Qdrant never builds or uses HNSW.** The collection is far below `indexing_threshold_kb: 10000` and `full_scan_threshold_kb: 10000`, so every search is a plain exact scan. `exact=True` is redundant but harmless. A payload index on `article` is recommended, not required (unless strict mode forbids unindexed filters, which is off by default).
4. **Top-level filters are propagated into every prefetch leaf** in v1.20.1 (`planned_query.rs`, "Filters are propagated into the leaves"). Still set the filter explicitly on each prefetch for readability.
5. **RRF k=60 was fixed in a pilot and never tuned in Cormack et al. 2009, and the paper's own Table 1 shows MAP is flat for k in 30..100** (between .2138 and .2147 for k=30..100, .2072 at k=0, .2098 at k=500). But they fused 30 configurations of one engine and TREC runs, not a dense+sparse pair. **Bruch et al. find RRF sensitive to its parameters and convex combination (CC) better in- and out-of-domain, and CC tunable with "a handful of labeled queries".** No query count is given in the text I read.
6. **Pitfall of k=60 with 30-deep lists** `[my arithmetic]`: a document at any rank <= 30 in *both* branches scores `>= 2/90 = 0.0222`, which beats a document at rank 1 in only one branch (`1/61 = 0.0164`). R@1 can therefore drop when the correct dense top-1 is absent from the sparse top-30.
7. **bge-m3 sparse is not available through sentence-transformers.** The HF repo's `modules.json` lists only Transformer, Pooling, Normalize. The sparse head is `sparse_linear.pt` (3.52 kB), applied as `relu(Linear(hidden_state))` per token, max-aggregated per token id, with cls/eos/pad/unk zeroed (FlagEmbedding code). Using it needs FlagEmbedding or manual use of `encode(output_value="token_embeddings")` `[my inference]`.
8. **There is no monolingual Vietnamese evidence for bge-m3 sparse or dense+sparse.** MIRACL has no Vietnamese. MKQA (Vietnamese queries over an English corpus) shows sparse at 51.8 Recall@100 vs dense 76.6 and dense+sparse 76.8, but cross-lingual overlap is near zero, so it says nothing about monolingual lexical matching. In M3's own tokenizer ablation, BM25 with a Lucene analyzer beats BM25 on XLM-R tokens by 6.6 nDCG@10 on MIRACL (38.5 vs 31.9): tokenisation matters.
9. **Vietnamese BM25 tokenisation: no controlled modern comparison found.** The only measured syllable-vs-word number I read is old (EVIA 2007, SMART ltc weighting, not BM25): syllable unigram 0.3636, syllable bigram 0.3778, dictionary word segmentation 0.5625 average precision. Vietnamese legal-retrieval papers use word segmentation (pyvi, VnCoreNLP RDRSegmenter) without ablating it. So syllable-level vs segmented is an open empirical question; pre-declare one and report the other as an exploratory run.
10. **Statistics for n = 90 (45 clusters):** the 90 queries are not independent; use the 45 gold clusters as the unit. For MRR@10 use a paired sign-flip permutation test on per-cluster mean differences (Smucker's randomization test with clusters as topics) plus a cluster bootstrap CI. For R@1 on binary outcomes use the exact (or mid-p) McNemar test on discordant pairs, per stratum, and a cluster-level version for the pooled set. Smucker/Urbano's "discontinue the sign test" is about comparing *means* of graded scores, not binary R@1. A small calculation shows how few queries can be significant (section 5).
11. **Do not tune fusion/BM25 hyper-parameters on the 90 frozen questions.** Fuhr (SIGIR Forum 2017) requires a tuning set disjoint from the test set, hypotheses fixed before the experiment, and multiple-testing correction, and notes that collection reuse is itself multiple testing. Qdrant's own docs say hand-tuned weights without measurement are "unlikely to beat the default reliably". Use fixed a-priori values.
12. **The "direct" questions are the lexically biased half.** BEIR (section 6) and ORQA both document that question-writers (or annotators) who see the passage create artificial lexical overlap that favours BM25-like systems. Gold drafted by an LLM from verbatim chunks has the same mechanism. Report direct, paraphrase and article_ref separately and make the paraphrase stratum the headline for hybrid-vs-dense.
13. **MRR is an ordinal-scale metric** (Fuhr 2017; Ferrante et al.). It is acceptable as the declared headline if R@1 and a rank-based second metric (mean first relevant rank, MFR, with a cap) are reported next to it. Recall@5 is saturated at 1.000, so switching the headline is justified; declare the switch before running.

---

## 1. Qdrant (v1.20)

### 1.1 Sparse vector config and named vectors

- Sparse vectors need a name; "Outside of a unique name, there are no required configuration parameters for sparse vectors". The distance for sparse vectors is always dot product. Source: https://qdrant.tech/articles/sparse-vectors/ and https://qdrant.tech/documentation/concepts/vectors/ (sparse vectors have no fixed length; non-zero entries limited by u32; stored in a separate sparse index; indices need not be sorted).
- The modifier is set at creation time and cannot be changed afterwards; the examples use `models.Modifier.IDF` (Python), `"modifier": "Idf"` (REST). Source: https://qdrant.tech/documentation/concepts/collections/ (sparse vectors section).
- Named vectors can be added to or removed from an existing collection without recreating it (docs mark this as available from v1.18.0). Source: https://qdrant.tech/documentation/concepts/vectors/.
- qdrant-client 1.20.0: `SparseVectorParams(modifier=...)`, `Rrf(k, weights)`, `RrfQuery`, `Fusion` enum (`RRF`, `DBSF`), `IntegerIndexParams`, `TextQuery`/`Bm25Scoring` exist in `qdrant_client/http/models/models.py` (wheel from https://pypi.org/project/qdrant-client/1.20.0/; I read `Rrf` at line 2965, `Fusion` at 1130, `IntegerIndexParams` at 1662, `TextQuery` at 3819).
- Not confirmed from docs or source: whether one collection may hold the current *unnamed* dense vector plus a named sparse vector, or whether the dense vector must be renamed. With 676 points it is cheap to re-create the collection with named `dense` + `bm25` vectors.

### 1.2 `Modifier.IDF`: exact formula and what the client still computes

- The docs pages I read (vectors, collections, sparse-vectors article) show the modifier but **do not define the formula**. The formula is in source.
- Source, `fancy_idf`: `((n - df + 0.5) / (df + 0.5) + 1.).ln()`, documented as "Advanced formula for Inverse Document Frequency (IDF) according to wikipedia": https://github.com/qdrant/qdrant/blob/v1.20.1/lib/segment/src/data_types/query_context.rs (lines 95-100).
- It is applied to the **query vector's weights**: `remap_idf_weights` does `weight *= fancy_idf(n, df)` for each query index, where `n` = number of indexed vectors in that sparse vector field and `df` = documents containing that dimension. Same file, lines 423-440; call site https://github.com/qdrant/qdrant/blob/v1.20.1/lib/segment/src/index/sparse_index/sparse_vector_index/read_view/search.rs (lines 45-56).
- Statistics are collection-wide by default; `SearchParams.idf` can restrict them to a filter-defined corpus ("By default (or with explicit "global") statistics are collection-wide"): https://github.com/qdrant/qdrant/blob/v1.20.1/lib/segment/src/types.rs (lines 713-736). **Consequence: filtering by `article` does not change IDF.**
- Compare Robertson & Zaragoza Eq. 3.3, `w_IDF = log((N - n_i + 0.5)/(n_i + 0.5))`: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf. Qdrant's `+ 1` inside the log keeps it non-negative; at df > N/2 the textbook form goes negative.
- **What the client must compute** (fastembed's BM25 is the reference; it is documented to be used with `modifier="idf"`): document-side weight per term `tf*(k+1) / (tf + k*(1 - b + b*|d|/avg_len))`, query-side weight 1.0 per token, token id = `abs(mmh3.hash(token))`. Defaults k=1.2, b=0.75, avg_len=256.0. Source: https://github.com/qdrant/fastembed/blob/main/fastembed/sparse/bm25.py (docstring ~lines 79-98, `_term_frequency`, `compute_token_id` at 362-363, query-embed comment at 365-367).
  - `avg_len` is a fixed constructor argument, not computed from the corpus. Set it to the real mean token count of the 676 chunks.
  - fastembed's default stemmer/stopword list cover 16-ish languages; "vietnamese" was not in the visible `supported_languages` list (I printed the first part of the list only), so use `disable_stemmer=True` or a custom tokenizer.
  - Hash collisions are possible with a 32-bit hash; negligible for a few thousand syllable types, but build a vocabulary dict (`term -> int id`) instead to remove the concern.
- **New in the v1.20.1 source (not verified in docs):** a server-side BM25 query over a payload text index (`TextQuery`, `Bm25Params{k, b}`, defaults k=1.2, b=0.75, avg_len default 256). Tokenizer options in the text index are `prefix`, `whitespace`, `word`, `multilingual`, `code`. Source: https://github.com/qdrant/qdrant/blob/v1.20.1/lib/api/src/rest/schema.rs (lines 180-224, 721-793) and https://github.com/qdrant/qdrant/blob/v1.20.1/lib/segment/src/data_types/index.rs (lines 306-315). It would remove the need to hash tokens client-side, but how the `word`/`whitespace` tokenizers treat Vietnamese diacritics and NFC is unverified, and there is no docs page I read. Treat it as an alternative to evaluate, not as the default path.

### 1.3 Query API `prefetch` + fusion: RRF k default, DBSF

- Docs (read): "Whenever a query has at least one prefetch, Qdrant" runs the prefetches then the main query on their results; prefetches can be nested; `offset` only affects the main query; prefetch `limit` should be at least `limit + offset`. RRF: `"query": { "rrf": {} }`; score = sum of `1/(k + rank)` with zero-based positions; "k is a constant (set to 2 by default)"; setting k "Available as of v1.16.0"; weighted RRF "available as of v1.17.0" (weights array in prefetch order). DBSF available as of v1.11.0: per-retriever mean and sample std, scores mapped from (mean - 3 sigma, mean + 3 sigma) to [0,1], summed, not clipped, `0.5` emitted if all scores equal. DBSF has no documented parameters. Source: https://qdrant.tech/documentation/concepts/hybrid-queries/.
- Source confirms: `pub const DEFAULT_RRF_K: usize = 2;` and `1.0 / ((position + 1) as f32 / weight + k as f32 - 1.0)`, which equals `1/(position + k)` at weight 1; "Does not break ties". https://github.com/qdrant/qdrant/blob/v1.20.1/lib/segment/src/common/reciprocal_rank_fusion.rs (lines 13-39, 54-99).
- REST schema: `Rrf.k` has `range(min = 1)`: https://github.com/qdrant/qdrant/blob/v1.20.1/lib/api/src/rest/schema.rs (lines 541-555). qdrant-client docstring: "`rrf` - Reciprocal Rank Fusion (with default parameters)" (`models.py` lines 1130-1133), so `Fusion.RRF` means k=2.
- Docs advice for choosing a fusion method: with an eval set, weighted RRF tuned on a split; trust raw scores without an eval set: DBSF; neither: default RRF. They also say "Hand-tuned weights without measurement are unlikely to beat the default reliably."
- Filter propagation: top-level filter is merged into each leaf prefetch: https://github.com/qdrant/qdrant/blob/v1.20.1/lib/shard/src/query/planned_query.rs (lines 344-368, `Filter::merge_opts(propagate_filter.clone(), filter)`). The docs page I read did not state this (page truncated at 100k chars).

### 1.4 Filtering with `MatchValue` / `MatchAny` on an integer field; payload index

- Docs: match on integer payloads uses the same condition form (`MatchValue(value=0)` example); Match Any "works as a logical OR for the given values", is the `IN` operator, applies to keyword and integer payloads, available since v1.1.0. https://qdrant.tech/documentation/concepts/filtering/.
- Index: "For performant filtering, create payload indexes for the fields you plan to filter on." and "For best results, create payload indexes before ingesting data." (filtering page); "Payload indexes should be created before ingesting data", "you need to rebuild the HNSW index to take advantage of the new payload indexes" (https://qdrant.tech/documentation/concepts/indexing/). So an index is **recommended, not required** for correctness.
- Strict mode can forbid unindexed filtering: `unindexed_filtering_retrieve` exists in `config/config.yaml` (lines 325-326, commented out, i.e. unset): https://github.com/qdrant/qdrant/blob/v1.20.1/config/config.yaml.
- Integer index params in qdrant-client 1.20.0: `lookup` (direct match, default true) and `range` (default true). `MatchValue/MatchAny` need `lookup`. Source: `models.py` lines 1662-1665.

### 1.5 Exact vs HNSW for a tiny collection

- Source `hnsw/read_view/dispatch.rs`: unfiltered search is plain when `exact || hnsw disabled || available_vector_count < full_scan_threshold`; filtered search is plain when `exact`, or when the estimated max cardinality is `< full_scan_threshold`. https://github.com/qdrant/qdrant/blob/v1.20.1/lib/segment/src/index/hnsw_index/hnsw/read_view/dispatch.rs (lines 43-70, 94-146).
- `full_scan_threshold` is configured in KB and converted to a vector count using the average vector size: https://github.com/qdrant/qdrant/blob/v1.20.1/lib/segment/src/index/hnsw_index/hnsw.rs (lines 186-213). Defaults: `full_scan_threshold_kb: 10000` and `indexing_threshold_kb: 10000` in https://github.com/qdrant/qdrant/blob/v1.20.1/config/config.yaml (lines 217-222, 255-261; "Maximum size (in KiloBytes) of vectors allowed for plain index"). Docs example output shows different sample numbers (full_scan 10000, indexing 20000), so read the live collection info rather than trusting the docs sample.
- Arithmetic `[my arithmetic]`: 676 x 1024 x 4 bytes = 2,768,896 bytes = about 2,704 KB, below both 10,000 KB thresholds. The segment is a plain (non-indexed) segment; search is a full scan. `SearchParams(exact=True)` additionally disables quantization (same dispatch file, lines 46-58) and is a no-op for correctness here.

### Implications for the Week 4 plan (Qdrant)

- Create a **new collection** `named vectors: dense (1024, cosine) + bm25 (SparseVectorParams(modifier=Modifier.IDF))`, plus an integer payload index on `article` (`lookup=True`). Recreate rather than patch, so the dense-only baseline stays reproducible in the old collection.
- Keep **RRF in Python** with k=60, 1-based ranks (`1/(60+rank)`), deterministic tie-break (higher dense rank first, then chunk id). If you ever call server-side RRF, pass `RrfQuery(rrf=Rrf(k=60))` and note that its 0-based form equals the paper's formula with k=59 `[my arithmetic]`. Add a unit test that fails if `Fusion.RRF` is used.
- Pull the sparse and dense branches with a separate `query_points(..., using=..., limit=30)` and fuse client-side, so the plan's "top 30 per branch" is explicit and testable. Optionally cross-check one query against server-side RRF with k=60.
- Document-side BM25 weights are computed once at ingest from the 676 chunks. Set `avg_len` to the measured mean token count, not 256. Query weights 1.0 (repeat terms may be summed; R&Z section 3.5.1 says qtf saturation "was unimportant").
- A hard filter on `article` is cheap (plain scan). Acceptance check: with the filter and `MatchValue(article=N)`, all returned chunks have `payload.article == N`; with an out-of-range N (> 220) the result is empty and the code falls back to unfiltered retrieval (design choice, see section 8).
- Check `client.get_collection(...).config` after creation to confirm `hnsw_config.full_scan_threshold`, `optimizer_config.indexing_threshold` and that `points_count == 676`; record them in the eval log.

---

## 2. RRF

### 2.1 Cormack, Clarke, Buttcher (SIGIR 2009)

Read: http://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf (redirect target of https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

- Formula: `RRFscore(d in D) = sum over r in R of 1 / (k + r(d))`, where rankings are "each a permutation on 1..|D|" (1-based rank).
- k: "where k = 60 was fixed during a pilot investigation and not altered during subsequent validation."
- Rationale quoted: highly ranked documents matter more, "the importance of lower-ranked documents does not vanish as it would were, say, an exponential function used. The constant k mitigates the impact of high rankings by outlier systems."
- Pilot (Table 1): MAP for RR Fusion of 30 model systems on TREC topics 351-400: k=0 .2072, k=10 .2123, k=20 .2134, k=30 .2139, k=40 .2138, k=50 .2144, k=60 .2145, k=70 .2146, k=80 .2147, k=90 .2145, k=100 .2142, k=500 .2098. Text: "indicated that k = 60 was near-optimal, but that the choice was not critical". Best individual .2016, Condorcet .2074, CombMNZ .2039.
- Significance method in that paper: "a simple sign test" over experiments (RRF beat Condorcet 7 of 7, p <= 0.008).
- Caveat for this project `[my inference]`: the pilot fused 30 configurations of one engine, and the TREC experiments fused many submitted runs. Nothing there is about fusing two heterogeneous retrievers (dense + sparse) over a 676-document corpus.

### 2.2 Bruch, Gai, Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval"

Read: arXiv v2 (4 May 2023), https://arxiv.org/pdf/2210.11934v2. The arXiv page lists DOI 10.1145/3596512 as a related DOI; I did not confirm volume/year of the published ACM TOIS version.

- Abstract findings: RRF is sensitive to its parameters; learning a convex combination (CC) is generally agnostic to the score normalisation; CC outperforms RRF in-domain and out-of-domain; CC is sample efficient, "requiring only a small set of training examples to tune its only parameter".
- Setup: BM25 (k1=0.9, b=0.4, whitespace tokenisation + stemming, PISA) fused with `all-MiniLM-L6-v2` (384-d cosine); in-domain MS MARCO plus out-of-domain BEIR sets; metric NDCG@1000 (cut at 100 for SciFact/NFCorpus). RRF set to k=60 "per [5]" (the Chen et al. hybrid study); CC tuned on validation, `alpha = 0.8` ("TM2C2": theoretical-minimum min-max normalisation).
- Section 5.1: "NDCG swings wildly as a function of RRF parameters", especially when separate k values per branch are allowed; best direction flips between in-domain (lexical discounted) and out-of-domain (semantic discounted); tuned RRF "does not generalize well to out-of-domain datasets".
- Sample efficiency (Fig. 12 text): "with less than 5% of the training data, which is often a small set of queries, TM2C2's alpha converges"; introduction: "just a handful of labeled queries". **No absolute query count appears in the text I read.**
- Note the paper's datasets are large; its tuning sets are subsets of train splits of ≥ thousands of queries (5% steps). Whether "a handful" holds for 45 clusters is `[my inference]`: unproven.

### Implications (RRF)

- k=60 is defensible as the a-priori value (Cormack: near-optimal and not critical; Bruch Table 2 baseline). Do not tune it on the 90 questions; declare it before running.
- Because Bruch finds CC better and RRF parameter-sensitive, an optional pre-declared secondary arm is **CC with min-max** `alpha=0.8 (dense) / 0.2 (sparse)`, a-priori, or `0.7` (see EACL 2026 in section 4). Mark it exploratory. Do not pick alpha from the frozen set.
- With 30-deep lists and k=60, a doc present in both lists at any rank <= 30 outscores any doc present in one list only (arithmetic in Summary item 6). Expect hybrid to promote "agreement" chunks above a dense-top-1 that sparse misses. Add a diagnostic: count queries where dense top-1 is correct but hybrid top-1 is not, and inspect them.
- Unit test for RRF: hand-computed example (two lists, overlapping ids), ties broken deterministically, ranks 1-based, missing doc contributes 0.
- Fuse at **chunk level, then collapse to article** (best-chunk) for the article-level metrics; pre-declare this. Duplicate chunks of the same article in one branch inflate that article's score if you collapse before fusing.

---

## 3. bge-m3

### 3.1 How sparse lexical weights are computed

Paper: Chen et al., "M3-Embedding", arXiv 2402.03216v5, https://arxiv.org/pdf/2402.03216v5 (read).

- "For each term t within the query (a term is corresponding to a token in our work), the term weight is computed as `w_qt = Relu(W_lex^T H_q[i])`, where `W_lex in R^{d x 1}` is the matrix mapping the hidden state to a float number. If a term t appears multiple times in the query, we only retain its max weight." Same for passages. Score `s_lex = sum over t in q intersect p of (w_qt * w_pt)`.
- Dense: normalised `[CLS]` hidden state, inner product. Hybrid: `s_rank = w1*s_dense + w2*s_lex + w3*s_mul` (Eq. 1; Eq. 3 for the re-ranking form).
- Weights reported: MIRACL and MKQA "Dense+Sparse": `w1=1, w2=0.3, w3=0`; "All": `w1=1, w2=0.3, w3=1`. MLDR (long documents): Dense+Sparse `w1=0.2, w2=0.8, w3=0`; All `w1=0.15, w2=0.5, w3=0.35`. Dense+Sparse re-ranks the union of top-1000 dense and top-1000 sparse candidates in MIRACL by this **weighted score sum** (not by rank fusion); sparse is searched with Lucene. The BM25 baseline in MIRACL Table 1 uses the same XLM-R tokenizer as M3 "to make the BM25 and M3 more comparable", which is why Table 11 (Analyzer vs XLM-R tokens) matters when reading the BM25 row.
- The paper does not list which tokens are dropped. FlagEmbedding code does: `unused_tokens = [cls, eos, pad, unk]` zeroed, plus `w > 0` filter and max per token id: https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/finetune/embedder/encoder_only/m3/modeling.py (`_sparse_embedding`, lines 116-160: `torch.relu(self.sparse_linear(hidden_state))`, `scatter_reduce(..., reduce="amax")`, `sparse_embedding[:, unused_tokens] *= 0.`) and the inference class https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py (`_process_token_weights`, lines 354-367; `compute_lexical_matching_score` sums `w1[t]*w2[t]` over shared token ids).
- Tokens are XLM-RoBERTa SentencePiece tokens (paper Table 11 uses "Tokenizer XLM-R" for M3 sparse), not Vietnamese syllables or words.
- HF README example weights (not a recommendation for Vietnamese): `weights_for_different_modes=[0.4, 0.2, 0.4]` for dense/sparse/colbert in `compute_score`: https://huggingface.co/BAAI/bge-m3/blob/main/README.md.

### 3.2 Vietnamese evidence in the paper

- MIRACL (monolingual, 18 languages, nDCG@10): ar bn en es fa fi fr hi id ja ko ru sw te th zh de yo. **Vietnamese is not in this list** (Table 1 header). Averages: BM25 31.9, Dense 69.2, Sparse 53.9, Multi-vec 70.5, Dense+Sparse 70.4, All 71.5.
- MKQA (Table 2, Recall@100; queries in 25 non-English languages, passages from English Wikipedia): Vietnamese row: BM25 46.6, Dense 76.6, Sparse 51.8, Multi-vec 76.7, Dense+Sparse 76.8, All 76.9. The paper itself explains weak sparse results: "only very limited co-existed terms for cross-lingual retrieval as the query and passage are presented in different languages". Recall@20 (Table 13): vi BM25 33.6, Dense 69.6, Sparse 42.2, Dense+Sparse 70.2, All 70.9.
- Appendix C.2 / Table 11 (tokenizer for BM25): BM25 with Lucene "Analyzer" 38.5 MIRACL / 40.9 MKQA / 64.1 MLDR; BM25 on XLM-R tokens 31.9 / 39.9 / 53.6; M3 sparse 53.9 / 45.3 / 62.2. The paper's own text: "BM25 remains a highly competitive baseline"; M3 sparse does not beat BM25 on MLDR.
- Conclusion: no published number for bge-m3 sparse or dense+sparse on monolingual Vietnamese retrieval in this paper.

### 3.3 Files in the HF repo; sentence-transformers

- HF repo `BAAI/bge-m3` (API snapshot: sha `5617a9f61b028005a4858fdac845db406aefb181`, lastModified 2024-07-03) contains `sparse_linear.pt` (listed as 3.52 kB), `colbert_linear.pt` (2.1 MB), `pytorch_model.bin` (2.27 GB), no `model.safetensors` in the listing I read. https://huggingface.co/BAAI/bge-m3/tree/main and https://huggingface.co/api/models/BAAI/bge-m3.
- `modules.json` = `sentence_transformers.models.Transformer` (idx 0), `Pooling` (1_Pooling, CLS pooling, dim 1024), `Normalize` (2_Normalize). No sparse module. https://huggingface.co/BAAI/bge-m3/raw/main/modules.json. README: "You also can use sentence-transformers and huggingface transformers to generate dense embeddings"; sparse and ColBERT outputs are shown only with `FlagEmbedding.BGEM3FlagModel` (`return_sparse=True`).
- sentence-transformers' own sparse class (`SparseEncoder`) is documented around SPLADE-style models (SPLADE pooling, SparseAutoEncoder, SparseStaticEmbedding); the docs page I read does not mention bge-m3: https://sbert.net/docs/sparse_encoder/usage/usage.html.
- `SentenceTransformer.encode` accepts `output_value="token_embeddings"` in current main (https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/model.py line 354). So you could load `sparse_linear.pt` yourself and apply `relu(linear(token_embeddings))` `[my inference]`; it would need to mimic the special-token and max-aggregation rules and has no official support. It is a pickle file (the Hub flags `.pt` and `.bin` as pickle): load with `torch.load(..., weights_only=True)`.

### Implications (bge-m3)

- Option A (BM25, client-side) is the lower-risk first arm: the repo's current dependency stack already supports it and the BM25 vs XLM-R-token result above shows that tokenisation, which you control, matters.
- Option B (M3 sparse) adds a `FlagEmbedding` dependency (or hand-rolled head), a second forward pass or a re-implementation, and has no Vietnamese-monolingual evidence. If tried, treat it as a pre-declared exploratory arm. Weighted sum with `w2 = 0.3` is the paper's choice for MIRACL; do not tune.
- If both arms run, count them as separate hypotheses in the multiple-testing budget (section 5/6).
- Acceptance check for any bge-m3 sparse code: reproduce the README example weights on the README's two English sentences (e.g. `'BM': 0.2515, '25': 0.3335`) within float tolerance before applying it to Vietnamese. (Values copied from the README; I did not run the model.)

---

## 4. BM25 for Vietnamese

### 4.1 Formula and parameters (Robertson and Zaragoza 2009)

Read: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf (author's copy of the Foundations and Trends monograph; the PDF is 59 pages, `inr-019.dvi`).

- Length normalisation `B = (1 - b) + b * dl/avdl`, `0 <= b <= 1`; term weight `w_BM25 = tf/(k1*B + tf) * w_RSJ`, equivalently `tf / (k1*((1 - b) + b*dl/avdl) + tf)` times an IDF-like weight (Eq. 3.13-3.15); without relevance information `w_RSJ` reduces to `log((N - n_i + 0.5)/(n_i + 0.5))` (Eq. 3.3).
- Parameters: "values such as 0.5 < b < 0.8 and 1.2 < k1 < 2 are reasonably good in many circumstances. However, there is also evidence that optimal values do depend on other factors (such as the type of documents or queries)." "A common combination would be b = 0.5 and k1 = 2. (However, many experiments suggest a somewhat lower value of k1 and a somewhat higher value of b.)" The `(k1 + 1)` numerator factor is rank-neutral.
- Other values reported in sources I read: fastembed/Qdrant defaults k=1.2, b=0.75; Bruch et al. k1=0.9, b=0.4; EACL 2026 Vietnamese study k1=1.5, b=0.75 (below).

### 4.2 Evidence on tokenisation for Vietnamese retrieval

| Source | Unit used | What was measured | Read? |
|---|---|---|---|
| Ho Bao Quoc, EVIA 2007, https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings6/EVIA/17.pdf | syllable unigram; syllable bigram; unigram + dictionary compounds; dictionary segmentation (about 30,000 entries) | SMART `ltc` weighting (not BM25); 11-pt average precision: 0.3636 (syllable unigram), 0.3778 (bigram), 0.4989 (unigram + dictionary compounds), 0.5625 (dictionary segmentation). Collections VN1 (11,398 docs) and VN2 (25,215 docs), 14 topics; the text does not say which collection produced the numbers. Bigram "lost of precision for high recall". | yes |
| Pham et al., Multi-stage IR for Vietnamese Legal Texts, https://arxiv.org/pdf/2209.14494 | BM25+ (`rank_bm25`) with VnCoreNLP word segmentation and stopword removal | Dense models trained at syllable vs word level, not BM25. BM25+ Recall@20 0.557, F2 0.221; sentence-transformer word-level models about 0.92-0.96 Recall@20. Authors: word segmentation helps PhoBERT-style models, syllable training helps ViBERT-style. No BM25 syllable-vs-word ablation. | yes |
| Miko team, ALQAC 2022, https://arxiv.org/pdf/2211.02200 | BM25 with pyvi word segmentation, lowercasing, stopwords; top-150 candidates | No ablation on tokenisation seen in what I read. | partly (grep) |
| NOWJ, ALQAC 2023, https://arxiv.org/pdf/2309.09070 | RDRSegmenter word segmentation before BM25 | They state segmenting compounds is intended to help the statistical models; no BM25 ablation seen. | partly (grep) |
| Nguyen and Quan, EACL 2026 Findings, "Which Works Best for Vietnamese?", https://aclanthology.org/2026.findings-eacl.110.pdf | BM25 "standard inverted-index configurations"; Underthesea segmenter is stated only for dataset statistics | ALQAC (BM25: P@1 89.25, R@10 97.92, MRR@10 92.20) and ZaloLegalQA (BM25 P@1 71.40, MRR@10 79.39) vs dense; BM25 k1=1.5, b=0.75, linear fusion alpha=0.7, RRF constant 60 (Table 5). Their summary: "linear BM25-based alpha-fusion consistently outperforming" standalone and rank-based fusion; "optimal alpha values typically lie between 0.6 and 0.8"; legal corpora "approaching saturation under strong hybrid". Tokenisation for BM25 retrieval not stated. | yes (tables partly garbled by extraction; I quote only unambiguous cells) |
| Son et al. (BKAI), https://arxiv.org/pdf/2412.00657 | BM25 baseline, tokenisation not stated | Zalo QA 2019 (Wikipedia QA, out-of-domain): BM25 MRR@10 45.33 vs BGE-M3 76.69 (Table 4). | yes |

- Syllable bigrams: only the 2007 EVIA paper above (bigram AP 0.3778 vs 0.3636 unigram, but far below dictionary word segmentation 0.5625). A practitioner repo using syllable bigrams for BM25 appeared in a search summary only (not opened): see Unverified.
- Everything about "which tokenisation is best for Vietnamese BM25 in modern retrieval" is therefore **unresolved by primary evidence I could read**. The strongest hints are (a) the 2007 gap between syllable and dictionary-segmented indexing, and (b) M3's finding that tokenisation changes BM25 by several nDCG points.

### Implications (BM25 for Vietnamese)

- Pre-declare **one primary BM25 tokenisation**. A defensible a-priori default is lowercase + NFC + strip punctuation + **syllable unigrams** (whitespace split), because (i) it is the simplest, (ii) it needs no segmenter dependency or model download, and (iii) legal queries share exact surface syllables with the law. Plus one **declared secondary arm**: syllable unigram + bigram, or word segmentation (pyvi / underthesea / VnCoreNLP). Do not run more than the declared arms on the frozen set. This is my recommendation; the literature does not decide it.
- Hyper-parameters: fixed a-priori k1=1.2, b=0.75 (Qdrant/fastembed defaults; inside R&Z's "reasonable" ranges). Optionally state that 1.5/0.75 (EACL 2026) is within the same range, but do not scan.
- Syllable unigrams make stopword-like tokens ("của", "và", "điều", "khoản") very frequent; IDF handles this. Check that "điều" and "khoản" do not dominate article_ref queries: compute the IDF of `điều` and `24`-like numeric tokens and record them.
- NFC-normalise both sides (project rule). Decide on diacritics up front: keep them (the corpus has them). Missing-diacritic user queries are an explicit failure case in EACL 2026 (they note informal queries without diacritics); out of scope for this frozen set but worth a note.
- Acceptance checks: (1) sparse-only recall@5 reported separately (expected to be much lower than dense on paraphrase; not a target); (2) for 5 direct questions, show the top sparse terms and their IDF; (3) the BM25 index uses the same `context_text` as the dense embedding (or declare the difference).

---

## 5. Significance testing for a small, clustered test collection

### 5.1 What the sources say

- **Smucker, Allan, Carterette, CIKM 2007** (read, https://ciir-publications.cs.umass.edu/getpdf.php?id=744): compared paired t, Wilcoxon, sign, bootstrap and randomization tests on MAP over TREC 3 and 5-8 run pairs. "Little practical difference between the randomization, bootstrap, and t tests"; Wilcoxon and sign "have a poor ability to detect significance and have the potential to lead to false detections of significance"; they "should be discontinued for measuring the significance of a difference between means"; recommend the randomization test "as the preferred test in all cases for which it is applicable" because it can use any statistic. Their example uses 100,000 permutations.
- **Urbano, Lima, Hanjalic, SIGIR 2019** (read, https://arxiv.org/pdf/1905.11096v2): simulation with known null (about 500 million p-values); measures AP, nDCG@20, ERR@20, P@10, RR; topic set sizes 25, 50, 100. t-test and permutation test keep Type I at alpha and behave almost identically; bootstrap-shift biased to small p-values; Wilcoxon and sign "should be discontinued for being unreliable"; Type III errors "for measures like P@10 and RR, or small topic sets, ... can be as high as 2%". They keep the permutation test as useful "because it can accommodate other test statistics besides the mean". **Binary R@1 was not one of the studied measures.**
- **Voorhees and Buckley, SIGIR 2002**: I could only read the NIST abstract record (https://www.nist.gov/node/707561): error rates computed directly for topic sets up to 25 and extrapolated; they are "larger than anticipated", so be careful declaring one method better with few topics. I did not read the paper's tables, so I quote no error-rate numbers.
- **Exact McNemar / sign on discordant pairs:** The McNemar test is the sign test on discordant pairs; the "exact conditional" version is conservative, and the mid-p version is recommended by Fagerland, Lydersen, Laake, BMC Med Res Methodol 2013, 13:91: "The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional" (title and abstract via search summary; full text not read; PMC link https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3716987/).
- **Related queries / clusters:** Miller, "Adding Error Bars to Evals" (arXiv 2411.00640, read): when questions are drawn in related groups, assuming independence gives inconsistent standard errors; use clustered standard errors, which "can be over 3X larger than naive" on real evals (their Table 4); analyse **paired question-level differences** when comparing two systems; report the cluster count with the question count; includes cluster-adjusted power formulas (Appendix C). Source: https://arxiv.org/pdf/2411.00640.
- **MRR scale and averaging:** Fuhr, SIGIR Forum 51(3), 2017, "Some Common Mistakes In IR Evaluation" (read, https://sigir.org/wp-content/uploads/2018/01/p032.pdf): RR is ordinal, "one cannot compute the mean for an ordinal scale"; proposes mean first relevant rank (MFR) as a ratio-scale alternative. Sakai disputes the strictness (secondary summary only; not opened). Ferrante, Ferro, Fuhr (arXiv 2101.02668, read in part) discuss mapping measures to interval scales.
- **Multiple testing:** Fuhr section 2.7: correct for several hypotheses (Bonferroni is "a simple (conservative) method"), and "When test collections are reused, chances increase that we observe random results"; one should count tests already run on the collection.

### 5.2 Arithmetic: how big a discordance is needed for R@1 `[my arithmetic]`

Exact two-sided sign/McNemar p-values (b = system B wins, c = A wins, discordant pairs only), computed with `math.comb`:

| b : c | exact two-sided p | mid-p two-sided |
|---|---|---|
| 5 : 0 | 0.0625 | 0.0312 |
| 6 : 0 | 0.0312 | 0.0156 |
| 7 : 0 | 0.0156 | 0.0078 |
| 7 : 1 | 0.0703 | 0.0391 |
| 8 : 1 | 0.0391 | 0.0215 |
| 8 : 2 | 0.1094 | 0.0654 |
| 9 : 2 | 0.0654 | 0.0386 |
| 10 : 3 | 0.0923 | 0.0574 |
| 12 : 4 | 0.0768 | 0.0490 |

So on the paraphrase stratum (45 questions, dense R@1 = 32/45), a hybrid that fixes 10 and breaks 3 is *not* significant by the exact test (p = 0.092). The 12 article_ref questions can show at most 3 fixes (9/12 baseline), so p can never fall below 0.25 there (3:0 exact two-sided p = 0.25).

### 5.3 Recommended tests for n = 90 (45 clusters)

Unit of analysis: the 45 gold clusters (each: one direct and one paraphrase question sharing gold). The 12 article_ref and 25 out-of-scope questions are analysed in their own strata (no cluster pairing unless several article_ref questions share an article).

1. **R@1 (binary), per stratum** (direct n=45, paraphrase n=45, article_ref n=12): exact McNemar (two-sided exact binomial on discordant pairs), report mid-p alongside. Treat as descriptive for article_ref (n=12).
2. **R@1, pooled direct + paraphrase (90 queries):** cluster-level paired permutation test: for each cluster compute `d_c = mean over its two queries of (R@1_B - R@1_A)`, in {-1, -0.5, 0, 0.5, 1}; under H0 each non-zero `d_c` flips sign with probability 1/2; the test statistic is the sum of `d_c`; enumerate exactly if the number of non-zero clusters is <= 20, else 100,000 random sign flips. This is Smucker's randomization test with clusters as "topics"; it is my synthesis of Smucker + Miller, not a published recipe for this exact case.
3. **MRR@10 (article-level):** the same cluster-level sign-flip permutation test on per-cluster mean RR differences, plus a **cluster bootstrap** 95% CI of the paired difference (10,000 resamples of the 45 clusters with replacement). Report effect size (difference in MRR and in R@1 counts), not only p.
4. **Second rank-based metric:** mean first relevant rank (article level, cap at 11 for "not in top 10"; cap value is a pre-declared choice, `[my inference]`) because MRR is ordinal; run the same permutation test.
5. **Multiple comparisons:** pre-declare the confirmatory contrasts (suggest 3: hybrid vs dense on paraphrase R@1; hybrid vs dense on pooled MRR@10; hybrid+filter vs hybrid on article_ref R@1). Everything else (sparse vs dense, secondary tokenisation arms, CC arm, M3 sparse arm) is exploratory. Apply Holm or Bonferroni to the confirmatory family and count prior runs on this frozen set (the dense baseline was already looked at) in the denominator `[my inference from Fuhr 2.7]`.
6. **Power note:** a +0.05 MRR change on 45 clusters is unlikely to be detectable. Miller's paper gives cluster-adjusted power formulas; to be honest about it, report the width of the cluster-bootstrap CI next to every comparison.

### Implications (significance)

- Report CI and effect size per stratum; avoid pooling the lexically easy "direct" half with the discriminating "paraphrase" half in the single headline.
- Do not use t-test on per-query binary scores of correlated pairs (naive SE too small, up to 3x per Miller). Do not use Wilcoxon (ties dominate; Smucker/Urbano).
- Implement the test code once with unit tests (known p-values from the table above; a permutation test on a toy example with a hand-enumerated null).
- Seed all random procedures; record the seed in `eval/results.md`.

---

## 6. Tuning on the evaluation set

### 6.1 What primary sources say

- Fuhr (2017), section 2.5-2.7 (read): "cross-validation ... should be a minimum requirement"; when a tuning set is needed ("determining the best parameter setting, or choosing the best one from a variety of methods") it "should be disjoint from the test set"; "the complete set of hypotheses to be tested must be known before the experiment"; "it would be incorrect if an author first determines which of the competing methods performs best on his data set, and then compares his own approach only to this one"; collection reuse counts as multiple hypothesis testing.
- Cormack et al. (2009): RRF's k=60 "was fixed during a pilot investigation and not altered during subsequent validation".
- Bruch et al. (2023): tuned CC alpha on the validation split of in-domain datasets and evaluated on the test split; for RRF they used k=60 from earlier work; they tuned RRF parameters on in-domain validation and then showed it does not transfer out-of-domain. The sample-efficiency experiment tunes on growing *subsets of a train split* and evaluates on a held-out test split.
- Qdrant docs: "Hand-tuned weights without measurement are unlikely to beat the default reliably"; if you have an eval set, "tune on a train/validation split".
- ORQA / BEIR do not address tuning discipline directly; I found no IR-methodology paper in this session that gives a sample-size rule for tuning with only dozens of queries.

### Implications (tuning)

- **Fix before running, write in the plan:** BM25 k1=1.2, b=0.75; tokenisation arm(s) as declared; RRF k=60 with equal weights; list depth 30 per branch; no per-branch k, no weights; any CC arm uses alpha fixed a priori (0.7 or 0.8 from the literature), never chosen from the frozen set.
- If you do want to learn anything (e.g. alpha or tokenisation choice), make it a **separate, pre-declared k-fold cross-validation over the 45 clusters** (e.g. 5 folds of 9; grid of at most 3-5 values listed in advance; pick on the training folds, report the held-out folds) and mark the result as exploratory. With 45 clusters the CV estimate will be very noisy; present it as a sensitivity check, not a selection procedure.
- Anything that was inspected on the frozen set (including failure analysis of individual queries that then leads to a code change, such as changing the regex because it missed an article_ref question) makes the set partially "used"; state that in `eval/results.md` and, if possible, add a small fresh holdout (e.g. 15-20 new clusters written *after* the design is frozen) for the final claim. This is my recommendation; the sources only support the principle.
- Record in `eval/results.md`: the arm list, the number of confirmatory contrasts, the correction method, and the seeds.

---

## 7. Lexical bias of evaluation sets

- **BEIR** (Thakur et al., NeurIPS 2021 Datasets and Benchmarks Track; arXiv 2104.08663v4; read): "we notice that there can be a strong lexical bias present in datasets included within the benchmark, likely as lexical models are pre-dominantly used during the annotation or creation of datasets. This can give an unfair disadvantage to non-lexical approaches." Section 6 (annotation selection bias): "Many BEIR datasets are found to be subject to a lexical bias, i.e. a lexical based retrieval system like TF-IDF or BM25 has been used to retrieve the candidates for annotation"; they quantify unjudged hits with **Hole@10** (top-10 results not seen by annotators) on TREC-COVID: BM25 6.4%, docT5query 2.8% (dense systems higher); adding the missing judgements raised ANCE from 0.65 (rest of the number not extracted) while lexical systems "improve only slightly". https://arxiv.org/pdf/2104.08663v4. This bias is *candidate-pool* bias, which is different from this project's mechanism; the effect direction (favouring lexical matching) is the same.
- **Question-writing bias:** Lee, Chang, Toutanova (ORQA, ACL 2019; arXiv 1906.00300; read): in SQuAD "the question askers are also prompted with a specific piece of evidence for the answer, leading to artificially large lexical overlap between the question and evidence"; question writing "is not motivated by an information need" and has "many hints in the question". https://arxiv.org/pdf/1906.00300. This is the closest documented analogue to LLM-drafted questions from verbatim chunks.
- **How papers quantify overlap:** BEIR uses Hole@10. I found, via search summary only (not opened), a QA-bias paper that defines a lexical-overlap coefficient and buckets it High (> 0.85), Medium (0.4 to 0.85), Low (<= 0.4) against answer-bearing sentences (arXiv 2109.11256). Nothing standard for retrieval test sets.

### Implications (lexical bias)

- **Quantify it yourself, cheaply, before reading any hybrid results** `[my proposal, not from a source]`: for each question compute (a) fraction of question syllable tokens (stopwords removed) that occur in the gold chunk's `context_text`, (b) the BM25 rank/score of the gold chunk, and report the distribution for direct vs paraphrase vs article_ref. If direct questions have much higher overlap than paraphrase, the pooled 90-query MRR@10 is dominated by an easy half.
- Make the **paraphrase stratum** the headline for dense vs hybrid; keep direct as a regression check; article_ref is the filter test.
- Expect sparse to look better on direct than on paraphrase; if sparse-only matches dense on direct but collapses on paraphrase, that is the bias showing, not evidence that BM25 is as good as dense.
- Out-of-scope set: current "top-1 cosine above lowest in-scope top-1" finding (12/25) stays a separate problem. RRF scores are rank-based and **cannot** express confidence; do not use fused scores as an abstention signal. If an abstention threshold is wanted, it has to come from the dense cosine (or a calibrated score), and that is a separate, separately-tuned task.

---

## 8. Explicit article references ("Điều N") in queries

- **Nothing found** that studies parsing explicit "Article N" references *in the user's question* and then filtering or boosting by them in a statute-retrieval benchmark, with a measured effect. The search returned tangential items only.
- Closest primary text I read: Fink, Recski, Kusa, Hanbury, "Statute-enhanced lexical retrieval of court cases for COLIEE 2022" (arXiv 2304.08188): statute sections are extracted from *case documents* by pattern, indexed as a separate statute field in Elasticsearch, scored with BM25 and added to the passage score; they report improvement over plain passage-level retrieval. This is reference extraction from documents for boosting, not query-side filtering, and is on Canadian court cases. https://arxiv.org/pdf/2304.08188. Their paper also finds passage-level retrieval with rank fusion beats document-level (abstract).
- Other items seen only in a search summary (not opened, so not relied on): a Lexis patent describing numeral and section-symbol patterns for statute citations; Korean and Italian dataset papers that treat reference spans as an extraction step; a graph-based reference-network approach to statute retrieval.

### Implications (article filter)

All of this is design reasoning `[my inference]`, not literature:

- The filter is **precision-oriented** and can only help article_ref questions. Its failure modes are in the *parser*, not in Qdrant: "Điều 24" vs "Điều 24 Bộ luật Dân sự" (other statute), a question that names an article number in passing ("sau 24 tháng"), "Điều 3 và Điều 5" (use `MatchAny`), numbers not in 1..220, "khoản 2 Điều 24" (clause is not filtered unless the chunk has a clause field), typos ("Dieu 24", "điều 24" lowercase, NBSP). The regex must be unit-tested on these.
- Frozen-set caveat: 12 article_ref questions with dense R@1 = 9/12. A hard filter will score about 12/12 by construction, which measures the regex, not retrieval. Add parser-level negative tests (the 25 out-of-scope questions plus 20 or so non-article questions that contain numbers) and require **zero false activations**; also a test that out-of-range N falls back to unfiltered search instead of returning an empty list.
- Soft alternative to pre-declare as exploratory: boost rather than filter (add the filtered branch as a third RRF input). It degrades gracefully if the regex is wrong.
- Both the `context_text` heading "Điều N." and the numeric token `N` are in the BM25 index, so sparse already gives partial credit to article_ref queries; report sparse-only on the 12 article_ref questions to see how much the filter adds beyond it.

---

## Unverified / could not access

- **Qdrant docs:** the IDF formula is not on any docs page I read (vectors, collections, sparse-vectors article); it is taken from v1.20.1 source. The docs "search" page and the HNSW/optimizer sections did not load via the fetch tool (it returned a "URL only" error), so full-scan behaviour is from source, not docs prose. Whether the filtering docs say anything about propagating a top-level filter into prefetches was not confirmed (page text truncated); the propagation claim is from `planned_query.rs`.
- **Qdrant:** whether a collection may mix the existing unnamed dense vector with a named sparse vector; behaviour of the new server-side `TextQuery` BM25 on Vietnamese text (no docs page read; qdrant-client 1.20.0 has the models, source has the schema).
- **fastembed:** I only read the beginning of `supported_languages` (up to "spanish"); I did not confirm that Vietnamese is absent or what its tokenizer does with Vietnamese.
- **Bruch et al.:** the ACM TOIS published version (volume, year, differences from arXiv v2) was not accessed; no absolute number of tuning queries appears in the arXiv text I read.
- **Cormack et al.:** read in full (2 pages); no issue.
- **bge-m3:** the paper does not list the dropped tokens (taken from FlagEmbedding code); sparse head size taken from the HF file listing (3.52 kB); I did not download or load `sparse_linear.pt`, did not run the model, and did not verify its tensor shape. No monolingual Vietnamese benchmark for M3 sparse found.
- **BM25 and Vietnamese:** no controlled syllable-vs-word BM25 comparison on Vietnamese legal retrieval found. Not opened: the Zalo AI Challenge 2021 legal retrieval official report and top-team write-ups; the ALQAC overview papers; a UIT 2026 competition repo that a search summary said uses syllable bigrams for BM25 (numbers 85.8 R@5 / 70.3 MRR were in a search summary only; I do not rely on them); Hackett and Oard's Tibetan syllable n-gram vs word study (search summary only, not Vietnamese). The Miko and NOWJ papers were only grepped for segmentation/BM25 mentions, not read in full. The EACL 2026 table cells I used were unambiguous after extraction, but the table text is partly garbled; check the PDF before quoting more cells.
- **Robertson and Zaragoza:** the PDF is the author's hosted copy; I read sections 3.4-3.5.1 and Eq. 3.2-3.3, not the full monograph.
- **Significance testing:** Voorhees and Buckley 2002 full text not read (only the NIST abstract record); Fagerland et al. 2013 full text not read (abstract via search summary); Sakai's rebuttal to Fuhr not read; Ferrante et al. only skimmed. The cluster-level permutation test and the mean-first-relevant-rank cap are my synthesis, not from a paper on binary cluster-paired IR data. No source gave a validated procedure for clusters of exactly two paired phrasings.
- **Tuning discipline:** no IR-methodology paper found that gives a sample-size rule for tuning with about 45 clusters; Hastie et al. (Elements of Statistical Learning section 7.10) is the usual reference for "wrong and right way to do cross-validation" but was not opened.
- **Explicit citation parsing (section 8):** nothing directly relevant found; the other works named there were not opened.
- **Qdrant benchmark claims:** I made no latency measurements; "plain exact scan" is derived from thresholds and code, not observed on the live collection. Verify with `get_collection` and a `exact=True` vs default result comparison.
