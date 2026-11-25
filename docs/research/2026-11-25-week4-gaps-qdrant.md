# Week 4 gaps: Qdrant 1.20 full-text / BM25 / sparse / client API

Research date: 2026-11-25. Scope: the five questions in the brief. No Qdrant server was contacted. No repository file was changed except this one.

## Summary

- Q1: Two thresholds are documented. `indexing_threshold` (optimizer, kilobytes) decides when a segment gets an HNSW index. `full_scan_threshold` (HNSW config) decides when the query planner prefers a full scan.
- Q1: The docs have no prose for sparse `full_scan_threshold`. Its only description is the Python model docstring, which says "number of vectors, not KiloBytes".
- Q1: The docs YAML sets the key `full_scan_threshold` but its comment says `full_scan_threshold_kb`. Rust accepts both names through a serde alias.
- Q2: Server-side BM25 exists as the `TextQuery` query type (`scoring: {bm25: {k, b}}`) over a full-text index created with `scoring: bm25`. The docs tag it `version="1.20.0"`, and the v1.20.0 release notes list it.
- Q2: Full-text tokenizers are `word` (default), `whitespace`, `prefix`, `multilingual` and `code`. Options include `lowercase` (default true), `ascii_folding` (default false), `stopwords`, `stemmer`, `phrase_matching` (default false), and token length limits.
- Q2: `ascii_folding` does fold Vietnamese precomposed letters, including `đ` to `d`. Script check: 152 of 152 precomposed Vietnamese letters are covered. It does not handle combining marks U+0300–U+036F, so NFD text is not folded.
- Q2: Vietnamese is in none of the Snowball stemmer, stopword, or `Language` lists. The `multilingual` tokenizer's charabia dependency is built with `default-features = false`, so charabia's Vietnamese normalizer is not compiled in. There is no Vietnamese segmenter.
- Q3: fastembed `Bm25` `supported_languages` has 18 entries, with no Vietnamese. Defaults: `k=1.2`, `b=0.75`, `avg_len=256.0`, `language="english"`, `token_max_length=40`. Token id = `abs(mmh3.hash(token))`.
- Q3: fastembed's tokenizer splits NFD Vietnamese into fragments. Tested locally: NFC "Việt Nam hợp đồng" gives 4 tokens, NFD gives 7. Text must be NFC before BM25 embedding.
- Q4: All requested classes, fields, defaults and client signatures were read from the local qdrant-client 1.20.0 package. Collection-level `metadata` is accepted by `create_collection` and `update_collection`.
- Q5: Collection metadata exists since Qdrant v1.16.0 (docs and changelog PR #7123). `update` merges keys, and a `null` value removes a key.

---

## Q1. HNSW build and full-scan thresholds (docs prose)

Source of prose: `landing_page` repo, `qdrant-landing/content/documentation/`, fetched from the `master` branch raw Markdown on 2026-11-25. Not pinned to a commit. Local copies: `scratchpad/docs/`.

**Index building (`indexing_threshold`)**

- `ops-optimization/optimizer.md:28`: "The indexing optimizer builds a vector index for a segment once its vector data exceeds the indexing threshold. New points go to an appendable segment, which has no vector index. While a segment is small, a full scan is fast, and building an index would cost more than it saves. As the segment grows, scans get slower and an index starts to pay off."
- `ops-optimization/optimizer.md:40`: "The `indexing_threshold` parameter sets the indexing threshold, in kilobytes."
- `ops-optimization/optimizer.md:159, 164`: "# Maximum size (in KiloBytes) of vectors allowed for plain index." / "indexing_threshold_kb: 10000"
- `manage-data/collections.md:349-352`: "A new index segment is built if the size of non-indexed vectors is higher than the value of `indexing_threshold`(in kB). If your collection is very small or the dimensionality of the vectors is low, there might be no HNSW segment created and `indexed_vectors_count` might be equal to `0`."

**Full-scan decision (`full_scan_threshold`, HNSW)**

- `manage-data/indexing.md:369-375`: "# Minimal size threshold (in KiloBytes) below which full-scan is preferred over HNSW search." / "# `full_scan_threshold_kb`, the query planner will use full-scan search instead of HNSW index" / "# traversal for better performance." / "# Note: 1Kb = 1 vector of size 256" / "full_scan_threshold: 10000"
- Inconsistency: the key is `full_scan_threshold` but the comment names `full_scan_threshold_kb`. Rust side: `lib/segment/src/types.rs:858-862` documents `full_scan_threshold_kb` and has `#[serde(alias = "full_scan_threshold_kb")] pub full_scan_threshold: usize` (v1.20.1). [Source: raw.githubusercontent.com/qdrant/qdrant/v1.20.1/lib/segment/src/types.rs]
- `search/search.md:550` (query planning): "* prefer a full scan if the amount of points is below a threshold"

**Sparse vectors**

- `manage-data/indexing.md:450`: "A sparse vector index in Qdrant is exact, meaning it does not use any approximation algorithms."
- `manage-data/indexing.md:453`: "With Qdrant, you can benefit from a more compact and efficient immutable sparse index, which is constructed during the same optimization process as the dense vector index."
- The docs have no prose for a sparse `full_scan_threshold`. The only definition is the Python model docstring, `qdrant_client/http/models/models.py:3469-3478` (`SparseIndexParams`): "We prefer a full scan search upto (excluding) this number of vectors. Note: this is number of vectors, not KiloBytes."
- Not located: the Rust code that applies the sparse `full_scan_threshold`. I searched `lib/segment/src/types.rs` only. [inference: it lives in another crate.]

---

## Q2. Server-side BM25 and full-text scoring (Qdrant 1.20.x)

**Documented query type and version**

- `manage-data/indexing.md:226`: "### BM25 Scoring {version="1.20.0"}"
- `manage-data/indexing.md:228`: "With scoring enabled, a full-text index can rank points by how relevant their text is to a query, using the BM25 ranking function."
- `search/text-search/full-text-search.md:35`: "A point matches the query if its text contains at least one of the query terms. Points that contain none of the terms are not returned."
- `full-text-search.md:43-44`: "`k`. Controls term frequency saturation. ... Defaults to 1.2." / "`b`. Controls document length normalization. ... Defaults to 0.75."
- `full-text-search.md:88`: "The full-text index computes the average document length from your data, so you no longer need to set `avg_len`."
- Rust, `lib/api/src/rest/schema.rs` at v1.20.1: line 721: "Rank by BM25 over the text index of the payload field named by `using`, which must have `scoring` set." `pub struct TextQuery` at line 726. `TextQueryInput` at 751 (`scoring: Option<TextQueryScoring>`). `Bm25Params` at 786 with `k` ("Term frequency saturation. Default is 1.2.") and `b` ("Document length normalization, from 0 (none) to 1 (full). Default is 0.75.").
- Python model: `qdrant_client/http/models/models.py:3819` `class TextQuery` (`text: TextInterface`), `:176` `class Bm25Params` (`k`, `b`, both default None), `:183` `Bm25Scoring`, `:4430` `Query` union (includes `TextQuery`).
- Release notes, v1.20.0 (`https://api.github.com/repos/qdrant/qdrant/releases/tags/v1.20.0`): "BM25 scoring directly on the full-text payload index [[docs]...]" (issue #10750).
- Not documented: the note `full-text-search.md:51` says "The term frequency parameter is named `k`, not `k1`."

**Full-text index: tokenizers** (`manage-data/indexing.md`)

- Line 248-258: `word` is the default. `whitespace`: "splits the string on whitespace ... Punctuation and special characters stay part of the tokens."
- Line 260-262 (`multilingual`): "based on multiple packages like charabia and vaporetto ..."; "See the charabia documentation for a full list of supported languages".
- Line 266 (`code`, `{version="1.20.0"}`).
- `prefix`: docs line 258.
- Rust enum `TokenizerType` (`lib/segment/src/data_types/index.rs:308-314`): Prefix, Whitespace, Word (default), Multilingual, Code. Python: `models.py:3860-3866` (`PREFIX, WHITESPACE, WORD, MULTILINGUAL, CODE`).

**Full-text index: options** (Rust `schema.rs` lines 230-262 and Python `models.py:3774-3817` `TextIndexParams`)

- `lowercase`: docs line 296: "Lowercasing is enabled by default." Rust schema.rs:237: "Lowercase the text before tokenization. Default is `true`."
- `ascii_folding`: docs line 302: "When enabled, ASCII folding converts Unicode characters into their corresponding ASCII equivalents, for example, by removing diacritics. For instance, the character `ã` is changed into `a`, `ç` becomes `c`, and `é` is converted to `e`." (`{version="1.16.0"}`, line 300). Rust schema.rs:241: "If true, normalize tokens by folding accented characters to ASCII (e.g., "ação" -> "acao"). Default is `false`." Python `models.py:3774+`: "If true, normalize tokens by folding accented characters to ASCII (e.g., 'ação' -> 'acao'). Default: false."
- `ascii_folding` table, `lib/segment/src/index/field_index/full_text_index/tokenizers/ascii_folding.rs`: lines 1-2 say the table is "inspired by Lucene's ASCIIFoldingFilter". Vietnamese precomposed letters are mapped (example lines 45-53 for `A/a`, line 88 for `đ`/`Đ` (U+0111 to `d`), lines 101-109 for `e`). Local check, `scratchpad/check_fold.py`, parsed 1241 match arms: 152 of 152 precomposed Vietnamese letters (U+00C0 to U+1EFF, with Vietnamese diacritics, plus U+0110/U+0111) are mapped. Examples: U+1EA1 `ạ` to `a`, U+1EBF `ế` to `e`, U+1EC7 `ệ` to `e`, U+0111 `đ` to `d`, U+01A1 `ơ` to `o`, U+01B0 `ư` to `u`.
- Combining marks U+0300–U+036F: zero entries in the table. [inference: NFD Vietnamese text (base letter + combining mark) is not folded. Text should be NFC.]
- Order of operations, `tokenizers/tokens_processor.rs:108-123`: truncate, ASCII folding, lowercase, stopwords, stemming, length limits.
- `stemmer`: docs line 315: "Qdrant provides an implementation of Snowball stemmer ... For the list of supported languages, please visit the rust-stemmers repository." Rust `SnowballLanguage` (`index.rs:495`): Arabic, Armenian, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, Turkish. No Vietnamese.
- `stopwords`: docs line 322-332. Rust `stop_words/` folder at v1.20.1 holds one `.rs` file per language; no Vietnamese file (`grep -ci vietnam stop_words/mod.rs` = 0). Rust `Language` enum (`index.rs:652`): no Vietnamese (grep = 0). Python `models.py:1749-1775`: no `VIETNAMESE`.
- `phrase_matching`: `models.py` docstring "If true, support phrase matching. Default: false." Docs line 340: "For efficient phrase matching, Qdrant requires building an additional data structure, so it needs to be configured during the creation of the full-text index".
- `min_token_len` / `max_token_len`: Rust schema.rs:256 and 260.

**Multilingual tokenizer and Vietnamese**

- Rust: `lib/segment/src/index/field_index/full_text_index/tokenizers/multilingual.rs:22-38`: "If the script of the input is latin and we don't need to stem early, tokenize as-is." Latin input goes to charabia; Japanese goes to vaporetto. `SUPPORTED_LANGUAGES` (lines 85-102) is used only for language detection and has no `Vie`.
- `lib/segment/Cargo.toml:108-115`: `charabia = { version = "0.10.0", default-features = false, features = ["greek", "hebrew", "thai", "chinese-segmentation", "chinese-normalization"] }`
- charabia v0.10.0 `charabia/Cargo.toml` `[features] default = [... "vietnamese", ...]`. Qdrant disables default features, so `vietnamese` is off.
- charabia v0.10.0 `charabia/src/normalizer/vietnamese.rs:9`: maps `Ð | Đ | đ | ð` to `d`. Compiled only with feature `vietnamese` (`normalizer/mod.rs:23, 45, 77`).
- charabia v0.10.0 `charabia/src/detection/script_language.rs:95, 326`: `Language::Vie` exists (detection only).
- charabia v0.10.0 segmenters (`charabia/src/segmenter/`): arabic, chinese, german, japanese, khmer, korean, latin, thai. No Vietnamese segmenter. `segmenter/latin/mod.rs`: the Latin segmenter returns the input unchanged unless feature `latin-camelcase` is on, which Qdrant does not enable.
- [inference] Vietnamese is whitespace-delimited, so the `multilingual` tokenizer produces one token per syllable. It does not segment compound words such as "hợp đồng". Not tested against the server.

---

## Q3. fastembed BM25 (`fastembed/sparse/bm25.py`, `main` branch, fetched 2026-11-25, not pinned)

**Languages**

- `bm25.py:35-52` `supported_languages`: arabic, danish, dutch, english, finnish, french, german, greek, hungarian, italian, norwegian, portuguese, romanian, russian, spanish, swedish, tamil, turkish (18 entries). No Vietnamese.
- `bm25.py:137-138`: `if not disable_stemmer and stemmer is None and language not in supported_languages: raise ValueError(f"{language} language is not supported")`
- Stopwords are not in the package. `bm25.py:193-201` loads `{language}.txt` from the model directory, downloaded from HF repo `Qdrant/bm25` (`bm25.py:66`, `additional_files`). Tamil has no stopword file (comment at `bm25.py:64`).
- Stemmer: `bm25.py:9` `from py_rust_stemmers import SnowballStemmer`; `bm25.py:179` `self.stemmer = SnowballStemmer(language)`.

**Defaults** (`bm25.py:118-126`)

- `k: float = 1.2`, `b: float = 0.75`, `avg_len: float = 256.0`, `language: str = "english"`, `token_max_length: int = 40`, `disable_stemmer: bool = False`.

**Tokenisation steps**

1. `raw_embed` (`bm25.py:309-320`) and `query_embed` (`bm25.py:365-382`) call `remove_non_alphanumeric(document)`. `fastembed/common/utils.py:156-157`: `return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)`.
2. `self.tokenizer = SimpleTokenizer` (`bm25.py:181`). `fastembed/sparse/utils/tokenizer.py`: `text = re.sub(r"[^\w]", " ", text.lower())`, then `re.sub(r"\s+", " ", text)`, then `text.strip().split()`.
3. `_stem` (`bm25.py:287-307`), per token: `lower_token = token.lower()`; skip if `token in self.punctuation` (punctuation = every code point with Unicode category `P`, `common/utils.py:150-153`); skip if `lower_token in self.stopwords`; skip if `len(token) > self.token_max_length`; stem with `stemmer.stem_word(lower_token)` if a stemmer is set, else keep `lower_token`; drop empty results.
4. Token id: `bm25.py:362-363` `return abs(mmh3.hash(token))`.
5. Document weight (`bm25.py:331-360`): `tf = count * (k + 1) / (count + k * (1 - b + b * doc_len / avg_len))`, where `doc_len = len(tokens)` after steps 1-3.
6. Query weight (`bm25.py:365-382`): unique token ids, each with value 1.0.

**Local check of step 2 with NFC vs NFD** (`scratchpad/nfd_test.py`, run with `.venv/Scripts/python.exe -I`):
- NFC "Việt Nam hợp đồng" gives 4 tokens: `việt`, `nam`, `hợp`, `đồng`.
- NFD of the same string gives 7 tokens: `vie`, `t`, `nam`, `ho`, `p`, `đo`, `ng`.
- So fastembed's BM25 breaks NFD Vietnamese words. Text must be normalised to NFC first.

---

## Q4. qdrant-client 1.20.0 Python API (local package `.venv/Lib/site-packages/qdrant_client/`)

Read from `http/models/models.py` and `qdrant_client.py` / `qdrant_remote.py` / `client_base.py`. Line numbers refer to `models.py` unless stated.

- `SparseVectorParams` (3567): `index: Optional[SparseIndexParams] = None` ("If none - values from collection configuration are used."), `modifier: Optional[Modifier] = None` ("Default: none").
- `SparseIndexParams` (3469): `full_scan_threshold: Optional[int] = None`, `on_disk: Optional[bool] = None` (deprecated), `memory: Optional[Memory] = None` ("Default: `pinned`"), `datatype: Optional[Datatype] = None`.
- `Modifier` (2032): `NONE = "none"`, `IDF = "idf"`.
- `Prefetch` (2484): `prefetch`, `query`, `using` (None; "For a `text` query, the payload field whose text index to search."), `filter`, `params`, `score_threshold`, `limit` (None; "Default is 10."), `lookup_from`. All default None.
- `Rrf` (2965): `k: Optional[int] = None`, `weights: Optional[List[float]] = None`. The client does not state the server default for `k`. [not verified]
- `RrfQuery` (2977): `rrf: Rrf` (required).
- `FusionQuery` (1142): `fusion: Fusion` (required). `Fusion` (1130): `RRF = "rrf"`, `DBSF = "dbsf"`.
- `MatchAny` (1881): `any: AnyVariants` (required; List[str] or List[int]).
- `MatchValue` (1945): `value: ValueVariants` (required).
- `FieldCondition` (1056): `key: str` (required); `match`, `range`, `geo_bounding_box`, `geo_radius`, `geo_polygon`, `values_count`, `is_empty`, `is_null` all default None.
- `Filter` (1080): `should`, `min_should`, `must`, `must_not`, all default None.
- `SearchParams` (3120): `hnsw_ef: Optional[int] = None`, `exact: Optional[bool] = False`, `quantization = None`, `indexed_only: Optional[bool] = False`, `acorn = None`, `idf: Optional[IdfParams] = None`. Docstring for `idf`: "Only applicable to sparse vectors with the IDF modifier enabled, and to BM25 over a text index."
- `IntegerIndexParams` (1662): `type: IntegerIndexType` (required, `INTEGER = "integer"`); `lookup`, `range`, `is_principal`, `on_disk`, `memory`, `enable_hnsw` all default None (server default true for lookup/range/enable_hnsw; false for is_principal).
- `PayloadSchemaType` (2333): `KEYWORD`, `INTEGER` (2342), `FLOAT`, `GEO`, `TEXT` (2345).
- `TextIndexParams` (3774): `type: TextIndexType` (required, `TEXT`), `tokenizer`, `min_token_len`, `max_token_len`, `lowercase`, `ascii_folding`, `phrase_matching`, `stopwords`, `on_disk`, `memory`, `stemmer`, `enable_hnsw`, `scoring: Optional[TextScoringInterface]` (`true` or `TextScoringParams(type="bm25")`), `nested`.
- `create_collection` (`qdrant_client.py:1690-1710`): `collection_name`, `vectors_config=None`, `sparse_vectors_config: Mapping[str, SparseVectorParams] = None`, `shard_number`, `sharding_method`, `replication_factor`, `write_consistency_factor`, `on_disk_payload`, `hnsw_config`, `optimizers_config`, `wal_config`, `quantization_config`, `timeout`, `strict_mode_config`, `metadata: types.Payload | None = None`, `payload`, `id_tracker`, `**kwargs`; returns `bool`.
- `create_payload_index` (`qdrant_client.py:2057`): `collection_name`, `field_name`, `field_schema: types.PayloadSchemaType | None = None`, `field_type` (deprecated), `wait=True`, `ordering`, `timeout`; returns `types.UpdateResult`. The annotation is narrower than the runtime: `qdrant_remote.py:2402-2403` also accepts `PayloadSchemaParams` (includes `TextIndexParams`, `IntegerIndexParams`), converted with `RestToGrpc.convert_payload_schema_params`.
- `query_points` (`qdrant_client.py:272-293`): `collection_name`, `query=None` (accepts `types.Query`, `types.Document`, etc.), `using=None`, `prefetch=None`, `query_filter=None`, `search_params=None`, `limit: int = 10`, `offset=None`, `with_payload=True`, `with_vectors=False`, `score_threshold=None`, `lookup_from=None`, `consistency=None`, `shard_key_selector=None`, `timeout=None`, `**kwargs`; returns `types.QueryResponse`.
- `get_collection` (`qdrant_client.py:1582`): `(collection_name: str, **kwargs) -> types.CollectionInfo`.
- `update_collection` (`qdrant_client.py:1614-1626`): `collection_name`, `optimizers_config`, `collection_params`, `vectors_config`, `hnsw_config`, `quantization_config`, `timeout`, `sparse_vectors_config`, `strict_mode_config`, `metadata: types.Payload | None = None`, `**kwargs`; returns `bool`.

---

## Q5. Collection-level metadata

- Docs, `manage-data/collections.md:356-369`: "### Collection Metadata {version="1.16.0"}" / "For convenience and better data organization, Qdrant allows attaching custom metadata to collections in the form of key-value pairs." / "Adding metadata is treated as a part of collection configuration and synchronized across all nodes in a cluster with consensus protocol." / "Note, that update operation only modifies the specified metadata fields, leaving other fields unchanged."
- Release notes, v1.16.0: "Add custom key-value metadata to collections" (PR #7123).
- Python model:
  - `CreateCollection.metadata` (`models.py:617-620`): "Arbitrary JSON metadata for the collection This can be used to store application-specific information such as creation time, migration data, inference model info, etc."
  - `CollectionConfig.metadata` (`models.py:302-305`), returned by `get_collection` through `CollectionInfo.config` (`models.py:354`).
  - `UpdateCollection.metadata` (`models.py:3950-3953`): "Metadata to update for the collection. If provided, this will merge with existing metadata. Individual keys can be removed by setting their value to `null`."
- [inference] Qdrant stores the value as opaque JSON. It does not read it. BM25 parameters that matter at query time (`k`, `b`) are set per query in the full-text API, and the docs say `avg_len` is no longer needed. So the stored values would be application metadata for reproducibility (tokenizer version, language, `k`, `b`, corpus statistics). Qdrant does not validate them.

---

## Caveats

- Upstream sources were fetched from branch heads (`master` for docs and landing page, `main` for fastembed), not pinned commits. Only the qdrant source (tag `v1.20.1`) and the charabia source (tag `v0.10.0`) are pinned.
- The rendered qdrant.tech pages were not used. The raw Markdown on GitHub had the same content.
- The Python client and the server were not run against each other. Every behaviour stated above comes from reading source or docs, not from running a query.

## NEED DOWNLOAD

- Not located (not blocked): the Rust code that applies sparse-vector `full_scan_threshold` at qdrant v1.20.1. Searched `lib/segment/src/types.rs` only. Next step: search the v1.20.1 tree under `lib/segment/` and `lib/sparse/` for `full_scan_threshold` with the sparse index config type.
- Not checked: the server default for `Rrf.k` (client states none).
