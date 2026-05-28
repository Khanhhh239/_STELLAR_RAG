# STELLAR-RAG v4 — Retrieval Pipeline

This document explains every stage of the query-time retrieval pipeline: from raw user input to the assembled context passed to the LLM.

---

## 1. Pipeline Overview

```
User query
    │
    ▼
[Guard]     InputGuardrail  →  block / warn / sanitise
    │
    ▼
[Cache]     LRUCache 256-entry  →  cache hit: return early
    │
    ▼
[Route]     QueryRouter  →  complexity: simple | medium | complex
    │
    ▼
[Expand]    QueryExpander  →  paraphrase variants (skip if simple)
    │
    ▼
[HyDE]      _should_hyde()?  →  hypothetical passage (analytical+complex only)
    │
    ▼
╔══ Critic Loop  (max 3 iterations) ════════════════════════════╗
║                                                               ║
║  [Retrieve]  Parallel: Dense FAISS │ BM25 │ Graph (PPR)      ║
║      │                                                        ║
║  [Fuse]      QDAP-S or RRF                                   ║
║      │                                                        ║
║  [Boost]     Doc-type intent cosine boost                    ║
║      │                                                        ║
║  [EHRAG]     Hypergraph diffusion rescore                     ║
║      │                                                        ║
║  [Rerank]    Cross-encoder (top-20)                          ║
║      │                                                        ║
║  [Self-RAG]  Quality < 0.15 → re-retrieve k*2, hops+1       ║
║      │                                                        ║
║  [Critic]    Validator YES → break                           ║
║              Validator NO  → Commenter → enrich query        ║
╚══════════════════════════════════════════════════════════════╝
    │
    ▼
[MMR]       Diversity selection (Jaccard, λ=0.7)
    │
    ▼
[Organizer] Context assembly with score-proportional budget
    │
    ▼
    context string  →  LLM
```

---

## 2. Query Routing

**Source**: `src/router.py`

The router assigns a complexity level to drive all downstream decisions.

| Level | Criteria | top_k | hops | use_graph | Expander | HyDE |
|-------|----------|-------|------|-----------|----------|------|
| `simple` | Short, single entity, no analytical keywords | 4 | 1 | False | Skip | No |
| `medium` | Multiple entities or mild complexity | 6 | 2 | True | Run | No |
| `complex` | Long, multi-entity, analytical keywords, conjunctions | 8 | 3 | True | Run | If analytical |

Routing happens **before** expansion and HyDE so that those expensive steps are only triggered when justified.

---

## 3. Query Expansion

**Source**: `src/query_expander.py`  
**Condition**: `complexity != 'simple'` and `len(sub_queries) == 1`

The LLM generates 2–3 paraphrase variants of the query. These become `sub_queries` in `ProcessedQuery`. Downstream retrieval runs `query_batch()` over all variants, effectively widening the recall net.

**Skipping for simple queries**: A factual lookup like "học phí kỳ 1 là bao nhiêu?" does not benefit from paraphrase variants — the correct chunk will rank top on BM25/dense regardless. Generating variants adds ~200 ms of LLM latency for zero gain.

---

## 4. HyDE — Hypothetical Document Embedding

**Source**: `src/agent.py` — `_hyde_expand()`  
**Condition**: `complexity == 'complex'` AND (analytical keyword in query OR ≥25 words)

The LLM generates a short 2–3 sentence passage that would *answer* the query, as if it were a paragraph from the actual document. The concatenation of `query + hypothetical_passage` is then used as the dense query vector.

```
Original:    "Tại sao sinh viên bị cảnh báo học tập?"
HyDE text:   "Sinh viên bị cảnh báo học tập khi điểm trung bình tích lũy
              giảm xuống dưới ngưỡng quy định..."
Dense query: "Tại sao sinh viên bị cảnh báo học tập?\n[hypothetical text]"
```

The dense query vector is now biased toward documents that *contain answers*, not just those that match the question's keywords.

---

## 5. Dense Retrieval

**Source**: `src/graphrag.py` — `_dense_search()`  
**Index**: FAISS `IndexFlatIP` (n < 500) or `IndexHNSWFlat` M=32 efSearch=64 (n ≥ 500)

Search returns the top $2k$ nearest neighbours by inner product (= cosine similarity for L2-normalised vectors).

**Contextual embedding** (enabled by default): each chunk is embedded with a metadata prefix:

```
[Loại: quy_che | Mục: Điều 15 | Nguồn: quy_che_2021.pdf]
{chunk text}
```

This steers the embedding space so that semantically similar chunks from the same document type cluster together — improving top-k recall for domain-specific queries.

---

## 6. BM25 Sparse Retrieval

**Source**: `src/graphrag.py` — `_bm25_search()`  
**Implementation**: `rank_bm25.BM25Okapi`, `k1=1.5`, `b=0.75`

Vietnamese tokenisation (`_tokenize_vi`): extract Unicode alphanumeric tokens of length ≥ 2 using a regex that handles all Vietnamese diacritics. Stop-words are not explicitly removed; IDF handles high-frequency terms naturally.

Returns top $2k$ by BM25 score, filtering zero-score results.

**Complementarity with dense**: BM25 excels at exact-match terms: article numbers ("Điều 15"), amounts ("4.5 triệu"), dates. Dense retrieval may embed these to semantically similar vectors from wrong contexts. The two signals are combined via QDAP-S.

---

## 7. Graph Retrieval

**Source**: `src/graphrag.py` — `_graph_retrieve()`

### 7.1 Entity Linking

Embed all unique entities at ingest time and store in `entity_vecs.npy`.  
At query time, find the top entities by cosine:

$$\text{linked} = \{e : \hat{\mathbf{v}}_e \cdot \hat{\mathbf{q}} \geq \theta_{\text{link}},\; e \in \text{top-10}\}, \quad \theta_{\text{link}} = 0.45$$

Cache: `_entity_name_to_idx` dict built once at `load()` / `build()` gives O(1) name→index lookup (was O(n) before Fix F).

**Fallbacks** (in order):
1. Substring match: entity name appears in query string
2. Section keyword overlap: ≥2 query words overlap with a section heading

### 7.2 Local Subgraph PPR

Expand a local subgraph around the seed entities using BFS up to `hops` steps (capped at `ppr_max_subgraph = 500` nodes), then run personalised PageRank on this subgraph.

The personalisation vector $\mathbf{p}$ places equal weight on all seed entity nodes:

$$p_v = \begin{cases} 1 / |\text{seeds}| & v \in \text{seed entities} \\ 0 & \text{otherwise} \end{cases}$$

Extract chunk nodes from the PPR result and rank by their PPR score.

**Relation weights** drive the random walk. Higher-weight edges are more "teleportable":

| Relation | Weight |
|----------|--------|
| `co_tien_quyet` (prerequisite) | 2.0 |
| `quy_dinh_ve` (regulates) | 1.8 |
| `yeu_cau` (requires) | 1.7 |
| `co_occurs_with` | 0.9 |
| `has_chunk` | 0.7 |

### 7.3 Weighted BFS Fallback

If PPR returns no chunk nodes (e.g., no chunk nodes in subgraph), fall back to weighted BFS:

$$s_{\text{BFS}}(v, t+1) = \sum_{u \to v} s_{\text{BFS}}(u, t) \cdot w_{u \to v} \cdot \delta^{t+1}$$

where $\delta = 0.7$ is the per-hop decay.

---

## 8. QDAP-S Fusion

**Source**: `src/graphrag.py` — `_qdap_fuse()`

See [MATH_FOUNDATIONS.md §4](MATH_FOUNDATIONS.md) for the full derivation.

**Summary**:
1. Predict $\alpha$ from query embedding via QDAP-S predictor.
2. Min-max normalise dense, BM25, and graph scores independently.
3. Blend dense and BM25: $s_{\text{db}} = \alpha \cdot s'_{\text{dense}} + (1-\alpha) \cdot s'_{\text{BM25}}$
4. Blend with graph: $s_{\text{final}} = 0.85 \cdot s_{\text{db}} + 0.15 \cdot s'_{\text{graph}}$
5. Sort descending and return.

Each result carries `qdap_alpha` for debugging.

---

## 9. Doc-Type Intent Boost

**Source**: `src/graphrag.py` — `_doc_type_boost()`

Classify query intent into one of six document types by cosine similarity to pre-embedded descriptions:

| Type | Description embedded |
|------|---------------------|
| `hoc_phi` | "học phí, chi phí học tập, tiền đóng học, miễn giảm học phí" |
| `quy_che` | "quy chế đào tạo, quy định, điều khoản, chính sách giáo dục" |
| `chuong_trinh` | "chương trình đào tạo, kế hoạch học tập, môn học, tín chỉ" |
| `lich_hoc` | "lịch học, thời khóa biểu, lịch thi, lịch giảng dạy" |
| `tuyen_sinh` | "tuyển sinh, nhập học, xét tuyển, điểm chuẩn, đăng ký" |
| `thong_bao` | "thông báo, thông tin cập nhật, thông cáo" |

Only activates when best-type similarity ≥ 0.40. Matching chunks get score multiplied by 1.35.

---

## 10. EHRAG Hypergraph Rescore

**Source**: `src/graphrag.py` — `_hypergraph_rescore()`  
See [HYPERGRAPH_EHRAG.md](HYPERGRAPH_EHRAG.md) for full algorithm.

**Steps**:
1. Link top-8 entities to the query via `_entity_link_embedding()`.
2. Build `seed_entity_scores = {name: cosine_sim}`.
3. Run `EntityHypergraph.diffuse()` → `entity_weights`, `cluster_scores`.
4. Run `topic_score_chunks(hits, entity_weights, cluster_scores)` → re-sorted hits.

Fails open: any exception returns the original hits unchanged.

---

## 11. Cross-Encoder Reranking

**Source**: `src/reranker.py`  
**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (≈22 MB, lazy-loaded)

Takes the top `reranker_top_k = 20` fused candidates and scores each pair `(query, chunk_text)` jointly. The cross-encoder attends across query and document, catching relevance nuances missed by bi-encoder cosine similarity.

Result: candidates re-sorted by CE score, with `retrieval_type` annotated with `+ce` suffix.

---

## 12. Self-RAG Quality Expansion

**Source**: `src/agent.py` — `_retrieve_and_build_context()`

After context assembly:

$$q_{\text{quality}} = \frac{|T_q \cap T_c|}{|T_q|}$$

If $q_{\text{quality}} < 0.15$:
- Re-retrieve with $k' = \min(2k, 20)$ and $\text{hops}' = \min(\text{hops}+1, 3)$
- Rerank the expanded set
- Rebuild context

This triggers for queries where the first retrieval retrieved the wrong document types or the query terms have low overlap with any stored chunk.

---

## 13. HybGRAG Critic Loop

**Source**: `src/agent.py` — `_retrieve_with_critic()`, `src/critic.py`  
**Paper**: HybGRAG (arXiv 2412.16311)

### 13.1 Fast-Path Bypass

Before calling the LLM validator, check Self-RAG quality:

$$\text{if } q_{\text{quality}} \geq 0.5 \Rightarrow \text{break (skip critic)}$$

This avoids 50–150 ms per-iteration LLM overhead for queries where the context clearly covers the question.

### 13.2 Validator (C_val)

A small fast LLM (`qwen2.5:0.5b`, temperature 0, max 8 tokens) answers YES/NO:

> "Does the retrieved context contain sufficient information to answer the query?"

YES → proceed to generation.  
NO → trigger Commenter.

Fail-open: LLM error returns True (proceed), so the pipeline never stalls.

### 13.3 Commenter (C_com)

A structured single-line feedback (temperature 0.2, max 120 tokens):

```
Thiếu thực thể: [entity/concept name]
Thiếu điều khoản: [article/decision reference]
Thiếu bảng số liệu: [table name or data type]
```

### 13.4 Query Enrichment

```python
enriched = f"{original_query} [Cần thêm: {feedback}]"
```

The enriched query is used in the next iteration's dense and BM25 retrieval — the structured feedback terms serve as additional query tokens.

### 13.5 Iteration Guard

Stop conditions:
- Validator returns YES
- Self-RAG fast-path triggered
- `critic_max_iterations = 3` reached
- Commenter returns empty string
- Enriched query equals previous query (no change)

---

## 14. Context Assembly (Organizer)

**Source**: `src/agent.py` — `Organizer.organize()`

### 14.1 MMR Selection

From the full fused hit list, apply MMR to select `top_k = 6` diverse, relevant chunks:

$$\text{MMR}(d_i) = \lambda \cdot s(d_i) - (1 - \lambda) \cdot \max_{d_j \in S} J(d_i, d_j), \quad \lambda = 0.7$$

### 14.2 Score-Proportional Budget

Total context budget: `max_context_chars = 6000`. Per-chunk allocation:

$$\text{budget}_i = \text{clip}\!\left(\frac{s_i}{\sum_j s_j} \cdot n \cdot \text{max\_chars\_per\_chunk},\; \text{min\_chars},\; \text{max\_chars}\right)$$

where $n$ is the number of selected chunks, `max_chars_per_chunk = 500`, `min_chars_per_chunk = 150`.

High-scoring chunks receive more characters; low-scoring chunks still get at least `min_chars`.

### 14.3 Sentence-Level Compression

Each chunk is compressed to its budget by `compress_text()`:

1. Split into sentences on `[.!?\n]` boundaries.
2. Score each sentence: $\text{score}(s) = |T_s \cap T_q| / \sqrt{|T_s|}$ (overlap normalised by sentence length).
3. Greedy: select highest-scoring sentences until budget is filled.
4. Restore original order.

This preserves the most query-relevant sentences from each chunk.

### 14.4 Context Sections

The assembled context string contains (in order):

1. **Tài liệu liên quan** — top-k hybrid chunks (with source/page/section/score tags)
2. **Quan hệ tri thức** — graph relation paths from PPR traversal
3. **Hội thoại liên quan** — relevant conversation memory (cosine ≥ 0.5)
4. **Câu trả lời chất lượng cao** — reinforced-recall high-rated answers
5. **Lịch sử hội thoại** — last 4 turns
6. **Ghi chú tra cứu** — critic feedback note (if applicable)
