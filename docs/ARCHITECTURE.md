# STELLAR-RAG v4 — System Architecture

> **Stack**: Python 3.10+ · PyMuPDF · EasyOCR · FAISS · NetworkX · Ollama · sentence-transformers  
> **Papers implemented**: EHRAG (arXiv 2604.17458) · HybGRAG (arXiv 2412.16311)

---

## 1. Bird's-Eye View

STELLAR-RAG v4 is a **hybrid retrieval-augmented generation** system for Vietnamese university Q&A.
Three independent retrieval signals — dense semantic search, BM25 keyword search, and knowledge-graph traversal — are fused by an adaptive predictor, post-processed by an entity hypergraph diffusion, validated by a critic loop, and reranked by a cross-encoder.

```
╔══════════════════════════════════════════════════════════════════╗
║  INGEST PHASE                                                    ║
║                                                                  ║
║  PDF ──► PdfPipeline ──► Chunks ──► GraphRAG.build()            ║
║            │                              │                      ║
║         OCR + table                  ┌────┼────┬─────────┐       ║
║         detection                    ▼    ▼    ▼         ▼       ║
║                                   FAISS BM25  KG    Hypergraph   ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  QUERY PHASE                                                     ║
║                                                                  ║
║  User query                                                      ║
║      │                                                           ║
║      ├─ InputGuardrail ──[block]──► Rejected                    ║
║      ├─ LRUCache ──[hit]──► Cached answer                       ║
║      ├─ QueryRouter → simple | medium | complex                  ║
║      ├─ [medium/complex] QueryExpander → paraphrase variants     ║
║      ├─ [analytical+complex] HyDE → hypothetical passage        ║
║      │                                                           ║
║      └─ Critic Loop (max 3 iterations)                          ║
║              │                                                   ║
║         ┌────▼─── Parallel Retrieval ──────────────────┐        ║
║         │  Dense FAISS │ BM25 │ Graph (entity+PPR)     │        ║
║         └────────────────────┬────────────────────────-┘        ║
║                               │                                  ║
║                          QDAP-S fusion                           ║
║                               │                                  ║
║                          Doc-type boost                          ║
║                               │                                  ║
║                     EHRAG hypergraph rescore                     ║
║                               │                                  ║
║                     Cross-encoder rerank                         ║
║                               │                                  ║
║              Self-RAG quality ──[low]──► expand k, hops         ║
║                               │                                  ║
║              Critic Validator YES ──► break                      ║
║                               │ NO                               ║
║              Critic Commenter → enrich query → next iter         ║
║                               │                                  ║
║                    MMR diversity selection                        ║
║                    Context assembly (Organizer)                  ║
║                    LLM generation (Ollama)                       ║
║                    OutputGuardrail                               ║
║                    Cache + Memory update                         ║
║                               │                                  ║
║         [Optional] Rating 1-5 ──► QDAP-S online update          ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 2. Module Reference

| Module | Responsibility |
|--------|----------------|
| `config.py` | Centralised settings; all parameters overridable via env-vars |
| `pdf_pipeline.py` | PDF → Chunks: OCR, table detection, section tracking, chunking |
| `embedding.py` | Text → L2-normalised float32 vectors (BAAI/bge-m3 or Ollama) |
| `vector_store.py` | FAISS index with auto FlatIP / HNSWFlat selection |
| `graphrag.py` | Core orchestrator: all indices, fusion, rescoring |
| `hypergraph.py` | EHRAG entity hypergraph (H^str + H^sem + diffusion + scoring) |
| `qdap.py` | QDAP-S α predictor (Linear → Conv1D → Softmax → E[α]) |
| `critic.py` | HybGRAG Validator + Commenter |
| `agent.py` | End-to-end pipeline: guardrail → retrieval → generation |
| `memory.py` | SQLite + FAISS conversational memory + reinforced recall |
| `reranker.py` | Singleton cross-encoder (ms-marco-MiniLM-L-6-v2) |
| `router.py` | Heuristic query complexity classifier |
| `guardrail.py` | Input sanitisation + output grounding check |
| `query_expander.py` | LLM paraphrase variant generation |

---

## 3. Ingest Data Flow

```
PDF file
  │
  ├─ fitz.open()
  │    ├─ page.get_text("text")  →  native_text
  │    └─ page.get_pixmap(dpi=220) → PIL Image → EasyOCR
  │         ├─ detail=1: (bbox, text, confidence) per token
  │         ├─ row clustering: sort by y-centre, group within row_tol
  │         ├─ table mode: ≥40% rows with ≥2 cells  → " | " separator
  │         └─ text reconstruction
  │
  └─ per-page:
       ├─ SECTION_HEADING_REGEX → update current_section
       ├─ _split_text() per kind (native_text, ocr_text)
       │    ├─ is_table (≥35% lines contain " | ") → _split_table()
       │    │    ├─ whole table ≤ 3 × chunk_size → single chunk
       │    │    └─ large table: row-boundary split, header row overlap
       │    └─ else: sliding window (chunk_size=750, overlap=120 chars)
       └─ MATH_HINT_REGEX → formula candidates (cap 15/page)

All Chunks → GraphRAG.build()
  ├─ embed_texts = [_enrich_chunk_text(c) for c in chunks]
  │    └─ prefix: "[Loại: X | Mục: Y | Nguồn: Z]\n{chunk.text}"
  ├─ Embedder.encode(embed_texts) → FaissStore.build()
  │    └─ auto-select: FlatIP (n<500) or HNSWFlat M=32 (n≥500)
  ├─ BM25Okapi(_tokenize_vi(raw_texts)) → bm25_index.pkl
  ├─ for each chunk:
  │    ├─ LLM: _extract_triplets() → {subject, relation, object}
  │    │    → entity nodes + weighted relation edges
  │    └─ LLM: _extract_entities() → {name, type}
  │         → typed entity nodes + co-occurrence edges (all pairs)
  ├─ _build_entity_index()
  │    ├─ deduplicate entities by unaccented name
  │    ├─ Embedder.encode(entity_names, batch=embed_batch_size)
  │    └─ entity_vecs.npy + entity_names.json
  └─ _build_hypergraph()
       ├─ chunk_entity_map from graph edges
       ├─ Embedder.encode(chunk_texts, batch=embed_batch_size)
       └─ EntityHypergraph.build()
            ├─ H^str (E×C): entity–chunk incidence (scipy sparse)
            └─ H^sem (E×K): BIRCH clusters → Gaussian weights
```

---

## 4. Query Data Flow (Detailed)

```
Agent.answer(user_query)
│
├─ [1] InputGuardrail.check()
│       ├─ length limit (guardrail_max_query_len = 2000)
│       ├─ injection pattern regex
│       ├─ OOD/toxic: LLM classifier (optional) or regex fallback
│       └─ sanitise: whitespace normalise, truncate
│
├─ [2] LRUCache.get()  (256 entries, optional semantic TTL)
│
├─ [3] QueryProcessor.process()
│       ├─ _needs_llm_processing()?
│       │    ├─ YES: LLM → {entities, sub_queries, expanded_terms}
│       │    └─ NO:  _fast_process() → domain keyword match (zero LLM)
│       └─ ProcessedQuery(original, entities, sub_queries, expanded_terms)
│
├─ [4] QueryRouter.classify() → complexity ∈ {simple, medium, complex}
│       └─ retrieval_params() → top_k, hops, use_graph
│
├─ [5] QueryExpander.expand()  [skipped when complexity == 'simple']
│       └─ LLM → 2-3 paraphrase variants
│
├─ [6] _should_hyde()? → _hyde_expand()
│       ├─ gate: complexity=='complex' AND (analytical keyword OR ≥25 words)
│       └─ LLM (max 80 tokens) → augment: "{query}\n{hypothetical_passage}"
│
└─ [7] _retrieve_with_critic()
         │
         └─ for t in range(critic_max_iterations=3):
              │
              ├─ _retrieve_and_build_context()
              │    ├─ GraphRAG.query() or .query_batch(sub_queries)
              │    │    ├─ Embedder.encode([hyde_query or q])  →  qv
              │    │    ├─ _retrieve_parallel() [ThreadPoolExecutor, 3 workers]
              │    │    │    ├─ _dense_search(qv, k*2)
              │    │    │    │    └─ FaissStore.search() → cosine scores
              │    │    │    ├─ _bm25_search(q, k*2)
              │    │    │    │    └─ BM25Okapi.get_scores() → BM25 scores
              │    │    │    └─ _graph_retrieve(q, qv, k, hops)
              │    │    │         ├─ _entity_link_embedding(qv, top_k=10)
              │    │    │         │    └─ entity_vecs @ qv → top cosine
              │    │    │         ├─ string/section fallback if no links
              │    │    │         ├─ _ppr_local(seeds, hops)
              │    │    │         │    └─ nx.pagerank on local subgraph
              │    │    │         └─ _weighted_bfs() [PPR fallback]
              │    │    ├─ _fuse(dense, bm25, graph, qv)
              │    │    │    ├─ qdap_s: QDAP-S α-weighted min-max blend
              │    │    │    └─ rrf:    Reciprocal Rank Fusion fallback
              │    │    ├─ _doc_type_boost() → cosine to doc-type embeddings
              │    │    └─ _hypergraph_rescore(hits, qv)
              │    │         ├─ _entity_link_embedding(qv) → seed scores
              │    │         ├─ EntityHypergraph.diffuse() → entity_weights, cluster_scores
              │    │         └─ EntityHypergraph.topic_score_chunks()
              │    │              └─ S(d) += λ1·entity_term + λ2·cluster_term
              │    │
              │    ├─ Reranker.rerank(query, hits, top_k=20)
              │    │    └─ CE scores → re-sort
              │    ├─ Self-RAG quality < self_rag_threshold
              │    │    └─ re-retrieve with k*2 and hops+1
              │    └─ Organizer.organize() → context string
              │
              ├─ FAST-PATH: Self-RAG quality ≥ critic_skip_threshold → break
              │
              ├─ Critic.validate(query, context, reasoning_paths)
              │    ├─ YES → break
              │    └─ NO  → continue
              │
              └─ Critic.comment() → feedback
                   └─ Critic.enrich_query() → enriched_query → next t

         └─ final context, dense_hits, graph_hits

         ├─ _build_messages() → [{system: SYSTEM_PROMPT}, {user: question+context}]
         ├─ ollama.chat() → answer
         ├─ OutputGuardrail.check(answer, context)
         ├─ LRUCache.put()
         ├─ Memory.add(user) + Memory.add(assistant)
         └─ [if rating] QDAP-S online update
```

---

## 5. Storage Layout

```
storage/
├── docs.faiss                FAISS index (FlatIP n<500 | HNSWFlat M=32 n≥500)
├── docs_meta.json            Chunk metadata list, parallel to FAISS rows
├── bm25_index.pkl            BM25Okapi object + metadata
├── knowledge.graphml         NetworkX DiGraph (GraphML format)
├── entity_vecs.npy           Entity embedding matrix  (E, d)  float32
├── entity_names.json         Entity name list  len=E
├── qdap_s.npz                QDAP-S weights: W (101, d), b (101,)
├── memory.sqlite             SQLite conversation history
├── memory.faiss              Memory FAISS index
├── memory_meta.json          Memory metadata
├── reward_memory.faiss       High-rated answer index
├── reward_memory_meta.json
├── chunk_vecs.npy            Chunk embeddings for hypergraph  (C, d)
├── chunk_ids.json            Chunk ID list  len=C
└── hypergraph/
    ├── hgraph_H_str.npz      Structural incidence matrix (scipy sparse, E×C)
    ├── hgraph_H_sem.npz      Semantic incidence matrix   (scipy sparse, E×K)
    ├── hgraph_meta.npz       cluster_ids, cluster_centroids, n_clusters
    ├── hgraph_chunk_ids.json
    └── hgraph_entity_names.json
```

---

## 6. Configuration Quick Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding model (1024-dim, multilingual) |
| `EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` or `ollama` |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Main LLM for generation |
| `CRITIC_MODEL` | `qwen2.5:0.5b` | Fast LLM for critic (≈50-150 ms/call) |
| `CHUNK_SIZE` | `750` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between consecutive chunks |
| `TOP_K` | `6` | Final retrieved chunks per query |
| `FUSION_METHOD` | `qdap_s` | `qdap_s` or `rrf` |
| `QDAP_GRAPH_WEIGHT` | `0.15` | Graph contribution weight $w_g$ |
| `CRITIC_ENABLED` | `true` | Enable HybGRAG critic loop |
| `CRITIC_MAX_ITERATIONS` | `3` | Max retrieval-refinement rounds |
| `CRITIC_SKIP_THRESHOLD` | `0.5` | Self-RAG quality threshold to skip critic |
| `RERANKER_ENABLED` | `true` | Cross-encoder reranking |
| `HYDE_ENABLED` | `true` | HyDE (analytical complex queries only) |
| `PARALLEL_RETRIEVAL` | `true` | BM25 + Dense + Graph in 3 threads |
| `BIRCH_THRESHOLD` | `0.5` | BIRCH merge distance |
| `HYPERGRAPH_TAU` | `1.0` | Gaussian temperature τ |
| `HYPERGRAPH_GAMMA` | `0.5` | Semantic expansion decay γ |
| `HYPERGRAPH_DIFFUSE_T` | `3` | Structural propagation iterations T |
| `HYPERGRAPH_LAMBDA1` | `0.3` | Entity evidence weight λ₁ |
| `HYPERGRAPH_LAMBDA2` | `0.2` | Cluster topic weight λ₂ |
