# STELLAR-RAG v4 — EHRAG + HybGRAG Hybrid Retrieval System

Vietnamese university Q&A system combining hypergraph-enhanced retrieval
(EHRAG, arxiv 2604.17458) with agentic critic validation (HybGRAG, arxiv
2412.16311) on top of the STELLAR-RAG v3 QDAP-S fusion baseline.

## Architecture

```
Query
  |
  v
[InputGuardrail]    -- injection detection + LLM safety classifier
  |
  v
[QueryProcessor]    -- heuristic or LLM entity extraction + sub-query split
  |
  v
[QueryExpander]     -- LLM paraphrase variants (robustness)
  |
  v
[QueryRouter]       -- simple / compound / complex tier
  |
  v
[HyDE]              -- hypothetical document embedding (complex only)
  |
  v
  +-------- BM25 search --------+
  |                              |
  +---- Dense FAISS search -----+-- parallel
  |                              |
  +---- Graph PPR retrieval ----+
  |
  v
[QDAP-S Fusion]     -- query-adaptive alpha prediction for dense/sparse blend
  |
  v
[Doc-type Boost]    -- embedding-based intent -> doc_type boost
  |
  v
[EHRAG Hypergraph Diffusion]
  |  1. Seed entity scores from dense retrieval (embedding similarity)
  |  2. Semantic expansion via H^sem (BIRCH cluster hyperedges)
  |  3. Structural propagation via H^str (co-occurrence/chunk hyperedges)
  |  4. Topic-aware 3-component re-scoring:
  |       S(d) = S_dense + lambda1 * entity_evidence + lambda2 * cluster_term
  |
  v
[CE Reranker]       -- optional CrossEncoder (top-20 candidates)
  |
  v
[HybGRAG Critic Loop]  max 3 iterations
  |  Validator: "Is this context sufficient?"  (qwen2.5:0.5b)
  |  Commenter: "What information is missing?" (structured feedback)
  |  Enrich query -> re-retrieve if insufficient
  |
  v
[Context Assembly]  -- MMR diversity + proportional char budget
  |
  v
[LLM Generation]    -- qwen2.5:7b-instruct (streaming or blocking)
  |
  v
[OutputGuardrail]   -- grounding check + hallucination marker detection
  |
  v
Answer
```

## New Features vs v3

| Feature | v3 | v4 |
|---------|----|----|
| Entity extraction | Triplets only (~2-5/chunk) | Triplets + ALL named entities with type classification |
| Co-occurrence edges | No | Yes (all entity pairs in same chunk) |
| Semantic clustering | No | BIRCH auto-clustering of entity embeddings |
| Structural hyperedge | No | H^str incidence matrix (scipy.sparse) |
| Semantic hyperedge | No | H^sem Gaussian-weighted cluster connections |
| Diffusion retrieval | No | 3-iteration structural propagation + 1-shot semantic |
| Topic-aware scoring | No | 3-component: dense + entity_evidence + cluster_term |
| Critic validation | No | HybGRAG validator + commenter (up to 3 refinement iters) |
| Verbalized paths | Used for display only | Passed to critic validator for evidence |
| Entity types | None | RULE, SUBJECT, AMOUNT, DATE, ORG, PERSON, CONDITION, PROCESS |
| Memory usage | Low | CPU-safe: scipy.sparse, batched embedding, 50k entity cap |

## Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Models: `ollama pull qwen2.5:7b-instruct` and `ollama pull nomic-embed-text`
- Optional (critic): `ollama pull qwen2.5:0.5b`

### Setup

```bash
cd C:\Users\Admin\Downloads\improve_RAG

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

**If you have no GPU** (CPU-only):
```bash
.venv\Scripts\pip install torch>=2.0.0
.venv\Scripts\pip install -r requirements.txt --no-deps
.venv\Scripts\pip install ollama python-dotenv networkx numpy faiss-cpu sentence-transformers PyMuPDF Pillow easyocr sympy==1.13.1 rapidfuzz tqdm rank_bm25 scipy scikit-learn
```

### New dependencies vs v3
```
scipy>=1.11.0        # sparse hyperedge matrices (H^str, H^sem)
scikit-learn>=1.3.0  # BIRCH clustering for semantic hyperedges
```

### Configuration

Copy `.env.example` to `.env` and edit:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
EMBEDDING_BACKEND=ollama
EMBED_MODEL=nomic-embed-text
USE_GPU=true
```

## Usage

### Ingest PDFs

Place PDF files in `data/raw/` and run:

```bash
.venv\Scripts\python ingest.py
```

This builds:
- `storage/docs.faiss` + `storage/docs_meta.json` — dense FAISS index
- `storage/bm25_index.pkl` — BM25 index
- `storage/knowledge.graphml` — knowledge graph
- `storage/entity_vecs.npy` + `storage/entity_names.json` — entity index
- `storage/hypergraph/` — EHRAG hypergraph artefacts (H^str, H^sem, clusters)
- `storage/chunk_vecs.npy` + `storage/chunk_ids.json` — chunk embeddings for diffusion

### Chat

```bash
.venv\Scripts\python app.py
```

### Evaluation

```bash
.venv\Scripts\python eval_pipeline.py [eval_prompts.txt]
```

## Configuration Reference

All settings can be overridden via environment variables.

### EHRAG Hypergraph Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `BIRCH_THRESHOLD` | `0.5` | BIRCH merge distance threshold |
| `BIRCH_N_CLUSTERS` | (auto) | Number of clusters; unset = auto-detect |
| `HYPERGRAPH_TOP_D` | `10` | Top-D nearest entity neighbours per cluster |
| `HYPERGRAPH_TAU` | `1.0` | Gaussian weight temperature τ |
| `HYPERGRAPH_GAMMA` | `0.5` | Semantic expansion decay γ |
| `HYPERGRAPH_DIFFUSE_T` | `3` | Structural propagation iterations T |
| `HYPERGRAPH_L` | `50` | Top-L chunks in query gating matrix |
| `HYPERGRAPH_EPSILON` | `0.01` | Activation threshold ε |
| `HYPERGRAPH_LAMBDA1` | `0.3` | Entity evidence weight λ₁ |
| `HYPERGRAPH_LAMBDA2` | `0.2` | Cluster topic weight λ₂ |

### HybGRAG Critic Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CRITIC_ENABLED` | `true` | Enable/disable critic loop |
| `CRITIC_MAX_ITERATIONS` | `3` | Max retrieval-refinement iterations |
| `CRITIC_MODEL` | `qwen2.5:0.5b` | Fast Ollama model for critic calls |

### Entity Extraction Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENTITY_EXTRACT_ENABLED` | `true` | Rich named entity extraction |
| `MAX_ENTITIES_PER_CHUNK` | `20` | Max entities extracted per chunk |

### Core Retrieval Settings (unchanged from v3)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Main LLM model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `TOP_K` | `6` | Number of chunks to retrieve |
| `FUSION_METHOD` | `qdap_s` | `qdap_s` or `rrf` |
| `QDAP_GRAPH_WEIGHT` | `0.15` | Graph contribution in QDAP fusion |
| `HYDE_ENABLED` | `true` | Enable HyDE for complex queries |
| `SELF_RAG_ENABLED` | `true` | Enable Self-RAG quality expansion |
| `SELF_RAG_THRESHOLD` | `0.15` | Token-overlap threshold for expansion |
| `RERANKER_ENABLED` | `false` | Enable CrossEncoder reranker |
| `GUARDRAIL_ENABLED` | `true` | Enable input/output guardrails |
| `GUARDRAIL_LLM_CLASSIFY` | `false` | Use LLM for safety classification |
| `QUERY_EXPANSION_ENABLED` | `true` | Enable query paraphrase expansion |
| `EMBED_BATCH_SIZE` | `32` | Batch size for embedding operations |

## Hardware Requirements

- **CPU-only**: Works fully. All new code (hypergraph.py, critic.py) uses
  NumPy + scipy only — no CUDA/PyTorch in new modules.
- **GPU**: Ollama handles GPU for LLM inference. PyTorch optional for
  sentence-transformers backend; set `EMBEDDING_BACKEND=ollama` for CPU-only.
- **RAM**: BIRCH clustering is skipped automatically if entity count > 50,000.
  scipy.sparse keeps hyperedge matrices memory-efficient for typical corpora.
  Recommended: 8 GB RAM minimum.
- **Disk**: ~500 MB for models + index files for a typical 50-document corpus.

## Performance vs v3

Expected improvements from EHRAG + HybGRAG:
- **nDCG@10**: +5–10% from topic-aware entity evidence scoring
- **Grounding overlap**: +3–8% from critic-driven context enrichment
- **Hallucination rate**: Lower due to critic validation ensuring context adequacy
- **Latency**: +50–200 ms per query (critic calls use fast 0.5b model)
  Set `CRITIC_ENABLED=false` to recover full v3 latency.

## File Structure

```
improve_RAG/
├── ingest.py           -- PDF ingestion entry point
├── app.py              -- Interactive chat
├── eval_pipeline.py    -- Evaluation pipeline
├── eval_prompts.txt    -- Test prompts
├── requirements.txt    -- Dependencies (+ scipy, scikit-learn)
├── .env.example        -- Environment variable template
└── src/
    ├── config.py       -- All settings + new EHRAG/HybGRAG config
    ├── agent.py        -- Agent with HybGRAG critic loop
    ├── graphrag.py     -- GraphRAG with EHRAG hypergraph
    ├── hypergraph.py   -- EHRAG EntityHypergraph (NEW)
    ├── critic.py       -- HybGRAG Critic validator + commenter (NEW)
    ├── qdap.py         -- QDAP-S alpha predictor (unchanged)
    ├── guardrail.py    -- Input/output safety (unchanged)
    ├── query_expander.py -- Query paraphrase expansion (unchanged)
    ├── router.py       -- Adaptive query router (unchanged)
    ├── memory.py       -- SQLite + FAISS memory (unchanged)
    ├── reranker.py     -- CrossEncoder reranker (unchanged)
    ├── embedding.py    -- Ollama/ST embedding backend (unchanged)
    ├── vector_store.py -- FAISS index wrapper (unchanged)
    └── pdf_pipeline.py -- PDF → Chunk pipeline (unchanged)
```
