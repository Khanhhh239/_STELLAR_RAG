# Mathematical Analysis of Techniques in STELLAR-RAG v4

This document explains the main techniques implemented in `improve_RAG` using mathematical notation. The relevant modules are `src/pdf_pipeline.py`, `src/embedding.py`, `src/vector_store.py`, `src/graphrag.py`, `src/hypergraph.py`, `src/qdap.py`, `src/agent.py`, `src/critic.py`, `src/guardrail.py`, `src/reranker.py`, and `src/memory.py`.

## 1. PDF Parsing, OCR, and Chunking

The ingestion pipeline extracts native PDF text with PyMuPDF, reconstructs OCR text with EasyOCR, detects formula-like lines, and splits text into overlapping chunks.

For a character sequence `x` of length `n`, chunk size `m`, and overlap `o`, the stride is:

$$
s = \max(1, m-o)
$$

The `i`-th chunk is:

$$
c_i = x_{is:\min(is+m,n)}
$$

Adjacent chunks preserve approximately `o` characters of context:

$$
|c_i \cap c_{i+1}| \approx o
$$

The OCR reconstruction groups bounding boxes into rows. For box center `y_i` and box height `h_i`, the row merge threshold is:

$$
\delta_y
=
\max\left(
0.6 \cdot \text{median}(h_i),
8
\right)
$$

Two boxes are assigned to the same row when:

$$
|y_i-y_j| \le \delta_y
$$

This is especially important for tables because row-wise reconstruction preserves relationships between columns.

## 2. Embeddings and Cosine Similarity

`src/embedding.py` supports two embedding backends:

- Ollama embeddings, with `nomic-embed-text` as the default model.
- SentenceTransformers, optionally using GPU.

Each text `d` is mapped to a vector:

$$
\mathbf{v}_d = f_\theta(d) \in \mathbb{R}^p
$$

The vector is L2-normalized:

$$
\hat{\mathbf{v}}_d
=
\frac{\mathbf{v}_d}
{\lVert \mathbf{v}_d \rVert_2 + \epsilon}
$$

For normalized vectors, inner product equals cosine similarity:

$$
\text{sim}(q,d)
=
\hat{\mathbf{v}}_q^\top \hat{\mathbf{v}}_d
=
\cos(\hat{\mathbf{v}}_q,\hat{\mathbf{v}}_d)
$$

For the Ollama backend, batch embedding is parallelized with a thread pool. In the ideal case, `N` independent requests take roughly:

$$
T_{\text{parallel}}
\approx
\max_i T_i
$$

instead of:

$$
T_{\text{serial}}
=
\sum_{i=1}^N T_i
$$

within the configured worker limit.

## 3. Dense Retrieval with FAISS

`src/vector_store.py` uses FAISS `IndexFlatIP`. Let `V` be the matrix of normalized chunk vectors:

$$
V =
\begin{bmatrix}
\hat{\mathbf{v}}_{d_1}^\top \\
\hat{\mathbf{v}}_{d_2}^\top \\
\cdots \\
\hat{\mathbf{v}}_{d_N}^\top
\end{bmatrix}
\in
\mathbb{R}^{N \times p}
$$

The dense retrieval score is:

$$
s_{\text{dense}}(q,d_i)
=
\hat{\mathbf{v}}_q^\top
\hat{\mathbf{v}}_{d_i}
$$

FAISS returns:

$$
\text{TopK}_{d_i}
\,
s_{\text{dense}}(q,d_i)
$$

Because the vectors are normalized, this is nearest-neighbor search under cosine similarity.

## 4. BM25 Sparse Retrieval

`src/graphrag.py` uses `rank_bm25.BM25Okapi`. For query `q` and document `d`, BM25 is:

$$
s_{\text{bm25}}(q,d)
=
\sum_{t \in q}
\text{IDF}(t)
\cdot
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}
$$

The inverse document frequency term is:

$$
\text{IDF}(t)
=
\log
\frac{N-n_t+0.5}{n_t+0.5}
$$

where:

| Symbol | Meaning |
| --- | --- |
| `f(t,d)` | Frequency of token `t` in document `d` |
| `N` | Number of documents |
| `n_t` | Number of documents containing token `t` |
| `doc_len(d)` | Document length |
| `avgdl` | Average document length |
| `k_1` | Term-frequency saturation parameter |
| `b` | Length-normalization parameter |

The default values are `BM25_K1=1.5` and `BM25_B=0.75`.

## 5. Knowledge Graph Construction

`GraphRAG._add_chunk_to_graph()` constructs a directed graph:

$$
G=(V,E)
$$

Each edge carries a relation label and weight:

$$
e = (u,v,r,w_r)
$$

The graph contains document hierarchy, extracted triplets, rich named entities, entity-to-chunk mention edges, chunk-to-entity containment edges, entity co-occurrence edges, and entity type edges.

### Entity Linking

Entity linking compares the query embedding with entity embeddings:

$$
\text{link}(q)
=
\left\{
e_i
\mid
\hat{\mathbf{v}}_{e_i}^\top
\hat{\mathbf{v}}_q
\ge
\tau_e
\right\}
$$

The default threshold is:

$$
\tau_e = 0.45
$$

## 6. Local Personalized PageRank

Graph retrieval starts from linked seed entities and extracts a local subgraph. For seed set `S`, the personalization vector is:

$$
p_i
=
\begin{cases}
\frac{1}{|S|}, & i \in S \\
0, & i \notin S
\end{cases}
$$

Personalized PageRank iterates:

$$
\mathbf{r}^{(t+1)}
=
\alpha P^\top\mathbf{r}^{(t)}
+
(1-\alpha)\mathbf{p}
$$

The code uses:

$$
\alpha = 0.85
$$

The graph score of a retrieved chunk is:

$$
s_{\text{graph}}(q,d)
=
r_{\text{chunk}(d)}
$$

If PageRank fails or returns no chunks, the code falls back to weighted BFS:

$$
s(v)
\leftarrow
s(v)
+
s(u)w_{uv}0.7^{h+1}
$$

where `h` is the current hop depth.

## 7. QDAP-S Adaptive Fusion

`src/qdap.py` predicts an adaptive dense/sparse coefficient:

$$
\alpha \in [0,1]
$$

Given query embedding `q`, QDAP-S first applies a linear projection:

$$
\mathbf{z}
=
W\hat{\mathbf{v}}_q+\mathbf{b}
$$

where:

$$
W \in \mathbb{R}^{101 \times p}
$$

A one-dimensional moving-average convolution smooths the logits:

$$
\tilde{z}_i
=
\frac{1}{7}
\sum_{j=-3}^{3}
z_{i+j}
$$

The smoothed logits are converted into a probability distribution:

$$
p_i
=
\frac{\exp(\tilde{z}_i)}
{\sum_{j=0}^{100}\exp(\tilde{z}_j)}
$$

The alpha grid is:

$$
\alpha_i = \frac{i}{100},
\quad
i=0,\dots,100
$$

The predicted alpha is the expectation:

$$
\alpha
=
\mathbb{E}[\alpha]
=
\sum_{i=0}^{100}
p_i\alpha_i
$$

If no trained weights exist at `storage/qdap_s.npz`, the initialized weights are zero, which gives a uniform distribution and:

$$
\alpha = 0.5
$$

### Min-Max Normalization

Each score channel is normalized to `[0,1]`:

$$
\tilde{s}(d)
=
\begin{cases}
\frac{s(d)-s_{\min}}{s_{\max}-s_{\min}}, & s_{\max}\ne s_{\min} \\
0.5, & s_{\max}=s_{\min}
\end{cases}
$$

Dense and BM25 scores are fused as:

$$
s_{db}(q,d)
=
\alpha\tilde{s}_{\text{dense}}(q,d)
+
(1-\alpha)\tilde{s}_{\text{bm25}}(q,d)
$$

Graph scores are blended as:

$$
s_{\text{hybrid}}(q,d)
=
(1-w_g)s_{db}(q,d)
+
w_g\tilde{s}_{\text{graph}}(q,d)
$$

The default graph weight is:

$$
w_g = 0.15
$$

## 8. Reciprocal Rank Fusion

If QDAP-S cannot be applied, the system falls back to Reciprocal Rank Fusion:

$$
s_{\text{rrf}}(d)
=
\sum_{\ell \in L}
\frac{1}
{k_{\text{rrf}}+\text{rank}_\ell(d)}
$$

where `L` is the set of retrieval lists. The default is:

$$
k_{\text{rrf}} = 60
$$

RRF is robust because it depends on ranks rather than raw score scales.

## 9. Document-Type Boost

The system embeds descriptions for document types such as `hoc_phi`, `quy_che`, `chuong_trinh`, `lich_hoc`, `tuyen_sinh`, and `thong_bao`.

The best document type is:

$$
t^*
=
\arg\max_t
\hat{\mathbf{v}}_q^\top
\hat{\mathbf{v}}_t
$$

If the similarity exceeds `0.40`, chunks with that document type are boosted:

$$
s'(q,d)
=
\begin{cases}
\beta s(q,d), & \text{type}(d)=t^* \\
s(q,d), & \text{otherwise}
\end{cases}
$$

The default is:

$$
\beta = 1.35
$$

## 10. EHRAG Hypergraph Construction

`src/hypergraph.py` builds a hypergraph over entities and chunks.

Let:

$$
E=\{e_1,\dots,e_M\}
$$

and:

$$
C=\{c_1,\dots,c_N\}
$$

### Structural Incidence Matrix

The structural incidence matrix is:

$$
H^{\text{str}}
\in
\mathbb{R}^{M \times N}
$$

with entries:

$$
H^{\text{str}}_{ij}
=
\begin{cases}
1, & e_i \text{ appears in } c_j \\
0, & \text{otherwise}
\end{cases}
$$

This matrix is stored as a SciPy sparse CSR matrix.

### Semantic Incidence Matrix

BIRCH clusters entity embeddings into `K` semantic clusters with centroids:

$$
\boldsymbol{\mu}_k
=
\frac{1}{|C_k|}
\sum_{e_i \in C_k}
\mathbf{x}_i
$$

For each centroid, the top-`D` nearest entities are connected with Gaussian weights:

$$
H^{\text{sem}}_{ik}
=
\begin{cases}
\exp\left(
-\frac{
\lVert \mathbf{x}_i-\boldsymbol{\mu}_k \rVert_2^2
}{\tau}
\right),
& e_i \in \mathcal{N}_D(\boldsymbol{\mu}_k) \\
0,
& \text{otherwise}
\end{cases}
$$

The defaults are:

$$
D = 10,
\quad
\tau = 1.0
$$

## 11. EHRAG Diffusion

Initial entity activation comes from entity linking:

$$
a_i^{(0)}
=
\max\left(
0,
\hat{\mathbf{v}}_{e_i}^\top
\hat{\mathbf{v}}_q
\right)
$$

Semantic expansion runs once:

$$
a_{\text{sem}}
=
\gamma
H^{\text{sem}}
(H^{\text{sem}})^\top
a^{(0)}
$$

The expanded activation is:

$$
a^{(1)}
=
a^{(0)}
+
a_{\text{sem}}
$$

The default semantic decay is:

$$
\gamma = 0.5
$$

### Structural Propagation

For each iteration:

$$
s^{(t)}
=
(H^{\text{str}})^\top
a^{(t)}
$$

The query-gating matrix `G_q` is diagonal and only keeps the top-`L` chunks by query similarity:

$$
G_q[j,j]
=
\begin{cases}
\hat{\mathbf{v}}_{c_j}^\top\hat{\mathbf{v}}_q,
& j \in \text{TopL}
\left(
\hat{\mathbf{v}}_{c_j}^\top\hat{\mathbf{v}}_q
\right) \\
0,
& \text{otherwise}
\end{cases}
$$

Chunk activation is propagated back to entities:

$$
\Delta a^{(t+1)}
=
H^{\text{str}}
G_q
s^{(t)}
$$

Thresholding removes weak activations:

$$
a_i^{(t+1)}
=
\begin{cases}
\Delta a_i^{(t+1)}, & \Delta a_i^{(t+1)} \ge \epsilon \\
0, & \Delta a_i^{(t+1)} < \epsilon
\end{cases}
$$

The defaults are:

$$
T = 3,
\quad
L = 50,
\quad
\epsilon = 0.01
$$

The cumulative entity weight is:

$$
w_i
=
\sum_{t=1}^{T}
a_i^{(t)}
$$

The topic score for cluster `k` is:

$$
S_{\text{topic}}(k)
=
\frac{1}{|C_k|}
\sum_{e_i \in C_k}
w_i
$$

## 12. EHRAG Topic-Aware Rescoring

After base fusion and document-type boosting, EHRAG adds entity and semantic-cluster evidence:

$$
S(d)
=
S_{\text{base}}(q,d)
+
\lambda_1
\sum_{v \in d}
\log(1+w(v))
+
\lambda_2
\log
\left(
1+
\sum_{k \in \mathcal{K}(d)}
S_{\text{topic}}(k)
\right)
$$

where:

| Term | Meaning |
| --- | --- |
| `S_base(q,d)` | Score after fusion and document-type boost |
| `w(v)` | Diffused activation weight of entity `v` |
| `K(d)` | Clusters represented by entities in chunk `d` |
| `lambda_1` | Entity evidence weight |
| `lambda_2` | Cluster topic weight |

The defaults are:

$$
\lambda_1 = 0.3,
\quad
\lambda_2 = 0.2
$$

The logarithms reduce the effect of very large activation values.

## 13. Cross-Encoder Reranking

The optional reranker uses a cross-encoder on query-document pairs.

Bi-encoder retrieval scores query and document independently:

$$
s_{\text{bi}}(q,d)
=
f(q)^\top f(d)
$$

Cross-encoder reranking scores the pair jointly:

$$
s_{\text{ce}}(q,d)
=
g_\phi([q;d])
$$

When `RERANKER_ENABLED=true`, the top candidate pool receives:

$$
s'(q,d)
=
s_{\text{ce}}(q,d)
$$

Candidates outside the reranking pool retain their existing fused scores.

## 14. Maximal Marginal Relevance

The organizer uses Maximal Marginal Relevance to balance relevance and diversity.

Jaccard overlap between two chunks is:

$$
J(d_i,d_j)
=
\frac{|T_i \cap T_j|}
{|T_i \cup T_j|}
$$

The next selected chunk maximizes:

$$
\text{MMR}(d)
=
\lambda s(d)
-
(1-\lambda)
\max_{d_j \in S_{\text{selected}}}
J(d,d_j)
$$

The default is:

$$
\lambda = 0.7
$$

This reduces redundant context while preserving high-ranking evidence.

## 15. Context Budgeting and Sentence Compression

Each selected chunk receives a character budget proportional to its score:

$$
B_i
=
\text{clip}
\left(
B_{\max}
n
\frac{s_i}{\sum_j s_j},
B_{\min},
B_{\max}
\right)
$$

The code defaults to:

$$
B_{\min}=150,
\quad
B_{\max}=500
$$

If a chunk is longer than its budget, sentences are ranked by query-token overlap:

$$
\text{sent\_score}(u,q)
=
\frac{|T(u)\cap T(q)|}
{\sqrt{|T(u)|}}
$$

The highest-scoring sentences are kept within the budget and then restored to their original order.

## 16. HyDE

For complex queries, HyDE asks the LLM to generate a short hypothetical answer-like passage `h`. Dense retrieval embeds:

$$
q'
=
q \oplus h
$$

and:

$$
\mathbf{v}_{q'}
=
f_\theta(q')
$$

The goal is to make the query embedding closer to chunks that contain the expected answer style and terminology.

## 17. Query Processing, Routing, and Expansion

The query processor returns:

$$
P(q)
=
(
\text{entities},
\text{subqueries},
\text{expanded terms}
)
$$

The router maps the processed query to:

$$
\text{tier}(q)
\in
\{\text{simple},\text{compound},\text{complex}\}
$$

The routing table is:

$$
\begin{array}{c|ccc}
\text{tier} & k & \text{hops} & \text{use graph} \\
\hline
\text{simple} & 4 & 0 & 0 \\
\text{compound} & 6 & 1 & 1 \\
\text{complex} & 10 & 2 & 1
\end{array}
$$

Query expansion creates:

$$
Q
=
\{q,q'_1,q'_2\}
$$

Retrieval then merges candidates from the relevant sub-queries and variants.

## 18. Self-RAG Quality Expansion

The lightweight Self-RAG heuristic estimates context quality with token overlap:

$$
\text{quality}(q,C)
=
\frac{|T(q)\cap T(C)|}
{|T(q)|}
$$

If:

$$
\text{quality}(q,C)
<
\theta_{\text{self}}
$$

the agent expands retrieval:

$$
k'=\min(2k,20)
$$

and:

$$
h'=\min(h+1,3)
$$

The default threshold is:

$$
\theta_{\text{self}} = 0.15
$$

## 19. HybGRAG Critic Loop

The critic has two LLM components:

| Component | Role |
| --- | --- |
| Validator | Decides whether context is sufficient |
| Commenter | Describes missing information when context is insufficient |

At iteration `t`, retrieval produces:

$$
C_t
=
R(q_t)
$$

The validator returns:

$$
y_t
=
C_{\text{val}}(q_0,C_t,P_t)
\in
\{\text{YES},\text{NO}\}
$$

If the answer is `NO`, the commenter generates feedback:

$$
f_t
=
C_{\text{com}}(q_0,C_t)
$$

The next retrieval query is:

$$
q_{t+1}
=
q_0 \oplus
\text{ "[Need more: " } f_t \text{ "]"}
$$

The maximum number of iterations is:

$$
T_{\text{critic}} = 3
$$

The critic is fail-open: if validation fails due to an LLM error, the context is treated as sufficient so the pipeline can continue.

## 20. Input and Output Guardrails

Input guardrails include length checking, prompt-injection detection, optional LLM safety classification, and sanitization.

The length constraint is:

$$
|q| \le L_{\max}
$$

with:

$$
L_{\max}=2000
$$

Output grounding overlap is:

$$
\text{ground}(a,C)
=
\frac{|T(a)\cap T(C)|}
{|T(a)|}
$$

If:

$$
\text{ground}(a,C) < 0.10
$$

the answer is flagged as potentially under-grounded. The output guardrail also detects speculative phrases such as "I think", "probably", and Vietnamese equivalents.

## 21. Memory and Reinforced Recall

`src/memory.py` stores conversation history in SQLite and builds FAISS indexes for:

- Interaction memory.
- Reward memory for feedback with `reward >= 4`.

Memory recall uses cosine similarity:

$$
s_{\text{memory}}(q,m)
=
\hat{\mathbf{v}}_q^\top
\hat{\mathbf{v}}_m
$$

The interaction vector index is rebuilt lazily when:

$$
n_{\text{pending}}
\ge
R
$$

The default rebuild threshold is:

$$
R=10
$$

## 22. Overall Scoring Pipeline

A compact view of the final ranking path is:

$$
S_{\text{final}}(q,d)
=
\text{MMR}
\left(
\text{CE}
\left(
\text{EHRAG}
\left(
\text{Boost}
\left(
\text{Fusion}
\left(
s_{\text{dense}},
s_{\text{bm25}},
s_{\text{graph}}
\right)
\right)
\right)
\right)
\right)
$$

Some operators are optional. If a component is unavailable, the system skips it and keeps the best available score from earlier stages.

The practical consequence is a failure-open retrieval pipeline:

- If BM25 is missing, dense and graph retrieval still work.
- If the graph is missing, dense and BM25 retrieval still work.
- If the hypergraph is missing, EHRAG rescoring is skipped.
- If the reranker is disabled, fusion scores are used directly.
- If critic validation fails operationally, generation proceeds with the current context.

This design improves retrieval quality when all artifacts are available while keeping the application usable under partial configuration.
