# Track C — Advanced RAG System (`dotkonnekt` Technical Assignment)

An advanced Retrieval-Augmented Generation (RAG) Q&A system built over a 15-document knowledge base (`documents/01_rag_fundamentals.txt` to `15_corrective_retrieval.txt`). This project implements **all 5 Track C advanced RAG techniques**, persistent multi-turn conversation memory, LLM call telemetry (prompt, tokens, latency), and an interactive Streamlit UI.

---

## 1. Architecture Notes & Query Flow

```text
                [15 .txt Documents in documents/]
                                │
             (Startup: Sliding-Window Chunking — 173 Chunks)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
      [Dense Vector Index]            [BM25 Keyword Index]
     (all-MiniLM-L6-v2)                  (rank-bm25)
                 └──────────────┬──────────────┘
                                │
================================│================================
                                ▼
1. USER QUERY + SESSION MEMORY (src/state.py)
   • Persists multi-turn chat history, retrieval traces, and telemetry in memory
                                │
                                ▼
2. QUERY TRANSFORMATION (src/prompts.py, src/pipeline.py)
   • Resolves conversational pronouns ("those results", "it") using chat history
   • Decomposes multi-part questions into 1–3 standalone search sub-queries
                                │
                                ▼
3. HYBRID RETRIEVAL + RRF FUSION (src/retriever.py)
   • Executes Cosine Similarity Vector Search + BM25 Lexical Search per sub-query
   • Merges rankings via Reciprocal Rank Fusion: RRF_score = sum(1 / (60 + rank))
                                │
                                ▼
4. CROSS-ENCODER RE-RANKING (src/reranker.py)
   • Scores top candidate pairs jointly using `cross-encoder/ms-marco-MiniLM-L-6-v2`
   • Selects Top-4 highest-scoring chunks
                                │
                                ▼
5. CORRECTIVE LOOP / CRAG (src/pipeline.py)
   • Grades retrieved context as SUFFICIENT or WEAK
   • If WEAK: automatically generates a broadened retry query and re-runs retrieval + re-ranking
                                │
                                ▼
6. GROUNDED GENERATION & ABSTENTION (src/pipeline.py)
   • Synthesizes answers strictly from verified chunks with inline [filename.txt] citations
   • Abstains cleanly when evidence is missing from the corpus
```

### Model & Stack Choices
* **Hosted LLM (`openai/gpt-oss-120b` via Groq API):** Because an advanced Corrective RAG pipeline makes 3 LLM calls per turn (Query Transformation $\rightarrow$ CRAG Context Grading $\rightarrow$ Grounded Generation), low time-to-first-token latency and reliable JSON-mode adherence are critical. Running `openai/gpt-oss-120b` on Groq's LPU inference engine delivers sub-second structured JSON reasoning.
* **Dense Embeddings (`sentence-transformers/all-MiniLM-L6-v2`):** Provides fast local semantic embeddings normalized for cosine similarity without external network latency.
* **Sparse Keyword Index (`rank-bm25`):** Ensures exact lexical matching on technical terms (`RRF`, `BM25`, `DiskANN`, `1536`, `3072`).
* **Re-Ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** Performs full cross-attention over `(query, chunk_text)` pairs locally to reorder RRF candidates before hitting the LLM context window.

---

## 2. Prompt Design Writeup

All prompts are modularized in `src/prompts.py` and executed with `temperature=0.0` and `response_format={"type": "json_object"}` to guarantee deterministic, parseable outputs:

1. **`QUERY_TRANSFORM_SYSTEM_PROMPT`:**
   * **Structure & Rationale:** Multi-turn RAG fails when users ask follow-up questions with pronouns (e.g., *"How does RRF combine those search results?"*) or bundle two distinct topics into one prompt. This prompt injects the last 3 turns of conversation history, requires a `reasoning` trace first, resolves coreferences into explicit nouns, and splits compound questions into up to 3 atomic sub-queries while preserving technical keywords.
2. **`CRAG_GRADER_SYSTEM_PROMPT`:**
   * **Structure & Rationale:** Separates *context evaluation* from *answer generation*. The prompt asks the model to audit whether the top re-ranked chunks cover all parts of the user question (`SUFFICIENT` vs. `WEAK`). If any aspect is missing, it outputs a `rewritten_query` with broader terminology so `src/pipeline.py` can execute a corrective retrieval pass before generating the final answer.
3. **`GROUNDED_GENERATOR_SYSTEM_PROMPT`:**
   * **Structure & Rationale:** Enforces a strict closed-world contract:
     * Prohibits external pre-trained knowledge.
     * Mandates inline `[filename.txt]` citations on every claim.
     * Handles partial coverage honestly: if a multi-part question only has evidence for Part A, it answers Part A with citations and explicitly states that Part B is not covered in the documents (demonstrated in Turn 2 of `sample_transcript.txt`).
     * Requires explicit abstention (`"abstained": true`) when no relevant chunks exist (demonstrated in Turn 4 of `sample_transcript.txt`).

---

## 3. Setup & How to Reproduce the Sample Transcript

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install sentence-transformers streamlit
```

### 2. Configure Environment Variable
Create a `.env` file in the root directory:
```ini
GROQ_API_KEY=gsk_your_api_key_here
```

### 3. Reproduce the Sample Transcript (`sample_transcript.txt`)
```bash
python main.py --demo
```
This runs the 4-turn benchmark conversation and writes the complete session transcript—including decomposed queries, CRAG retries, re-ranker scores, citations, and per-call LLM token/latency telemetry—to `sample_transcript.txt`.

### 4. Run Interactive CLI
```bash
python main.py
```
* Type `/state` to inspect the full in-memory session state as JSON.
* Type `/transcript` to print the formatted multi-turn transcript.
* Type `/exit` to quit.

### 5. Launch Bonus Streamlit Web UI
```bash
streamlit run app.py
```

---

## 4. Known Limitations & Failure Modes

1. **Fixed Word-Window Chunking vs. Semantic Boundaries:**
   * `src/retriever.py` chunks documents using a 180-word sliding window with a 35-word overlap. If a long table or numbered list spans more than 180 words, a list item can be separated from its header context.
2. **Global Corpus Aggregation Queries:**
   * Because the Cross-Encoder narrows candidates down to `RERANK_TOP_K = 4` chunks, broad meta-questions such as *"Summarize all 15 files"* or *"How many documents mention embeddings?"* will only see the top 4 chunks rather than the entire 173-chunk index.
3. **Single-Pass Corrective Loop Cap:**
   * To bound latency and prevent infinite loops on out-of-scope questions, the CRAG loop executes at most 1 corrective re-query when context is graded `WEAK`. If a concept requires 3+ hops across obscure synonyms, a single retry may still miss it and trigger a partial or full abstention.
4. **In-Memory Index Rebuilt on Startup:**
   * Embeddings for the 173 chunks are computed in memory when `AdvancedRAGPipeline` initializes (~2 seconds). For larger corpora (10,000+ documents), vectors should be persisted to disk (e.g., Qdrant, Chroma, or FAISS).