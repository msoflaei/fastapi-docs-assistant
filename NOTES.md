# Development notes

## Data cleaning
- 156 documents processed, 459 code includes resolved.
- Known gap: how-to/configure-swagger-ui.md includes ../../fastapi/openapi/docs.py,
  which is FastAPI's library source rather than a docs example. Not downloaded,
  since the full file would add mostly unrelated code. Impact: one section of one page.
- Console (terminal output) blocks contained HTML used for coloring on the website.
  Stripped the tags and unescaped HTML entities inside console blocks only;
  real code blocks are never modified.

## Chunking
- Structure-aware: split at headings, then at blank lines if a section exceeds 350 words.
  Code blocks are never split.
- Each chunk starts with a heading breadcrumb and links to its exact section via the anchor.
- No overlap between chunks: since splits follow section boundaries, overlap adds little.
- First version: 2017 chunks, median 72 words, 427 under 30 words (21%).
  Fix: merge sections under 100 words into the following section (staying under 350).
- 40 chunks exceed 350 words (max 1470) because of long code examples; kept whole,
  since splitting code hurts more and the embedding model's limit is much higher.

## Embeddings
- Model: BAAI/bge-small-en-v1.5 (open-source, 384 dimensions, 512-token limit), run locally.
- Chunks longer than the token limit: [your number] of 1059 (truncated; mostly long code examples).
- Vectors normalized, so cosine similarity = dot product.
- Queries use the bge retrieval prefix; documents don't.

## Vector database
- PostgreSQL + pgvector, installed via conda-forge; data stored outside the repo (~/pgdata).
- Table `chunks` with a vector(384) column and an HNSW index (cosine distance).
- Database URL read from the DATABASE_URL environment variable, with a local default.
- Verified: database results match brute-force numpy search on test questions.

## Retrieval observations (to test with evals later)
- Release notes chunks appear in results (e.g. "make a parameter optional"); consider excluding them.
- Some queries return several chunks from the same page; consider limiting results per page.

## Generation
- Claude via the Anthropic SDK; model set by LLM_MODEL (default claude-haiku-5-5), temperature 0.
- System prompt rules: answer only from documents, cite as [n], admit when the answer isn't there.
- Retrieved chunks passed in XML <document> tags with their source URLs.
- API key in .env (git-ignored); .env.example documents the required settings.
- Responses can contain thinking blocks before the text; extract only blocks of type "text".
- max_tokens 2048 (thinking counts toward it); warn if stop_reason is "max_tokens".
- Out-of-scope test ("connect to MongoDB"): assistant correctly said the docs don't cover it,
  instead of hallucinating; retrieval returned weak matches, generation recognized them.

## API
- FastAPI app (src/docs_assistant/api.py) with GET /health, POST /search, POST /ask.
- Embedding model and Anthropic client loaded once at startup via lifespan.
- Pydantic validates input (question 3-500 chars, k 1-10).
- Errors: database down -> 503, LLM API failure -> 502.
- Sync endpoints, since retrieval and LLM calls are blocking; FastAPI runs them in a thread pool.

## Testing
- pytest; unit tests for cleaning, chunking, prompt building, and answer extraction.
- API tests use monkeypatch to replace the model, database, and LLM with fakes: fast, free, offline.
- Error paths tested: database down -> 503, LLM failure -> 502, invalid input -> 422.
- Regression tests for bugs found during development (console HTML, thinking blocks, temperature).
- Integration tests (real database + model) marked and run separately: pytest -m integration.
- Exact versions recorded in requirements-lock.txt.

## Evaluation dataset
- evals/questions.jsonl: [N] questions ([X] in scope, [Y] out of scope), hand-verified against the docs.
- Labels at page level (relevant_docs), so the dataset survives chunking changes.
- answer_points: key facts a correct answer must contain (used for answer grading).
- Validated by docs_assistant.eval_data (unknown page IDs, duplicates, missing fields).

## Retrieval evaluation (40 in-scope questions)

Metrics: hit@k (any relevant page in top k), recall@k (share of relevant pages in top k),
MRR (rewards ranking the first relevant result high).

| Run | hit@1 | hit@3 | hit@5 | recall@5 | hit@10 | MRR | ms/question |
|---|---|---|---|---|---|---|---|
| baseline | 0.750 | 0.875 | 0.950 | 0.912 | 0.950 | 0.825 | 64 |
| exclude release notes | 0.750 | 0.900 | 0.950 | 0.912 | 0.950 | 0.832 | 39 |
| exclude release notes + reference | 0.750 | 0.925 | 0.950 | 0.912 | 0.950 | 0.834 | 50 |
| exclude release notes + rerank | 0.700 | 0.925 | 0.950 | 0.938 | 1.000 | 0.819 | 642 |
| exclude both + rerank | 0.700 | 0.925 | 0.950 | 0.938 | 1.000 | 0.820 | 636 |
| final (exclude both, no rerank) | 0.750 | 0.925 | 0.950 | 0.912 | 0.950 | 0.834 | 44 |

Findings and decisions:
- Baseline was already strong (hit@5 0.95); the weak spot was ranking (hit@1 0.75).
- Excluding release notes: +1 question at hit@3 (with 40 questions, 1 question = 2.5 points,
  so not significant), no regressions, faster. Adopted; trade-off: can't answer
  version-history questions.
- Excluding reference pages (25 chunks): these are short pointer pages whose real content
  (API docs generated from source code) isn't in our data. Moved q15 from rank 4 to 3,
  nothing else changed. Adopted.
- Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2, 30 candidates): per-question comparison
  showed 12 of 40 questions changed, 5 better, 7 worse. Gains were large (q10 not found -> 2,
  q04 not found -> 8, q35 5 -> 1); most losses were small (rank 1 -> 2). Since the LLM reads
  all 5 chunks, hit@5/recall@5 matter more than hit@1/MRR for us. Not adopted: same hit@5,
  +0.026 recall@5 (about 1 question), 13x slower. Kept as an option (--rerank) for testing
  a larger reranker later.
- Hypothesis "excluding reference pages gives the reranker a fair second chance" was not
  confirmed: rerank results barely changed.
- Remaining misses (q04 "JSON in a POST request" -> Request Body; q10 "share common logic"
  -> Dependencies) are vocabulary mismatches. Hybrid (keyword) search deprioritized, since
  it helps when words match exactly, not when they differ. The agent (Stage 3) can rephrase
  and search again, which targets this directly.
- Final default config: vector search, k=5, excluding release-notes and reference/%.
  Defined in retrieve.DEFAULT_EXCLUDE and used by the API, generate.py, and the evaluation.
- To check: whether how-to/custom-docs-ui-assets should be an additional relevant page for q15.