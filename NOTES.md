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
- evals/questions.jsonl: 45 questions (40 in scope, 5 out of scope), hand-verified against the docs.
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

## Answer evaluation

Method:
- 45 questions answered (Haiku, k=5), graded by an LLM judge (Sonnet) plus code checks.
- Code checks: refusal detection, citation validity (code blocks excluded).
- Judge: answer-point coverage and faithfulness (yes/partial/no), JSON output, retries on bad output.
- Judge validation: 8 of 10 manually reviewed answers agreed with the judge.

Baseline results (Haiku, k=5):
- Fully correct 85% (34/40), point coverage 86.3%, faithful 87.5% (5 partial, 0 unfaithful),
  out-of-scope refused 5/5, 0 invalid citations, 2 false refusals, 3.0 s/answer.
- Failures fall into three groups:
  A. Retrieval misses (q04, q10): honest refusal or partial answer. Same as retrieval eval misses.
  B. Right page, wrong chunk (q05, q09, q19): explanation chunk retrieved, code chunk not.
     Invisible to page-level retrieval metrics -> test k=8.
  C. Faithfulness slips (q01, q03, q15, q39): correct facts added from model knowledge,
     not from the documents -> candidate for a stricter prompt.
- q24 label demanded "version", which the question didn't ask for -> label fixed, baseline rerun.

Experiments:
- baseline-v2 (q24 label fixed, otherwise identical): fully correct 87.5%, coverage 88.7%,
  faithful 80% (8 partial), 2 false refusals, 1 invalid citation, 2.92 s.
  Faithfulness changed on 8 questions with no system change -> run-to-run variance from
  non-deterministic generation and judging. Faithfulness differences of a few points are noise.
- k=8: fully correct 92.5%, coverage 92.5%, faithful 85% (6 partial), 1 false refusal,
  0 invalid citations, 3.1 s. Fixed q05 and q19 (Group B), as predicted; q09 still fails.
  Correctness gain matches the hypothesis -> adopt k=8, pending a repeat run.
- To do: repeat k8 run to confirm; validate judge on 10 answers.
- Repeated runs (3 each):
  | Metric | k=5 mean (range) | k=8 mean (range) |
  |---|---|---|
  | Fully correct | 0.883 (0.875-0.900) | 0.908 (0.900-0.925) |
  | Point coverage | 0.895 (0.887-0.912) | 0.908 (0.900-0.925) |
  | Faithful | 0.833 (0.800-0.875) | 0.817 (0.750-0.850) |
  | False refusals | 1.67 (1-2) | 0.67 (0-1) |
- k=8 at least as good as k=5 on correctness in every run; average gain +2.5 points (~1 question),
  half of what the single run suggested (+5). Faithfulness difference within noise.
- Decision: k=8 adopted as DEFAULT_K (used by the API, CLI, and evals).
  Trade-off: about 60% more input tokens per question.

## Continuous integration
- GitHub Actions (.github/workflows/tests.yml): unit and API tests on every push and pull request.
- CPU-only PyTorch on CI to avoid multi-GB GPU builds; versions constrained by requirements-lock.txt.
- No secrets or database needed: the LLM, model, and database are mocked in these tests.
- FastAPI docs pinned to commit [first 7 characters of your commit ID] in download_data.sh;
  full rebuild reproduced identical results (156 docs, 1059 chunks, hit@5 0.950).
- retrieval-eval workflow: rebuilds the index from scratch on CI with a pgvector service
  container, runs integration tests, then the retrieval eval with a quality gate (hit@5 >= 0.90).
  Results uploaded as an artifact.
- Answer evaluation stays manual: it costs money and varies between runs.

## Agent
- DocsAgent: Claude in a tool-use loop with search_docs(query) and read_page(doc_id).
- Guardrails: max 6 model calls per question, tool errors returned to the model as text,
  long pages truncated at 15,000 characters.
- Trace (every tool call) and token usage recorded per question, for evaluation.
- Source numbers stay stable across multiple searches, so citations remain valid.
- System prompt kept general: no hints targeting specific eval questions (avoids overfitting).
- First manual test (4 questions): q04 fixed (agent added "body" to its query, then read the
  Request Body page); q10 mostly fixed (found Dependencies, but led with query models);
  q23 answered in 1 search; MongoDB: correct conclusion after 3 searches (20.6k input tokens).
- Problems found: (1) cited doc_ids like [tutorial/body] instead of numbers, because tool output
  shows both index and doc_id; (2) out-of-scope answer reworded the refusal and added advice
  from outside knowledge. Fixed by making the citation and refusal rules explicit in the prompt.
- After the prompt fix: all 4 answers cited numbers; MongoDB answer gave the exact refusal plus
  one cited sentence, in 2 searches (13.8k tokens, down from 20.6k). q10 now shows a dependency
  class with Depends(). The agent's path varies between runs (q04: search + read_page, then 2 searches).

## Agent evaluation (45 questions, judge Sonnet)

| Metric | RAG k=8 (3 runs) | Agent r1 | Agent r2 |
|---|---|---|---|
| Fully correct | 0.908 (0.900-0.925) | 1.000 | 1.000 |
| Point coverage | 0.908 (0.900-0.925) | 1.000 | 1.000 |
| Faithful | 0.817 (0.750-0.850) | 0.850 | 0.700 |
| False refusals | 0.67 | 0 | 0 |
| Out-of-scope refused | 5/5 | 5/5 | 5/5 |
| Avg tool calls | 1 (fixed) | 1.62 | 1.56 |
| Avg input tokens | ~3,200 (estimated, chars/4) | 7,197 | 6,890 |
| Avg seconds | ~3 | 4.24 | 3.89 |

Findings:
- Correctness: the agent fixed every remaining RAG failure (q04, q10: vocabulary mismatch;
  q05, q09, q19: right page, wrong chunk) and was 100% fully correct in both runs.
  A large, consistent gain (~4 questions).
- Faithfulness: not improved (mean 0.775 vs 0.817, within noise). Longer answers combining
  more documents give more room for small slips (misattributed citations, simplified code,
  calling something "recommended"). q03's "convert and validate" slip appears with both RAG
  and the agent: it's the answering model's habit, not a retrieval issue.
- Efficiency: ~1.6 tool calls per question, never hit the step limit, ~1 s slower than RAG.
- Cost: agent ~7,000 input tokens/question (measured) vs RAG ~3,200 (estimated with
  scripts/estimate_rag_tokens.py) -> ~2.2x. Caveats: chars/4 is approximate (code may use
  more tokens per character); output tokens not compared, and the agent writes longer
  answers (~600 output tokens).
- 100% correctness means the eval set no longer discriminates between agent versions:
  harder, multi-part questions are needed for future comparisons.
- Limitations: a third agent run stopped at q32 when API credit ran out; the eval script
  saves results only at the end, so partial results were lost (improvement: save
  incrementally). RAG tokens estimated rather than measured, since credit wasn't topped up.

Conclusion:
- The agent trades ~2x cost and ~1 s latency for about +9 points of correctness (100% vs ~91%).
  Both are offered in the API: /ask (fast, cheap) and /agent (thorough). The /agent response
  includes the trace (tool calls) and token usage. Tested with a mocked agent.


## Framework comparison: LlamaIndex (retrieval evaluation, 40 in-scope questions)
| | Hand-built (merged) | Hand-built (no merge) | LlamaIndex Markdown | LlamaIndex defaults (1024) |
|---|---|---|---|---|
| Chunks | 1,059 | 2,017 | 1,156 | 381 |
| Median words | 168 | 72 | 91 | 451 |
| hit@1 | 0.750 | 0.750 | 0.850 | 0.750 |
| hit@5 | 0.950 | 0.950 | 0.950 | 0.950 |
| recall@5 | 0.912 | 0.938 | 0.938 | 0.912 |
| MRR | 0.834 | 0.847 | 0.896 | 0.841 |
- Same cleaned docs, embedding model, query prefix, database, and exclusions; LlamaIndex did
  chunking, indexing, and retrieval (~45 lines vs 338).
- LlamaIndex Markdown matched hit@5 and ranked better (hit@1 +0.10 = 4 questions; MRR +0.062).
- Hypothesis "our merging of small sections hurts ranking": only partly confirmed. No-merge
  improved MRR by 0.013 and recall@5 to 0.938, but hit@1 unchanged. The remaining gap is likely
  from different splitting (2,017 vs 1,156 chunks) and metadata embedding; not isolated further.
- The merge decision (Step 13) was made from size statistics before the eval set existed;
  re-testing it with retrieval metrics showed a small cost.
- LlamaIndex defaults: 1,024-token chunks exceed the model's 512-token limit (median 451 words),
  so chunk endings are invisible to search. Page-level metrics favor big chunks.
- Other framework issues: metadata embedded by default, API differs from tutorials (hnsw_kwargs),
  no section-level citation links, heading anchors left in text, many extra dependencies.
- Not adopted: chunking changes would need answer evaluation (smaller chunks may bring back
  "right page, wrong chunk"), which requires API credit. Promising lead for future work.
- Main index rebuilt afterwards; final-check matched final exactly.