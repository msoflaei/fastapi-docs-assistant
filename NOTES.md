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