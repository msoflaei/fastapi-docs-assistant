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