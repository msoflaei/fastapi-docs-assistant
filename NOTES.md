# Development notes

## Data cleaning
- 156 documents processed, 459 code includes resolved.
- Known gap: how-to/configure-swagger-ui.md includes ../../fastapi/openapi/docs.py,
  which is FastAPI's library source rather than a docs example. Not downloaded,
  since the full file would add mostly unrelated code. Impact: one section of one page.
- Console (terminal output) blocks contained HTML used for coloring on the website.
  Stripped the tags and unescaped HTML entities inside console blocks only;
  real code blocks are never modified.