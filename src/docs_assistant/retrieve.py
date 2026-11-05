"""Retrieve the most relevant chunks for a question from PostgreSQL."""
import sys

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.search import QUERY_PREFIX


# Pages excluded from search by default, chosen by retrieval evaluation (see NOTES.md):
# release notes mention many features briefly; reference pages are thin pointers in our data.
DEFAULT_EXCLUDE = ("release-notes", "reference/%")

# Number of chunks given to the LLM, chosen by answer evaluation (see NOTES.md):
# k=8 beat k=5 on correctness in all 3 repeated runs (+2.5 points on average).
DEFAULT_K = 8

# <=> is pgvector's cosine DISTANCE operator; 1 - distance is cosine SIMILARITY.
# The WHERE clause filters out excluded pages while searching (a metadata filter).
# ::text[] tells PostgreSQL the parameter is a list of text values.
QUERY = """
SELECT chunk_id, section, url, content, 1 - (embedding <=> %(q)s) AS score
FROM chunks
WHERE NOT (doc_id LIKE ANY(%(exclude)s::text[]))
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""


def retrieve(query: str, model, conn, k: int = 5, exclude_docs: list[str] | tuple = DEFAULT_EXCLUDE) -> list[dict]:
    """Return the k most relevant chunks as dictionaries, skipping pages in exclude_docs."""
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    rows = conn.execute(QUERY, {"q": q, "k": k, "exclude": list(exclude_docs)}).fetchall()
    columns = ["chunk_id", "section", "url", "content", "score"]
    return [dict(zip(columns, row)) for row in rows]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How do I run my app in development mode?"
    model = load_model()
    with connect() as conn:
        results = retrieve(query, model, conn)

    print(f"Question: {query}\n")
    for r in results:
        print(f"{r['score']:.3f}  {r['section']}")
        print(f"       {r['url']}")


if __name__ == "__main__":
    main()