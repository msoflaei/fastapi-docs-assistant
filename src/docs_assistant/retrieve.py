"""Retrieve the most relevant chunks for a question from PostgreSQL."""
import sys

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.search import QUERY_PREFIX

# <=> is pgvector's cosine DISTANCE operator (0 = same direction, 2 = opposite).
# 1 - distance gives cosine SIMILARITY, the same score search.py printed.
# ORDER BY distance ... LIMIT k returns the k closest chunks, using the HNSW index.
QUERY = """
SELECT chunk_id, section, url, content, 1 - (embedding <=> %(q)s) AS score
FROM chunks
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""


def retrieve(query: str, model, conn, k: int = 5) -> list[dict]:
    """Return the k most relevant chunks as dictionaries."""
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    rows = conn.execute(QUERY, {"q": q, "k": k}).fetchall()
    columns = ["chunk_id", "section", "url", "content", "score"]
    return [dict(zip(columns, row)) for row in rows]   # turn each row into a dictionary


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