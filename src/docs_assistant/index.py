"""Load chunks and their embeddings into PostgreSQL."""
import numpy as np

from docs_assistant.db import connect
from docs_assistant.embed import EMB_FILE, load_chunks

# The table: one row per chunk. vector(384) is pgvector's column type for
# a 384-number vector, matching our embedding model.
CREATE_TABLE = """
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    chunk_id  text PRIMARY KEY,   -- e.g. "tutorial/first-steps#1", must be unique
    doc_id    text NOT NULL,      -- the page the chunk came from
    title     text,               -- the page title
    section   text,               -- the breadcrumb, e.g. "First Steps > Check it"
    url       text,               -- link to the exact section
    content   text NOT NULL,      -- the chunk's text
    n_words   integer,
    embedding vector(384) NOT NULL
);
"""

# An HNSW index makes similarity search fast: instead of comparing the question
# with every row, it navigates a graph of vectors to find close ones quickly.
# vector_cosine_ops tells it we measure closeness with cosine distance.
CREATE_INDEX = "CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);"

INSERT = """
INSERT INTO chunks (chunk_id, doc_id, title, section, url, content, n_words, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""
# The %s are placeholders: psycopg fills them in safely with the values we pass.
# Never build SQL by pasting values into the string yourself; that invites SQL injection.


def main() -> None:
    chunks = load_chunks()
    vectors = np.load(EMB_FILE)
    # Safety check: row i of the vectors must belong to chunk i.
    assert len(chunks) == len(vectors), "chunks and embeddings are out of sync; rerun embed.py"

    rows = [
        (c["chunk_id"], c["doc_id"], c["title"], c["section"], c["url"], c["text"], c["n_words"], v)
        for c, v in zip(chunks, vectors)     # zip pairs each chunk with its vector
    ]

    with connect() as conn:                  # "with" closes the connection when done
        conn.execute(CREATE_TABLE)           # start fresh each time we re-index
        with conn.cursor() as cur:
            cur.executemany(INSERT, rows)    # insert all rows
        conn.execute(CREATE_INDEX)           # build the index after inserting (faster)
        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    print(f"Indexed {count} chunks into PostgreSQL")


if __name__ == "__main__":
    main()