"""Connecting to the PostgreSQL database."""
import os                                       # to read environment variables

import psycopg                                  # the PostgreSQL driver for Python
from pgvector.psycopg import register_vector    # adds support for the vector type

# The database address. Reading it from an environment variable (with a default)
# means it can be changed later, e.g. for a cloud database, without editing code.
DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/docs_assistant")


def connect() -> psycopg.Connection:
    """Open a connection with vector support enabled."""
    conn = psycopg.connect(DB_URL, autocommit=True)   # autocommit: each statement is saved immediately
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")   # harmless if it already exists
    register_vector(conn)                       # lets us pass numpy arrays as vectors
    return conn