"""Integration tests: use the real database and embedding model.

Skipped by default. Run with:  pytest -m integration   (PostgreSQL must be running)
"""
import pytest

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.retrieve import retrieve

pytestmark = pytest.mark.integration    # marks every test in this file as "integration"


@pytest.fixture(scope="module")         # load the model once for all tests in this file
def model():
    return load_model()


def test_dev_mode_question_retrieves_cli_docs(model):
    with connect() as conn:
        results = retrieve("How do I run my app in development mode?", model, conn, k=5)
    assert len(results) == 5
    assert any("fastapi-cli" in r["url"] for r in results)


def test_security_question_retrieves_security_docs(model):
    with connect() as conn:
        results = retrieve("How do I protect my endpoints with a password?", model, conn, k=5)
    assert any("/security/" in r["url"] for r in results)