"""Shared test fixtures."""
import pytest
from fastapi.testclient import TestClient

import docs_assistant.api as api

FAKE_CHUNKS = [
    {
        "chunk_id": "tutorial/first-steps#0",
        "section": "First Steps",
        "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "content": "Run your app with fastapi dev.",
        "score": 0.75123,
    }
]


class FakeConn:
    """Stands in for a database connection; supports 'with' and execute()."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):
        return self


@pytest.fixture
def client(monkeypatch):
    """A test client for the API, with the model, database, and LLM all replaced by fakes."""
    monkeypatch.setattr(api, "load_model", lambda: "fake-model")
    monkeypatch.setattr(api.anthropic, "Anthropic", lambda: "fake-llm-client")
    monkeypatch.setattr(api, "connect", lambda: FakeConn())
    monkeypatch.setattr(api, "retrieve", lambda question, model, conn, k: FAKE_CHUNKS[:k])
    monkeypatch.setattr(api, "generate_answer", lambda question, chunks, llm: "Use fastapi dev [1].")
    # Using TestClient in a "with" block runs the app's startup (lifespan) code.
    with TestClient(api.app) as test_client:
        yield test_client       # the test runs here; afterwards monkeypatch restores everything