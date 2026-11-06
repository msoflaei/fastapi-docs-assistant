"""API tests. The 'client' fixture (from conftest.py) uses fakes, so these are fast and free."""
import anthropic
import httpx
import psycopg

import docs_assistant.api as api

QUESTION = {"question": "How do I run my app?"}


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_numbered_sources(client):
    response = client.post("/ask", json=QUESTION)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use fastapi dev [1]."
    assert body["sources"] == [{
        "index": 1,
        "section": "First Steps",
        "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "score": 0.751,                    # rounded to 3 decimals
    }]
    assert body["model"] == api.MODEL


def test_search_returns_sources_without_answer(client):
    body = client.post("/search", json=QUESTION).json()
    assert "results" in body and "answer" not in body


def test_too_short_question_is_rejected(client):
    assert client.post("/ask", json={"question": "a"}).status_code == 422   # 422 = invalid input


def test_k_out_of_range_is_rejected(client):
    assert client.post("/ask", json={"question": "How do I run my app?", "k": 50}).status_code == 422


def test_database_down_returns_503(client, monkeypatch):
    def broken_connect():
        raise psycopg.OperationalError("connection refused")
    monkeypatch.setattr(api, "connect", broken_connect)
    assert client.post("/ask", json=QUESTION).status_code == 503


def test_llm_failure_returns_502(client, monkeypatch):
    def failing_generate(question, chunks, llm):
        raise anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
    monkeypatch.setattr(api, "generate_answer", failing_generate)
    assert client.post("/ask", json=QUESTION).status_code == 502


class FakeAgent:
    """Stands in for DocsAgent: returns a fixed result without calling Claude."""

    def __init__(self, embed_model, client, conn):
        pass

    def run(self, question):
        return {
            "answer": "Use dependencies with Depends() [1].",
            "sources": [{"index": 1, "section": "Dependencies",
                         "url": "https://fastapi.tiangolo.com/tutorial/dependencies/"}],
            "documents": [],
            "steps": [{"tool": "search_docs", "input": {"query": "dependencies"}}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "stopped": False,
        }


class FailingAgent(FakeAgent):
    def run(self, question):
        raise anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )


def test_agent_returns_answer_with_trace_and_usage(client, monkeypatch):
    monkeypatch.setattr(api, "DocsAgent", FakeAgent)
    response = client.post("/agent", json={"question": "How do I share logic?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use dependencies with Depends() [1]."
    assert body["steps"] == [{"tool": "search_docs", "input": {"query": "dependencies"}}]
    assert body["sources"][0]["index"] == 1
    assert body["usage"]["input_tokens"] == 100
    assert body["stopped"] is False


def test_agent_llm_failure_returns_502(client, monkeypatch):
    monkeypatch.setattr(api, "DocsAgent", FailingAgent)
    assert client.post("/agent", json={"question": "How do I share logic?"}).status_code == 502


def test_agent_too_short_question_is_rejected(client):
    assert client.post("/agent", json={"question": "a"}).status_code == 422