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