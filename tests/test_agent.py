"""Unit tests for the agent loop, using a scripted fake Claude client (no API calls)."""
from types import SimpleNamespace

import docs_assistant.agent as agent_mod
from docs_assistant.agent import MAX_STEPS, DocsAgent

FAKE_CHUNK = {
    "chunk_id": "fastapi-cli#0",
    "section": "FastAPI CLI",
    "url": "https://fastapi.tiangolo.com/fastapi-cli/",
    "content": "Run your app with fastapi dev.",
    "score": 0.8,
}


class ScriptedClient:
    """Pretends to be the Anthropic client, returning prepared responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5)


def tool_call(name, args, call_id="t1"):
    block = SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)
    return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage())


def final_answer(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           stop_reason="end_turn", usage=usage())


def test_agent_searches_then_answers(monkeypatch):
    monkeypatch.setattr(agent_mod, "retrieve", lambda query, model, conn, k: [FAKE_CHUNK])
    client = ScriptedClient([tool_call("search_docs", {"query": "run app"}), final_answer("Use fastapi dev [1].")])

    result = DocsAgent("fake-model", client, conn=None).run("How do I run my app?")

    assert result["answer"] == "Use fastapi dev [1]."
    assert result["steps"] == [{"tool": "search_docs", "input": {"query": "run app"}}]
    assert result["sources"] == [{"index": 1, "section": "FastAPI CLI", "url": FAKE_CHUNK["url"]}]
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 10}
    assert result["stopped"] is False
    # The tool result went back to Claude with the matching id and a numbered document.
    tool_result = client.calls[1]["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "t1"
    assert '<document index="1"' in tool_result["content"]


def test_same_chunk_keeps_its_number_across_searches(monkeypatch):
    monkeypatch.setattr(agent_mod, "retrieve", lambda query, model, conn, k: [FAKE_CHUNK])
    client = ScriptedClient([tool_call("search_docs", {"query": "a"}, "t1"),
                             tool_call("search_docs", {"query": "b"}, "t2"),
                             final_answer("done [1]")])
    result = DocsAgent("fake-model", client, conn=None).run("q")
    assert len(result["sources"]) == 1          # found twice, numbered once


def test_step_limit_stops_the_loop(monkeypatch):
    monkeypatch.setattr(agent_mod, "retrieve", lambda query, model, conn, k: [FAKE_CHUNK])
    client = ScriptedClient([tool_call("search_docs", {"query": "again"})] * MAX_STEPS)
    result = DocsAgent("fake-model", client, conn=None).run("q")
    assert result["stopped"] is True
    assert len(result["steps"]) == MAX_STEPS


def test_tool_errors_are_returned_to_claude_not_raised():
    client = ScriptedClient([tool_call("delete_everything", {}), final_answer("Sorry.")])
    result = DocsAgent("fake-model", client, conn=None).run("q")
    assert result["answer"] == "Sorry."
    tool_result = client.calls[1]["messages"][2]["content"][0]
    assert tool_result["content"].startswith("Error: unknown tool")