"""Unit tests for prompt building and answer extraction, using a fake LLM client."""
from types import SimpleNamespace   # a quick way to build simple objects with attributes

from docs_assistant.generate import build_prompt, generate_answer

CHUNKS = [
    {"url": "https://a.example/", "content": "Alpha"},
    {"url": "https://b.example/", "content": "Beta"},
]


class FakeClient:
    """Pretends to be anthropic.Anthropic: client.messages.create(...) returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.calls = []             # records what our code sent, so tests can inspect it
        self.messages = self        # so that client.messages.create calls our create below

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def test_prompt_numbers_documents_and_ends_with_question():
    prompt = build_prompt("What is FastAPI?", CHUNKS)
    assert '<document index="1" source="https://a.example/">' in prompt
    assert '<document index="2" source="https://b.example/">' in prompt
    assert prompt.endswith("Question: What is FastAPI?")


def test_answer_skips_thinking_blocks():
    # Regression test: the first block can be "thinking", not the answer.
    response = make_response([
        SimpleNamespace(type="thinking", thinking="Let me look at the documents..."),
        SimpleNamespace(type="text", text="Use fastapi dev [1]."),
    ])
    assert generate_answer("q", CHUNKS, FakeClient(response)) == "Use fastapi dev [1]."


def test_cut_off_answer_gets_a_warning():
    response = make_response([SimpleNamespace(type="text", text="Partial")], stop_reason="max_tokens")
    assert "Warning" in generate_answer("q", CHUNKS, FakeClient(response))


def test_request_has_system_prompt_and_no_temperature():
    client = FakeClient(make_response([SimpleNamespace(type="text", text="ok")]))
    generate_answer("q", CHUNKS, client)
    sent = client.calls[0]
    assert "ONLY the information in the provided documents" in sent["system"]
    assert "temperature" not in sent        # regression test: newer SDKs reject it