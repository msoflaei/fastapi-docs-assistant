"""Answer questions using retrieved chunks and an LLM (the "generation" in RAG)."""
import os
import sys

from anthropic import Anthropic         # the official Claude SDK
from dotenv import load_dotenv          # reads settings from the .env file

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.retrieve import retrieve

load_dotenv()   # copies the values in .env into environment variables, where the SDK finds the key

# Which model to use. Haiku is fast and cheap, good while developing; it can be
# switched (e.g. to claude-sonnet-5-5) by setting LLM_MODEL in .env, with no code changes.
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")

# The system prompt: standing instructions that shape every answer.
# Each rule targets a known RAG failure mode.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the FastAPI web framework.

Rules:
- Answer using ONLY the information in the provided documents. Do not rely on outside knowledge.
- If the documents don't contain the answer, say "I couldn't find this in the FastAPI documentation." and stop.
- Cite the documents you used with their number in square brackets, like [1] or [2][3], right after the sentence they support.
- Include short code examples from the documents when they help answer the question.
- Be concise and practical."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Put the retrieved chunks and the question into one message.

    XML-style tags clearly separate the documents from the question, which
    helps the model tell instructions apart from data.
    """
    docs = "\n\n".join(
        f'<document index="{i}" source="{c["url"]}">\n{c["content"]}\n</document>'
        for i, c in enumerate(chunks, start=1)      # number the documents 1, 2, 3...
    )
    return f"<documents>\n{docs}\n</documents>\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[dict], client: Anthropic) -> str:
    """Send the prompt to the LLM and return its answer text."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,         # upper limit on output length; thinking counts toward it too
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    # The response is a list of content blocks of different types (e.g. "thinking", then "text").
    # Keep only the text blocks and join them, rather than assuming the first block is the answer.
    answer = "".join(block.text for block in response.content if block.type == "text")

    # If the model hit the length limit, the answer may be cut off; warn so we notice.
    if response.stop_reason == "max_tokens":
        answer += "\n\n[Warning: answer was cut off because it reached max_tokens]"
    return answer


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I run my app in development mode?"
    model = load_model()
    client = Anthropic()                 # reads ANTHROPIC_API_KEY from the environment

    with connect() as conn:
        chunks = retrieve(question, model, conn, k=5)    # step 1: retrieval
    answer = generate_answer(question, chunks, client)   # step 2: generation

    print(f"Question: {question}\n")
    print(answer)
    print("\nSources:")
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] {c['section']}\n    {c['url']}")


if __name__ == "__main__":
    main()