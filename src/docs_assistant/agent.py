"""An agent that answers FastAPI questions by deciding itself what to search and read.

Claude runs in a loop with two tools (search_docs, read_page). Each turn it either
asks for a tool, which our code runs, or gives its final answer.
"""
import sys

from anthropic import Anthropic

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.generate import MODEL, response_text      # also loads .env
from docs_assistant.retrieve import retrieve

MAX_STEPS = 6            # guardrail: at most this many model calls per question
SEARCH_K = 5             # chunks returned per search (the agent can search several times)
MAX_PAGE_CHARS = 15000   # guardrail: very long pages are truncated to limit cost

# Tool definitions: a name, a description telling Claude what the tool does and when
# to use it, and a JSON schema describing its inputs.
TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Search the official FastAPI documentation by meaning. Returns the most relevant "
            "sections, each with a document number for citing and the doc_id of its page. "
            "If the results don't answer the question, search again with different words, "
            "preferably the technical terms the FastAPI docs are likely to use."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for: a few words or a short question."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": (
            "Read a whole FastAPI documentation page by its doc_id (shown in search results). "
            "Use it when a search result comes from the right page but doesn't include the "
            "details or code example you need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "The page's doc_id, e.g. 'tutorial/first-steps'."}
            },
            "required": ["doc_id"],
        },
    },
]

AGENT_SYSTEM = """You are a helpful assistant that answers questions about the FastAPI web framework, using tools to look things up in the official FastAPI documentation.

How to work:
- Always search the documentation before answering. Never answer from memory.
- If the results don't answer the question, search again with different words. The docs often use their own technical terms rather than everyday words (for example, "path operation" for an endpoint), so try the terms the docs are likely to use.
- If a result comes from the right page but lacks the details or code you need, use read_page on that page.
- Be efficient: usually 1 to 3 tool calls are enough. Stop once you have what you need.

How to answer:
- Answer using ONLY information from the tool results.
- Cite the document numbers from the tool results in square brackets, like [1] or [2][3], right after the sentence they support.
- If the documentation doesn't contain the answer after a reasonable search, say "I couldn't find this in the FastAPI documentation." and stop.
- Include short code examples from the documents when they help. Be concise and practical."""


class DocsAgent:
    """Holds the shared resources and runs the agent loop for one question at a time."""

    def __init__(self, embed_model, client: Anthropic, conn, search_k: int = SEARCH_K):
        self.embed_model = embed_model
        self.client = client
        self.conn = conn
        self.search_k = search_k
        self.sources: list[dict] = []      # every document shown to Claude, numbered from 1
        self._source_ids: dict[str, int] = {}

    # ---- Source numbering: the same chunk or page always keeps the same number ----

    def add_source(self, key: str, section: str, url: str) -> int:
        if key not in self._source_ids:
            self.sources.append({"index": len(self.sources) + 1, "section": section, "url": url})
            self._source_ids[key] = len(self.sources)
        return self._source_ids[key]

    # ---- Tools ----

    def search_docs(self, query: str) -> str:
        chunks = retrieve(query, self.embed_model, self.conn, k=self.search_k)
        parts = []
        for c in chunks:
            doc_id = c["chunk_id"].split("#")[0]
            n = self.add_source(c["chunk_id"], c["section"], c["url"])
            parts.append(f'<document index="{n}" doc_id="{doc_id}" source="{c["url"]}">\n{c["content"]}\n</document>')
        return "\n\n".join(parts) or "No results."

    def read_page(self, doc_id: str) -> str:
        rows = self.conn.execute("SELECT chunk_id, url, content FROM chunks WHERE doc_id = %s", (doc_id,)).fetchall()
        if not rows:
            return f"No page with doc_id '{doc_id}'. Use a doc_id exactly as shown in search results."
        rows.sort(key=lambda r: int(r[0].split("#")[1]))    # put the chunks back in page order
        page_url = rows[0][1].split("#")[0]
        text = "\n\n".join(r[2] for r in rows)
        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + "\n\n[Page truncated]"
        n = self.add_source(f"page:{doc_id}", f"{doc_id} (full page)", page_url)
        return f'<document index="{n}" doc_id="{doc_id}" source="{page_url}">\n{text}\n</document>'

    def run_tool(self, name: str, args: dict) -> str:
        """Run a tool. Errors are returned to Claude as text instead of crashing the program,
        so it can correct itself (e.g. retry with a valid doc_id)."""
        try:
            if name == "search_docs":
                return self.search_docs(args["query"])
            if name == "read_page":
                return self.read_page(args["doc_id"])
            return f"Error: unknown tool '{name}'."
        except Exception as e:
            return f"Error running {name}: {type(e).__name__}: {e}"

    # ---- The agent loop ----

    def run(self, question: str) -> dict:
        self.sources, self._source_ids = [], {}
        messages = [{"role": "user", "content": question}]
        steps = []                                           # the trace: every tool call made
        usage = {"input_tokens": 0, "output_tokens": 0}     # to measure cost

        for _ in range(MAX_STEPS):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=AGENT_SYSTEM,
                tools=TOOLS,
                messages=messages,
            )
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
            # Keep Claude's whole reply (including any thinking and tool requests) in the history.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":          # no tool requested: this is the final answer
                return {"answer": response_text(response), "sources": self.sources,
                        "steps": steps, "usage": usage, "stopped": False}

            # Run every tool Claude asked for, and send all results back in one message.
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    steps.append({"tool": block.name, "input": block.input})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": self.run_tool(block.name, block.input)})
            messages.append({"role": "user", "content": results})

        # Guardrail: the step limit was reached without a final answer.
        return {"answer": "I couldn't complete this question within the step limit.",
                "sources": self.sources, "steps": steps, "usage": usage, "stopped": True}


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I accept JSON data in a POST request?"
    with connect() as conn:
        agent = DocsAgent(load_model(), Anthropic(), conn)
        result = agent.run(question)

    print(f"Question: {question}\n")
    print("Steps taken:")
    for i, step in enumerate(result["steps"], start=1):
        print(f"  {i}. {step['tool']}({step['input']})")
    print(f"\n{result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"[{s['index']}] {s['section']}\n    {s['url']}")
    print(f"\nTokens: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out"
          f"{'   (STOPPED at step limit)' if result['stopped'] else ''}")


if __name__ == "__main__":
    main()