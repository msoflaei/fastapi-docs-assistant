"""A web API for the docs assistant, built with FastAPI."""
from contextlib import asynccontextmanager

import anthropic
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.generate import MODEL, generate_answer   # importing generate also loads .env
from docs_assistant.retrieve import DEFAULT_K, retrieve

# Shared resources, created once when the server starts. Loading the embedding
# model takes seconds, so we must not do it on every request.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Code before 'yield' runs at startup; code after it runs at shutdown."""
    state["model"] = load_model()
    state["client"] = anthropic.Anthropic()
    yield
    state.clear()


app = FastAPI(
    title="FastAPI Docs Assistant",
    description="Answers questions about FastAPI using retrieval-augmented generation.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request and response shapes. Pydantic validates incoming data automatically:
# a question that's empty or too long is rejected with a clear error before
# our code even runs. These models also generate the interactive docs.
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500,
                          examples=["How do I run my app in development mode?"])
    k: int = Field(default=DEFAULT_K, ge=1, le=10, description="How many chunks to retrieve")


class Source(BaseModel):
    index: int          # the number the answer uses in its citations, e.g. [1]
    section: str
    url: str
    score: float        # similarity to the question


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str


class SearchResponse(BaseModel):
    results: list[Source]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_chunks(req: AskRequest) -> list[dict]:
    """Retrieve chunks, turning a database failure into a clear HTTP error."""
    try:
        with connect() as conn:
            return retrieve(req.question, state["model"], conn, k=req.k)
    except psycopg.OperationalError:
        # 503 = Service Unavailable: the database isn't reachable (is the server started?)
        raise HTTPException(status_code=503, detail="Database unavailable")


def to_sources(chunks: list[dict]) -> list[Source]:
    return [
        Source(index=i, section=c["section"], url=c["url"], score=round(c["score"], 3))
        for i, c in enumerate(chunks, start=1)
    ]


# ---------------------------------------------------------------------------
# Endpoints. They're plain "def" functions (not "async def") because our
# database and LLM calls are blocking; FastAPI runs them in a thread pool,
# so one slow request doesn't freeze the others.
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Check that the service and the database are up."""
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: AskRequest) -> SearchResponse:
    """Return the most relevant documentation sections, without generating an answer."""
    return SearchResponse(results=to_sources(get_chunks(req)))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question about FastAPI, with cited sources."""
    chunks = get_chunks(req)
    try:
        answer = generate_answer(req.question, chunks, state["client"])
    except anthropic.APIError as e:
        # 502 = Bad Gateway: a service we depend on (the LLM API) failed
        raise HTTPException(status_code=502, detail=f"LLM request failed: {type(e).__name__}")
    return AskResponse(answer=answer, sources=to_sources(chunks), model=MODEL)