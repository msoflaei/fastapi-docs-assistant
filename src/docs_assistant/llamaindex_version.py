"""The retrieval pipeline rebuilt with LlamaIndex, for comparison with the hand-built version.

Same cleaned documents, embedding model, query prefix, database, and page exclusions;
LlamaIndex does the chunking, indexing, and retrieval. Evaluated with the same metrics,
and results are saved in the same format, so scripts/compare_runs.py works on them.

Usage:
    python -m docs_assistant.llamaindex_version --splitter markdown
    python -m docs_assistant.llamaindex_version --splitter sentence
"""
import argparse
import getpass
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import psycopg
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

from docs_assistant.db import DB_URL
from docs_assistant.embed import MODEL_NAME
from docs_assistant.eval_data import load_questions
from docs_assistant.eval_retrieval import CUTOFFS, RESULTS_DIR, score_question
from docs_assistant.search import QUERY_PREFIX

DOCS_FILE = Path("data/processed/docs.jsonl")
EMBED_DIM = 384


def load_documents() -> list[Document]:
    """Our cleaned pages as LlamaIndex Documents, with the same exclusions as our pipeline."""
    documents = []
    with DOCS_FILE.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["doc_id"] == "release-notes" or d["doc_id"].startswith("reference/"):
                continue
            documents.append(Document(
                text=d["text"],
                metadata={"doc_id": d["doc_id"], "title": d["title"], "url": d["url"]},
                # LlamaIndex adds metadata to the text it embeds by default. Keep the title
                # (useful context, like our breadcrumbs), but not IDs and URLs.
                excluded_embed_metadata_keys=["doc_id", "url"],
            ))
    return documents


def make_vector_store(table: str) -> PGVectorStore:
    """A LlamaIndex vector store in our PostgreSQL database, in its own table."""
    url = make_url(DB_URL)
    return PGVectorStore.from_params(
        database=url.database,
        host=url.host or "localhost",
        port=url.port or 5432,
        user=url.username or getpass.getuser(),     # locally, your Mac user name
        password=url.password or "",
        table_name=table,
        embed_dim=EMBED_DIM,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate a LlamaIndex retrieval pipeline.")
    parser.add_argument("--splitter", choices=["markdown", "sentence"], default="markdown")
    parser.add_argument("--chunk-size", type=int, default=1024, help="for the sentence splitter (LlamaIndex default: 1024)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="for the sentence splitter (LlamaIndex default: 200)")
    args = parser.parse_args()

    name = "markdown" if args.splitter == "markdown" else f"sentence{args.chunk_size}"
    label = f"li-{name}"
    table = f"li_{name}"

    # ---- Build ----
    documents = load_documents()
    node_parser = (MarkdownNodeParser() if args.splitter == "markdown"
                   else SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap))
    nodes = node_parser.get_nodes_from_documents(documents)
    words = sorted(len(n.get_content().split()) for n in nodes)
    print(f"{label}: {len(documents)} documents -> {len(nodes)} chunks "
          f"(words: min {words[0]}, median {words[len(words) // 2]}, max {words[-1]})")

    # Start fresh: drop this experiment's table if it exists (PGVectorStore adds a "data_" prefix).
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS data_{table}")

    embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME, query_instruction=QUERY_PREFIX)
    storage = StorageContext.from_defaults(vector_store=make_vector_store(table))
    index = VectorStoreIndex(nodes, storage_context=storage, embed_model=embed_model, show_progress=True)

    # ---- Evaluate (same questions and metrics as eval_retrieval) ----
    retriever = index.as_retriever(similarity_top_k=max(CUTOFFS))
    questions = [q for q in load_questions() if q["category"] == "in_scope"]
    results = []
    start = time.perf_counter()
    for q in questions:
        docs = [r.node.metadata["doc_id"] for r in retriever.retrieve(q["question"])]
        results.append({"id": q["id"], "question": q["question"], "relevant_docs": q["relevant_docs"],
                        "retrieved_docs": docs, **score_question(docs, q["relevant_docs"])})
    seconds = time.perf_counter() - start

    metric_names = [f"{m}@{k}" for m in ("hit", "recall") for k in CUTOFFS] + ["mrr"]
    summary = {m: round(mean(r[m] for r in results), 3) for m in metric_names}
    print(f"\nRun '{label}': {len(results)} in-scope questions, "
          f"{seconds / len(results) * 1000:.0f} ms per question\n")
    for m in metric_names:
        print(f"  {m:<10} {summary[m]:.3f}")

    misses = [r for r in results if r["hit@5"] == 0]
    print(f"\nMisses at k=5: {len(misses)}")
    for r in misses:
        print(f"  {r['id']}: {r['question']}\n     expected: {', '.join(r['relevant_docs'])}"
              f"\n     got:      {', '.join(r['retrieved_docs'][:5])}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"retrieval_{label}.json"
    out_file.write_text(json.dumps({
        "label": label, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "framework": "llamaindex", "splitter": args.splitter, "n_chunks": len(nodes),
        "n_questions": len(results), "summary": summary, "per_question": results,
    }, indent=2))
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()