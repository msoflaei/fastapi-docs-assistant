"""Evaluate retrieval quality on the in-scope evaluation questions.

Usage:
    python -m docs_assistant.eval_retrieval --label baseline
    python -m docs_assistant.eval_retrieval --label no-release-notes --exclude release-notes
"""
import argparse                         # reads options like --label from the command line
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.eval_data import load_questions
from docs_assistant.retrieve import retrieve
from docs_assistant.rerank import load_reranker, rerank
from docs_assistant.retrieve import DEFAULT_EXCLUDE, retrieve
import sys

RESULTS_DIR = Path("evals/results")
CUTOFFS = (1, 3, 5, 10)                 # the k values we report metrics at


def doc_of(chunk: dict) -> str:
    """The page a chunk came from: 'tutorial/cors#2' -> 'tutorial/cors'."""
    return chunk["chunk_id"].split("#")[0]


def score_question(retrieved_docs: list[str], relevant_docs: list[str]) -> dict:
    """Compute hit@k, recall@k and reciprocal rank for one question.

    retrieved_docs is the page of each retrieved chunk, in rank order (may repeat).
    """
    relevant = set(relevant_docs)
    scores = {}
    for k in CUTOFFS:
        found = relevant & set(retrieved_docs[:k])          # relevant pages among the top k
        scores[f"hit@{k}"] = 1.0 if found else 0.0
        scores[f"recall@{k}"] = len(found) / len(relevant)
    # Rank (1-based) of the first chunk from a relevant page, or None if there isn't one.
    first = next((rank for rank, doc in enumerate(retrieved_docs, start=1) if doc in relevant), None)
    scores["mrr"] = 1 / first if first else 0.0
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--label", default="baseline", help="name for this run, used in the results file")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                        help="doc_id patterns to exclude (default: the app's defaults; pass --exclude alone for none)")
    parser.add_argument("--rerank", action="store_true", help="rerank candidates with a cross-encoder")
    parser.add_argument("--candidates", type=int, default=30, help="how many chunks to fetch before reranking")
    parser.add_argument("--min-hit5", type=float,
                        help="quality gate: exit with an error if hit@5 is below this value")
    args = parser.parse_args()

    questions = [q for q in load_questions() if q["category"] == "in_scope"]
    model = load_model()
    reranker = load_reranker() if args.rerank else None

    results = []
    start = time.perf_counter()
    with connect() as conn:
        for q in questions:
            if reranker:
                # Stage 1: fetch many candidates quickly. Stage 2: rerank them, keep the top 10.
                candidates = retrieve(q["question"], model, conn, k=args.candidates, exclude_docs=args.exclude)
                chunks = rerank(q["question"], candidates, reranker, top_k=max(CUTOFFS))
            else:
                chunks = retrieve(q["question"], model, conn, k=max(CUTOFFS), exclude_docs=args.exclude)
            docs = [doc_of(c) for c in chunks]
            results.append({
                "id": q["id"],
                "question": q["question"],
                "relevant_docs": q["relevant_docs"],
                "retrieved_docs": docs,
                **score_question(docs, q["relevant_docs"]),    # ** merges the scores into this dict
            })
    seconds = time.perf_counter() - start

    # Average each metric over all questions.
    metric_names = [f"{m}@{k}" for m in ("hit", "recall") for k in CUTOFFS] + ["mrr"]
    summary = {name: round(mean(r[name] for r in results), 3) for name in metric_names}

    print(f"Run '{args.label}': {len(results)} in-scope questions, excluded: {args.exclude or 'none'}, "
          f"rerank: {f'yes ({args.candidates} candidates)' if args.rerank else 'no'}")
    print(f"Average time per question: {seconds / len(results) * 1000:.0f} ms\n")
    for name in metric_names:
        print(f"  {name:<10} {summary[name]:.3f}")

    # Error analysis: list every question where no relevant page made the top 5.
    misses = [r for r in results if r["hit@5"] == 0]
    print(f"\nMisses at k=5: {len(misses)}")
    for r in misses:
        print(f"\n  {r['id']}: {r['question']}")
        print(f"     expected: {', '.join(r['relevant_docs'])}")
        print(f"     got:      {', '.join(r['retrieved_docs'][:5])}")

    # Save everything, so different runs can be compared later.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"retrieval_{args.label}.json"
    out_file.write_text(json.dumps({
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "excluded_docs": args.exclude,
        "n_questions": len(results),
        "summary": summary,
        "per_question": results,
    }, indent=2))
    print(f"\nSaved -> {out_file}")

 # Quality gate: a non-zero exit code makes the CI job fail.
    if args.min_hit5 is not None:
        if summary["hit@5"] < args.min_hit5:
            print(f"\nQUALITY GATE FAILED: hit@5 {summary['hit@5']:.3f} < {args.min_hit5:.3f}")
            sys.exit(1)
        print(f"\nQuality gate passed: hit@5 {summary['hit@5']:.3f} >= {args.min_hit5:.3f}")


if __name__ == "__main__":
    main()

       