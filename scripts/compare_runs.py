"""Show which questions changed between two retrieval evaluation runs.

Usage: python scripts/compare_runs.py baseline rerank
"""
import json
import sys
from pathlib import Path


def load(label: str) -> dict:
    return json.loads(Path(f"evals/results/retrieval_{label}.json").read_text())


def first_rank(result: dict) -> str:
    """Rank of the first relevant chunk (from the reciprocal rank), or '-' if not in the top 10."""
    return str(round(1 / result["mrr"])) if result["mrr"] else "-"


run_a, run_b = load(sys.argv[1]), load(sys.argv[2])
results_b = {r["id"]: r for r in run_b["per_question"]}   # look up run B's results by question id

print(f"Rank of first relevant result: {sys.argv[1]} -> {sys.argv[2]}\n")
changed = 0
for ra in run_a["per_question"]:
    rb = results_b[ra["id"]]
    if ra["mrr"] != rb["mrr"]:
        changed += 1
        direction = "better" if rb["mrr"] > ra["mrr"] else "WORSE"
        print(f"{ra['id']}: {first_rank(ra):>2} -> {first_rank(rb):>2}  ({direction})  {ra['question']}")
print(f"\n{changed} of {len(run_a['per_question'])} questions changed")